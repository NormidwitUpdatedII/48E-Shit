"""
LSTM with Feature Engineering - First Sample

DATA PIPELINE:
    Raw FRED-MD --> fred_md_loader.py (stationarity) --> feature_engineering.py (additional FE)
    
NOTE: Since rawdata.csv is already transformed, we use skip_basic_transforms=True
"""
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts, embed, calculate_errors
from feature_engineering import StationaryFeatureEngineer
from feature_utils import (
    standardize_features, 
    handle_missing_values, 
    apply_3stage_feature_selection
)
from feature_config import (
    CONSTANT_VARIANCE_THRESHOLD,
    CORRELATION_THRESHOLD,
    LOW_VARIANCE_THRESHOLD
)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    # Suppress TF warnings and limit GPU memory growth
    tf.get_logger().setLevel('ERROR')
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available")

from joblib import Parallel, delayed


# Number of parallel jobs for LSTM (limited to avoid GPU memory issues)
N_JOBS_LSTM = 2  # Use fewer workers for LSTM due to memory constraints


def run_lstm_fe(Y, indice, lag, lstm_units=64, dropout_rate=0.2):
    """Run LSTM with Feature Engineering (OPTIMIZED).
    
    OPTIMIZATION: Only lag raw features, not engineered features.
    Reduces features from ~20,000 to ~5,000, then to ~200 for LSTM efficiency.
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras required")
    
    indice = indice - 1
    Y = np.array(Y)
    n_raw_features = Y.shape[1]
    
    # OPTIMIZATION: Separate raw lagging from feature engineering
    # Step 1: Create lags of RAW features only
    aux_raw = embed(Y, 4)
    
    # Step 2: Engineer features on CURRENT data (no lags)
    fe = StationaryFeatureEngineer()
    Y_engineered = fe.get_all_features(Y, include_raw=False, skip_basic_transforms=True)
    Y_engineered, _ = handle_missing_values(Y_engineered, strategy='mean')
    
    # Step 3: Align lengths
    min_len = min(len(aux_raw), len(Y_engineered))
    aux_raw = aux_raw[:min_len]
    Y_eng_current = Y_engineered[:min_len]
    
    # Step 4: Create target
    y_target = embed(Y[:, indice].reshape(-1, 1), 4)[:, 0]
    y_target = y_target[:min_len]
    
    # Step 5: Build feature matrix
    if lag == 1:
        X_raw_lagged = aux_raw
    else:
        lag_start = n_raw_features * (lag - 1)
        lag_end = lag_start + (n_raw_features * 4)
        if lag_end <= aux_raw.shape[1]:
            X_raw_lagged = aux_raw[:, lag_start:lag_end]
        else:
            X_raw_lagged = aux_raw[:, lag_start:]
    
    # Step 6: Combine lagged raw + current engineered
    X = np.hstack([X_raw_lagged, Y_eng_current])
    X_out = X[-1, :]
    
    # Step 7: Adjust for forecast horizon
    y = y_target[:len(y_target) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    # Step 8: Feature selection for LSTM efficiency
    # Apply 3-stage selection first
    X, selection_info = apply_3stage_feature_selection(
        X,
        constant_threshold=CONSTANT_VARIANCE_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        variance_threshold=LOW_VARIANCE_THRESHOLD * 2,
        verbose=False
    )
    X_out = X_out[selection_info['combined_mask']]
    
    # Limit features for LSTM efficiency (keep top ~200 features)
    if X.shape[1] > 200:
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-200:]
        X = X[:, top_indices]
        X_out = X_out[top_indices]
    
    # Standardize
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std
    X_out_scaled = (X_out - X_mean) / X_std
    
    # Reshape for LSTM (samples, timesteps, features)
    lookback = min(4, X.shape[1])
    n_features = X.shape[1] // lookback
    
    if X.shape[1] % lookback != 0:
        n_features = X.shape[1] // lookback + 1
        pad_width = n_features * lookback - X.shape[1]
        X_scaled = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode='constant')
        X_out_scaled = np.pad(X_out_scaled, (0, pad_width), mode='constant')
    
    X_lstm = X_scaled.reshape(X_scaled.shape[0], lookback, n_features)
    X_out_lstm = X_out_scaled.reshape(1, lookback, n_features)
    
    # Build LSTM model
    model = Sequential([
        LSTM(lstm_units, activation='tanh', return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, activation='tanh'),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_lstm, y, epochs=100, batch_size=16, verbose=0, callbacks=[early_stop])
    
    pred = model.predict(X_out_lstm, verbose=0)[0, 0]
    
    return {'model': model, 'pred': pred}


def _lstm_fe_single_iteration(i, Y, indice, lag, lstm_units, dropout_rate):
    """Single iteration for parallel LSTM-FE rolling window."""
    result = run_lstm_fe(Y[:i, :], indice, lag, lstm_units, dropout_rate)
    actual = Y[i + lag - 1, indice - 1]
    return i, result['pred'], actual


def lstm_fe_rolling_window(Y, nprev, indice, lag, lstm_units=64, dropout_rate=0.2):
    """Run LSTM with FE using rolling window (PARALLELIZED with limited workers)."""
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras required")
    
    Y = np.array(Y)
    nobs = Y.shape[0]
    
    # PARALLEL execution with limited workers (LSTM is memory intensive)
    print(f"    Running {nobs - lag + 1 - nprev} LSTM-FE iterations in parallel (n_jobs={N_JOBS_LSTM})...")
    results = Parallel(n_jobs=N_JOBS_LSTM, verbose=1)(
        delayed(_lstm_fe_single_iteration)(i, Y, indice, lag, lstm_units, dropout_rate)
        for i in range(nprev, nobs - lag + 1)
    )
    
    # Sort by index and extract predictions/actuals
    results.sort(key=lambda x: x[0])
    predictions = np.array([r[1] for r in results])
    actuals = np.array([r[2] for r in results])
    
    return {'pred': predictions, 'actuals': actuals, 'errors': calculate_errors(actuals, predictions)}


def main():
    if not KERAS_AVAILABLE:
        print("ERROR: TensorFlow not installed")
        return
    
    Y = load_csv(DATA_PATH)
    nprev = 132
    np.random.seed(123)
    
    results = {}
    
    print("Running LSTM with Feature Engineering (First Sample)...")
    print("=" * 60)
    
    for lag in range(1, 13):
        print(f"  Processing lag={lag}...")
        results[f'lstm_fe{lag}c'] = lstm_fe_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'lstm_fe{lag}p'] = lstm_fe_rolling_window(Y, nprev, indice=2, lag=lag)
    
    cpi = np.column_stack([results[f'lstm_fe{lag}c']['pred'] for lag in range(1, 13)])
    pce = np.column_stack([results[f'lstm_fe{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    save_forecasts(cpi, os.path.join(FORECAST_DIR, 'lstm-fe-cpi.csv'))
    save_forecasts(pce, os.path.join(FORECAST_DIR, 'lstm-fe-pce.csv'))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    for lag in [1, 6, 12]:
        print(f"Lag {lag}: CPI RMSE={results[f'lstm_fe{lag}c']['errors']['rmse']:.6f}")
    
    print(f"\nForecasts saved to {FORECAST_DIR}")
    return results


if __name__ == '__main__':
    main()
