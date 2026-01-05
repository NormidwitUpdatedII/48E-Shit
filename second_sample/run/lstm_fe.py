"""
LSTM with Feature Engineering - Second Sample

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

from utils import load_csv, save_forecasts, embed, calculate_errors, add_outlier_dummy
from feature_engineering import StationaryFeatureEngineer
from feature_utils import standardize_features, handle_missing_values, select_features_by_variance

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


def run_lstm_fe(Y, indice, lag, lstm_units=64, dropout_rate=0.2):
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras required")
    
    indice = indice - 1
    Y = np.array(Y)
    
    fe = StationaryFeatureEngineer()
    Y_engineered = fe.get_all_features(Y, include_raw=True, skip_basic_transforms=True)
    Y_engineered = handle_missing_values(Y_engineered, strategy='mean')
    Y_engineered, _ = select_features_by_variance(Y_engineered, threshold=0.01)
    
    if Y_engineered.shape[1] > 100:
        Y_engineered = Y_engineered[:, :100]
    
    aux = embed(Y_engineered, 4 + lag)
    y_target = embed(Y[:, indice].reshape(-1, 1), 4 + lag)[:, 0]
    
    min_len = min(len(aux), len(y_target))
    aux, y_target = aux[:min_len], y_target[:min_len]
    
    n_cols = Y_engineered.shape[1]
    X = aux[:, n_cols * lag:]
    
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        aux_trimmed = aux[:, :aux.shape[1] - n_cols * (lag - 1)]
        X_out = aux_trimmed[-1, :X.shape[1]]
    
    y = y_target[:len(y_target) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std
    X_out_scaled = (X_out - X_mean) / X_std
    
    lookback = min(4, X.shape[1])
    n_features = X.shape[1] // lookback
    
    if X.shape[1] % lookback != 0:
        n_features = X.shape[1] // lookback + 1
        pad_width = n_features * lookback - X.shape[1]
        X_scaled = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode='constant')
        X_out_scaled = np.pad(X_out_scaled, (0, pad_width), mode='constant')
    
    X_lstm = X_scaled.reshape(X_scaled.shape[0], lookback, n_features)
    X_out_lstm = X_out_scaled.reshape(1, lookback, n_features)
    
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
    
    return {'model': model, 'pred': model.predict(X_out_lstm, verbose=0)[0, 0]}


def lstm_fe_rolling_window(Y, nprev, indice, lag, lstm_units=64, dropout_rate=0.2):
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras required")
    
    Y = np.array(Y)
    nobs = Y.shape[0]
    predictions, actuals = [], []
    
    for i in range(nprev, nobs - lag + 1):
        result = run_lstm_fe(Y[:i, :], indice, lag, lstm_units, dropout_rate)
        predictions.append(result['pred'])
        actuals.append(Y[i + lag - 1, indice - 1])
    
    predictions, actuals = np.array(predictions), np.array(actuals)
    return {'pred': predictions, 'actuals': actuals, 'errors': calculate_errors(actuals, predictions)}


def main():
    if not KERAS_AVAILABLE:
        print("ERROR: TensorFlow not installed")
        return
    
    Y = load_csv(DATA_PATH)
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 298
    np.random.seed(123)
    results = {}
    
    print("Running LSTM with Feature Engineering (Second Sample)...")
    
    for lag in range(1, 13):
        print(f"  Lag={lag}...")
        results[f'lstm_fe{lag}c'] = lstm_fe_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'lstm_fe{lag}p'] = lstm_fe_rolling_window(Y, nprev, indice=2, lag=lag)
    
    cpi = np.column_stack([results[f'lstm_fe{lag}c']['pred'] for lag in range(1, 13)])
    pce = np.column_stack([results[f'lstm_fe{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    save_forecasts(cpi, os.path.join(FORECAST_DIR, 'lstm-fe-cpi.csv'))
    save_forecasts(pce, os.path.join(FORECAST_DIR, 'lstm-fe-pce.csv'))
    
    print(f"\nForecasts saved to {FORECAST_DIR}")
    return results


if __name__ == '__main__':
    main()
