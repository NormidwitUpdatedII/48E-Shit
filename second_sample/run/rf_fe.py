"""
Random Forest with Feature Engineering - Second Sample

DATA PIPELINE:
    Raw FRED-MD --> fred_md_loader.py (stationarity) --> feature_engineering.py (additional FE)
    
NOTE: Since rawdata.csv is already transformed, we use skip_basic_transforms=True
"""
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata_1990_2022.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts, embed, calculate_errors, add_outlier_dummy
from feature_engineering.feature_engineering import StationaryFeatureEngineer
from feature_engineering.feature_utils import (
    standardize_features, 
    handle_missing_values,
    apply_3stage_feature_selection
)

from feature_engineering.feature_config import (
    CONSTANT_VARIANCE_THRESHOLD,
    CORRELATION_THRESHOLD,
    LOW_VARIANCE_THRESHOLD
)

from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

# Optimized RF parameters
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 20,
    'random_state': 42,
    'n_jobs': -1,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}


def run_rf_fe(Y, indice, lag):
    """Run Random Forest with Feature Engineering (OPTIMIZED).
    
    OPTIMIZATION: Only lag raw features, not engineered features.
    Reduces features from ~20,000 to ~5,000 (75% reduction).
    """
    indice = indice - 1
    Y = np.array(Y)
    n_raw_features = Y.shape[1]
    
    # Step 1: Create lags of RAW features only
    aux_raw = embed(Y, 4)
    
    # Step 2: Engineer features on CURRENT data (no lags)
    fe = StationaryFeatureEngineer()
    Y_engineered = fe.get_all_features(Y, include_raw=False, skip_basic_transforms=True)
    Y_engineered, _ = handle_missing_values(Y_engineered, strategy='mean')
    
    # Step 3: Align lengths (starts at index 3 due to embed 4)
    Y_eng_aligned = Y_engineered[3:3+len(aux_raw)]
    X_combined = np.hstack([aux_raw, Y_eng_aligned])
    
    # Step 4: Create target from original Y (aligned with X_combined)
    y_target = Y[3:3+len(aux_raw), indice]
    
    # Step 5: Supervised Learning Alignment (CRITICAL FIX)
    # Match Month t info (X_t) with Month t+lag target (y_{t+lag})
    if len(X_combined) <= lag:
        return {'model': None, 'pred': np.nan}
        
    y_train = y_target[lag:]
    X_train = X_combined[:-lag, :]
    
    # Step 6: Prepare features for the out-of-sample forecast
    # We use the most recent info set (Month T) to predict Month T+lag
    X_out = X_combined[-1, :].reshape(1, -1)
    
    # Step 7: Apply 3-stage feature selection
    X_train, selection_info = apply_3stage_feature_selection(
        X_train, 
        constant_threshold=CONSTANT_VARIANCE_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        variance_threshold=LOW_VARIANCE_THRESHOLD,
        verbose=False
    )
    
    # Apply same selection to out-of-sample features
    X_out = X_out[:, selection_info['combined_mask']]
    
    # Step 8: Standardize
    X_train, scaler = standardize_features(X_train)
    X_out = scaler.transform(X_out)
    
    # Step 9: Train Random Forest
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    
    # Step 10: Predict
    pred = model.predict(X_out)[0]
    
    return {'model': model, 'pred': pred}


def _rf_fe_single_iteration(i, Y, indice, lag):
    """Single iteration for parallel RF-FE rolling window."""
    Y_train = Y[:i, :]
    result = run_rf_fe(Y_train, indice, lag)
    actual = Y[i + lag - 1, indice - 1]
    return i, result['pred'], actual


def rf_fe_rolling_window(Y, nprev, indice, lag):
    """Run RF with FE using rolling window (PARALLELIZED)."""
    Y = np.array(Y)
    nobs = Y.shape[0]
    
    # PARALLEL execution of rolling window
    print(f"    Running {nobs - lag + 1 - nprev} RF-FE iterations in parallel...")
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_rf_fe_single_iteration)(i, Y, indice, lag)
        for i in range(nprev, nobs - lag + 1)
    )
    
    # Sort by index and extract predictions/actuals
    results.sort(key=lambda x: x[0])
    predictions = np.array([r[1] for r in results])
    actuals = np.array([r[2] for r in results])
    errors = calculate_errors(actuals, predictions)
    
    return {'pred': predictions, 'actuals': actuals, 'errors': errors}


def main():
    """Main function."""
    Y = load_csv(DATA_PATH)
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 298  # Second sample: 2001-2025 out-of-sample
    np.random.seed(123)
    
    results = {}
    
    print("Running Random Forest with Feature Engineering (Second Sample)...")
    print("=" * 60)
    
    for lag in range(1, 13):
        print(f"  Processing lag={lag}...")
        results[f'rf_fe{lag}c'] = rf_fe_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'rf_fe{lag}p'] = rf_fe_rolling_window(Y, nprev, indice=2, lag=lag)
    
    cpi_rf_fe = np.column_stack([results[f'rf_fe{lag}c']['pred'] for lag in range(1, 13)])
    pce_rf_fe = np.column_stack([results[f'rf_fe{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    save_forecasts(cpi_rf_fe, os.path.join(FORECAST_DIR, 'rf-fe-cpi.csv'))
    save_forecasts(pce_rf_fe, os.path.join(FORECAST_DIR, 'rf-fe-pce.csv'))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    for lag in [1, 6, 12]:
        print(f"Lag {lag}: CPI RMSE={results[f'rf_fe{lag}c']['errors']['rmse']:.6f}, "
              f"PCE RMSE={results[f'rf_fe{lag}p']['errors']['rmse']:.6f}")
    
    print(f"\nForecasts saved to {FORECAST_DIR}")
    return results


if __name__ == '__main__':
    main()
