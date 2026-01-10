"""
XGBoost with Feature Engineering - Second Sample

DATA PIPELINE:
    Raw FRED-MD --> fred_md_loader.py (stationarity) --> feature_engineering.py (additional FE)
    
NOTE: Since rawdata.csv is already transformed, we use skip_basic_transforms=True
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata_1990_2022.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts, embed, calculate_errors, add_outlier_dummy
from feature_engineering.feature_engineering import StationaryFeatureEngineer
from feature_engineering.feature_utils import standardize_features, handle_missing_values, apply_3stage_feature_selection
from feature_engineering.feature_config import CONSTANT_VARIANCE_THRESHOLD, CORRELATION_THRESHOLD, LOW_VARIANCE_THRESHOLD

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from joblib import Parallel, delayed


# Number of parallel jobs (4 for stability, prevents OOM)
N_JOBS = 4

XGB_PARAMS = {
    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0
}


def run_xgb_fe(Y, indice, lag):
    """Run XGBoost with Feature Engineering (OPTIMIZED).
    
    OPTIMIZATION: Only lag raw features, not engineered features.
    Reduces features from ~20,000 to ~5,000 (75% reduction).
    """
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost required")
    
    indice = indice - 1
    Y = np.array(Y)
    n_raw_features = Y.shape[1]
    
    # OPTIMIZATION: Separate raw lagging from feature engineering
    aux_raw = embed(Y, 4)
    
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
    
    # Step 9: Train XGBoost
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    
    return {'model': model, 'pred': model.predict(X_out)[0]}


def _xgb_fe_single_iteration(i, Y, indice, lag):
    """Single iteration for parallel XGB-FE rolling window."""
    result = run_xgb_fe(Y[:i, :], indice, lag)
    actual = Y[i + lag - 1, indice - 1]
    return i, result['pred'], actual


def xgb_fe_rolling_window(Y, nprev, indice, lag):
    """Run XGB with FE using rolling window (PARALLELIZED)."""
    Y = np.array(Y)
    nobs = Y.shape[0]
    
    # PARALLEL execution of rolling window
    print(f"    Running {nobs - lag + 1 - nprev} XGB-FE iterations in parallel...")
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_xgb_fe_single_iteration)(i, Y, indice, lag)
        for i in range(nprev, nobs - lag + 1)
    )
    
    # Sort by index and extract predictions/actuals
    results.sort(key=lambda x: x[0])
    predictions = np.array([r[1] for r in results])
    actuals = np.array([r[2] for r in results])
    
    return {'pred': predictions, 'actuals': actuals, 'errors': calculate_errors(actuals, predictions)}


def main():
    if not XGB_AVAILABLE:
        print("ERROR: XGBoost not installed")
        return
    
    Y = load_csv(DATA_PATH)
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 298
    np.random.seed(123)
    results = {}
    
    print("Running XGBoost with Feature Engineering (Second Sample)...")
    
    for lag in range(1, 13):
        print(f"  Lag={lag}...")
        results[f'xgb_fe{lag}c'] = xgb_fe_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'xgb_fe{lag}p'] = xgb_fe_rolling_window(Y, nprev, indice=2, lag=lag)
    
    cpi = np.column_stack([results[f'xgb_fe{lag}c']['pred'] for lag in range(1, 13)])
    pce = np.column_stack([results[f'xgb_fe{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    save_forecasts(cpi, os.path.join(FORECAST_DIR, 'xgb-fe-cpi.csv'))
    save_forecasts(pce, os.path.join(FORECAST_DIR, 'xgb-fe-pce.csv'))
    
    print(f"\nForecasts saved to {FORECAST_DIR}")
    return results


if __name__ == '__main__':
    main()
