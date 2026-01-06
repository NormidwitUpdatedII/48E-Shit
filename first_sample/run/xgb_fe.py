"""
XGBoost with Feature Engineering - Run Script

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
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts, embed, calculate_errors
from feature_engineering import StationaryFeatureEngineer
from feature_utils import (
    standardize_features, 
    handle_missing_values,
    select_features_by_variance,
    remove_highly_correlated
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

from joblib import Parallel, delayed


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

# Optimized XGBoost parameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}


def run_xgb_fe(Y, indice, lag):
    """
    Run XGBoost with Feature Engineering.
    
    Parameters:
    -----------
    Y : np.ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    dict : Dictionary with 'model' and 'pred'
    """
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost is required")
    
    indice = indice - 1  # Convert to 0-indexed
    Y = np.array(Y)
    
    # Apply feature engineering (skip basic transforms - data already transformed)
    fe = StationaryFeatureEngineer()
    Y_engineered = fe.get_all_features(Y, include_raw=True, skip_basic_transforms=True)
    
    # Handle NaN values
    Y_engineered = handle_missing_values(Y_engineered, strategy='mean')
    
    # Create embedded matrix with lags
    aux = embed(Y_engineered, 4 + lag)
    
    # Target is from original Y
    y_target = embed(Y[:, indice].reshape(-1, 1), 4 + lag)[:, 0]
    
    # Align dimensions
    min_len = min(len(aux), len(y_target))
    aux = aux[:min_len]
    y_target = y_target[:min_len]
    
    # Features (lagged values)
    n_cols = Y_engineered.shape[1]
    X = aux[:, n_cols * lag:]
    
    # Out-of-sample features
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        aux_trimmed = aux[:, :aux.shape[1] - n_cols * (lag - 1)]
        X_out = aux_trimmed[-1, :X.shape[1]]
    
    # Adjust for lag
    y = y_target[:len(y_target) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    # Feature selection
    X, var_mask = select_features_by_variance(X, threshold=0.001)
    X_out = X_out[var_mask]
    
    if X.shape[1] > 100:
        X, corr_mask = remove_highly_correlated(X, threshold=0.95)
        X_out = X_out[corr_mask]
    
    # Standardize
    X, scaler = standardize_features(X)
    X_out = scaler.transform(X_out.reshape(1, -1))
    
    # Train XGBoost
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    
    # Predict
    pred = model.predict(X_out)[0]
    
    return {
        'model': model,
        'pred': pred
    }


def _xgb_fe_single_iteration(i, Y, indice, lag):
    """Single iteration for parallel XGB-FE rolling window."""
    Y_train = Y[:i, :]
    result = run_xgb_fe(Y_train, indice, lag)
    actual = Y[i + lag - 1, indice - 1]
    return i, result['pred'], actual


def xgb_fe_rolling_window(Y, nprev, indice, lag):
    """
    Run XGBoost with Feature Engineering using rolling window (PARALLELIZED).
    
    Parameters:
    -----------
    Y : np.ndarray
        Input data matrix
    nprev : int
        Number of observations for initial training
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    dict : Dictionary with 'pred' array and 'errors'
    """
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
    
    # Calculate errors
    errors = calculate_errors(actuals, predictions)
    
    return {
        'pred': predictions,
        'actuals': actuals,
        'errors': errors
    }


def main():
    """Main function to run XGBoost with Feature Engineering forecasts."""
    if not XGB_AVAILABLE:
        print("ERROR: XGBoost not installed. Run: pip install xgboost")
        return
    
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = 132  # Initial training observations
    np.random.seed(123)
    
    results = {}
    
    print("Running XGBoost with Feature Engineering...")
    print("=" * 50)
    
    # Run for each lag
    for lag in range(1, 13):
        print(f"  Processing lag={lag}...")
        
        # CPI (indice=1)
        results[f'xgb_fe{lag}c'] = xgb_fe_rolling_window(Y, nprev, indice=1, lag=lag)
        
        # PCE (indice=2)
        results[f'xgb_fe{lag}p'] = xgb_fe_rolling_window(Y, nprev, indice=2, lag=lag)
    
    # Combine results for CPI
    cpi_xgb_fe = np.column_stack([results[f'xgb_fe{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE
    pce_xgb_fe = np.column_stack([results[f'xgb_fe{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_xgb_fe, os.path.join(FORECAST_DIR, 'xgb-fe-cpi.csv'))
    save_forecasts(pce_xgb_fe, os.path.join(FORECAST_DIR, 'xgb-fe-pce.csv'))
    
    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    for lag in [1, 6, 12]:
        print(f"\nLag {lag}:")
        print(f"  CPI RMSE: {results[f'xgb_fe{lag}c']['errors']['rmse']:.6f}")
        print(f"  PCE RMSE: {results[f'xgb_fe{lag}p']['errors']['rmse']:.6f}")
    
    print(f"\nDone! Forecasts saved to {FORECAST_DIR}")
    
    return results


if __name__ == '__main__':
    main()
