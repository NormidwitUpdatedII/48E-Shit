"""
Random Forest with Feature Engineering - Run Script

This script uses:
1. FRED-MD transformed data (stationarity transforms already applied)
2. Additional feature engineering (momentum, volatility, z-scores)
3. Feature selection and preprocessing
4. Random Forest with optimized parameters

DATA PIPELINE:
    Raw FRED-MD --> fred_md_loader.py (stationarity) --> feature_engineering.py (additional FE)
    
NOTE: Since rawdata.csv is already transformed, we use skip_basic_transforms=True
"""
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts, embed, compute_pca_scores, calculate_errors
from feature_engineering import (
    StationaryFeatureEngineer, 
    engineer_features_for_model,
    apply_feature_engineering_to_rolling_window
)
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

from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

# Optimized RF parameters (from EC48E project)
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 20,
    'random_state': 42,
    'n_jobs': -1,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}


def run_rf_fe(Y, indice, lag):
    """
    Run Random Forest with Feature Engineering (OPTIMIZED).
    
    OPTIMIZATION: Only lag raw features, not engineered features.
    Engineered features (rolling stats, momentum, etc.) already contain
    historical information. Lagging them is 92% redundant.
    
    This reduces features from ~20,000 to ~5,000 (75% reduction).
    
    Parameters:
    -----------
    Y : np.ndarray
        Input data matrix (already FRED-MD transformed)
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    dict : Dictionary with 'model' and 'pred'
    """
    indice = indice - 1  # Convert to 0-indexed
    Y = np.array(Y)
    n_raw_features = Y.shape[1]
    
    # OPTIMIZATION: Separate raw lagging from feature engineering
    # Step 1: Create lags of RAW features only (126 vars × 4 lags = 504 features)
    aux_raw = embed(Y, 4)
    
    # Step 2: Engineer features on CURRENT data (no lags)
    # This gives ~4,410 features with historical info already encoded
    fe = StationaryFeatureEngineer()
    Y_engineered = fe.get_all_features(Y, include_raw=False, skip_basic_transforms=True)
    
    # Handle NaN values
    Y_engineered, _ = handle_missing_values(Y_engineered, strategy='mean')
    
    # Step 3: Align lengths
    min_len = min(len(aux_raw), len(Y_engineered))
    aux_raw = aux_raw[:min_len]
    Y_eng_current = Y_engineered[:min_len]
    
    # Step 4: Create target from original Y
    y_target = embed(Y[:, indice].reshape(-1, 1), 4)[:, 0]
    y_target = y_target[:min_len]
    
    # Step 5: Build feature matrix based on forecast horizon
    # For lag=1: use lags [1,2,3,4] of raw + current engineered
    # For lag=12: use lags [12,13,14,15] of raw + current engineered
    if lag == 1:
        # Use all 4 lags of raw features
        X_raw_lagged = aux_raw
    else:
        # Shift to use lags [lag, lag+1, lag+2, lag+3]
        # This ensures we're using appropriate historical lags for the forecast horizon
        lag_start = n_raw_features * (lag - 1)
        lag_end = lag_start + (n_raw_features * 4)
        
        if lag_end <= aux_raw.shape[1]:
            X_raw_lagged = aux_raw[:, lag_start:lag_end]
        else:
            # If we don't have enough lags, use what we have
            X_raw_lagged = aux_raw[:, lag_start:]
    
    # Step 6: Combine lagged raw + current engineered
    # Total: ~504 lagged + ~4,410 current = ~4,914 features (vs 20,000 before!)
    X = np.hstack([X_raw_lagged, Y_eng_current])
    
    # Step 7: Prepare out-of-sample features
    X_out = X[-1, :]
    
    # Step 8: Adjust for forecast horizon
    y = y_target[:len(y_target) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    # Apply 3-stage feature selection (now much faster with 5k vs 20k features!)
    X, selection_info = apply_3stage_feature_selection(
        X, 
        constant_threshold=CONSTANT_VARIANCE_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        variance_threshold=LOW_VARIANCE_THRESHOLD,
        verbose=False
    )
    
    # Apply selection mask to out-of-sample features
    X_out = X_out[selection_info['combined_mask']]
    
    # Standardize
    X, scaler = standardize_features(X)
    X_out = scaler.transform(X_out.reshape(1, -1))
    
    # Train Random Forest
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X, y)
    
    # Predict
    pred = model.predict(X_out)[0]
    
    return {
        'model': model,
        'pred': pred
    }


def _rf_fe_single_iteration(i, Y, indice, lag):
    """Single iteration for parallel RF-FE rolling window."""
    Y_train = Y[:i, :]
    result = run_rf_fe(Y_train, indice, lag)
    actual = Y[i + lag - 1, indice - 1]
    return i, result['pred'], actual


def rf_fe_rolling_window(Y, nprev, indice, lag):
    """
    Run Random Forest with Feature Engineering using rolling window (PARALLELIZED).
    
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
    print(f"    Running {nobs - lag + 1 - nprev} RF-FE iterations in parallel...")
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_rf_fe_single_iteration)(i, Y, indice, lag)
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
    """Main function to run RF with Feature Engineering forecasts."""
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = 132  # Initial training observations
    np.random.seed(123)
    
    results = {}
    
    print("Running Random Forest with Feature Engineering...")
    print("=" * 50)
    
    # Run for each lag
    for lag in range(1, 13):
        print(f"  Processing lag={lag}...")
        
        # CPI (indice=1)
        results[f'rf_fe{lag}c'] = rf_fe_rolling_window(Y, nprev, indice=1, lag=lag)
        
        # PCE (indice=2)
        results[f'rf_fe{lag}p'] = rf_fe_rolling_window(Y, nprev, indice=2, lag=lag)
    
    # Combine results for CPI
    cpi_rf_fe = np.column_stack([results[f'rf_fe{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE
    pce_rf_fe = np.column_stack([results[f'rf_fe{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_rf_fe, os.path.join(FORECAST_DIR, 'rf-fe-cpi.csv'))
    save_forecasts(pce_rf_fe, os.path.join(FORECAST_DIR, 'rf-fe-pce.csv'))
    
    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    for lag in [1, 6, 12]:
        print(f"\nLag {lag}:")
        print(f"  CPI RMSE: {results[f'rf_fe{lag}c']['errors']['rmse']:.6f}")
        print(f"  PCE RMSE: {results[f'rf_fe{lag}p']['errors']['rmse']:.6f}")
    
    print(f"\nDone! Forecasts saved to {FORECAST_DIR}")
    
    return results


if __name__ == '__main__':
    main()
