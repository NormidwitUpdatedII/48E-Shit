"""
Random Forest with Feature Engineering - Example Run Script
Demonstrates how to integrate feature engineering with existing models.

This script shows improved forecasting using:
1. Advanced feature engineering (momentum, volatility, z-scores)
2. Feature selection and preprocessing
3. Random Forest with optimized parameters
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
    select_features_by_variance,
    remove_highly_correlated
)

from sklearn.ensemble import RandomForestRegressor


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
    Run Random Forest with Feature Engineering.
    
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
    indice = indice - 1  # Convert to 0-indexed
    Y = np.array(Y)
    
    # Apply feature engineering to raw data
    fe = StationaryFeatureEngineer()
    Y_engineered = fe.get_all_features(Y, include_raw=True)
    
    # Handle NaN values
    Y_engineered = handle_missing_values(Y_engineered, strategy='mean')
    
    # Create embedded matrix with lags
    aux = embed(Y_engineered, 4 + lag)
    
    # Target is from original Y (not engineered)
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
    
    # Feature selection (remove low variance and highly correlated)
    X, var_mask = select_features_by_variance(X, threshold=0.001)
    X_out = X_out[var_mask]
    
    if X.shape[1] > 50:
        X, corr_mask = remove_highly_correlated(X, threshold=0.95)
        X_out = X_out[corr_mask]
    
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


def rf_fe_rolling_window(Y, nprev, indice, lag):
    """
    Run Random Forest with Feature Engineering using rolling window.
    
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
    
    predictions = []
    actuals = []
    
    # Rolling window forecasting
    for i in range(nprev, nobs - lag + 1):
        Y_train = Y[:i, :]
        
        result = run_rf_fe(Y_train, indice, lag)
        pred = result['pred']
        predictions.append(pred)
        
        # Get actual value
        actual = Y[i + lag - 1, indice - 1]
        actuals.append(actual)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
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
