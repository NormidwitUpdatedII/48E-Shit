"""
XGBoost functions for inflation forecasting.
"""

import numpy as np
import xgboost as xgb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import embed, compute_pca_scores, calculate_errors, plot_forecast


def run_xgb(Y, indice, lag):
    """
    Run XGBoost model for forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    dict : Dictionary with 'model' and 'pred'
    """
    # Convert to 0-indexed
    indice = indice - 1
    
    Y = np.array(Y)
    
    # Compute PCA scores (returns tuple: scores, Y_filled)
    scores, _ = compute_pca_scores(Y, n_components=4, scale=False)
    
    # Combine original data with PCA scores
    Y2 = np.column_stack([Y, scores])
    
    # Create embedded matrix
    aux = embed(Y2, 4 + lag)
    y = aux[:, indice]
    n_cols_Y2 = Y2.shape[1]
    X = aux[:, n_cols_Y2 * lag:]
    
    # Prepare out-of-sample data
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        aux_trimmed = aux[:, :aux.shape[1] - n_cols_Y2 * (lag - 1)]
        X_out = aux_trimmed[-1, :X.shape[1]]
    
    # Adjust for lag
    y = y[:len(y) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    # Fit XGBoost model with parameters similar to R code
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,  # eta
        max_depth=4,
        colsample_bylevel=2/3,
        subsample=1.0,
        min_child_weight=X.shape[0] / 200,
        random_state=42,
        verbosity=0,
        n_jobs=1
    )
    model.fit(X, y, verbose=False)
    
    # Make prediction
    pred = model.predict(X_out.reshape(1, -1))[0]
    
    return {'model': model, 'pred': pred}


def xgb_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window XGBoost forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    nprev : int
        Number of out-of-sample predictions
    indice : int
        Column index of target variable (1-indexed)
    lag : int
        Forecast horizon
    
    Returns:
    --------
    dict : Dictionary with 'pred' and 'errors'
    """
    Y = np.array(Y)
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        # Window selection
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        
        # Run XGBoost model
        result = run_xgb(Y_window, indice, lag)
        
        idx = nprev - i
        save_pred[idx, 0] = result['pred']
        
        print(f"iteration {idx + 1}")
    
    # Calculate errors
    real = Y[:, indice - 1]  # Convert to 0-indexed
    actual = real[-nprev:]
    errors = calculate_errors(actual, save_pred.flatten())
    
    return {'pred': save_pred, 'errors': errors}


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    Y = np.random.randn(200, 5)
    
    result = xgb_rolling_window(Y, nprev=10, indice=1, lag=1)
    print(f"XGB RMSE: {result['errors']['rmse']:.4f}")
    print(f"XGB MAE: {result['errors']['mae']:.4f}")
