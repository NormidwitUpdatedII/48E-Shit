"""
XGBoost Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import embed, compute_pca_scores, calculate_errors


def run_xgb(Y, indice, lag, n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Run XGBoost for inflation forecasting.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    has_dummy = Y.shape[1] > 2
    if has_dummy:
        Y_main = Y[:, :-1]
    else:
        Y_main = Y
    
    pca_scores = compute_pca_scores(Y_main)[:, :4]
    Y2 = np.column_stack([Y_main, pca_scores])
    
    aux = embed(Y2, 4 + lag)
    y = aux[:, indice - 1]
    X = aux[:, (Y2.shape[1] * lag):]
    
    X_out = aux[-1, :X.shape[1]].reshape(1, -1)
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)
    
    pred = model.predict(X_out)[0]
    
    return {
        'pred': pred,
        'model': model
    }


def xgb_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with XGBoost.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        
        result = run_xgb(Y_window, indice, lag)
        save_pred[nprev - i, 0] = result['pred']
        
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
