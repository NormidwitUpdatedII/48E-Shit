"""
RF-OLS Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from joblib import Parallel, delayed
from utils import embed, compute_pca_scores, calculate_errors


# Number of parallel jobs (-1 = use all CPU cores)
N_JOBS = -1

def run_rfols(Y, indice, lag, n_estimators=500, top_k=10):
    """
    Run RF-OLS for inflation forecasting.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    has_dummy = Y.shape[1] > 2
    if has_dummy:
        Y_main = Y[:, :-1]
    else:
        Y_main = Y
    
    pca_scores, _ = compute_pca_scores(Y_main)
    pca_scores = pca_scores[:, :4]
    Y2 = np.column_stack([Y_main, pca_scores])
    
    aux = embed(Y2, 4 + lag)
    y = aux[:, indice - 1]
    X = aux[:, (Y2.shape[1] * lag):]
    
    X_out = aux[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    # Fit RF to get importance
    rf = RandomForestRegressor(n_estimators=n_estimators, max_features='sqrt', random_state=42)
    rf.fit(X, y)
    
    importance = rf.feature_importances_
    top_indices = np.argsort(importance)[-top_k:]
    
    # Fit OLS with top features
    X_selected = X[:, top_indices]
    X_out_selected = X_out[top_indices]
    
    model = LinearRegression()
    model.fit(X_selected, y)
    
    pred = model.intercept_ + X_out_selected @ model.coef_
    
    return {
        'pred': pred,
        'model': model,
        'selected_features': top_indices
    }


def rfols_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with RF-OLS.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        
        result = run_rfols(Y_window, indice, lag)
        save_pred[nprev - i, 0] = result['pred']
        
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
