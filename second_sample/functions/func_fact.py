"""
Factor Model Functions for Inflation Forecasting - Second Sample
With dummy variable handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import embed, compute_pca_scores, calculate_errors


def run_fact(Y, indice, lag, n_factors=4):
    """
    Run Factor Model for inflation forecasting.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    has_dummy = Y.shape[1] > 2
    if has_dummy:
        Y_main = Y[:, :-1]
    else:
        Y_main = Y
    
    # Get target variable and factors
    y_target = Y_main[:, indice - 1]
    pca_scores, _ = compute_pca_scores(Y_main)
    pca_scores = pca_scores[:, :n_factors]
    
    # Combine target with factors
    Y2 = np.column_stack([y_target, pca_scores])
    
    aux = embed(Y2, 4 + lag)
    y = aux[:, 0]
    X = aux[:, (Y2.shape[1] * lag):]
    
    X_out = aux[-1, :X.shape[1]]
    
    y = y[:(len(y) - lag + 1)]
    X = X[:(X.shape[0] - lag + 1), :]
    
    model = LinearRegression()
    model.fit(X, y)
    
    pred = model.intercept_ + X_out @ model.coef_
    
    return {
        'pred': pred,
        'model': model
    }


def fact_rolling_window(Y, nprev, indice=1, lag=1):
    """
    Rolling window forecasting with Factor Model.
    """
    Y = np.array(Y)
    n_obs = Y.shape[0]
    
    save_pred = np.full((nprev, 1), np.nan)
    
    for i in range(nprev, 0, -1):
        start_idx = nprev - i
        end_idx = n_obs - i
        Y_window = Y[start_idx:end_idx, :]
        
        result = run_fact(Y_window, indice, lag)
        save_pred[nprev - i, 0] = result['pred']
        
        print(f"iteration {nprev - i + 1}")
    
    real = Y[:, indice - 1]
    errors = calculate_errors(real[-nprev:], save_pred.flatten())
    
    return {
        'pred': save_pred,
        'errors': errors
    }
