"""
Adaptive LASSO Rolling Window Forecasts
"""
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path constants for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts
from first_sample.functions.func_lasso import lasso_rolling_window, pols_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = 132
    alpha = 1  # alpha=1 for LASSO
    
    results = {}
    pols_results = {}
    
    # Adaptive LASSO models
    print("Running Adaptive LASSO models...")
    for lag in range(1, 13):
        print(f"  AdaLASSO lag={lag}")
        results[f'adalasso{lag}c'] = lasso_rolling_window(Y, nprev, indice=1, lag=lag, alpha=alpha, model_type='adalasso')
        results[f'adalasso{lag}p'] = lasso_rolling_window(Y, nprev, indice=2, lag=lag, alpha=alpha, model_type='adalasso')
    
    # POLS (Post-OLS) models using AdaLASSO selected variables
    print("Running POLS models...")
    for lag in range(1, 13):
        print(f"  POLS lag={lag}")
        pols_results[f'pols{lag}c'] = pols_rolling_window(Y, nprev, indice=1, lag=lag, 
                                                           coef_matrix=results[f'adalasso{lag}c']['coef'])
        pols_results[f'pols{lag}p'] = pols_rolling_window(Y, nprev, indice=2, lag=lag,
                                                           coef_matrix=results[f'adalasso{lag}p']['coef'])
    
    # Combine results for CPI (indice=1)
    cpi_adalasso = np.column_stack([results[f'adalasso{lag}c']['pred'] for lag in range(1, 13)])
    cpi_pols = np.column_stack([pols_results[f'pols{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE (indice=2)
    pce_adalasso = np.column_stack([results[f'adalasso{lag}p']['pred'] for lag in range(1, 13)])
    pce_pols = np.column_stack([pols_results[f'pols{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_adalasso, os.path.join(FORECAST_DIR, 'adalasso-cpi.csv'))
    save_forecasts(pce_adalasso, os.path.join(FORECAST_DIR, 'adalasso-pce.csv'))
    save_forecasts(cpi_pols, os.path.join(FORECAST_DIR, 'pols-adalasso-cpi.csv'))
    save_forecasts(pce_pols, os.path.join(FORECAST_DIR, 'pols-adalasso-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    
    return results, pols_results


if __name__ == '__main__':
    main()
