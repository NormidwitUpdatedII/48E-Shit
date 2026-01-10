"""
LASSO Rolling Window Forecasts - Second Sample
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path constants for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata_1990_2022.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts, add_outlier_dummy
from second_sample.functions.func_lasso import lasso_rolling_window, pols_rolling_window


def main():
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 298  # Out-of-sample 2001-2025
    alpha = 1
    
    results = {}
    pols_results = {}
    
    print("Running LASSO models...")
    for lag in range(1, 13):
        print(f"  LASSO lag={lag}")
        results[f'lasso{lag}c'] = lasso_rolling_window(Y, nprev, indice=1, lag=lag, alpha=alpha, type_='lasso')
        results[f'lasso{lag}p'] = lasso_rolling_window(Y, nprev, indice=2, lag=lag, alpha=alpha, type_='lasso')
    
    print("Running POLS models...")
    for lag in range(1, 13):
        print(f"  POLS lag={lag}")
        pols_results[f'pols{lag}c'] = pols_rolling_window(Y, nprev, indice=1, lag=lag, 
                                                           coefs=results[f'lasso{lag}c']['coef'])
        pols_results[f'pols{lag}p'] = pols_rolling_window(Y, nprev, indice=2, lag=lag,
                                                           coefs=results[f'lasso{lag}p']['coef'])
    
    cpi_lasso = np.column_stack([results[f'lasso{lag}c']['pred'] for lag in range(1, 13)])
    cpi_pols = np.column_stack([pols_results[f'pols{lag}c']['pred'] for lag in range(1, 13)])
    
    pce_lasso = np.column_stack([results[f'lasso{lag}p']['pred'] for lag in range(1, 13)])
    pce_pols = np.column_stack([pols_results[f'pols{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    save_forecasts(cpi_lasso, os.path.join(FORECAST_DIR, 'lasso-cpi.csv'))
    save_forecasts(pce_lasso, os.path.join(FORECAST_DIR, 'lasso-pce.csv'))
    save_forecasts(cpi_pols, os.path.join(FORECAST_DIR, 'pols-lasso-cpi.csv'))
    save_forecasts(pce_pols, os.path.join(FORECAST_DIR, 'pols-lasso-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    return results, pols_results


if __name__ == '__main__':
    main()
