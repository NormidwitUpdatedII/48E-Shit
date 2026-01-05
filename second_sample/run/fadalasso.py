"""
FLASSO (with dummy variable) Rolling Window Forecasts - Second Sample
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
from second_sample.functions.func_flasso import flasso_rolling_window, pols_dummy_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = 300
    alpha = 1
    
    results = {}
    pols_results = {}
    
    # LASSO with dummy models
    print("Running LASSO with dummy models...")
    for lag in range(1, 13):
        print(f"  FLASSO lag={lag}")
        results[f'flasso{lag}c'] = flasso_rolling_window(Y, nprev, indice=1, lag=lag, alpha=alpha, type_='lasso')
        results[f'flasso{lag}p'] = flasso_rolling_window(Y, nprev, indice=2, lag=lag, alpha=alpha, type_='lasso')
    
    # POLS models
    print("Running POLS models...")
    for lag in range(1, 13):
        print(f"  POLS lag={lag}")
        pols_results[f'pols{lag}c'] = pols_dummy_rolling_window(Y, nprev, indice=1, lag=lag, 
                                                                 coefs=results[f'flasso{lag}c']['coef'])
        pols_results[f'pols{lag}p'] = pols_dummy_rolling_window(Y, nprev, indice=2, lag=lag,
                                                                 coefs=results[f'flasso{lag}p']['coef'])
    
    # Combine results
    cpi_flasso = np.column_stack([results[f'flasso{lag}c']['pred'] for lag in range(1, 13)])
    cpi_pols = np.column_stack([pols_results[f'pols{lag}c']['pred'] for lag in range(1, 13)])
    
    pce_flasso = np.column_stack([results[f'flasso{lag}p']['pred'] for lag in range(1, 13)])
    pce_pols = np.column_stack([pols_results[f'pols{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_flasso, os.path.join(FORECAST_DIR, 'flasso-cpi.csv'))
    save_forecasts(pce_flasso, os.path.join(FORECAST_DIR, 'flasso-pce.csv'))
    save_forecasts(cpi_pols, os.path.join(FORECAST_DIR, 'pols-flasso-cpi.csv'))
    save_forecasts(pce_pols, os.path.join(FORECAST_DIR, 'pols-flasso-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    
    return results, pols_results


if __name__ == '__main__':
    main()
