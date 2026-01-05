"""
RF-LASSO Rolling Window Forecasts - Second Sample
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

from utils import \1, add_outlier_dummynd_sample.functions.func_rflasso import rflasso_rolling_window


def main():
    # Load data
    \1
    
    # Add dummy variable for outliers (COVID period) as in the R code
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 180  # Paper specification for second sample
    np.random.seed(123)
    
    results = {}
    
    # RF-LASSO models
    print("Running RF-LASSO models...")
    for lag in range(1, 13):
        print(f"  RFLASSO lag={lag}")
        results[f'rflasso{lag}c'] = rflasso_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'rflasso{lag}p'] = rflasso_rolling_window(Y, nprev, indice=2, lag=lag)
    
    # Combine results
    cpi_rflasso = np.column_stack([results[f'rflasso{lag}c']['pred'] for lag in range(1, 13)])
    pce_rflasso = np.column_stack([results[f'rflasso{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_rflasso, os.path.join(FORECAST_DIR, 'rflasso-cpi.csv'))
    save_forecasts(pce_rflasso, os.path.join(FORECAST_DIR, 'rflasso-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    
    return results


if __name__ == '__main__':
    main()
