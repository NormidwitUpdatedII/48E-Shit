"""
AR Model Rolling Window Forecasts - Second Sample
"""
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path constants for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata_1990_2022.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts, add_outlier_dummy
from second_sample.functions.func_ar import ar_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    # dum[which.min(Y[,1])]=1  # Creates dummy at minimum CPI value
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 298  # Out-of-sample 2001-2025
    
    results = {}
    
    # Fixed AR models
    print("Running fixed AR models...")
    for lag in range(1, 13):
        print(f"  AR lag={lag}")
        results[f'ar{lag}c'] = ar_rolling_window(Y, nprev, indice=1, lag=lag, model_type='fixed')
        results[f'ar{lag}p'] = ar_rolling_window(Y, nprev, indice=2, lag=lag, model_type='fixed')
    
    # BIC AR models
    print("Running BIC AR models...")
    for lag in range(1, 13):
        print(f"  BIC AR lag={lag}")
        results[f'bar{lag}c'] = ar_rolling_window(Y, nprev, indice=1, lag=lag, model_type='bic')
        results[f'bar{lag}p'] = ar_rolling_window(Y, nprev, indice=2, lag=lag, model_type='bic')
    
    # Combine results
    cpi_fixed = np.column_stack([results[f'ar{lag}c']['pred'] for lag in range(1, 13)])
    cpi_bic = np.column_stack([results[f'bar{lag}c']['pred'] for lag in range(1, 13)])
    
    pce_fixed = np.column_stack([results[f'ar{lag}p']['pred'] for lag in range(1, 13)])
    pce_bic = np.column_stack([results[f'bar{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_fixed, os.path.join(FORECAST_DIR, 'ar-fixed-cpi.csv'))
    save_forecasts(pce_fixed, os.path.join(FORECAST_DIR, 'ar-fixed-pce.csv'))
    save_forecasts(cpi_bic, os.path.join(FORECAST_DIR, 'ar-bic-cpi.csv'))
    save_forecasts(pce_bic, os.path.join(FORECAST_DIR, 'ar-bic-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    
    return results


if __name__ == '__main__':
    main()
