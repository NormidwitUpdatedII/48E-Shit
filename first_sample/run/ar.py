"""
AR Model Rolling Window Forecasts
"""
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import load_csv, save_forecasts
from first_sample.functions.func_ar import ar_rolling_window

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'rawdata.csv')
FORECAST_DIR = os.path.join(SCRIPT_DIR, '..', 'forecasts')


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = 132
    
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
    
    # Combine results for CPI (indice=1)
    cpi_fixed = np.column_stack([results[f'ar{lag}c']['pred'] for lag in range(1, 13)])
    cpi_bic = np.column_stack([results[f'bar{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE (indice=2)
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
