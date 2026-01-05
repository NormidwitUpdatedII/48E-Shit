"""
Random Forest Rolling Window Forecasts
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
from first_sample.functions.func_rf import rf_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = 132
    np.random.seed(123)
    
    results = {}
    
    # Random Forest models
    print("Running Random Forest models...")
    for lag in range(1, 13):
        print(f"  RF lag={lag}")
        results[f'rf{lag}c'] = rf_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'rf{lag}p'] = rf_rolling_window(Y, nprev, indice=2, lag=lag)
    
    # Combine results for CPI (indice=1)
    cpi_rf = np.column_stack([results[f'rf{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE (indice=2)
    pce_rf = np.column_stack([results[f'rf{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_rf, os.path.join(FORECAST_DIR, 'rf-cpi.csv'))
    save_forecasts(pce_rf, os.path.join(FORECAST_DIR, 'rf-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    
    return results


if __name__ == '__main__':
    main()
