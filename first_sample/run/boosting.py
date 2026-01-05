"""
Boosting Rolling Window Forecasts
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
from first_sample.functions.func_boosting import boosting_rolling_window


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = 132
    np.random.seed(123)
    
    results = {}
    
    # Boosting models
    print("Running Boosting models...")
    for lag in range(1, 13):
        print(f"  Boosting lag={lag}")
        results[f'boost{lag}c'] = boosting_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'boost{lag}p'] = boosting_rolling_window(Y, nprev, indice=2, lag=lag)
    
    # Combine results for CPI (indice=1)
    cpi_boost = np.column_stack([results[f'boost{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE (indice=2)
    pce_boost = np.column_stack([results[f'boost{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_boost, os.path.join(FORECAST_DIR, 'boosting-cpi.csv'))
    save_forecasts(pce_boost, os.path.join(FORECAST_DIR, 'boosting-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    
    return results


if __name__ == '__main__':
    main()
