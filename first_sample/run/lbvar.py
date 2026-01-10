"""
LBVAR Rolling Window Forecasts
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

from utils import load_csv, save_forecasts
from first_sample.functions.func_lbvar import lbvar_rw


def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    nprev = 132
    np.random.seed(123)
    
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    # LBVAR models
    print("Running LBVAR models...")
    for lag in range(1, 13):
        print(f"  LBVAR lag={lag}")
        results[f'lbvar{lag}c'] = lbvar_rw(Y, nprev, indice=1, lag=lag)
        results[f'lbvar{lag}p'] = lbvar_rw(Y, nprev, indice=2, lag=lag)
    
    # Combine results for CPI (indice=1)
    cpi_lbvar = np.column_stack([results[f'lbvar{lag}c']['pred'] for lag in range(1, 13)])
    
    # Combine results for PCE (indice=2)
    pce_lbvar = np.column_stack([results[f'lbvar{lag}p']['pred'] for lag in range(1, 13)])
    
    # Create output directory
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    # Save forecasts
    save_forecasts(cpi_lbvar, os.path.join(FORECAST_DIR, 'lbvar-cpi.csv'))
    save_forecasts(pce_lbvar, os.path.join(FORECAST_DIR, 'lbvar-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    # Print RMSE by horizon
    print("
RMSE BY HORIZON:")
    print(f"{'Horizon':<8} {'CPI':<12} {'PCE':<12}")
    for h in range(1, 13):
        print(f"h={h:<6} {rmse_cpi.get(h, 0):<12.6f} {rmse_pce.get(h, 0):<12.6f}")
    print(f"Average: {np.mean(list(rmse_cpi.values())):.6f}  {np.mean(list(rmse_pce.values())):.6f}")

    
    return results


if __name__ == '__main__':
    main()
