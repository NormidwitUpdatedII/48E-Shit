"""
LBVAR Rolling Window Forecasts - Second Sample
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path constants for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts
from second_sample.functions.func_lbvar import lbvar_rw


def main():
    Y = load_csv(DATA_PATH)
    
    nprev = 300
    np.random.seed(123)
    
    results = {}
    
    print("Running LBVAR models...")
    for lag in range(1, 13):
        print(f"  LBVAR lag={lag}")
        results[f'lbvar{lag}c'] = lbvar_rw(Y, nprev, indice=1, lag=lag)
        results[f'lbvar{lag}p'] = lbvar_rw(Y, nprev, indice=2, lag=lag)
    
    cpi_lbvar = np.column_stack([results[f'lbvar{lag}c']['pred'] for lag in range(1, 13)])
    pce_lbvar = np.column_stack([results[f'lbvar{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    save_forecasts(cpi_lbvar, os.path.join(FORECAST_DIR, 'lbvar-cpi.csv'))
    save_forecasts(pce_lbvar, os.path.join(FORECAST_DIR, 'lbvar-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    return results


if __name__ == '__main__':
    main()
