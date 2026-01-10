"""
CSR Rolling Window Forecasts - Second Sample
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path constants for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata_1990_2022.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts, add_outlier_dummy
from second_sample.functions.func_csr import csr_rolling_window


def main():
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 298  # Out-of-sample 2001-2025
    np.random.seed(123)
    
    results = {}
    
    print("Running CSR models...")
    for lag in range(1, 13):
        print(f"  CSR lag={lag}")
        results[f'csr{lag}c'] = csr_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'csr{lag}p'] = csr_rolling_window(Y, nprev, indice=2, lag=lag)
    
    cpi_csr = np.column_stack([results[f'csr{lag}c']['pred'] for lag in range(1, 13)])
    pce_csr = np.column_stack([results[f'csr{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    save_forecasts(cpi_csr, os.path.join(FORECAST_DIR, 'csr-cpi.csv'))
    save_forecasts(pce_csr, os.path.join(FORECAST_DIR, 'csr-pce.csv'))
    
    print(f"Done! Forecasts saved to {FORECAST_DIR}")
    return results


if __name__ == '__main__':
    main()
