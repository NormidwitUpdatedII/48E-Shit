"""
SCAD Rolling Window Forecasts - Second Sample
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
from second_sample.functions.func_scad import scad_rolling_window


def main():
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 298  # Out-of-sample 2001-2025
    np.random.seed(123)
    
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    print("Running SCAD models...")
    for lag in range(1, 13):
        print(f"  SCAD lag={lag}")
        results[f'scad{lag}c'] = scad_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'scad{lag}p'] = scad_rolling_window(Y, nprev, indice=2, lag=lag)
        rmse_cpi[lag] = results[f'scad{lag}c']['errors']['rmse']
        rmse_pce[lag] = results[f'scad{lag}p']['errors']['rmse']
    
    cpi_scad = np.column_stack([results[f'scad{lag}c']['pred'] for lag in range(1, 13)])
    pce_scad = np.column_stack([results[f'scad{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    save_forecasts(cpi_scad, os.path.join(FORECAST_DIR, 'scad-cpi.csv'))
    save_forecasts(pce_scad, os.path.join(FORECAST_DIR, 'scad-pce.csv'))
    
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
