"""
Adaptive LASSO + Random Forest forecasting run script for second sample
Runs adalasso-rf with dummy variable for all lags (PCE only based on R code)
"""
import sys
import os
import numpy as np
import pandas as pd

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Path constants for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from second_sample.functions.func_adalassorf import lasso_rolling_window
from utils import \1, add_outlier_dummy

def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 180  # Paper specification for second sample
    
    print("Running AdaLASSO+RF forecasts (second sample with dummy)...")
    
    # Storage for results - Note: R code only runs PCE for this model
    pce_results = []
    
    # Run for each lag (only PCE based on original R code)
    for lag in range(1, 13):
        print(f"  Lag {lag}/12...")
        
        # PCE (indice=2) only
        result_pce = lasso_rolling_window(Y, nprev, indice=2, lag=lag,
                                          type_='adalasso')
        pce_results.append(result_pce['predictions'])
    
    # Combine results
    pce = np.column_stack(pce_results)
    
    # Save forecasts
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    save_forecasts(pce, os.path.join(FORECAST_DIR, 'adalassorf-pce.csv'))
    
    print(f"AdaLASSO+RF forecasts saved to {FORECAST_DIR}")

if __name__ == "__main__":
    main()
