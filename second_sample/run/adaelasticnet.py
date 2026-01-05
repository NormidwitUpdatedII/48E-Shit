"""
Adaptive Elastic Net forecasting run script for second sample
Runs adaptive elastic net with dummy variable for all lags and both indices
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

from second_sample.functions.func_lasso import lasso_rolling_window
from utils import \1, add_outlier_dummy

def main():
    # Load data
    Y = load_csv(DATA_PATH)
    
    # Add dummy variable for outliers (COVID period) as in the R code
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 298  # Out-of-sample 2001-2025
    alpha = 0.5  # Elastic Net
    
    print("Running Adaptive Elastic Net forecasts (second sample with dummy)...")
    
    # Storage for results
    cpi_results = []
    pce_results = []
    
    # Run for each lag
    for lag in range(1, 13):
        print(f"  Lag {lag}/12...")
        
        # CPI (indice=1)
        result_cpi = lasso_rolling_window(Y, nprev, indice=1, lag=lag, 
                                          alpha=alpha, type_='adalasso')
        cpi_results.append(result_cpi['predictions'])
        
        # PCE (indice=2)
        result_pce = lasso_rolling_window(Y, nprev, indice=2, lag=lag,
                                          alpha=alpha, type_='adalasso')
        pce_results.append(result_pce['predictions'])
    
    # Combine results
    cpi = np.column_stack(cpi_results)
    pce = np.column_stack(pce_results)
    
    # Save forecasts
    os.makedirs(FORECAST_DIR, exist_ok=True)
    
    save_forecasts(cpi, os.path.join(FORECAST_DIR, 'adaelasticnet-cpi.csv'))
    save_forecasts(pce, os.path.join(FORECAST_DIR, 'adaelasticnet-pce.csv'))
    
    print(f"Adaptive Elastic Net forecasts saved to {FORECAST_DIR}")

if __name__ == "__main__":
    main()
