"""
Template for run files that output RMSE by horizon (h1-h12)

This template shows the pattern that all run files should follow
to print RMSE for all 12 horizons.

Key features:
1. Run model for all horizons h=1 to h=12
2. Print RMSE summary table
3. Save RMSE to CSV file
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


def run_model_template(Y, nprev, model_name, model_func):
    """
    Template function to run any model for all horizons h1-h12 and print RMSE.
    
    Parameters:
    -----------
    Y : ndarray
        Input data matrix
    nprev : int
        Number of out-of-sample predictions
    model_name : str
        Name of the model for display
    model_func : callable
        Function that takes (Y, nprev, indice, lag) and returns dict with 'pred' and 'errors'
        
    Returns:
    --------
    tuple : (results, rmse_cpi, rmse_pce)
    """
    results = {}
    rmse_cpi = {}
    rmse_pce = {}
    
    print("=" * 60)
    print(f"{model_name} - RMSE by Horizon (h1-h12)")
    print("=" * 60)
    
    for lag in range(1, 13):
        print(f"  Running h={lag}...")
        
        # Run for CPI (indice=1)
        results[f'{model_name}{lag}c'] = model_func(Y, nprev, indice=1, lag=lag)
        rmse_cpi[lag] = results[f'{model_name}{lag}c']['errors']['rmse']
        
        # Run for PCE (indice=2)
        results[f'{model_name}{lag}p'] = model_func(Y, nprev, indice=2, lag=lag)
        rmse_pce[lag] = results[f'{model_name}{lag}p']['errors']['rmse']
    
    # Print RMSE summary by horizon
    print("\n" + "=" * 60)
    print("RMSE BY HORIZON")
    print("=" * 60)
    print(f"{'Horizon':<10} {'CPI RMSE':<15} {'PCE RMSE':<15}")
    print("-" * 40)
    for h in range(1, 13):
        print(f"h={h:<7} {rmse_cpi[h]:<15.6f} {rmse_pce[h]:<15.6f}")
    
    # Print average RMSE
    avg_cpi = np.mean(list(rmse_cpi.values()))
    avg_pce = np.mean(list(rmse_pce.values()))
    print("-" * 40)
    print(f"{'Average':<10} {avg_cpi:<15.6f} {avg_pce:<15.6f}")
    print("=" * 60)
    
    return results, rmse_cpi, rmse_pce


def save_results_template(results, rmse_cpi, rmse_pce, model_name, forecast_dir):
    """
    Template function to save forecasts and RMSE to CSV files.
    
    Parameters:
    -----------
    results : dict
        Model results
    rmse_cpi : dict
        RMSE by horizon for CPI
    rmse_pce : dict
        RMSE by horizon for PCE
    model_name : str
        Name of the model
    forecast_dir : str
        Directory to save files
    """
    os.makedirs(forecast_dir, exist_ok=True)
    
    # Combine forecasts
    cpi_forecasts = np.column_stack([results[f'{model_name}{lag}c']['pred'] for lag in range(1, 13)])
    pce_forecasts = np.column_stack([results[f'{model_name}{lag}p']['pred'] for lag in range(1, 13)])
    
    # Save forecasts
    save_forecasts(cpi_forecasts, os.path.join(forecast_dir, f'{model_name}-cpi.csv'))
    save_forecasts(pce_forecasts, os.path.join(forecast_dir, f'{model_name}-pce.csv'))
    
    # Save RMSE summary
    rmse_df = pd.DataFrame({
        'Horizon': list(range(1, 13)),
        'CPI_RMSE': [rmse_cpi[h] for h in range(1, 13)],
        'PCE_RMSE': [rmse_pce[h] for h in range(1, 13)]
    })
    rmse_df.to_csv(os.path.join(forecast_dir, f'{model_name}-rmse.csv'), index=False)
    
    print(f"\nForecasts saved to {forecast_dir}")
    print(f"RMSE summary saved to {os.path.join(forecast_dir, f'{model_name}-rmse.csv')}")


# Example usage:
if __name__ == '__main__':
    print("This is a template file. See individual model run files for usage.")
    print("\nExpected output format:")
    print("=" * 60)
    print("Model Name - RMSE by Horizon (h1-h12)")
    print("=" * 60)
    print(f"{'Horizon':<10} {'CPI RMSE':<15} {'PCE RMSE':<15}")
    print("-" * 40)
    for h in range(1, 13):
        print(f"h={h:<7} {'0.123456':<15} {'0.123456':<15}")
    print("-" * 40)
    print(f"{'Average':<10} {'0.123456':<15} {'0.123456':<15}")
    print("=" * 60)
