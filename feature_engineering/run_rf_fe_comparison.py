"""
RF-FE Model Comparison Script
==============================
Runs Random Forest with Feature Engineering on both first_sample and second_sample
with 12-month forecast horizon and computes out-of-sample RMSE.

Features:
- Parallelized rolling window estimation using joblib
- 12-month forecast horizon (lag=12)
- Out-of-sample RMSE computation
- Results for both CPI (indice=1) and PCE (indice=2)

Author: Naghiayik Project
Date: January 2026
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # feature_engineering folder
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # project root
sys.path.insert(0, PROJECT_ROOT)

from utils import load_csv, calculate_errors

# Import RF-FE functions from both samples
from first_sample.run.rf_fe import rf_fe_rolling_window as rf_fe_first
from second_sample.run.rf_fe import rf_fe_rolling_window as rf_fe_second


def print_header():
    """Print script header."""
    print("=" * 70)
    print("RF-FE (Random Forest with Feature Engineering) MODEL COMPARISON")
    print("=" * 70)
    print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Forecast Horizon: 12 months")
    print("=" * 70)


def run_rf_fe_for_sample(Y, nprev, sample_name, rf_fe_func):
    """
    Run RF-FE model for a single sample.
    
    Parameters:
    -----------
    Y : np.ndarray
        Data matrix
    nprev : int
        Number of out-of-sample predictions
    sample_name : str
        Name of sample for display
    rf_fe_func : callable
        RF-FE rolling window function for the sample
        
    Returns:
    --------
    dict : Results containing predictions, actuals, and errors
    """
    results = {}
    lag = 12  # 12-month forecast horizon
    
    print(f"\n{'='*60}")
    print(f"Running RF-FE for {sample_name}")
    print(f"{'='*60}")
    print(f"Data shape: {Y.shape}")
    print(f"Out-of-sample periods: {nprev}")
    print(f"Forecast horizon: {lag} months")
    
    # CPI (indice=1)
    print(f"\n--- CPI Forecasting (indice=1) ---")
    start_time = time.time()
    cpi_result = rf_fe_func(Y, nprev, indice=1, lag=lag)
    cpi_time = time.time() - start_time
    
    print(f"  Completed in {cpi_time:.1f} seconds")
    print(f"  RMSE: {cpi_result['errors']['rmse']:.6f}")
    print(f"  MAE:  {cpi_result['errors']['mae']:.6f}")
    
    results['CPI'] = {
        'pred': cpi_result['pred'],
        'actual': cpi_result['actuals'],
        'rmse': cpi_result['errors']['rmse'],
        'mae': cpi_result['errors']['mae'],
        'time': cpi_time
    }
    
    # PCE (indice=2)
    print(f"\n--- PCE Forecasting (indice=2) ---")
    start_time = time.time()
    pce_result = rf_fe_func(Y, nprev, indice=2, lag=lag)
    pce_time = time.time() - start_time
    
    print(f"  Completed in {pce_time:.1f} seconds")
    print(f"  RMSE: {pce_result['errors']['rmse']:.6f}")
    print(f"  MAE:  {pce_result['errors']['mae']:.6f}")
    
    results['PCE'] = {
        'pred': pce_result['pred'],
        'actual': pce_result['actuals'],
        'rmse': pce_result['errors']['rmse'],
        'mae': pce_result['errors']['mae'],
        'time': pce_time
    }
    
    results['total_time'] = cpi_time + pce_time
    
    return results


def generate_results_table(first_results, second_results):
    """Generate results comparison table."""
    data = {
        'Sample': ['First Sample', 'First Sample', 'Second Sample', 'Second Sample'],
        'Target': ['CPI', 'PCE', 'CPI', 'PCE'],
        'RMSE': [
            first_results['CPI']['rmse'],
            first_results['PCE']['rmse'],
            second_results['CPI']['rmse'],
            second_results['PCE']['rmse']
        ],
        'MAE': [
            first_results['CPI']['mae'],
            first_results['PCE']['mae'],
            second_results['CPI']['mae'],
            second_results['PCE']['mae']
        ],
        'Time (s)': [
            first_results['CPI']['time'],
            first_results['PCE']['time'],
            second_results['CPI']['time'],
            second_results['PCE']['time']
        ]
    }
    
    df = pd.DataFrame(data)
    return df


def save_results(first_results, second_results, output_dir):
    """Save results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary table
    summary_df = generate_results_table(first_results, second_results)
    summary_df.to_csv(os.path.join(output_dir, 'rf_fe_summary.csv'), index=False)
    
    # Save predictions for first sample
    first_pred_df = pd.DataFrame({
        'CPI_Actual': first_results['CPI']['actual'],
        'CPI_Predicted': first_results['CPI']['pred'],
        'PCE_Actual': first_results['PCE']['actual'],
        'PCE_Predicted': first_results['PCE']['pred']
    })
    first_pred_df.to_csv(os.path.join(output_dir, 'rf_fe_first_sample_predictions.csv'), index=False)
    
    # Save predictions for second sample
    second_pred_df = pd.DataFrame({
        'CPI_Actual': second_results['CPI']['actual'],
        'CPI_Predicted': second_results['CPI']['pred'],
        'PCE_Actual': second_results['PCE']['actual'],
        'PCE_Predicted': second_results['PCE']['pred']
    })
    second_pred_df.to_csv(os.path.join(output_dir, 'rf_fe_second_sample_predictions.csv'), index=False)
    
    print(f"\n✓ Results saved to {output_dir}/")


def main():
    """Main function to run RF-FE comparison."""
    total_start = time.time()
    
    print_header()
    
    # ==========================================================================
    # Load Data
    # ==========================================================================
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    first_sample_path = os.path.join(PROJECT_ROOT, 'first_sample', 'rawdata.csv')
    second_sample_path = os.path.join(PROJECT_ROOT, 'second_sample', 'rawdata.csv')
    
    Y_first = load_csv(first_sample_path)
    Y_second = load_csv(second_sample_path)
    
    print(f"\nFirst Sample:")
    print(f"  Path: {first_sample_path}")
    print(f"  Shape: {Y_first.shape}")
    
    print(f"\nSecond Sample:")
    print(f"  Path: {second_sample_path}")
    print(f"  Shape: {Y_second.shape}")
    
    # Define nprev (out-of-sample periods)
    nprev_first = 132   # Standard for first sample
    nprev_second = 298  # Standard for second sample
    
    # ==========================================================================
    # Run RF-FE for First Sample
    # ==========================================================================
    first_results = run_rf_fe_for_sample(
        Y_first, nprev_first, "First Sample", rf_fe_first
    )
    
    # ==========================================================================
    # Run RF-FE for Second Sample
    # ==========================================================================
    second_results = run_rf_fe_for_sample(
        Y_second, nprev_second, "Second Sample", rf_fe_second
    )
    
    # ==========================================================================
    # Results Summary
    # ==========================================================================
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY - RF-FE (12-month horizon)")
    print("="*70)
    
    # Generate and display results table
    results_df = generate_results_table(first_results, second_results)
    results_df['RMSE'] = results_df['RMSE'].apply(lambda x: f"{x:.6f}")
    results_df['MAE'] = results_df['MAE'].apply(lambda x: f"{x:.6f}")
    results_df['Time (s)'] = results_df['Time (s)'].apply(lambda x: f"{x:.1f}")
    
    print("\n" + results_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("DETAILED COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n📊 First Sample (nprev={nprev_first}):")
    print(f"   CPI RMSE: {first_results['CPI']['rmse']:.6f}")
    print(f"   PCE RMSE: {first_results['PCE']['rmse']:.6f}")
    print(f"   Total Time: {first_results['total_time']:.1f}s")
    
    print(f"\n📊 Second Sample (nprev={nprev_second}):")
    print(f"   CPI RMSE: {second_results['CPI']['rmse']:.6f}")
    print(f"   PCE RMSE: {second_results['PCE']['rmse']:.6f}")
    print(f"   Total Time: {second_results['total_time']:.1f}s")
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    output_dir = os.path.join(PROJECT_ROOT, 'rf_fe_results')
    save_results(first_results, second_results, output_dir)
    
    print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    
    print("\n" + "="*70)
    print("RF-FE COMPARISON COMPLETE!")
    print("="*70)
    
    return first_results, second_results


if __name__ == '__main__':
    first_results, second_results = main()
