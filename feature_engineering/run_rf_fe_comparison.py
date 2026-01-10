"""
RF-FE Model Comparison Script - ALL SAMPLE PERIODS
====================================================
Runs Random Forest with Feature Engineering on ALL 5 sample periods
for both first_sample and second_sample directories.

Sample Periods:
- 1990_2000: Low volatility (Great Moderation)
- 2001_2015: Financial crisis and recovery
- 2016_2022: COVID-19 and inflation surge
- 2020_2022: Pandemic subset
- 1990_2022: Full extended sample

Usage:
    python run_rf_fe_comparison.py           # Run all periods
    python run_rf_fe_comparison.py 1990_2000 # Run specific period
"""

import os
import sys
import time
import warnings
import argparse
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

# Available periods with their nprev values
SAMPLE_PERIODS = {
    '1990_2000': {'nprev': 60, 'description': 'Low volatility (Great Moderation)'},
    '2001_2015': {'nprev': 84, 'description': 'Financial crisis and recovery'},
    '2016_2022': {'nprev': 48, 'description': 'COVID-19 and inflation surge'},
    '2020_2022': {'nprev': 24, 'description': 'Pandemic subset'},
    '1990_2022': {'nprev': 132, 'description': 'Full extended sample'},
}


def print_header(periods):
    """Print script header."""
    print("=" * 70)
    print("RF-FE (Random Forest + Feature Engineering) - ALL PERIODS")
    print("=" * 70)
    print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Periods to run: {', '.join(periods)}")
    print("=" * 70)


def run_rf_fe_for_period(period, sample_dir, rf_fe_func):
    """
    Run RF-FE model for a specific period and sample.
    
    Parameters:
    -----------
    period : str
        Period name (e.g., '1990_2000')
    sample_dir : str
        'first_sample' or 'second_sample'
    rf_fe_func : callable
        RF-FE rolling window function
        
    Returns:
    --------
    dict : Results with predictions, actuals, and errors
    """
    # Construct data path
    data_path = os.path.join(PROJECT_ROOT, sample_dir, f'rawdata_{period}.csv')
    
    if not os.path.exists(data_path):
        print(f"  ⚠ Data file not found: {data_path}")
        return None
    
    Y = load_csv(data_path)
    nprev = SAMPLE_PERIODS[period]['nprev']
    
    # Adjust nprev if dataset is too small
    if len(Y) < nprev + 24:  # Need at least 24 months for training
        nprev = max(len(Y) // 3, 12)
        print(f"  ⚠ Adjusted nprev to {nprev} (dataset too small)")
    
    results = {}
    lag = 12  # 12-month forecast horizon
    
    print(f"\n  {sample_dir} | {period}")
    print(f"  Data shape: {Y.shape}, nprev: {nprev}")
    
    # CPI (indice=1)
    try:
        start_time = time.time()
        cpi_result = rf_fe_func(Y, nprev, indice=1, lag=lag)
        cpi_time = time.time() - start_time
        
        results['CPI'] = {
            'rmse': cpi_result['errors']['rmse'],
            'mae': cpi_result['errors']['mae'],
            'time': cpi_time
        }
        print(f"    CPI RMSE: {cpi_result['errors']['rmse']:.6f} ({cpi_time:.1f}s)")
    except Exception as e:
        print(f"    CPI Error: {e}")
        results['CPI'] = {'rmse': np.nan, 'mae': np.nan, 'time': 0}
    
    # PCE (indice=2)
    try:
        start_time = time.time()
        pce_result = rf_fe_func(Y, nprev, indice=2, lag=lag)
        pce_time = time.time() - start_time
        
        results['PCE'] = {
            'rmse': pce_result['errors']['rmse'],
            'mae': pce_result['errors']['mae'],
            'time': pce_time
        }
        print(f"    PCE RMSE: {pce_result['errors']['rmse']:.6f} ({pce_time:.1f}s)")
    except Exception as e:
        print(f"    PCE Error: {e}")
        results['PCE'] = {'rmse': np.nan, 'mae': np.nan, 'time': 0}
    
    return results


def run_all_periods(periods=None):
    """
    Run RF-FE for all specified periods.
    
    Parameters:
    -----------
    periods : list, optional
        List of period names. If None, run all periods.
        
    Returns:
    --------
    pd.DataFrame : Summary results
    """
    if periods is None:
        periods = list(SAMPLE_PERIODS.keys())
    
    print_header(periods)
    
    all_results = []
    total_start = time.time()
    
    for period in periods:
        print(f"\n{'='*60}")
        print(f"PERIOD: {period} - {SAMPLE_PERIODS[period]['description']}")
        print(f"{'='*60}")
        
        # First sample
        first_results = run_rf_fe_for_period(period, 'first_sample', rf_fe_first)
        if first_results:
            all_results.append({
                'Period': period,
                'Sample': 'first_sample',
                'CPI_RMSE': first_results['CPI']['rmse'],
                'PCE_RMSE': first_results['PCE']['rmse'],
                'CPI_Time': first_results['CPI']['time'],
                'PCE_Time': first_results['PCE']['time']
            })
        
        # Second sample
        second_results = run_rf_fe_for_period(period, 'second_sample', rf_fe_second)
        if second_results:
            all_results.append({
                'Period': period,
                'Sample': 'second_sample',
                'CPI_RMSE': second_results['CPI']['rmse'],
                'PCE_RMSE': second_results['PCE']['rmse'],
                'CPI_Time': second_results['CPI']['time'],
                'PCE_Time': second_results['PCE']['time']
            })
    
    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY - RF-FE (12-month horizon)")
    print("=" * 70)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    output_dir = os.path.join(PROJECT_ROOT, 'rf_fe_results')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'rf_fe_all_periods_summary.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes")
    print("\n" + "=" * 70)
    print("RF-FE COMPARISON COMPLETE!")
    print("=" * 70)
    
    return results_df


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Run RF-FE for sample periods')
    parser.add_argument('period', nargs='?', default=None,
                       help='Specific period to run (or all if not specified)')
    parser.add_argument('--list', action='store_true',
                       help='List available periods')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable periods:")
        for name, info in SAMPLE_PERIODS.items():
            print(f"  {name}: {info['description']} (nprev={info['nprev']})")
        return
    
    if args.period:
        if args.period not in SAMPLE_PERIODS:
            print(f"Error: Unknown period '{args.period}'")
            print(f"Valid periods: {list(SAMPLE_PERIODS.keys())}")
            return
        periods = [args.period]
    else:
        periods = None  # Run all
    
    results = run_all_periods(periods)
    return results


if __name__ == '__main__':
    main()
