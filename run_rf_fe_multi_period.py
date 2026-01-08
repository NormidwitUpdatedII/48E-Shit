"""
Random Forest with Feature Engineering - Multi-Period Out-of-Sample Forecasting
================================================================================

This script runs RF-FE model across four distinct time periods:
1. Training Period: up to 1990
2. Test Period 1: 2000-2016
3. Test Period 2: 2016-2022
4. Test Period 3: 2022-2023

The model is trained on historical data and evaluated on subsequent periods
to assess out-of-sample forecasting performance.

Features:
- FRED-MD transformed macroeconomic data
- Advanced feature engineering (momentum, volatility, z-scores)
- Rolling window forecasting with parallel processing
- 3-stage feature selection (constant removal, correlation filter, variance filter)
- Optimized Random Forest hyperparameters
- Comprehensive error metrics (RMSE, MAE, MAPE)

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
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import necessary modules
from utils import embed, calculate_errors
from feature_engineering import StationaryFeatureEngineer
from feature_utils import (
    standardize_features, 
    handle_missing_values,
    apply_3stage_feature_selection
)
from feature_config import (
    CONSTANT_VARIANCE_THRESHOLD,
    CORRELATION_THRESHOLD,
    LOW_VARIANCE_THRESHOLD
)
from fred_md_loader import FREDMDLoader
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model parameters
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 20,
    'random_state': 42,
    'n_jobs': -1,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}

# Forecast horizon
FORECAST_HORIZON = 12  # 12-month ahead forecast

# Target variable names in FRED-MD data
CPI_COLUMN = 'CPIAUCSL'  # Consumer Price Index for All Urban Consumers
PCE_COLUMN = 'PCEPI'     # Personal Consumption Expenditures Price Index

# Number of parallel jobs
N_JOBS = -1

# Output directory
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'multi_period_rf_fe'


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def load_data_with_dates(data_path):
    """
    Load FRED-MD data with proper transformations and parse dates.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to FRED-MD CSV file
        
    Returns:
    --------
    tuple : (data_matrix, dates, column_names)
    """
    print(f"\nLoading FRED-MD data from: {data_path}")
    
    # Use FREDMDLoader to properly load and transform data
    loader = FREDMDLoader(data_path=str(data_path))
    
    # Load raw data
    raw_df = loader.load_raw_data()
    
    # Apply FRED-MD transformations
    print("Applying FRED-MD stationarity transformations...")
    transformed_df = loader.transform_all_variables()
    
    # Extract data matrix and dates
    dates = transformed_df.index
    data_matrix = transformed_df.values
    column_names = transformed_df.columns.tolist()
    
    print(f"  Data shape: {data_matrix.shape}")
    print(f"  Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    print(f"  Variables: {len(column_names)}")
    print(f"  Transformation codes applied: {len(loader.transformation_codes)}")
    
    return data_matrix, dates, column_names


def split_data_by_period(data_matrix, dates, train_end, test1_start, test1_end, test2_end, test3_end):
    """
    Split data into four periods.
    
    Parameters:
    -----------
    data_matrix : np.ndarray
        Full data matrix
    dates : pd.Series
        Date index
    train_end : str
        End date for training period (e.g., '1990-12-31')
    test1_start : str
        Start date for test period 1 (e.g., '2000-01-01')
    test1_end : str
        End date for test period 1 (e.g., '2016-12-31')
    test2_end : str
        End date for test period 2 (e.g., '2022-12-31')
    test3_end : str
        End date for test period 3 (e.g., '2023-12-31')
        
    Returns:
    --------
    dict : Dictionary with period data and indices
    """
    train_end_dt = pd.to_datetime(train_end)
    test1_start_dt = pd.to_datetime(test1_start)
    test1_end_dt = pd.to_datetime(test1_end)
    test2_end_dt = pd.to_datetime(test2_end)
    test3_end_dt = pd.to_datetime(test3_end)
    
    # Find indices
    train_mask = dates <= train_end_dt
    test1_mask = (dates >= test1_start_dt) & (dates <= test1_end_dt)
    test2_mask = (dates > test1_end_dt) & (dates <= test2_end_dt)
    test3_mask = (dates > test2_end_dt) & (dates <= test3_end_dt)
    
    train_idx = np.where(train_mask)[0]
    test1_idx = np.where(test1_mask)[0]
    test2_idx = np.where(test2_mask)[0]
    test3_idx = np.where(test3_mask)[0]
    
    print(f"\n{'='*70}")
    print("DATA SPLIT SUMMARY")
    print(f"{'='*70}")
    print(f"Training Period: {dates[train_idx[0]].strftime('%Y-%m')} to {dates[train_idx[-1]].strftime('%Y-%m')}")
    print(f"  Observations: {len(train_idx)}")
    print(f"\nTest Period 1 (2000-2016): {dates[test1_idx[0]].strftime('%Y-%m')} to {dates[test1_idx[-1]].strftime('%Y-%m')}")
    print(f"  Observations: {len(test1_idx)}")
    print(f"\nTest Period 2 (2016-2022): {dates[test2_idx[0]].strftime('%Y-%m')} to {dates[test2_idx[-1]].strftime('%Y-%m')}")
    print(f"  Observations: {len(test2_idx)}")
    print(f"\nTest Period 3 (2022-2023): {dates[test3_idx[0]].strftime('%Y-%m')} to {dates[test3_idx[-1]].strftime('%Y-%m')}")
    print(f"  Observations: {len(test3_idx)}")
    print(f"{'='*70}")
    
    return {
        'train': {'data': data_matrix[train_mask], 'dates': dates[train_mask], 'idx': train_idx},
        'test1': {'data': data_matrix[test1_mask], 'dates': dates[test1_mask], 'idx': test1_idx},
        'test2': {'data': data_matrix[test2_mask], 'dates': dates[test2_mask], 'idx': test2_idx},
        'test3': {'data': data_matrix[test3_mask], 'dates': dates[test3_mask], 'idx': test3_idx},
        'full_data': data_matrix,
        'full_dates': dates
    }


def run_rf_fe(Y, target_col_idx, lag):
    """
    Run Random Forest with Feature Engineering for a single forecast.
    
    Parameters:
    -----------
    Y : np.ndarray
        Training data matrix (already FRED-MD transformed)
    target_col_idx : int
        Column index of target variable (0-indexed)
    lag : int
        Forecast horizon
        
    Returns:
    --------
    dict : Dictionary with 'model', 'pred', and 'feature_info'
    """
    Y = np.array(Y)
    
    # Apply feature engineering
    fe = StationaryFeatureEngineer()
    Y_engineered = fe.get_all_features(Y, include_raw=True, skip_basic_transforms=True)
    
    # Handle NaN values
    Y_engineered = handle_missing_values(Y_engineered, strategy='mean')
    
    # Create embedded matrix with lags
    aux = embed(Y_engineered, 4 + lag)
    
    # Target from original Y
    y_target = embed(Y[:, target_col_idx].reshape(-1, 1), 4 + lag)[:, 0]
    
    # Align dimensions
    min_len = min(len(aux), len(y_target))
    aux = aux[:min_len]
    y_target = y_target[:min_len]
    
    # Features (lagged values)
    n_cols = Y_engineered.shape[1]
    X = aux[:, n_cols * lag:]
    
    # Out-of-sample features
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        aux_trimmed = aux[:, :aux.shape[1] - n_cols * (lag - 1)]
        X_out = aux_trimmed[-1, :X.shape[1]]
    
    # Adjust for lag
    y = y_target[:len(y_target) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    # Apply 3-stage feature selection
    X, selection_info = apply_3stage_feature_selection(
        X, 
        constant_threshold=CONSTANT_VARIANCE_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        variance_threshold=LOW_VARIANCE_THRESHOLD,
        verbose=False
    )
    
    # Apply selection to out-of-sample
    X_out = X_out[selection_info['combined_mask']]
    
    # Standardize
    X, scaler = standardize_features(X)
    X_out = scaler.transform(X_out.reshape(1, -1))
    
    # Train Random Forest
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X, y)
    
    # Predict
    pred = model.predict(X_out)[0]
    
    return {
        'model': model,
        'pred': pred,
        'n_features': X.shape[1]
    }


def rolling_forecast_for_period(train_data, test_data, indice, lag, target_name):
    """target_col_idx, lag, target_name):
    """
    Perform rolling window forecasting for a test period.
    
    Parameters:
    -----------
    train_data : np.ndarray
        Initial training data
    test_data : np.ndarray
        Test period data
    target_col_idx : int
        Target variable column index (0-indexed)
    lag : int
        Forecast horizon
    target_name : str
        Name of target variable (for logging)
        
    Returns:
    --------
    dict : Results with predictions, actuals, errors
    """
    n_test = len(test_data)
    predictions = []
    actuals = []
    
    print(f"\n  Rolling forecast for {target_name}")
    print(f"  Test observations: {n_test}")
    print(f"  Forecast horizon: {lag} months")
    
    # Parallel processing function
    def process_iteration(i):
        """Process a single iteration."""
        # Expanding window: use all data up to current point
        Y_window = np.vstack([train_data, test_data[:i]])
        
        # Run RF-FE
        result = run_rf_fe(Y_window, target_col_idx, lag)
        
        # Actual value (lag periods ahead from current window end)
        actual_idx = i + lag - 1
        if actual_idx < n_test:
            actual = test_data[actual_idx, target_col_idx
            actual = np.nan
        
        return result['pred'], actual, result['n_features']
    
    # Run parallel forecasting
    start_time = time.time()
    
    # We need at least lag observations to make a forecast
    n_forecasts = n_test - lag + 1
    
    print(f"  Running {n_forecasts} forecasts in parallel...")
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_iteration)(i) for i in range(1, n_forecasts + 1)
    )
    
    # Extract results
    predictions = [r[0] for r in results]
    actuals = [r[1] for r in results]
    n_features_used = results[0][2]  # Features from first iteration
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")
    print(f"  Features used: {n_features_used}")
    
    # Remove NaN actuals (if any at the end)
    valid_mask = ~np.isnan(actuals)
    predictions = np.array(predictions)[valid_mask]
    actuals = np.array(actuals)[valid_mask]
    
    # Calculate errors
    errors = calculate_errors(actuals, predictions)
    
    print(f"  RMSE: {errors['rmse']:.6f}")
    print(f"  MAE:  {errors['mae']:.6f}")
    print(f"  MAPE: {errors['mape']:.2f}%")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'errors': errors,
        'n_features': n_features_used,
        'time': elapsed
    }


def run_multi_period_analysis(data_path, train_end='1990-12-31', 
                              test1_start='2000-01-01', test1_end='2016-12-31', 
                              test2_end='2022-12-31', test3_end='2023-12-31'):
    """
    Main function to run multi-period analysis.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to data file
    train_end : str
        End date for training period
    test1_start : str
        Start date for test period 1
    test1_end : str
        End date for test period 1
    test2_end : str
        End date for test period 2
    test3_end : str
        End date for test period 3
        
    Returns:
    --------
    dict : Complete results for all periods and targets
    """
    print(f"\n{'='*70}")
    print("RANDOM FOREST WITH FEATURE ENGINEERING")
    print("Multi-Period Out-of-Sample Forecasting")
    print(f"{'='*70}")
    print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Forecast Horizon: {FORECAST_HORIZON} months")
    print(f"{'='*70}")
    
    # Load data
    data_matrix, dates, column_names = load_data_with_dates(data_path)
    
    # Find target column indices
    try:
        cpi_idx = column_names.index(CPI_COLUMN)
        pce_idx = column_names.index(PCE_COLUMN)
        print(f"\nTarget variables located:")
        print(f"  CPI ({CPI_COLUMN}): column {cpi_idx}")
        print(f"  PCE ({PCE_COLUMN}): column {pce_idx}")
    except ValueError as e:
        raise ValueError(
            f"Could not find target variables in data. "
            f"Looking for '{CPI_COLUMN}' and '{PCE_COLUMN}' in columns: {column_names[:10]}..."
        )f"FORECASTING CPI INFLATION ({CPI_COLUMN})")
    print(f"{'='*70}")
    
    print("\n--- Test Period 1: 2000-2016 ---")
    results['CPI']['test1'] = rolling_forecast_for_period(
        periods['train']['data'],
        periods['test1']['data'],
        cpi_idx,
        FORECAST_HORIZON,
        'CPI'
    )
    
    print("\n--- Test Period 2: 2016-2022 ---")
    # Train on data up to 2016 for test period 2
    train_extended1 = np.vstack([periods['train']['data'], periods['test1']['data']])
    results['CPI']['test2'] = rolling_forecast_for_period(
        train_extended1,
        periods['test2']['data'],
        cpi_idx,
        FORECAST_HORIZON,
        'CPI'
    )
    
    print("\n--- Test Period 3: 2022-2023 ---")
    # Train on data up to 2022 for test period 3
    train_extended2 = np.vstack([train_extended1, periods['test2']['data']])
    results['CPI']['test3'] = rolling_forecast_for_period(
        train_extended2,
        periods['test3']['data'],
        cpi_idx,
        FORECAST_HORIZON,
        'CPI'
    )
    
    # Run for PCE
    print(f"\n{'='*70}")
    print(f"FORECASTING PCE INFLATION ({PCE_COLUMN})")
    print(f"{'='*70}")
    
    print("\n--- Test Period 1: 2000-2016 ---")
    results['PCE']['test1'] = rolling_forecast_for_period(
        periods['train']['data'],
        periods['test1']['data'],
        pce_idx,
        FORECAST_HORIZON,
        'PCE'
    )
    
    print("\n--- Test Period 2: 2016-2022 ---")
    train_extended1 = np.vstack([periods['train']['data'], periods['test1']['data']])
    results['PCE']['test2'] = rolling_forecast_for_period(
        train_extended1,
        periods['test2']['data'],
        pce_idx,
        FORECAST_HORIZON,
        'PCE'
    )
    
    print("\n--- Test Period 3: 2022-2023 ---")
    train_extended2 = np.vstack([train_extended1, periods['test2']['data']])
    results['PCE']['test3'] = rolling_forecast_for_period(
        train_extended2,
        periods['test3']['data'],
        pce_idx]['test2'] = rolling_forecast_for_period(
        train_extended1,
        periods['test2']['data'],
        PCE_INDEX,
        FORECAST_HORIZON,
        'PCE'
    )
    
    print("\n--- Test Period 3: 2022-2023 ---")
    train_extended2 = np.vstack([train_extended1, periods['test2']['data']])
    results['PCE']['test3'] = rolling_forecast_for_period(
        train_extended2,
        periods['test3']['data'],
        PCE_INDEX,
        FORECAST_HORIZON,
        'PCE'
    )
    
    # Store period info
    results['periods'] = periods
    results['metadata'] = {
        'data_path': str(data_path),
        'train_end': train_end,
        'test1_start': test1_start,
        'test1_end': test1_end,
        'test2_end': test2_end,
        'test3_end': test3_end,
        'forecast_horizon': FORECAST_HORIZON,
        'rf_params': RF_PARAMS,
        'run_time': datetime.now().isoformat()
    }
    
    return results


def generate_summary_table(results):
    """Generate summary table of all results."""
    data = []
    
    for target in ['CPI', 'PCE']:
        for period in ['test1', 'test2', 'test3']:
            res = results[target][period]
            if period == 'test1':
                period_label = 'Test 1 (2000-2016)'
            elif period == 'test2':
                period_label = 'Test 2 (2016-2022)'
            else:
                period_label = 'Test 3 (2022-2023)'
            
            data.append({
                'Target': target,
                'Period': period_label,
                'N_Forecasts': len(res['predictions']),
                'RMSE': res['errors']['rmse'],
                'MAE': res['errors']['mae'],
                'MAPE': res['errors']['mape'],
                'Features': res['n_features'],
                'Time (s)': res['time']
            })
    
    df = pd.DataFrame(data)
    return df


def save_results(results, output_dir):
    """Save all results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    
    # Summary table
    summary = generate_summary_table(results)
    summary_path = output_dir / 'summary_table.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary table: {summary_path}")
    print("\n" + summary.to_string(index=False))
    
    # Detailed forecasts for each target and period
    for target in ['CPI', 'PCE']:
        for period in ['test1', 'test2', 'test3']:
            res = results[target][period]
            period_dates = results['periods'][period]['dates']
            
            # Get forecast dates (accounting for lag)
            forecast_dates = period_dates[FORECAST_HORIZON-1:FORECAST_HORIZON-1+len(res['predictions'])]
            
            df = pd.DataFrame({
                'Date': forecast_dates,
                'Actual': res['actuals'],
                'Predicted': res['predictions'],
                'Error': res['actuals'] - res['predictions'],
                'Abs_Error': np.abs(res['actuals'] - res['predictions'])
            })
            
            if period == 'test1':
                period_str = 'test1_2000_2016'
            elif period == 'test2':
                period_str = 'test2_2016_2022'
            else:
                period_str = 'test3_2022_2023'
            
            forecast_path = output_dir / f'{target.lower()}_{period_str}_forecasts.csv'
            df.to_csv(forecast_path, index=False)
            print(f"  {target} {period_str}: {forecast_path}")
    
    # Metadata
    metadata_path = output_dir / 'metadata.txt'
    with open(metadata_path, 'w') as f:
        f.write("RF-FE Multi-Period Forecasting Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Run Time: {results['metadata']['run_time']}\n")
        f.write(f"Data Path: {results['metadata']['data_path']}\n")
        f.write(f"Training Period: Start - {results['metadata']['train_end']}\n")
        f.write(f"Test Period 1: {results['metadata']['test1_start']} - {results['metadata']['test1_end']}\n")
        f.write(f"Test Period 2: {results['metadata']['test1_end']} - {results['metadata']['test2_end']}\n")
        f.write(f"Test Period 3: {results['metadata']['test2_end']} - {results['metadata']['test3_end']}\n")
        f.write(f"Forecast Horizon: {results['metadata']['forecast_horizon']} months\n\n")
        f.write("Random Forest Parameters:\n")
        for key, val in results['metadata']['rf_params'].items():
            f.write(f"  {key}: {val}\n")
    
    print(f"\nMetadata: {metadata_path}")
    print(f"\n{'='*70}")
    print("ALL RESULTS SAVED SUCCESSFULLY")
    print(f"{'='*70}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    ""Use the main FRED-MD data file
    data_path = SCRIPT_DIR / 'data' / '2025-11-MD.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"FRED-MD data file not found: {data_path}\n"
            f"Please ensure '2025-11-MD.csv' exists in the 'data/' directory."
        )
    
    print(f"Using FRED-MD data: {data_path}
    else:
        raise FileNotFoundError("No data file found in first_sample or second_sample directories")
    
    # Run analysis
    try:
        results = run_multi_period_analysis(
            data_path=data_path,
            train_end='1990-12-31',
            test1_start='2000-01-01',
            test1_end='2016-12-31',
            test2_end='2022-12-31',
            test3_end='2023-12-31'
        )
        
        # Save results
        save_results(results, OUTPUT_DIR)
        
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"\nTotal execution time: {sum([results[t][p]['time'] for t in ['CPI', 'PCE'] for p in ['test1', 'test2', 'test3']]):.1f} seconds")
        print(f"Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR OCCURRED")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
