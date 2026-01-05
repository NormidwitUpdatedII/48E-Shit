"""
Data Preparation Script for FRED-MD
Applies stationarity transformations as described in McCracken & Ng (2016) and 
used in Medeiros et al. (2018) "Forecasting Inflation in a Data-Rich Environment"

FRED-MD Transformation Codes:
1 = no transformation (levels)
2 = first difference: Δx_t
3 = second difference: Δ²x_t  
4 = log: log(x_t)
5 = first difference of log: Δlog(x_t) = log(x_t) - log(x_{t-1})
6 = second difference of log: Δ²log(x_t)
7 = first difference of percent change: Δ(x_t/x_{t-1} - 1)

First sample: Out-of-sample = 1990-2000 (nprev = 132)
Second sample: Out-of-sample = 2001-2025 (nprev calculated dynamically)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def apply_transformation(x, tcode):
    """
    Apply FRED-MD transformation to a series.
    
    Parameters:
    -----------
    x : array-like
        Raw data series
    tcode : int
        Transformation code (1-7)
    
    Returns:
    --------
    transformed : ndarray
        Transformed series (with NaN for lost observations)
    """
    x = np.array(x, dtype=float)
    n = len(x)
    y = np.full(n, np.nan)
    
    small = 1e-6  # Small value to handle zeros in log
    
    if tcode == 1:
        # No transformation (levels)
        y = x.copy()
        
    elif tcode == 2:
        # First difference: Δx_t = x_t - x_{t-1}
        y[1:] = np.diff(x)
        
    elif tcode == 3:
        # Second difference: Δ²x_t = Δx_t - Δx_{t-1}
        dx = np.diff(x)
        y[2:] = np.diff(dx)
        
    elif tcode == 4:
        # Log transformation
        y = np.log(np.maximum(x, small))
        
    elif tcode == 5:
        # First difference of log: Δlog(x_t)
        # This gives approximate percent change (growth rate)
        log_x = np.log(np.maximum(x, small))
        y[1:] = np.diff(log_x)
        
    elif tcode == 6:
        # Second difference of log: Δ²log(x_t)
        log_x = np.log(np.maximum(x, small))
        dlog_x = np.diff(log_x)
        y[2:] = np.diff(dlog_x)
        
    elif tcode == 7:
        # First difference of percent change
        # Δ(x_t/x_{t-1} - 1)
        pct_change = x[1:] / np.maximum(x[:-1], small) - 1
        y[2:] = np.diff(pct_change)
        
    else:
        raise ValueError(f"Unknown transformation code: {tcode}")
    
    return y


def remove_outliers(x, n_mad=10):
    """
    Remove outliers using median absolute deviation (MAD).
    Replaces outliers with NaN.
    
    Parameters:
    -----------
    x : array-like
        Data series
    n_mad : float
        Number of MADs from median to consider outlier (default 10)
    
    Returns:
    --------
    cleaned : ndarray
        Series with outliers replaced by NaN
    """
    x = np.array(x, dtype=float)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    
    if mad < 1e-10:  # Avoid division by zero
        return x
    
    # Identify outliers
    outlier_mask = np.abs(x - median) > n_mad * mad
    
    # Replace outliers with NaN
    x_clean = x.copy()
    x_clean[outlier_mask] = np.nan
    
    return x_clean


def fill_missing_values(Y):
    """
    Fill missing values using forward fill then backward fill.
    This is similar to the EM algorithm approach in the FRED-MD literature.
    
    Parameters:
    -----------
    Y : ndarray
        Data matrix with potential NaN values
    
    Returns:
    --------
    Y_filled : ndarray
        Data matrix with NaN values filled
    """
    Y_filled = Y.copy()
    
    for col in range(Y.shape[1]):
        series = Y_filled[:, col]
        
        # Forward fill
        mask = np.isnan(series)
        idx = np.where(~mask)[0]
        
        if len(idx) == 0:
            # All NaN - fill with zeros
            Y_filled[:, col] = 0
            continue
        
        # Interpolate
        Y_filled[:, col] = np.interp(
            np.arange(len(series)),
            idx,
            series[idx]
        )
    
    return Y_filled


def prepare_fred_md(input_file, output_dir=None):
    """
    Load and transform FRED-MD data for inflation forecasting.
    
    Parameters:
    -----------
    input_file : str
        Path to the raw FRED-MD CSV file
    output_dir : str, optional
        Directory to save output files (default: script directory)
    
    Returns:
    --------
    config : dict
        Configuration with nprev values for both samples
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("FRED-MD Data Preparation")
    print("Applying stationarity transformations as in the paper")
    print("=" * 60)
    
    # =========================================
    # Step 1: Load raw data with transform codes
    # =========================================
    print("\n[Step 1] Loading raw FRED-MD data...")
    
    # Read the full file
    df_raw = pd.read_csv(input_file)
    
    # First row contains "Transform:" and the transformation codes
    transform_row = df_raw.iloc[0]
    
    # Get column names (excluding date column)
    date_col = df_raw.columns[0]  # Usually 'sasdate'
    var_names = df_raw.columns[1:].tolist()
    
    # Extract transformation codes
    tcodes = {}
    for var in var_names:
        try:
            tcodes[var] = int(float(transform_row[var]))
        except:
            tcodes[var] = 5  # Default to first diff of log
    
    print(f"  Loaded {len(var_names)} variables")
    print(f"  Transform codes distribution:")
    tcode_counts = pd.Series(list(tcodes.values())).value_counts().sort_index()
    for tc, count in tcode_counts.items():
        print(f"    Code {tc}: {count} variables")
    
    # =========================================
    # Step 2: Extract data (skip transform row)
    # =========================================
    print("\n[Step 2] Extracting data...")
    
    df_data = df_raw.iloc[1:].copy()  # Skip transform row
    df_data[date_col] = pd.to_datetime(df_data[date_col], format='%m/%d/%Y')
    df_data = df_data.set_index(date_col)
    
    # Convert to numeric
    for col in var_names:
        df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
    
    print(f"  Date range: {df_data.index[0].strftime('%Y-%m-%d')} to {df_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total observations: {len(df_data)}")
    
    # =========================================
    # Step 3: Apply transformations
    # =========================================
    print("\n[Step 3] Applying stationarity transformations...")
    
    Y_raw = df_data.values
    n_obs, n_vars = Y_raw.shape
    Y_transformed = np.full((n_obs, n_vars), np.nan)
    
    for i, var in enumerate(var_names):
        tcode = tcodes[var]
        Y_transformed[:, i] = apply_transformation(Y_raw[:, i], tcode)
    
    # Count NaN rows at the beginning (due to differencing)
    # Maximum differencing is 2 (for tcode 3, 6, 7), so we lose first 2 observations
    nan_rows = 2  # Always remove first 2 rows to be safe
    
    print(f"  Transformations applied successfully")
    print(f"  Dropping first {nan_rows} observations due to differencing")
    
    # =========================================
    # Step 4: Handle outliers and missing values
    # =========================================
    print("\n[Step 4] Handling outliers and missing values...")
    
    # Remove extreme outliers (optional - comment out if not needed)
    # for i in range(n_vars):
    #     Y_transformed[:, i] = remove_outliers(Y_transformed[:, i])
    
    # Fill remaining missing values
    Y_clean = fill_missing_values(Y_transformed)
    
    # Remove initial NaN rows
    Y_clean = Y_clean[nan_rows:, :]
    dates_clean = df_data.index[nan_rows:]
    
    print(f"  Final data shape: {Y_clean.shape}")
    print(f"  Final date range: {dates_clean[0].strftime('%Y-%m-%d')} to {dates_clean[-1].strftime('%Y-%m-%d')}")
    
    # Check for any remaining NaN
    nan_count = np.sum(np.isnan(Y_clean))
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN values remain (will be filled with column mean)")
        col_means = np.nanmean(Y_clean, axis=0)
        for i in range(Y_clean.shape[1]):
            mask = np.isnan(Y_clean[:, i])
            Y_clean[mask, i] = col_means[i]
    
    # =========================================
    # Step 5: Create sample splits
    # =========================================
    print("\n[Step 5] Creating sample splits...")
    
    # Define date cutoffs
    date_1990 = pd.Timestamp('1990-01-01')
    date_2001 = pd.Timestamp('2001-01-01')
    
    # Find indices
    idx_1990 = dates_clean.searchsorted(date_1990)
    idx_2001 = dates_clean.searchsorted(date_2001)
    
    # First sample: data up to end of 2000
    first_sample_end = idx_2001
    first_sample_data = Y_clean[:first_sample_end, :]
    nprev_first = idx_2001 - idx_1990  # Out-of-sample: 1990-2000
    
    print(f"\n  === First Sample (1990-2000 out-of-sample) ===")
    print(f"  Total observations: {first_sample_data.shape[0]}")
    print(f"  Number of variables: {first_sample_data.shape[1]}")
    print(f"  Training period: {dates_clean[0].strftime('%Y-%m')} to {dates_clean[idx_1990-1].strftime('%Y-%m')}")
    print(f"  Out-of-sample period: {dates_clean[idx_1990].strftime('%Y-%m')} to {dates_clean[first_sample_end-1].strftime('%Y-%m')}")
    print(f"  nprev = {nprev_first}")
    
    # Second sample: all data (2001-2025 out-of-sample)
    second_sample_data = Y_clean
    nprev_second = len(Y_clean) - idx_2001  # Out-of-sample: 2001-2025
    
    print(f"\n  === Second Sample (2001-2025 out-of-sample) ===")
    print(f"  Total observations: {second_sample_data.shape[0]}")
    print(f"  Number of variables: {second_sample_data.shape[1]}")
    print(f"  Training period: {dates_clean[0].strftime('%Y-%m')} to {dates_clean[idx_2001-1].strftime('%Y-%m')}")
    print(f"  Out-of-sample period: {dates_clean[idx_2001].strftime('%Y-%m')} to {dates_clean[-1].strftime('%Y-%m')}")
    print(f"  nprev = {nprev_second}")
    
    # =========================================
    # Step 6: Save transformed data
    # =========================================
    print("\n[Step 6] Saving transformed data...")
    
    first_sample_dir = os.path.join(output_dir, 'first_sample')
    second_sample_dir = os.path.join(output_dir, 'second_sample')
    
    os.makedirs(first_sample_dir, exist_ok=True)
    os.makedirs(second_sample_dir, exist_ok=True)
    
    first_sample_path = os.path.join(first_sample_dir, 'rawdata.csv')
    second_sample_path = os.path.join(second_sample_dir, 'rawdata.csv')
    
    # Save without header and index (just numeric matrix like R)
    pd.DataFrame(first_sample_data).to_csv(first_sample_path, index=False, header=False)
    pd.DataFrame(second_sample_data).to_csv(second_sample_path, index=False, header=False)
    
    print(f"  First sample saved to: {first_sample_path}")
    print(f"  Second sample saved to: {second_sample_path}")
    
    # Also save variable names for reference
    var_names_path = os.path.join(output_dir, 'variable_names.txt')
    with open(var_names_path, 'w') as f:
        for i, var in enumerate(var_names):
            f.write(f"{i+1},{var},{tcodes[var]}\n")
    print(f"  Variable names saved to: {var_names_path}")
    
    # =========================================
    # Step 7: Summary
    # =========================================
    print("\n" + "=" * 60)
    print("SUMMARY - UPDATE YOUR RUN SCRIPTS WITH THESE VALUES:")
    print("=" * 60)
    print(f"\nFirst sample:")
    print(f"  nprev = {nprev_first}")
    print(f"\nSecond sample:")
    print(f"  nprev = {nprev_second}")
    print("\n" + "=" * 60)
    
    config = {
        'first_sample': {
            'nprev': nprev_first,
            'n_obs': first_sample_data.shape[0],
            'n_vars': first_sample_data.shape[1],
            'oos_start': dates_clean[idx_1990].strftime('%Y-%m-%d'),
            'oos_end': dates_clean[first_sample_end-1].strftime('%Y-%m-%d')
        },
        'second_sample': {
            'nprev': nprev_second,
            'n_obs': second_sample_data.shape[0],
            'n_vars': second_sample_data.shape[1],
            'oos_start': dates_clean[idx_2001].strftime('%Y-%m-%d'),
            'oos_end': dates_clean[-1].strftime('%Y-%m-%d')
        }
    }
    
    return config


if __name__ == '__main__':
    # Path to your downloaded FRED-MD file
    input_file = r"C:\Users\asus\Downloads\2025-11-MD.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please download FRED-MD from: https://research.stlouisfed.org/econ/mccracken/fred-databases/")
        print("And update the input_file path.")
    else:
        config = prepare_fred_md(input_file)
        
        print("\n\nTo update nprev in run scripts, the values are:")
        print(f"  first_sample/run/*.py: nprev = {config['first_sample']['nprev']}")
        print(f"  second_sample/run/*.py: nprev = {config['second_sample']['nprev']}")
