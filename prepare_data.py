"""
Data Preparation Script
Splits the FRED-MD data into first_sample and second_sample datasets.

First sample: Out-of-sample = 1990-2000 (training = data before 1990)
Second sample: Out-of-sample = 2001-2025 (training = data before 2001)
"""

import pandas as pd
import numpy as np
import os


def prepare_data(input_file):
    """
    Load and prepare FRED-MD data for forecasting.
    
    Parameters:
    -----------
    input_file : str
        Path to the raw FRED-MD CSV file
    """
    # Read the CSV file
    # Skip the transform row (row 2) and use the header (row 1)
    df = pd.read_csv(input_file, skiprows=[1])  # Skip the transform codes row
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df['sasdate'].iloc[0]} to {df['sasdate'].iloc[-1]}")
    
    # Parse dates
    df['date'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y')
    df = df.set_index('date')
    df = df.drop('sasdate', axis=1)
    
    # Get the numeric data (all columns except date)
    Y = df.values
    
    print(f"\nData matrix shape: {Y.shape}")
    print(f"Number of observations: {Y.shape[0]}")
    print(f"Number of variables: {Y.shape[1]}")
    
    # Define date cutoffs
    date_1990 = pd.Timestamp('1990-01-01')
    date_2001 = pd.Timestamp('2001-01-01')
    
    # Find indices
    idx_1990 = df.index.searchsorted(date_1990)
    idx_2001 = df.index.searchsorted(date_2001)
    
    print(f"\nDate indices:")
    print(f"  Index for 1990-01-01: {idx_1990} (date: {df.index[idx_1990]})")
    print(f"  Index for 2001-01-01: {idx_2001} (date: {df.index[idx_2001]})")
    
    # First sample: All data up to end of 2000
    # Training: before 1990, Out-of-sample: 1990-2000
    first_sample_end = idx_2001  # Data through end of 2000
    first_sample_data = Y[:first_sample_end, :]
    nprev_first = idx_2001 - idx_1990  # Number of out-of-sample observations (1990-2000)
    
    print(f"\n=== First Sample ===")
    print(f"  Total observations: {first_sample_data.shape[0]}")
    print(f"  Training observations (before 1990): {idx_1990}")
    print(f"  Out-of-sample observations (1990-2000): {nprev_first}")
    print(f"  Date range: {df.index[0]} to {df.index[first_sample_end-1]}")
    
    # Second sample: All data up to 2025
    # Training: before 2001, Out-of-sample: 2001-2025
    second_sample_data = Y  # All data
    nprev_second = Y.shape[0] - idx_2001  # Number of out-of-sample observations (2001-2025)
    
    print(f"\n=== Second Sample ===")
    print(f"  Total observations: {second_sample_data.shape[0]}")
    print(f"  Training observations (before 2001): {idx_2001}")
    print(f"  Out-of-sample observations (2001-2025): {nprev_second}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create output directories
    first_sample_dir = os.path.join(os.path.dirname(__file__), 'first_sample')
    second_sample_dir = os.path.join(os.path.dirname(__file__), 'second_sample')
    
    # Save data files
    first_sample_path = os.path.join(first_sample_dir, 'rawdata.csv')
    second_sample_path = os.path.join(second_sample_dir, 'rawdata.csv')
    
    # Save without header and index (just the numeric data matrix)
    pd.DataFrame(first_sample_data).to_csv(first_sample_path, index=False, header=False)
    pd.DataFrame(second_sample_data).to_csv(second_sample_path, index=False, header=False)
    
    print(f"\n=== Files Saved ===")
    print(f"  First sample: {first_sample_path}")
    print(f"  Second sample: {second_sample_path}")
    
    # Also save nprev values to a config file for reference
    config = {
        'first_sample': {
            'nprev': nprev_first,
            'training_end': str(df.index[idx_1990-1]),
            'oos_start': str(df.index[idx_1990]),
            'oos_end': str(df.index[first_sample_end-1])
        },
        'second_sample': {
            'nprev': nprev_second,
            'training_end': str(df.index[idx_2001-1]),
            'oos_start': str(df.index[idx_2001]),
            'oos_end': str(df.index[-1])
        }
    }
    
    print(f"\n=== Configuration Values ===")
    print(f"First sample nprev (for run scripts): {nprev_first}")
    print(f"Second sample nprev (for run scripts): {nprev_second}")
    
    return config


if __name__ == '__main__':
    input_file = r"C:\Users\asus\Downloads\2025-11-MD.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please update the input_file path to point to your data file.")
    else:
        config = prepare_data(input_file)
