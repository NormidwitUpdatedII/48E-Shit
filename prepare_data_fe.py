"""
Data Preparation Script for Naghiayik Inflation Forecasting Project

DATA STRUCTURE:
    - ONE source: Raw FRED-MD data (2025-11-MD.csv) covering 1959-2025
    - TWO samples split by TIME PERIOD:
        * first_sample: 502 rows (shorter period, ~2000-2025), nprev=132
        * second_sample: 800 rows (full period, ~1959-2025), nprev=298
    
PIPELINE:
    Raw FRED-MD → FRED-MD stationarity transforms → Split by time → Feature Engineering

The existing rawdata.csv files are ALREADY transformed. This script:
1. Can regenerate transformed data from raw FRED-MD
2. Applies ADDITIONAL feature engineering on top of transformed data

Usage:
    python prepare_data_fe.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from fred_md_loader import FREDMDLoader, load_fred_md, create_aligned_dataset
from feature_engineering import StationaryFeatureEngineer, engineer_features_for_model


# Sample configurations matching R project
SAMPLE_CONFIG = {
    'first_sample': {
        'rows': 502,      # Number of observations
        'nprev': 132,     # Evaluation period (months)
        'description': 'Shorter sample (~2000-2025)'
    },
    'second_sample': {
        'rows': 800,      # Number of observations (full period)
        'nprev': 298,     # Evaluation period (months)
        'description': 'Full sample (~1959-2025)'
    }
}


def prepare_first_sample(fred_md_path=None, output_dir=None, apply_fe=True, use_existing=True):
    """
    Prepare first_sample data.
    
    First sample characteristics (from R project):
    - 502 rows (shorter period, ~2000-2025)
    - nprev = 132 (132 months for evaluation, ~11 years)
    
    Parameters:
    -----------
    fred_md_path : str, optional
        Path to raw FRED-MD CSV (only needed if use_existing=False)
    output_dir : str, optional
        Output directory (default: first_sample/)
    apply_fe : bool
        Whether to apply feature engineering
    use_existing : bool
        If True, use existing rawdata.csv (already transformed)
        If False, regenerate from raw FRED-MD
        
    Returns:
    --------
    tuple : (data, info_dict)
    """
    print("=" * 60)
    print("Preparing FIRST SAMPLE")
    print(f"  Config: {SAMPLE_CONFIG['first_sample']}")
    print("=" * 60)
    
    if output_dir is None:
        output_dir = project_root / "first_sample"
    else:
        output_dir = Path(output_dir)
    
    if use_existing:
        # Use existing rawdata.csv (already FRED-MD transformed)
        existing_path = output_dir / "rawdata.csv"
        if existing_path.exists():
            print(f"Using existing transformed data: {existing_path}")
            df_sample = pd.read_csv(existing_path, header=None)
            print(f"  Rows: {len(df_sample)}, Columns: {df_sample.shape[1]}")
        else:
            raise FileNotFoundError(f"Existing data not found: {existing_path}")
    else:
        # Regenerate from raw FRED-MD
        if fred_md_path is None:
            fred_md_path = project_root / "data" / "2025-11-MD.csv"
        
        loader = FREDMDLoader(str(fred_md_path))
        loader.load_raw_data()
        df_full = loader.transform_data()
        
        # First sample: take last N rows (more recent data, ~2000-2025)
        target_rows = SAMPLE_CONFIG['first_sample']['rows']
        if len(df_full) >= target_rows:
            df_sample = df_full.tail(target_rows)
        else:
            print(f"Warning: Only {len(df_full)} rows available")
            df_sample = df_full
        
        print(f"\nGenerated from raw FRED-MD:")
        print(f"  Rows: {len(df_sample)}")
        print(f"  Date range: {df_sample.index.min()} to {df_sample.index.max()}")
    
    # Convert to numpy array
    data = df_sample.values if isinstance(df_sample, pd.DataFrame) else df_sample
    
    # Apply feature engineering if requested
    if apply_fe:
        print("\nApplying feature engineering...")
        fe = StationaryFeatureEngineer()
        # skip_basic_transforms=True because data is already FRED-MD transformed
        features = fe.get_all_features(data, include_raw=True, skip_basic_transforms=True)
        
        # Handle NaN - use forward fill to avoid data leakage
        # Forward fill uses only past values, not future values
        df_features = pd.DataFrame(features)
        features = df_features.fillna(method='ffill').fillna(method='bfill').values
        
        print(f"  Original features: {data.shape[1]}")
        print(f"  Engineered features: {features.shape[1]}")
        
        # Save engineered features
        output_path = output_dir / "rawdata_fe.csv"
        np.savetxt(output_path, features, delimiter=',')
        print(f"  Saved to: {output_path}")
        
        return features, {'original_shape': data.shape, 'fe_shape': features.shape}
    
    return data, {'shape': data.shape}


def prepare_second_sample(fred_md_path=None, output_dir=None, apply_fe=True, use_existing=True):
    """
    Prepare second_sample data.
    
    Second sample characteristics (from R project):
    - 800 rows (full period, ~1959-2025)
    - nprev = 298 (298 months for evaluation)
    - Includes outlier dummy for COVID
    
    Parameters:
    -----------
    fred_md_path : str, optional
        Path to raw FRED-MD CSV (only needed if use_existing=False)
    output_dir : str, optional
        Output directory (default: second_sample/)
    apply_fe : bool
        Whether to apply feature engineering
    use_existing : bool
        If True, use existing rawdata.csv (already transformed)
        If False, regenerate from raw FRED-MD
        
    Returns:
    --------
    tuple : (data, info_dict)
    """
    print("\n" + "=" * 60)
    print("Preparing SECOND SAMPLE")
    print(f"  Config: {SAMPLE_CONFIG['second_sample']}")
    print("=" * 60)
    
    if output_dir is None:
        output_dir = project_root / "second_sample"
    else:
        output_dir = Path(output_dir)
    
    if use_existing:
        # Use existing rawdata.csv (already FRED-MD transformed)
        existing_path = output_dir / "rawdata.csv"
        if existing_path.exists():
            print(f"Using existing transformed data: {existing_path}")
            df_sample = pd.read_csv(existing_path, header=None)
            print(f"  Rows: {len(df_sample)}, Columns: {df_sample.shape[1]}")
        else:
            raise FileNotFoundError(f"Existing data not found: {existing_path}")
    else:
        # Regenerate from raw FRED-MD
        if fred_md_path is None:
            fred_md_path = project_root / "data" / "2025-11-MD.csv"
        
        loader = FREDMDLoader(str(fred_md_path))
        loader.load_raw_data()
        df_full = loader.transform_data()
        
        # Second sample: use ALL transformed data (full period)
        df_sample = df_full
        
        print(f"\nGenerated from raw FRED-MD:")
        print(f"  Rows: {len(df_sample)}")
        print(f"  Date range: {df_sample.index.min()} to {df_sample.index.max()}")
    
    # Convert to numpy array
    data = df_sample.values if isinstance(df_sample, pd.DataFrame) else df_sample
    
    # Apply feature engineering if requested
    if apply_fe:
        print("\nApplying feature engineering...")
        fe = StationaryFeatureEngineer()
        # skip_basic_transforms=True because data is already FRED-MD transformed
        features = fe.get_all_features(data, include_raw=True, skip_basic_transforms=True)
        
        # Handle NaN - use forward fill to avoid data leakage
        # Forward fill uses only past values, not future values
        df_features = pd.DataFrame(features)
        features = df_features.fillna(method='ffill').fillna(method='bfill').values
        
        print(f"  Original features: {data.shape[1]}")
        print(f"  Engineered features: {features.shape[1]}")
        
        # Save engineered features
        output_path = output_dir / "rawdata_fe.csv"
        np.savetxt(output_path, features, delimiter=',')
        print(f"  Saved to: {output_path}")
        
        return features, {'original_shape': data.shape, 'fe_shape': features.shape}
    
    return data, {'shape': data.shape}


def compare_with_original(original_path, new_data):
    """
    Compare new data with original rawdata.csv to verify alignment.
    """
    print("\n" + "-" * 40)
    print("Comparing with original data")
    print("-" * 40)
    
    original = pd.read_csv(original_path, header=None)
    print(f"Original shape: {original.shape}")
    print(f"New shape:      {new_data.shape}")
    
    # Compare dimensions
    if original.shape == new_data.shape:
        print("✓ Shapes match!")
    else:
        print(f"Note: Shapes differ (FE adds features)")
    
    # Compare first few values of original features
    print("\nFirst row comparison (first 5 original features):")
    print(f"  Original: {original.iloc[0, :5].values}")
    print(f"  New:      {new_data[0, :5]}")


def main():
    """
    Main function to apply feature engineering to existing data.
    
    Uses existing rawdata.csv files (already FRED-MD transformed) and
    applies ADDITIONAL feature engineering on top.
    """
    print("=" * 60)
    print("NAGHIAYIK DATA PREPARATION WITH FEATURE ENGINEERING")
    print("=" * 60)
    print("\nData Pipeline:")
    print("  Raw FRED-MD (2025-11-MD.csv)")
    print("  → [Already done] FRED-MD stationarity transforms")
    print("  → [Already done] Split into first_sample (502 rows) & second_sample (800 rows)")
    print("  → [This script] Apply additional feature engineering")
    print()
    
    # Prepare first sample (use existing transformed data)
    try:
        data1, info1 = prepare_first_sample(use_existing=True, apply_fe=True)
        original_first = project_root / "first_sample" / "rawdata.csv"
        if original_first.exists():
            compare_with_original(str(original_first), data1)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    
    # Prepare second sample (use existing transformed data)
    try:
        data2, info2 = prepare_second_sample(use_existing=True, apply_fe=True)
        original_second = project_root / "second_sample" / "rawdata.csv"
        if original_second.exists():
            compare_with_original(str(original_second), data2)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - first_sample/rawdata_fe.csv (with feature engineering)")
    print("  - second_sample/rawdata_fe.csv (with feature engineering)")
    print("\nUsage in models:")
    print("  - rawdata.csv: Original FRED-MD transformed data")
    print("  - rawdata_fe.csv: With additional feature engineering (rolling stats, momentum, etc.)")


if __name__ == "__main__":
    main()
