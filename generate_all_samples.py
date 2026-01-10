"""
Generate All Sample Period Datasets

This script generates datasets for all 5 sample periods used in the analysis:
1. 1990-2000: Low volatility period (Great Moderation)
2. 2001-2015: Financial crisis and recovery  
3. 2016-2022: COVID-19 and inflation surge
4. 2020-2022: Pandemic and inflation surge (subset)
5. 1990-2022: Full extended sample

For each period, generates:
- rawdata_{period}.csv: FRED-MD transformed data
- rawdata_fe_{period}.csv: Feature-engineered data

Usage:
    python generate_all_samples.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from fred_md_loader import FREDMDLoader
from feature_engineering.feature_engineering import StationaryFeatureEngineer

# =============================================================================
# SAMPLE PERIOD CONFIGURATIONS
# =============================================================================

SAMPLE_PERIODS = {
    '1990_2000': {
        'start_date': '1990-01-01',
        'end_date': '2000-12-31',
        'description': 'Low volatility period (Great Moderation)',
        'nprev': 60  # 5 years evaluation
    },
    '2001_2015': {
        'start_date': '2001-01-01',
        'end_date': '2015-12-31',
        'description': 'Financial crisis and recovery',
        'nprev': 84  # 7 years evaluation
    },
    '2016_2022': {
        'start_date': '2016-01-01',
        'end_date': '2022-12-31',
        'description': 'COVID-19 and inflation surge',
        'nprev': 48  # 4 years evaluation
    },
    '2020_2022': {
        'start_date': '2020-01-01',
        'end_date': '2022-12-31',
        'description': 'Pandemic and inflation surge',
        'nprev': 24  # 2 years evaluation
    },
    '1990_2022': {
        'start_date': '1990-01-01',
        'end_date': '2022-12-31',
        'description': 'Full extended sample',
        'nprev': 132  # 11 years evaluation
    }
}

# Original sample configurations for backward compatibility
ORIGINAL_SAMPLES = {
    'first_sample': {
        'rows': 502,
        'nprev': 132,
        'description': 'Shorter sample (~2000-2025)'
    },
    'second_sample': {
        'rows': 800,
        'nprev': 298,
        'description': 'Full sample (~1959-2025)'
    }
}


def load_fred_md_data(data_path=None):
    """
    Load and transform FRED-MD data.
    
    Parameters
    ----------
    data_path : str, optional
        Path to raw FRED-MD CSV file
        
    Returns
    -------
    pd.DataFrame
        Transformed FRED-MD data with datetime index
    """
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "2025-11-MD.csv"
    
    loader = FREDMDLoader(str(data_path))
    loader.load_raw_data()
    df = loader.transform_data()
    
    return df


def generate_period_data(df_full, period_name, config, output_dir):
    """
    Generate data for a specific sample period.
    
    Parameters
    ----------
    df_full : pd.DataFrame
        Full FRED-MD transformed data
    period_name : str
        Period identifier (e.g., '1990_2000')
    config : dict
        Period configuration with start_date, end_date, nprev
    output_dir : Path
        Output directory for CSV files
        
    Returns
    -------
    dict
        Info about generated data
    """
    print(f"\n{'='*60}")
    print(f"Generating: {period_name}")
    print(f"  {config['description']}")
    print(f"  Period: {config['start_date']} to {config['end_date']}")
    print(f"{'='*60}")
    
    # Filter by date
    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    
    df_period = df_full[(df_full.index >= start_date) & (df_full.index <= end_date)]
    
    if len(df_period) == 0:
        print(f"  WARNING: No data for period {period_name}")
        return None
    
    print(f"  Observations: {len(df_period)}")
    print(f"  Actual range: {df_period.index.min()} to {df_period.index.max()}")
    
    # Save raw data (FRED-MD transformed)
    raw_filename = f"rawdata_{period_name}.csv"
    raw_path = output_dir / raw_filename
    df_period.to_csv(raw_path, header=False, index=False)
    print(f"  Saved: {raw_filename}")
    
    # Apply feature engineering
    print(f"  Applying feature engineering...")
    data = df_period.values
    fe = StationaryFeatureEngineer()
    features = fe.get_all_features(data, include_raw=True, skip_basic_transforms=True)
    
    # Handle NaN - forward fill only to avoid leakage
    df_features = pd.DataFrame(features)
    features = df_features.fillna(method='ffill').fillna(0).values
    
    print(f"    Raw features: {data.shape[1]}")
    print(f"    Engineered features: {features.shape[1]}")
    
    # Save feature-engineered data
    fe_filename = f"rawdata_fe_{period_name}.csv"
    fe_path = output_dir / fe_filename
    np.savetxt(fe_path, features, delimiter=',')
    print(f"  Saved: {fe_filename}")
    
    return {
        'period': period_name,
        'observations': len(df_period),
        'raw_features': data.shape[1],
        'fe_features': features.shape[1],
        'nprev': config['nprev'],
        'raw_path': str(raw_path),
        'fe_path': str(fe_path)
    }


def generate_original_sample_data(df_full, sample_name, config, output_dir):
    """
    Generate data for original sample configuration (backward compatibility).
    
    Parameters
    ----------
    df_full : pd.DataFrame
        Full FRED-MD transformed data
    sample_name : str
        Sample identifier ('first_sample' or 'second_sample')
    config : dict
        Sample configuration
    output_dir : Path
        Output directory
        
    Returns
    -------
    dict
        Info about generated data
    """
    print(f"\n{'='*60}")
    print(f"Generating: {sample_name} (original configuration)")
    print(f"  {config['description']}")
    print(f"  Target rows: {config['rows']}, nprev: {config['nprev']}")
    print(f"{'='*60}")
    
    target_rows = config['rows']
    
    if len(df_full) >= target_rows:
        # Take last N rows for more recent data
        df_sample = df_full.tail(target_rows)
    else:
        print(f"  WARNING: Only {len(df_full)} rows available")
        df_sample = df_full
    
    print(f"  Observations: {len(df_sample)}")
    print(f"  Date range: {df_sample.index.min()} to {df_sample.index.max()}")
    
    # Save raw data (backward compatible name)
    raw_path = output_dir / "rawdata_1990_2022.csv"
    df_sample.to_csv(raw_path, header=False, index=False)
    print(f"  Saved: rawdata.csv")
    
    # Apply feature engineering
    print(f"  Applying feature engineering...")
    data = df_sample.values
    fe = StationaryFeatureEngineer()
    features = fe.get_all_features(data, include_raw=True, skip_basic_transforms=True)
    
    df_features = pd.DataFrame(features)
    features = df_features.fillna(method='ffill').fillna(0).values
    
    print(f"    Raw features: {data.shape[1]}")
    print(f"    Engineered features: {features.shape[1]}")
    
    # Save feature-engineered data
    fe_path = output_dir / "rawdata_fe_1990_2022.csv"
    np.savetxt(fe_path, features, delimiter=',')
    print(f"  Saved: rawdata_fe.csv")
    
    return {
        'sample': sample_name,
        'observations': len(df_sample),
        'raw_features': data.shape[1],
        'fe_features': features.shape[1]
    }


def generate_all_samples(data_path=None, samples_to_generate='all'):
    """
    Generate all sample period datasets.
    
    Parameters
    ----------
    data_path : str, optional
        Path to raw FRED-MD CSV
    samples_to_generate : str or list
        'all' for all samples, or list of sample names
        
    Returns
    -------
    dict
        Summary of all generated samples
    """
    print("="*60)
    print("MULTI-SAMPLE DATA GENERATION")
    print("="*60)
    print(f"\nLoading FRED-MD data...")
    
    df_full = load_fred_md_data(data_path)
    print(f"Full data: {len(df_full)} observations")
    print(f"Date range: {df_full.index.min()} to {df_full.index.max()}")
    
    results = {'periods': [], 'original_samples': []}
    
    # Generate period-based samples for both directories
    for sample_dir in ['first_sample', 'second_sample']:
        output_dir = PROJECT_ROOT / sample_dir
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'#'*60}")
        print(f"# Processing: {sample_dir}")
        print(f"{'#'*60}")
        
        # Generate original sample data (backward compatibility)
        if sample_dir in ORIGINAL_SAMPLES:
            result = generate_original_sample_data(
                df_full, 
                sample_dir, 
                ORIGINAL_SAMPLES[sample_dir],
                output_dir
            )
            if result:
                results['original_samples'].append(result)
        
        # Generate period-based samples
        for period_name, config in SAMPLE_PERIODS.items():
            if samples_to_generate != 'all' and period_name not in samples_to_generate:
                continue
                
            result = generate_period_data(df_full, period_name, config, output_dir)
            if result:
                result['sample_dir'] = sample_dir
                results['periods'].append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    
    print("\nOriginal Samples:")
    for r in results['original_samples']:
        print(f"  {r['sample']}: {r['observations']} obs, {r['fe_features']} FE features")
    
    print("\nPeriod-Based Samples:")
    for r in results['periods']:
        print(f"  {r['sample_dir']}/{r['period']}: {r['observations']} obs, {r['fe_features']} FE features")
    
    return results


def get_sample_data_path(sample_dir, period=None, feature_engineered=False):
    """
    Get the path to a specific sample data file.
    
    Parameters
    ----------
    sample_dir : str
        'first_sample' or 'second_sample'
    period : str, optional
        Period name (e.g., '1990_2000'). If None, uses original rawdata.csv
    feature_engineered : bool
        If True, return path to FE data
        
    Returns
    -------
    Path
        Path to the data file
    """
    base_dir = PROJECT_ROOT / sample_dir
    
    if period is None:
        # Original data
        filename = "rawdata_fe_1990_2022.csv" if feature_engineered else "rawdata_1990_2022.csv"
    else:
        # Period-specific data
        filename = f"rawdata_fe_{period}.csv" if feature_engineered else f"rawdata_{period}.csv"
    
    return base_dir / filename


def load_sample_data(sample_dir, period=None, feature_engineered=True):
    """
    Load sample data as numpy array.
    
    Parameters
    ----------
    sample_dir : str
        'first_sample' or 'second_sample'
    period : str, optional
        Period name
    feature_engineered : bool
        If True, load FE data
        
    Returns
    -------
    np.ndarray
        Data matrix
    """
    path = get_sample_data_path(sample_dir, period, feature_engineered)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    data = pd.read_csv(path, header=None).values
    return data


if __name__ == "__main__":
    # Generate all samples
    results = generate_all_samples()
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print("\nGenerated files can be found in:")
    print("  - first_sample/")
    print("  - second_sample/")
    print("\nFile naming convention:")
    print("  - rawdata.csv: Original sample (backward compatible)")
    print("  - rawdata_fe.csv: FE version of original sample")
    print("  - rawdata_{period}.csv: Period-specific raw data")
    print("  - rawdata_fe_{period}.csv: Period-specific FE data")
