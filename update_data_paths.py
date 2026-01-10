"""
Update all model runner scripts to use period-specific data files

This script replaces references to:
- rawdata.csv -> rawdata_1990_2022.csv (or configurable period)
- rawdata_fe.csv -> rawdata_fe_1990_2022.csv (or configurable period)

Run from project root:
    python update_data_paths.py
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Default period to use
DEFAULT_PERIOD = '1990_2022'

def update_file(file_path, period=DEFAULT_PERIOD):
    """Update data path references in a single file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original_content = content
    
    # Replace rawdata.csv references
    # Pattern: 'rawdata.csv' -> 'rawdata_{period}.csv'
    content = re.sub(
        r"(['\"])rawdata\.csv(['\"])",
        rf"\1rawdata_{period}.csv\2",
        content
    )
    
    # Replace rawdata_fe.csv references
    # Pattern: 'rawdata_fe.csv' -> 'rawdata_fe_{period}.csv'
    content = re.sub(
        r"(['\"])rawdata_fe\.csv(['\"])",
        rf"\1rawdata_fe_{period}.csv\2",
        content
    )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def update_all_files(period=DEFAULT_PERIOD):
    """Update all Python files in the project."""
    print("="*60)
    print(f"UPDATING DATA PATHS TO USE: {period}")
    print("="*60)
    
    directories = [
        PROJECT_ROOT / 'first_sample' / 'run',
        PROJECT_ROOT / 'second_sample' / 'run',
        PROJECT_ROOT / 'first_sample' / 'functions',
        PROJECT_ROOT / 'second_sample' / 'functions',
        PROJECT_ROOT,
        PROJECT_ROOT / 'feature_engineering',
    ]
    
    updated_files = []
    
    for directory in directories:
        if not directory.exists():
            continue
        
        if directory == PROJECT_ROOT:
            py_files = list(directory.glob('*.py'))
        else:
            py_files = list(directory.glob('*.py'))
        
        for file_path in py_files:
            if file_path.name == 'update_data_paths.py':
                continue
            
            if update_file(file_path, period):
                updated_files.append(file_path)
                print(f"  Updated: {file_path.relative_to(PROJECT_ROOT)}")
    
    print(f"\nTotal files updated: {len(updated_files)}")
    return updated_files


if __name__ == "__main__":
    import sys
    
    period = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERIOD
    
    print(f"\nUsing period: {period}")
    print("Valid periods: 1990_2000, 2001_2015, 2016_2022, 2020_2022, 1990_2022\n")
    
    updated = update_all_files(period)
    
    print("\n" + "="*60)
    print("UPDATE COMPLETE")
    print("="*60)
