"""
Setup Validation Script for RF-FE Multi-Period Forecasting
============================================================

This script checks that all dependencies and data files are in place
before running the main RF-FE multi-period script.

Run this script FIRST to ensure everything is set up correctly:
    python validate_rf_fe_setup.py

Author: Naghiayik Project
Date: January 2026
"""

import sys
import os
from pathlib import Path

def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

def check_dependencies():
    """Check if all required packages are installed."""
    print_section("CHECKING DEPENDENCIES")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    optional_packages = {
        'tqdm': 'tqdm',
        'statsmodels': 'statsmodels'
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            installed.append(package_name)
            print(f"✓ {package_name} - installed")
        except ImportError:
            missing.append(package_name)
            print(f"✗ {package_name} - MISSING (required)")
    
    print("\nOptional packages:")
    for import_name, package_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} - installed")
        except ImportError:
            print(f"○ {package_name} - not installed (optional)")
    
    if missing:
        print(f"\n⚠️  MISSING REQUIRED PACKAGES: {', '.join(missing)}")
        print(f"\nTo install, run:")
        print(f"  pip install {' '.join(missing)}")
        print(f"\nOr install from requirements file:")
        print(f"  pip install -r requirements_rf_fe.txt")
        return False
    else:
        print(f"\n✓ All required dependencies are installed!")
        return True

def check_project_files():
    """Check if all required project files exist."""
    print_section("CHECKING PROJECT FILES")
    
    script_dir = Path(__file__).parent
    
    required_files = [
        'utils.py',
        'feature_engineering.py',
        'feature_utils.py',
        'feature_config.py',
        'run_rf_fe_multi_period.py'
    ]
    
    missing = []
    
    for filename in required_files:
        filepath = script_dir / filename
        if filepath.exists():
            print(f"✓ {filename} - found")
        else:
            print(f"✗ {filename} - MISSING")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠️  MISSING FILES: {', '.join(missing)}")
        return False
    else:
        print(f"\n✓ All required project files found!")
        return True

def check_data_files():
    """Check if data files exist."""
    print_section("CHECKING DATA FILES")
    
    script_dir = Path(__file__).parent
    
    data_locations = [
        script_dir / 'first_sample' / 'rawdata.csv',
        script_dir / 'second_sample' / 'rawdata.csv'
    ]
    
    found_data = []
    
    for data_path in data_locations:
        if data_path.exists():
            print(f"✓ {data_path.relative_to(script_dir)} - found")
            found_data.append(data_path)
            
            # Check file size and basic structure
            try:
                import pandas as pd
                df = pd.read_csv(data_path, nrows=5)
                print(f"    Columns: {len(df.columns)}")
                print(f"    Sample shape: {df.shape}")
            except Exception as e:
                print(f"    ⚠️  Warning: Could not read file: {e}")
        else:
            print(f"○ {data_path.relative_to(script_dir)} - not found")
    
    if not found_data:
        print(f"\n⚠️  NO DATA FILES FOUND!")
        print(f"\nExpected data locations:")
        for loc in data_locations:
            print(f"  - {loc.relative_to(script_dir)}")
        return False
    else:
        print(f"\n✓ Data file(s) found: {len(found_data)}")
        print(f"\nWill use: {found_data[0].relative_to(script_dir)}")
        return True

def check_module_imports():
    """Check if custom modules can be imported."""
    print_section("CHECKING MODULE IMPORTS")
    
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    
    modules_to_test = [
        ('utils', ['embed', 'calculate_errors']),
        ('feature_engineering', ['StationaryFeatureEngineer']),
        ('feature_utils', ['standardize_features', 'handle_missing_values', 'apply_3stage_feature_selection']),
        ('feature_config', ['CONSTANT_VARIANCE_THRESHOLD', 'CORRELATION_THRESHOLD'])
    ]
    
    all_ok = True
    
    for module_name, items in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✓ {module_name} - imports successfully")
            
            # Check specific items
            missing_items = []
            for item in items:
                if not hasattr(module, item):
                    missing_items.append(item)
            
            if missing_items:
                print(f"  ⚠️  Missing: {', '.join(missing_items)}")
                all_ok = False
            else:
                print(f"    All required items present")
                
        except Exception as e:
            print(f"✗ {module_name} - IMPORT ERROR: {e}")
            all_ok = False
    
    if all_ok:
        print(f"\n✓ All modules import successfully!")
        return True
    else:
        print(f"\n⚠️  Some module imports failed!")
        return False

def estimate_runtime():
    """Provide runtime estimates."""
    print_section("RUNTIME ESTIMATES")
    
    print("Expected runtime for run_rf_fe_multi_period.py:")
    print("\nDepends on:")
    print("  - Number of observations in each period")
    print("  - Number of CPU cores (parallel processing)")
    print("  - Feature engineering complexity")
    print("\nTypical estimates:")
    print("  - CPI forecasting (both periods): 10-30 minutes")
    print("  - PCE forecasting (both periods): 10-30 minutes")
    print("  - Total runtime: 20-60 minutes")
    print("\nThe script uses parallel processing to speed up computation.")
    print("More CPU cores = faster execution.")

def main():
    """Main validation routine."""
    print(f"\n{'='*70}")
    print("RF-FE MULTI-PERIOD SETUP VALIDATION")
    print(f"{'='*70}")
    print(f"Validation script location: {Path(__file__).parent}")
    
    # Run all checks
    checks = [
        ("Dependencies", check_dependencies),
        ("Project Files", check_project_files),
        ("Data Files", check_data_files),
        ("Module Imports", check_module_imports)
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n⚠️  Error during {check_name} check: {e}")
            results[check_name] = False
    
    # Runtime estimates
    estimate_runtime()
    
    # Final summary
    print_section("VALIDATION SUMMARY")
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {check_name}")
    
    print(f"\n{'='*70}")
    
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nYou are ready to run the main script:")
        print("  python run_rf_fe_multi_period.py")
        print("\nNote: The script will take 20-60 minutes to complete.")
        print("      Results will be saved to: results/multi_period_rf_fe/")
        return 0
    else:
        print("✗ SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before running the main script.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
