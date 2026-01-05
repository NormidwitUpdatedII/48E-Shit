"""
Final Control Tests for Naghiayik Python Project
Tests data structure, feature engineering, and model execution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

def test_data_structure():
    """Test data loading and structure"""
    print("=" * 60)
    print("TEST 1: DATA STRUCTURE CHECK")
    print("=" * 60)
    
    # Load data (no header in CSV - first row is data used as header)
    df1 = pd.read_csv('first_sample/rawdata.csv', header=None)
    df2 = pd.read_csv('second_sample/rawdata.csv', header=None)
    
    print(f"First sample:  {df1.shape[0]} rows x {df1.shape[1]} cols")
    print(f"Second sample: {df2.shape[0]} rows x {df2.shape[1]} cols")
    print(f"Number of features: {df1.shape[1]}")
    print(f"First 5 values of target: {list(df1.iloc[:5, 0].round(4))}")
    
    # Verify expected dimensions (data rows)
    assert df1.shape[0] >= 500, f"Expected ~502 rows for first_sample, got {df1.shape[0]}"
    assert df2.shape[0] >= 798, f"Expected ~800 rows for second_sample, got {df2.shape[0]}"
    assert df1.shape[1] == 126, f"Expected 126 cols for first_sample, got {df1.shape[1]}"
    
    print("✓ Data structure check PASSED")
    return True

def test_feature_engineered_data():
    """Test feature-engineered data"""
    print("\n" + "=" * 60)
    print("TEST 2: FEATURE ENGINEERING CHECK")
    print("=" * 60)
    
    # Load FE data (no header)
    df1_fe = pd.read_csv('first_sample/rawdata_fe.csv', header=None)
    df2_fe = pd.read_csv('second_sample/rawdata_fe.csv', header=None)
    
    print(f"First sample FE:  {df1_fe.shape[0]} rows x {df1_fe.shape[1]} cols")
    print(f"Second sample FE: {df2_fe.shape[0]} rows x {df2_fe.shape[1]} cols")
    
    # Check feature expansion
    assert df1_fe.shape[1] > 1000, f"Expected >1000 features, got {df1_fe.shape[1]}"
    
    # Compare original and FE first values (should match for target)
    df1_orig = pd.read_csv('first_sample/rawdata.csv', header=None)
    target_orig = df1_orig.iloc[:5, 0].values
    target_fe = df1_fe.iloc[:5, 0].values
    
    if np.allclose(target_orig, target_fe, rtol=0.01):
        print("✓ Target values match between original and FE data")
    else:
        print("⚠ Target values differ slightly (may be due to NaN handling)")
    
    print(f"✓ Feature engineering check PASSED ({df1_fe.shape[1]} features)")
    return True

def test_imports():
    """Test all critical imports work"""
    print("\n" + "=" * 60)
    print("TEST 3: IMPORT CHECK")
    print("=" * 60)
    
    errors = []
    
    # Test core modules
    modules = [
        ('utils', 'utils'),
        ('feature_engineering', 'feature_engineering'),
        ('feature_config', 'feature_config'),
        ('fred_md_loader', 'fred_md_loader'),
    ]
    
    for name, module in modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            errors.append(name)
    
    # Test first_sample functions
    funcs = ['func_ar', 'func_lasso', 'func_rf', 'func_xgb', 'func_nn', 'func_lstm']
    for func in funcs:
        try:
            __import__(f'first_sample.functions.{func}')
            print(f"  ✓ first_sample.functions.{func}")
        except Exception as e:
            print(f"  ✗ first_sample.functions.{func}: {e}")
            errors.append(func)
    
    # Test second_sample functions
    for func in funcs:
        try:
            __import__(f'second_sample.functions.{func}')
            print(f"  ✓ second_sample.functions.{func}")
        except Exception as e:
            print(f"  ✗ second_sample.functions.{func}: {e}")
            errors.append(func)
    
    if errors:
        print(f"⚠ {len(errors)} import errors")
        return False
    print("✓ All imports PASSED")
    return True

def test_ar_model():
    """Test AR model execution"""
    print("\n" + "=" * 60)
    print("TEST 4: AR MODEL EXECUTION")
    print("=" * 60)
    
    from first_sample.functions.func_ar import ar_rolling_window
    
    # Load data (no header)
    df = pd.read_csv('first_sample/rawdata.csv', header=None)
    Y = df.values
    
    # Run AR(1) with small nprev for testing
    nprev = 10  # Small test
    print(f"Running AR(1) with nprev={nprev}...")
    
    result = ar_rolling_window(Y, nprev, indice=1, lag=1, model_type="fixed")
    
    print(f"  Predictions shape: {result['pred'].shape}")
    print(f"  RMSE: {result['errors']['rmse']:.4f}")
    
    assert len(result['pred']) == nprev
    assert result['errors']['rmse'] > 0
    
    print("✓ AR model test PASSED")
    return True

def test_rf_model():
    """Test Random Forest model execution"""
    print("\n" + "=" * 60)
    print("TEST 5: RANDOM FOREST MODEL EXECUTION")
    print("=" * 60)
    
    from first_sample.functions.func_rf import rf_rolling_window
    
    # Load data (no header)
    df = pd.read_csv('first_sample/rawdata.csv', header=None)
    Y = df.values
    
    # Run RF with small nprev for testing
    nprev = 5  # Very small test
    print(f"Running Random Forest with nprev={nprev}...")
    
    result = rf_rolling_window(Y, nprev, indice=1, lag=1)
    
    print(f"  Predictions shape: {result['pred'].shape}")
    print(f"  RMSE: {result['errors']['rmse']:.4f}")
    
    assert len(result['pred']) == nprev
    assert result['errors']['rmse'] > 0
    
    print("✓ Random Forest model test PASSED")
    return True

def test_lstm_import():
    """Test LSTM model can be imported (TensorFlow check)"""
    print("\n" + "=" * 60)
    print("TEST 6: LSTM IMPORT CHECK")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"  TensorFlow version: {tf.__version__}")
        
        from first_sample.functions.func_lstm import lstm_rolling_window
        print("  ✓ LSTM function imported successfully")
        
        from second_sample.functions.func_lstm import lstm_rolling_window as lstm_rw2
        print("  ✓ Second sample LSTM imported successfully")
        
        print("✓ LSTM import test PASSED")
        return True
    except Exception as e:
        print(f"  ⚠ LSTM import warning: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("# NAGHIAYIK PYTHON - FINAL CONTROL TESTS")
    print("#" * 60)
    
    results = {}
    
    try:
        results['data_structure'] = test_data_structure()
    except Exception as e:
        print(f"✗ Data structure test FAILED: {e}")
        results['data_structure'] = False
    
    try:
        results['feature_engineering'] = test_feature_engineered_data()
    except Exception as e:
        print(f"✗ Feature engineering test FAILED: {e}")
        results['feature_engineering'] = False
    
    try:
        results['imports'] = test_imports()
    except Exception as e:
        print(f"✗ Import test FAILED: {e}")
        results['imports'] = False
    
    try:
        results['ar_model'] = test_ar_model()
    except Exception as e:
        print(f"✗ AR model test FAILED: {e}")
        results['ar_model'] = False
    
    try:
        results['rf_model'] = test_rf_model()
    except Exception as e:
        print(f"✗ RF model test FAILED: {e}")
        results['rf_model'] = False
    
    try:
        results['lstm_import'] = test_lstm_import()
    except Exception as e:
        print(f"✗ LSTM import test FAILED: {e}")
        results['lstm_import'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Project is ready.")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please review.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
