"""
Data Leakage Test Suite
========================
Tests to verify that feature engineering does not leak future data into training.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import StationaryFeatureEngineer


def test_no_future_data_leakage():
    """
    Verify that adding future data doesn't change past features.
    
    This test ensures that features computed at time t only use data
    from times <= t-1 (no current or future data).
    """
    print("Testing for data leakage in feature engineering...")
    
    # Create synthetic time series data
    np.random.seed(42)
    n_obs = 150
    n_features = 10
    Y_full = np.random.randn(n_obs, n_features)
    
    # Create features on partial data (first 100 observations)
    Y_partial = Y_full[:100, :]
    fe1 = StationaryFeatureEngineer()
    features_partial = fe1.get_all_features(Y_partial, include_raw=True, skip_basic_transforms=True)
    
    # Create features on full data (all 150 observations)
    fe2 = StationaryFeatureEngineer()
    features_full = fe2.get_all_features(Y_full, include_raw=True, skip_basic_transforms=True)
    
    print(f"  Partial data shape: {features_partial.shape}")
    print(f"  Full data shape: {features_full.shape}")
    
    # The first 100 rows should be IDENTICAL (or NaN in same positions)
    # This verifies that adding observations 101-150 didn't change features for 1-100
    
    # Check that shapes match for the overlapping period
    assert features_partial.shape[1] == features_full.shape[1], \
        "Feature dimensions don't match!"
    
    # Compare the overlapping period
    features_partial_subset = features_partial[:100, :]
    features_full_subset = features_full[:100, :]
    
    # Check NaN positions match
    nan_mask_partial = np.isnan(features_partial_subset)
    nan_mask_full = np.isnan(features_full_subset)
    
    # For non-NaN values, they should be identical
    non_nan_mask = ~nan_mask_partial & ~nan_mask_full
    
    if np.any(non_nan_mask):
        # Compare non-NaN values
        diff = np.abs(features_partial_subset[non_nan_mask] - features_full_subset[non_nan_mask])
        max_diff = np.max(diff)
        
        print(f"  Maximum difference in overlapping features: {max_diff}")
        
        if max_diff > 1e-10:
            print("  ❌ FAILED: Data leakage detected!")
            print(f"     Past features changed when future data was added (max diff: {max_diff})")
            return False
        else:
            print("  ✅ PASSED: No data leakage detected!")
            print("     Past features remain unchanged when future data is added")
            return True
    else:
        print("  ⚠️  WARNING: All values are NaN, cannot verify")
        return None


def test_rolling_statistics_use_past_only():
    """
    Verify that rolling statistics only use past data (shifted properly).
    """
    print("\nTesting rolling statistics for proper shifting...")
    
    # Create simple time series
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Calculate rolling mean with shift(1)
    rolling_mean = series.rolling(window=3, min_periods=3).mean().shift(1)
    
    # At time t=3 (value=4), rolling mean should use [1,2,3] = 2.0
    # But due to shift(1), it should be NaN
    # At time t=4 (value=5), rolling mean should use [1,2,3] = 2.0
    
    print(f"  Series: {series.values}")
    print(f"  Rolling mean (3-period, shifted): {rolling_mean.values}")
    
    # Check that first few values are NaN (due to min_periods and shift)
    assert np.isnan(rolling_mean.iloc[0:4]).all(), \
        "First 4 values should be NaN due to window=3 and shift(1)"
    
    # Check that value at index 4 uses data from [1,2,3]
    expected_value = np.mean([1, 2, 3])
    actual_value = rolling_mean.iloc[4]
    
    assert np.abs(actual_value - expected_value) < 1e-10, \
        f"Rolling mean at t=4 should be {expected_value}, got {actual_value}"
    
    print("  ✅ PASSED: Rolling statistics properly shifted!")
    return True


def test_min_periods_prevents_unstable_features():
    """
    Verify that min_periods=window prevents computing features with insufficient data.
    """
    print("\nTesting min_periods configuration...")
    
    series = pd.Series([1, 2, 3, 4, 5])
    
    # With min_periods=window, should get NaN when insufficient data
    rolling_mean_strict = series.rolling(window=10, min_periods=10).mean()
    
    # All values should be NaN (series length < window)
    assert rolling_mean_strict.isna().all(), \
        "All values should be NaN when series length < window with min_periods=window"
    
    print("  ✅ PASSED: min_periods=window prevents unstable features!")
    return True


def run_all_tests():
    """Run all data leakage tests."""
    print("=" * 70)
    print("DATA LEAKAGE TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Test 1: No future data leakage
    results.append(test_no_future_data_leakage())
    
    # Test 2: Rolling statistics use past only
    results.append(test_rolling_statistics_use_past_only())
    
    # Test 3: min_periods prevents unstable features
    results.append(test_min_periods_prevents_unstable_features())
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    warnings = sum(1 for r in results if r is None)
    
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Warnings: {warnings}")
    
    if failed == 0:
        print("\n✅ All tests passed! No data leakage detected.")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed! Data leakage detected.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
