"""
Test script to verify 3-stage feature selection works correctly.
"""
import numpy as np
import pandas as pd
from feature_engineering import StationaryFeatureEngineer
from feature_utils import handle_missing_values, apply_3stage_feature_selection
from feature_config import CONSTANT_VARIANCE_THRESHOLD, CORRELATION_THRESHOLD, LOW_VARIANCE_THRESHOLD

print("=" * 70)
print("TESTING 3-STAGE FEATURE SELECTION")
print("=" * 70)

# Load rawdata
print("\n[Step 1] Loading rawdata.csv...")
rawdata_path = "first_sample/rawdata.csv"
Y = pd.read_csv(rawdata_path, header=None).values
print(f"  Original data shape: {Y.shape}")

# Apply feature engineering
print("\n[Step 2] Applying feature engineering...")
fe = StationaryFeatureEngineer()
Y_engineered = fe.get_all_features(Y, include_raw=True, skip_basic_transforms=True)
Y_engineered, _ = handle_missing_values(Y_engineered, strategy='mean')
print(f"  Engineered features shape: {Y_engineered.shape}")
print(f"  Feature expansion: {Y.shape[1]} → {Y_engineered.shape[1]} ({Y_engineered.shape[1]/Y.shape[1]:.1f}x)")

# Apply 3-stage selection
print("\n[Step 3] Applying 3-stage feature selection...")
Y_selected, selection_info = apply_3stage_feature_selection(
    Y_engineered,
    constant_threshold=CONSTANT_VARIANCE_THRESHOLD,
    correlation_threshold=CORRELATION_THRESHOLD,
    variance_threshold=LOW_VARIANCE_THRESHOLD,
    verbose=True
)

print("\n" + "=" * 70)
print("FEATURE SELECTION SUMMARY")
print("=" * 70)
print(f"Initial features:         {selection_info['initial_features']}")
print(f"After Stage 1a (const):   {selection_info['initial_features'] - selection_info['stage_1a_removed']}")
print(f"After Stage 1b (corr):    {selection_info['initial_features'] - selection_info['stage_1a_removed'] - selection_info['stage_1b_removed']}")
print(f"Final features:           {selection_info['final_features']}")
print(f"\nTotal reduction:          {100 * (1 - selection_info['final_features']/selection_info['initial_features']):.1f}%")
print(f"Compression ratio:        {selection_info['initial_features']/selection_info['final_features']:.2f}x")

print("\n" + "=" * 70)
print("CONFIGURATION USED")
print("=" * 70)
print(f"Constant variance threshold:     {CONSTANT_VARIANCE_THRESHOLD}")
print(f"Correlation threshold:           {CORRELATION_THRESHOLD}")
print(f"Low variance threshold:          {LOW_VARIANCE_THRESHOLD}")

print("\n✅ Test completed successfully!")
print(f"   Final feature count: {Y_selected.shape[1]} (target: 3000-4000)")
