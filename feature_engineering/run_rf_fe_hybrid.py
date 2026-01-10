"""
Hybrid RF-FE: Best of Both Worlds
==================================
Combines:
- YOUR optimization: Smart embedding (only lag raw features)
- FRIEND's features: SelectKBest, Random Walk benchmark, float32, better error handling

Performance: 4-8 hours (vs 80 hours original)
Features: ~5,000 (vs 20,000 original)
Benchmarking: Relative RMSE vs Random Walk
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# --- Configuration ---
warnings.filterwarnings('ignore')

# Handle both script and notebook environments
try:
    SCRIPT_DIR = Path(__file__).parent  # feature_engineering folder
    PROJECT_ROOT = SCRIPT_DIR.parent    # project root
except NameError:
    # Running in Kaggle/Jupyter notebook
    SCRIPT_DIR = Path.cwd()
    PROJECT_ROOT = SCRIPT_DIR

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

CPI_COLUMN = 'CPIAUCSL'
PCE_COLUMN = 'PCEPI'
N_JOBS = 4                # Optimized for memory
MAX_FEATURES = 500        # Pre-screening limit
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 12,
    'random_state': 42,
    'n_jobs': 1
}

# Import project modules
try:
    from utils import embed, calculate_errors
    from .feature_engineering import StationaryFeatureEngineer
    from .feature_utils import handle_missing_values, apply_3stage_feature_selection
    from .feature_config import (
        CONSTANT_VARIANCE_THRESHOLD,
        CORRELATION_THRESHOLD,
        LOW_VARIANCE_THRESHOLD
    )
    from fred_md_loader import FREDMDLoader
except ImportError as e:
    print(f"❌ ERROR: Missing required modules - {e}")
    sys.exit(1)


# ==============================================================================
# BENCHMARK UTILITIES (from friend's code)
# ==============================================================================

def run_rw_benchmark(data, raw_cols, test_mask, h, t_name):
    """
    Random Walk benchmark: y_{t+h} = y_t
    
    This is the naive forecast baseline. A good model should beat this.
    
    Parameters:
    -----------
    data : np.ndarray
        Full data matrix
    raw_cols : list
        Column names
    test_mask : np.ndarray
        Boolean mask for test period
    h : int
        Forecast horizon
    t_name : str
        Target variable name
        
    Returns:
    --------
    np.ndarray : Squared errors from random walk forecast
    """
    t_idx = raw_cols.index(t_name)
    data_test = data[test_mask]
    n_test = len(data_test)
    n_forecasts = n_test - h
    
    if n_forecasts <= 0:
        return np.array([])
    
    sq_errors = []
    for i in range(n_forecasts):
        actual = data_test[i + h, t_idx]
        pred = data_test[i, t_idx]  # Naive: use current value
        sq_errors.append((actual - pred) ** 2)
    
    return np.array(sq_errors)


# ==============================================================================
# OPTIMIZED RF-FE ENGINE (hybrid approach)
# ==============================================================================

def run_rf_fe_hybrid(Y_raw_matrix, target_name, lag, raw_column_names):
    """
    Hybrid RF-FE combining:
    - Smart embedding (only lag raw features) - YOUR optimization
    - SelectKBest pre-screening - FRIEND's feature
    - float32 for memory efficiency - FRIEND's feature
    - Defensive error handling - FRIEND's feature
    
    Parameters:
    -----------
    Y_raw_matrix : np.ndarray
        Raw FRED-MD data matrix
    target_name : str
        Name of target variable
    lag : int
        Forecast horizon
    raw_column_names : list
        Column names
        
    Returns:
    --------
    dict : {'pred': float, 'n_features': int}
    """
    n_raw_features = Y_raw_matrix.shape[1]
    
    # Defensive check
    if len(Y_raw_matrix) < 20:
        return {'pred': np.nan, 'n_features': 0}
    
    # OPTIMIZATION 1: Use most recent raw features (lags 0, 1, 2, 3)
    # aux_raw[t] contains [Y_t, Y_{t-1}, Y_{t-2}, Y_{t-3}] for all 126 variables
    aux_raw = embed(Y_raw_matrix, 4)
    if len(aux_raw) < 2:
        return {'pred': np.nan, 'n_features': 0}
    
    # OPTIMIZATION 2: Engineer features on CURRENT data
    fe = StationaryFeatureEngineer()
    df_raw = pd.DataFrame(Y_raw_matrix, columns=raw_column_names)
    eng_out = fe.get_all_features(df_raw, include_raw=False, skip_basic_transforms=True)
    Y_eng = eng_out.values if isinstance(eng_out, pd.DataFrame) else eng_out
    
    # Trim engineered features to match aux_raw length (starts at index 3 due to embed 4)
    Y_eng_aligned = Y_eng[3:]
    
    # FRIEND's OPTIMIZATION: Use float32 for memory efficiency
    Y_eng_aligned, _ = handle_missing_values(Y_eng_aligned, strategy='mean')
    Y_eng_aligned = Y_eng_aligned.astype(np.float32)
    
    # Combine into feature matrix where each row 't' is info available at time 't'
    min_len = min(len(aux_raw), len(Y_eng_aligned))
    X_combined = np.hstack([aux_raw[:min_len], Y_eng_aligned[:min_len]]).astype(np.float32)
    
    # Target values (Month 3 to T)
    if target_name in raw_column_names:
        t_idx = raw_column_names.index(target_name)
        y_target = Y_raw_matrix[3:3+min_len, t_idx]
    else:
        return {'pred': np.nan, 'n_features': 0}
    
    # --- SUPERVISED LEARNING ALIGNMENT (CRITICAL) ---
    # We want to match: X_t (info at time t) -> y_{t+h} (target at time t+h)
    # Features X_t: [Y_t, Y_{t-1}, ..., engineered_t]
    # Prediction target is h months ahead
    
    if len(X_combined) <= lag:
        return {'pred': np.nan, 'n_features': 0}
    
    # X used for training (all but last 'lag' months)
    X_train_full = X_combined[:-lag, :]
    # y used for training (shifted forward by 'lag')
    y_train_full = y_target[lag:]
    
    # X used for prediction (the VERY LAST information set available)
    X_forecast = X_combined[-1:, :]
    
    # SelectKBest pre-screening on TRAINING DATA ONLY (Leakage Prevention)
    num_k = min(MAX_FEATURES, X_train_full.shape[1])
    if X_train_full.shape[1] > num_k:
        selector = SelectKBest(score_func=f_regression, k=num_k)
        selector.fit(X_train_full, y_train_full)
        X_train_screened = X_train_full[:, selector.get_support()]
        X_forecast_screened = X_forecast[:, selector.get_support()]
    else:
        X_train_screened = X_train_full
        X_forecast_screened = X_forecast
    
    # Apply 3-stage feature selection to training data
    try:
        X_final_train, selection_info = apply_3stage_feature_selection(
            X_train_screened,
            constant_threshold=CONSTANT_VARIANCE_THRESHOLD,
            correlation_threshold=CORRELATION_THRESHOLD,
            variance_threshold=LOW_VARIANCE_THRESHOLD,
            verbose=False
        )
        # Apply same selection to forecast features
        X_final_forecast = X_forecast_screened[:, selection_info['selected_indices']]
    except Exception as e:
        print(f"  ⚠ Feature selection failed: {e}")
        X_final_train = X_train_screened
        X_final_forecast = X_forecast_screened
    
    if X_final_train.shape[1] == 0:
        return {'pred': np.nan, 'n_features': 0}
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_final_train)
    y_train = y_train_full
    
    # Train Random Forest
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    X_today_scaled = scaler.transform(X_final_forecast)
    pred = model.predict(X_today_scaled)[0]
    
    return {
        'pred': float(pred),
        'n_features': X_final_train.shape[1]
    }


# ==============================================================================
# ROLLING FORECAST WRAPPER
# ==============================================================================

def rolling_forecast_hybrid(data, raw_cols, train_mask, test_mask, h, t_name, p_label):
    """
    Rolling window forecast with hybrid RF-FE.
    
    Parameters:
    -----------
    data : np.ndarray
        Full data matrix
    raw_cols : list
        Column names
    train_mask : np.ndarray
        Boolean mask for training period
    test_mask : np.ndarray
        Boolean mask for test period
    h : int
        Forecast horizon
    t_name : str
        Target variable name
    p_label : str
        Period label for logging
        
    Returns:
    --------
    dict : Results with predictions, actuals, squared errors
    """
    data_train = data[train_mask]
    data_test = data[test_mask]
    n_test = len(data_test)
    n_forecasts = n_test - h
    
    if n_forecasts <= 0:
        print(f"  ⚠ WARNING: Test period too short for h={h}")
        return {
            'preds': np.array([]),
            'actuals': np.array([]),
            'sq_errors': np.array([]),
            'n_feat': 0
        }
    
    t_raw_idx = raw_cols.index(t_name)
    
    def worker(i):
        """Process single forecast iteration.
        
        DATA LEAKAGE FIX:
        - At time i, we forecast i + h
        - Training data must EXCLUDE observation i (the forecast origin)
        - Only use data BEFORE time i
        """
        # Expanding window: use all data BEFORE current point (exclude i)
        if i == 0:
            # First forecast: use only training data
            Y_window = data_train
        else:
            # Subsequent forecasts: training + test data up to (but NOT including) i
            Y_window = np.vstack([data_train, data_test[:i]])
        
        # Run hybrid RF-FE
        res = run_rf_fe_hybrid(Y_window, t_name, h, raw_cols)
        
        # Actual value is h periods ahead
        actual = data_test[i + h, t_raw_idx]
        
        return res['pred'], actual, res['n_features']
    
    print(f"\n▶ FORECASTING: {p_label} | {t_name} | h={h}")
    print(f"  Valid forecast horizon: {n_forecasts} months")
    
    results = []
    batch_size = 20
    start_time = time.time()
    
    # Process in batches to manage memory
    for b in range(0, n_forecasts, batch_size):
        end = min(b + batch_size, n_forecasts)
        batch_start_t = time.time()
        
        batch_res = Parallel(n_jobs=N_JOBS)(
            delayed(worker)(i) for i in range(b, end)
        )
        results.extend(batch_res)
        
        batch_elapsed = time.time() - batch_start_t
        total_elapsed = time.time() - start_time
        print(f"  ✅ Batch [{b:3d}-{end-1:3d}] | "
              f"Time: {batch_elapsed:5.1f}s | "
              f"Elapsed: {total_elapsed/60:5.1f}m")
    
    # Extract results
    preds = np.array([r[0] for r in results])
    acts = np.array([r[1] for r in results])
    
    # Filter out NaN predictions
    valid = ~np.isnan(preds) & ~np.isnan(acts)
    
    if not np.any(valid):
        print(f"  ⚠ WARNING: All predictions are NaN!")
        return {
            'preds': np.array([]),
            'actuals': np.array([]),
            'sq_errors': np.array([]),
            'n_feat': 0
        }
    
    return {
        'preds': preds[valid],
        'actuals': acts[valid],
        'sq_errors': (acts[valid] - preds[valid]) ** 2,
        'n_feat': np.mean([r[2] for r in results if r[2] > 0])
    }


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "=" * 70)
    print("HYBRID RF-FE: OPTIMIZED MULTIVARIATE MACRO-FORECASTING")
    print("=" * 70)
    print("\nCombining:")
    print("  ✓ Smart embedding (only lag raw features)")
    print("  ✓ SelectKBest pre-screening")
    print("  ✓ Random Walk benchmarking")
    print("  ✓ Memory-efficient float32")
    print("  ✓ Defensive error handling")
    print("\nExpected: 4-8 hours runtime (vs 80 hours original)")
    print("=" * 70)
    
    # Load data
    data_path = PROJECT_ROOT / 'data' / '2025-11-MD.csv'
    
    if not data_path.exists():
        print(f"\n❌ ERROR: Data file not found: {data_path}")
        return
    
    print(f"\nLoading data from: {data_path}")
    loader = FREDMDLoader(data_path=str(data_path))
    loader.load_raw_data()
    df = loader.transform_data()
    
    raw_cols = df.columns.tolist()
    data = df.values
    
    print(f"  Data shape: {data.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Define test periods
    periods = [
        ('Test1 (2000-2016)', '2000-01-01', '2016-12-31'),
        ('Test2 (2017-2022)', '2017-01-01', '2022-12-31'),
        ('Test3 (2023-2025)', '2023-01-01', '2025-10-31')
    ]
    
    horizons = [1, 6, 12]
    final_output = []
    
    # Run forecasts for each horizon
    for h in horizons:
        print(f"\n{'='*70}")
        print(f"HORIZON: {h} months")
        print(f"{'='*70}")
        
        h_pool_rf = []
        h_pool_rw = []
        
        for p_name, start, end in periods:
            train_m = df.index < start
            test_m = (df.index >= start) & (df.index <= end)
            
            # Run hybrid RF-FE
            rf_results = rolling_forecast_hybrid(
                data, raw_cols, train_m, test_m, h, CPI_COLUMN, p_name
            )
            
            # Run Random Walk benchmark
            rw_sq = run_rw_benchmark(data, raw_cols, test_m, h, CPI_COLUMN)
            
            if len(rf_results['sq_errors']) > 0 and len(rw_sq) > 0:
                # Calculate RMSE
                rf_rmse = np.sqrt(np.mean(rf_results['sq_errors']))
                rw_rmse = np.sqrt(np.mean(rw_sq))
                rel_rmse = rf_rmse / rw_rmse
                
                print(f"\n  📊 {p_name}:")
                print(f"     RF-FE RMSE: {rf_rmse:.6f}")
                print(f"     RW RMSE:    {rw_rmse:.6f}")
                print(f"     Relative:   {rel_rmse:.4f} ({'✓ Better' if rel_rmse < 1 else '✗ Worse'} than RW)")
                print(f"     Features:   {rf_results['n_feat']:.0f}")
                
                final_output.append({
                    'Horizon': h,
                    'Period': p_name,
                    'RF_RMSE': rf_rmse,
                    'RW_RMSE': rw_rmse,
                    'Rel_RMSE': rel_rmse,
                    'N_Features': rf_results['n_feat']
                })
                
                # Pool for total horizon stats
                h_pool_rf.extend(rf_results['sq_errors'])
                h_pool_rw.extend(rw_sq)
        
        # Calculate pooled statistics across all periods
        if len(h_pool_rf) > 0 and len(h_pool_rw) > 0:
            total_rf_rmse = np.sqrt(np.mean(h_pool_rf))
            total_rw_rmse = np.sqrt(np.mean(h_pool_rw))
            total_rel = total_rf_rmse / total_rw_rmse
            
            print(f"\n  📊 POOLED (All Periods):")
            print(f"     RF-FE RMSE: {total_rf_rmse:.6f}")
            print(f"     RW RMSE:    {total_rw_rmse:.6f}")
            print(f"     Relative:   {total_rel:.4f} ({'✓ Better' if total_rel < 1 else '✗ Worse'} than RW)")
            
            final_output.append({
                'Horizon': h,
                'Period': 'POOLED',
                'RF_RMSE': total_rf_rmse,
                'RW_RMSE': total_rw_rmse,
                'Rel_RMSE': total_rel,
                'N_Features': np.mean([r['N_Features'] for r in final_output if r['Horizon'] == h and r['Period'] != 'POOLED'])
            })
    
    # Final summary
    results_df = pd.DataFrame(final_output)
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: RELATIVE RMSE (RF-FE / Random Walk)")
    print("=" * 70)
    print("\nValues < 1.0 indicate RF-FE beats the naive Random Walk forecast")
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    output_dir = PROJECT_ROOT / 'results' / 'hybrid_rf_fe'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'hybrid_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
