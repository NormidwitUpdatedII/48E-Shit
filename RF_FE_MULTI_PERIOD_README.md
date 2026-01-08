# RF-FE Multi-Period Out-of-Sample Forecasting

## Overview

This script performs **Random Forest with Feature Engineering (RF-FE)** forecasting across four distinct time periods:

1. **Training Period**: up to 1990
2. **Test Period 1**: 2000-2016
3. **Test Period 2**: 2016-2022
4. **Test Period 3**: 2022-2023

The model uses rolling window forecasting with a **12-month forecast horizon** for both CPI and PCE inflation.

## Files Created

### Main Script
- **`run_rf_fe_multi_period.py`**: Main script for multi-period RF-FE forecasting

### Support Files
- **`validate_rf_fe_setup.py`**: Setup validation script (run this FIRST)
- **`requirements_rf_fe.txt`**: Minimal requirements file for this script

## Quick Start

### Step 1: Validate Setup

Run the validation script to check dependencies and data files:

```bash
python validate_rf_fe_setup.py
```

This will check:
- ✓ All required Python packages are installed
- ✓ Project files (utils.py, feature_engineering.py, etc.) exist
- ✓ Data files (rawdata.csv) are available
- ✓ Modules can be imported correctly

### Step 2: Install Dependencies (if needed)

If validation fails due to missing packages:

```bash
pip install -r requirements_rf_fe.txt
```

Or install specific packages:

```bash
pip install numpy pandas scikit-learn joblib
```

### Step 3: Run the Main Script

**⚠️ WARNING: This will take 20-60 minutes to complete!**

```bash
python run_rf_fe_multi_period.py
```

The script will:
1. Load data from `first_sample/rawdata.csv` (or `second_sample/rawdata.csv`)
2. Split data into three periods
3. Run rolling window forecasts for CPI (both test periods)
4. Run rolling window forecasts for PCE (both test periods)
5. Calculate error metrics (RMSE, MAE, MAPE)
6. Save all results to `results/multi_period_rf_fe/`

## Output Files

All results are saved to: `results/multi_period_rf_fe/`

### Summary Files
- **`summary_table.csv`**: Overall results for all targets and periods
- **`metadata.txt`**: Run configuration and parameters

### Detailed Forecasts
- **`cpi_test1_2000_2016_forecasts.csv`**: CPI forecasts for 2000-2016
- **`cpi_test2_2016_2022_forecasts.csv`**: CPI forecasts for 2016-2022
- **`cpi_test3_2022_2023_forecasts.csv`**: CPI forecasts for 2022-2023
- **`pce_test1_2000_2016_forecasts.csv`**: PCE forecasts for 2000-2016
- **`pce_test2_2016_2022_forecasts.csv`**: PCE forecasts for 2016-2022
- **`pce_test3_2022_2023_forecasts.csv`**: PCE forecasts for 2022-2023

Each forecast file contains:
- `Date`: Forecast date
- `Actual`: Actual inflation value
- `Predicted`: Model prediction
- `Error`: Prediction error (Actual - Predicted)
- `Abs_Error`: Absolute prediction error

## Technical Details

### Data Pipeline

```
Raw FRED-MD Data
    ↓
FRED-MD Stationarity Transformations (already applied in rawdata.csv)
    ↓
Feature Engineering (momentum, volatility, z-scores, rolling stats)
    ↓
3-Stage Feature Selection:
    - Stage 1: Remove constants (variance < 1e-8)
    - Stage 2: Remove correlated features (correlation > 0.95)
    - Stage 3: Remove low variance features (variance < 0.001)
    ↓
Standardization (zero mean, unit variance)
    ↓
Random Forest Model (300 trees, max_depth=20)
    ↓
12-Month Ahead Forecast
```

### Model Configuration

**Random Forest Parameters:**
- `n_estimators`: 300
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `random_state`: 42
- `n_jobs`: -1 (use all CPU cores)

**Forecast Horizon:** 12 months

**Target Variables:**
- CPI inflation (column 1)
- PCE inflation (column 2)

### Parallel Processing

The script uses `joblib` for parallel processing:
- Each rolling window iteration runs in parallel
- Utilizes all available CPU cores (`n_jobs=-1`)
- Significantly reduces runtime compared to sequential processing

### Feature Engineering

**Advanced features created:**
1. **Rolling Statistics** (windows: 3, 6, 12 months)
   - Mean, Std, Max, Min, Range, Skewness

2. **Momentum Features** (horizons: 3, 6, 12 months)
   - Rate of change, Acceleration, Momentum

3. **Volatility Features** (windows: 3, 6, 12 months)
   - Realized volatility, Volatility of volatility

4. **Z-scores** (window: 12 months)
   - Rolling standardization for outlier detection

5. **Cross-sectional Features**
   - Spreads, ratios between related variables

## Expected Runtime

**Total runtime: 30-90 minutes** (depends on CPU cores and data size)

Breakdown:
- CPI Test Period 1 (2000-2016): ~8-20 minutes
- CPI Test Period 2 (2016-2022): ~5-15 minutes
- CPI Test Period 3 (2022-2023): ~2-5 minutes
- PCE Test Period 1 (2000-2016): ~8-20 minutes
- PCE Test Period 2 (2016-2022): ~5-15 minutes
- PCE Test Period 3 (2022-2023): ~2-5 minutes

## Troubleshooting

### Issue: Missing dependencies
**Solution:** Run `pip install -r requirements_rf_fe.txt`

### Issue: No data file found
**Solution:** Ensure `rawdata.csv` exists in either:
- `first_sample/rawdata.csv`
- `second_sample/rawdata.csv`

### Issue: Module import errors
**Solution:** Run `validate_rf_fe_setup.py` to diagnose the issue

### Issue: Out of memory
**Solution:** 
- Close other applications
- Reduce `n_estimators` in RF_PARAMS (e.g., from 300 to 100)
- Use a machine with more RAM

### Issue: Too slow
**Solution:**
- Ensure `n_jobs=-1` (uses all cores)
- Reduce `n_estimators` (e.g., 100 instead of 300)
- Run on a machine with more CPU cores

## Customization

### Change Time Periods

Edit in `run_multi_period_analysis()`:

```python
results = run_multi_period_analysis(
    data_path=data_path,
    train_end='1990-12-31',      # ← End of training
    test1_start='2000-01-01',    # ← Start of test 1
    test1_end='2016-12-31',      # ← End of test 1
    test2_end='2022-12-31',      # ← End of test 2
    test3_end='2023-12-31'       # ← End of test 3
)
```

### Change Forecast Horizon

Edit at top of script:

```python
FORECAST_HORIZON = 12  # ← Change to 1, 3, 6, or 24
```

### Change RF Parameters

Edit `RF_PARAMS` dictionary:

```python
RF_PARAMS = {
    'n_estimators': 300,        # ← Number of trees
    'max_depth': 20,            # ← Maximum tree depth
    'min_samples_split': 5,     # ← Min samples to split
    'min_samples_leaf': 2,      # ← Min samples per leaf
    'random_state': 42,
    'n_jobs': -1
}
```

## Dependencies

### Required
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0

### Optional
- tqdm >= 4.65.0 (progress bars)
- statsmodels >= 0.14.0 (stationarity tests)

## Citation

If using this code, please cite:

Naghiayik Project - Inflation Forecasting with Machine Learning Methods (2026)

## Contact

For issues or questions, refer to the main project README.

---

**Last Updated:** January 2026
