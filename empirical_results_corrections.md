# Key Corrections Needed for Empirical Results Section

## Verified Technical Details from Our Implementation:

### 1. Feature Engineering Specifics
- **Raw FRED-MD variables**: 126 (not 127)
- **Rolling windows**: [3, 6, 12] months (NOT [3, 6, 12, 24])
- **Momentum horizons**: [3, 6, 12] months
- **Volatility windows**: [3, 6, 12] months
- **Feature-engineered dataset**: 5,061 total features (126 raw + 4,935 engineered)
- **After smart embedding**: ~5,000 features (504 lagged raw + ~4,500 current engineered)
- **After 3-stage selection**: ~450-550 features typically used

### 2. Sample Periods (from README)
- **First sample**: 502 rows, 132 out-of-sample forecasts (~2000-2025)
- **Second sample**: 800 rows, 298 out-of-sample forecasts (~1959-2025)

### 3. Configuration Parameters
- **N_JOBS**: 4 (for RF/XGB), 2 (for LSTM) - NOT -1
- **RF Parameters**: n_estimators=200, max_depth=12 (Hybrid), or 300/20 (standard)
- **SelectKBest**: MAX_FEATURES = 500 (pre-screening limit)

### 4. Feature Selection Thresholds
- **Constant variance**: 1e-8
- **Correlation**: 0.95
- **Low variance**: 0.001

### 5. Critical Fixes Implemented
- **Data leakage prevention**: All rolling stats use .shift(1) and min_periods=window
- **Supervised learning alignment**: X_t paired with y_{t+h} (not y_t)
- **Smart embedding**: Only lag raw features, keep engineered features at lag 0
- **MAD implementation**: Added Median Absolute Deviation for robust dispersion

### 6. Computational Performance
- **Before optimization**: ~20,000 features, 80 hours runtime
- **After optimization**: ~5,000 features, 4-8 hours runtime
- **Speedup**: 10-20× for RF/XGB, 10-15× for LSTM
- **Memory reduction**: ~16× (from 16GB to 1GB)

## Corrections to Make in empirical_results.tex:

1. Change "3, 6, 12, and 24-month windows" to "3, 6, and 12-month windows"
2. Verify feature counts: 126 raw → 5,061 total → ~5,000 after embedding → ~500 after selection
3. Update sample descriptions to match actual periods
4. Ensure MAD is mentioned as a robust dispersion measure we implemented
5. Verify all table structures match the 5 sub-periods the user will have data for
