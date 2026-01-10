# Inflation Forecasting with Machine Learning and Feature Engineering

A comprehensive Python implementation for inflation forecasting using machine learning methods with advanced feature engineering. This project extends the methodology from **"Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods"** by Medeiros et al. (2021) with optimized feature engineering and computational efficiency improvements.

## 📖 Overview

This repository implements state-of-the-art machine learning methods for inflation forecasting, featuring:

- **Advanced Feature Engineering**: Transforms 126 FRED-MD variables into 5,061 engineered features with smart embedding optimization
- **21+ Forecasting Methods**: Random Forest, XGBoost, LSTM, LASSO, Ridge, and more
- **Optimized Performance**: 10-20× runtime improvement through smart embedding (4-8 hours vs 80 hours)
- **Rigorous Data Leakage Prevention**: All features use proper temporal alignment and .shift(1) operations
- **Two Sample Periods**: First sample (2000-2025, 502 obs) and Second sample (1959-2025, 800 obs)
- **Rolling Window Evaluation**: Expanding window forecasts with comprehensive error metrics
- **Hybrid RF-FE Model**: Best-in-class performance combining SelectKBest, smart embedding, and Random Walk benchmarking

## 📁 Project Structure

```
48E-Project-Files/
├── requirements.txt          # Python dependencies
├── utils.py                  # Shared utility functions
├── fred_md_loader.py         # FRED-MD data loader with transformations
├── generate_all_samples.py   # Generate data for all 5 sample periods
├── run_all_models.py         # Master script to run all models
├── run_rf_fe_hybrid.py       # Optimized hybrid RF-FE model
├── feature_engineering/      # Feature engineering module
│   ├── __init__.py           # Package exports
│   ├── feature_engineering.py # Core FE transformations
│   ├── feature_config.py     # FE configuration parameters
│   ├── feature_utils.py      # FE utility functions
│   └── prepare_data_fe.py    # Data preparation for FE models
├── first_sample/             # First sample period analysis (2000-2025, 502 obs)
│   ├── rawdata.csv           # Original FRED-MD transformed data
│   ├── rawdata_fe.csv        # Feature-engineered data
│   ├── rawdata_1990_2000.csv # Period-specific data
│   ├── rawdata_fe_1990_2000.csv
│   ├── rawdata_2001_2015.csv
│   ├── rawdata_fe_2001_2015.csv
│   ├── rawdata_2016_2022.csv
│   ├── rawdata_fe_2016_2022.csv
│   ├── rawdata_2020_2022.csv
│   ├── rawdata_fe_2020_2022.csv
│   ├── rawdata_1990_2022.csv
│   ├── rawdata_fe_1990_2022.csv
│   ├── functions/            # Model function implementations
│   │   ├── func_rw.py        # Random Walk (benchmark)
│   │   ├── func_ar.py        # Autoregressive models
│   │   ├── func_rf.py        # Random Forest
│   │   ├── func_xgb.py       # XGBoost
│   │   ├── func_lstm.py      # LSTM Deep Learning
│   │   └── ... (21+ models)
│   └── run/                  # Execution scripts
│       ├── rf_fe.py          # Run RF with Feature Engineering
│       ├── xgb_fe.py         # Run XGBoost with Feature Engineering
│       ├── lstm_fe.py        # Run LSTM with Feature Engineering
│       └── ... (all runners)
└── second_sample/            # Second sample period (1959-2025, 800 obs)
    └── ... (same structure)
```

## 🔄 Sample Periods

The project supports 5 sample periods for analysis:

| Period | Date Range | Description | nprev |
|--------|-----------|-------------|-------|
| 1990_2000 | Jan 1990 - Dec 2000 | Low volatility (Great Moderation) | 60 |
| 2001_2015 | Jan 2001 - Dec 2015 | Financial crisis and recovery | 84 |
| 2016_2022 | Jan 2016 - Dec 2022 | COVID-19 and inflation surge | 48 |
| 2020_2022 | Jan 2020 - Dec 2022 | Pandemic subset | 24 |
| 1990_2022 | Jan 1990 - Dec 2022 | Full extended sample | 132 |

## 🔧 Installation

### Prerequisites
- Python 3.10+ (tested with Python 3.14)
- pip package manager

### Install Dependencies

```bash
cd Naghiayik-python
pip install -r requirements.txt
```

### Required Packages
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Numerical computing |
| pandas | ≥2.0.0 | Data manipulation |
| scipy | ≥1.10.0 | Scientific computing |
| scikit-learn | ≥1.3.0 | Machine learning algorithms |
| xgboost | ≥2.0.0 | Gradient boosting |
| tensorflow | ≥2.13.0 | Deep learning (LSTM, Neural Networks) |
| pyreadr | ≥0.5.0 | Load R data files (.rda, .RData) |
| matplotlib | ≥3.7.0 | Visualization (optional) |
| statsmodels | ≥0.14.0 | Statistical models (optional) |

## 🔬 Feature Engineering

The project includes an advanced feature engineering module that expands the original 126 FRED-MD variables to 5,000+ features:

### Features Generated
| Feature Type | Description |
|--------------|-------------|
| Rolling Statistics | Mean, std, min, max, skew, kurtosis (windows: 3, 6, 12) |
| Momentum | Price changes over multiple periods |
| Volatility | Rolling standard deviation |
| Z-Scores | Normalized rolling statistics |
| Cross-sectional | Mean, std, percentile rankings |

### Using Feature Engineering

```python
# Generate feature-engineered dataset
python prepare_data_fe.py

# This creates rawdata_fe.csv in each sample folder
# Features: 126 → 5,061 (rolling stats, momentum, volatility, z-scores)
```

### Running Feature-Engineered Models

```python
# Run Random Forest with feature engineering
python first_sample/run/rf_fe.py

# Run XGBoost with feature engineering  
python first_sample/run/xgb_fe.py

# Run LSTM with feature engineering
python first_sample/run/lstm_fe.py
```

## 🧠 FRED-MD Data Pipeline

The project handles FRED-MD data with official stationarity transformations:

### Transformation Codes (FRED-MD Official)
| Code | Transformation |
|------|----------------|
| 1 | No transformation |
| 2 | First difference: Δx_t |
| 3 | Second difference: Δ²x_t |
| 4 | Log: log(x_t) |
| 5 | Log first difference: Δlog(x_t) |
| 6 | Log second difference: Δ²log(x_t) |
| 7 | Percentage change: (x_t/x_{t-1} - 1) |

### Loading Raw FRED-MD Data

```python
from fred_md_loader import FREDMDLoader

# Load and transform raw FRED-MD data
loader = FREDMDLoader('2025-11-MD.csv')
transformed = loader.get_transformed_data()
```

## 🚀 Usage

### Quick Start

```python
# Load data and run AR model
import pyreadr
from first_sample.functions.func_ar import ar_rolling_window

# Load R data file
result = pyreadr.read_r('first-sample/rawdata.rda')
Y = result['dados'].values

# Run AR(1) with rolling window
nprev = 132  # Number of out-of-sample forecasts
forecasts = ar_rolling_window(Y, nprev, indice=1, lag=1, model_type="fixed")
```

### Running Individual Models

```python
# Example: Run LASSO model
from first_sample.functions.func_lasso import lasso_rolling_window

result = lasso_rolling_window(Y, nprev, indice=1, alpha=1.0)
print(f"RMSE: {result['rmse']}")
```

### Running Complete Experiments

```python
# Run the complete AR experiment (from run scripts)
from first_sample.run.ar import ar_main
results = ar_main(data_path='first-sample/rawdata.rda', nprev=132)
```

## 📊 Methods Implemented

### Benchmark Model
| Method | Description | Function |
|--------|-------------|----------|
| Random Walk | Simple benchmark (no change forecast) | `func_rw.py` |

### Linear Methods
| Method | Description | Function |
|--------|-------------|----------|
| AR | Autoregressive models (lags 1-12) | `func_ar.py` |
| Ridge | Ridge regression | `func_lasso.py` |
| LASSO | Least Absolute Shrinkage | `func_lasso.py` |
| Elastic Net | L1 + L2 regularization | `func_lasso.py` |
| Adaptive LASSO | Weighted LASSO | `func_lasso.py` |
| SCAD | Smoothly Clipped Absolute Deviation | `func_scad.py` |

### Machine Learning Methods
| Method | Description | Function |
|--------|-------------|----------|
| Random Forest | Ensemble of decision trees | `func_rf.py` |
| XGBoost | Extreme Gradient Boosting | `func_xgb.py` |
| Neural Network | Feedforward neural network | `func_nn.py` |
| LSTM | Long Short-Term Memory networks | `func_lstm.py` |
| Gradient Boosting | Sequential ensemble | `func_boosting.py` |
| Bagging | Bootstrap aggregating | `func_bag.py` |

### Factor and Dimension Reduction Methods
| Method | Description | Function |
|--------|-------------|----------|
| Factor Models | Principal component regression | `func_fact.py` |
| Targeted Factors | Targeted principal components | `func_tfact.py` |
| CSR | Complete Subset Regression | `func_csr.py` |
| RF-OLS | Random Forest variable selection + OLS | `func_rfols.py` |

### Bayesian and Other Methods
| Method | Description | Function |
|--------|-------------|----------|
| LBVAR | Large Bayesian VAR | `func_lbvar.py` |
| UC-SV | Unobserved Components with Stochastic Volatility | `func_ucsv.py` |
| Jackknife | Jackknife Model Averaging | `func_jn.py` |

## 📈 Rolling Window Evaluation

The forecasting is performed using an expanding or rolling window approach:

```
Training Period     |  Test Point
[==================]|[*]
     └─ Estimate    └─ Forecast

nprev = 132 (132 out-of-sample forecast points)
```

### Parameters
- **nprev**: Number of out-of-sample forecasts (default: 132)
- **indice**: Target variable index (1 = CPI, 2 = PCE)
- **lag**: Number of lags to include
- **model_type**: "fixed" (fixed window) or "bic" (BIC selection)

## 🧪 Testing

Run individual models:

```bash
cd Naghiayik-python

# Run AR model
python first_sample/run/ar.py

# Run LSTM model
python first_sample/run/lstm.py

# Run feature-engineered models
python first_sample/run/rf_fe.py
python first_sample/run/xgb_fe.py
python first_sample/run/lstm_fe.py
```

Expected output:
```
Running Random Forest with Feature Engineering...
Loading feature-engineered data: first_sample/rawdata_fe.csv
Feature dimensions: (502, 5061)
Rolling window evaluation: 132 forecasts
RMSE: 0.XXX, MAE: 0.XXX
```

## 📚 Data Format

### Input Data Structure
The input data (`rawdata.csv`) contains:
- A DataFrame with time series observations (FRED-MD format)
- First column: target variable (inflation measure - CPI)
- Remaining columns: predictor variables (126 macroeconomic indicators)
- Data is pre-transformed using official FRED-MD stationarity codes

### Sample Periods
| Sample | Rows | nprev | Period |
|--------|------|-------|--------|
| first_sample | 502 | 132 | ~2000-2025 |
| second_sample | 800 | 298 | ~1959-2025 |

### Feature-Engineered Data
- `rawdata_fe.csv`: Extended features (5,061 columns)
- Rolling statistics, momentum, volatility, z-scores
- Generated by `prepare_data_fe.py`

### Output Format
Each model returns a dictionary containing:
```python
{
    'forecasts': np.array,    # Out-of-sample predictions
    'actual': np.array,       # Actual values
    'rmse': float,            # Root Mean Square Error
    'mae': float,             # Mean Absolute Error
    'model_info': dict        # Model-specific information
}
```

## 🔬 Methodology

### Forecast Horizons
- h = 1: One-step ahead forecast
- h = 3: Three-month ahead
- h = 6: Six-month ahead
- h = 12: Twelve-month ahead (annual)

### Model Selection
- **BIC**: Bayesian Information Criterion for lag selection
- **Cross-Validation**: For regularization parameters
- **Grid Search**: For hyperparameter tuning (ML methods)

## 📖 References

1. Medeiros, M. C., Vasconcelos, G., Veiga, A., & Zilberman, E. (2018). **Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods.** Journal of Business & Economic Statistics.

2. Original R Implementation: [HDeconometrics](https://github.com/gabrielrvsc/HDeconometrics)

## 📄 License

MIT License - See [LICENSE](../Naghiayik/LICENSE) for details.


