# Naghiayik Python - Inflation Forecasting with Machine Learning

A comprehensive Python implementation of the R codebase from **"Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods"** by Medeiros, Vasconcelos, Veiga and Zilberman (2018).

## 📖 Overview

This repository contains Python implementations of various machine learning and econometric methods for inflation forecasting using rolling window evaluation. The original R code has been faithfully converted to Python while maintaining the same structure and methodology.

### Key Features
- **17+ forecasting methods** implemented
- **Rolling window evaluation** with configurable forecast horizons
- **Two sample periods** for analysis (first-sample, second-sample)
- **Modular design** with separate function and run modules
- **Comprehensive error metrics** (RMSE, MAE, MAPE)

## 📁 Project Structure

```
Naghiayik-python/
├── requirements.txt          # Python dependencies
├── utils.py                  # Shared utility functions
├── test_all.py              # Comprehensive test suite
├── first_sample/            # First sample period analysis
│   ├── __init__.py
│   ├── functions/           # Model function implementations
│   │   ├── func_ar.py       # Autoregressive models
│   │   ├── func_lasso.py    # LASSO regression
│   │   ├── func_rf.py       # Random Forest
│   │   ├── func_xgb.py      # XGBoost
│   │   ├── func_nn.py       # Neural Networks
│   │   ├── func_boosting.py # Gradient Boosting
│   │   ├── func_bag.py      # Bagging
│   │   ├── func_csr.py      # Complete Subset Regression
│   │   ├── func_fact.py     # Factor Models
│   │   ├── func_tfact.py    # Targeted Factor Models
│   │   ├── func_scad.py     # SCAD Penalized Regression
│   │   ├── func_jn.py       # Jackknife
│   │   ├── func_rfols.py    # Random Forest OLS
│   │   ├── func_lbvar.py    # Large Bayesian VAR
│   │   ├── func_ucsv.py     # Unobserved Components SV
│   │   ├── func_polilasso.py    # Polynomial LASSO
│   │   ├── func_adalassorf.py   # Adaptive LASSO RF
│   │   └── __init__.py
│   └── run/                 # Execution scripts
│       ├── ar.py            # Run AR models
│       ├── lasso.py         # Run LASSO
│       ├── adalasso.py      # Run Adaptive LASSO
│       ├── elasticnet.py    # Run Elastic Net
│       ├── ridge.py         # Run Ridge Regression
│       ├── rf.py            # Run Random Forest
│       ├── xgb.py           # Run XGBoost
│       ├── nn.py            # Run Neural Networks
│       ├── boosting.py      # Run Boosting
│       ├── bagging.py       # Run Bagging
│       ├── csr.py           # Run CSR
│       ├── factors.py       # Run Factor Models
│       ├── tfactors.py      # Run Targeted Factors
│       ├── scad.py          # Run SCAD
│       ├── jackknife.py     # Run Jackknife
│       ├── rfols.py         # Run RF-OLS
│       ├── lbvar.py         # Run LBVAR
│       ├── ucsv.py          # Run UC-SV
│       └── __init__.py
└── second_sample/           # Second sample period analysis
    ├── __init__.py
    ├── functions/           # Same structure as first_sample
    │   ├── func_flasso.py   # Forecast LASSO (unique)
    │   ├── func_rflasso.py  # RF LASSO (unique)
    │   └── ... (same as first_sample)
    └── run/
        ├── cm.py            # Combination Methods (unique)
        ├── fadalasso.py     # Forecast Adaptive LASSO
        ├── rflasso.py       # RF LASSO
        ├── rlasso.py        # Robust LASSO
        └── ... (same as first_sample)
```

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
| pyreadr | ≥0.5.0 | Load R data files (.rda, .RData) |
| matplotlib | ≥3.7.0 | Visualization (optional) |
| statsmodels | ≥0.14.0 | Statistical models (optional) |

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

Run the comprehensive test suite:

```bash
cd Naghiayik-python
python test_all.py
```

Expected output:
```
============================================================
NAGHIAYIK PYTHON TEST SUITE
============================================================
[1] SYNTAX CHECK
----------------------------------------
  OK: utils.py
  OK: first_sample/functions/func_ar.py
  ... (88 files)

Syntax Results: 88 passed, 0 failed
============================================================
```

## 📚 Data Format

### Input Data Structure
The input data (`rawdata.rda` or `rawdata.RData`) should contain:
- A DataFrame/matrix with time series observations
- First column: target variable (inflation measure)
- Remaining columns: predictor variables (macroeconomic indicators)

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


