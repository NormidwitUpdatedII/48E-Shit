# Inflation Forecasting with Machine Learning

A comprehensive Python implementation for inflation forecasting using machine learning methods. This project implements the methodology from **"Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods"** by Medeiros et al. (2021).

## Overview

This repository implements state-of-the-art machine learning methods for inflation forecasting:

- **21+ Forecasting Methods**: Random Forest, XGBoost, LSTM, LASSO, Ridge, and more
- **Two Sample Directories**: first_sample and second_sample for different analyses
- **5 Sample Periods**: 1990-2000, 2001-2015, 2016-2022, 2020-2022, 1990-2022
- **RMSE by Horizon (h1-h12)**: All models output RMSE for each forecast horizon
- **Rolling Window Evaluation**: Expanding window forecasts with comprehensive error metrics

## Project Structure

```
48E-Project-Files/
├── requirements.txt          # Python dependencies
├── utils.py                  # Shared utility functions
├── fred_md_loader.py         # FRED-MD data loader with transformations
├── generate_all_samples.py   # Generate data for all 5 sample periods
├── run_all_models.py         # Master script to run all models
├── data/                     # Raw FRED-MD data
│   └── 2025-11-MD.csv
├── first_sample/
│   ├── rawdata_1990_2000.csv   # 132 observations
│   ├── rawdata_2001_2015.csv   # 180 observations
│   ├── rawdata_2016_2022.csv   # 84 observations
│   ├── rawdata_2020_2022.csv   # 36 observations
│   ├── rawdata_1990_2022.csv   # 396 observations (default)
│   ├── functions/              # Model implementations (21 files)
│   └── run/                    # Execution scripts (26 files)
└── second_sample/              # Same structure as first_sample
```

## Sample Periods

| Period | Date Range | Description | nprev |
|--------|-----------|-------------|-------|
| 1990_2000 | Jan 1990 - Dec 2000 | Low volatility (Great Moderation) | 60 |
| 2001_2015 | Jan 2001 - Dec 2015 | Financial crisis and recovery | 84 |
| 2016_2022 | Jan 2016 - Dec 2022 | COVID-19 and inflation surge | 48 |
| 2020_2022 | Jan 2020 - Dec 2022 | Pandemic subset | 24 |
| 1990_2022 | Jan 1990 - Dec 2022 | Full extended sample (default) | 132 |

## Installation

```bash
pip install -r requirements.txt
```

### Required Packages
- numpy ≥1.24.0
- pandas ≥2.0.0
- scipy ≥1.10.0
- scikit-learn ≥1.3.0
- xgboost ≥2.0.0
- tensorflow ≥2.13.0
- joblib

## Usage

### Running Individual Models

```bash
# Run Random Forest model
python first_sample/run/rf.py

# Run AR model
python first_sample/run/ar.py

# Run XGBoost model  
python first_sample/run/xgb.py
```

### Output Format (RMSE by Horizon)

All models now output RMSE for each forecast horizon (h1-h12):

```
RMSE BY HORIZON:
Horizon  CPI          PCE
h=1      0.123456     0.123456
h=2      0.123456     0.123456
...
h=12     0.123456     0.123456
Average: 0.123456     0.123456
```

### Generating Sample Period Data

```bash
python generate_all_samples.py
```

This creates `rawdata_{period}.csv` files in both `first_sample/` and `second_sample/`.

## Methods Implemented

### Benchmark
- **Random Walk** (`func_rw.py`)

### Linear Methods
- AR (Autoregressive)
- Ridge Regression
- LASSO
- Elastic Net
- Adaptive LASSO
- SCAD

### Machine Learning
- **Random Forest** (`func_rf.py`)
- **XGBoost** (`func_xgb.py`)
- **Neural Network** (`func_nn.py`)
- **LSTM** (`func_lstm.py`)
- Gradient Boosting
- Bagging

### Factor Methods
- Factor Models
- Targeted Factors
- CSR (Complete Subset Regression)
- RF-OLS

### Bayesian
- LBVAR (Large Bayesian VAR)
- UC-SV (Unobserved Components with Stochastic Volatility)
- Jackknife

## FRED-MD Transformations

| Code | Transformation |
|------|----------------|
| 1 | No transformation |
| 2 | First difference: Δx_t |
| 3 | Second difference: Δ²x_t |
| 4 | Log: log(x_t) |
| 5 | Log first difference: Δlog(x_t) |
| 6 | Log second difference: Δ²log(x_t) |
| 7 | Percentage change |

## Parameters

- **nprev**: Number of out-of-sample forecasts
- **indice**: Target variable (1 = CPI, 2 = PCE)
- **lag**: Forecast horizon (1-12)

## References

1. Medeiros, M. C., Vasconcelos, G., Veiga, A., & Zilberman, E. (2018). **Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods.** Journal of Business & Economic Statistics.

2. Original R Implementation: [HDeconometrics](https://github.com/gabrielrvsc/HDeconometrics)

## License

MIT License
