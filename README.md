# Inflation Forecasting with Machine Learning

A comprehensive Python implementation for inflation forecasting using machine learning methods. This project implements the methodology from **"Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods"** by Medeiros et al. (2021).

## Overview

- **21+ Forecasting Methods**: Random Forest, XGBoost, LSTM, LASSO, Ridge, and more
- **Dynamic Period Selection**: Each model can run on different sample periods
- **RMSE by Horizon (h1-h12)**: All models output RMSE for each forecast horizon
- **Outlier Handling**: `with_dummy` sample includes 2008 crisis dummy variable
- **Rolling Window Evaluation**: Expanding window forecasts with comprehensive error metrics

## Project Structure

```
48E-Project-Files/
├── requirements.txt           # Python dependencies
├── utils.py                   # Shared utility functions
├── fred_md_loader.py          # FRED-MD data loader
├── without_dummy/             # Periods WITHOUT 2008 crisis (no outlier dummy)
│   ├── rawdata_1990_2000.csv  # Great Moderation period
│   ├── rawdata_2016_2022.csv  # COVID/inflation period (default)
│   ├── rawdata_2020_2022.csv  # Pandemic subset
│   ├── functions/             # Model implementations
│   └── run/                   # Execution scripts
└── with_dummy/                # Periods WITH 2008 crisis (with outlier dummy)
    ├── rawdata_2001_2015.csv  # Financial crisis period
    ├── rawdata_1990_2022.csv  # Full sample (default)
    ├── functions/             # Model implementations
    └── run/                   # Execution scripts
```

## Sample Periods

### without_dummy (No 2008 Crisis, No Outlier Dummy)
| Period | Observations | nprev | Description |
|--------|--------------|-------|-------------|
| 1990_2000 | 132 | 60 | Great Moderation |
| **2016_2022** | 84 | 48 | COVID/inflation (**default**) |
| 2020_2022 | 36 | 24 | Pandemic subset |

### with_dummy (Has 2008 Crisis, With Outlier Dummy)
| Period | Observations | nprev | Description |
|--------|--------------|-------|-------------|
| 2001_2015 | 180 | 84 | Financial crisis & recovery |
| **1990_2022** | 396 | 132 | Full sample (**default**) |

## Changing the Period

Edit `CURRENT_PERIOD` in any run file:

```python
PERIOD_CONFIG = {
    '1990_2000': {'nprev': 60},
    '2016_2022': {'nprev': 48},
    '2020_2022': {'nprev': 24},
}
CURRENT_PERIOD = '2016_2022'  # Change this to run different period
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run Random Forest (without_dummy - 2016-2022, no outlier dummy)
python without_dummy/run/rf.py

# Run Random Forest (with_dummy - 1990-2022, with outlier dummy)
python with_dummy/run/rf.py
```

### Output Format

```
RMSE BY HORIZON:
Horizon  CPI          PCE
h=1      0.123456     0.123456
...
h=12     0.123456     0.123456
Average: 0.123456     0.123456
```

## Methods Implemented

- **Benchmark**: Random Walk
- **Linear**: AR, Ridge, LASSO, Elastic Net, Adaptive LASSO, SCAD
- **Machine Learning**: Random Forest, XGBoost, Neural Network, LSTM, Boosting, Bagging
- **Factor Methods**: Factor Models, Targeted Factors, CSR, RF-OLS
- **Bayesian**: LBVAR, UC-SV, Jackknife

## References

Medeiros, M. C., Vasconcelos, G., Veiga, A., & Zilberman, E. (2018). **Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods.** Journal of Business & Economic Statistics.
