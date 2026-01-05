"""
XGBoost with Feature Engineering - Second Sample
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'rawdata.csv')
FORECAST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'forecasts')

from utils import load_csv, save_forecasts, embed, calculate_errors, add_outlier_dummy
from feature_engineering import StationaryFeatureEngineer
from feature_utils import standardize_features, handle_missing_values, select_features_by_variance, remove_highly_correlated

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

XGB_PARAMS = {
    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0
}


def run_xgb_fe(Y, indice, lag):
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost required")
    
    indice = indice - 1
    Y = np.array(Y)
    
    fe = StationaryFeatureEngineer()
    Y_engineered = fe.get_all_features(Y, include_raw=True)
    Y_engineered = handle_missing_values(Y_engineered, strategy='mean')
    
    aux = embed(Y_engineered, 4 + lag)
    y_target = embed(Y[:, indice].reshape(-1, 1), 4 + lag)[:, 0]
    
    min_len = min(len(aux), len(y_target))
    aux, y_target = aux[:min_len], y_target[:min_len]
    
    n_cols = Y_engineered.shape[1]
    X = aux[:, n_cols * lag:]
    
    if lag == 1:
        X_out = aux[-1, :X.shape[1]]
    else:
        aux_trimmed = aux[:, :aux.shape[1] - n_cols * (lag - 1)]
        X_out = aux_trimmed[-1, :X.shape[1]]
    
    y = y_target[:len(y_target) - lag + 1]
    X = X[:X.shape[0] - lag + 1, :]
    
    X, var_mask = select_features_by_variance(X, threshold=0.001)
    X_out = X_out[var_mask]
    
    if X.shape[1] > 100:
        X, corr_mask = remove_highly_correlated(X, threshold=0.95)
        X_out = X_out[corr_mask]
    
    X, scaler = standardize_features(X)
    X_out = scaler.transform(X_out.reshape(1, -1))
    
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    
    return {'model': model, 'pred': model.predict(X_out)[0]}


def xgb_fe_rolling_window(Y, nprev, indice, lag):
    Y = np.array(Y)
    nobs = Y.shape[0]
    predictions, actuals = [], []
    
    for i in range(nprev, nobs - lag + 1):
        result = run_xgb_fe(Y[:i, :], indice, lag)
        predictions.append(result['pred'])
        actuals.append(Y[i + lag - 1, indice - 1])
    
    predictions, actuals = np.array(predictions), np.array(actuals)
    return {'pred': predictions, 'actuals': actuals, 'errors': calculate_errors(actuals, predictions)}


def main():
    if not XGB_AVAILABLE:
        print("ERROR: XGBoost not installed")
        return
    
    Y = load_csv(DATA_PATH)
    Y = add_outlier_dummy(Y, target_col=0)
    
    nprev = 298
    np.random.seed(123)
    results = {}
    
    print("Running XGBoost with Feature Engineering (Second Sample)...")
    
    for lag in range(1, 13):
        print(f"  Lag={lag}...")
        results[f'xgb_fe{lag}c'] = xgb_fe_rolling_window(Y, nprev, indice=1, lag=lag)
        results[f'xgb_fe{lag}p'] = xgb_fe_rolling_window(Y, nprev, indice=2, lag=lag)
    
    cpi = np.column_stack([results[f'xgb_fe{lag}c']['pred'] for lag in range(1, 13)])
    pce = np.column_stack([results[f'xgb_fe{lag}p']['pred'] for lag in range(1, 13)])
    
    os.makedirs(FORECAST_DIR, exist_ok=True)
    save_forecasts(cpi, os.path.join(FORECAST_DIR, 'xgb-fe-cpi.csv'))
    save_forecasts(pce, os.path.join(FORECAST_DIR, 'xgb-fe-pce.csv'))
    
    print(f"\nForecasts saved to {FORECAST_DIR}")
    return results


if __name__ == '__main__':
    main()
