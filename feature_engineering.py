"""
Advanced Feature Engineering for Inflation Forecasting
Creates stationary features from raw macroeconomic data for improved model performance.

This module provides comprehensive feature transformations including:
- Basic transformations (pct_change, diff, log_diff, yoy, qoq)
- Rolling statistics (mean, std, max, min)
- Momentum features (rate of change, acceleration)
- Volatility features
- Z-score and outlier detection
- Cross-sectional features (spreads, ratios)

Compatible with existing Naghiayik project structure.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from feature_config import (
    ROLLING_WINDOWS, MOMENTUM_HORIZONS, VOLATILITY_WINDOWS,
    ZSCORE_WINDOW, OUTLIER_THRESHOLD, N_PCA_COMPONENTS
)


# =============================================================================
# Utility Functions for Feature Engineering
# =============================================================================

def calculate_zscore(series, window=12):
    """Calculate rolling z-score for a series."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    rolling_std = rolling_std.replace(0, np.nan)  # Avoid division by zero
    return (series - rolling_mean) / rolling_std


def check_stationarity(series, significance=0.05):
    """
    Check stationarity using Augmented Dickey-Fuller test.
    
    Returns:
    --------
    dict : {'stationary': bool, 'p_value': float}
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna())
        return {'stationary': result[1] < significance, 'p_value': result[1]}
    except:
        return {'stationary': False, 'p_value': 1.0}


def calculate_rolling_statistics(series, windows=None):
    """
    Calculate rolling statistics for different window sizes.
    
    Parameters:
    -----------
    series : pd.Series or np.array
        Input time series
    windows : list
        List of window sizes (default: [3, 6, 12])
    
    Returns:
    --------
    pd.DataFrame : Rolling statistics
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    
    stats = pd.DataFrame(index=series.index if hasattr(series, 'index') else range(len(series)))
    
    for w in windows:
        stats[f'mean_{w}'] = series.rolling(w).mean()
        stats[f'std_{w}'] = series.rolling(w).std()
        stats[f'max_{w}'] = series.rolling(w).max()
        stats[f'min_{w}'] = series.rolling(w).min()
        stats[f'range_{w}'] = stats[f'max_{w}'] - stats[f'min_{w}']
        stats[f'skew_{w}'] = series.rolling(w).skew()
    
    return stats


def calculate_momentum_features(series, horizons=None):
    """
    Calculate momentum and acceleration features.
    
    Parameters:
    -----------
    series : pd.Series or np.array
        Input time series
    horizons : list
        List of momentum horizons (default: [3, 6, 12])
    
    Returns:
    --------
    pd.DataFrame : Momentum features
    """
    if horizons is None:
        horizons = MOMENTUM_HORIZONS
    
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    
    momentum = pd.DataFrame(index=series.index if hasattr(series, 'index') else range(len(series)))
    
    for h in horizons:
        # Rate of change
        momentum[f'roc_{h}'] = series.pct_change(h)
        # Acceleration (second derivative)
        momentum[f'acc_{h}'] = series.diff(h).diff(h)
        # Momentum (current vs h periods ago)
        momentum[f'mom_{h}'] = series - series.shift(h)
    
    return momentum


def calculate_volatility_features(series, windows=None):
    """
    Calculate volatility-based features.
    
    Parameters:
    -----------
    series : pd.Series or np.array
        Input time series
    windows : list
        List of window sizes (default: [3, 6, 12])
    
    Returns:
    --------
    pd.DataFrame : Volatility features
    """
    if windows is None:
        windows = VOLATILITY_WINDOWS
    
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    
    vol = pd.DataFrame(index=series.index if hasattr(series, 'index') else range(len(series)))
    
    returns = series.pct_change()
    
    for w in windows:
        # Standard deviation of returns
        vol[f'vol_{w}'] = returns.rolling(w).std()
        # Realized volatility (sum of squared returns)
        vol[f'realized_vol_{w}'] = (returns ** 2).rolling(w).sum().apply(np.sqrt)
        # Volatility of volatility
        vol[f'vol_of_vol_{w}'] = vol[f'vol_{w}'].rolling(w).std()
    
    return vol


def safe_divide(a, b):
    """Safe division handling zeros."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def exponential_smoothing(series, alpha=0.3):
    """Apply exponential smoothing."""
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    return series.ewm(alpha=alpha).mean()


# =============================================================================
# Main Feature Engineering Class
# =============================================================================

class StationaryFeatureEngineer:
    """
    Creates stationary features for inflation forecasting.
    Transforms raw economic data into predictive features.
    """
    
    def __init__(self, n_features=None):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        n_features : int, optional
            Number of features in raw data
        """
        self.n_features = n_features
        self.transformed_features = []
        self.feature_info = {}
        self.stationarity_results = {}
    
    def apply_basic_transforms(self, Y):
        """
        Apply basic stationary transformations to all features.
        
        Parameters:
        -----------
        Y : np.ndarray
            Raw data matrix (n_obs x n_features)
        
        Returns:
        --------
        np.ndarray : Basic transformed features
        """
        Y = np.array(Y)
        n_obs, n_cols = Y.shape
        transforms = []
        
        for col in range(n_cols):
            series = pd.Series(Y[:, col])
            
            # Skip if all NaN or constant
            if series.isna().all() or series.std() == 0:
                continue
            
            # Percentage change
            pct_change = series.pct_change().values
            transforms.append(pct_change)
            
            # First difference
            diff = series.diff().values
            transforms.append(diff)
            
            # Log difference (for positive series)
            if (series > 0).all():
                log_diff = np.log(series).diff().values
                transforms.append(log_diff)
            
            # Year-over-year change (12 periods)
            if n_obs > 12:
                yoy = ((series - series.shift(12)) / series.shift(12).abs()).values
                transforms.append(yoy)
            
            # Quarter-over-quarter change (3 periods)
            if n_obs > 3:
                qoq = ((series - series.shift(3)) / series.shift(3).abs()).values
                transforms.append(qoq)
        
        if transforms:
            return np.column_stack(transforms)
        return np.array([]).reshape(n_obs, 0)
    
    def apply_rolling_statistics(self, Y, windows=None):
        """
        Apply rolling statistics to all features.
        
        Parameters:
        -----------
        Y : np.ndarray
            Raw data matrix
        windows : list
            Rolling window sizes
        
        Returns:
        --------
        np.ndarray : Rolling statistics features
        """
        if windows is None:
            windows = ROLLING_WINDOWS
        
        Y = np.array(Y)
        n_obs, n_cols = Y.shape
        transforms = []
        
        for col in range(n_cols):
            series = pd.Series(Y[:, col])
            
            if series.isna().all() or series.std() == 0:
                continue
            
            rolling_stats = calculate_rolling_statistics(series, windows)
            
            for stat_col in rolling_stats.columns:
                transforms.append(rolling_stats[stat_col].values)
        
        if transforms:
            return np.column_stack(transforms)
        return np.array([]).reshape(n_obs, 0)
    
    def apply_momentum_features(self, Y, horizons=None):
        """
        Apply momentum features to all columns.
        
        Parameters:
        -----------
        Y : np.ndarray
            Raw data matrix
        horizons : list
            Momentum horizons
        
        Returns:
        --------
        np.ndarray : Momentum features
        """
        if horizons is None:
            horizons = MOMENTUM_HORIZONS
        
        Y = np.array(Y)
        n_obs, n_cols = Y.shape
        transforms = []
        
        for col in range(n_cols):
            series = pd.Series(Y[:, col])
            
            if series.isna().all() or series.std() == 0:
                continue
            
            momentum = calculate_momentum_features(series, horizons)
            
            for mom_col in momentum.columns:
                transforms.append(momentum[mom_col].values)
        
        if transforms:
            return np.column_stack(transforms)
        return np.array([]).reshape(n_obs, 0)
    
    def apply_volatility_features(self, Y, windows=None):
        """
        Apply volatility features to all columns.
        
        Parameters:
        -----------
        Y : np.ndarray
            Raw data matrix
        windows : list
            Volatility windows
        
        Returns:
        --------
        np.ndarray : Volatility features
        """
        if windows is None:
            windows = VOLATILITY_WINDOWS
        
        Y = np.array(Y)
        n_obs, n_cols = Y.shape
        transforms = []
        
        for col in range(n_cols):
            series = pd.Series(Y[:, col])
            
            if series.isna().all() or series.std() == 0:
                continue
            
            vol_features = calculate_volatility_features(series, windows)
            
            for vol_col in vol_features.columns:
                transforms.append(vol_features[vol_col].values)
        
        if transforms:
            return np.column_stack(transforms)
        return np.array([]).reshape(n_obs, 0)
    
    def apply_zscore_features(self, Y, window=None, threshold=None):
        """
        Apply z-score and outlier detection features.
        
        Parameters:
        -----------
        Y : np.ndarray
            Raw data matrix
        window : int
            Z-score window (default: 12)
        threshold : float
            Outlier threshold (default: 2.0)
        
        Returns:
        --------
        np.ndarray : Z-score features
        """
        if window is None:
            window = ZSCORE_WINDOW
        if threshold is None:
            threshold = OUTLIER_THRESHOLD
        
        Y = np.array(Y)
        n_obs, n_cols = Y.shape
        transforms = []
        
        for col in range(n_cols):
            series = pd.Series(Y[:, col])
            
            if series.isna().all() or series.std() == 0:
                continue
            
            # Z-score
            zscore = calculate_zscore(series, window)
            transforms.append(zscore.values)
            
            # Outlier flag
            outlier_flag = (abs(zscore) > threshold).astype(int)
            transforms.append(outlier_flag.values)
            
            # Z-score change
            zscore_change = zscore.diff()
            transforms.append(zscore_change.values)
        
        if transforms:
            return np.column_stack(transforms)
        return np.array([]).reshape(n_obs, 0)
    
    def apply_cross_sectional_features(self, Y):
        """
        Apply cross-sectional features (spreads, ratios between columns).
        
        Parameters:
        -----------
        Y : np.ndarray
            Raw data matrix
        
        Returns:
        --------
        np.ndarray : Cross-sectional features
        """
        Y = np.array(Y)
        n_obs, n_cols = Y.shape
        transforms = []
        
        if n_cols < 2:
            return np.array([]).reshape(n_obs, 0)
        
        # Create spreads between first few columns (typically CPI, PCE, etc.)
        for i in range(min(5, n_cols)):
            for j in range(i + 1, min(5, n_cols)):
                # Spread
                spread = Y[:, i] - Y[:, j]
                transforms.append(spread)
                
                # Ratio (safe)
                ratio = safe_divide(Y[:, i], np.abs(Y[:, j]) + 1e-8)
                transforms.append(ratio)
        
        # Rolling correlation between first two columns (CPI and PCE)
        if n_cols >= 2:
            series1 = pd.Series(Y[:, 0])
            series2 = pd.Series(Y[:, 1])
            rolling_corr = series1.rolling(12).corr(series2)
            transforms.append(rolling_corr.values)
        
        if transforms:
            return np.column_stack(transforms)
        return np.array([]).reshape(n_obs, 0)
    
    def get_all_features(self, Y, include_raw=False):
        """
        Apply all feature engineering transformations.
        
        Parameters:
        -----------
        Y : np.ndarray
            Raw data matrix
        include_raw : bool
            Whether to include raw features (default: False)
        
        Returns:
        --------
        np.ndarray : All engineered features
        """
        Y = np.array(Y)
        n_obs = Y.shape[0]
        
        # Apply all transformations
        basic = self.apply_basic_transforms(Y)
        rolling = self.apply_rolling_statistics(Y)
        momentum = self.apply_momentum_features(Y)
        volatility = self.apply_volatility_features(Y)
        zscore = self.apply_zscore_features(Y)
        cross = self.apply_cross_sectional_features(Y)
        
        # Combine all features
        feature_sets = [basic, rolling, momentum, volatility, zscore, cross]
        
        if include_raw:
            feature_sets.insert(0, Y)
        
        # Filter out empty arrays
        feature_sets = [f for f in feature_sets if f.shape[1] > 0]
        
        if feature_sets:
            all_features = np.hstack(feature_sets)
        else:
            all_features = Y  # Fallback to raw
        
        # Replace inf with nan, then handle nans
        all_features = np.where(np.isinf(all_features), np.nan, all_features)
        
        # Store feature count
        self.n_transformed_features = all_features.shape[1]
        
        return all_features
    
    def get_feature_summary(self):
        """Get summary of created features."""
        return {
            'n_transformed_features': getattr(self, 'n_transformed_features', 0),
            'feature_types': ['basic', 'rolling', 'momentum', 'volatility', 'zscore', 'cross_sectional']
        }


# =============================================================================
# Integration Functions for Existing Models
# =============================================================================

def engineer_features_for_model(Y, include_raw=True, handle_nan='fill'):
    """
    Convenience function to engineer features for model training.
    
    Parameters:
    -----------
    Y : np.ndarray
        Raw data matrix
    include_raw : bool
        Whether to include original features
    handle_nan : str
        How to handle NaN values: 'fill' (mean), 'drop', or 'keep'
    
    Returns:
    --------
    np.ndarray : Engineered features ready for modeling
    """
    fe = StationaryFeatureEngineer()
    features = fe.get_all_features(Y, include_raw=include_raw)
    
    if handle_nan == 'fill':
        # Fill NaN with column mean
        col_means = np.nanmean(features, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        inds = np.where(np.isnan(features))
        features[inds] = np.take(col_means, inds[1])
    elif handle_nan == 'drop':
        # Drop rows with NaN (risky for time series)
        mask = ~np.any(np.isnan(features), axis=1)
        features = features[mask]
    # else 'keep': do nothing
    
    return features


def apply_feature_engineering_to_rolling_window(Y_train, Y_test_row, include_raw=True):
    """
    Apply feature engineering in a rolling window context.
    
    Parameters:
    -----------
    Y_train : np.ndarray
        Training data matrix
    Y_test_row : np.ndarray
        Single test observation (1D array)
    include_raw : bool
        Whether to include original features
    
    Returns:
    --------
    tuple : (X_train_engineered, X_test_engineered)
    """
    fe = StationaryFeatureEngineer()
    
    # Combine for consistent transformation
    Y_combined = np.vstack([Y_train, Y_test_row.reshape(1, -1)])
    
    # Engineer features
    features_combined = fe.get_all_features(Y_combined, include_raw=include_raw)
    
    # Handle NaN
    col_means = np.nanmean(features_combined[:-1], axis=0)  # Mean from train only
    col_means = np.where(np.isnan(col_means), 0, col_means)
    inds = np.where(np.isnan(features_combined))
    features_combined[inds] = np.take(col_means, inds[1])
    
    # Split back
    X_train = features_combined[:-1]
    X_test = features_combined[-1:]
    
    return X_train, X_test


# =============================================================================
# Selective Feature Engineering (for specific columns)
# =============================================================================

def create_target_features(Y, target_col=0, lags=[1, 2, 3, 4]):
    """
    Create lagged features specifically for the target variable.
    
    Parameters:
    -----------
    Y : np.ndarray
        Raw data matrix
    target_col : int
        Index of target column
    lags : list
        Lag periods to create
    
    Returns:
    --------
    np.ndarray : Lagged target features
    """
    target = Y[:, target_col]
    n_obs = len(target)
    
    lagged_features = []
    for lag in lags:
        lagged = np.roll(target, lag)
        lagged[:lag] = np.nan
        lagged_features.append(lagged)
    
    return np.column_stack(lagged_features)


def create_interaction_features(Y, col_pairs=None, max_pairs=10):
    """
    Create interaction features between column pairs.
    
    Parameters:
    -----------
    Y : np.ndarray
        Raw data matrix
    col_pairs : list of tuples
        Specific column pairs to interact (default: auto-select)
    max_pairs : int
        Maximum number of pairs if auto-selecting
    
    Returns:
    --------
    np.ndarray : Interaction features
    """
    Y = np.array(Y)
    n_obs, n_cols = Y.shape
    
    if col_pairs is None:
        # Auto-select: use first few columns
        col_pairs = []
        for i in range(min(5, n_cols)):
            for j in range(i + 1, min(5, n_cols)):
                col_pairs.append((i, j))
                if len(col_pairs) >= max_pairs:
                    break
            if len(col_pairs) >= max_pairs:
                break
    
    interactions = []
    for i, j in col_pairs:
        if i < n_cols and j < n_cols:
            # Product
            interactions.append(Y[:, i] * Y[:, j])
            # Ratio
            interactions.append(safe_divide(Y[:, i], np.abs(Y[:, j]) + 1e-8))
    
    if interactions:
        return np.column_stack(interactions)
    return np.array([]).reshape(n_obs, 0)
