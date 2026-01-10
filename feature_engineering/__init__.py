"""
Feature Engineering Module

This module provides comprehensive feature engineering for inflation forecasting.
All imports should go through this __init__.py.

Usage:
    from feature_engineering import StationaryFeatureEngineer
    from feature_engineering import engineer_features_for_model
    from feature_engineering.feature_utils import handle_missing_values
    from feature_engineering.feature_config import ROLLING_WINDOWS
"""

# Import main classes and functions for easy access
from .feature_engineering import (
    StationaryFeatureEngineer,
    engineer_features_for_model,
    calculate_rolling_statistics,
    calculate_momentum_features,
    calculate_volatility_features,
    calculate_zscore,
    create_target_features,
    create_interaction_features
)

from .feature_config import (
    ROLLING_WINDOWS,
    MOMENTUM_HORIZONS,
    VOLATILITY_WINDOWS,
    ZSCORE_WINDOW,
    OUTLIER_THRESHOLD,
    N_PCA_COMPONENTS,
    CONSTANT_VARIANCE_THRESHOLD,
    CORRELATION_THRESHOLD,
    LOW_VARIANCE_THRESHOLD,
    EMBED_DIMENSION
)

from .feature_utils import (
    standardize_features,
    handle_missing_values,
    apply_3stage_feature_selection,
    remove_highly_correlated,
    select_features_by_variance,
    select_features_by_correlation
)

__all__ = [
    # Main class
    'StationaryFeatureEngineer',
    
    # Functions
    'engineer_features_for_model',
    'calculate_rolling_statistics',
    'calculate_momentum_features',
    'calculate_volatility_features',
    'calculate_zscore',
    'create_target_features',
    'create_interaction_features',
    'standardize_features',
    'handle_missing_values',
    'apply_3stage_feature_selection',
    'remove_highly_correlated',
    'select_features_by_variance',
    'select_features_by_correlation',
    
    # Config
    'ROLLING_WINDOWS',
    'MOMENTUM_HORIZONS',
    'VOLATILITY_WINDOWS',
    'ZSCORE_WINDOW',
    'OUTLIER_THRESHOLD',
    'N_PCA_COMPONENTS',
    'CONSTANT_VARIANCE_THRESHOLD',
    'CORRELATION_THRESHOLD',
    'LOW_VARIANCE_THRESHOLD',
    'EMBED_DIMENSION'
]
