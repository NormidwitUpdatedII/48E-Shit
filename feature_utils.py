"""
Feature Engineering Utilities
Additional utility functions for feature engineering operations.
Extends the main utils.py with feature-specific helpers.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def standardize_features(X, scaler=None, fit=True):
    """
    Standardize features using StandardScaler.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    scaler : StandardScaler, optional
        Pre-fitted scaler (for test data)
    fit : bool
        Whether to fit the scaler
    
    Returns:
    --------
    tuple : (X_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def reduce_dimensionality(X, n_components=None, variance_ratio=0.95, pca=None, fit=True):
    """
    Reduce dimensionality using PCA.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    n_components : int, optional
        Number of components (if None, use variance_ratio)
    variance_ratio : float
        Target explained variance ratio
    pca : PCA, optional
        Pre-fitted PCA (for test data)
    fit : bool
        Whether to fit the PCA
    
    Returns:
    --------
    tuple : (X_reduced, pca)
    """
    if pca is None:
        if n_components is None:
            # Determine components from variance ratio
            temp_pca = PCA()
            temp_pca.fit(X)
            cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_ratio) + 1
            n_components = max(1, min(n_components, X.shape[1]))
        
        pca = PCA(n_components=n_components)
    
    if fit:
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = pca.transform(X)
    
    return X_reduced, pca


def select_features_by_variance(X, threshold=0.01):
    """
    Select features with variance above threshold.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    threshold : float
        Minimum variance threshold
    
    Returns:
    --------
    tuple : (X_selected, mask)
    """
    variances = np.nanvar(X, axis=0)
    mask = variances > threshold
    return X[:, mask], mask


def select_features_by_correlation(X, y, top_k=50):
    """
    Select top-k features by correlation with target.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target variable
    top_k : int
        Number of features to select
    
    Returns:
    --------
    tuple : (X_selected, indices)
    """
    correlations = []
    for col in range(X.shape[1]):
        # Handle NaN
        valid_mask = ~(np.isnan(X[:, col]) | np.isnan(y))
        if valid_mask.sum() > 10:
            corr = np.corrcoef(X[valid_mask, col], y[valid_mask])[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        else:
            correlations.append(0)
    
    correlations = np.array(correlations)
    top_indices = np.argsort(correlations)[-top_k:]
    
    return X[:, top_indices], top_indices


def remove_highly_correlated(X, threshold=0.95):
    """
    Remove highly correlated features.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    threshold : float
        Correlation threshold
    
    Returns:
    --------
    tuple : (X_selected, mask)
    """
    n_features = X.shape[1]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0)
    
    # Find pairs above threshold
    to_remove = set()
    for i in range(n_features):
        if i in to_remove:
            continue
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                to_remove.add(j)
    
    mask = np.ones(n_features, dtype=bool)
    mask[list(to_remove)] = False
    
    return X[:, mask], mask


def handle_missing_values(X, strategy='mean'):
    """
    Handle missing values in feature matrix.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    strategy : str
        'mean', 'median', 'zero', or 'forward_fill'
    
    Returns:
    --------
    np.ndarray : Cleaned feature matrix
    """
    X = X.copy()
    
    if strategy == 'mean':
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    
    elif strategy == 'median':
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0, col_medians)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
    
    elif strategy == 'zero':
        X = np.nan_to_num(X, nan=0.0)
    
    elif strategy == 'forward_fill':
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                # Forward fill
                idx = np.where(~mask, np.arange(len(mask)), 0)
                idx = np.maximum.accumulate(idx)
                X[:, col] = X[idx, col]
    
    # Handle any remaining NaN (e.g., at start for forward_fill)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X


def create_feature_pipeline(X, y=None, standardize=True, reduce_dim=False, 
                           select_by_var=True, select_by_corr=False,
                           remove_correlated=True, handle_nan=True):
    """
    Complete feature preprocessing pipeline.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray, optional
        Target (needed for correlation selection)
    standardize : bool
        Whether to standardize
    reduce_dim : bool or int
        Whether to reduce dimensions (or number of components)
    select_by_var : bool
        Whether to select by variance
    select_by_corr : bool or int
        Whether to select by correlation (or top_k)
    remove_correlated : bool
        Whether to remove highly correlated
    handle_nan : bool
        Whether to handle missing values
    
    Returns:
    --------
    np.ndarray : Processed feature matrix
    """
    # Handle NaN first
    if handle_nan:
        X = handle_missing_values(X, strategy='mean')
    
    # Select by variance
    if select_by_var:
        X, _ = select_features_by_variance(X, threshold=0.01)
    
    # Remove highly correlated
    if remove_correlated and X.shape[1] > 10:
        X, _ = remove_highly_correlated(X, threshold=0.95)
    
    # Select by correlation with target
    if select_by_corr and y is not None:
        top_k = select_by_corr if isinstance(select_by_corr, int) else 50
        X, _ = select_features_by_correlation(X, y, top_k=top_k)
    
    # Standardize
    if standardize:
        X, _ = standardize_features(X)
    
    # Reduce dimensionality
    if reduce_dim:
        n_comp = reduce_dim if isinstance(reduce_dim, int) else None
        X, _ = reduce_dimensionality(X, n_components=n_comp)
    
    return X
