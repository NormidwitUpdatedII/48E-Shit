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
    
    **DATA LEAKAGE PREVENTION:**
    - When fit=True: Only use this on TRAINING data to learn statistics
    - When fit=False: Use this on TEST data with pre-fitted scaler from training
    - Never fit on combined train+test data!
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    scaler : StandardScaler, optional
        Pre-fitted scaler (for test data). If provided with fit=True, will raise error.
    fit : bool
        Whether to fit the scaler on X (True) or just transform (False)
    
    Returns:
    --------
    tuple : (X_scaled, scaler)
    
    Raises:
    -------
    ValueError : If scaler is provided with fit=True (ambiguous intent)
    """
    if scaler is not None and fit:
        raise ValueError(
            "Cannot fit=True when scaler is provided. This may cause data leakage.\n"
            "Use fit=True with scaler=None (fit on training data) OR\n"
            "Use fit=False with provided scaler (transform test data with train statistics)."
        )
    
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
    Remove highly correlated features (keeps feature with HIGHER variance).
    OPTIMIZED for large feature sets using vectorized operations.
    
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
    
    # Calculate variance for each feature (to decide which to keep)
    variances = np.nanvar(X, axis=0)
    
    # Calculate correlation matrix (this is the bottleneck for large n_features)
    print(f"    Computing correlation matrix for {n_features} features...")
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0)
    
    # Find pairs above threshold - VECTORIZED approach
    # Set diagonal to 0 (don't compare feature to itself)
    np.fill_diagonal(corr_matrix, 0)
    
    # Get upper triangle indices where correlation > threshold
    high_corr_pairs = np.where(np.abs(np.triu(corr_matrix, k=1)) > threshold)
    
    to_remove = set()
    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
        if i in to_remove or j in to_remove:
            continue
        # Remove the one with LOWER variance
        if variances[i] >= variances[j]:
            to_remove.add(j)
        else:
            to_remove.add(i)
    
    mask = np.ones(n_features, dtype=bool)
    mask[list(to_remove)] = False
    
    return X[:, mask], mask


def handle_missing_values(X, strategy='mean', fill_values=None):
    """
    Handle missing values in feature matrix.
    
    **DATA LEAKAGE PREVENTION:**
    - When fill_values=None: Compute statistics from X (USE ONLY FOR TRAINING DATA)
    - When fill_values provided: Use precomputed statistics (FOR TEST DATA)
    - Never compute statistics on combined train+test data!
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    strategy : str
        'mean', 'median', 'zero', or 'forward_fill'
    fill_values : np.ndarray, optional
        Precomputed fill values (e.g., mean/median from training data).
        If provided, these will be used instead of computing from X.
        This prevents test data leakage.
    
    Returns:
    --------
    tuple : (X_cleaned, fill_values)
        - X_cleaned: Cleaned feature matrix
        - fill_values: The fill values used (for applying to test data)
    """
    X = X.copy()
    computed_fill_values = fill_values
    
    if strategy == 'mean':
        if fill_values is None:
            # Use forward fill instead of global mean to avoid future data leakage
            # Forward fill uses only past values at each point
            df = pd.DataFrame(X)
            filled = df.fillna(method='ffill').fillna(method='bfill')  # bfill for very beginning
            X = filled.values
            computed_fill_values = None  # Forward fill doesn't have precomputable values
        else:
            # If fill_values provided, use them (should be from training data)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(fill_values, inds[1])
    
    elif strategy == 'median':
        if fill_values is None:
            computed_fill_values = np.nanmedian(X, axis=0)
            computed_fill_values = np.where(np.isnan(computed_fill_values), 0, computed_fill_values)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(computed_fill_values, inds[1])
    
    elif strategy == 'zero':
        X = np.nan_to_num(X, nan=0.0)
        computed_fill_values = np.zeros(X.shape[1])  # For consistency
    
    elif strategy == 'forward_fill':
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                # Forward fill
                idx = np.where(~mask, np.arange(len(mask)), 0)
                idx = np.maximum.accumulate(idx)
                X[:, col] = X[idx, col]
        computed_fill_values = None  # Forward fill doesn't have precomputable values
    
    # Handle any remaining NaN (e.g., at start for forward_fill)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, computed_fill_values


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


def apply_3stage_feature_selection(X, constant_threshold=1e-8, correlation_threshold=0.95, 
                                   variance_threshold=0.005, verbose=True):
    """
    Apply 3-stage feature selection (RECOMMENDED APPROACH).
    
    Stage 1a: Remove TRUE constants (near-zero variance)
    Stage 1b: Remove highly correlated features (keep higher variance)
    Stage 1c: Remove low variance features (conservative threshold)
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    constant_threshold : float
        Threshold for Stage 1a (default: 1e-8)
    correlation_threshold : float
        Threshold for Stage 1b (default: 0.95)
    variance_threshold : float
        Threshold for Stage 1c (default: 0.005)
    verbose : bool
        Whether to print selection statistics
    
    Returns:
    --------
    tuple : (X_selected, selection_info)
        - X_selected: Filtered feature matrix
        - selection_info: Dictionary with selection statistics
    """
    initial_features = X.shape[1]
    selection_info = {
        'initial_features': initial_features,
        'stage_1a_removed': 0,
        'stage_1b_removed': 0,
        'stage_1c_removed': 0,
        'final_features': 0
    }
    
    if verbose:
        print(f"  [Feature Selection] Starting with {initial_features} features")
    
    # Stage 1a: Remove TRUE constants
    if verbose:
        print(f"  [Stage 1a] Removing constants (var < {constant_threshold})...")
    X_stage1a, mask_1a = select_features_by_variance(X, threshold=constant_threshold)
    removed_1a = initial_features - X_stage1a.shape[1]
    selection_info['stage_1a_removed'] = removed_1a
    if verbose:
        print(f"    Removed {removed_1a} constant features → {X_stage1a.shape[1]} features")
    
    # Stage 1b: Remove highly correlated features
    if verbose:
        print(f"  [Stage 1b] Removing correlated features (corr > {correlation_threshold})...")
    X_stage1b, mask_1b = remove_highly_correlated(X_stage1a, threshold=correlation_threshold)
    removed_1b = X_stage1a.shape[1] - X_stage1b.shape[1]
    selection_info['stage_1b_removed'] = removed_1b
    if verbose:
        print(f"    Removed {removed_1b} correlated features → {X_stage1b.shape[1]} features")
    
    # Stage 1c: Remove low variance features (conservative threshold)
    if verbose:
        print(f"  [Stage 1c] Removing low variance (var < {variance_threshold})...")
    X_final, mask_1c = select_features_by_variance(X_stage1b, threshold=variance_threshold)
    removed_1c = X_stage1b.shape[1] - X_final.shape[1]
    selection_info['stage_1c_removed'] = removed_1c
    selection_info['final_features'] = X_final.shape[1]
    
    if verbose:
        print(f"    Removed {removed_1c} low-variance features → {X_final.shape[1]} features")
        print(f"  [Summary] Total reduction: {initial_features} → {X_final.shape[1]} " +
              f"({100*(1-X_final.shape[1]/initial_features):.1f}% reduction)")
    
    # Build combined mask for traceability
    combined_mask = mask_1a.copy()
    combined_mask[combined_mask] = mask_1b
    temp_mask = combined_mask[combined_mask]
    temp_mask[temp_mask] = mask_1c
    combined_mask[combined_mask] = temp_mask
    
    selection_info['combined_mask'] = combined_mask
    
    return X_final, selection_info
