"""Utility functions for PenguinBoost."""

import numpy as np


def check_array(X, dtype=np.float64, copy=False):
    """Validate and convert input array."""
    X = np.asarray(X, dtype=dtype)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if copy:
        X = X.copy()
    return X


def check_target(y):
    """Validate target array."""
    y = np.asarray(y, dtype=np.float64).ravel()
    return y


def detect_categorical(X, threshold=20):
    """Auto-detect categorical features based on unique value count.

    Returns list of feature indices with <= threshold unique values.
    """
    cat_features = []
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if len(np.unique(valid)) <= threshold:
            cat_features.append(j)
    return cat_features
