"""Feature neutralization and orthogonal gradient projection for financial ML.

Numerai-inspired techniques for reducing feature exposure and preventing
the model from learning feature-level linear effects.
"""

import numpy as np


class FeatureNeutralizer:
    """Post-training feature neutralization (Numerai-style).

    Removes the component of predictions that is linearly explained by
    the feature matrix, reducing feature exposure and making predictions
    more orthogonal to the feature space.

    Mathematical formulation:
        p_neutralized = p - proportion * X(X^TX + ε·I)^{-1}X^T p
                      = p - proportion * proj_X(p)

    where proj_X(p) is the orthogonal projection of p onto column space of X.
    When proportion=1.0, predictions are made fully orthogonal to features.
    When proportion=0.5, half the feature exposure is removed.

    Per-era neutralization applies this operation within each time period
    separately, which is the standard Numerai approach.
    """

    def __init__(self, eps=1e-5):
        """
        Parameters
        ----------
        eps : float
            Tikhonov regularization for numerical stability of (X^TX + ε·I)^{-1}.
        """
        self.eps = eps

    def neutralize(self, predictions, X, features=None, proportion=1.0,
                   per_era=False, eras=None):
        """Remove feature exposure from predictions.

        Parameters
        ----------
        predictions : np.ndarray of shape (n_samples,)
            Model predictions to neutralize.
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        features : list of int or None
            Subset of feature indices to neutralize against. None = all features.
        proportion : float in [0, 1]
            Fraction of feature exposure to remove. 1.0 = fully orthogonalized.
        per_era : bool
            If True, apply neutralization separately within each era.
        eras : np.ndarray of shape (n_samples,) or None
            Era labels for per-era neutralization.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Neutralized predictions.
        """
        X_sub = X[:, features] if features is not None else X

        if per_era and eras is not None:
            return self._neutralize_per_era(predictions, X_sub, proportion, eras)

        return self._neutralize_block(predictions, X_sub, proportion)

    def _neutralize_block(self, predictions, X, proportion):
        """Neutralize a single block of predictions against X."""
        # Tikhonov-regularized least squares: (X^TX + ε·I)^{-1} X^T p
        n_feat = X.shape[1]
        XtX = X.T @ X + self.eps * np.eye(n_feat)
        coef = np.linalg.solve(XtX, X.T @ predictions)
        projection = X @ coef
        return predictions - proportion * projection

    def _neutralize_per_era(self, predictions, X, proportion, eras):
        """Apply neutralization independently within each era."""
        result = predictions.copy()
        for era in np.unique(eras):
            mask = eras == era
            if mask.sum() < X.shape[1] + 1:
                # Too few samples for stable inversion — skip this era
                continue
            result[mask] = self._neutralize_block(
                predictions[mask], X[mask], proportion)
        return result

    def feature_exposure(self, predictions, X, features=None):
        """Compute Pearson correlation of predictions with each feature.

        Parameters
        ----------
        predictions : np.ndarray of shape (n_samples,)
        X : np.ndarray of shape (n_samples, n_features)
        features : list of int or None

        Returns
        -------
        np.ndarray of shape (n_features_used,)
            Correlation of predictions with each feature.
        """
        X_sub = X[:, features] if features is not None else X
        n_features = X_sub.shape[1]
        exposures = np.zeros(n_features)
        p_centered = predictions - predictions.mean()
        p_std = predictions.std() + 1e-9
        for j in range(n_features):
            x_j = X_sub[:, j]
            x_std = x_j.std() + 1e-9
            exposures[j] = (p_centered @ (x_j - x_j.mean())) / (
                len(predictions) * p_std * x_std)
        return exposures

    def max_feature_exposure(self, predictions, X, features=None):
        """Maximum absolute feature exposure (scalar summary statistic)."""
        return float(np.max(np.abs(self.feature_exposure(predictions, X, features))))

    def feature_exposure_per_era(self, predictions, X, eras, features=None):
        """Mean absolute feature exposure averaged over eras.

        Useful for verifying that neutralization is effective within eras.
        """
        X_sub = X[:, features] if features is not None else X
        era_labels = np.unique(eras)
        mean_exposures = np.zeros(X_sub.shape[1])
        for era in era_labels:
            mask = eras == era
            if mask.sum() < 2:
                continue
            exp_era = self.feature_exposure(predictions[mask], X_sub[mask])
            mean_exposures += np.abs(exp_era)
        mean_exposures /= max(len(era_labels), 1)
        return mean_exposures


class OrthogonalGradientProjector:
    """Projects boosting gradients orthogonal to the feature subspace.

    Mathematical formulation:
        g_orth = g - strength · X(X^TX + ε·I)^{-1}X^T g

    This removes the component of gradients that is linearly explainable
    by the feature matrix, forcing the boosting algorithm to discover
    true nonlinear alpha signals rather than re-learning feature-level
    effects. The result is a model with lower feature exposure.

    The regularization parameter ε (eps) prevents numerical instability
    when features are highly correlated (near-singular X^TX).

    Parameters
    ----------
    strength : float in [0, 1]
        Fraction of feature-explainable gradient to remove.
        0 = no projection, 1 = full orthogonalization.
    eps : float
        Tikhonov regularization strength for (X^TX + ε·I)^{-1}.
    features : list of int or None
        Feature indices to project against. None = all features.
    """

    def __init__(self, strength=1.0, eps=1e-4, features=None):
        self.strength = strength
        self.eps = eps
        self.features = features
        self._XtX_inv = None
        self._X_sub = None

    def fit(self, X):
        """Pre-compute (X^TX + ε·I)^{-1} for efficient per-iteration projection.

        Call once before training; the cached matrix is reused at every
        boosting iteration.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        """
        self._X_sub = X[:, self.features] if self.features is not None else X
        n_feat = self._X_sub.shape[1]
        XtX = self._X_sub.T @ self._X_sub
        self._XtX_inv = np.linalg.inv(XtX + self.eps * np.eye(n_feat))
        return self

    def project(self, gradients):
        """Remove feature-explainable component from gradients.

        Parameters
        ----------
        gradients : np.ndarray of shape (n_samples,)

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Orthogonalized gradients.
        """
        if self._X_sub is None:
            raise RuntimeError("Call fit() before project().")
        X = self._X_sub
        # g_orth = g - strength * X @ (X^TX + εI)^{-1} @ X^T @ g
        Xtg = X.T @ gradients                       # (n_features,)
        coef = self._XtX_inv @ Xtg                  # (n_features,)
        projection = X @ coef                        # (n_samples,)
        return gradients - self.strength * projection
