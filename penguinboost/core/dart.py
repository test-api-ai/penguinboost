"""DART (Dropout Additive Regression Trees) from XGBoost.

Also implements Era-aware DART: trees that are only helpful in specific eras
(high era-variance of per-era Spearman correlation) are dropped more often,
keeping only trees that contribute uniformly across all time periods.
"""

import numpy as np


class DARTManager:
    """Manages tree dropout for DART regularization.

    At each iteration, drops existing trees with probability `drop_rate`.
    The new tree's predictions are scaled by 1/(1 + n_dropped) to compensate.

    Parameters
    ----------
    drop_rate : float
        Probability of dropping each existing tree (0 to 1).
    skip_drop : float
        Probability of skipping dropout entirely for an iteration.
    """

    def __init__(self, drop_rate=0.1, skip_drop=0.0):
        self.drop_rate = drop_rate
        self.skip_drop = skip_drop
        self._dropped_indices = []

    def sample_drops(self, n_trees, rng):
        """Determine which trees to drop for this iteration.

        Parameters
        ----------
        n_trees : int
            Number of existing trees.
        rng : np.random.RandomState
            Random state for reproducibility.

        Returns
        -------
        list of int
            Indices of trees to drop.
        """
        if n_trees == 0:
            self._dropped_indices = []
            return self._dropped_indices

        # ドロップアウトを完全にスキップする可能性
        if self.skip_drop > 0 and rng.random() < self.skip_drop:
            self._dropped_indices = []
            return self._dropped_indices

        # 各ツリーを drop_rate の確率で独立にドロップ
        mask = rng.random(n_trees) < self.drop_rate
        self._dropped_indices = np.where(mask)[0].tolist()

        # 少なくとも 1 本のツリーを残す（全ドロップを防止）
        if len(self._dropped_indices) == n_trees and n_trees > 0:
            keep = rng.randint(n_trees)
            self._dropped_indices.remove(keep)

        return self._dropped_indices

    def compute_scale_factor(self):
        """Compute scale factor for the new tree: 1/(1 + n_dropped).

        Returns
        -------
        float
            Scale factor for the new tree's predictions.
        """
        n_dropped = len(self._dropped_indices)
        return 1.0 / (1.0 + n_dropped)

    def adjust_predictions(self, predictions, trees, learning_rate, dropped_indices):
        """Remove contributions of dropped trees from predictions.

        Parameters
        ----------
        predictions : np.ndarray
            Current cumulative predictions.
        trees : list
            List of (tree, col_indices) tuples.
        learning_rate : float
            Learning rate used during training.
        dropped_indices : list of int
            Indices of trees to drop.

        Returns
        -------
        np.ndarray
            Adjusted predictions with dropped trees removed.
        """
        adjusted = predictions.copy()
        for idx in dropped_indices:
            tree, col_indices, X_binned = trees[idx]
            if col_indices is not None:
                pred = tree.predict(X_binned[:, col_indices])
            else:
                pred = tree.predict(X_binned)
            adjusted -= learning_rate * pred
        return adjusted


class EraAwareDARTManager(DARTManager):
    """Era-aware DART: drop trees with high era-instability more often.

    Motivation
    ----------
    A tree that achieves high Spearman correlation in some eras but near-zero
    in others increases overall prediction variance (hurts Sharpe).  By
    tracking the era-variance of each tree's per-era Spearman correlation we
    can assign era-unstable trees a higher drop probability, keeping only the
    trees that contribute uniformly across time.

    Drop probability formula
    ------------------------
    Given per-era variances var_1, …, var_M for the M trained trees:

        p_drop(m) = sigmoid(scale * (var_m - median(var)))

    Centring on the median ensures that roughly half of trees are above the
    0.5 probability line, so the overall drop rate stays close to the base
    ``drop_rate``.  ``scale`` controls the sharpness of the selection.

    Parameters
    ----------
    drop_rate : float
        Fallback base drop probability used when era statistics are
        unavailable.
    skip_drop : float
        Probability of skipping dropout entirely for one iteration.
    era_var_scale : float
        Sigmoid sharpness parameter (``scale`` above). Larger values →
        more aggressive selection against era-unstable trees (default 20).
    """

    def __init__(self, drop_rate=0.1, skip_drop=0.0, era_var_scale=20.0):
        super().__init__(drop_rate=drop_rate, skip_drop=skip_drop)
        self.era_var_scale = era_var_scale
        self._tree_era_vars = []   # per-tree era-variance of Spearman corr

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _spearman_corr(x, y):
        """Fast Spearman rank correlation (pure numpy)."""
        if len(x) < 2:
            return 0.0
        rx = np.argsort(np.argsort(x, kind='stable'), kind='stable').astype(np.float64)
        ry = np.argsort(np.argsort(y, kind='stable'), kind='stable').astype(np.float64)
        c = np.corrcoef(rx, ry)
        v = float(c[0, 1])
        return v if np.isfinite(v) else 0.0

    # ── public interface ──────────────────────────────────────────────────────

    def record_tree_era_variance(self, tree_pred, y, era_indices):
        """Record the era-variance of Spearman correlations for a new tree.

        Call this once per boosting round, immediately after predicting with
        the newly added tree (before accumulating into ensemble predictions).

        Parameters
        ----------
        tree_pred : np.ndarray of shape (n_samples,)
            Predictions of the *single* new tree (not the full ensemble).
        y : np.ndarray of shape (n_samples,)
            Training targets (1-D).
        era_indices : np.ndarray of shape (n_samples,)
            Era label for each sample.
        """
        era_labels = np.unique(era_indices)
        corrs = []
        for era in era_labels:
            mask = era_indices == era
            if mask.sum() < 2:
                continue
            corrs.append(self._spearman_corr(tree_pred[mask], y[mask]))
        if len(corrs) < 2:
            self._tree_era_vars.append(0.0)
        else:
            self._tree_era_vars.append(float(np.var(corrs)))

    def sample_drops(self, n_trees, rng):
        """Era-aware drop sampling.

        Uses era-variance-weighted drop probabilities when era statistics
        are available; falls back to the parent class otherwise.
        """
        if n_trees == 0:
            self._dropped_indices = []
            return self._dropped_indices

        if self.skip_drop > 0 and rng.random() < self.skip_drop:
            self._dropped_indices = []
            return self._dropped_indices

        n_stats = len(self._tree_era_vars)
        if n_stats < n_trees:
            # Not enough era stats yet → use base DART
            return super().sample_drops(n_trees, rng)

        vars_arr = np.array(self._tree_era_vars[:n_trees], dtype=np.float64)
        median_var = np.median(vars_arr)

        # sigmoid(scale * (var_m - median_var)): centred so that trees with
        # above-median era-variance get p > 0.5 and below-median get p < 0.5
        z = self.era_var_scale * (vars_arr - median_var)
        p_drops = 1.0 / (1.0 + np.exp(-z))

        mask = rng.random(n_trees) < p_drops
        self._dropped_indices = np.where(mask)[0].tolist()

        # Ensure at least one tree survives
        if len(self._dropped_indices) == n_trees and n_trees > 0:
            keep = rng.randint(n_trees)
            self._dropped_indices.remove(keep)

        return self._dropped_indices
