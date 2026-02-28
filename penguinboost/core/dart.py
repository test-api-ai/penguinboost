"""DART (Dropout Additive Regression Trees) from XGBoost."""

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
