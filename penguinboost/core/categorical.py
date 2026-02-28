"""Ordered Target Statistics for categorical features (CatBoost-style)."""

import numpy as np


class OrderedTargetEncoder:
    """Encode categorical features using ordered target statistics.

    CatBoost-style approach: for each sample, the target statistic is computed
    using only preceding samples in a random permutation, preventing target leakage.

    For each sample i (in permutation order):
        encoded_i = (sum_targets_before_i + prior * prior_weight) / (count_before_i + prior_weight)
    """

    def __init__(self, smoothing=10.0, random_state=None):
        self.smoothing = smoothing
        self.random_state = random_state
        self.cat_feature_indices_ = None
        self.global_mean_ = None
        self.category_stats_ = {}  # {feature_idx: {category: (sum, count)}}

    def fit(self, X, y, cat_features=None):
        """Fit encoder: store category-level statistics for transform at predict time."""
        self.global_mean_ = np.mean(y)

        if cat_features is None:
            self.cat_feature_indices_ = []
            return self

        self.cat_feature_indices_ = list(cat_features)

        for j in self.cat_feature_indices_:
            stats = {}
            col = X[:, j]
            for cat in np.unique(col):
                if np.isnan(cat):
                    continue
                mask = col == cat
                stats[cat] = (y[mask].sum(), mask.sum())
            self.category_stats_[j] = stats

        return self

    def transform_train(self, X, y):
        """Apply ordered target statistics to training data.

        Uses a random permutation so each sample only sees preceding targets.
        """
        if not self.cat_feature_indices_:
            return X.copy()

        rng = np.random.RandomState(self.random_state)
        X_encoded = X.astype(np.float64).copy()
        perm = rng.permutation(len(y))

        for j in self.cat_feature_indices_:
            col = X[:, j]
            encoded = np.full(len(y), self.global_mean_, dtype=np.float64)

            # Accumulate stats in permutation order
            cat_sum = {}
            cat_count = {}

            for idx in perm:
                cat = col[idx]
                if np.isnan(cat):
                    continue
                s = cat_sum.get(cat, 0.0)
                c = cat_count.get(cat, 0)
                encoded[idx] = (s + self.global_mean_ * self.smoothing) / (c + self.smoothing)
                cat_sum[cat] = s + y[idx]
                cat_count[cat] = c + 1

            X_encoded[:, j] = encoded

        return X_encoded

    def transform(self, X):
        """Apply target statistics to test data using stored category stats."""
        if not self.cat_feature_indices_:
            return X.copy()

        X_encoded = X.astype(np.float64).copy()

        for j in self.cat_feature_indices_:
            col = X[:, j]
            encoded = np.full(len(col), self.global_mean_, dtype=np.float64)
            stats = self.category_stats_.get(j, {})

            for i in range(len(col)):
                cat = col[i]
                if np.isnan(cat):
                    continue
                if cat in stats:
                    s, c = stats[cat]
                    encoded[i] = (s + self.global_mean_ * self.smoothing) / (c + self.smoothing)

            X_encoded[:, j] = encoded

        return X_encoded
