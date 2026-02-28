"""Histogram-based split finding with subtraction trick (LightGBM-style).

v2: Bayesian adaptive gain, stability penalty, and monotone constraint support.
C++: build_histogram and find_best_split_basic are accelerated via _core.
"""

import numpy as np

try:
    from penguinboost import _core as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


class HistogramBuilder:
    """Builds gradient/hessian histograms for binned features and finds best splits."""

    def __init__(self, max_bins=255):
        self.max_bins = max_bins

    def build_histogram(self, X_binned, gradients, hessians, sample_indices=None):
        """Build gradient and hessian histograms for all features.

        Uses the C++ single-pass implementation when available; falls back to
        numpy bincount otherwise.

        Returns:
            grad_hist: (n_features, max_bins+1) gradient sums per bin
            hess_hist: (n_features, max_bins+1) hessian sums per bin
            count_hist: (n_features, max_bins+1) sample counts per bin
        """
        if _HAS_CPP:
            return _cpp.build_histogram(
                X_binned, gradients, hessians,
                sample_indices, self.max_bins)

        # ---- Pure-Python fallback ----
        if sample_indices is not None:
            X_sub = X_binned[sample_indices]
            g_sub = gradients[sample_indices]
            h_sub = hessians[sample_indices]
        else:
            X_sub = X_binned
            g_sub = gradients
            h_sub = hessians

        n_features = X_sub.shape[1]
        n_bins = self.max_bins + 1  # +1 for NaN bin

        grad_hist = np.zeros((n_features, n_bins), dtype=np.float64)
        hess_hist = np.zeros((n_features, n_bins), dtype=np.float64)
        count_hist = np.zeros((n_features, n_bins), dtype=np.int64)

        for j in range(n_features):
            bins = X_sub[:, j]
            grad_hist[j] = np.bincount(bins, weights=g_sub, minlength=n_bins)
            hess_hist[j] = np.bincount(bins, weights=h_sub, minlength=n_bins)
            count_hist[j] = np.bincount(bins, minlength=n_bins)

        return grad_hist, hess_hist, count_hist

    def subtract_histograms(self, parent_hist, sibling_hist):
        """Subtraction trick: child = parent - sibling (avoids rebuilding)."""
        grad_p, hess_p, count_p = parent_hist
        grad_s, hess_s, count_s = sibling_hist
        return (grad_p - grad_s, hess_p - hess_s, count_p - count_s)

    def find_best_split(self, grad_hist, hess_hist, count_hist,
                        reg_lambda=1.0, reg_alpha=0.0, min_child_weight=1.0,
                        min_child_samples=1, adaptive_reg=None, iteration=0,
                        total_iterations=1, stability_tracker=None,
                        monotone_checker=None):
        """Find the best split across all features using histogram bins.

        Fully vectorized over features and bins: computes all splits in a single
        set of numpy operations, avoiding Python-level feature/bin loops.

        v2 additions:
        - adaptive_reg: AdaptiveRegularizer for per-node lambda
        - stability_tracker: FeatureStabilityTracker for gain variance penalty
        - monotone_checker: MonotoneConstraintChecker for constraint enforcement

        Returns:
            best_feature, best_bin, best_gain, best_nan_direction
        """
        # ---- C++ fast path (no adaptive_reg / no monotone constraints) ----
        if (_HAS_CPP
                and adaptive_reg is None
                and monotone_checker is None):
            feat, bin_t, gain, nan_dir = _cpp.find_best_split_basic(
                grad_hist, hess_hist, count_hist,
                self.max_bins, reg_lambda, reg_alpha,
                min_child_weight, min_child_samples)
            return feat, bin_t, gain, nan_dir

        n_features = grad_hist.shape[0]
        mb = self.max_bins

        # --- Slice valid-bin and NaN-bin parts ---
        g_valid = grad_hist[:, :mb]      # (n_features, max_bins)
        h_valid = hess_hist[:, :mb]
        c_valid = count_hist[:, :mb]     # int64 (no cast needed)

        g_nan = grad_hist[:, mb]         # (n_features,)
        h_nan = hess_hist[:, mb]
        c_nan = count_hist[:, mb]        # already int64

        g_total = g_valid.sum(axis=1)    # (n_features,)
        h_total = h_valid.sum(axis=1)
        c_total = c_valid.sum(axis=1)    # int64

        g_sum = g_total + g_nan          # (n_features,)
        h_sum = h_total + h_nan
        c_sum = c_total + c_nan          # (n_features,)

        # --- Prefix cumulative sums along bins ---
        # Shape: (n_features, max_bins), element [j, b] = sum(hist[j, 0..b])
        g_cumL = np.cumsum(g_valid, axis=1)
        h_cumL = np.cumsum(h_valid, axis=1)
        c_cumL = np.cumsum(c_valid, axis=1)   # int64, no cast needed

        # Broadcast helpers: add singleton dims for nan_dir and bin axes
        g_nan_e = g_nan[:, np.newaxis]          # (n_features, 1)
        h_nan_e = h_nan[:, np.newaxis]
        c_nan_e = c_nan[:, np.newaxis]          # int64, (n_features, 1)

        # --- Build left-child stats for both nan_dirs simultaneously ---
        # Stack shape: (2, n_features, max_bins)
        #   axis-0 index 0 → nan_dir=0 (NaN goes left)
        #   axis-0 index 1 → nan_dir=1 (NaN goes right)
        g_L = np.stack([g_cumL + g_nan_e, g_cumL])
        h_L = np.stack([h_cumL + h_nan_e, h_cumL])
        c_L = np.stack([c_cumL + c_nan_e, c_cumL])

        # Right-child stats via complement
        g_sum_e = g_sum[np.newaxis, :, np.newaxis]   # (1, n_features, 1)
        h_sum_e = h_sum[np.newaxis, :, np.newaxis]
        c_sum_e = c_sum[np.newaxis, :, np.newaxis]

        g_R = g_sum_e - g_L              # (2, n_features, max_bins)
        h_R = h_sum_e - h_L
        c_R = c_sum_e - c_L

        # --- Validity mask ---
        valid = ((h_L >= min_child_weight) & (h_R >= min_child_weight) &
                 (c_L >= min_child_samples) & (c_R >= min_child_samples))
        # (2, n_features, max_bins) bool

        # Early exit if no valid split anywhere
        if not valid.any():
            return -1, -1, 0.0, 0

        # --- Compute split gains (fully vectorized) ---
        # Parent score per feature: (n_features,) → (1, n_features, 1)
        s_total = self._score_vec(g_sum, h_sum, reg_lambda, reg_alpha)
        s_total_e = s_total[np.newaxis, :, np.newaxis]

        if adaptive_reg is not None:
            _schedule = 1.0 + adaptive_reg.alpha * iteration / max(total_iterations, 1)
            lambda_l = (adaptive_reg.lambda_base * _schedule
                        + adaptive_reg.mu / np.sqrt(np.maximum(c_L, 1)))
            lambda_r = (adaptive_reg.lambda_base * _schedule
                        + adaptive_reg.mu / np.sqrt(np.maximum(c_R, 1)))
            s_L = self._score_vec(g_L, h_L, lambda_l, reg_alpha)
            s_R = self._score_vec(g_R, h_R, lambda_r, reg_alpha)
        else:
            s_L = self._score_vec(g_L, h_L, reg_lambda, reg_alpha)
            s_R = self._score_vec(g_R, h_R, reg_lambda, reg_alpha)

        gains = 0.5 * (s_L + s_R - s_total_e)   # (2, n_features, max_bins)

        # --- v2: monotone constraint filter (per-constrained-feature) ---
        if monotone_checker is not None and monotone_checker.has_constraints():
            for j, constraint in monotone_checker.constraints.items():
                if j >= n_features or constraint == 0:
                    continue
                left_val = self._leaf_value_vec(
                    g_L[:, j, :], h_L[:, j, :], reg_lambda, reg_alpha)
                right_val = self._leaf_value_vec(
                    g_R[:, j, :], h_R[:, j, :], reg_lambda, reg_alpha)
                if constraint == 1:
                    valid[:, j, :] &= (right_val >= left_val)
                else:
                    valid[:, j, :] &= (right_val <= left_val)

        # --- Mask invalid splits and find global optimum ---
        gains = np.where(valid, gains, 0.0)
        flat_idx = int(np.argmax(gains))
        max_gain = float(gains.ravel()[flat_idx])

        if max_gain <= 0.0:
            return -1, -1, 0.0, 0

        best_nan_dir, best_feature, best_bin = np.unravel_index(flat_idx, gains.shape)

        # v2: stability_tracker penalty (no-op as in original)
        if stability_tracker is not None:
            pass

        return int(best_feature), int(best_bin), max_gain, int(best_nan_dir)

    # --- Vectorized helpers (shape-agnostic, work on any ndarray shape) ---

    @staticmethod
    def _score_vec(g, h, lam, reg_alpha):
        """Vectorized node score. Works with any shape numpy arrays."""
        if reg_alpha == 0.0:
            # Fast path: no L1 regularization (common default)
            return g * g / (h + lam)
        abs_g = np.abs(g)
        g_adj = g - np.sign(g) * reg_alpha
        return np.where(abs_g > reg_alpha, g_adj * g_adj / (h + lam), 0.0)

    @staticmethod
    def _leaf_value_vec(g, h, reg_lambda, reg_alpha):
        """Vectorized leaf value for monotone constraint checking."""
        if reg_alpha == 0.0:
            return -g / (h + reg_lambda)
        abs_g = np.abs(g)
        g_adj = g - np.sign(g) * reg_alpha
        return np.where(abs_g > reg_alpha, -g_adj / (h + reg_lambda), 0.0)

    # --- Original scalar methods (kept for compatibility) ---

    def _compute_gain(self, g_left, h_left, g_right, h_right,
                      g_total, h_total, reg_lambda, reg_alpha):
        """Compute split gain with L1/L2 regularization (v1 formula)."""
        def score(g, h):
            if abs(g) <= reg_alpha:
                return 0.0
            g_adj = g - np.sign(g) * reg_alpha
            return (g_adj ** 2) / (h + reg_lambda)

        return 0.5 * (score(g_left, h_left) + score(g_right, h_right)
                       - score(g_total, h_total))

    def _compute_gain_v2(self, g_left, h_left, g_right, h_right,
                         g_total, h_total, lambda_l, lambda_r,
                         reg_lambda, reg_alpha):
        """Compute split gain with adaptive per-child lambda (v2 formula).

        Gain = 0.5 * [G_L^2/(H_L+lambda_L) + G_R^2/(H_R+lambda_R) - G^2/(H+lambda)] - gamma
        """
        def score_adaptive(g, h, lam):
            if abs(g) <= reg_alpha:
                return 0.0
            g_adj = g - np.sign(g) * reg_alpha
            return (g_adj ** 2) / (h + lam)

        def score_parent(g, h):
            if abs(g) <= reg_alpha:
                return 0.0
            g_adj = g - np.sign(g) * reg_alpha
            return (g_adj ** 2) / (h + reg_lambda)

        return 0.5 * (score_adaptive(g_left, h_left, lambda_l)
                       + score_adaptive(g_right, h_right, lambda_r)
                       - score_parent(g_total, h_total))

    @staticmethod
    def _leaf_value(g, h, reg_lambda, reg_alpha):
        """Compute leaf value for monotone constraint checking."""
        if abs(g) <= reg_alpha:
            return 0.0
        return -(g - np.sign(g) * reg_alpha) / (h + reg_lambda)
