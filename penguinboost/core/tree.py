"""Decision tree with Leaf-wise (LightGBM), Symmetric (CatBoost), and Hybrid growth strategies.

v2: Hybrid growth (symmetric -> leaf-wise), v2 parameter propagation to histogram.
"""

import numpy as np
from penguinboost.core.histogram import HistogramBuilder


class TreeNode:
    """A node in the decision tree."""

    __slots__ = [
        "feature", "threshold", "nan_direction",
        "left", "right", "value", "depth", "gain",
        "sample_indices", "grad_sum", "hess_sum",
    ]

    def __init__(self):
        self.feature = -1       # split feature index
        self.threshold = 0      # split bin threshold
        self.nan_direction = 0  # 0=left, 1=right
        self.left = None
        self.right = None
        self.value = 0.0        # leaf prediction
        self.depth = 0
        self.gain = 0.0
        self.sample_indices = None
        self.grad_sum = 0.0
        self.hess_sum = 0.0


class DecisionTree:
    """Histogram-based decision tree supporting multiple growth strategies.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    max_leaves : int
        Maximum number of leaves (for leaf-wise growth).
    growth : str
        'leafwise', 'symmetric', 'depthwise', or 'hybrid'.
    reg_lambda : float
        L2 regularization on leaf weights.
    reg_alpha : float
        L1 regularization on leaf weights.
    min_child_weight : float
        Minimum sum of hessians in a leaf.
    min_child_samples : int
        Minimum number of samples in a leaf.
    max_bins : int
        Maximum number of histogram bins.
    symmetric_depth : int
        For hybrid growth: depth at which to switch from symmetric to leaf-wise.
    adaptive_reg : AdaptiveRegularizer or None
        v2 adaptive regularization.
    monotone_checker : MonotoneConstraintChecker or None
        v2 monotone constraints.
    iteration : int
        Current boosting iteration (for adaptive regularization).
    total_iterations : int
        Total boosting iterations planned.
    """

    def __init__(self, max_depth=6, max_leaves=31, growth="leafwise",
                 reg_lambda=1.0, reg_alpha=0.0, min_child_weight=1.0,
                 min_child_samples=1, max_bins=255, symmetric_depth=3,
                 adaptive_reg=None, monotone_checker=None,
                 iteration=0, total_iterations=1):
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.growth = growth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.max_bins = max_bins
        self.symmetric_depth = symmetric_depth
        self.adaptive_reg = adaptive_reg
        self.monotone_checker = monotone_checker
        self.iteration = iteration
        self.total_iterations = total_iterations
        self.hist_builder = HistogramBuilder(max_bins)
        self.root = None
        self.split_features_ = []
        self.split_gains_ = []

    def build(self, X_binned, gradients, hessians):
        """Build the tree given binned features and gradient/hessian arrays."""
        n_samples = X_binned.shape[0]
        all_indices = np.arange(n_samples)

        root = TreeNode()
        root.sample_indices = all_indices
        root.depth = 0
        root.grad_sum = gradients.sum()
        root.hess_sum = hessians.sum()
        root.value = self._leaf_value(root.grad_sum, root.hess_sum)

        if self.growth == "symmetric":
            self._build_symmetric(root, X_binned, gradients, hessians)
        elif self.growth == "leafwise":
            self._build_leafwise(root, X_binned, gradients, hessians)
        elif self.growth == "hybrid":
            self._build_hybrid(root, X_binned, gradients, hessians)
        else:
            self._build_depthwise(root, X_binned, gradients, hessians)

        self.root = root
        return self

    def predict(self, X_binned):
        """Predict values for binned feature matrix."""
        predictions = np.empty(X_binned.shape[0], dtype=np.float64)
        self._predict_batch(self.root, X_binned, np.arange(X_binned.shape[0]), predictions)
        return predictions

    def to_arrays(self, col_indices=None):
        """Serialize tree to flat pre-order DFS arrays for C++ prediction.

        Parameters
        ----------
        col_indices : array-like or None
            If the tree was built on a column-subsampled matrix, supply the
            original column indices so that feature references are remapped to
            absolute positions in the full feature matrix.

        Returns
        -------
        features, thresholds, nan_dirs, values, lefts, rights
            Each a numpy array of shape (n_nodes,).
        """
        features   = []
        thresholds = []
        nan_dirs   = []
        values     = []
        lefts      = []
        rights     = []

        def dfs(node):
            idx = len(features)
            features  .append(node.feature)
            thresholds.append(node.threshold)
            nan_dirs  .append(node.nan_direction)
            values    .append(node.value)
            lefts     .append(-1)
            rights    .append(-1)

            if node.left is not None:
                left_idx      = len(features)
                lefts[idx]    = left_idx
                dfs(node.left)
                right_idx     = len(features)
                rights[idx]   = right_idx
                dfs(node.right)

        if self.root is not None:
            dfs(self.root)

        feat_arr = np.array(features, dtype=np.int32)

        # Remap local feature indices → absolute column indices
        if col_indices is not None and len(feat_arr) > 0:
            ci = np.asarray(col_indices, dtype=np.int32)
            valid = feat_arr >= 0          # interior (non-leaf) nodes only
            feat_arr[valid] = ci[feat_arr[valid]]

        return (
            feat_arr,
            np.array(thresholds, dtype=np.int32),
            np.array(nan_dirs,   dtype=np.int32),
            np.array(values,     dtype=np.float64),
            np.array(lefts,      dtype=np.int32),
            np.array(rights,     dtype=np.int32),
        )

    def _predict_batch(self, node, X_binned, indices, out):
        """Recursively assign leaf values to a batch of samples at once."""
        if node.left is None:  # leaf node
            out[indices] = node.value
            return
        bins = X_binned[indices, node.feature]
        nan_mask = bins == self.max_bins
        if node.nan_direction == 0:
            left_mask = (bins <= node.threshold) | nan_mask
        else:
            left_mask = (bins <= node.threshold) & ~nan_mask
        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]
        if len(left_idx) > 0:
            self._predict_batch(node.left, X_binned, left_idx, out)
        if len(right_idx) > 0:
            self._predict_batch(node.right, X_binned, right_idx, out)

    # --- Growth strategies ---

    def _build_leafwise(self, root, X_binned, gradients, hessians):
        """Leaf-wise (best-first) growth: always split the leaf with max gain."""
        import heapq

        leaves = []
        n_leaves = 1

        split, root_hist = self._find_split_with_hist(root, X_binned, gradients, hessians)
        if split is not None:
            heapq.heappush(leaves, (-split[2], id(root), root, split, root_hist))

        while leaves and n_leaves < self.max_leaves:
            neg_gain, _, node, (feat, bin_thresh, gain, nan_dir), node_hist = \
                heapq.heappop(leaves)

            if node.depth >= self.max_depth:
                continue

            self._apply_split(node, feat, bin_thresh, gain, nan_dir,
                              X_binned, gradients, hessians)
            n_leaves += 1

            # Subtraction trick: one build + one subtract instead of two builds
            left_hist, right_hist = self._compute_children_hists(
                node, node_hist, X_binned, gradients, hessians)

            for child, child_hist in ((node.left, left_hist), (node.right, right_hist)):
                if child.depth < self.max_depth:
                    child_split, child_hist = self._find_split_with_hist(
                        child, X_binned, gradients, hessians, hist=child_hist)
                    if child_split is not None:
                        heapq.heappush(leaves, (-child_split[2], id(child),
                                                child, child_split, child_hist))

    def _build_symmetric(self, root, X_binned, gradients, hessians):
        """Symmetric growth: all nodes at same depth use the same split."""
        level_nodes = [root]

        for depth in range(self.max_depth):
            if not level_nodes:
                break

            combined_grad = None
            combined_hess = None
            combined_count = None

            for node in level_nodes:
                gh, hh, ch = self.hist_builder.build_histogram(
                    X_binned, gradients, hessians, node.sample_indices)
                if combined_grad is None:
                    combined_grad = gh
                    combined_hess = hh
                    combined_count = ch
                else:
                    combined_grad += gh
                    combined_hess += hh
                    combined_count += ch

            feat, bin_thresh, gain, nan_dir = self.hist_builder.find_best_split(
                combined_grad, combined_hess, combined_count,
                self.reg_lambda, self.reg_alpha,
                self.min_child_weight, self.min_child_samples,
                adaptive_reg=self.adaptive_reg,
                iteration=self.iteration,
                total_iterations=self.total_iterations,
                monotone_checker=self.monotone_checker)

            if feat < 0 or gain <= 0:
                break

            self.split_features_.append(feat)
            self.split_gains_.append(gain)

            next_level = []
            for node in level_nodes:
                self._apply_split(node, feat, bin_thresh, gain, nan_dir,
                                  X_binned, gradients, hessians)
                next_level.extend([node.left, node.right])

            level_nodes = next_level

    def _build_depthwise(self, root, X_binned, gradients, hessians):
        """Standard depth-wise growth."""
        root_split, root_hist = self._find_split_with_hist(
            root, X_binned, gradients, hessians)
        if root_split is None:
            return

        # Queue carries (node, precomputed_hist, precomputed_split)
        queue = [(root, root_hist, root_split)]

        while queue:
            next_queue = []
            for node, node_hist, split in queue:
                if node.depth >= self.max_depth:
                    continue
                feat, bin_thresh, gain, nan_dir = split
                self._apply_split(node, feat, bin_thresh, gain, nan_dir,
                                  X_binned, gradients, hessians)

                # Subtraction trick: one build + one subtract instead of two builds
                left_hist, right_hist = self._compute_children_hists(
                    node, node_hist, X_binned, gradients, hessians)

                for child, child_hist in ((node.left, left_hist), (node.right, right_hist)):
                    if child.depth < self.max_depth:
                        child_split, child_hist = self._find_split_with_hist(
                            child, X_binned, gradients, hessians, hist=child_hist)
                        if child_split is not None:
                            next_queue.append((child, child_hist, child_split))
            queue = next_queue

    def _build_hybrid(self, root, X_binned, gradients, hessians):
        """Hybrid growth: symmetric for levels 0..D_sym, then leaf-wise.

        Combines CatBoost's oblivious tree regularization with
        LightGBM's flexible leaf-wise splitting.
        """
        import heapq

        # Phase 1: Symmetric growth up to symmetric_depth
        level_nodes = [root]
        n_leaves = 1

        for depth in range(min(self.symmetric_depth, self.max_depth)):
            if not level_nodes:
                break

            combined_grad = None
            combined_hess = None
            combined_count = None

            for node in level_nodes:
                gh, hh, ch = self.hist_builder.build_histogram(
                    X_binned, gradients, hessians, node.sample_indices)
                if combined_grad is None:
                    combined_grad = gh
                    combined_hess = hh
                    combined_count = ch
                else:
                    combined_grad += gh
                    combined_hess += hh
                    combined_count += ch

            feat, bin_thresh, gain, nan_dir = self.hist_builder.find_best_split(
                combined_grad, combined_hess, combined_count,
                self.reg_lambda, self.reg_alpha,
                self.min_child_weight, self.min_child_samples,
                adaptive_reg=self.adaptive_reg,
                iteration=self.iteration,
                total_iterations=self.total_iterations,
                monotone_checker=self.monotone_checker)

            if feat < 0 or gain <= 0:
                break

            self.split_features_.append(feat)
            self.split_gains_.append(gain)

            next_level = []
            for node in level_nodes:
                self._apply_split(node, feat, bin_thresh, gain, nan_dir,
                                  X_binned, gradients, hessians)
                next_level.extend([node.left, node.right])
                n_leaves += 1  # net +1 per split

            level_nodes = next_level

        # Phase 2: Leaf-wise growth from symmetric leaves
        if level_nodes and n_leaves < self.max_leaves:
            leaf_heap = []
            for node in level_nodes:
                if node.depth < self.max_depth:
                    split, node_hist = self._find_split_with_hist(
                        node, X_binned, gradients, hessians)
                    if split is not None:
                        heapq.heappush(leaf_heap, (-split[2], id(node),
                                                   node, split, node_hist))

            while leaf_heap and n_leaves < self.max_leaves:
                neg_gain, _, node, (feat, bin_thresh, gain, nan_dir), node_hist = \
                    heapq.heappop(leaf_heap)

                if node.depth >= self.max_depth:
                    continue

                self._apply_split(node, feat, bin_thresh, gain, nan_dir,
                                  X_binned, gradients, hessians)
                n_leaves += 1

                # Subtraction trick: one build + one subtract instead of two builds
                left_hist, right_hist = self._compute_children_hists(
                    node, node_hist, X_binned, gradients, hessians)

                for child, child_hist in ((node.left, left_hist), (node.right, right_hist)):
                    if child.depth < self.max_depth:
                        child_split, child_hist = self._find_split_with_hist(
                            child, X_binned, gradients, hessians, hist=child_hist)
                        if child_split is not None:
                            heapq.heappush(leaf_heap, (-child_split[2], id(child),
                                                       child, child_split, child_hist))

    # --- Helpers ---

    def _find_split(self, node, X_binned, gradients, hessians):
        """Find best split for a node. Returns (feature, bin, gain, nan_dir) or None."""
        split, _ = self._find_split_with_hist(node, X_binned, gradients, hessians)
        return split

    def _find_split_with_hist(self, node, X_binned, gradients, hessians, hist=None):
        """Like _find_split but also returns the histogram that was used/built.

        Parameters
        ----------
        hist : tuple (grad_hist, hess_hist, count_hist) or None
            Pre-built histogram to reuse. Built from node.sample_indices if None.

        Returns
        -------
        split : (feature, bin, gain, nan_dir) or None
        hist  : tuple or None  — always returned so callers can pass it to
                _compute_children_hists even when no valid split is found.
        """
        if len(node.sample_indices) < 2 * self.min_child_samples:
            return None, hist

        if hist is None:
            hist = self.hist_builder.build_histogram(
                X_binned, gradients, hessians, node.sample_indices)

        grad_hist, hess_hist, count_hist = hist

        feat, bin_thresh, gain, nan_dir = self.hist_builder.find_best_split(
            grad_hist, hess_hist, count_hist,
            self.reg_lambda, self.reg_alpha,
            self.min_child_weight, self.min_child_samples,
            adaptive_reg=self.adaptive_reg,
            iteration=self.iteration,
            total_iterations=self.total_iterations,
            monotone_checker=self.monotone_checker)

        if feat < 0 or gain <= 0:
            return None, hist

        return (feat, bin_thresh, gain, nan_dir), hist

    def _compute_children_hists(self, node, parent_hist, X_binned, gradients, hessians):
        """Compute both children's histograms using the subtraction trick.

        Builds the histogram for the smaller child from scratch, then derives
        the larger child's histogram as parent_hist - smaller_child_hist.
        This saves one full histogram build per split.

        Returns
        -------
        left_hist, right_hist : tuples of (grad_hist, hess_hist, count_hist)
        """
        left, right = node.left, node.right
        if len(left.sample_indices) <= len(right.sample_indices):
            left_hist = self.hist_builder.build_histogram(
                X_binned, gradients, hessians, left.sample_indices)
            right_hist = self.hist_builder.subtract_histograms(parent_hist, left_hist)
        else:
            right_hist = self.hist_builder.build_histogram(
                X_binned, gradients, hessians, right.sample_indices)
            left_hist = self.hist_builder.subtract_histograms(parent_hist, right_hist)
        return left_hist, right_hist

    def _apply_split(self, node, feat, bin_thresh, gain, nan_dir,
                     X_binned, gradients, hessians):
        """Apply a split to a node, creating left and right children."""
        node.feature = feat
        node.threshold = bin_thresh
        node.gain = gain
        node.nan_direction = nan_dir

        self.split_features_.append(feat)
        self.split_gains_.append(gain)

        indices = node.sample_indices
        bins = X_binned[indices, feat]

        left_mask = bins <= bin_thresh

        nan_mask = bins == self.max_bins
        if nan_dir == 0:
            left_mask = left_mask | nan_mask
        else:
            left_mask = left_mask & ~nan_mask

        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]

        node.left = TreeNode()
        node.left.sample_indices = left_idx
        node.left.depth = node.depth + 1
        node.left.grad_sum = gradients[left_idx].sum()
        node.left.hess_sum = hessians[left_idx].sum()
        node.left.value = self._leaf_value(node.left.grad_sum, node.left.hess_sum)

        node.right = TreeNode()
        node.right.sample_indices = right_idx
        node.right.depth = node.depth + 1
        node.right.grad_sum = gradients[right_idx].sum()
        node.right.hess_sum = hessians[right_idx].sum()
        node.right.value = self._leaf_value(node.right.grad_sum, node.right.hess_sum)

    def _leaf_value(self, grad_sum, hess_sum):
        """Compute optimal leaf value with regularization."""
        if abs(grad_sum) <= self.reg_alpha:
            return 0.0
        return -(grad_sum - np.sign(grad_sum) * self.reg_alpha) / (hess_sum + self.reg_lambda)

    def _predict_one(self, node, x):
        """Traverse tree for a single sample."""
        if node.left is None:
            return node.value
        bin_val = x[node.feature]
        if bin_val == self.max_bins:  # NaN
            if node.nan_direction == 0:
                return self._predict_one(node.left, x)
            else:
                return self._predict_one(node.right, x)
        if bin_val <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)
