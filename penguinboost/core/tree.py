"""Leaf-wise（LightGBM）、Symmetric（CatBoost）、Hybrid 成長戦略を持つ決定木。"""

import numpy as np
from penguinboost.core.histogram import HistogramBuilder


class TreeNode:
    """決定木のノード。"""

    __slots__ = [
        "feature", "threshold", "nan_direction",
        "left", "right", "value", "depth", "gain",
        "sample_indices", "grad_sum", "hess_sum",
    ]

    def __init__(self):
        self.feature = -1       # 分割特徴量インデックス
        self.threshold = 0      # 分割ビン閾値
        self.nan_direction = 0  # 0=左, 1=右
        self.left = None
        self.right = None
        self.value = 0.0        # 葉ノードの予測値
        self.depth = 0
        self.gain = 0.0
        self.sample_indices = None
        self.grad_sum = 0.0
        self.hess_sum = 0.0


class DecisionTree:
    """複数の成長戦略をサポートするヒストグラムベースの決定木。

    Parameters
    ----------
    max_depth : int
        ツリーの最大深さ。
    max_leaves : int
        最大葉数（Leaf-wise 成長で使用）。
    growth : str
        'leafwise', 'symmetric', 'depthwise', 'hybrid'。
    reg_lambda : float
        葉の重みに対する L2 正則化。
    reg_alpha : float
        葉の重みに対する L1 正則化。
    min_child_weight : float
        葉ノードの最小ヘッセ行列和。
    min_child_samples : int
        葉ノードの最小サンプル数。
    max_bins : int
        ヒストグラムの最大ビン数。
    symmetric_depth : int
        ハイブリッド成長で対称から Leaf-wise へ切り替える深さ。
    adaptive_reg : AdaptiveRegularizer or None
        適応的正則化器。
    monotone_checker : MonotoneConstraintChecker or None
        単調制約チェッカー。
    iteration : int
        現在のブースティングイテレーション（適応的正則化用）。
    total_iterations : int
        計画されたブースティングイテレーションの総数。
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
        """ビニング済み特徴量と勾配・ヘッセ行列からツリーを構築する。"""
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
        """ビニング済み特徴量行列の予測値を返す。"""
        predictions = np.empty(X_binned.shape[0], dtype=np.float64)
        self._predict_batch(self.root, X_binned, np.arange(X_binned.shape[0]), predictions)
        return predictions

    def to_arrays(self, col_indices=None):
        """C++ 予測用に DFS 前順の平坦配列へシリアライズする。

        Parameters
        ----------
        col_indices : array-like or None
            列サブサンプリングで構築された場合、元の列インデックスを指定して
            特徴量参照を絶対位置へ変換する。

        Returns
        -------
        features, thresholds, nan_dirs, values, lefts, rights
            各要素は (n_nodes,) の numpy 配列。
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

        # ローカル特徴量インデックスを絶対列インデックスへ変換
        if col_indices is not None and len(feat_arr) > 0:
            ci = np.asarray(col_indices, dtype=np.int32)
            valid = feat_arr >= 0          # 内部（非葉）ノードのみ
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
        """サンプルのバッチに葉の値を再帰的に割り当てる。"""
        if node.left is None:  # 葉ノード
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

    # --- 成長戦略 ---

    def _build_leafwise(self, root, X_binned, gradients, hessians):
        """Leaf-wise（ベストファースト）成長：常に最大ゲインの葉を分割する。"""
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

            # 差分トリック：2 回のビルドの代わりに 1 回のビルドと 1 回の減算
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
        """対称成長：同じ深さの全ノードが同一の分割を使用する。"""
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
        """標準の深さ優先成長。"""
        root_split, root_hist = self._find_split_with_hist(
            root, X_binned, gradients, hessians)
        if root_split is None:
            return

        # キューに (ノード, ヒストグラム, 分割情報) を格納
        queue = [(root, root_hist, root_split)]

        while queue:
            next_queue = []
            for node, node_hist, split in queue:
                if node.depth >= self.max_depth:
                    continue
                feat, bin_thresh, gain, nan_dir = split
                self._apply_split(node, feat, bin_thresh, gain, nan_dir,
                                  X_binned, gradients, hessians)

                # 差分トリック：2 回のビルドの代わりに 1 回のビルドと 1 回の減算
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
        """ハイブリッド成長：レベル 0..D_sym は対称成長、その後 Leaf-wise 成長。

        CatBoost の oblibious tree 正則化と LightGBM の柔軟な Leaf-wise
        分割を組み合わせる。
        """
        import heapq

        # フェーズ 1: symmetric_depth までの対称成長
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
                n_leaves += 1  # 1 分割あたり純 +1

            level_nodes = next_level

        # フェーズ 2: 対称葉からの Leaf-wise 成長
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

                # 差分トリック：2 回のビルドの代わりに 1 回のビルドと 1 回の減算
                left_hist, right_hist = self._compute_children_hists(
                    node, node_hist, X_binned, gradients, hessians)

                for child, child_hist in ((node.left, left_hist), (node.right, right_hist)):
                    if child.depth < self.max_depth:
                        child_split, child_hist = self._find_split_with_hist(
                            child, X_binned, gradients, hessians, hist=child_hist)
                        if child_split is not None:
                            heapq.heappush(leaf_heap, (-child_split[2], id(child),
                                                       child, child_split, child_hist))

    # --- ヘルパー ---

    def _find_split(self, node, X_binned, gradients, hessians):
        """ノードの最適分割を探す。(feature, bin, gain, nan_dir) または None を返す。"""
        split, _ = self._find_split_with_hist(node, X_binned, gradients, hessians)
        return split

    def _find_split_with_hist(self, node, X_binned, gradients, hessians, hist=None):
        """_find_split と同様だが、使用・構築したヒストグラムも返す。

        Parameters
        ----------
        hist : tuple (grad_hist, hess_hist, count_hist) or None
            再利用する事前構築済みヒストグラム。None の場合は node.sample_indices から構築する。

        Returns
        -------
        split : (feature, bin, gain, nan_dir) or None
        hist  : tuple or None — 有効な分割が見つからない場合でも常に返される。
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
        """差分トリックで両子ノードのヒストグラムを計算する。

        小さい方の子のヒストグラムをゼロから構築し、大きい方を
        parent_hist - 小さい子のヒストグラム として導出する。
        これにより分割ごとのヒストグラム構築を 1 回節約できる。

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
        """ノードに分割を適用し、左右の子ノードを作成する。"""
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
        """正則化付きの最適葉値を計算する。"""
        if abs(grad_sum) <= self.reg_alpha:
            return 0.0
        return -(grad_sum - np.sign(grad_sum) * self.reg_alpha) / (hess_sum + self.reg_lambda)

    def _predict_one(self, node, x):
        """単一サンプルのツリー探索。"""
        if node.left is None:
            return node.value
        bin_val = x[node.feature]
        if bin_val == self.max_bins:  # NaN 値
            if node.nan_direction == 0:
                return self._predict_one(node.left, x)
            else:
                return self._predict_one(node.right, x)
        if bin_val <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)
