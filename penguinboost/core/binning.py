"""Feature binning and Exclusive Feature Bundling (EFB) from LightGBM."""

import numpy as np


class FeatureBinner:
    """Equal-frequency binning of continuous features (max 255 bins).

    Also implements Exclusive Feature Bundling (EFB) to reduce the number
    of features by bundling mutually exclusive sparse features.
    """

    def __init__(self, max_bins=255, efb_threshold=0.0):
        self.max_bins = max_bins
        self.efb_threshold = efb_threshold  # EFB の競合率閾値
        self.bin_edges_ = []
        self.bundles_ = None  # list of lists of feature indices
        self.n_features_original_ = 0

    def fit(self, X):
        """Compute bin edges for each feature and optionally bundle features."""
        self.n_features_original_ = X.shape[1]

        # 特徴量ごとのビンエッジを計算
        self.bin_edges_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                self.bin_edges_.append(np.array([]))
                continue
            # 等頻度分位数エッジ
            n_bins = min(self.max_bins, len(np.unique(valid)))
            quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
            edges = np.unique(np.percentile(valid, quantiles))
            self.bin_edges_.append(edges)

        # EFB: 相互排他的特徴量をバンドル
        if self.efb_threshold > 0:
            self.bundles_ = self._find_bundles(X)
        else:
            self.bundles_ = [[i] for i in range(X.shape[1])]

        return self

    def transform(self, X):
        """Bin features and apply bundling. Returns integer bin indices.

        NaN is mapped to max_bins (a special bin).
        """
        n_samples = X.shape[0]
        n_bundles = len(self.bundles_)
        X_binned = np.zeros((n_samples, n_bundles), dtype=np.uint8)

        for bundle_idx, bundle in enumerate(self.bundles_):
            if len(bundle) == 1:
                feat = bundle[0]
                X_binned[:, bundle_idx] = self._bin_feature(X[:, feat], feat)
            else:
                # バンドル特徴量をマージ：各サブ特徴量のビンをオフセット
                X_binned[:, bundle_idx] = self._bin_bundle(X, bundle)

        return X_binned

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _bin_feature(self, col, feat_idx):
        """Digitize a single feature column into bin indices."""
        edges = self.bin_edges_[feat_idx]
        nan_mask = np.isnan(col)
        binned = np.digitize(col, edges).astype(np.uint8)
        binned[nan_mask] = self.max_bins  # special NaN bin
        return binned

    def _bin_bundle(self, X, bundle):
        """Merge multiple mutually exclusive features into one binned column."""
        n_samples = X.shape[0]
        result = np.zeros(n_samples, dtype=np.uint8)
        offset = 0
        for feat in bundle:
            col = X[:, feat]
            edges = self.bin_edges_[feat]
            n_bins = len(edges) + 1
            non_zero = ~np.isnan(col) & (col != 0)
            binned = np.digitize(col[non_zero], edges).astype(np.uint8)
            result[non_zero] = binned + offset
            offset += n_bins
        return result

    def _find_bundles(self, X):
        """Greedy EFB: find groups of mutually exclusive features."""
        n_features = X.shape[1]
        # 各特徴量の非ゼロ数を計算
        nonzero = np.array([(~np.isnan(X[:, j]) & (X[:, j] != 0)).sum()
                            for j in range(n_features)])

        # 非ゼロ数の降順で特徴量をソート
        order = np.argsort(-nonzero)
        bundles = []
        assigned = set()

        for i in order:
            if i in assigned:
                continue
            bundle = [i]
            assigned.add(i)
            nz_i = ~np.isnan(X[:, i]) & (X[:, i] != 0)
            for j in order:
                if j in assigned:
                    continue
                nz_j = ~np.isnan(X[:, j]) & (X[:, j] != 0)
                conflict = (nz_i & nz_j).sum()
                conflict_ratio = conflict / max(nz_i.sum(), 1)
                if conflict_ratio <= self.efb_threshold:
                    # バンドル合計ビン数が 255 を超えないかチェック
                    total_bins = sum(len(self.bin_edges_[f]) + 1 for f in bundle)
                    total_bins += len(self.bin_edges_[j]) + 1
                    if total_bins <= self.max_bins:
                        bundle.append(j)
                        assigned.add(j)
                        nz_i = nz_i | nz_j
            bundles.append(bundle)

        return bundles

    @property
    def n_features_bundled(self):
        return len(self.bundles_) if self.bundles_ else self.n_features_original_
