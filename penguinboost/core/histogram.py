"""ヒストグラムベースの分割探索（差分トリック / LightGBM スタイル）。

C++ アクセラレーション: build_histogram と find_best_split_basic は _core で高速化される。
"""

import numpy as np

try:
    from penguinboost import _core as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


class HistogramBuilder:
    """ビニング済み特徴量の勾配・ヘッセ行列ヒストグラムを構築し、最良分割を探す。"""

    def __init__(self, max_bins=255):
        self.max_bins = max_bins

    def build_histogram(self, X_binned, gradients, hessians, sample_indices=None):
        """全特徴量の勾配・ヘッセ行列ヒストグラムを構築する。

        利用可能な場合は C++ のシングルパス実装を使用し、
        それ以外は numpy の bincount にフォールバックする。

        Returns:
            grad_hist:  (n_features, max_bins+1) ビンごとの勾配和
            hess_hist:  (n_features, max_bins+1) ビンごとのヘッセ行列和
            count_hist: (n_features, max_bins+1) ビンごとのサンプル数
        """
        if _HAS_CPP:
            return _cpp.build_histogram(
                X_binned, gradients, hessians,
                sample_indices, self.max_bins)

        # Pure-Python フォールバック
        if sample_indices is not None:
            X_sub = X_binned[sample_indices]
            g_sub = gradients[sample_indices]
            h_sub = hessians[sample_indices]
        else:
            X_sub = X_binned
            g_sub = gradients
            h_sub = hessians

        n_features = X_sub.shape[1]
        n_bins = self.max_bins + 1  # NaN ビン用に +1

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
        """差分トリック: child = parent - sibling（再構築を回避）。"""
        grad_p, hess_p, count_p = parent_hist
        grad_s, hess_s, count_s = sibling_hist
        return (grad_p - grad_s, hess_p - hess_s, count_p - count_s)

    def find_best_split(self, grad_hist, hess_hist, count_hist,
                        reg_lambda=1.0, reg_alpha=0.0, min_child_weight=1.0,
                        min_child_samples=1, adaptive_reg=None, iteration=0,
                        total_iterations=1, stability_tracker=None,
                        monotone_checker=None):
        """全特徴量・全ビンにわたって最良分割を探す。

        特徴量とビンに対して完全ベクトル化されており、Python レベルのループを避ける。

        Returns:
            best_feature, best_bin, best_gain, best_nan_direction
        """
        # C++ 高速パス（適応的正則化・単調制約なし時）
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

        # 有効ビンと NaN ビンに分割
        g_valid = grad_hist[:, :mb]      # (n_features, max_bins)
        h_valid = hess_hist[:, :mb]
        c_valid = count_hist[:, :mb]

        g_nan = grad_hist[:, mb]         # (n_features,)
        h_nan = hess_hist[:, mb]
        c_nan = count_hist[:, mb]

        g_total = g_valid.sum(axis=1)    # (n_features,)
        h_total = h_valid.sum(axis=1)
        c_total = c_valid.sum(axis=1)

        g_sum = g_total + g_nan          # (n_features,)
        h_sum = h_total + h_nan
        c_sum = c_total + c_nan

        # ビン方向の前置累積和
        # 形状: (n_features, max_bins)、要素 [j, b] = sum(hist[j, 0..b])
        g_cumL = np.cumsum(g_valid, axis=1)
        h_cumL = np.cumsum(h_valid, axis=1)
        c_cumL = np.cumsum(c_valid, axis=1)

        # ブロードキャスト用シングルトン次元
        g_nan_e = g_nan[:, np.newaxis]          # (n_features, 1)
        h_nan_e = h_nan[:, np.newaxis]
        c_nan_e = c_nan[:, np.newaxis]

        # 両 nan_dir の左子ノード統計を同時構築
        # スタック形状: (2, n_features, max_bins)
        #   axis-0 インデックス 0 → nan_dir=0（NaN を左へ）
        #   axis-0 インデックス 1 → nan_dir=1（NaN を右へ）
        g_L = np.stack([g_cumL + g_nan_e, g_cumL])
        h_L = np.stack([h_cumL + h_nan_e, h_cumL])
        c_L = np.stack([c_cumL + c_nan_e, c_cumL])

        # 補集合から右子ノード統計を計算
        g_sum_e = g_sum[np.newaxis, :, np.newaxis]   # (1, n_features, 1)
        h_sum_e = h_sum[np.newaxis, :, np.newaxis]
        c_sum_e = c_sum[np.newaxis, :, np.newaxis]

        g_R = g_sum_e - g_L              # (2, n_features, max_bins)
        h_R = h_sum_e - h_L
        c_R = c_sum_e - c_L

        # 有効スプリットマスク
        valid = ((h_L >= min_child_weight) & (h_R >= min_child_weight) &
                 (c_L >= min_child_samples) & (c_R >= min_child_samples))

        # 有効な分割がなければ早期終了
        if not valid.any():
            return -1, -1, 0.0, 0

        # 分割ゲインを計算（完全ベクトル化）
        # 親スコア (n_features,) → (1, n_features, 1)
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

        # 単調制約フィルター（制約特徴量ごと）
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

        # 無効なスプリットをマスクしてグローバル最適を探す
        gains = np.where(valid, gains, 0.0)
        flat_idx = int(np.argmax(gains))
        max_gain = float(gains.ravel()[flat_idx])

        if max_gain <= 0.0:
            return -1, -1, 0.0, 0

        best_nan_dir, best_feature, best_bin = np.unravel_index(flat_idx, gains.shape)

        if stability_tracker is not None:
            pass

        return int(best_feature), int(best_bin), max_gain, int(best_nan_dir)

    # --- ベクトル化ヘルパー（形状非依存、任意の ndarray 形状で動作）---

    @staticmethod
    def _score_vec(g, h, lam, reg_alpha):
        """ベクトル化ノードスコア。任意形状の numpy 配列で動作する。"""
        if reg_alpha == 0.0:
            # 高速パス：L1 正則化なし（一般的なデフォルト）
            return g * g / (h + lam)
        abs_g = np.abs(g)
        g_adj = g - np.sign(g) * reg_alpha
        return np.where(abs_g > reg_alpha, g_adj * g_adj / (h + lam), 0.0)

    @staticmethod
    def _leaf_value_vec(g, h, reg_lambda, reg_alpha):
        """単調制約チェック用のベクトル化葉値。"""
        if reg_alpha == 0.0:
            return -g / (h + reg_lambda)
        abs_g = np.abs(g)
        g_adj = g - np.sign(g) * reg_alpha
        return np.where(abs_g > reg_alpha, -g_adj / (h + reg_lambda), 0.0)

    # --- スカラーメソッド（後方互換性）---

    def _compute_gain(self, g_left, h_left, g_right, h_right,
                      g_total, h_total, reg_lambda, reg_alpha):
        """L1/L2 正則化付きの分割ゲインを計算する。"""
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
        """子ノードごとの適応的ラムダを使った分割ゲインを計算する。

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
        """単調制約チェック用の葉値を計算する。"""
        if abs(g) <= reg_alpha:
            return 0.0
        return -(g - np.sign(g) * reg_alpha) / (h + reg_lambda)
