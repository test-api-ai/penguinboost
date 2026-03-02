"""勾配ブースティングエンジン。

DART、勾配摂動、複数置換順序付きブースティング、時系列正則化、
適応的正則化、ハイブリッド成長、単調制約を含む。

C++ アクセラレーション: predict() は利用可能な場合 _core.predict_trees を使用する。
"""

import numpy as np

try:
    from penguinboost import _core as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
from penguinboost.core.tree import DecisionTree
from penguinboost.core.sampling import GOSSSampler, TemporallyWeightedGOSSSampler
from penguinboost.core.binning import FeatureBinner
from penguinboost.core.categorical import OrderedTargetEncoder
from penguinboost.core.regularization import (
    AdaptiveRegularizer, GradientPerturber, EraAdaptiveGradientClipper)
from penguinboost.core.dart import DARTManager, EraAwareDARTManager
from penguinboost.core.monotone import MonotoneConstraintChecker
from penguinboost.core.financial import TemporalRegularizer


def _override_leaf_values(node, residuals, objective):
    """Walk the tree and replace each leaf's value with the objective-optimal estimate.

    Called when the objective provides ``compute_leaf_value(residuals)``.
    This is required for non-smooth losses (MAE, Huber) where the standard
    Newton step  −G/(H + λ)  gives systematically biased leaf values due to
    the constant or near-zero hessian approximation.

    Parameters
    ----------
    node : TreeNode
        Root (or any subtree root) of the decision tree.
    residuals : np.ndarray, shape (n_samples,)
        Per-sample residuals y − current_predictions (full dataset).
    objective : object
        Must implement ``compute_leaf_value(residuals_in_leaf) → float``.
    """
    if node.left is None:  # leaf node
        leaf_res = residuals[node.sample_indices]
        if len(leaf_res) > 0:
            node.value = objective.compute_leaf_value(leaf_res)
        return
    _override_leaf_values(node.left,  residuals, objective)
    _override_leaf_values(node.right, residuals, objective)


def _era_to_int(era_indices):
    """Convert arbitrary era labels to consecutive integers 0..n_eras-1."""
    labels = np.unique(era_indices)
    label_map = {lab: i for i, lab in enumerate(labels)}
    return np.array([label_map[e] for e in era_indices], dtype=np.int32)


class BoostingEngine:
    """LightGBM・CatBoost・XGBoost の技術を組み合わせたコア勾配ブースティングエンジン。

    Parameters
    ----------
    n_estimators : int
        ブースティングラウンド数。
    learning_rate : float
        シュリンケージ率。
    max_depth : int
        ツリーの最大深さ。
    max_leaves : int
        ツリーあたりの最大葉数。
    growth : str
        ツリー成長戦略: 'leafwise', 'symmetric', 'depthwise', 'hybrid'。
    reg_lambda : float
        L2 正則化。
    reg_alpha : float
        L1 正則化。
    min_child_weight : float
        葉ノードの最小ヘッセ行列和。
    min_child_samples : int
        葉ノードの最小サンプル数。
    subsample : float
        行サブサンプリング率（1.0 = サブサンプリングなし）。
    colsample_bytree : float
        ツリーごとの列サブサンプリング率。
    max_bins : int
        ヒストグラムの最大ビン数。
    use_goss : bool
        GOSS サンプリングを使用するか。
    goss_top_rate : float
        GOSS の上位勾配割合。
    goss_other_rate : float
        GOSS のその他サンプル割合。
    use_ordered_boosting : bool
        CatBoost スタイルの順序付きブースティングを使用するか。
    n_permutations : int
        複数置換順序付きブースティングの置換数。
    cat_features : list or None
        カテゴリ特徴量のインデックス。
    cat_smoothing : float
        順序付きターゲット統計のスムージング。
    efb_threshold : float
        EFB の競合閾値（0 = 無効）。
    early_stopping_rounds : int or None
        この回数改善がなければ停止。
    use_dart : bool
        DART（Dropout Trees）正則化を有効化するか。
    dart_drop_rate : float
        DART: 各ツリーをドロップする確率。
    dart_skip_drop : float
        DART: ドロップアウトをスキップする確率。
    use_gradient_perturbation : bool
        勾配摂動（クリッピング＋ノイズ）を有効化するか。
    gradient_clip_tau : float
        勾配クリッピング閾値。
    gradient_noise_eta : float
        勾配ノイズスケール（標準偏差に対する相対値）。
    use_adaptive_reg : bool
        ベイズ適応的正則化を有効化するか。
    adaptive_alpha : float
        適応的正則化のスケジュールスケーリング。
    adaptive_mu : float
        ベイズ子ノードペナルティ係数。
    monotone_constraints : dict or None
        特徴量インデックス -> 単調制約方向（+1/-1）。
    use_temporal_reg : bool
        時系列正則化（金融向け）を有効化するか。
    temporal_rho : float
        時系列スムーズネスペナルティの強度。
    symmetric_depth : int
        ハイブリッド成長で対称からLeaf-wiseへ切り替える深さ。
    verbose : int
        詳細出力レベル。
    random_state : int or None
        乱数シード。
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 max_leaves=31, growth="leafwise", reg_lambda=1.0, reg_alpha=0.0,
                 min_child_weight=1.0, min_child_samples=1, subsample=1.0,
                 colsample_bytree=1.0, max_bins=255, use_goss=False,
                 goss_top_rate=0.2, goss_other_rate=0.1,
                 use_ordered_boosting=False, n_permutations=4,
                 cat_features=None, cat_smoothing=10.0, efb_threshold=0.0,
                 early_stopping_rounds=None,
                 use_dart=False, dart_drop_rate=0.1, dart_skip_drop=0.0,
                 use_gradient_perturbation=False,
                 gradient_clip_tau=5.0, gradient_noise_eta=0.1,
                 use_adaptive_reg=False, adaptive_alpha=0.5, adaptive_mu=1.0,
                 monotone_constraints=None,
                 use_temporal_reg=False, temporal_rho=0.1,
                 symmetric_depth=3,
                 use_orthogonal_gradients=False,
                 orthogonal_strength=1.0, orthogonal_eps=1e-4,
                 orthogonal_features=None,
                 use_era_boosting=False,
                 era_boosting_method='hard_era', era_boosting_temp=1.0,
                 use_feature_exposure_penalty=False,
                 feature_exposure_lambda=0.1, exposure_penalty_features=None,
                 # ── New financial-domain features ──────────────────────────
                 use_tw_goss=False, tw_goss_decay=0.01,
                 use_era_gradient_clipping=False, era_clip_multiplier=4.0,
                 use_era_aware_dart=False, era_dart_var_scale=20.0,
                 use_sharpe_early_stopping=False,
                 sharpe_es_patience=50,
                 use_sharpe_tree_reg=False, sharpe_reg_threshold=0.5,
                 use_era_adversarial_split=False, era_adversarial_beta=0.3,
                 verbose=0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.growth = growth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bins = max_bins
        self.use_goss = use_goss
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.use_ordered_boosting = use_ordered_boosting
        self.n_permutations = n_permutations
        self.cat_features = cat_features
        self.cat_smoothing = cat_smoothing
        self.efb_threshold = efb_threshold
        self.early_stopping_rounds = early_stopping_rounds
        self.use_dart = use_dart
        self.dart_drop_rate = dart_drop_rate
        self.dart_skip_drop = dart_skip_drop
        self.use_gradient_perturbation = use_gradient_perturbation
        self.gradient_clip_tau = gradient_clip_tau
        self.gradient_noise_eta = gradient_noise_eta
        self.use_adaptive_reg = use_adaptive_reg
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_mu = adaptive_mu
        self.monotone_constraints = monotone_constraints
        self.use_temporal_reg = use_temporal_reg
        self.temporal_rho = temporal_rho
        self.symmetric_depth = symmetric_depth
        self.use_orthogonal_gradients = use_orthogonal_gradients
        self.orthogonal_strength = orthogonal_strength
        self.orthogonal_eps = orthogonal_eps
        self.orthogonal_features = orthogonal_features
        self.use_era_boosting = use_era_boosting
        self.era_boosting_method = era_boosting_method
        self.era_boosting_temp = era_boosting_temp
        self.use_feature_exposure_penalty = use_feature_exposure_penalty
        self.feature_exposure_lambda = feature_exposure_lambda
        self.exposure_penalty_features = exposure_penalty_features
        self.use_tw_goss = use_tw_goss
        self.tw_goss_decay = tw_goss_decay
        self.use_era_gradient_clipping = use_era_gradient_clipping
        self.era_clip_multiplier = era_clip_multiplier
        self.use_era_aware_dart = use_era_aware_dart
        self.era_dart_var_scale = era_dart_var_scale
        self.use_sharpe_early_stopping = use_sharpe_early_stopping
        self.sharpe_es_patience = sharpe_es_patience
        self.use_sharpe_tree_reg = use_sharpe_tree_reg
        self.sharpe_reg_threshold = sharpe_reg_threshold
        self.use_era_adversarial_split = use_era_adversarial_split
        self.era_adversarial_beta = era_adversarial_beta
        self.verbose = verbose
        self.random_state = random_state

        # 学習済み属性
        self.trees_ = []
        self.tree_lr_ = []          # DART 用ツリーごとの有効学習率
        self.binner_ = None
        self.cat_encoder_ = None
        self.base_score_ = 0.0
        self.best_iteration_ = 0
        self.train_losses_ = []
        self._cpp_trees = None      # C++ 予測用にシリアライズされた平坦配列

    def fit(self, X, y, objective, eval_set=None, eval_metric=None,
            era_indices=None):
        """ブースティングモデルを学習する。

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,) or (n_samples, 2) for survival
        objective : object with gradient(y, pred) and hessian(y, pred) methods
        eval_set : tuple (X_val, y_val) or None
        eval_metric : callable(y_true, y_pred) -> float, or None
        era_indices : array-like of shape (n_samples,) or None
            エラ（時間期間）ラベル。エラブースティングとエラ条件付き目的関数を有効化する。
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # カテゴリ変数エンコーディング
        self.cat_encoder_ = OrderedTargetEncoder(
            smoothing=self.cat_smoothing, random_state=self.random_state)
        y_for_encode = y[:, 0] if y.ndim == 2 else y
        self.cat_encoder_.fit(X, y_for_encode, cat_features=self.cat_features)
        X_encoded = self.cat_encoder_.transform_train(X, y_for_encode)

        # ビニングと EFB
        self.binner_ = FeatureBinner(max_bins=self.max_bins,
                                     efb_threshold=self.efb_threshold)
        X_binned = self.binner_.fit_transform(X_encoded)

        # 検証セットのビニング
        X_val_binned = None
        y_val = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_encoded = self.cat_encoder_.transform(X_val)
            X_val_binned = self.binner_.transform(X_val_encoded)

        # 予測値を初期化
        self.base_score_ = objective.init_score(y)
        predictions = np.full(n_samples, self.base_score_, dtype=np.float64)

        if X_val_binned is not None:
            val_predictions = np.full(len(y_val), self.base_score_, dtype=np.float64)

        # GOSS / TW-GOSS サンプラー (互いに排他)
        goss = None
        tw_goss = None
        if self.use_tw_goss and era_indices is not None:
            tw_goss = TemporallyWeightedGOSSSampler(
                top_rate=self.goss_top_rate,
                other_rate=self.goss_other_rate,
                temporal_decay=self.tw_goss_decay,
                random_state=self.random_state)
        elif self.use_goss:
            goss = GOSSSampler(self.goss_top_rate, self.goss_other_rate)

        # 適応的正則化器
        adaptive_reg = None
        if self.use_adaptive_reg:
            adaptive_reg = AdaptiveRegularizer(
                lambda_base=self.reg_lambda,
                alpha=self.adaptive_alpha,
                mu=self.adaptive_mu)

        # 勾配摂動器
        perturber = None
        if self.use_gradient_perturbation:
            perturber = GradientPerturber(
                tau=self.gradient_clip_tau,
                eta=self.gradient_noise_eta)

        # DART / Era-aware DART マネージャー
        dart_mgr = None
        dart_tree_data = []  # (ツリー, 列インデックス, X_binned 参照)
        if self.use_era_aware_dart and era_indices is not None:
            dart_mgr = EraAwareDARTManager(
                drop_rate=self.dart_drop_rate,
                skip_drop=self.dart_skip_drop,
                era_var_scale=self.era_dart_var_scale)
        elif self.use_dart:
            dart_mgr = DARTManager(
                drop_rate=self.dart_drop_rate,
                skip_drop=self.dart_skip_drop)

        # 単調制約チェッカー
        monotone_checker = None
        if self.monotone_constraints:
            monotone_checker = MonotoneConstraintChecker(self.monotone_constraints)

        # 時系列正則化器
        temporal_reg = None
        if self.use_temporal_reg:
            temporal_reg = TemporalRegularizer(rho=self.temporal_rho)

        # 直交勾配射影器
        orth_projector = None
        if self.use_orthogonal_gradients:
            from penguinboost.core.neutralization import OrthogonalGradientProjector
            orth_projector = OrthogonalGradientProjector(
                strength=self.orthogonal_strength,
                eps=self.orthogonal_eps,
                features=self.orthogonal_features)
            orth_projector.fit(X_encoded)   # (X^TX + εI)^{-1} を事前計算

        # エラブースティング再重み付け器
        era_reweighter = None
        era_indices_arr = None
        y_1d = y[:, 0] if y.ndim == 2 else y   # エラメトリクス用の 1 次元ターゲット
        if self.use_era_boosting and era_indices is not None:
            from penguinboost.core.era_boost import EraBoostingReweighter
            era_reweighter = EraBoostingReweighter(
                method=self.era_boosting_method,
                temperature=self.era_boosting_temp)
            era_indices_arr = np.asarray(era_indices)

        # フィーチャーエクスポージャーペナルティ（センタリングと標準偏差を事前計算）
        exposure_penalty_data = None
        if self.use_feature_exposure_penalty:
            feats = self.exposure_penalty_features
            X_ep = X_encoded[:, feats] if feats is not None else X_encoded
            exposure_penalty_data = {
                'X_centered': X_ep - X_ep.mean(axis=0),
                'X_std':      X_ep.std(axis=0) + 1e-9,
                'lambda':     self.feature_exposure_lambda,
            }

        # Era adaptive gradient clipper
        era_clipper = None
        if self.use_era_gradient_clipping:
            era_clipper = EraAdaptiveGradientClipper(
                clip_multiplier=self.era_clip_multiplier)

        # Sharpe-ratio early stopping state
        best_sr = float('-inf')
        sr_patience_counter = 0

        # Sharpe-ratio tree regularization helper
        def _era_spearman_sharpe(tree_pred, targets, eras):
            """Compute Sharpe ratio of per-era Spearman correlations."""
            from penguinboost.core.era_boost import _spearman_corr
            labels = np.unique(eras)
            corrs = []
            for era in labels:
                mask = eras == era
                if mask.sum() < 2:
                    continue
                corrs.append(_spearman_corr(tree_pred[mask], targets[mask]))
            if len(corrs) < 2:
                return 0.0
            arr = np.array(corrs)
            return float(arr.mean() / (arr.std() + 1e-9))

        # Era-adversarial split: precompute integer era IDs (once, before loop)
        _era_int_ids = None
        _n_eras = 0
        if self.use_era_adversarial_split and era_indices_arr is not None:
            _era_int_ids = _era_to_int(era_indices_arr)
            _n_eras = int(_era_int_ids.max()) + 1

        # エラ条件付き目的関数に era_indices を設定
        if era_indices is not None and hasattr(objective, 'set_era_indices'):
            objective.set_era_indices(era_indices)

        # 順序付きブースティング用置換リスト
        if self.use_ordered_boosting:
            n_perms = min(self.n_permutations, n_samples)
            permutations = [rng.permutation(n_samples) for _ in range(n_perms)]

        best_val_loss = float("inf")
        rounds_no_improve = 0
        self.trees_ = []
        self.tree_lr_ = []
        self.train_losses_ = []

        for iteration in range(self.n_estimators):
            # DART：ツリーをドロップして予測を調整
            current_predictions = predictions.copy()
            if dart_mgr is not None and len(self.trees_) > 0:
                dropped = dart_mgr.sample_drops(len(self.trees_), rng)
                if dropped:
                    current_predictions = dart_mgr.adjust_predictions(
                        predictions, dart_tree_data, self.learning_rate, dropped)
            else:
                current_predictions = predictions

            # 勾配とヘッセ行列を計算
            gradients = objective.gradient(y, current_predictions)
            hessians = objective.hessian(y, current_predictions)

            # 時系列正則化：時系列勾配を追加
            if temporal_reg is not None:
                g_temp = temporal_reg.compute_temporal_gradient(current_predictions)
                gradients = gradients + g_temp

            # エラブースティング：エラ難易度重みで勾配をスケーリング
            if era_reweighter is not None:
                era_w = era_reweighter.compute_sample_weights(
                    current_predictions, y_1d, era_indices_arr)
                gradients = gradients * era_w
                hessians = hessians * era_w

            # フィーチャーエクスポージャーペナルティ：特徴相関を低減する勾配を追加
            if exposure_penalty_data is not None:
                n = len(current_predictions)
                std_P = current_predictions.std() + 1e-9
                P_c = current_predictions - current_predictions.mean()
                Xc = exposure_penalty_data['X_centered']
                Xs = exposure_penalty_data['X_std']
                lam = exposure_penalty_data['lambda']
                corr = (Xc.T @ P_c) / (n * std_P * Xs)           # 形状: (n_feats,)
                linear = (Xc @ (corr / Xs)) / (n * std_P)         # 形状: (n_samples,)
                quadratic = P_c * float(corr @ corr) / (n * std_P**2)
                gradients = gradients + 2.0 * lam * (linear - quadratic)

            # Era adaptive gradient clipping (per-era MAD)
            if era_clipper is not None and era_indices_arr is not None:
                gradients = era_clipper.clip(gradients, era_indices_arr)

            # 直交勾配射影
            if orth_projector is not None:
                gradients = orth_projector.project(gradients)

            # 勾配摂動
            if perturber is not None:
                gradients = perturber.perturb(gradients, rng)

            # Multi-target iteration update
            if hasattr(objective, 'set_iteration'):
                objective.set_iteration(iteration)

            # GOSS / TW-GOSS サンプリング
            sample_indices = np.arange(n_samples)

            if tw_goss is not None and era_indices_arr is not None:
                # TW-GOSS: combine gradient magnitude with temporal recency
                # Use era IDs as ordinal time index
                era_int = _era_to_int(era_indices_arr)
                tw_indices, tw_weights = tw_goss.sample(gradients, era_int)
                weighted_gradients = gradients.copy()
                weighted_hessians = hessians.copy()
                weight_map = np.ones(n_samples)
                weight_map[tw_indices] = tw_weights
                weighted_gradients *= weight_map
                weighted_hessians *= weight_map
                sample_indices = tw_indices
            elif goss is not None:
                goss_indices, goss_weights = goss.sample(gradients)
                weighted_gradients = gradients.copy()
                weighted_hessians = hessians.copy()
                weight_map = np.ones(n_samples)
                weight_map[goss_indices] = goss_weights
                weighted_gradients *= weight_map
                weighted_hessians *= weight_map
                sample_indices = goss_indices
            else:
                weighted_gradients = gradients
                weighted_hessians = hessians

                if self.subsample < 1.0:
                    n_sub = max(1, int(n_samples * self.subsample))
                    sample_indices = rng.choice(n_samples, size=n_sub, replace=False)

            # 複数置換による順序付きブースティング
            if self.use_ordered_boosting:
                if len(permutations) > 1:
                    all_grads = []
                    all_hess = []
                    for perm in permutations:
                        ordered_g = np.empty_like(weighted_gradients)
                        ordered_h = np.empty_like(weighted_hessians)
                        # ベクトル化ファンシーインデックス代入
                        ordered_g[perm] = weighted_gradients[perm]
                        ordered_h[perm] = weighted_hessians[perm]
                        all_grads.append(ordered_g)
                        all_hess.append(ordered_h)
                    # 置換間の中央値集約（外れ値に対して頑健）
                    weighted_gradients = np.median(all_grads, axis=0)
                    weighted_hessians = np.median(all_hess, axis=0)
                else:
                    perm = permutations[0]
                    ordered_grads = np.empty_like(weighted_gradients)
                    ordered_hess = np.empty_like(weighted_hessians)
                    ordered_grads[perm] = weighted_gradients[perm]
                    ordered_hess[perm] = weighted_hessians[perm]
                    weighted_gradients = ordered_grads
                    weighted_hessians = ordered_hess

            # 列サブサンプリング
            if self.colsample_bytree < 1.0:
                n_cols = X_binned.shape[1]
                n_sel = max(1, int(n_cols * self.colsample_bytree))
                col_indices = rng.choice(n_cols, size=n_sel, replace=False)
                X_tree = X_binned[:, col_indices]
            else:
                X_tree = X_binned
                col_indices = None

            # Era-adversarial split data (uses precomputed IDs from before loop)
            era_adv_data = None
            if _era_int_ids is not None:
                era_adv_data = {
                    'era_ids': _era_int_ids,
                    'n_eras': _n_eras,
                    'beta': self.era_adversarial_beta,
                }

            # ツリーを構築
            tree = DecisionTree(
                max_depth=self.max_depth,
                max_leaves=self.max_leaves,
                growth=self.growth,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                min_child_weight=self.min_child_weight,
                min_child_samples=self.min_child_samples,
                max_bins=self.max_bins,
                symmetric_depth=self.symmetric_depth,
                adaptive_reg=adaptive_reg,
                monotone_checker=monotone_checker,
                iteration=iteration,
                total_iterations=self.n_estimators,
                era_adversarial_data=era_adv_data,
            )
            tree.build(X_tree, weighted_gradients, weighted_hessians)

            # Non-smooth objectives (MAE, Huber): replace leaf values with the
            # true L1/Huber-optimal estimate (median / mean of residuals).
            # The Newton step −G/(H+λ) with constant hessian saturates at ±1
            # for MAE and can diverge for Huber when H≈0.
            if hasattr(objective, 'compute_leaf_value'):
                residuals_all = (y if y.ndim == 1 else y[:, 0]) - current_predictions
                _override_leaf_values(tree.root, residuals_all, objective)

            # 予測用に列マッピングを保存
            tree._col_indices = col_indices

            # ツリー予測値を計算
            if col_indices is not None:
                tree_pred = tree.predict(X_binned[:, col_indices])
            else:
                tree_pred = tree.predict(X_binned)

            # Era-aware DART: record era-variance for this tree
            if isinstance(dart_mgr, EraAwareDARTManager) and era_indices_arr is not None:
                dart_mgr.record_tree_era_variance(tree_pred, y_1d, era_indices_arr)

            # Sharpe-ratio tree regularization (design doc D):
            # Scale down trees whose per-era Spearman Sharpe is below threshold
            if (self.use_sharpe_tree_reg
                    and era_indices_arr is not None
                    and len(np.unique(era_indices_arr)) >= 2):
                tree_sr = _era_spearman_sharpe(tree_pred, y_1d, era_indices_arr)
                if tree_sr < self.sharpe_reg_threshold and tree_sr >= 0:
                    sr_scale = max(0.0, tree_sr / self.sharpe_reg_threshold)
                    tree_pred = tree_pred * sr_scale

            # DART スケーリング
            lr = self.learning_rate
            if dart_mgr is not None:
                scale = dart_mgr.compute_scale_factor()
                lr = self.learning_rate * scale

            predictions += lr * tree_pred
            self.trees_.append(tree)
            self.tree_lr_.append(lr)

            # DART 用にツリーデータを保存
            if dart_mgr is not None:
                dart_tree_data.append((tree, col_indices, X_binned))

            # 訓練損失
            train_loss = objective.loss(y, predictions)
            self.train_losses_.append(train_loss)

            # 検証と早期停止
            if X_val_binned is not None and eval_metric is not None:
                if col_indices is not None:
                    val_pred = tree.predict(X_val_binned[:, col_indices])
                else:
                    val_pred = tree.predict(X_val_binned)
                val_predictions += lr * val_pred
                val_loss = eval_metric(y_val, val_predictions)

                if self.verbose >= 1:
                    print(f"[{iteration}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_iteration_ = iteration
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1

                if (self.early_stopping_rounds is not None
                        and rounds_no_improve >= self.early_stopping_rounds):
                    if self.verbose >= 1:
                        print(f"Early stopping at iteration {iteration}")
                    break
            else:
                if self.verbose >= 1 and iteration % 10 == 0:
                    print(f"[{iteration}] train_loss={train_loss:.6f}")
                self.best_iteration_ = iteration

            # Sharpe-ratio early stopping (design doc M):
            # Uses era-wise Spearman Sharpe on *training* data as the stopping
            # criterion.  Requires era_indices; eval_set is not needed.
            if (self.use_sharpe_early_stopping
                    and era_indices_arr is not None
                    and len(np.unique(era_indices_arr)) >= 2):
                sr = _era_spearman_sharpe(predictions, y_1d, era_indices_arr)
                if sr > best_sr:
                    best_sr = sr
                    self.best_iteration_ = iteration
                    sr_patience_counter = 0
                else:
                    sr_patience_counter += 1
                if self.verbose >= 1 and iteration % 10 == 0:
                    print(f"[{iteration}] era_sharpe={sr:.4f}")
                if sr_patience_counter >= self.sharpe_es_patience:
                    if self.verbose >= 1:
                        print(f"Sharpe early stopping at iteration {iteration} "
                              f"(best SR={best_sr:.4f})")
                    break

        # C++ 予測用にツリーを平坦配列にシリアライズ
        self._cpp_trees = self._serialize_trees() if _HAS_CPP else None

        return self

    def _serialize_trees(self):
        """全ツリーを C++ predict_trees 用の平坦配列にシリアライズする。"""
        n_trees = self.best_iteration_ + 1
        trees_slice = self.trees_[:n_trees]

        all_feat, all_thresh, all_nan, all_val, all_left, all_right = [], [], [], [], [], []
        offsets  = []
        lrs      = []
        total    = 0

        for i, tree in enumerate(trees_slice):
            col_idx = tree._col_indices
            f, t, nd, v, lc, rc = tree.to_arrays(col_idx)
            offsets.append(total)
            all_feat  .append(f)
            all_thresh.append(t)
            all_nan   .append(nd)
            all_val   .append(v)
            # 子ノードインデックスをツリーローカルからグローバルへ変換
            all_left  .append(np.where(lc >= 0, lc + total, -1))
            all_right .append(np.where(rc >= 0, rc + total, -1))
            lrs.append(self.tree_lr_[i] if i < len(self.tree_lr_) else self.learning_rate)
            total += len(f)

        def cat(lst, dtype):
            return np.concatenate(lst).astype(dtype) if lst else np.array([], dtype=dtype)

        return {
            'features'  : cat(all_feat,   np.int32),
            'thresholds': cat(all_thresh, np.int32),
            'nan_dirs'  : cat(all_nan,    np.int32),
            'values'    : cat(all_val,    np.float64),
            'lefts'     : cat(all_left,   np.int32),
            'rights'    : cat(all_right,  np.int32),
            'offsets'   : np.array(offsets, dtype=np.int32),
            'lrs'       : np.array(lrs,     dtype=np.float64),
            'n_trees'   : n_trees,
        }

    def predict(self, X):
        """新しいデータの生スコアを予測する。"""
        X_encoded = self.cat_encoder_.transform(X)
        X_binned  = self.binner_.transform(X_encoded)

        # C++ 高速パス
        if _HAS_CPP and self._cpp_trees is not None:
            ct = self._cpp_trees
            return _cpp.predict_trees(
                ct['features'], ct['thresholds'], ct['nan_dirs'],
                ct['values'],   ct['lefts'],      ct['rights'],
                ct['offsets'],  ct['lrs'],
                X_binned, self.base_score_,
                ct['n_trees'], self.max_bins,
            )

        # Pure-Python フォールバック
        predictions = np.full(X_binned.shape[0], self.base_score_, dtype=np.float64)

        n_trees = self.best_iteration_ + 1
        for i, tree in enumerate(self.trees_[:n_trees]):
            lr = self.tree_lr_[i] if i < len(self.tree_lr_) else self.learning_rate
            if tree._col_indices is not None:
                pred = tree.predict(X_binned[:, tree._col_indices])
            else:
                pred = tree.predict(X_binned)
            predictions += lr * pred

        return predictions

    def feature_importances(self, importance_type="gain"):
        """特徴量重要度を計算する（分割回数またはゲイン合計）。"""
        n_features = self.binner_.n_features_bundled
        importances = np.zeros(n_features)

        for tree in self.trees_:
            if importance_type == "gain":
                for feat, gain in zip(tree.split_features_, tree.split_gains_):
                    if tree._col_indices is not None:
                        feat = tree._col_indices[feat]
                    if 0 <= feat < n_features:
                        importances[feat] += gain
            else:  # 分割回数
                for feat in tree.split_features_:
                    if tree._col_indices is not None:
                        feat = tree._col_indices[feat]
                    if 0 <= feat < n_features:
                        importances[feat] += 1

        total = importances.sum()
        if total > 0:
            importances /= total

        return importances
