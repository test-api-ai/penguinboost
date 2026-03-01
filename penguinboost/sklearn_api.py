"""PenguinBoost の scikit-learn 互換 API ラッパー。"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from penguinboost.core.boosting import BoostingEngine
from penguinboost.objectives.regression import MSEObjective, MAEObjective, HuberObjective
from penguinboost.objectives.classification import BinaryLoglossObjective, SoftmaxObjective
from penguinboost.objectives.ranking import LambdaRankObjective
from penguinboost.objectives.survival import CoxObjective
from penguinboost.objectives.quantile import QuantileObjective, CVaRObjective
from penguinboost.objectives.regression import AsymmetricHuberObjective
from penguinboost.objectives.multi_target import MultiTargetAuxiliaryObjective
from penguinboost.metrics.metrics import rmse, logloss
from penguinboost.utils import check_array, check_target


class _PenguinBoostBase(BaseEstimator):
    """共通パラメータとロジックを持つ基底クラス。"""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 max_leaves=31, growth="leafwise", reg_lambda=1.0,
                 reg_alpha=0.0, min_child_weight=1.0, min_child_samples=1,
                 subsample=1.0, colsample_bytree=1.0, max_bins=255,
                 use_goss=False, goss_top_rate=0.2, goss_other_rate=0.1,
                 use_ordered_boosting=False, n_permutations=4,
                 cat_features=None, cat_smoothing=10.0, efb_threshold=0.0,
                 early_stopping_rounds=None, importance_type="gain",
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
                 use_tw_goss=False, tw_goss_decay=0.01,
                 use_era_gradient_clipping=False, era_clip_multiplier=4.0,
                 use_era_aware_dart=False, era_dart_var_scale=20.0,
                 use_sharpe_early_stopping=False, sharpe_es_patience=50,
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
        self.importance_type = importance_type
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

    def _build_engine(self):
        return BoostingEngine(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_leaves=self.max_leaves,
            growth=self.growth,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            min_child_weight=self.min_child_weight,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            max_bins=self.max_bins,
            use_goss=self.use_goss,
            goss_top_rate=self.goss_top_rate,
            goss_other_rate=self.goss_other_rate,
            use_ordered_boosting=self.use_ordered_boosting,
            n_permutations=self.n_permutations,
            cat_features=self.cat_features,
            cat_smoothing=self.cat_smoothing,
            efb_threshold=self.efb_threshold,
            early_stopping_rounds=self.early_stopping_rounds,
            use_dart=self.use_dart,
            dart_drop_rate=self.dart_drop_rate,
            dart_skip_drop=self.dart_skip_drop,
            use_gradient_perturbation=self.use_gradient_perturbation,
            gradient_clip_tau=self.gradient_clip_tau,
            gradient_noise_eta=self.gradient_noise_eta,
            use_adaptive_reg=self.use_adaptive_reg,
            adaptive_alpha=self.adaptive_alpha,
            adaptive_mu=self.adaptive_mu,
            monotone_constraints=self.monotone_constraints,
            use_temporal_reg=self.use_temporal_reg,
            temporal_rho=self.temporal_rho,
            symmetric_depth=self.symmetric_depth,
            use_orthogonal_gradients=self.use_orthogonal_gradients,
            orthogonal_strength=self.orthogonal_strength,
            orthogonal_eps=self.orthogonal_eps,
            orthogonal_features=self.orthogonal_features,
            use_era_boosting=self.use_era_boosting,
            era_boosting_method=self.era_boosting_method,
            era_boosting_temp=self.era_boosting_temp,
            use_feature_exposure_penalty=self.use_feature_exposure_penalty,
            feature_exposure_lambda=self.feature_exposure_lambda,
            exposure_penalty_features=self.exposure_penalty_features,
            use_tw_goss=self.use_tw_goss,
            tw_goss_decay=self.tw_goss_decay,
            use_era_gradient_clipping=self.use_era_gradient_clipping,
            era_clip_multiplier=self.era_clip_multiplier,
            use_era_aware_dart=self.use_era_aware_dart,
            era_dart_var_scale=self.era_dart_var_scale,
            use_sharpe_early_stopping=self.use_sharpe_early_stopping,
            sharpe_es_patience=self.sharpe_es_patience,
            use_sharpe_tree_reg=self.use_sharpe_tree_reg,
            sharpe_reg_threshold=self.sharpe_reg_threshold,
            use_era_adversarial_split=self.use_era_adversarial_split,
            era_adversarial_beta=self.era_adversarial_beta,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    @property
    def feature_importances_(self):
        return self.engine_.feature_importances(self.importance_type)


class PenguinBoostRegressor(_PenguinBoostBase, RegressorMixin):
    """PenguinBoost の scikit-learn 互換回帰器。

    Parameters
    ----------
    objective : str
        損失関数: 'mse'（デフォルト）、'mae'、'huber'。
    huber_delta : float
        Huber 損失のデルタパラメータ（デフォルト 1.0）。

    その他のパラメータは _PenguinBoostBase を参照。
    """

    def __init__(self, objective="mse", huber_delta=1.0, asymmetric_kappa=2.0,
                 n_estimators=100, learning_rate=0.1, max_depth=6,
                 max_leaves=31, growth="leafwise", reg_lambda=1.0,
                 reg_alpha=0.0, min_child_weight=1.0, min_child_samples=1,
                 subsample=1.0, colsample_bytree=1.0, max_bins=255,
                 use_goss=False, goss_top_rate=0.2, goss_other_rate=0.1,
                 use_ordered_boosting=False, n_permutations=4,
                 cat_features=None, cat_smoothing=10.0, efb_threshold=0.0,
                 early_stopping_rounds=None, importance_type="gain",
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
                 use_tw_goss=False, tw_goss_decay=0.01,
                 use_era_gradient_clipping=False, era_clip_multiplier=4.0,
                 use_era_aware_dart=False, era_dart_var_scale=20.0,
                 use_sharpe_early_stopping=False, sharpe_es_patience=50,
                 use_sharpe_tree_reg=False, sharpe_reg_threshold=0.5,
                 use_era_adversarial_split=False, era_adversarial_beta=0.3,
                 verbose=0, random_state=None):
        super().__init__(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, max_leaves=max_leaves, growth=growth,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            min_child_weight=min_child_weight, min_child_samples=min_child_samples,
            subsample=subsample, colsample_bytree=colsample_bytree,
            max_bins=max_bins, use_goss=use_goss, goss_top_rate=goss_top_rate,
            goss_other_rate=goss_other_rate,
            use_ordered_boosting=use_ordered_boosting, n_permutations=n_permutations,
            cat_features=cat_features, cat_smoothing=cat_smoothing,
            efb_threshold=efb_threshold, early_stopping_rounds=early_stopping_rounds,
            importance_type=importance_type,
            use_dart=use_dart, dart_drop_rate=dart_drop_rate,
            dart_skip_drop=dart_skip_drop,
            use_gradient_perturbation=use_gradient_perturbation,
            gradient_clip_tau=gradient_clip_tau, gradient_noise_eta=gradient_noise_eta,
            use_adaptive_reg=use_adaptive_reg, adaptive_alpha=adaptive_alpha,
            adaptive_mu=adaptive_mu, monotone_constraints=monotone_constraints,
            use_temporal_reg=use_temporal_reg, temporal_rho=temporal_rho,
            symmetric_depth=symmetric_depth,
            use_orthogonal_gradients=use_orthogonal_gradients,
            orthogonal_strength=orthogonal_strength, orthogonal_eps=orthogonal_eps,
            orthogonal_features=orthogonal_features,
            use_era_boosting=use_era_boosting,
            era_boosting_method=era_boosting_method, era_boosting_temp=era_boosting_temp,
            use_feature_exposure_penalty=use_feature_exposure_penalty,
            feature_exposure_lambda=feature_exposure_lambda,
            exposure_penalty_features=exposure_penalty_features,
            use_tw_goss=use_tw_goss, tw_goss_decay=tw_goss_decay,
            use_era_gradient_clipping=use_era_gradient_clipping,
            era_clip_multiplier=era_clip_multiplier,
            use_era_aware_dart=use_era_aware_dart,
            era_dart_var_scale=era_dart_var_scale,
            use_sharpe_early_stopping=use_sharpe_early_stopping,
            sharpe_es_patience=sharpe_es_patience,
            use_sharpe_tree_reg=use_sharpe_tree_reg,
            sharpe_reg_threshold=sharpe_reg_threshold,
            use_era_adversarial_split=use_era_adversarial_split,
            era_adversarial_beta=era_adversarial_beta,
            verbose=verbose, random_state=random_state)
        self.objective = objective
        self.huber_delta = huber_delta
        self.asymmetric_kappa = asymmetric_kappa

    def fit(self, X, y, eval_set=None, era_indices=None):
        """回帰器を学習する。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        eval_set : tuple (X_val, y_val) or None
        era_indices : array-like of shape (n_samples,) or None
            エラ（時間期間）ラベル。エラブースティングと
            エラ条件付き目的関数（MaxSharpeEraObjective）に必要。

        Returns
        -------
        self
        """
        X = check_array(X)
        y = check_target(y)

        obj_map = {
            "mse": MSEObjective,
            "mae": MAEObjective,
            "huber": lambda: HuberObjective(delta=self.huber_delta),
            "asymmetric_huber": lambda: AsymmetricHuberObjective(
                delta=self.huber_delta, kappa=self.asymmetric_kappa),
        }
        obj_cls = obj_map[self.objective]
        obj = obj_cls() if callable(obj_cls) else obj_cls()

        self.engine_ = self._build_engine()
        self.engine_.fit(X, y, obj, eval_set=eval_set,
                         eval_metric=rmse if eval_set else None,
                         era_indices=era_indices)
        self.objective_ = obj
        return self

    def predict(self, X):
        X = check_array(X)
        return self.engine_.predict(X)

    # ── 事後学習ユーティリティ ─────────────────────────────────────────

    def neutralize(self, predictions, X, proportion=1.0, per_era=False,
                   eras=None, features=None):
        """事後学習フィーチャー中立化（Numerai スタイル）。

        予測値の線形特徴量説明可能成分を除去する。

        Parameters
        ----------
        predictions : array-like of shape (n_samples,)
            中立化する予測値。
        X : array-like of shape (n_samples, n_features)
            中立化に使用する特徴量行列。
        proportion : float in [0, 1]
            除去する特徴量エクスポージャーの割合。デフォルト 1.0（完全除去）。
        per_era : bool
            True の場合、各エラ内で個別に中立化を適用する。
        eras : array-like or None
            エララベル（per_era=True 時に必要）。
        features : list of int or None
            中立化対象の特徴量列インデックス。None = 全特徴量。

        Returns
        -------
        np.ndarray of shape (n_samples,)
            中立化後の予測値。
        """
        from penguinboost.core.neutralization import FeatureNeutralizer
        predictions = np.asarray(predictions, dtype=np.float64)
        X = check_array(X)
        return FeatureNeutralizer().neutralize(
            predictions, X,
            features=features, proportion=proportion,
            per_era=per_era, eras=eras)

    def feature_exposure(self, X, predictions=None, features=None):
        """予測値と X の特徴量ごとのピアソン相関を計算する。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        predictions : array-like or None
            None の場合、このモデルで X から予測値を計算する。
        features : list of int or None

        Returns
        -------
        np.ndarray of shape (n_features,)
            各特徴量列との予測値の相関。
        """
        from penguinboost.core.neutralization import FeatureNeutralizer
        X = check_array(X)
        if predictions is None:
            predictions = self.predict(X)
        return FeatureNeutralizer().feature_exposure(
            np.asarray(predictions, dtype=np.float64), X, features=features)


class PenguinBoostClassifier(_PenguinBoostBase, ClassifierMixin):
    """PenguinBoost の scikit-learn 互換分類器。

    2 値分類と多値分類をサポートする。
    """

    def fit(self, X, y, eval_set=None):
        X = check_array(X)
        y = check_target(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[v] for v in y], dtype=np.float64)

        if self.n_classes_ == 2:
            obj = BinaryLoglossObjective()
            self.engine_ = self._build_engine()
            eval_metric = logloss if eval_set else None
            if eval_set is not None:
                X_val, y_val = eval_set
                y_val_mapped = np.array([label_map[v] for v in y_val], dtype=np.float64)
                eval_set = (check_array(X_val), y_val_mapped)
            self.engine_.fit(X, y_mapped, obj, eval_set=eval_set,
                             eval_metric=eval_metric)
            self.objective_ = obj
        else:
            obj = SoftmaxObjective(n_classes=self.n_classes_)
            init_scores = obj.init_score(y_mapped)
            self._fit_multiclass(X, y_mapped, obj, init_scores)

        return self

    def _fit_multiclass(self, X, y, obj, init_scores):
        """クラスごとに個別のブースティングエンジンを学習する。"""
        n_samples = len(y)
        self.engines_ = []
        predictions = np.tile(init_scores, (n_samples, 1))

        for k in range(self.n_classes_):
            engine = self._build_engine()
            y_k = (y == k).astype(np.float64)
            obj_k = BinaryLoglossObjective()
            engine.fit(X, y_k, obj_k)
            self.engines_.append(engine)

        self.objective_ = obj

    def predict_proba(self, X):
        X = check_array(X)

        if self.n_classes_ == 2:
            raw = self.engine_.predict(X)
            prob_pos = self.objective_.transform(raw)
            return np.column_stack([1 - prob_pos, prob_pos])
        else:
            scores = np.zeros((X.shape[0], self.n_classes_))
            for k, engine in enumerate(self.engines_):
                raw = engine.predict(X)
                scores[:, k] = raw

            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    @property
    def feature_importances_(self):
        if self.n_classes_ == 2:
            return self.engine_.feature_importances(self.importance_type)
        else:
            importances = np.zeros_like(
                self.engines_[0].feature_importances(self.importance_type))
            for engine in self.engines_:
                importances += engine.feature_importances(self.importance_type)
            total = importances.sum()
            if total > 0:
                importances /= total
            return importances


class PenguinBoostRanker(_PenguinBoostBase):
    """PenguinBoost の学習ランキング用ランカー。"""

    def fit(self, X, y, group, eval_set=None, eval_group=None):
        X = check_array(X)
        y = check_target(y)

        obj = LambdaRankObjective()
        obj.set_group(group)

        self.engine_ = self._build_engine()
        self.engine_.fit(X, y, obj)
        self.objective_ = obj
        return self

    def predict(self, X):
        X = check_array(X)
        return self.engine_.predict(X)


class PenguinBoostSurvival(_PenguinBoostBase):
    """PenguinBoost の生存分析器（Cox PH モデル）。"""

    def fit(self, X, times, events, eval_set=None):
        X = check_array(X)
        times = np.asarray(times, dtype=np.float64)
        events = np.asarray(events, dtype=np.float64)

        y = np.column_stack([times, events])

        obj = CoxObjective()
        obj.set_data(times, events.astype(bool))

        self.engine_ = self._build_engine()
        self.engine_.fit(X, y, obj)
        self.objective_ = obj
        return self

    def predict(self, X):
        X = check_array(X)
        raw = self.engine_.predict(X)
        return np.exp(raw)

    def predict_hazard(self, X):
        return self.predict(X)


class PenguinBoostQuantileRegressor(_PenguinBoostBase, RegressorMixin):
    """PenguinBoost の分位点回帰器（VaR/CVaR 推定用）。

    Parameters
    ----------
    objective : str
        'quantile'（ピンボール損失）または 'cvar'（CVaR/期待ショートフォール）。
    alpha : float
        分位点水準（例: 5% VaR の場合は 0.05）。

    その他のパラメータは _PenguinBoostBase を参照。
    """

    def __init__(self, objective="quantile", alpha=0.5,
                 n_estimators=100, learning_rate=0.1, max_depth=6,
                 max_leaves=31, growth="leafwise", reg_lambda=1.0,
                 reg_alpha=0.0, min_child_weight=1.0, min_child_samples=1,
                 subsample=1.0, colsample_bytree=1.0, max_bins=255,
                 use_goss=False, goss_top_rate=0.2, goss_other_rate=0.1,
                 use_ordered_boosting=False, n_permutations=4,
                 cat_features=None, cat_smoothing=10.0, efb_threshold=0.0,
                 early_stopping_rounds=None, importance_type="gain",
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
                 verbose=0, random_state=None):
        super().__init__(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, max_leaves=max_leaves, growth=growth,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            min_child_weight=min_child_weight, min_child_samples=min_child_samples,
            subsample=subsample, colsample_bytree=colsample_bytree,
            max_bins=max_bins, use_goss=use_goss, goss_top_rate=goss_top_rate,
            goss_other_rate=goss_other_rate,
            use_ordered_boosting=use_ordered_boosting, n_permutations=n_permutations,
            cat_features=cat_features, cat_smoothing=cat_smoothing,
            efb_threshold=efb_threshold, early_stopping_rounds=early_stopping_rounds,
            importance_type=importance_type,
            use_dart=use_dart, dart_drop_rate=dart_drop_rate,
            dart_skip_drop=dart_skip_drop,
            use_gradient_perturbation=use_gradient_perturbation,
            gradient_clip_tau=gradient_clip_tau, gradient_noise_eta=gradient_noise_eta,
            use_adaptive_reg=use_adaptive_reg, adaptive_alpha=adaptive_alpha,
            adaptive_mu=adaptive_mu, monotone_constraints=monotone_constraints,
            use_temporal_reg=use_temporal_reg, temporal_rho=temporal_rho,
            symmetric_depth=symmetric_depth,
            use_orthogonal_gradients=use_orthogonal_gradients,
            orthogonal_strength=orthogonal_strength, orthogonal_eps=orthogonal_eps,
            orthogonal_features=orthogonal_features,
            use_era_boosting=use_era_boosting,
            era_boosting_method=era_boosting_method, era_boosting_temp=era_boosting_temp,
            use_feature_exposure_penalty=use_feature_exposure_penalty,
            feature_exposure_lambda=feature_exposure_lambda,
            exposure_penalty_features=exposure_penalty_features,
            verbose=verbose, random_state=random_state)
        self.objective = objective
        self.alpha = alpha

    def fit(self, X, y, eval_set=None):
        X = check_array(X)
        y = check_target(y)

        if self.objective == "cvar":
            obj = CVaRObjective(alpha=self.alpha)
        else:
            obj = QuantileObjective(alpha=self.alpha)

        self.engine_ = self._build_engine()
        self.engine_.fit(X, y, obj, eval_set=eval_set)
        self.objective_ = obj
        return self

    def predict(self, X):
        X = check_array(X)
        return self.engine_.predict(X)
