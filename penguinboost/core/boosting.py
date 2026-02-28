"""Boosting engine with all v2 features: DART, gradient perturbation, multi-permutation
ordered boosting, temporal regularization, adaptive regularization, hybrid growth,
monotone constraints, and regime detection.

C++ acceleration: predict() uses _core.predict_trees when available.
"""

import numpy as np

try:
    from penguinboost import _core as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
from penguinboost.core.tree import DecisionTree
from penguinboost.core.sampling import GOSSSampler
from penguinboost.core.binning import FeatureBinner
from penguinboost.core.categorical import OrderedTargetEncoder
from penguinboost.core.regularization import AdaptiveRegularizer, GradientPerturber
from penguinboost.core.dart import DARTManager
from penguinboost.core.monotone import MonotoneConstraintChecker
from penguinboost.core.financial import TemporalRegularizer


class BoostingEngine:
    """Core gradient boosting engine combining LightGBM, CatBoost, and XGBoost techniques.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage rate.
    max_depth : int
        Maximum tree depth.
    max_leaves : int
        Maximum number of leaves per tree.
    growth : str
        Tree growth strategy: 'leafwise', 'symmetric', 'depthwise', or 'hybrid'.
    reg_lambda : float
        L2 regularization.
    reg_alpha : float
        L1 regularization.
    min_child_weight : float
        Minimum hessian sum in leaves.
    min_child_samples : int
        Minimum samples in leaves.
    subsample : float
        Row subsampling ratio (1.0 = no subsampling).
    colsample_bytree : float
        Column subsampling ratio per tree.
    max_bins : int
        Maximum histogram bins.
    use_goss : bool
        Use GOSS sampling.
    goss_top_rate : float
        GOSS top gradient fraction.
    goss_other_rate : float
        GOSS other samples fraction.
    use_ordered_boosting : bool
        Use CatBoost-style ordered boosting.
    n_permutations : int
        Number of permutations for multi-permutation ordered boosting (v2).
    cat_features : list or None
        Indices of categorical features.
    cat_smoothing : float
        Smoothing for ordered target statistics.
    efb_threshold : float
        EFB conflict threshold (0 = disabled).
    early_stopping_rounds : int or None
        Stop if no improvement for this many rounds.
    use_dart : bool
        Enable DART (Dropout Trees) regularization.
    dart_drop_rate : float
        DART: probability of dropping each tree.
    dart_skip_drop : float
        DART: probability of skipping dropout for an iteration.
    use_gradient_perturbation : bool
        Enable gradient perturbation (clipping + noise).
    gradient_clip_tau : float
        Gradient clipping threshold.
    gradient_noise_eta : float
        Gradient noise scale (relative to std).
    use_adaptive_reg : bool
        Enable Bayesian adaptive regularization.
    adaptive_alpha : float
        Schedule scaling for adaptive regularization.
    adaptive_mu : float
        Bayesian child-node penalty coefficient.
    monotone_constraints : dict or None
        Feature index -> +1/-1 monotone constraint direction.
    use_temporal_reg : bool
        Enable temporal regularization (financial).
    temporal_rho : float
        Temporal smoothness penalty strength.
    symmetric_depth : int
        For hybrid growth: depth to switch from symmetric to leaf-wise.
    verbose : int
        Verbosity level.
    random_state : int or None
        Random seed.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 max_leaves=31, growth="leafwise", reg_lambda=1.0, reg_alpha=0.0,
                 min_child_weight=1.0, min_child_samples=1, subsample=1.0,
                 colsample_bytree=1.0, max_bins=255, use_goss=False,
                 goss_top_rate=0.2, goss_other_rate=0.1,
                 use_ordered_boosting=False, n_permutations=4,
                 cat_features=None, cat_smoothing=10.0, efb_threshold=0.0,
                 early_stopping_rounds=None,
                 # v2 DART
                 use_dart=False, dart_drop_rate=0.1, dart_skip_drop=0.0,
                 # v2 gradient perturbation
                 use_gradient_perturbation=False,
                 gradient_clip_tau=5.0, gradient_noise_eta=0.1,
                 # v2 adaptive regularization
                 use_adaptive_reg=False, adaptive_alpha=0.5, adaptive_mu=1.0,
                 # v2 monotone constraints
                 monotone_constraints=None,
                 # v2 temporal regularization
                 use_temporal_reg=False, temporal_rho=0.1,
                 # v2 hybrid growth
                 symmetric_depth=3,
                 # v3 orthogonal gradient projection
                 use_orthogonal_gradients=False,
                 orthogonal_strength=1.0, orthogonal_eps=1e-4,
                 orthogonal_features=None,
                 # v3 era boosting
                 use_era_boosting=False,
                 era_boosting_method='hard_era', era_boosting_temp=1.0,
                 # v3 feature exposure penalty
                 use_feature_exposure_penalty=False,
                 feature_exposure_lambda=0.1, exposure_penalty_features=None,
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
        # v3
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
        self.verbose = verbose
        self.random_state = random_state

        # Fitted attributes
        self.trees_ = []
        self.tree_lr_ = []  # effective learning rate per tree (for DART)
        self.binner_ = None
        self.cat_encoder_ = None
        self.base_score_ = 0.0
        self.best_iteration_ = 0
        self.train_losses_ = []
        self._cpp_trees = None   # serialized flat-array representation for C++

    def fit(self, X, y, objective, eval_set=None, eval_metric=None,
            era_indices=None):
        """Train the boosting model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,) or (n_samples, 2) for survival
        objective : object with gradient(y, pred) and hessian(y, pred) methods
        eval_set : tuple (X_val, y_val) or None
        eval_metric : callable(y_true, y_pred) -> float, or None
        era_indices : array-like of shape (n_samples,) or None
            Era (time-period) labels. Enables era-aware features:
            era boosting reweighting and era-conditional objectives.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # Categorical encoding
        self.cat_encoder_ = OrderedTargetEncoder(
            smoothing=self.cat_smoothing, random_state=self.random_state)
        y_for_encode = y[:, 0] if y.ndim == 2 else y
        self.cat_encoder_.fit(X, y_for_encode, cat_features=self.cat_features)
        X_encoded = self.cat_encoder_.transform_train(X, y_for_encode)

        # Binning + EFB
        self.binner_ = FeatureBinner(max_bins=self.max_bins,
                                     efb_threshold=self.efb_threshold)
        X_binned = self.binner_.fit_transform(X_encoded)

        # Bin validation set if provided
        X_val_binned = None
        y_val = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_encoded = self.cat_encoder_.transform(X_val)
            X_val_binned = self.binner_.transform(X_val_encoded)

        # Initialize predictions
        self.base_score_ = objective.init_score(y)
        predictions = np.full(n_samples, self.base_score_, dtype=np.float64)

        if X_val_binned is not None:
            val_predictions = np.full(len(y_val), self.base_score_, dtype=np.float64)

        # GOSS sampler
        goss = None
        if self.use_goss:
            goss = GOSSSampler(self.goss_top_rate, self.goss_other_rate)

        # v2: Adaptive regularizer
        adaptive_reg = None
        if self.use_adaptive_reg:
            adaptive_reg = AdaptiveRegularizer(
                lambda_base=self.reg_lambda,
                alpha=self.adaptive_alpha,
                mu=self.adaptive_mu)

        # v2: Gradient perturber
        perturber = None
        if self.use_gradient_perturbation:
            perturber = GradientPerturber(
                tau=self.gradient_clip_tau,
                eta=self.gradient_noise_eta)

        # v2: DART manager
        dart_mgr = None
        dart_tree_data = []  # (tree, col_indices, X_binned_ref)
        if self.use_dart:
            dart_mgr = DARTManager(
                drop_rate=self.dart_drop_rate,
                skip_drop=self.dart_skip_drop)

        # v2: Monotone constraint checker
        monotone_checker = None
        if self.monotone_constraints:
            monotone_checker = MonotoneConstraintChecker(self.monotone_constraints)

        # v2: Temporal regularizer
        temporal_reg = None
        if self.use_temporal_reg:
            temporal_reg = TemporalRegularizer(rho=self.temporal_rho)

        # v3: Orthogonal gradient projector
        orth_projector = None
        if self.use_orthogonal_gradients:
            from penguinboost.core.neutralization import OrthogonalGradientProjector
            orth_projector = OrthogonalGradientProjector(
                strength=self.orthogonal_strength,
                eps=self.orthogonal_eps,
                features=self.orthogonal_features)
            orth_projector.fit(X_encoded)   # pre-compute (X^TX + εI)^{-1}

        # v3: Era boosting reweighter
        era_reweighter = None
        era_indices_arr = None
        y_1d = y[:, 0] if y.ndim == 2 else y   # 1-D target for era metrics
        if self.use_era_boosting and era_indices is not None:
            from penguinboost.core.era_boost import EraBoostingReweighter
            era_reweighter = EraBoostingReweighter(
                method=self.era_boosting_method,
                temperature=self.era_boosting_temp)
            era_indices_arr = np.asarray(era_indices)

        # v3: Feature exposure penalty (pre-compute centering/std once)
        exposure_penalty_data = None
        if self.use_feature_exposure_penalty:
            feats = self.exposure_penalty_features
            X_ep = X_encoded[:, feats] if feats is not None else X_encoded
            exposure_penalty_data = {
                'X_centered': X_ep - X_ep.mean(axis=0),
                'X_std':      X_ep.std(axis=0) + 1e-9,
                'lambda':     self.feature_exposure_lambda,
            }

        # v3: Wire era_indices into era-conditional objectives (e.g. MaxSharpeEra)
        if era_indices is not None and hasattr(objective, 'set_era_indices'):
            objective.set_era_indices(era_indices)

        # Ordered boosting permutations (v2: multi-permutation with median)
        if self.use_ordered_boosting:
            n_perms = min(self.n_permutations, n_samples)
            permutations = [rng.permutation(n_samples) for _ in range(n_perms)]

        best_val_loss = float("inf")
        rounds_no_improve = 0
        self.trees_ = []
        self.tree_lr_ = []
        self.train_losses_ = []

        for iteration in range(self.n_estimators):
            # v2: DART - drop trees and adjust predictions
            current_predictions = predictions.copy()
            if dart_mgr is not None and len(self.trees_) > 0:
                dropped = dart_mgr.sample_drops(len(self.trees_), rng)
                if dropped:
                    current_predictions = dart_mgr.adjust_predictions(
                        predictions, dart_tree_data, self.learning_rate, dropped)
            else:
                current_predictions = predictions

            # Compute gradients and hessians
            gradients = objective.gradient(y, current_predictions)
            hessians = objective.hessian(y, current_predictions)

            # v2: Temporal regularization - add temporal gradient
            if temporal_reg is not None:
                g_temp = temporal_reg.compute_temporal_gradient(current_predictions)
                gradients = gradients + g_temp

            # v3: Era boosting — scale gradients by era difficulty weights
            if era_reweighter is not None:
                era_w = era_reweighter.compute_sample_weights(
                    current_predictions, y_1d, era_indices_arr)
                gradients = gradients * era_w
                hessians = hessians * era_w

            # v3: Feature exposure penalty — add gradient to reduce feature correlation
            if exposure_penalty_data is not None:
                n = len(current_predictions)
                std_P = current_predictions.std() + 1e-9
                P_c = current_predictions - current_predictions.mean()
                Xc = exposure_penalty_data['X_centered']
                Xs = exposure_penalty_data['X_std']
                lam = exposure_penalty_data['lambda']
                corr = (Xc.T @ P_c) / (n * std_P * Xs)           # (n_feats,)
                linear = (Xc @ (corr / Xs)) / (n * std_P)         # (n_samples,)
                quadratic = P_c * float(corr @ corr) / (n * std_P**2)
                gradients = gradients + 2.0 * lam * (linear - quadratic)

            # v3: Orthogonal gradient projection
            if orth_projector is not None:
                gradients = orth_projector.project(gradients)

            # v2: Gradient perturbation
            if perturber is not None:
                gradients = perturber.perturb(gradients, rng)

            # GOSS sampling
            sample_indices = np.arange(n_samples)

            if goss is not None:
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

            # v2: Multi-permutation Ordered Boosting with median aggregation
            if self.use_ordered_boosting:
                if len(permutations) > 1:
                    all_grads = []
                    all_hess = []
                    for perm in permutations:
                        ordered_g = np.empty_like(weighted_gradients)
                        ordered_h = np.empty_like(weighted_hessians)
                        # Vectorized fancy-index assignment (replaces Python for-loop)
                        ordered_g[perm] = weighted_gradients[perm]
                        ordered_h[perm] = weighted_hessians[perm]
                        all_grads.append(ordered_g)
                        all_hess.append(ordered_h)
                    # Median aggregation across permutations (robust to outliers)
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

            # Column subsampling
            if self.colsample_bytree < 1.0:
                n_cols = X_binned.shape[1]
                n_sel = max(1, int(n_cols * self.colsample_bytree))
                col_indices = rng.choice(n_cols, size=n_sel, replace=False)
                X_tree = X_binned[:, col_indices]
            else:
                X_tree = X_binned
                col_indices = None

            # Build tree (v2: pass adaptive_reg, monotone_checker, growth params)
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
            )
            tree.build(X_tree, weighted_gradients, weighted_hessians)

            # Store column mapping for prediction
            tree._col_indices = col_indices

            # Compute tree predictions
            if col_indices is not None:
                tree_pred = tree.predict(X_binned[:, col_indices])
            else:
                tree_pred = tree.predict(X_binned)

            # v2: DART scaling
            lr = self.learning_rate
            if dart_mgr is not None:
                scale = dart_mgr.compute_scale_factor()
                lr = self.learning_rate * scale

            predictions += lr * tree_pred
            self.trees_.append(tree)
            self.tree_lr_.append(lr)

            # v2: Store tree data for DART
            if dart_mgr is not None:
                dart_tree_data.append((tree, col_indices, X_binned))

            # Training loss
            train_loss = objective.loss(y, predictions)
            self.train_losses_.append(train_loss)

            # Validation & early stopping
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

        # Serialize trees to flat arrays for C++ prediction
        self._cpp_trees = self._serialize_trees() if _HAS_CPP else None

        return self

    def _serialize_trees(self):
        """Serialize all active trees to flat arrays for C++ predict_trees."""
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
            # Shift left/right child indices from tree-local to global
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
        """Predict raw scores for new data."""
        X_encoded = self.cat_encoder_.transform(X)
        X_binned  = self.binner_.transform(X_encoded)

        # ---- C++ fast path ----
        if _HAS_CPP and self._cpp_trees is not None:
            ct = self._cpp_trees
            return _cpp.predict_trees(
                ct['features'], ct['thresholds'], ct['nan_dirs'],
                ct['values'],   ct['lefts'],      ct['rights'],
                ct['offsets'],  ct['lrs'],
                X_binned, self.base_score_,
                ct['n_trees'], self.max_bins,
            )

        # ---- Pure-Python fallback ----
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
        """Compute feature importances (split count or total gain)."""
        n_features = self.binner_.n_features_bundled
        importances = np.zeros(n_features)

        for tree in self.trees_:
            if importance_type == "gain":
                for feat, gain in zip(tree.split_features_, tree.split_gains_):
                    if tree._col_indices is not None:
                        feat = tree._col_indices[feat]
                    if 0 <= feat < n_features:
                        importances[feat] += gain
            else:  # split count
                for feat in tree.split_features_:
                    if tree._col_indices is not None:
                        feat = tree._col_indices[feat]
                    if 0 <= feat < n_features:
                        importances[feat] += 1

        total = importances.sum()
        if total > 0:
            importances /= total

        return importances
