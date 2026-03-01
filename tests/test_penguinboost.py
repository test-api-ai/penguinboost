"""Tests for PenguinBoost v2 library."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression, make_classification
from sklearn.model_selection import train_test_split

from penguinboost import (
    PenguinBoostClassifier, PenguinBoostRegressor,
    PenguinBoostRanker, PenguinBoostSurvival, PenguinBoostQuantileRegressor,
)
from penguinboost.core.binning import FeatureBinner
from penguinboost.core.histogram import HistogramBuilder
from penguinboost.core.sampling import GOSSSampler
from penguinboost.core.categorical import OrderedTargetEncoder
from penguinboost.core.regularization import AdaptiveRegularizer, GradientPerturber, FeatureStabilityTracker
from penguinboost.core.dart import DARTManager
from penguinboost.core.monotone import MonotoneConstraintChecker
from penguinboost.core.financial import PurgedKFold, TemporalRegularizer, RegimeDetector
from penguinboost.objectives.quantile import QuantileObjective, CVaRObjective
from penguinboost.metrics.metrics import (
    rmse, mae, r2_score, logloss, auc, accuracy, ndcg_at_k,
    concordance_index, sharpe_ratio, max_drawdown, quantile_loss,
)


# --- Core module tests (v1, preserved) ---

class TestFeatureBinner:
    def test_basic_binning(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        binner = FeatureBinner(max_bins=10)
        X_binned = binner.fit_transform(X)
        assert X_binned.shape == X.shape
        assert X_binned.dtype == np.uint8
        assert X_binned.max() <= 10

    def test_nan_handling(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
        binner = FeatureBinner(max_bins=5)
        X_binned = binner.fit_transform(X)
        assert X_binned[0, 1] == 5 or X_binned[2, 0] == 5

    def test_efb(self):
        rng = np.random.RandomState(42)
        X = np.zeros((100, 5))
        X[:20, 0] = rng.randn(20)
        X[20:40, 1] = rng.randn(20)
        X[40:60, 2] = rng.randn(20)
        X[:, 3] = rng.randn(100)
        X[:, 4] = rng.randn(100)

        binner = FeatureBinner(max_bins=255, efb_threshold=0.05)
        X_binned = binner.fit_transform(X)
        assert X_binned.shape[1] <= 5


class TestHistogramBuilder:
    def test_build_and_find_split(self):
        rng = np.random.RandomState(42)
        X = rng.randint(0, 10, size=(100, 2)).astype(np.uint8)
        grads = rng.randn(100)
        hess = np.ones(100)

        builder = HistogramBuilder(max_bins=255)
        g_hist, h_hist, c_hist = builder.build_histogram(X, grads, hess)

        assert g_hist.shape == (2, 256)
        feat, bin_t, gain, nan_dir = builder.find_best_split(
            g_hist, h_hist, c_hist)
        assert gain >= 0

    def test_subtraction_trick(self):
        rng = np.random.RandomState(42)
        X = rng.randint(0, 5, size=(50, 2)).astype(np.uint8)
        grads = rng.randn(50)
        hess = np.ones(50)

        builder = HistogramBuilder(max_bins=255)
        parent = builder.build_histogram(X, grads, hess)
        left = builder.build_histogram(X, grads, hess, sample_indices=np.arange(25))
        right = builder.subtract_histograms(parent, left)

        right_direct = builder.build_histogram(X, grads, hess, sample_indices=np.arange(25, 50))
        np.testing.assert_allclose(right[0], right_direct[0], atol=1e-10)

    def test_v2_adaptive_reg_split(self):
        """Test that adaptive regularization changes split gains."""
        rng = np.random.RandomState(42)
        X = rng.randint(0, 10, size=(100, 2)).astype(np.uint8)
        grads = rng.randn(100)
        hess = np.ones(100)

        builder = HistogramBuilder(max_bins=255)
        g_hist, h_hist, c_hist = builder.build_histogram(X, grads, hess)

        # Without adaptive reg
        _, _, gain_v1, _ = builder.find_best_split(g_hist, h_hist, c_hist)

        # With adaptive reg (higher regularization)
        adaptive_reg = AdaptiveRegularizer(lambda_base=1.0, alpha=2.0, mu=5.0)
        _, _, gain_v2, _ = builder.find_best_split(
            g_hist, h_hist, c_hist,
            adaptive_reg=adaptive_reg, iteration=50, total_iterations=100)

        # Adaptive reg should reduce gain (more regularization)
        assert gain_v2 <= gain_v1 + 1e-10


class TestGOSS:
    def test_sampling(self):
        rng = np.random.RandomState(42)
        grads = rng.randn(1000)
        sampler = GOSSSampler(top_rate=0.2, other_rate=0.1)
        indices, weights = sampler.sample(grads)

        assert len(indices) < 1000
        assert len(indices) > 100
        assert weights[:200].mean() == 1.0


class TestCategoricalEncoder:
    def test_ordered_target_stats(self):
        rng = np.random.RandomState(42)
        X = np.column_stack([
            rng.choice([0, 1, 2], size=100),
            rng.randn(100),
        ])
        y = X[:, 0] + rng.randn(100) * 0.1

        encoder = OrderedTargetEncoder(smoothing=10, random_state=42)
        encoder.fit(X, y, cat_features=[0])
        X_train = encoder.transform_train(X, y)
        X_test = encoder.transform(X)

        assert X_train[:, 0].dtype == np.float64
        assert not np.array_equal(X_train[:, 0], X[:, 0])
        np.testing.assert_allclose(X_train[:, 1], X[:, 1])


# --- Metrics tests ---

class TestMetrics:
    def test_rmse(self):
        assert rmse(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0
        assert rmse(np.array([0, 0]), np.array([1, 1])) == 1.0

    def test_r2(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, y) == 1.0

    def test_auc(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0.1, 0.2, 0.8, 0.9])
        assert auc(y, pred) == 1.0

    def test_accuracy(self):
        y = np.array([0, 1, 1, 0])
        pred = np.array([0.1, 0.9, 0.8, 0.2])
        assert accuracy(y, pred) == 1.0

    def test_ndcg(self):
        rel = np.array([3, 2, 1, 0])
        scores = np.array([3.0, 2.0, 1.0, 0.0])
        assert ndcg_at_k(rel, scores) == pytest.approx(1.0)

    def test_concordance_index(self):
        times = np.array([1, 2, 3, 4])
        events = np.array([1, 1, 1, 1])
        risk = np.array([4.0, 3.0, 2.0, 1.0])
        assert concordance_index(times, events, risk) == 1.0


class TestFinancialMetrics:
    def test_sharpe_ratio(self):
        # Constant positive returns -> high sharpe
        returns = np.full(100, 0.01)
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_sharpe_ratio_zero_vol(self):
        returns = np.zeros(100)
        assert sharpe_ratio(returns) == 0.0

    def test_max_drawdown(self):
        # Simple drawdown: 100 -> 90 -> 95
        cum = np.array([100, 95, 90, 92, 95, 100])
        dd = max_drawdown(cum)
        assert dd == pytest.approx(0.10, abs=1e-6)

    def test_quantile_loss(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        assert quantile_loss(y, pred) == 0.0


# --- Regressor tests ---

class TestPenguinBoostRegressor:
    def test_basic_regression(self):
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1,
                               random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        model = PenguinBoostRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=4,
            random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = r2_score(y_test, preds)
        assert score > 0.5, f"R2 too low: {score}"

    def test_feature_importances(self):
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        model = PenguinBoostRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        imp = model.feature_importances_
        assert len(imp) == 5
        assert abs(imp.sum() - 1.0) < 1e-6

    def test_symmetric_growth(self):
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=30, growth="symmetric", max_depth=4, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.3

    def test_goss(self):
        X, y = make_regression(n_samples=300, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=30, use_goss=True, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.3

    def test_ordered_boosting(self):
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=30, use_ordered_boosting=True, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.3


# --- Classifier tests ---

class TestPenguinBoostClassifier:
    def test_binary_classification(self):
        X, y = make_classification(n_samples=200, n_features=5,
                                   n_informative=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        model = PenguinBoostClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=4,
            random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc = accuracy(y_test, preds)
        assert acc > 0.7, f"Accuracy too low: {acc}"
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass_iris(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        model = PenguinBoostClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=3,
            random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc = accuracy(y_test, preds.astype(float))
        assert acc > 0.7, f"Iris accuracy too low: {acc}"
        assert proba.shape == (len(X_test), 3)

    def test_classes_attribute(self):
        X, y = make_classification(n_samples=100, random_state=42)
        model = PenguinBoostClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert hasattr(model, "classes_")
        assert len(model.classes_) == 2

    def test_get_set_params(self):
        model = PenguinBoostClassifier(n_estimators=50, learning_rate=0.05)
        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["learning_rate"] == 0.05

        model.set_params(n_estimators=100)
        assert model.n_estimators == 100


# --- Ranker tests ---

class TestPenguinBoostRanker:
    def test_basic_ranking(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = rng.randint(0, 4, size=50).astype(float)
        group = [25, 25]

        model = PenguinBoostRanker(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X, y, group=group)
        preds = model.predict(X)
        assert len(preds) == 50


# --- Survival tests ---

class TestPenguinBoostSurvival:
    def test_basic_survival(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        times = rng.exponential(10, size=100)
        events = rng.binomial(1, 0.7, size=100)

        model = PenguinBoostSurvival(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X, times, events)
        risk = model.predict(X)
        assert len(risk) == 100
        assert np.all(risk > 0)


# --- Integration tests ---

class TestIntegration:
    def test_early_stopping(self):
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42)

        model = PenguinBoostRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            early_stopping_rounds=10, random_state=42)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        assert len(model.engine_.trees_) <= 200

    def test_all_growth_strategies(self):
        X, y = make_classification(n_samples=150, n_features=5,
                                   n_informative=3, random_state=42)
        for growth in ["leafwise", "symmetric", "depthwise"]:
            model = PenguinBoostClassifier(
                n_estimators=20, growth=growth, max_depth=3, random_state=42)
            model.fit(X, y)
            preds = model.predict(X)
            acc = np.mean(preds == y)
            assert acc > 0.6, f"Growth={growth}, accuracy={acc}"

    def test_hybrid_goss_ordered(self):
        """Test combining GOSS + Ordered Boosting (PenguinBoost unique)."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=30, use_goss=True, use_ordered_boosting=True,
            random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.2


# ===== v2 Tests =====

class TestAdaptiveRegularization:
    def test_lambda_increases_with_iteration(self):
        reg = AdaptiveRegularizer(lambda_base=1.0, alpha=1.0, mu=0.0)
        lam_early = reg.compute_lambda(100, 0, 100)
        lam_late = reg.compute_lambda(100, 99, 100)
        assert lam_late > lam_early

    def test_lambda_increases_with_small_node(self):
        reg = AdaptiveRegularizer(lambda_base=1.0, alpha=0.0, mu=2.0)
        lam_big = reg.compute_lambda(1000, 0, 100)
        lam_small = reg.compute_lambda(10, 0, 100)
        assert lam_small > lam_big

    def test_gradient_perturber(self):
        rng = np.random.RandomState(42)
        grads = rng.randn(100)
        perturber = GradientPerturber(tau=2.0, eta=0.1)
        perturbed = perturber.perturb(grads, rng)
        # Clipping should be applied
        assert np.all(perturbed <= 2.0 + 1.0)  # tau + noise margin
        # Should not be identical to original
        assert not np.array_equal(grads, perturbed)

    def test_stability_tracker(self):
        tracker = FeatureStabilityTracker(psi=1.0)
        gains = [0.5, 0.6, 0.4, 0.55]
        penalty = tracker.compute_stability_penalty(gains)
        assert penalty > 0
        # Single gain -> no penalty
        assert tracker.compute_stability_penalty([0.5]) == 0.0


class TestDART:
    def test_drop_and_scale(self):
        rng = np.random.RandomState(42)
        mgr = DARTManager(drop_rate=0.5)
        dropped = mgr.sample_drops(10, rng)
        assert len(dropped) <= 10
        scale = mgr.compute_scale_factor()
        assert 0 < scale <= 1.0

    def test_no_trees_no_drop(self):
        rng = np.random.RandomState(42)
        mgr = DARTManager(drop_rate=0.5)
        dropped = mgr.sample_drops(0, rng)
        assert dropped == []
        assert mgr.compute_scale_factor() == 1.0

    def test_skip_drop(self):
        rng = np.random.RandomState(0)  # deterministic
        mgr = DARTManager(drop_rate=0.5, skip_drop=1.0)  # always skip
        dropped = mgr.sample_drops(10, rng)
        assert dropped == []

    def test_dart_regression(self):
        """Full regression with DART enabled."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=30, use_dart=True, dart_drop_rate=0.2,
            random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.2


class TestMonotoneConstraints:
    def test_increasing_valid(self):
        checker = MonotoneConstraintChecker({0: 1, 1: -1})
        assert checker.is_valid_split(0, left_value=1.0, right_value=2.0)
        assert not checker.is_valid_split(0, left_value=2.0, right_value=1.0)

    def test_decreasing_valid(self):
        checker = MonotoneConstraintChecker({0: 1, 1: -1})
        assert checker.is_valid_split(1, left_value=2.0, right_value=1.0)
        assert not checker.is_valid_split(1, left_value=1.0, right_value=2.0)

    def test_unconstrained(self):
        checker = MonotoneConstraintChecker({0: 1})
        # Feature 5 has no constraint -> always valid
        assert checker.is_valid_split(5, left_value=10.0, right_value=1.0)

    def test_monotone_regression(self):
        """Regression with monotone constraints."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = 2 * X[:, 0] - X[:, 1] + rng.randn(200) * 0.1

        model = PenguinBoostRegressor(
            n_estimators=30, max_depth=4,
            monotone_constraints={0: 1, 1: -1},
            random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.2


class TestFinancialModules:
    def test_purged_kfold(self):
        X = np.random.randn(100, 3)
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.02)
        folds = list(pkf.split(X))
        assert len(folds) == 5
        for train_idx, test_idx in folds:
            # No overlap between train and test
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            # Embargo: train indices right after test should be removed
            assert len(train_idx) + len(test_idx) <= 100

    def test_purged_kfold_n_splits(self):
        pkf = PurgedKFold(n_splits=3)
        assert pkf.get_n_splits() == 3

    def test_temporal_regularizer(self):
        pred = np.array([1.0, 2.0, 3.0, 2.5, 2.0])
        reg = TemporalRegularizer(rho=0.5)
        g_temp = reg.compute_temporal_gradient(pred)
        assert len(g_temp) == 5
        penalty = reg.compute_penalty(pred)
        assert penalty > 0

    def test_temporal_regularizer_constant(self):
        pred = np.array([1.0, 1.0, 1.0, 1.0])
        reg = TemporalRegularizer(rho=1.0)
        g_temp = reg.compute_temporal_gradient(pred)
        np.testing.assert_allclose(g_temp, 0.0, atol=1e-10)
        assert reg.compute_penalty(pred) == 0.0

    def test_regime_detector(self):
        rng = np.random.RandomState(42)
        # Low vol period + high vol period
        returns = np.concatenate([
            rng.randn(100) * 0.01,
            rng.randn(100) * 0.10,
        ])
        detector = RegimeDetector(window=20, n_regimes=3)
        detector.fit(returns)
        regimes = detector.predict(returns)
        assert len(regimes) == 200
        assert set(regimes).issubset({0, 1, 2})
        # High vol period should have higher regime values on average
        assert regimes[150:].mean() > regimes[:80].mean()

    def test_temporal_reg_regression(self):
        """Regression with temporal regularization enabled."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=30, use_temporal_reg=True, temporal_rho=0.01,
            random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.2


class TestQuantileObjectives:
    def test_quantile_gradient(self):
        obj = QuantileObjective(alpha=0.5)
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.5, 1.5, 3.5])
        g = obj.gradient(y, pred)
        # y < pred for y=1.0: g = 0.5 - 1 = -0.5
        # y > pred for y=2.0: g = 0.5
        # y < pred for y=3.0: g = 0.5 - 1 = -0.5
        assert g[0] == pytest.approx(-0.5)
        assert g[1] == pytest.approx(0.5)

    def test_quantile_init_score(self):
        obj = QuantileObjective(alpha=0.1)
        y = np.arange(100, dtype=float)
        score = obj.init_score(y)
        assert score == pytest.approx(np.quantile(y, 0.1))

    def test_cvar_gradient(self):
        obj = CVaRObjective(alpha=0.05)
        y = np.array([1.0, 2.0])
        pred = np.array([1.5, 1.0])
        g = obj.gradient(y, pred)
        # y < pred for y=1: indicator=1, g = 1 - 1/0.05 = 1-20 = -19
        # y > pred for y=2: indicator=0, g = 1 - 0 = 1
        assert g[0] == pytest.approx(-19.0)
        assert g[1] == pytest.approx(1.0)

    def test_quantile_regressor(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = X[:, 0] * 2 + rng.randn(200) * 0.5

        model = PenguinBoostQuantileRegressor(
            objective="quantile", alpha=0.5,
            n_estimators=30, max_depth=4, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 200
        # Median prediction should be reasonably close
        assert np.abs(np.mean(preds - y)) < 5.0

    def test_cvar_regressor(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = X[:, 0] + rng.randn(200)

        model = PenguinBoostQuantileRegressor(
            objective="cvar", alpha=0.1,
            n_estimators=30, max_depth=4, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 200


class TestHybridGrowth:
    def test_hybrid_growth_regression(self):
        """Hybrid growth: symmetric -> leafwise."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=30, growth="hybrid", max_depth=6,
            symmetric_depth=3, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.3

    def test_hybrid_growth_classification(self):
        X, y = make_classification(n_samples=200, n_features=5,
                                   n_informative=3, random_state=42)
        model = PenguinBoostClassifier(
            n_estimators=30, growth="hybrid", max_depth=6,
            symmetric_depth=2, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.6


class TestMultiPermutationOrdered:
    def test_multi_permutation_regression(self):
        """Multi-permutation ordered boosting with median aggregation."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=30, use_ordered_boosting=True,
            n_permutations=4, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.2


class TestGradientPerturbation:
    def test_perturbed_regression(self):
        """Regression with gradient perturbation."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=50, use_gradient_perturbation=True,
            gradient_clip_tau=5.0, gradient_noise_eta=0.05,
            random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.2


class TestAdaptiveRegRegression:
    def test_adaptive_reg_regression(self):
        """Regression with adaptive regularization."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=50, use_adaptive_reg=True,
            adaptive_alpha=0.5, adaptive_mu=1.0,
            random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.3


class TestV2FullIntegration:
    def test_all_v2_features_combined(self):
        """Test combining DART + gradient perturbation + adaptive reg + temporal reg."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=42)
        model = PenguinBoostRegressor(
            n_estimators=30, growth="hybrid", max_depth=6,
            symmetric_depth=2,
            use_dart=True, dart_drop_rate=0.1,
            use_gradient_perturbation=True,
            gradient_clip_tau=5.0, gradient_noise_eta=0.02,
            use_adaptive_reg=True,
            adaptive_alpha=0.3, adaptive_mu=0.5,
            use_temporal_reg=True, temporal_rho=0.001,
            random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        # Should still learn something despite all regularizations
        assert r2_score(y, preds) > 0.1

    def test_v2_params_in_get_params(self):
        """Ensure all v2 params are visible via sklearn get_params."""
        model = PenguinBoostRegressor(
            use_dart=True, use_gradient_perturbation=True,
            use_adaptive_reg=True, monotone_constraints={0: 1},
            use_temporal_reg=True, growth="hybrid", symmetric_depth=4)
        params = model.get_params()
        assert params["use_dart"] is True
        assert params["use_gradient_perturbation"] is True
        assert params["use_adaptive_reg"] is True
        assert params["monotone_constraints"] == {0: 1}
        assert params["use_temporal_reg"] is True
        assert params["growth"] == "hybrid"
        assert params["symmetric_depth"] == 4

    def test_version_updated(self):
        import penguinboost
        assert penguinboost.__version__ == "0.3.6"

    def test_quantile_regressor_exported(self):
        from penguinboost import PenguinBoostQuantileRegressor
        model = PenguinBoostQuantileRegressor(alpha=0.1)
        assert model.alpha == 0.1


# ── v3: Feature Neutralization ────────────────────────────────────────────────

class TestFeatureNeutralizer:
    def test_neutralize_reduces_exposure(self):
        """Neutralization should reduce feature exposure towards zero."""
        from penguinboost import FeatureNeutralizer
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5))
        # Predictions correlated with feature 0
        pred = 2.0 * X[:, 0] + 0.1 * rng.standard_normal(200)
        fn = FeatureNeutralizer()
        neutralized = fn.neutralize(pred, X, proportion=1.0)
        # After full neutralization, max feature exposure should be small
        exp_before = fn.feature_exposure(pred, X)
        exp_after = fn.feature_exposure(neutralized, X)
        assert np.max(np.abs(exp_after)) < np.max(np.abs(exp_before))
        assert np.max(np.abs(exp_after)) < 0.05

    def test_neutralize_proportion_zero_is_noop(self):
        """proportion=0 should return predictions unchanged."""
        from penguinboost import FeatureNeutralizer
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 3))
        pred = rng.standard_normal(100)
        fn = FeatureNeutralizer()
        result = fn.neutralize(pred, X, proportion=0.0)
        np.testing.assert_allclose(result, pred)

    def test_neutralize_per_era(self):
        """Per-era neutralization should work for all era samples."""
        from penguinboost import FeatureNeutralizer
        rng = np.random.default_rng(2)
        n = 300
        X = rng.standard_normal((n, 4))
        pred = X[:, 0] + 0.2 * rng.standard_normal(n)
        eras = np.repeat(np.arange(10), 30)
        fn = FeatureNeutralizer()
        result = fn.neutralize(pred, X, proportion=1.0, per_era=True, eras=eras)
        assert result.shape == (n,)
        assert not np.allclose(result, pred)

    def test_max_feature_exposure(self):
        """max_feature_exposure returns scalar float."""
        from penguinboost import FeatureNeutralizer
        rng = np.random.default_rng(3)
        X = rng.standard_normal((100, 5))
        pred = rng.standard_normal(100)
        fn = FeatureNeutralizer()
        mfe = fn.max_feature_exposure(pred, X)
        assert isinstance(mfe, float)
        assert 0.0 <= mfe <= 1.0


class TestOrthogonalGradientProjector:
    def test_project_reduces_feature_alignment(self):
        """Projected gradient should be less correlated with features."""
        from penguinboost import OrthogonalGradientProjector
        rng = np.random.default_rng(4)
        X = rng.standard_normal((200, 5))
        g = X[:, 0] + 0.1 * rng.standard_normal(200)  # gradient aligned with X[:,0]
        proj = OrthogonalGradientProjector(strength=1.0)
        proj.fit(X)
        g_orth = proj.project(g)
        # Correlation between g_orth and X[:,0] should be near zero
        corr_before = float(np.corrcoef(g, X[:, 0])[0, 1])
        corr_after = float(np.corrcoef(g_orth, X[:, 0])[0, 1])
        assert abs(corr_after) < abs(corr_before)
        assert abs(corr_after) < 0.05

    def test_project_raises_without_fit(self):
        """project() without fit() should raise RuntimeError."""
        from penguinboost import OrthogonalGradientProjector
        proj = OrthogonalGradientProjector()
        with pytest.raises(RuntimeError):
            proj.project(np.ones(10))

    def test_strength_zero_is_noop(self):
        """strength=0 should leave gradients unchanged."""
        from penguinboost import OrthogonalGradientProjector
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 4))
        g = rng.standard_normal(100)
        proj = OrthogonalGradientProjector(strength=0.0)
        proj.fit(X)
        g_out = proj.project(g)
        np.testing.assert_allclose(g_out, g)


# ── v3: Era Boosting ──────────────────────────────────────────────────────────

class TestEraBoostingReweighter:
    def _make_era_data(self, n_eras=5, n_per_era=40, seed=0):
        rng = np.random.default_rng(seed)
        n = n_eras * n_per_era
        X = rng.standard_normal((n, 4))
        y = X[:, 0] + 0.5 * rng.standard_normal(n)
        pred = rng.standard_normal(n)
        eras = np.repeat(np.arange(n_eras), n_per_era)
        return pred, y, eras

    def test_hard_era_weights_sum_to_n(self):
        """Sample weights should sum to n_samples."""
        from penguinboost import EraBoostingReweighter
        pred, y, eras = self._make_era_data()
        rew = EraBoostingReweighter(method='hard_era')
        w = rew.compute_sample_weights(pred, y, eras)
        assert w.shape == pred.shape
        np.testing.assert_allclose(w.sum(), len(pred), rtol=1e-6)

    def test_sharpe_reweight_weights_sum_to_n(self):
        from penguinboost import EraBoostingReweighter
        pred, y, eras = self._make_era_data()
        rew = EraBoostingReweighter(method='sharpe_reweight')
        w = rew.compute_sample_weights(pred, y, eras)
        np.testing.assert_allclose(w.sum(), len(pred), rtol=1e-6)

    def test_proportional_weights_sum_to_n(self):
        from penguinboost import EraBoostingReweighter
        pred, y, eras = self._make_era_data()
        rew = EraBoostingReweighter(method='proportional')
        w = rew.compute_sample_weights(pred, y, eras)
        np.testing.assert_allclose(w.sum(), len(pred), rtol=1e-6)

    def test_weights_non_negative(self):
        from penguinboost import EraBoostingReweighter
        pred, y, eras = self._make_era_data()
        for method in ('hard_era', 'sharpe_reweight', 'proportional'):
            rew = EraBoostingReweighter(method=method)
            w = rew.compute_sample_weights(pred, y, eras)
            assert np.all(w >= 0)

    def test_min_era_weight_enforced(self):
        """min_era_weight should prevent any era from being completely starved."""
        from penguinboost import EraBoostingReweighter
        rng = np.random.default_rng(7)
        n_eras = 5
        n_per_era = 40
        eras = np.repeat(np.arange(n_eras), n_per_era)
        y = rng.standard_normal(n_eras * n_per_era)
        # Perfect prediction for era 0 → it would get near-zero weight without floor
        pred = y.copy()
        pred[eras != 0] = rng.standard_normal((eras != 0).sum())
        # With large min_era_weight, all eras should get non-trivial weight
        rew_floor = EraBoostingReweighter(method='hard_era', min_era_weight=0.5)
        w_floor = rew_floor.compute_sample_weights(pred, y, eras)
        # No era should be starved to near zero
        for era in range(n_eras):
            assert w_floor[eras == era].sum() > 0.1
        # Compare: without floor, era 0 (perfect pred) gets much less weight
        rew_no_floor = EraBoostingReweighter(method='hard_era', min_era_weight=0.0)
        w_no_floor = rew_no_floor.compute_sample_weights(pred, y, eras)
        era0_floor = w_floor[eras == 0].sum()
        era0_no_floor = w_no_floor[eras == 0].sum()
        # With floor, era 0 gets more weight than without
        assert era0_floor > era0_no_floor

    def test_era_stats_returns_dict(self):
        from penguinboost import EraBoostingReweighter
        pred, y, eras = self._make_era_data()
        rew = EraBoostingReweighter()
        stats = rew.era_stats(pred, y, eras)
        assert isinstance(stats, dict)
        assert len(stats) == 5
        for v in stats.values():
            assert -1.0 <= v <= 1.0


class TestEraMetrics:
    def test_update_and_mean_corr(self):
        from penguinboost import EraMetrics
        rng = np.random.default_rng(8)
        n = 100
        eras = np.repeat(np.arange(5), 20)
        y = rng.standard_normal(n)
        pred = y + 0.1 * rng.standard_normal(n)
        em = EraMetrics(eras)
        em.update(pred, y)
        mc = em.mean_corr()
        assert isinstance(mc, float)
        assert mc > 0.5  # strong correlation expected

    def test_sharpe_positive_for_good_model(self):
        from penguinboost import EraMetrics
        rng = np.random.default_rng(9)
        n = 200
        eras = np.repeat(np.arange(10), 20)
        y = rng.standard_normal(n)
        pred = y + 0.05 * rng.standard_normal(n)
        em = EraMetrics(eras)
        em.update(pred, y)
        assert em.sharpe() > 0

    def test_worst_era_is_valid_label(self):
        from penguinboost import EraMetrics
        rng = np.random.default_rng(10)
        n = 100
        eras = np.repeat(np.arange(5), 20)
        y = rng.standard_normal(n)
        pred = rng.standard_normal(n)
        em = EraMetrics(eras)
        em.update(pred, y)
        worst = em.worst_era()
        assert worst in np.unique(eras)


# ── v3: Financial Objectives ──────────────────────────────────────────────────

class TestSpearmanObjective:
    def test_gradient_shape(self):
        from penguinboost import SpearmanObjective
        rng = np.random.default_rng(11)
        y = rng.standard_normal(100)
        pred = rng.standard_normal(100)
        obj = SpearmanObjective()
        obj.init_score(y)
        g = obj.gradient(y, pred)
        assert g.shape == (100,)
        assert np.all(np.isfinite(g))

    def test_hessian_ones(self):
        from penguinboost import SpearmanObjective
        rng = np.random.default_rng(12)
        y = rng.standard_normal(50)
        pred = rng.standard_normal(50)
        obj = SpearmanObjective()
        h = obj.hessian(y, pred)
        np.testing.assert_array_equal(h, np.ones(50))

    def test_loss_decreases_with_better_ranking(self):
        from penguinboost import SpearmanObjective
        rng = np.random.default_rng(13)
        y = rng.standard_normal(100)
        obj = SpearmanObjective()
        obj.init_score(y)
        # Perfect rank prediction vs. random
        pred_good = y + 0.01 * rng.standard_normal(100)
        pred_bad = rng.standard_normal(100)
        loss_good = obj.loss(y, pred_good)
        loss_bad = obj.loss(y, pred_bad)
        assert loss_good < loss_bad

    def test_init_score_is_finite(self):
        from penguinboost import SpearmanObjective
        y = np.arange(50, dtype=float)
        obj = SpearmanObjective()
        s = obj.init_score(y)
        assert np.isfinite(s)


class TestMaxSharpeEraObjective:
    def _make_data(self, n_eras=5, n_per_era=40, seed=14):
        rng = np.random.default_rng(seed)
        n = n_eras * n_per_era
        y = rng.standard_normal(n)
        pred = y + 0.5 * rng.standard_normal(n)
        eras = np.repeat(np.arange(n_eras), n_per_era)
        return y, pred, eras

    def test_gradient_shape_with_eras(self):
        from penguinboost import MaxSharpeEraObjective
        y, pred, eras = self._make_data()
        obj = MaxSharpeEraObjective()
        obj.set_era_indices(eras)
        obj.init_score(y)
        g = obj.gradient(y, pred)
        assert g.shape == y.shape
        assert np.all(np.isfinite(g))

    def test_gradient_fallback_no_eras(self):
        """Without era indices, should fall back to Spearman gradient."""
        from penguinboost import MaxSharpeEraObjective
        rng = np.random.default_rng(15)
        y = rng.standard_normal(80)
        pred = rng.standard_normal(80)
        obj = MaxSharpeEraObjective()
        obj.init_score(y)
        g = obj.gradient(y, pred)
        assert g.shape == (80,)
        assert np.all(np.isfinite(g))

    def test_loss_is_negative_sharpe(self):
        from penguinboost import MaxSharpeEraObjective
        y, pred, eras = self._make_data()
        obj = MaxSharpeEraObjective()
        obj.set_era_indices(eras)
        loss = obj.loss(y, pred)
        # Loss is negative Sharpe; good model should have lower (more negative) loss
        assert np.isfinite(loss)

    def test_hessian_positive(self):
        from penguinboost import MaxSharpeEraObjective
        y, pred, eras = self._make_data()
        obj = MaxSharpeEraObjective()
        obj.set_era_indices(eras)
        h = obj.hessian(y, pred)
        assert np.all(h > 0)


class TestFeatureExposurePenalizedObjective:
    def test_gradient_shape(self):
        from penguinboost import FeatureExposurePenalizedObjective, SpearmanObjective
        rng = np.random.default_rng(16)
        n = 100
        X = rng.standard_normal((n, 5))
        y = rng.standard_normal(n)
        pred = rng.standard_normal(n)
        base = SpearmanObjective()
        base.init_score(y)
        obj = FeatureExposurePenalizedObjective(base, X, lambda_fe=0.1)
        g = obj.gradient(y, pred)
        assert g.shape == (n,)
        assert np.all(np.isfinite(g))

    def test_penalty_differs_from_base(self):
        """Penalized gradient should differ from base gradient."""
        from penguinboost import FeatureExposurePenalizedObjective, SpearmanObjective
        rng = np.random.default_rng(17)
        n = 100
        X = rng.standard_normal((n, 5))
        y = rng.standard_normal(n)
        pred = X[:, 0] + 0.1 * rng.standard_normal(n)  # aligned with feature 0
        base = SpearmanObjective()
        base.init_score(y)
        g_base = base.gradient(y, pred)
        obj = FeatureExposurePenalizedObjective(base, X, lambda_fe=1.0)
        g_penalized = obj.gradient(y, pred)
        assert not np.allclose(g_base, g_penalized)

    def test_feature_exposure_method(self):
        """feature_exposure() returns per-feature correlations."""
        from penguinboost import FeatureExposurePenalizedObjective, SpearmanObjective
        rng = np.random.default_rng(18)
        n = 100
        X = rng.standard_normal((n, 5))
        y = rng.standard_normal(n)
        pred = X[:, 2].copy()  # fully aligned with feature 2
        base = SpearmanObjective()
        obj = FeatureExposurePenalizedObjective(base, X, lambda_fe=0.1)
        exposures = obj.feature_exposure(pred)
        assert exposures.shape == (5,)
        assert abs(exposures[2]) > 0.9  # feature 2 should have high exposure


# ── v3: Integration — PenguinBoostRegressor with v3 params ─────────────────────

class TestPenguinBoostRegressorV3:
    def test_v3_params_in_get_params(self):
        """All v3 params visible via sklearn get_params()."""
        model = PenguinBoostRegressor(
            use_orthogonal_gradients=True,
            use_era_boosting=True,
            use_feature_exposure_penalty=True,
        )
        params = model.get_params()
        assert params["use_orthogonal_gradients"] is True
        assert params["use_era_boosting"] is True
        assert params["use_feature_exposure_penalty"] is True

    def test_era_boosting_fit_predict(self):
        """PenguinBoostRegressor with era boosting should fit and predict."""
        from sklearn.metrics import r2_score
        rng = np.random.default_rng(20)
        n_eras, n_per_era = 5, 40
        n = n_eras * n_per_era
        X = rng.standard_normal((n, 6))
        y = X[:, 0] + 0.3 * rng.standard_normal(n)
        eras = np.repeat(np.arange(n_eras), n_per_era)
        model = PenguinBoostRegressor(
            n_estimators=30,
            use_era_boosting=True,
            era_boosting_method='hard_era',
            random_state=42)
        model.fit(X, y, era_indices=eras)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.3

    def test_orthogonal_gradients_fit_predict(self):
        """PenguinBoostRegressor with orthogonal gradient projection should fit."""
        from sklearn.metrics import r2_score
        rng = np.random.default_rng(21)
        n = 200
        X = rng.standard_normal((n, 5))
        y = X[:, 0] + 0.3 * rng.standard_normal(n)
        model = PenguinBoostRegressor(
            n_estimators=30,
            use_orthogonal_gradients=True,
            orthogonal_strength=0.5,
            random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.1

    def test_feature_exposure_penalty_fit_predict(self):
        """PenguinBoostRegressor with feature exposure penalty should fit."""
        from sklearn.metrics import r2_score
        rng = np.random.default_rng(22)
        n = 200
        X = rng.standard_normal((n, 5))
        y = X[:, 0] + 0.3 * rng.standard_normal(n)
        model = PenguinBoostRegressor(
            n_estimators=30,
            use_feature_exposure_penalty=True,
            feature_exposure_lambda=0.1,
            random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert r2_score(y, preds) > 0.1

    def test_neutralize_method(self):
        """neutralize() should reduce feature exposure of predictions."""
        rng = np.random.default_rng(23)
        n = 200
        X = rng.standard_normal((n, 5))
        y = X[:, 0] + 0.3 * rng.standard_normal(n)
        model = PenguinBoostRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        neutralized = model.neutralize(preds, X, proportion=1.0)
        assert neutralized.shape == (n,)
        # Feature exposure should decrease after neutralization
        exp_before = model.feature_exposure(X, preds)
        exp_after = model.feature_exposure(X, neutralized)
        assert np.max(np.abs(exp_after)) < np.max(np.abs(exp_before))

    def test_feature_exposure_method(self):
        """feature_exposure() without predictions uses self.predict(X)."""
        rng = np.random.default_rng(24)
        n = 200
        X = rng.standard_normal((n, 5))
        y = X[:, 0] + 0.3 * rng.standard_normal(n)
        model = PenguinBoostRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        exp = model.feature_exposure(X)
        assert exp.shape == (5,)
        assert np.all(np.abs(exp) <= 1.0 + 1e-9)

    def test_v3_exports(self):
        """All v3 classes should be importable from penguinboost top-level."""
        import penguinboost
        assert hasattr(penguinboost, 'FeatureNeutralizer')
        assert hasattr(penguinboost, 'OrthogonalGradientProjector')
        assert hasattr(penguinboost, 'EraBoostingReweighter')
        assert hasattr(penguinboost, 'EraMetrics')
        assert hasattr(penguinboost, 'SpearmanObjective')
        assert hasattr(penguinboost, 'MaxSharpeEraObjective')
        assert hasattr(penguinboost, 'FeatureExposurePenalizedObjective')


# ── 追加テスト群 ───────────────────────────────────────────────────────────────
# 複雑な動作・相互作用・数値的正確性を検証する

from sklearn.metrics import r2_score as _r2_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. ヒストグラム差分トリックの数値正確性
# ─────────────────────────────────────────────────────────────────────────────

class TestSubtractionTrickCorrectness:
    """Verify that histogram subtraction is numerically identical to direct builds."""

    def _make_binned(self, n=400, f=8, max_bins=15, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, f)
        X[rng.random((n, f)) < 0.08] = np.nan   # ~8 % NaN values
        from penguinboost.core.binning import FeatureBinner
        X_binned = FeatureBinner(max_bins=max_bins).fit_transform(X)
        g = rng.randn(n)
        h = np.abs(rng.randn(n)) + 0.1
        return X_binned, g, h, max_bins

    def test_parent_equals_left_plus_right(self):
        """hist(parent) == hist(left) + hist(right) for any partition."""
        X_binned, g, h, mb = self._make_binned()
        builder = HistogramBuilder(max_bins=mb)

        left_idx  = np.arange(0, 200)
        right_idx = np.arange(200, 400)

        gp, hp, cp = builder.build_histogram(X_binned, g, h)
        gl, hl, cl = builder.build_histogram(X_binned, g, h, left_idx)
        gr, hr, cr = builder.build_histogram(X_binned, g, h, right_idx)

        np.testing.assert_allclose(gp, gl + gr, atol=1e-10,
                                   err_msg="grad_hist: parent ≠ left + right")
        np.testing.assert_allclose(hp, hl + hr, atol=1e-10,
                                   err_msg="hess_hist: parent ≠ left + right")
        np.testing.assert_array_equal(cp, cl + cr)

    def test_subtract_matches_direct_build(self):
        """subtract_histograms(parent, left) is numerically equal to direct right build."""
        X_binned, g, h, mb = self._make_binned(seed=7)
        builder = HistogramBuilder(max_bins=mb)

        rng = np.random.RandomState(7)
        left_idx  = np.sort(rng.choice(400, 180, replace=False))
        right_idx = np.setdiff1d(np.arange(400), left_idx)

        parent_hist     = builder.build_histogram(X_binned, g, h)
        left_hist       = builder.build_histogram(X_binned, g, h, left_idx)
        right_direct    = builder.build_histogram(X_binned, g, h, right_idx)
        right_subtract  = builder.subtract_histograms(parent_hist, left_hist)

        np.testing.assert_allclose(right_subtract[0], right_direct[0], atol=1e-10,
                                   err_msg="grad_hist mismatch after subtraction")
        np.testing.assert_allclose(right_subtract[1], right_direct[1], atol=1e-10,
                                   err_msg="hess_hist mismatch after subtraction")
        np.testing.assert_allclose(right_subtract[2], right_direct[2], atol=1e-10,
                                   err_msg="count_hist mismatch after subtraction")

    def test_subtraction_with_all_nan_feature(self):
        """Column entirely composed of NaN is handled (all samples in NaN bin)."""
        X_binned, g, h, mb = self._make_binned()
        # Force feature 0 to all-NaN bin
        X_binned[:, 0] = mb
        builder = HistogramBuilder(max_bins=mb)

        left_idx  = np.arange(200)
        right_idx = np.arange(200, 400)

        parent_hist    = builder.build_histogram(X_binned, g, h)
        left_hist      = builder.build_histogram(X_binned, g, h, left_idx)
        right_direct   = builder.build_histogram(X_binned, g, h, right_idx)
        right_subtract = builder.subtract_histograms(parent_hist, left_hist)

        np.testing.assert_allclose(right_subtract[0], right_direct[0], atol=1e-10)

    def test_end_to_end_model_quality_preserved(self):
        """The subtraction trick should not degrade model accuracy."""
        X, y = make_regression(n_samples=500, n_features=10, n_informative=5,
                               noise=0.5, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        model = PenguinBoostRegressor(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)
        score = _r2_score(y_te, model.predict(X_te))
        assert score > 0.90, (
            f"Subtraction trick degraded accuracy: R²={score:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. depthwise 成長戦略
# ─────────────────────────────────────────────────────────────────────────────

class TestDepthwiseGrowth:
    """Explicit tests for the depthwise growth strategy (no dedicated class existed)."""

    def test_depthwise_regression_quality(self):
        """depthwise growth should learn the signal."""
        X, y = make_regression(n_samples=500, n_features=10, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        model = PenguinBoostRegressor(
            n_estimators=50, growth="depthwise", max_depth=4, random_state=42)
        model.fit(X_tr, y_tr)
        assert _r2_score(y_te, model.predict(X_te)) > 0.5

    def test_depthwise_stump_single_split_per_tree(self):
        """max_depth=1 with depthwise → each tree has at most 1 split."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=0)
        model = PenguinBoostRegressor(
            n_estimators=10, growth="depthwise", max_depth=1,
            max_leaves=64, random_state=0)
        model.fit(X, y)
        for tree in model.engine_.trees_:
            assert len(tree.split_features_) <= 1, (
                f"max_depth=1 tree has {len(tree.split_features_)} splits")

    def test_depthwise_respects_max_depth(self):
        """depthwise growth must not exceed max_depth in any node."""
        from penguinboost.core.tree import DecisionTree, TreeNode
        from penguinboost.core.binning import FeatureBinner

        rng = np.random.RandomState(0)
        X = rng.randn(200, 5)
        y = X[:, 0] + rng.randn(200) * 0.1
        X_binned = FeatureBinner(max_bins=15).fit_transform(X)
        g = -(y - y.mean())
        h = np.ones(200)

        for max_d in [1, 2, 4]:
            tree = DecisionTree(max_depth=max_d, growth="depthwise")
            tree.build(X_binned, g, h)

            def max_node_depth(node, d=0):
                if node is None or node.left is None:
                    return d
                return max(max_node_depth(node.left, d+1),
                           max_node_depth(node.right, d+1))

            actual_depth = max_node_depth(tree.root)
            assert actual_depth <= max_d, (
                f"max_depth={max_d} violated: tree reached depth {actual_depth}")

    def test_depthwise_same_as_leafwise_for_depth_one(self):
        """Depthwise and leafwise should pick the same best split for depth=1."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=1)

        kw = dict(n_estimators=5, max_depth=1, max_leaves=2, random_state=1)
        m_dw = PenguinBoostRegressor(growth="depthwise", **kw)
        m_lw = PenguinBoostRegressor(growth="leafwise",  **kw)
        m_dw.fit(X, y)
        m_lw.fit(X, y)

        # At depth=1 both strategies make the same greedy split decision
        np.testing.assert_allclose(
            m_dw.predict(X), m_lw.predict(X), rtol=1e-10,
            err_msg="depthwise and leafwise differ at max_depth=1")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAE / Huber 目的関数
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectiveMAEHuber:
    """MAE and Huber objectives are not tested elsewhere."""

    def _data(self, seed=42):
        X, y = make_regression(n_samples=400, n_features=8, noise=0.5,
                               random_state=seed)
        # Normalize y so that MAE/Huber objectives converge within 80 iterations.
        # (Gradient magnitudes for MAE/Huber are bounded, so large-scale targets
        # require normalization for the model to make meaningful progress.)
        y = (y - y.mean()) / (y.std() + 1e-9)
        return train_test_split(X, y, test_size=0.2, random_state=seed)

    def test_mae_objective_learns(self):
        X_tr, X_te, y_tr, y_te = self._data()
        m = PenguinBoostRegressor(
            objective="mae", n_estimators=80, learning_rate=0.1, random_state=42)
        m.fit(X_tr, y_tr)
        assert _r2_score(y_te, m.predict(X_te)) > 0.5

    def test_huber_objective_learns(self):
        X_tr, X_te, y_tr, y_te = self._data()
        m = PenguinBoostRegressor(
            objective="huber", huber_delta=1.0, n_estimators=80,
            learning_rate=0.1, random_state=42)
        m.fit(X_tr, y_tr)
        assert _r2_score(y_te, m.predict(X_te)) > 0.5

    def test_mae_init_score_is_median(self):
        """MAEObjective.init_score should return the median, not the mean."""
        from penguinboost.objectives.regression import MAEObjective
        y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        obj = MAEObjective()
        assert obj.init_score(y) == pytest.approx(np.median(y))
        assert obj.init_score(y) != pytest.approx(np.mean(y))

    def test_huber_gradient_clips_at_delta(self):
        """Huber gradient should be clipped to [-delta, delta] for large residuals."""
        from penguinboost.objectives.regression import HuberObjective
        delta = 2.0
        obj = HuberObjective(delta=delta)
        pred = np.array([0.0])
        y    = np.array([100.0])   # huge residual

        grad = obj.gradient(y, pred)
        assert abs(grad[0]) == pytest.approx(delta)

    def test_huber_equals_mse_for_small_residuals(self):
        """For |residual| < delta, Huber gradient equals MSE gradient."""
        from penguinboost.objectives.regression import HuberObjective, MSEObjective
        delta = 5.0
        obj_h = HuberObjective(delta=delta)
        obj_m = MSEObjective()
        y    = np.array([0.0, 0.5, -0.5])
        pred = np.array([1.0, 2.0, -1.5])   # residuals < delta

        np.testing.assert_allclose(
            obj_h.gradient(y, pred), obj_m.gradient(y, pred), atol=1e-12)

    def test_objectives_produce_different_predictions(self):
        """MSE, MAE, Huber should not all produce identical predictions."""
        rng = np.random.RandomState(0)
        X = rng.randn(300, 5)
        # Target with heavy-tailed noise to create differences between objectives
        y = X[:, 0] + rng.standard_cauchy(300) * 0.5

        kw = dict(n_estimators=50, random_state=0)
        m_mse   = PenguinBoostRegressor(objective="mse",   **kw).fit(X, y)
        m_mae   = PenguinBoostRegressor(objective="mae",   **kw).fit(X, y)
        m_huber = PenguinBoostRegressor(objective="huber", **kw).fit(X, y)

        p_mse   = m_mse.predict(X)
        p_mae   = m_mae.predict(X)
        p_huber = m_huber.predict(X)

        assert not np.allclose(p_mse, p_mae,   atol=1e-3), "MSE == MAE predictions"
        assert not np.allclose(p_mse, p_huber, atol=1e-3), "MSE == Huber predictions"


# ─────────────────────────────────────────────────────────────────────────────
# 4. sklearn 互換性
# ─────────────────────────────────────────────────────────────────────────────

class TestSklearnCompatibility:
    """clone(), Pipeline, cross_val_score, GridSearchCV."""

    def test_clone_regressor(self):
        from sklearn.base import clone
        m = PenguinBoostRegressor(n_estimators=20, learning_rate=0.05, random_state=7)
        c = clone(m)
        assert c.n_estimators  == 20
        assert c.learning_rate == pytest.approx(0.05)
        assert c.random_state  == 7
        assert not hasattr(c, "engine_")   # clone must be unfitted

    def test_clone_classifier(self):
        from sklearn.base import clone
        m = PenguinBoostClassifier(max_depth=3, reg_lambda=2.0)
        c = clone(m)
        assert c.max_depth  == 3
        assert c.reg_lambda == pytest.approx(2.0)
        assert not hasattr(c, "engine_")

    def test_pipeline_regression(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = make_regression(n_samples=200, n_features=8, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

        pipe = Pipeline([
            ("sc",    StandardScaler()),
            ("model", PenguinBoostRegressor(n_estimators=30, random_state=0)),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)

        assert preds.shape == (len(y_te),)
        assert np.isfinite(preds).all()
        assert _r2_score(y_te, preds) > 0.3

    def test_pipeline_classification(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = make_classification(n_samples=200, n_features=8, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

        pipe = Pipeline([
            ("sc",    StandardScaler()),
            ("model", PenguinBoostClassifier(n_estimators=20, random_state=0)),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)

        assert preds.shape == (len(y_te),)
        assert set(preds).issubset(set(y))

    def test_cross_val_score_regressor(self):
        from sklearn.model_selection import cross_val_score
        X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        scores = cross_val_score(
            PenguinBoostRegressor(n_estimators=20, random_state=0),
            X, y, cv=3, scoring="r2")
        assert scores.shape == (3,)
        assert np.all(scores > -1.0), f"Unexpectedly poor CV scores: {scores}"

    def test_gridsearchcv_finds_best_param(self):
        from sklearn.model_selection import GridSearchCV
        X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        gs = GridSearchCV(
            PenguinBoostRegressor(n_estimators=20, random_state=0),
            param_grid={"learning_rate": [0.01, 0.1]},
            cv=2, scoring="r2")
        gs.fit(X, y)
        assert gs.best_params_["learning_rate"] in [0.01, 0.1]
        assert gs.best_score_ > -1.0

    def test_set_params_then_fit(self):
        """set_params should be respected on subsequent fit calls."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        m = PenguinBoostRegressor(n_estimators=5, random_state=0)
        m.set_params(n_estimators=15)
        m.fit(X, y)
        assert m.engine_.best_iteration_ == 14   # 0-indexed → 14 means 15 trees


# ─────────────────────────────────────────────────────────────────────────────
# 5. 再現性
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """Same random_state → identical output; different seeds → different output."""

    def test_same_seed_identical_predictions(self):
        X, y = make_regression(n_samples=200, n_features=10, random_state=0)
        kw = dict(n_estimators=30, random_state=42)
        p1 = PenguinBoostRegressor(**kw).fit(X, y).predict(X)
        p2 = PenguinBoostRegressor(**kw).fit(X, y).predict(X)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ_with_colsample(self):
        """Column subsampling introduces randomness; different seeds must give different models."""
        X, y = make_regression(n_samples=300, n_features=20, random_state=0)
        kw = dict(n_estimators=30, colsample_bytree=0.5)
        p1 = PenguinBoostRegressor(**kw, random_state=1).fit(X, y).predict(X)
        p2 = PenguinBoostRegressor(**kw, random_state=99).fit(X, y).predict(X)
        assert not np.allclose(p1, p2), "Different seeds produced identical models"

    def test_predict_is_idempotent(self):
        """predict() called twice on the same input returns identical arrays."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        m = PenguinBoostRegressor(n_estimators=20, random_state=0).fit(X, y)
        np.testing.assert_array_equal(m.predict(X), m.predict(X))

    def test_classifier_same_seed_identical_proba(self):
        X, y = make_classification(n_samples=200, n_features=8, random_state=0)
        kw = dict(n_estimators=20, random_state=42)
        p1 = PenguinBoostClassifier(**kw).fit(X, y).predict_proba(X)
        p2 = PenguinBoostClassifier(**kw).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 特徴量重要度の性質
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureImportanceProperties:
    """Feature importances must satisfy mathematical invariants."""

    def _fit(self, n_features=10, importance_type="gain", seed=42, **kw):
        X, y = make_regression(n_samples=500, n_features=n_features,
                               n_informative=5, random_state=seed)
        m = PenguinBoostRegressor(
            n_estimators=80, importance_type=importance_type,
            random_state=seed, **kw)
        m.fit(X, y)
        return m, X, y

    def test_gain_importances_sum_to_one(self):
        m, _, _ = self._fit(importance_type="gain")
        assert np.allclose(m.feature_importances_.sum(), 1.0, atol=1e-9)

    def test_split_importances_sum_to_one(self):
        m, _, _ = self._fit(importance_type="split")
        assert np.allclose(m.feature_importances_.sum(), 1.0, atol=1e-9)

    def test_importances_non_negative(self):
        m, _, _ = self._fit()
        assert np.all(m.feature_importances_ >= 0)

    def test_informative_features_rank_high(self):
        """The 5 informative features should dominate the top-6 ranks."""
        rng = np.random.RandomState(0)
        X = rng.randn(600, 10)
        # Strong signal only in features 0..4
        y = X[:, :5].sum(axis=1) + rng.randn(600) * 0.1
        m = PenguinBoostRegressor(n_estimators=100, random_state=0).fit(X, y)
        top5 = set(np.argsort(m.feature_importances_)[-5:])
        # At least 4 of the top-5 should be the informative features
        overlap = len(top5 & set(range(5)))
        assert overlap >= 4, (
            f"Only {overlap}/5 informative features in top-5: {top5}")

    def test_colsample_reduces_nonzero_features(self):
        """With aggressive colsample_bytree and few trees, fewer features should be non-zero.

        With colsample_bytree=0.2 and only 5 trees (6 features/tree from 30),
        P(feature never selected) ≈ 0.8^5 ≈ 33%, so ~10 of 30 features stay at 0.
        With colsample_bytree=1.0, all features are eligible every tree.
        """
        X, y = make_regression(n_samples=500, n_features=30, n_informative=30,
                               random_state=42)
        m_full = PenguinBoostRegressor(
            n_estimators=5, colsample_bytree=1.0, random_state=42).fit(X, y)
        m_low = PenguinBoostRegressor(
            n_estimators=5, colsample_bytree=0.2, random_state=42).fit(X, y)
        n_full = np.sum(m_full.feature_importances_ > 0)
        n_low  = np.sum(m_low.feature_importances_ > 0)
        assert n_low < n_full, (
            f"colsample_bytree=0.2 didn't reduce active features: "
            f"{n_low} vs {n_full}")

    def test_multiclass_importances_sum_to_one(self):
        X, y = make_classification(n_samples=400, n_features=8,
                                   n_classes=3, n_informative=6,
                                   n_clusters_per_class=1, random_state=0)
        m = PenguinBoostClassifier(n_estimators=20, random_state=0).fit(X, y)
        assert np.allclose(m.feature_importances_.sum(), 1.0, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 7. エッジケース
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_stump_max_depth_one(self):
        """max_depth=1 → each tree has at most one split."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=0)
        m = PenguinBoostRegressor(
            n_estimators=10, max_depth=1, max_leaves=2, random_state=0)
        m.fit(X, y)
        for tree in m.engine_.trees_:
            assert len(tree.split_features_) <= 1

    def test_min_child_samples_prevents_all_splits(self):
        """min_child_samples > n/2 must prevent any split from occurring."""
        n = 100
        X, y = make_regression(n_samples=n, n_features=5, random_state=0)
        # Every potential split would leave < 51 samples on one side → impossible
        m = PenguinBoostRegressor(
            n_estimators=5, min_child_samples=51, random_state=0)
        m.fit(X, y)
        for tree in m.engine_.trees_:
            assert len(tree.split_features_) == 0, (
                "Split occurred despite min_child_samples constraint")

    def test_single_feature(self):
        """A model with a single input feature should work end-to-end."""
        rng = np.random.RandomState(0)
        X = rng.randn(300, 1)
        y = X[:, 0] ** 2 + rng.randn(300) * 0.1
        m = PenguinBoostRegressor(n_estimators=50, random_state=0).fit(X, y)
        assert np.isfinite(m.predict(X)).all()
        assert _r2_score(y, m.predict(X)) > 0.5

    def test_constant_target_predicts_constant(self):
        """With a constant target, all predictions should equal that constant."""
        rng = np.random.RandomState(0)
        X = rng.randn(100, 4)
        c = 7.5
        y = np.full(100, c)
        m = PenguinBoostRegressor(n_estimators=5, random_state=0).fit(X, y)
        np.testing.assert_allclose(m.predict(X), c, atol=1e-6)

    def test_single_sample_vs_batch_prediction(self):
        """Predicting one sample at a time must match batch prediction."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        m = PenguinBoostRegressor(n_estimators=20, random_state=0).fit(X, y)
        batch = m.predict(X[:5])
        singles = np.array([m.predict(X[i:i+1])[0] for i in range(5)])
        np.testing.assert_allclose(batch, singles, rtol=1e-10)

    def test_large_reg_lambda_shrinks_predictions(self):
        """Heavy L2 regularization should shrink leaf values toward zero."""
        X, y = make_regression(n_samples=300, n_features=5, noise=5.0, random_state=0)
        kw = dict(n_estimators=50, learning_rate=0.3, random_state=0)
        m_low  = PenguinBoostRegressor(reg_lambda=0.01, **kw).fit(X, y)
        m_high = PenguinBoostRegressor(reg_lambda=1000.0, **kw).fit(X, y)
        # High regularization → predictions closer to base score → smaller variance
        assert m_high.predict(X).std() < m_low.predict(X).std()

    def test_high_reg_alpha_produces_zero_leaf_values(self):
        """Very high L1 penalty (reg_alpha) should force many leaf values to zero."""
        rng = np.random.RandomState(0)
        X = rng.randn(200, 5)
        y = X[:, 0] * 0.01 + rng.randn(200) * 0.001  # very small signal
        m = PenguinBoostRegressor(
            n_estimators=5, reg_alpha=1e6, random_state=0).fit(X, y)
        # With huge L1, gradient must exceed reg_alpha to get non-zero leaf → all zero
        preds = m.predict(X)
        assert np.allclose(preds, preds[0], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 8. 早期終了の正確性
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyStoppingBehavior:

    def test_stops_before_n_estimators(self):
        """early_stopping_rounds should stop training before n_estimators."""
        X, y = make_regression(n_samples=500, n_features=10, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        m = PenguinBoostRegressor(
            n_estimators=200, early_stopping_rounds=10, random_state=0)
        m.fit(X_tr, y_tr, eval_set=(X_te, y_te))
        assert m.engine_.best_iteration_ < 199

    def test_no_early_stopping_uses_all_iterations(self):
        """Without early stopping, best_iteration_ == n_estimators - 1."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        n_est = 25
        m = PenguinBoostRegressor(n_estimators=n_est, random_state=0).fit(X, y)
        assert m.engine_.best_iteration_ == n_est - 1

    def test_eval_set_no_effect_on_predictions_without_early_stop(self):
        """When early stopping is off, eval_set must not alter training predictions."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

        m_no  = PenguinBoostRegressor(n_estimators=20, random_state=0)
        m_ev  = PenguinBoostRegressor(n_estimators=20, random_state=0)
        m_no.fit(X_tr, y_tr)
        m_ev.fit(X_tr, y_tr, eval_set=(X_te, y_te))

        np.testing.assert_allclose(
            m_no.predict(X_tr), m_ev.predict(X_tr), rtol=1e-10,
            err_msg="eval_set changed training predictions without early stopping")

    def test_train_losses_recorded(self):
        """train_losses_ should be recorded for every completed iteration."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=0)
        n_est = 15
        m = PenguinBoostRegressor(n_estimators=n_est, random_state=0).fit(X, y)
        assert len(m.engine_.train_losses_) == n_est
        assert all(np.isfinite(l) for l in m.engine_.train_losses_)

    def test_train_losses_generally_decrease(self):
        """Training loss should trend downward (not necessarily monotone each step)."""
        X, y = make_regression(n_samples=500, n_features=10, random_state=0)
        m = PenguinBoostRegressor(n_estimators=50, learning_rate=0.1,
                                   random_state=0).fit(X, y)
        losses = np.array(m.engine_.train_losses_)
        # First quarter vs last quarter
        assert losses[:10].mean() > losses[-10:].mean()


# ─────────────────────────────────────────────────────────────────────────────
# 9. スレッド制御 API (OpenMP)
# ─────────────────────────────────────────────────────────────────────────────

class TestThreadControl:
    """set_num_threads / get_num_threads correctness and training invariance."""

    def test_get_num_threads_positive_int(self):
        import penguinboost
        n = penguinboost.get_num_threads()
        assert isinstance(n, int) and n >= 1

    def test_set_then_get(self):
        import penguinboost
        original = penguinboost.get_num_threads()
        try:
            penguinboost.set_num_threads(2)
            assert penguinboost.get_num_threads() == 2
        finally:
            penguinboost.set_num_threads(original)

    def test_set_zero_resets_to_max(self):
        """set_num_threads(0) should restore the maximum thread count."""
        import multiprocessing
        import penguinboost
        original = penguinboost.get_num_threads()
        try:
            penguinboost.set_num_threads(1)
            assert penguinboost.get_num_threads() == 1
            penguinboost.set_num_threads(0)   # reset
            assert penguinboost.get_num_threads() == multiprocessing.cpu_count()
        finally:
            penguinboost.set_num_threads(original)

    def test_single_thread_same_predictions(self):
        """Predictions with 1 thread must match default thread count."""
        import penguinboost
        X, y = make_regression(n_samples=300, n_features=10, random_state=0)
        original = penguinboost.get_num_threads()
        try:
            # Train and predict with default threads
            m_default = PenguinBoostRegressor(n_estimators=20, random_state=0)
            m_default.fit(X, y)
            p_default = m_default.predict(X)

            # Train and predict with forced 1 thread
            penguinboost.set_num_threads(1)
            m_1t = PenguinBoostRegressor(n_estimators=20, random_state=0)
            m_1t.fit(X, y)
            p_1t = m_1t.predict(X)
        finally:
            penguinboost.set_num_threads(original)

        # Float-point ordering may differ at epsilon level; use generous tolerance
        np.testing.assert_allclose(
            p_default, p_1t, atol=1e-6,
            err_msg="1-thread result differs from multi-thread result")

    def test_set_get_num_threads_exported(self):
        """set_num_threads and get_num_threads must be importable from top-level."""
        import penguinboost
        assert hasattr(penguinboost, 'set_num_threads')
        assert hasattr(penguinboost, 'get_num_threads')
        assert callable(penguinboost.set_num_threads)
        assert callable(penguinboost.get_num_threads)


# ─────────────────────────────────────────────────────────────────────────────
# New financial-domain feature tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_era_data(n_samples=300, n_features=10, n_eras=5, seed=42):
    """Helper: create regression data with era labels."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = X[:, 0] * 0.5 + rng.randn(n_samples) * 0.3
    eras = np.repeat(np.arange(n_eras), n_samples // n_eras)
    eras = np.concatenate([eras, np.full(n_samples - len(eras), n_eras - 1)])
    return X, y, eras


class TestAsymmetricHuberObjective:
    def test_gradient_sign_overprediction(self):
        """Heavy-tail gradient fires when ŷ >> y (overprediction)."""
        from penguinboost.objectives.regression import AsymmetricHuberObjective
        obj = AsymmetricHuberObjective(delta=1.0, kappa=3.0)
        y = np.zeros(5)
        pred = np.array([0.0, 0.5, 1.5, 3.0, 5.0])   # increasing overprediction
        g = obj.gradient(y, pred)
        # In quadratic zone (r <= delta): g ≈ r
        np.testing.assert_allclose(g[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(g[1], 0.5, atol=1e-9)
        # In tail zone (r > delta=1): g = kappa * delta * sign(r) = 3.0
        np.testing.assert_allclose(g[2], 3.0 * 1.0, atol=1e-9)
        np.testing.assert_allclose(g[3], 3.0 * 1.0, atol=1e-9)
        # All tail gradients equal (linear growth, not quadratic)
        assert g[3] == g[4]

    def test_loss_increases_faster_on_overprediction(self):
        """Loss should grow faster than L2 for large overpredictions."""
        from penguinboost.objectives.regression import AsymmetricHuberObjective, MSEObjective
        obj_ah = AsymmetricHuberObjective(delta=0.5, kappa=5.0)
        obj_mse = MSEObjective()
        y = np.zeros(1)
        pred_large = np.array([3.0])   # large overprediction
        # AsymmetricHuber penalises overprediction harder → loss should differ from MSE
        loss_ah = obj_ah.loss(y, pred_large)
        loss_mse = obj_mse.loss(y, pred_large)
        assert loss_ah != loss_mse   # they should differ

    def test_kappa_must_be_ge_1(self):
        from penguinboost.objectives.regression import AsymmetricHuberObjective
        with pytest.raises(ValueError):
            AsymmetricHuberObjective(delta=1.0, kappa=0.5)

    def test_regressor_asymmetric_huber(self):
        X, y, _ = _make_era_data()
        model = PenguinBoostRegressor(
            objective="asymmetric_huber", huber_delta=1.0, asymmetric_kappa=2.0,
            n_estimators=20)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)


class TestTemporallyWeightedGOSS:
    def test_output_shapes(self):
        from penguinboost.core.sampling import TemporallyWeightedGOSSSampler
        rng = np.random.RandomState(0)
        gradients = rng.randn(200)
        time_idx = np.repeat(np.arange(10), 20)
        sampler = TemporallyWeightedGOSSSampler(top_rate=0.2, other_rate=0.1,
                                                 temporal_decay=0.05)
        indices, weights = sampler.sample(gradients, time_idx)
        assert len(indices) == len(weights)
        assert len(indices) <= len(gradients)
        assert (weights > 0).all()

    def test_recent_bias(self):
        """With high decay, recent samples should dominate the top bucket."""
        from penguinboost.core.sampling import TemporallyWeightedGOSSSampler
        n = 100
        # All gradients equal; selection driven purely by recency
        gradients = np.ones(n)
        time_idx = np.arange(n)    # 0..99, recent = high index
        sampler = TemporallyWeightedGOSSSampler(top_rate=0.1, other_rate=0.05,
                                                 temporal_decay=1.0,
                                                 random_state=0)
        indices, _ = sampler.sample(gradients, time_idx)
        n_top = max(1, int(n * 0.1))
        top_indices = indices[:n_top]
        # With decay=1.0, top samples should be mostly from the last 50%
        assert np.mean(top_indices > n // 2) > 0.5

    def test_tw_goss_in_regressor(self):
        X, y, eras = _make_era_data()
        model = PenguinBoostRegressor(
            use_tw_goss=True, tw_goss_decay=0.05,
            goss_top_rate=0.3, goss_other_rate=0.1,
            n_estimators=20, random_state=0)
        model.fit(X, y, era_indices=eras)
        preds = model.predict(X)
        assert preds.shape == (len(y),)


class TestEraAdaptiveGradientClipper:
    def test_clipping_reduces_outliers(self):
        from penguinboost.core.regularization import EraAdaptiveGradientClipper
        rng = np.random.RandomState(0)
        gradients = rng.randn(100)
        gradients[0] = 1000.0   # extreme outlier in era 0
        eras = np.repeat(np.arange(5), 20)
        clipper = EraAdaptiveGradientClipper(clip_multiplier=3.0)
        clipped = clipper.clip(gradients, eras)
        assert abs(clipped[0]) < abs(gradients[0])   # outlier was clipped

    def test_clip_with_stats(self):
        from penguinboost.core.regularization import EraAdaptiveGradientClipper
        rng = np.random.RandomState(0)
        gradients = rng.randn(100)
        eras = np.repeat(np.arange(5), 20)
        clipper = EraAdaptiveGradientClipper(clip_multiplier=4.0)
        clipped, stats = clipper.clip_with_stats(gradients, eras)
        assert len(stats) == 5
        for era, s in stats.items():
            assert 'mad' in s and 'threshold' in s and 'frac_clipped' in s

    def test_in_regressor(self):
        X, y, eras = _make_era_data()
        model = PenguinBoostRegressor(
            use_era_gradient_clipping=True, era_clip_multiplier=4.0,
            n_estimators=20)
        model.fit(X, y, era_indices=eras)
        preds = model.predict(X)
        assert preds.shape == (len(y),)


class TestEraAwareDARTManager:
    def test_era_variance_recording(self):
        from penguinboost.core.dart import EraAwareDARTManager
        from penguinboost.core.era_boost import _spearman_corr
        mgr = EraAwareDARTManager(drop_rate=0.3, era_var_scale=20.0)
        rng = np.random.RandomState(0)
        n = 100
        y = rng.randn(n)
        eras = np.repeat(np.arange(5), 20)

        for _ in range(3):
            tree_pred = rng.randn(n)
            mgr.record_tree_era_variance(tree_pred, y, eras)

        assert len(mgr._tree_era_vars) == 3
        assert all(v >= 0 for v in mgr._tree_era_vars)

    def test_drop_probabilities_use_era_variance(self):
        from penguinboost.core.dart import EraAwareDARTManager
        mgr = EraAwareDARTManager(drop_rate=0.3, era_var_scale=50.0)
        # Manually inject era variances: first tree stable, second tree unstable
        mgr._tree_era_vars = [0.0, 0.1]
        rng = np.random.RandomState(0)
        drops = [mgr.sample_drops(2, rng) for _ in range(200)]
        # Tree 1 (high variance) should be dropped more often than tree 0
        drop_counts = np.zeros(2)
        for d in drops:
            for idx in d:
                drop_counts[idx] += 1
        assert drop_counts[1] > drop_counts[0]   # unstable tree dropped more

    def test_in_regressor(self):
        X, y, eras = _make_era_data()
        model = PenguinBoostRegressor(
            use_era_aware_dart=True, dart_drop_rate=0.1,
            era_dart_var_scale=20.0, n_estimators=20)
        model.fit(X, y, era_indices=eras)
        preds = model.predict(X)
        assert preds.shape == (len(y),)


class TestMultiTargetAuxiliaryObjective:
    def test_gradient_mixing(self):
        from penguinboost.objectives.multi_target import MultiTargetAuxiliaryObjective
        from penguinboost.objectives.regression import MSEObjective
        rng = np.random.RandomState(0)
        n = 50
        y_main = rng.randn(n)
        Y_aux = rng.randn(n, 3)
        pred = rng.randn(n)

        obj = MultiTargetAuxiliaryObjective(MSEObjective(), alpha=0.7)
        obj.set_aux_targets(Y_aux)

        g_main = MSEObjective().gradient(y_main, pred)
        g_mixed = obj.gradient(y_main, pred)

        # With alpha=0.7, mixed gradient should differ from pure main gradient
        assert not np.allclose(g_main, g_mixed)
        # But they should be correlated (same direction)
        assert np.corrcoef(g_main, g_mixed)[0, 1] > 0.5

    def test_schedule_increases_alpha(self):
        from penguinboost.objectives.multi_target import MultiTargetAuxiliaryObjective
        from penguinboost.objectives.regression import MSEObjective
        rng = np.random.RandomState(0)
        n = 20
        y = rng.randn(n)
        Y_aux = rng.randn(n, 2)
        pred = rng.randn(n)

        obj = MultiTargetAuxiliaryObjective(
            MSEObjective(), alpha=0.9, use_schedule=True,
            alpha_start=0.2, n_estimators=100)
        obj.set_aux_targets(Y_aux)

        obj.set_iteration(0)
        alpha_early = obj._effective_alpha()
        obj.set_iteration(99)
        alpha_late = obj._effective_alpha()
        assert alpha_late > alpha_early

    def test_no_aux_targets_returns_main_gradient(self):
        from penguinboost.objectives.multi_target import MultiTargetAuxiliaryObjective
        from penguinboost.objectives.regression import MSEObjective
        rng = np.random.RandomState(0)
        y = rng.randn(30)
        pred = rng.randn(30)
        obj = MultiTargetAuxiliaryObjective(MSEObjective(), alpha=0.7)
        g_main = MSEObjective().gradient(y, pred)
        g_multi = obj.gradient(y, pred)
        np.testing.assert_array_equal(g_main, g_multi)


class TestConformalPredictor:
    def test_calibration_coverage(self):
        from penguinboost.core.conformal import ConformalPredictor
        rng = np.random.RandomState(42)
        n_cal, n_test = 500, 200
        y_cal = rng.randn(n_cal)
        pred_cal = y_cal + rng.randn(n_cal) * 0.3   # near-perfect predictor

        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(y_cal, pred_cal)

        y_test = rng.randn(n_test)
        pred_test = y_test + rng.randn(n_test) * 0.3
        cov = cp.empirical_coverage(y_test, pred_test)
        # Coverage should be ≥ 1 - alpha (90%) ± some tolerance
        assert cov >= 0.80   # generous lower bound for a small test set

    def test_interval_shape(self):
        from penguinboost.core.conformal import ConformalPredictor
        rng = np.random.RandomState(0)
        cp = ConformalPredictor(alpha=0.05)
        cp.calibrate(rng.randn(100), rng.randn(100))
        pred = rng.randn(50)
        lower, upper = cp.predict_interval(pred)
        assert lower.shape == pred.shape == upper.shape
        assert (upper >= lower).all()

    def test_asymmetric_mode(self):
        from penguinboost.core.conformal import ConformalPredictor
        rng = np.random.RandomState(0)
        y_cal = rng.randn(200)
        pred_cal = y_cal + np.abs(rng.randn(200)) * 0.5   # biased (under)predictor
        cp = ConformalPredictor(alpha=0.1, asymmetric=True)
        cp.calibrate(y_cal, pred_cal)
        # With asymmetric errors, upper and lower quantiles differ
        assert cp._q_upper != cp._q_lower

    def test_era_conformal(self):
        from penguinboost.core.conformal import EraConformalPredictor
        rng = np.random.RandomState(0)
        n = 300
        y = rng.randn(n)
        pred = y + rng.randn(n) * 0.2
        eras = np.repeat(np.arange(10), 30)
        cp = EraConformalPredictor(alpha=0.1, min_era_samples=10)
        cp.calibrate(y, pred, eras)
        lower, upper = cp.predict_interval(pred[:30], eras[:30])
        assert lower.shape == (30,)
        assert (upper >= lower).all()

    def test_not_calibrated_raises(self):
        from penguinboost.core.conformal import ConformalPredictor
        cp = ConformalPredictor()
        with pytest.raises(RuntimeError):
            cp.predict_interval(np.zeros(5))


class TestSharpeEarlyStopping:
    def test_sharpe_es_runs_without_error(self):
        """SR early stopping should run and produce valid predictions."""
        X, y, eras = _make_era_data(n_samples=200, n_eras=5)
        model = PenguinBoostRegressor(
            use_sharpe_early_stopping=True,
            sharpe_es_patience=10,
            n_estimators=50,
            random_state=0)
        model.fit(X, y, era_indices=eras)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert model.engine_.best_iteration_ >= 0


class TestSharpeTreeRegularization:
    def test_sharpe_tree_reg_runs(self):
        X, y, eras = _make_era_data()
        model = PenguinBoostRegressor(
            use_sharpe_tree_reg=True, sharpe_reg_threshold=0.3,
            n_estimators=30, random_state=0)
        model.fit(X, y, era_indices=eras)
        preds = model.predict(X)
        assert preds.shape == (len(y),)


class TestEraAdversarialSplit:
    def test_era_adversarial_split_runs(self):
        X, y, eras = _make_era_data(n_samples=200, n_eras=4)
        model = PenguinBoostRegressor(
            use_era_adversarial_split=True, era_adversarial_beta=0.3,
            n_estimators=15, random_state=0)
        model.fit(X, y, era_indices=eras)
        preds = model.predict(X)
        assert preds.shape == (len(y),)

    def test_era_histogram_builder(self):
        from penguinboost.core.histogram import HistogramBuilder
        rng = np.random.RandomState(0)
        n, p, n_eras = 100, 4, 3
        X = rng.randint(0, 10, (n, p)).astype(np.uint8)
        g = rng.randn(n)
        h = np.ones(n)
        era_ids = (np.arange(n) % n_eras).astype(np.int32)

        builder = HistogramBuilder(max_bins=15)
        era_g, era_h = builder.build_era_histograms(X, g, h, era_ids, n_eras)
        assert era_g.shape == (n_eras, p, 16)   # 15 + 1 NaN bin
        assert era_h.shape == (n_eras, p, 16)

        penalty = builder._era_adversarial_penalty(era_g, era_h, reg_lambda=1.0)
        assert penalty.shape == (2, p, 15)      # (nan_dirs, features, bins)
        assert (penalty >= 0).all()


class TestNeutralizationAwareObjective:
    def test_gradient_shape(self):
        from penguinboost.objectives.corr import NeutralizationAwareObjective
        rng = np.random.RandomState(0)
        n, p = 80, 5
        X = rng.randn(n, p)
        y = rng.randn(n)
        pred = rng.randn(n)

        obj = NeutralizationAwareObjective(X_ref=X, lambda_ridge=1e-3)
        g = obj.gradient(y, pred)
        assert g.shape == (n,)
        assert np.isfinite(g).all()

    def test_loss_is_scalar(self):
        from penguinboost.objectives.corr import NeutralizationAwareObjective
        rng = np.random.RandomState(0)
        n, p = 60, 4
        X = rng.randn(n, p)
        y = rng.randn(n)
        pred = rng.randn(n)
        obj = NeutralizationAwareObjective(X_ref=X)
        loss = obj.loss(y, pred)
        assert isinstance(loss, float)
        assert np.isfinite(loss)
