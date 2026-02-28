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
        assert penguinboost.__version__ == "0.3.0"

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
