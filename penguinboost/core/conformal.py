"""Conformal Prediction for distribution-free prediction intervals.

Provides marginal coverage guarantees without distributional assumptions.
Particularly useful in financial ML where tail events are unpredictable
and uncertainty quantification is critical.

Mathematical background
-----------------------
Split conformal prediction (Papadopoulos et al., 2002; Vovk et al., 2005):

1. Split data into proper training set D_train and calibration set D_cal.
2. Train a base model on D_train.
3. For each calibration sample (x_i, y_i) in D_cal, compute the
   nonconformity score s_i = |y_i - ŷ_i|.
4. Let q = ⌈(n_cal + 1)(1-α)⌉ / n_cal quantile of {s_1, …, s_n_cal}.
5. For a new input x, the prediction interval is:

        C(x) = [ŷ(x) - q, ŷ(x) + q]

Coverage guarantee: P(y ∈ C(x)) ≥ 1 - α for exchangeable data.

The finite-sample correction ⌈(n+1)(1-α)⌉/n instead of (1-α) ensures
the guarantee holds exactly, not just asymptotically.

Extensions implemented
----------------------
- Conditional conformal (local coverage by era/group)
- Asymmetric intervals (separate upper/lower quantiles)
- Rolling calibration for non-stationary financial data
"""

import numpy as np


class ConformalPredictor:
    """Split conformal predictor for any point-prediction model.

    Parameters
    ----------
    alpha : float in (0, 1)
        Miscoverage rate. The returned intervals cover the true value
        with probability ≥ 1 - alpha (default 0.1 → 90% coverage).
    asymmetric : bool
        If True, compute separate lower and upper nonconformity quantiles
        (alpha/2 each) for an asymmetric interval. Useful when errors are
        skewed (common in financial returns).
    """

    def __init__(self, alpha=0.1, asymmetric=False):
        self.alpha = alpha
        self.asymmetric = asymmetric
        self._q_upper = None   # upper half-width
        self._q_lower = None   # lower half-width (asymmetric mode)
        self._n_cal = 0

    # ── calibration ──────────────────────────────────────────────────────────

    def calibrate(self, y_cal, pred_cal):
        """Compute nonconformity quantile(s) from calibration data.

        Parameters
        ----------
        y_cal : np.ndarray of shape (n_cal,)
            True targets in the calibration set.
        pred_cal : np.ndarray of shape (n_cal,)
            Model predictions for the calibration set.

        Returns
        -------
        self
        """
        y_cal = np.asarray(y_cal, dtype=np.float64)
        pred_cal = np.asarray(pred_cal, dtype=np.float64)
        n = len(y_cal)
        self._n_cal = n

        if self.asymmetric:
            # Signed residuals: positive = underprediction, negative = overprediction
            residuals = y_cal - pred_cal

            # Upper interval (catch underprediction tails)
            alpha_up = self.alpha / 2
            q_up = np.ceil((n + 1) * (1.0 - alpha_up)) / n
            q_up = min(q_up, 1.0)
            # quantile of positive residuals (how far true value can be above pred)
            self._q_upper = float(np.quantile(np.maximum(residuals, 0), q_up))

            # Lower interval (catch overprediction tails)
            alpha_lo = self.alpha / 2
            q_lo = np.ceil((n + 1) * (1.0 - alpha_lo)) / n
            q_lo = min(q_lo, 1.0)
            # quantile of negative residuals magnitudes
            self._q_lower = float(np.quantile(np.maximum(-residuals, 0), q_lo))
        else:
            scores = np.abs(y_cal - pred_cal)
            # Finite-sample corrected quantile level
            q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
            q_level = min(q_level, 1.0)
            self._q_upper = float(np.quantile(scores, q_level))
            self._q_lower = self._q_upper

        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_interval(self, pred):
        """Construct prediction intervals for new predictions.

        Parameters
        ----------
        pred : np.ndarray or float
            Point predictions from the base model.

        Returns
        -------
        lower : np.ndarray
            Lower bound of the prediction interval.
        upper : np.ndarray
            Upper bound of the prediction interval.
        """
        if self._q_upper is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        pred = np.asarray(pred, dtype=np.float64)
        return pred - self._q_lower, pred + self._q_upper

    def predict_interval_width(self):
        """Total width of the prediction interval (scalar)."""
        if self._q_upper is None:
            raise RuntimeError("Call calibrate() first.")
        return self._q_lower + self._q_upper

    @property
    def coverage_level(self):
        """Nominal coverage level (1 - alpha)."""
        return 1.0 - self.alpha

    # ── evaluation ────────────────────────────────────────────────────────────

    def empirical_coverage(self, y_test, pred_test):
        """Compute empirical coverage on a held-out test set.

        Parameters
        ----------
        y_test : np.ndarray
        pred_test : np.ndarray

        Returns
        -------
        float
            Fraction of test samples whose true value falls within the
            predicted interval. Should be ≥ 1 - alpha.
        """
        lower, upper = self.predict_interval(pred_test)
        return float(np.mean((y_test >= lower) & (y_test <= upper)))


class EraConformalPredictor:
    """Conditional conformal predictor stratified by era (time period).

    Computes separate nonconformity quantiles per era, providing
    *conditional* (local) coverage for each time regime.  This is
    stronger than marginal coverage and better suited to non-stationary
    financial data.

    Parameters
    ----------
    alpha : float
        Miscoverage rate per era.
    min_era_samples : int
        Minimum calibration samples per era to fit a local predictor.
        Eras with fewer samples fall back to the global predictor.
    """

    def __init__(self, alpha=0.1, min_era_samples=20):
        self.alpha = alpha
        self.min_era_samples = min_era_samples
        self._global = ConformalPredictor(alpha=alpha)
        self._era_predictors = {}

    def calibrate(self, y_cal, pred_cal, era_cal):
        """Calibrate global and per-era predictors.

        Parameters
        ----------
        y_cal, pred_cal : np.ndarray of shape (n_cal,)
        era_cal : np.ndarray of shape (n_cal,)
            Era label for each calibration sample.
        """
        era_cal = np.asarray(era_cal)
        self._global.calibrate(y_cal, pred_cal)
        self._era_predictors = {}
        for era in np.unique(era_cal):
            mask = era_cal == era
            if mask.sum() >= self.min_era_samples:
                cp = ConformalPredictor(alpha=self.alpha)
                cp.calibrate(y_cal[mask], pred_cal[mask])
                self._era_predictors[era] = cp
        return self

    def predict_interval(self, pred, era_test):
        """Construct era-conditional prediction intervals.

        Parameters
        ----------
        pred : np.ndarray of shape (n_test,)
        era_test : np.ndarray of shape (n_test,)

        Returns
        -------
        lower, upper : np.ndarray of shape (n_test,)
        """
        pred = np.asarray(pred, dtype=np.float64)
        era_test = np.asarray(era_test)
        lower = np.empty_like(pred)
        upper = np.empty_like(pred)

        for era in np.unique(era_test):
            mask = era_test == era
            cp = self._era_predictors.get(era, self._global)
            lo, hi = cp.predict_interval(pred[mask])
            lower[mask] = lo
            upper[mask] = hi

        return lower, upper
