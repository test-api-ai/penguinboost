"""Financial-specific tools: Purged K-Fold CV, Temporal Regularization, Regime Detection."""

import numpy as np


class PurgedKFold:
    """Purged K-Fold cross-validation for time-series financial data.

    Removes samples from the training set that are within `embargo_pct`
    of the test set boundaries, preventing information leakage through
    overlapping time windows.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    embargo_pct : float
        Fraction of total samples to embargo after each test fold.
    """

    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        """Generate purged train/test indices.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (assumed sorted by time).
        y : ignored
        groups : ignored

        Yields
        ------
        train_indices, test_indices : np.ndarray, np.ndarray
        """
        n_samples = X.shape[0]
        embargo_size = int(n_samples * self.embargo_pct)
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)

            # エンバーゴゾーン：テストセット直後のサンプルを除外
            embargo_end = min(test_end + embargo_size, n_samples)

            train_indices = np.concatenate([
                np.arange(0, test_start),
                np.arange(embargo_end, n_samples),
            ])
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class TemporalRegularizer:
    """Temporal regularization penalizing rapid prediction changes over time.

    Adds penalty: Omega_temporal = rho * sum_t (F(x_t) - F(x_{t-1}))^2
    Additional gradient contribution:
        g_temp[t] = 2*rho * [2*F_t - F_{t-1} - F_{t+1}]

    Parameters
    ----------
    rho : float
        Temporal smoothness penalty strength.
    """

    def __init__(self, rho=0.1):
        self.rho = rho

    def compute_temporal_gradient(self, predictions):
        """Compute additional gradient from temporal regularization.

        Parameters
        ----------
        predictions : np.ndarray of shape (n_samples,)
            Current predictions, assumed sorted by time.

        Returns
        -------
        np.ndarray
            Additional gradient component for temporal smoothness.
        """
        n = len(predictions)
        if n < 2:
            return np.zeros(n)

        g_temp = np.zeros(n)

        # 内部点: g[t] = 2*rho*(2*F[t] - F[t-1] - F[t+1])
        if n >= 3:
            g_temp[1:-1] = 2.0 * self.rho * (
                2.0 * predictions[1:-1] - predictions[:-2] - predictions[2:]
            )

        # 境界点
        g_temp[0] = 2.0 * self.rho * (predictions[0] - predictions[1])
        g_temp[-1] = 2.0 * self.rho * (predictions[-1] - predictions[-2])

        return g_temp

    def compute_penalty(self, predictions):
        """Compute the temporal penalty value.

        Parameters
        ----------
        predictions : np.ndarray
            Current predictions sorted by time.

        Returns
        -------
        float
            rho * sum of squared consecutive differences.
        """
        if len(predictions) < 2:
            return 0.0
        diffs = np.diff(predictions)
        return self.rho * np.sum(diffs ** 2)


class RegimeDetector:
    """Simple regime detection using rolling volatility.

    Classifies time periods into volatility regimes (low/medium/high)
    based on rolling standard deviation of returns.

    Parameters
    ----------
    window : int
        Rolling window size for volatility estimation.
    n_regimes : int
        Number of regimes to detect (using quantile thresholds).
    """

    def __init__(self, window=20, n_regimes=3):
        self.window = window
        self.n_regimes = n_regimes
        self.thresholds_ = None

    def fit(self, returns):
        """Fit regime detector on historical returns.

        Parameters
        ----------
        returns : np.ndarray
            Time series of returns.

        Returns
        -------
        self
        """
        vol = self._rolling_std(returns)
        # Compute quantile thresholds
        valid = vol[~np.isnan(vol)]
        if len(valid) == 0:
            self.thresholds_ = np.zeros(self.n_regimes - 1)
            return self
        # 分位数閾値を計算
        quantiles = np.linspace(0, 1, self.n_regimes + 1)[1:-1]
        self.thresholds_ = np.quantile(valid, quantiles)
        return self

    def predict(self, returns):
        """Assign regime labels (0 = low vol, n_regimes-1 = high vol).

        Parameters
        ----------
        returns : np.ndarray
            Time series of returns.

        Returns
        -------
        np.ndarray of int
            Regime labels for each time step.
        """
        vol = self._rolling_std(returns)
        regimes = np.zeros(len(returns), dtype=np.int32)
        for i, threshold in enumerate(self.thresholds_):
            regimes[vol > threshold] = i + 1
        return regimes

    def _rolling_std(self, x):
        """Compute rolling standard deviation."""
        n = len(x)
        result = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            result[i] = np.std(x[i - self.window + 1:i + 1])
        return result
