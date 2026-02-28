"""分位点回帰および CVaR 目的関数。"""

import numpy as np


class QuantileObjective:
    """Quantile (pinball) loss objective.

    Loss: L(y, f) = alpha * max(y - f, 0) + (1 - alpha) * max(f - y, 0)
    Gradient: g_i = alpha - 1{y_i < pred_i}
    Hessian: h_i = 1.0 (constant, as pinball loss is piecewise linear)

    Parameters
    ----------
    alpha : float
        Quantile level, between 0 and 1. E.g., 0.05 for 5th percentile (VaR).
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def init_score(self, y):
        """Initialize with the sample quantile."""
        return float(np.quantile(y, self.alpha))

    def gradient(self, y, pred):
        """Compute gradient of quantile loss."""
        return np.where(y < pred, self.alpha - 1.0, self.alpha)

    def hessian(self, y, pred):
        """Constant hessian (second derivative of piecewise linear is 0, use 1 for stability)."""
        return np.ones_like(y)

    def loss(self, y, pred):
        """Compute quantile (pinball) loss."""
        residual = y - pred
        return np.mean(np.where(residual >= 0,
                                self.alpha * residual,
                                (self.alpha - 1.0) * residual))

    def transform(self, raw_predictions):
        """No transformation needed for quantile regression."""
        return raw_predictions


class CVaRObjective:
    """Conditional Value-at-Risk (CVaR / Expected Shortfall) objective.

    Based on the Rockafellar-Uryasev formulation.
    Gradient: g_i = 1 - 1{y_i < pred_i} / alpha
    Hessian: h_i = 1.0

    Parameters
    ----------
    alpha : float
        Tail probability level (e.g., 0.05 for 5% CVaR).
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def init_score(self, y):
        """Initialize with the sample quantile at alpha."""
        return float(np.quantile(y, self.alpha))

    def gradient(self, y, pred):
        """Compute CVaR gradient (Rockafellar-Uryasev)."""
        indicator = (y < pred).astype(np.float64)
        return 1.0 - indicator / self.alpha

    def hessian(self, y, pred):
        """Constant hessian for stability."""
        return np.ones_like(y)

    def loss(self, y, pred):
        """Approximate CVaR loss."""
        shortfall = np.maximum(pred - y, 0)
        return pred.mean() + np.mean(shortfall) / self.alpha

    def transform(self, raw_predictions):
        """No transformation needed."""
        return raw_predictions
