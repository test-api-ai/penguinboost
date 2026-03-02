"""分位点回帰および CVaR 目的関数。"""

import numpy as np


class QuantileObjective:
    """Quantile (pinball) loss objective.

    Loss: L(y, f) = alpha * max(y - f, 0) + (1 - alpha) * max(f - y, 0)

    Gradient ∂L/∂f:
      f > y  →  +(1 - alpha)   [over-prediction: push f down]
      f < y  →  -alpha          [under-prediction: push f up]
      f == y →  -alpha (limiting value)

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
        """Compute gradient of quantile loss.

        Using XGBoost/LightGBM sign convention: gradient = ∂L/∂pred.
        Leaf value formula is -G/(H + λ), so positive gradient → negative leaf
        (pushes prediction down), negative gradient → positive leaf (pushes up).
        """
        return np.where(y < pred, 1.0 - self.alpha, -self.alpha)

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

    def compute_leaf_value(self, residuals):
        """Optimal quantile leaf value: alpha-quantile of (y - pred) in leaf.

        argmin_c  Σ [α·max(r_i−c,0) + (1−α)·max(c−r_i,0)]  =  quantile(r, α)

        The constant-hessian Newton step −G/(H+λ) is bounded in [−1,1] and
        cannot represent the true scale of the target; this direct quantile
        estimate gives the correct optimal leaf correction.
        """
        return float(np.quantile(residuals, self.alpha))


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
