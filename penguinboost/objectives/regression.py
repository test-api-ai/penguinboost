"""Regression objective functions: MSE, MAE, Huber."""

import numpy as np


class MSEObjective:
    """Mean Squared Error (L2) objective."""

    def init_score(self, y):
        return np.mean(y)

    def gradient(self, y, pred):
        return pred - y

    def hessian(self, y, pred):
        return np.ones_like(y, dtype=np.float64)

    def loss(self, y, pred):
        return np.mean((y - pred) ** 2)

    def transform(self, pred):
        return pred


class MAEObjective:
    """Mean Absolute Error (L1) objective.

    Gradient / hessian are used only for split-finding (the same as LightGBM).
    Leaf values are computed via ``compute_leaf_value`` (weighted median of
    residuals), which is the true L1-optimal leaf update and avoids the ±1
    saturation that occurs when the Newton step is used with constant hessian.
    """

    def init_score(self, y):
        return np.median(y)

    def gradient(self, y, pred):
        return np.sign(pred - y)

    def hessian(self, y, pred):
        return np.ones_like(y, dtype=np.float64)

    def loss(self, y, pred):
        return np.mean(np.abs(y - pred))

    def transform(self, pred):
        return pred

    @staticmethod
    def compute_leaf_value(residuals):
        """Optimal L1 leaf value: median of (y - pred) for samples in leaf."""
        return float(np.median(residuals))


class HuberObjective:
    """Huber loss objective (smooth transition between L1 and L2)."""

    def __init__(self, delta=1.0):
        self.delta = delta

    def init_score(self, y):
        return np.median(y)

    def gradient(self, y, pred):
        diff = pred - y
        grad = np.where(np.abs(diff) <= self.delta, diff,
                        self.delta * np.sign(diff))
        return grad

    def hessian(self, y, pred):
        # Use delta / |r| in the linear region instead of 0.0.
        # This keeps split-gain computation numerically stable and gives the
        # IRLS-equivalent hessian that produces sensible leaf values even when
        # all samples are in the linear region.
        diff = pred - y
        linear_h = self.delta / (np.abs(diff) + 1e-8)
        return np.where(np.abs(diff) <= self.delta, 1.0, linear_h).astype(np.float64)

    def loss(self, y, pred):
        diff = np.abs(y - pred)
        return np.mean(np.where(diff <= self.delta,
                                0.5 * diff ** 2,
                                self.delta * (diff - 0.5 * self.delta)))

    def transform(self, pred):
        return pred

    def compute_leaf_value(self, residuals):
        """Approximate optimal Huber leaf value: mean of residuals.

        For the quadratic region this is exact (same as MSE optimal).
        For the linear region it is a reasonable first-order approximation
        that avoids the ±delta saturation of the constant-hessian Newton step.
        """
        return float(np.mean(residuals))


class AsymmetricHuberObjective:
    """Asymmetric Huber loss for financial return prediction.

    Applies heavier-than-quadratic penalty when the model *over*predicts
    returns (residual r = ŷ - y > δ), matching the asymmetric risk profile
    of financial portfolios where being too optimistic is more costly.

    Mathematical definition
    -----------------------
    Let r = ŷ - y (signed residual, positive = overprediction).

        L(r) = { ½ r²                          if  r ≤ δ
               { κ · (δ|r| - ½δ²)             if  r > δ

    where κ > 1 amplifies the penalty for overprediction and δ is the
    boundary between the quadratic and linear regimes.

    Gradient (∂L/∂ŷ)
    -----------------
        g = r                 if r ≤ δ   (standard MSE gradient)
        g = κ · δ · sign(r)  if r > δ   (clipped, scaled by κ)

    Since r > δ > 0 in the tail, sign(r) = +1, so g = +κδ (pushes ŷ down).

    Hessian (∂²L/∂ŷ²)
    ------------------
        h = 1.0  (quadratic region)
        h = 1e-3 (linear region – small but positive for tree stability)

    Parameters
    ----------
    delta : float
        Huber boundary between quadratic and linear regime (default 1.0).
    kappa : float
        Downside penalty multiplier κ > 1 for overprediction (default 2.0).
    """

    def __init__(self, delta=1.0, kappa=2.0):
        if kappa < 1.0:
            raise ValueError(f"kappa must be >= 1.0, got {kappa}")
        self.delta = delta
        self.kappa = kappa

    def init_score(self, y):
        return float(np.median(y))

    def gradient(self, y, pred):
        """Asymmetric Huber gradient.

        Quadratic for r ≤ δ (both slight over- and underprediction treated
        equally); scaled linear slope κδ for r > δ (heavy overprediction).
        """
        r = pred - y
        in_quadratic = r <= self.delta
        g = np.where(in_quadratic, r, self.kappa * self.delta * np.sign(r))
        return g

    def hessian(self, y, pred):
        r = pred - y
        return np.where(r <= self.delta, 1.0, 1e-3).astype(np.float64)

    def loss(self, y, pred):
        r = pred - y
        in_quadratic = r <= self.delta
        return float(np.mean(np.where(
            in_quadratic,
            0.5 * r ** 2,
            self.kappa * (self.delta * np.abs(r) - 0.5 * self.delta ** 2),
        )))

    def transform(self, pred):
        return pred
