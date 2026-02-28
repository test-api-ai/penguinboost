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
    """Mean Absolute Error (L1) objective."""

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
        diff = pred - y
        return np.where(np.abs(diff) <= self.delta, 1.0, 0.0).astype(np.float64)

    def loss(self, y, pred):
        diff = np.abs(y - pred)
        return np.mean(np.where(diff <= self.delta,
                                0.5 * diff ** 2,
                                self.delta * (diff - 0.5 * self.delta)))

    def transform(self, pred):
        return pred
