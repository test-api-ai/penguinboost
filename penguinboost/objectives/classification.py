"""Classification objective functions: Binary Logloss, Multiclass Softmax."""

import numpy as np


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class BinaryLoglossObjective:
    """Binary cross-entropy (logloss) objective."""

    def init_score(self, y):
        p = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))  # log-odds

    def gradient(self, y, pred):
        p = _sigmoid(pred)
        return p - y

    def hessian(self, y, pred):
        p = _sigmoid(pred)
        return np.maximum(p * (1 - p), 1e-7)

    def loss(self, y, pred):
        p = _sigmoid(pred)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def transform(self, pred):
        return _sigmoid(pred)


class SoftmaxObjective:
    """Multiclass softmax (cross-entropy) objective.

    This handles K classes by treating pred as shape (n_samples, n_classes).
    For the boosting engine, gradients/hessians are flattened and we build
    one tree per class per iteration.
    """

    def __init__(self, n_classes=None):
        self.n_classes = n_classes

    def init_score(self, y):
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        # Return class prior log-probabilities
        counts = np.bincount(y.astype(int), minlength=self.n_classes)
        priors = np.clip(counts / len(y), 1e-7, 1.0)
        return np.log(priors)

    def softmax(self, pred):
        """pred: (n_samples, n_classes)"""
        exp_pred = np.exp(pred - pred.max(axis=1, keepdims=True))
        return exp_pred / exp_pred.sum(axis=1, keepdims=True)

    def gradient(self, y, pred):
        """Returns (n_samples, n_classes) gradient."""
        probs = self.softmax(pred)
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y.astype(int)] = 1
        return probs - y_onehot

    def hessian(self, y, pred):
        """Returns (n_samples, n_classes) hessian (diagonal approximation)."""
        probs = self.softmax(pred)
        return np.maximum(probs * (1 - probs), 1e-7)

    def loss(self, y, pred):
        probs = self.softmax(pred)
        probs = np.clip(probs, 1e-7, 1.0)
        return -np.mean(np.log(probs[np.arange(len(y)), y.astype(int)]))

    def transform(self, pred):
        return self.softmax(pred)
