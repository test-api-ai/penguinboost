"""適応的正則化、勾配摂動、安定性ペナルティ。"""

import numpy as np


class AdaptiveRegularizer:
    """Bayesian adaptive regularization with schedule and child-node penalty.

    Computes per-node lambda values:
        lambda_node = lambda_base * (1 + alpha * t/T) + mu / sqrt(n_node)

    Parameters
    ----------
    lambda_base : float
        Base L2 regularization strength.
    alpha : float
        Schedule scaling factor for iteration-dependent regularization.
    mu : float
        Bayesian child-node penalty coefficient (scales with 1/sqrt(n)).
    """

    def __init__(self, lambda_base=1.0, alpha=0.5, mu=1.0):
        self.lambda_base = lambda_base
        self.alpha = alpha
        self.mu = mu

    def compute_lambda(self, n_samples, iteration, total_iterations):
        """Compute adaptive lambda for a node.

        Parameters
        ----------
        n_samples : int
            Number of samples in the node.
        iteration : int
            Current boosting iteration (0-indexed).
        total_iterations : int
            Total planned boosting iterations.

        Returns
        -------
        float
            Adaptive lambda value.
        """
        schedule = 1.0 + self.alpha * iteration / max(total_iterations, 1)
        bayesian_penalty = self.mu / np.sqrt(np.maximum(n_samples, 1))
        return self.lambda_base * schedule + bayesian_penalty


class GradientPerturber:
    """Gradient perturbation for implicit regularization.

    Applies clipping and Gaussian noise to gradients:
        g_perturbed = clip(g, -tau, tau) + epsilon
        epsilon ~ N(0, (eta * std(g))^2)

    Parameters
    ----------
    tau : float
        Clipping threshold for gradients.
    eta : float
        Noise scale relative to gradient standard deviation.
    """

    def __init__(self, tau=5.0, eta=0.1):
        self.tau = tau
        self.eta = eta

    def perturb(self, gradients, rng):
        """Apply clipping and Gaussian noise to gradients.

        Parameters
        ----------
        gradients : np.ndarray
            Raw gradient values.
        rng : np.random.RandomState
            Random state for reproducibility.

        Returns
        -------
        np.ndarray
            Perturbed gradients.
        """
        clipped = np.clip(gradients, -self.tau, self.tau)
        std_g = np.std(gradients)
        if std_g > 0:
            noise = rng.normal(0, self.eta * std_g, size=gradients.shape)
        else:
            noise = 0.0
        return clipped + noise


class FeatureStabilityTracker:
    """Tracks gain variance across permutations for stability penalty.

    Computes Var(gain) across K permutation-based gain estimates,
    used as an additive penalty in the split gain formula:
        Gain_v2 = ... - psi * Var(gain)

    Parameters
    ----------
    psi : float
        Stability penalty coefficient.
    n_permutations : int
        Number of permutations (K) for variance estimation.
    """

    def __init__(self, psi=0.5, n_permutations=4):
        self.psi = psi
        self.n_permutations = n_permutations

    def compute_stability_penalty(self, gains):
        """Compute stability penalty from multiple gain estimates.

        Parameters
        ----------
        gains : array-like
            Gain values from K permutations.

        Returns
        -------
        float
            psi * Var(gains), the stability penalty.
        """
        if len(gains) < 2:
            return 0.0
        return self.psi * np.var(gains)
