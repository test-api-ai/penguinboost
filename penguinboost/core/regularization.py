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


class EraAdaptiveGradientClipper:
    """Adaptive gradient clipping using per-era Median Absolute Deviation.

    Standard gradient clipping uses a fixed global threshold, which fails
    in financial data where each era (time period) has its own volatility
    regime. This clipper computes the MAD within each era independently
    and clips each sample relative to its era's local scale:

        g_i^{clipped} = sign(g_i) * min(|g_i|, c * MAD_{era(i)})

    where MAD_{era}(g) = median(|g_i - median(g_i)|) for all i in the era.

    This prevents fat-tail samples in high-volatility eras from dominating
    the gradient signal while preserving within-era gradient structure.

    Parameters
    ----------
    clip_multiplier : float
        Number of MADs to use as the clipping threshold (c).
        Recommended range 3.0–6.0. Larger values → softer clipping.
    min_mad : float
        Minimum MAD to avoid division by zero in flat regions.
    """

    def __init__(self, clip_multiplier=4.0, min_mad=1e-8):
        self.clip_multiplier = clip_multiplier
        self.min_mad = min_mad

    def clip(self, gradients, era_indices):
        """Apply per-era MAD-based gradient clipping.

        Parameters
        ----------
        gradients : np.ndarray of shape (n_samples,)
            Raw gradient values from the objective function.
        era_indices : np.ndarray of shape (n_samples,)
            Era label for each sample.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Clipped gradients. Samples outside the era-adaptive threshold
            are hard-clipped to ±(c * MAD_era).
        """
        clipped = gradients.copy()
        for era in np.unique(era_indices):
            mask = era_indices == era
            g_era = gradients[mask]
            if len(g_era) < 2:
                continue
            med = np.median(g_era)
            mad = np.median(np.abs(g_era - med))
            mad = max(mad, self.min_mad)
            threshold = self.clip_multiplier * mad
            abs_g = np.abs(g_era)
            clipped[mask] = np.sign(g_era) * np.minimum(abs_g, threshold)
        return clipped

    def clip_with_stats(self, gradients, era_indices):
        """Clip gradients and return per-era clipping statistics.

        Returns
        -------
        clipped : np.ndarray
            Clipped gradients.
        stats : dict
            Mapping era_label → {'mad': float, 'threshold': float,
                                  'frac_clipped': float}
        """
        clipped = gradients.copy()
        stats = {}
        for era in np.unique(era_indices):
            mask = era_indices == era
            g_era = gradients[mask]
            if len(g_era) < 2:
                stats[era] = {'mad': 0.0, 'threshold': float('inf'),
                              'frac_clipped': 0.0}
                continue
            med = np.median(g_era)
            mad = np.median(np.abs(g_era - med))
            mad = max(mad, self.min_mad)
            threshold = self.clip_multiplier * mad
            abs_g = np.abs(g_era)
            clipped[mask] = np.sign(g_era) * np.minimum(abs_g, threshold)
            frac = float((abs_g > threshold).mean())
            stats[era] = {'mad': float(mad), 'threshold': float(threshold),
                          'frac_clipped': frac}
        return clipped, stats
