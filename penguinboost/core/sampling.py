"""Gradient-based One-Side Sampling (GOSS) from LightGBM."""

import numpy as np


class GOSSSampler:
    """GOSS: keep all samples with large gradients, randomly sample from the rest.

    Parameters
    ----------
    top_rate : float
        Fraction of samples with largest gradients to keep (default 0.2).
    other_rate : float
        Fraction of remaining samples to randomly keep (default 0.1).
    """

    def __init__(self, top_rate=0.2, other_rate=0.1, random_state=None):
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.random_state = random_state

    def sample(self, gradients):
        """Select sample indices and compute weight multipliers.

        Returns
        -------
        indices : np.ndarray
            Selected sample indices.
        weights : np.ndarray
            Weight multiplier for each selected sample (1.0 for top, amplified for others).
        """
        rng = np.random.RandomState(self.random_state)
        n = len(gradients)

        n_top = max(1, int(n * self.top_rate))
        n_other = max(1, int(n * self.other_rate))

        # Sort by absolute gradient descending
        abs_grad = np.abs(gradients)
        sorted_idx = np.argsort(-abs_grad)

        top_indices = sorted_idx[:n_top]
        rest_indices = sorted_idx[n_top:]

        # Random sample from the rest
        if len(rest_indices) > n_other:
            sampled_rest = rng.choice(rest_indices, size=n_other, replace=False)
        else:
            sampled_rest = rest_indices

        # Weight amplification for the sampled small-gradient samples
        if len(rest_indices) > 0 and len(sampled_rest) > 0:
            amplify = len(rest_indices) / len(sampled_rest)
        else:
            amplify = 1.0

        indices = np.concatenate([top_indices, sampled_rest])
        weights = np.ones(len(indices), dtype=np.float64)
        weights[n_top:] = amplify

        return indices, weights
