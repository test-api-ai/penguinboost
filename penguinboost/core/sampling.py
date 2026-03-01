"""Gradient-based One-Side Sampling (GOSS) from LightGBM.

Also implements Temporally-Weighted GOSS (TW-GOSS): a financial-domain
extension that combines gradient magnitude with temporal recency, so that
both "difficult" and "recent" samples are prioritised during training.
"""

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

        # 絶対勾配の降順でソート
        abs_grad = np.abs(gradients)
        sorted_idx = np.argsort(-abs_grad)

        top_indices = sorted_idx[:n_top]
        rest_indices = sorted_idx[n_top:]

        # 残りからランダムサンプリング
        if len(rest_indices) > n_other:
            sampled_rest = rng.choice(rest_indices, size=n_other, replace=False)
        else:
            sampled_rest = rest_indices

        # 小勾配サンプルの重みを増幅
        if len(rest_indices) > 0 and len(sampled_rest) > 0:
            amplify = len(rest_indices) / len(sampled_rest)
        else:
            amplify = 1.0

        indices = np.concatenate([top_indices, sampled_rest])
        weights = np.ones(len(indices), dtype=np.float64)
        weights[n_top:] = amplify

        return indices, weights


class TemporallyWeightedGOSSSampler:
    """Temporally-Weighted GOSS (TW-GOSS) for financial time-series.

    Standard GOSS prioritises samples with large gradients, but in finance
    large gradients often correspond to outlier market events rather than
    informative recent signals. TW-GOSS combines gradient magnitude with
    an exponential temporal recency weight:

        w_i = |g_i| * exp(-lambda * (t_max - t_i))

    where t_i is the ordinal time (era index) for sample i and lambda
    controls how quickly older samples lose importance.

    Algorithm
    ---------
    1. Compute combined weight w_i for every sample.
    2. Select the top ``top_rate`` fraction by weight as "confirmed" samples.
    3. From the remainder, sample ``other_rate`` fraction with probability
       proportional to w_i (recent, high-gradient samples more likely).
    4. Assign compensation factor (1 - top_rate) / other_rate to the
       randomly sampled subset so that the effective gradient sum is
       unbiased.

    Parameters
    ----------
    top_rate : float
        Fraction of samples selected deterministically (largest combined
        weight). Analogous to the ``a`` parameter in the original GOSS.
    other_rate : float
        Fraction of remaining samples selected stochastically (weighted).
        Analogous to ``b`` in the original GOSS.
    temporal_decay : float
        Exponential decay rate lambda for the temporal kernel. Larger
        values → stronger recency bias (recommended range 0.001–0.1).
    random_state : int or None
    """

    def __init__(self, top_rate=0.2, other_rate=0.1,
                 temporal_decay=0.01, random_state=None):
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.temporal_decay = temporal_decay
        self.random_state = random_state

    def sample(self, gradients, time_indices):
        """Select sample indices with temporal-gradient combined weight.

        Parameters
        ----------
        gradients : np.ndarray of shape (n_samples,)
            Current boosting gradients.
        time_indices : np.ndarray of shape (n_samples,)
            Ordinal time index for each sample (e.g., era integer id).
            Need not start at 0; only relative differences matter.

        Returns
        -------
        indices : np.ndarray
            Selected sample indices (length varies).
        weights : np.ndarray
            Per-selected-sample amplification weights. Top samples get 1.0;
            stochastic samples get the compensation factor (1-a)/b.
        """
        rng = np.random.RandomState(self.random_state)
        n = len(gradients)
        time_indices = np.asarray(time_indices, dtype=np.float64)

        # Combined weight: gradient magnitude × temporal recency kernel
        t_max = float(time_indices.max())
        temporal_kernel = np.exp(-self.temporal_decay * (t_max - time_indices))
        combined_w = np.abs(gradients) * temporal_kernel
        # Avoid all-zero weights (e.g., all gradients are 0)
        total_w = combined_w.sum()
        if total_w < 1e-12:
            combined_w = np.ones(n, dtype=np.float64)

        n_top = max(1, int(n * self.top_rate))
        n_other = max(1, int(n * self.other_rate))

        # Top samples: highest combined weight
        sorted_idx = np.argsort(-combined_w)
        top_indices = sorted_idx[:n_top]
        rest_indices = sorted_idx[n_top:]

        # Stochastic sample from the remainder, proportional to combined_w
        n_sample = min(n_other, len(rest_indices))
        if len(rest_indices) == 0:
            sampled_rest = np.array([], dtype=np.int64)
        else:
            rest_w = combined_w[rest_indices]
            rest_probs = rest_w / rest_w.sum()
            sampled_rest = rng.choice(
                rest_indices, size=n_sample, replace=False, p=rest_probs)

        # Compensation factor: unbiased gradient estimation
        # Weight for stochastic samples: (1 - a) / b  (GOSS formula)
        if n_sample > 0 and len(rest_indices) > 0:
            amplify = (1.0 - self.top_rate) / self.other_rate
        else:
            amplify = 1.0

        indices = np.concatenate([top_indices, sampled_rest])
        weights = np.ones(len(indices), dtype=np.float64)
        weights[len(top_indices):] = amplify

        return indices, weights
