"""Era-aware sample reweighting for financial ML (Numerai-style).

In financial time-series (especially Numerai), data is divided into
"eras" (time periods). Standard gradient boosting treats all samples
equally, which can cause overfitting to easy eras. Era boosting
upweights hard eras (low Spearman correlation) to force the model
to improve uniformly across all time periods.
"""

import numpy as np


# ── Shared helpers ────────────────────────────────────────────────────────────

def _rankdata(x):
    """Rank with stable argsort (1-indexed, simple tie order)."""
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x, kind='stable')] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def _spearman_corr(x, y):
    """Spearman rank correlation (pure numpy)."""
    if len(x) < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    c = np.corrcoef(rx, ry)
    return float(c[0, 1]) if np.isfinite(c[0, 1]) else 0.0


def _softmax(x, temperature=1.0):
    """Numerically stable softmax."""
    z = x / max(temperature, 1e-9)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ── Era boosting ──────────────────────────────────────────────────────────────

class EraBoostingReweighter:
    """Era-aware sample reweighting for gradient boosting.

    Computes per-sample gradient weights based on era-level Spearman
    correlation. Eras where the current model performs poorly receive
    higher weight, forcing the model to improve uniformly across time.

    This addresses the key challenge in financial ML: models that achieve
    high average accuracy but fail catastrophically in specific market
    regimes (eras) are not useful in practice.

    Parameters
    ----------
    method : str
        Reweighting strategy:
        - 'hard_era'   : upweight low-correlation eras via softmax(-ρ/T).
        - 'sharpe_reweight' : weight eras proportional to their gradient
          w.r.t. Sharpe(ρ). Eras that would increase Sharpe receive more
          weight. Reduces variance across eras.
        - 'proportional' : weight inversely proportional to |ρ|. Harder
          eras get more weight, but never zero-weights easy eras.
    temperature : float
        Softmax temperature for 'hard_era' and 'sharpe_reweight'. Lower
        temperature → sharper focus on the worst era.
    min_era_weight : float
        Minimum fraction of weight assigned to each era (prevents
        starvation). Default 0.02 (2% per era minimum).
    """

    def __init__(self, method='hard_era', temperature=1.0, min_era_weight=0.02):
        self.method = method
        self.temperature = temperature
        self.min_era_weight = min_era_weight

    def compute_sample_weights(self, predictions, targets, eras):
        """Compute per-sample importance weights based on era performance.

        Parameters
        ----------
        predictions : np.ndarray of shape (n_samples,)
            Current model predictions.
        targets : np.ndarray of shape (n_samples,)
            True targets (1-D).
        eras : np.ndarray of shape (n_samples,)
            Era label for each sample.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Non-negative weights. Within each era, all samples receive
            the same weight. Weights are scaled so that
            sum(weights) == n_samples (unbiased gradient scaling).
        """
        era_labels = np.unique(eras)
        n_eras = len(era_labels)
        n_samples = len(predictions)

        # --- Per-era Spearman correlation ---
        era_corrs = np.array([
            _spearman_corr(predictions[eras == e], targets[eras == e])
            for e in era_labels
        ])

        # --- Era-level weights ---
        if self.method == 'hard_era':
            # Upweight eras with low Spearman correlation
            era_weights = _softmax(-era_corrs, self.temperature)

        elif self.method == 'sharpe_reweight':
            mu = era_corrs.mean()
            sigma = era_corrs.std() + 1e-9
            # ∂Sharpe/∂ρ_e = (σ² - μ·(ρ_e - μ)) / (n_eras · σ³)
            # Eras below mean get higher Sharpe gradient → more weight
            sharpe_grad = (sigma**2 - mu * (era_corrs - mu)) / (n_eras * sigma**3)
            # Use softmax over positive Sharpe gradients
            era_weights = _softmax(sharpe_grad, self.temperature)

        else:  # 'proportional'
            # Weight ∝ 1 - |ρ| (harder eras get more weight, never zero)
            raw = 1.0 - np.abs(era_corrs)
            raw = np.clip(raw, 1e-6, None)
            era_weights = raw / raw.sum()

        # --- Enforce minimum era weight to prevent starvation ---
        min_w = self.min_era_weight / n_eras
        era_weights = np.clip(era_weights, min_w, None)
        era_weights /= era_weights.sum()

        # --- Map era weights → sample weights (uniform within each era) ---
        sample_weights = np.ones(n_samples)
        for i, era in enumerate(era_labels):
            mask = eras == era
            n_in_era = mask.sum()
            if n_in_era == 0:
                continue
            # Scale: era weight × (n_samples / n_in_era) so weights sum to n_eras
            sample_weights[mask] = era_weights[i] * n_samples / n_in_era

        return sample_weights

    def era_stats(self, predictions, targets, eras):
        """Return per-era Spearman correlations for monitoring.

        Returns
        -------
        dict mapping era_label → spearman_correlation
        """
        era_labels = np.unique(eras)
        return {
            era: _spearman_corr(predictions[eras == era], targets[eras == era])
            for era in era_labels
        }


class EraMetrics:
    """Track and summarize per-era performance metrics during training.

    Useful for monitoring era overfitting: a healthy model has a stable,
    high mean Spearman correlation with low cross-era variance (high Sharpe).

    Parameters
    ----------
    eras : np.ndarray
        Era labels for training data.
    """

    def __init__(self, eras):
        self.eras = eras
        self.era_labels_ = np.unique(eras)
        self._history = []   # list of {era: corr} dicts per iteration

    def update(self, predictions, targets):
        """Record per-era Spearman correlations for the current iteration."""
        stats = {
            era: _spearman_corr(
                predictions[self.eras == era], targets[self.eras == era])
            for era in self.era_labels_
        }
        self._history.append(stats)
        return stats

    def mean_corr(self, iteration=-1):
        """Mean Spearman correlation over eras at a given iteration."""
        return float(np.mean(list(self._history[iteration].values())))

    def sharpe(self, iteration=-1):
        """Sharpe ratio of per-era correlations (mean/std)."""
        vals = np.array(list(self._history[iteration].values()))
        return float(vals.mean() / (vals.std() + 1e-9))

    def worst_era(self, iteration=-1):
        """Era with lowest Spearman correlation."""
        stats = self._history[iteration]
        return min(stats, key=stats.get)
