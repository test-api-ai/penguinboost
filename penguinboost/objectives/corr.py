"""ランク相関および金融向け目的関数。

Numerai にインスパイアされた株式リターン予測向け目的関数:
- SpearmanObjective: ランクターゲット MSE によるランク相関の最適化
- MaxSharpeEraObjective: エラごとのスピアマン相関のシャープ比を最大化
- FeatureExposurePenalizedObjective: 任意の目的関数に露出削減ペナルティを追加
"""

import numpy as np
from penguinboost.core.era_boost import _rankdata, _spearman_corr


# ── ヘルパー ───────────────────────────────────────────────────────────────────

def _rank_normalize(y):
    """Map y to uniformly-spaced ranks in (-1, 1).

    Returns Van der Waerden-style scores (signed, zero-mean) scaled to [-1, 1],
    which provides a target that is invariant to the scale of y and directly
    interpretable as rank percentile deviation from the median.
    """
    n = len(y)
    ranks = _rankdata(y)                   # 1..n
    normalized = (ranks - 1) / max(n - 1, 1)  # 0..1
    return normalized * 2 - 1             # -1..1  (zero-mean when symmetric)


# ── スピアマン目的関数 ─────────────────────────────────────────────────────────

class SpearmanObjective:
    """Approximate Spearman rank correlation objective.

    Formulation (rank-target MSE):
        L(P, Y) = Σ_i (P_i - r_i)²
        where r_i = normalize_rank(Y_i) ∈ [-1, 1]

    The minimum of this loss coincides with the predictions that best track
    the rank order of Y. Gradients:
        ∂L/∂P_i = 2(P_i - r_i)   (divided by n in the engine)

    Additionally, the correlation-gradient correction is applied to make the
    objective directly maximize Pearson(P, rank(Y)):
        g_i = (P_i - r_i) - α·(P_i - P̄)·corr(P, r)

    The second term penalizes prediction dispersion that does not contribute
    to correlation, improving the quality of the gradient signal.

    Parameters
    ----------
    corr_correction : float in [0, 1]
        Strength of the correlation-gradient correction. 0 = pure rank-MSE,
        1 = full Pearson gradient correction. Default 0.5.
    """

    def __init__(self, corr_correction=0.5):
        self.corr_correction = corr_correction
        self._y_rank = None

    # ── 目的関数インターフェース ───────────────────────────────────────────────

    def init_score(self, y):
        """Initialize predictions to median rank (0.0 after normalization)."""
        self._y_rank = _rank_normalize(y).astype(np.float64)
        return float(self._y_rank.mean())

    def gradient(self, y, pred):
        """Gradient of rank-MSE with optional Pearson correction.

        The correction steers the gradient towards directly maximizing
        the rank correlation rather than just minimizing squared rank error.
        """
        if self._y_rank is None:
            self._y_rank = _rank_normalize(y).astype(np.float64)

        r = self._y_rank
        g = pred - r                                 # rank-MSE gradient

        if self.corr_correction > 0.0 and pred.std() > 1e-9:
            # ピアソン corr(P, r) 勾配補正:
            # P と r のスケールのミスマッチを補正
            corr = float(np.corrcoef(pred, r)[0, 1]) if np.isfinite(np.corrcoef(pred, r)[0, 1]) else 0.0
            p_centered = pred - pred.mean()
            correction = corr * p_centered / (pred.std() + 1e-9) / (r.std() + 1e-9)
            g = g - self.corr_correction * correction

        return g

    def hessian(self, y, pred):
        """Constant hessian (rank-MSE has constant second derivative)."""
        return np.ones(len(y), dtype=np.float64)

    def loss(self, y, pred):
        """Rank-MSE loss (lower = better rank tracking)."""
        if self._y_rank is None:
            self._y_rank = _rank_normalize(y).astype(np.float64)
        return float(np.mean((pred - self._y_rank) ** 2))


# ── MaxSharpe エラ目的関数 ────────────────────────────────────────────────────

class MaxSharpeEraObjective:
    """Maximize the Sharpe ratio of per-era Spearman correlations.

    Objective:
        maximize  Sharpe(ρ) = μ_era(ρ) / σ_era(ρ)

    where ρ_e = Spearman(P_e, Y_e) is the rank correlation within era e.

    Gradient derivation (chain rule through Sharpe):
        ∂Sharpe/∂P_i = Σ_e w_e · ∂ρ_e/∂P_i

    where the Sharpe weights are:
        w_e = ∂(μ/σ)/∂ρ_e = [σ² - μ(ρ_e - μ)] / (n_eras · σ³)

    Eras below the mean correlation receive positive, larger weights
    (improving them raises both μ and reduces variance → increases Sharpe).
    Eras above the mean receive smaller weights (focus shifts away).

    The era-conditional gradient ∂ρ_e/∂P_i is approximated by the
    Pearson correlation gradient using rank-normalized targets within era:
        ∂ρ_e/∂P_i ≈ (r_ei - r̄_e) / (n_e · σ_{P_e} · σ_{r_e})

    Parameters
    ----------
    fallback_to_spearman : bool
        If no era indices are set, fall back to SpearmanObjective behaviour.
    corr_eps : float
        Small constant added to std estimates for numerical stability.
    """

    def __init__(self, fallback_to_spearman=True, corr_eps=1e-9):
        self.fallback_to_spearman = fallback_to_spearman
        self.corr_eps = corr_eps
        self._eras = None
        self._era_labels = None
        self._y_rank = None

    def set_era_indices(self, eras):
        """Set era labels. Must be called before fitting.

        Parameters
        ----------
        eras : array-like of shape (n_samples,)
            Era identifier for each sample (integer or string).
        """
        self._eras = np.asarray(eras)
        self._era_labels = np.unique(self._eras)

    def init_score(self, y):
        """Initialize to median rank."""
        self._y_rank = _rank_normalize(y).astype(np.float64)
        return float(self._y_rank.mean())

    def gradient(self, y, pred):
        """Compute Sharpe-maximizing gradient over eras."""
        if self._eras is None:
            # フォールバック: 標準スピアマン勾配
            if self._y_rank is None:
                self._y_rank = _rank_normalize(y).astype(np.float64)
            return pred - self._y_rank

        n = len(pred)
        g = np.zeros(n, dtype=np.float64)
        n_eras = len(self._era_labels)

        # --- 1. エラごとのスピアマン相関 ---
        era_corrs = np.array([
            _spearman_corr(pred[self._eras == e], y[self._eras == e])
            for e in self._era_labels
        ])

        # --- 2. 各エラの相関に対するシャープ勾配 ---
        mu = era_corrs.mean()
        sigma = era_corrs.std() + self.corr_eps
        # ∂(μ/σ)/∂ρ_e  — エンジンで最小化するため負
        sharpe_weights = (sigma**2 - mu * (era_corrs - mu)) / (n_eras * sigma**3)

        # --- 3. ピアソン近似によるエラ条件付き勾配 ∂ρ_e/∂P_i ---
        for idx, era in enumerate(self._era_labels):
            mask = self._eras == era
            n_e = int(mask.sum())
            if n_e < 2:
                continue

            P_e = pred[mask]
            Y_e = y[mask]

            # Rank-normalize target within this era
            r_e = _rank_normalize(Y_e)
            r_bar = r_e.mean()
            sigma_r = r_e.std() + self.corr_eps
            sigma_P = P_e.std() + self.corr_eps

            # ∂ρ_e/∂P_i ≈ (r_ei - r̄_e) / (n_e · σ_P · σ_r)
            corr_grad = (r_e - r_bar) / (n_e * sigma_P * sigma_r)

            # シャープ勾配: 負のシャープを最小化 → 符号反転
            g[mask] = -sharpe_weights[idx] * corr_grad

        return g

    def hessian(self, y, pred):
        """Positive-definite constant hessian per era."""
        h = np.ones(len(y), dtype=np.float64)
        if self._eras is not None:
            n_eras = len(self._era_labels)
            for era in self._era_labels:
                mask = self._eras == era
                n_e = mask.sum()
                h[mask] = 1.0 / (max(n_e, 1) * n_eras)
        return h

    def loss(self, y, pred):
        """Negative Sharpe of era Spearman correlations (lower = better Sharpe)."""
        if self._eras is None:
            if self._y_rank is None:
                self._y_rank = _rank_normalize(y).astype(np.float64)
            return float(np.mean((pred - self._y_rank) ** 2))
        era_corrs = np.array([
            _spearman_corr(pred[self._eras == e], y[self._eras == e])
            for e in self._era_labels
        ])
        mu = era_corrs.mean()
        sigma = era_corrs.std() + 1e-9
        return float(-mu / sigma)   # negative Sharpe (we minimise)


# ── 特徴量露出ペナルティ ───────────────────────────────────────────────────────

class FeatureExposurePenalizedObjective:
    """Wraps any objective with a feature exposure reduction penalty.

    At each boosting iteration, adds a gradient term that penalizes
    linear correlation between predictions and features:

        R(P) = λ · Σ_k corr(P, X_k)²

    The gradient of this penalty w.r.t. prediction P_i is:

        ∂R/∂P_i = 2λ · Σ_k ρ_k · [(X_ki - X̄_k)/(n·σ_P·σ_k)
                                    - ρ_k·(P_i - P̄)/(n·σ_P²)]

    where ρ_k = corr(P, X_k). This is equivalent to penalising the sum of
    squared feature-prediction correlations, encouraging the model to find
    signals that are orthogonal to the individual feature directions.

    Parameters
    ----------
    base_objective : objective object
        Any PenguinBoost objective (e.g. SpearmanObjective, MSEObjective).
    X_ref : np.ndarray of shape (n_samples, n_features)
        Feature matrix used to compute exposure. Should be the training X.
    lambda_fe : float
        Penalty strength. Larger values reduce exposure more aggressively.
    features : list of int or None
        Subset of feature column indices to penalise against. None = all.
    """

    def __init__(self, base_objective, X_ref, lambda_fe=0.1, features=None):
        self.base_objective = base_objective
        self.lambda_fe = lambda_fe
        self.features = features

        X_sub = X_ref[:, features] if features is not None else X_ref
        self._X_centered = X_sub - X_sub.mean(axis=0)
        self._X_std = X_sub.std(axis=0) + 1e-9
        self._n_feat = X_sub.shape[1]

    # ── 目的関数インターフェース ───────────────────────────────────────────────

    def init_score(self, y):
        return self.base_objective.init_score(y)

    def gradient(self, y, pred):
        return self.base_objective.gradient(y, pred) + self._exposure_gradient(pred)

    def hessian(self, y, pred):
        return self.base_objective.hessian(y, pred)

    def loss(self, y, pred):
        return self.base_objective.loss(y, pred)

    # ── 露出勾配 ──────────────────────────────────────────────────────────────

    def _exposure_gradient(self, pred):
        """Gradient of Σ_k corr(P, X_k)² w.r.t. P.

        Efficient O(n · n_features) computation using vectorized ops.
        """
        n = len(pred)
        std_P = pred.std() + 1e-9
        P_centered = pred - pred.mean()

        # corr_k = (X_k - X̄_k)^T P_centered / (n · std_P · std_k)
        # 形状: (n_features,)
        corr = (self._X_centered.T @ P_centered) / (n * std_P * self._X_std)

        # 線形項: Σ_k (corr_k / std_k) · (X_ki - X̄_k) / (n · std_P)
        linear = (self._X_centered @ (corr / self._X_std)) / (n * std_P)

        # 二次項: P_centered · Σ_k corr_k² / (n · std_P²)
        quadratic = P_centered * float(corr @ corr) / (n * std_P**2)

        return 2.0 * self.lambda_fe * (linear - quadratic)

    def feature_exposure(self, pred):
        """Current per-feature exposure (for monitoring)."""
        n = len(pred)
        std_P = pred.std() + 1e-9
        P_centered = pred - pred.mean()
        return (self._X_centered.T @ P_centered) / (n * std_P * self._X_std)


# ── Neutralization-aware Loss (design doc K) ──────────────────────────────────

class NeutralizationAwareObjective:
    """Objective that maximises Spearman correlation of *neutralised* predictions.

    Numerai evaluates submissions on their neutralized correlation (FNC):

        FNC = corr(ŷ_neutral, y)

    where ŷ_neutral = ŷ - X β  and  β = (X^TX + λI)^{-1} X^T ŷ.

    Training directly on FNC ties the learning objective to the evaluation
    metric and structurally discourages feature-level linear exposures.

    Mathematical gradient derivation
    ---------------------------------
    Let H_X = X (X^TX + λI)^{-1} X^T (hat matrix) and P = I - H_X.

        ŷ_neutral = P ŷ    (P is symmetric and idempotent: P² = P, P^T = P)

    The gradient of corr(P ŷ, y) w.r.t. ŷ (chain rule through P):

        ∂ corr(P ŷ, y) / ∂ ŷ_i = Σ_k P_{ki} · [∂ corr / ∂ (Pŷ)_k]
                                 = P^T · ∇_{Pŷ} corr(Pŷ, y)
                                 = P · ∇_{Pŷ} corr(Pŷ, y)

    We approximate the inner gradient using rank-normalised targets
    (SpearmanObjective approach applied to ŷ_neutral within each era).

    Parameters
    ----------
    X_ref : np.ndarray of shape (n_samples, n_features)
        Feature matrix used to compute the neutralisation projection.
    lambda_ridge : float
        Ridge regularisation for (X^TX + λI)^{-1}. Stabilises neutralisation
        when features are correlated.
    features : list of int or None
        Subset of feature indices to neutralise against. None = all.
    corr_eps : float
        Small constant for numerical stability.
    """

    def __init__(self, X_ref, lambda_ridge=1e-4, features=None, corr_eps=1e-9):
        self.lambda_ridge = lambda_ridge
        self.features = features
        self.corr_eps = corr_eps
        self._eras = None
        self._era_labels = None

        # Pre-compute projection matrix P = I - H_X
        X_sub = X_ref[:, features] if features is not None else X_ref
        n, p = X_sub.shape
        XtX = X_sub.T @ X_sub + lambda_ridge * np.eye(p)
        self._P = np.eye(n) - X_sub @ np.linalg.solve(XtX, X_sub.T)

    def set_era_indices(self, eras):
        """Set era labels for era-conditional Spearman gradient."""
        self._eras = np.asarray(eras)
        self._era_labels = np.unique(self._eras)

    def init_score(self, y):
        return float(_rank_normalize(y).mean())

    def gradient(self, y, pred):
        """Gradient of -corr(P·ŷ, y) w.r.t. ŷ (negated since we minimise)."""
        # Neutralise current predictions
        y_neutral = self._P @ pred

        # Compute Spearman gradient w.r.t. the neutralised predictions
        if self._eras is not None:
            g_neutral = self._spearman_era_gradient(y, y_neutral)
        else:
            r = _rank_normalize(y)
            g_neutral = y_neutral - r

        # Project back through P: ∂L/∂ŷ = P^T · g_neutral = P · g_neutral
        return self._P @ g_neutral

    def _spearman_era_gradient(self, y, y_neutral):
        """Per-era rank-correlation gradient w.r.t. y_neutral."""
        n = len(y)
        g = np.zeros(n, dtype=np.float64)
        for era in self._era_labels:
            mask = self._eras == era
            if mask.sum() < 2:
                continue
            r_e = _rank_normalize(y[mask])
            g[mask] = y_neutral[mask] - r_e
        return g

    def hessian(self, y, pred):
        return np.ones(len(y), dtype=np.float64)

    def loss(self, y, pred):
        """Negative Spearman corr of neutralised predictions (lower = better)."""
        y_neutral = self._P @ pred
        return -float(_spearman_corr_vec(y_neutral, y))


def _spearman_corr_vec(x, y):
    """Spearman rank correlation (vectorised)."""
    from penguinboost.core.era_boost import _rankdata, _spearman_corr
    return _spearman_corr(x, y)
