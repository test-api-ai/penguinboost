"""Evaluation metrics: RMSE, MAE, R2, Logloss, AUC, Accuracy, NDCG, C-index, Sharpe, Max Drawdown, Quantile Loss."""

import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def logloss(y_true, y_pred):
    """Binary log loss. y_pred should be probabilities."""
    p = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def accuracy(y_true, y_pred):
    """y_pred can be probabilities (threshold 0.5) or class labels."""
    if np.all((y_pred >= 0) & (y_pred <= 1)):
        pred_labels = (y_pred >= 0.5).astype(int)
    else:
        pred_labels = y_pred.astype(int)
    return np.mean(y_true == pred_labels)


def auc(y_true, y_pred):
    """Area Under the ROC Curve (binary classification)."""
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc_val = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            auc_val += tp

    return auc_val / (n_pos * n_neg)


def ndcg_at_k(y_true, y_pred, k=None):
    """Normalized Discounted Cumulative Gain."""
    if k is None:
        k = len(y_true)
    k = min(k, len(y_true))

    order = np.argsort(-y_pred)[:k]
    dcg = np.sum((2.0 ** y_true[order] - 1) / np.log2(np.arange(k) + 2))

    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum((2.0 ** y_true[ideal_order] - 1) / np.log2(np.arange(k) + 2))

    if idcg == 0:
        return 1.0
    return dcg / idcg


def concordance_index(times, events, risk_scores):
    """Harrell's concordance index for survival analysis."""
    concordant = 0
    discordant = 0
    tied = 0
    n = len(times)

    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[j] <= times[i]:
                continue
            if risk_scores[i] > risk_scores[j]:
                concordant += 1
            elif risk_scores[i] < risk_scores[j]:
                discordant += 1
            else:
                tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


# --- v2 Financial metrics ---

def sharpe_ratio(returns, risk_free_rate=0.0):
    """Annualized Sharpe Ratio.

    Parameters
    ----------
    returns : np.ndarray
        Array of period returns.
    risk_free_rate : float
        Risk-free rate per period.

    Returns
    -------
    float
        Sharpe ratio (annualized assuming 252 trading days).
    """
    excess = returns - risk_free_rate
    if np.std(excess) == 0:
        return 0.0
    return np.mean(excess) / np.std(excess) * np.sqrt(252)


def max_drawdown(cumulative_returns):
    """Maximum drawdown from cumulative return series.

    Parameters
    ----------
    cumulative_returns : np.ndarray
        Cumulative return series (e.g. portfolio value over time).

    Returns
    -------
    float
        Maximum drawdown (positive number, e.g. 0.15 = 15% drawdown).
    """
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (peak - cumulative_returns) / np.where(peak > 0, peak, 1.0)
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def quantile_loss(y_true, y_pred, alpha=0.5):
    """Quantile (pinball) loss.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted quantile values.
    alpha : float
        Quantile level.

    Returns
    -------
    float
        Mean quantile loss.
    """
    residual = y_true - y_pred
    return float(np.mean(np.where(residual >= 0,
                                  alpha * residual,
                                  (alpha - 1.0) * residual)))


METRIC_REGISTRY = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2_score,
    "logloss": logloss,
    "accuracy": accuracy,
    "auc": auc,
    "ndcg": ndcg_at_k,
    "c_index": concordance_index,
    "sharpe_ratio": sharpe_ratio,
    "max_drawdown": max_drawdown,
    "quantile_loss": quantile_loss,
}
