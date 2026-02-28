"""Survival analysis objective: Cox Proportional Hazards."""

import numpy as np


class CoxObjective:
    """Cox Proportional Hazards partial likelihood objective.

    Expects y to be a structured array or 2D array with columns:
        - y[:, 0] or y['time']: event/censoring time
        - y[:, 1] or y['event']: event indicator (1=event, 0=censored)

    For the boosting engine, y is passed as a 2D array.
    """

    def __init__(self):
        self._times = None
        self._events = None

    def _parse_y(self, y):
        if y.ndim == 2:
            return y[:, 0], y[:, 1].astype(bool)
        return self._times, self._events

    def set_data(self, times, events):
        self._times = np.asarray(times, dtype=np.float64)
        self._events = np.asarray(events, dtype=bool)

    def init_score(self, y):
        return 0.0

    def gradient(self, y, pred):
        """Negative gradient of Cox partial log-likelihood."""
        times, events = self._parse_y(y)
        n = len(times)
        risk_scores = np.exp(pred)

        # リスクセット計算を効率化するために時刻降順でソート
        order = np.argsort(-times)
        gradients = np.zeros(n, dtype=np.float64)

        # リスクスコアの累積和（最大時刻から最小時刻へ）
        cum_risk = 0.0
        cum_risk_arr = np.zeros(n, dtype=np.float64)

        for idx in order:
            cum_risk += risk_scores[idx]
            cum_risk_arr[idx] = cum_risk

        # 勾配計算のために時刻昇順で再ソート
        order_asc = np.argsort(times)

        # 各イベントの勾配寄与を計算
        for idx in order_asc:
            if events[idx]:
                # リスクセット: times[idx] 以上の全サンプル
                at_risk = times >= times[idx]
                risk_sum = risk_scores[at_risk].sum()
                if risk_sum > 0:
                    gradients[idx] -= 1.0  # イベント項から
                    gradients[at_risk] += risk_scores[at_risk] / risk_sum

        self._cached_pred = pred
        self._cached_risk = risk_scores
        return gradients

    def hessian(self, y, pred):
        """Approximate diagonal Hessian for Cox model."""
        times, events = self._parse_y(y)
        n = len(times)
        risk_scores = np.exp(pred)
        hessians = np.zeros(n, dtype=np.float64)

        for idx in range(n):
            if events[idx]:
                at_risk = times >= times[idx]
                risk_sum = risk_scores[at_risk].sum()
                if risk_sum > 0:
                    p = risk_scores[at_risk] / risk_sum
                    hessians[at_risk] += p * (1 - p)

        return np.maximum(hessians, 1e-7)

    def loss(self, y, pred):
        """Negative Cox partial log-likelihood."""
        times, events = self._parse_y(y)
        risk_scores = np.exp(pred)
        loss = 0.0

        for idx in range(len(times)):
            if events[idx]:
                at_risk = times >= times[idx]
                risk_sum = risk_scores[at_risk].sum()
                if risk_sum > 0:
                    loss -= pred[idx] - np.log(risk_sum)

        n_events = events.sum()
        return loss / max(n_events, 1)

    def transform(self, pred):
        """Return risk scores (exp of predictions)."""
        return np.exp(pred)
