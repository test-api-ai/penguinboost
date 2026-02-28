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

        # Sort by time descending for efficient risk set computation
        order = np.argsort(-times)
        gradients = np.zeros(n, dtype=np.float64)

        # Cumulative sum of risk scores (from largest time to smallest)
        cum_risk = 0.0
        cum_risk_arr = np.zeros(n, dtype=np.float64)

        for idx in order:
            cum_risk += risk_scores[idx]
            cum_risk_arr[idx] = cum_risk

        # Re-sort by time ascending for computing gradients
        order_asc = np.argsort(times)

        # For each event, compute gradient contribution
        for idx in order_asc:
            if events[idx]:
                # Risk set: all samples with time >= times[idx]
                at_risk = times >= times[idx]
                risk_sum = risk_scores[at_risk].sum()
                if risk_sum > 0:
                    gradients[idx] -= 1.0  # from the event term
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
