"""Ranking objective: LambdaRank (NDCG optimization)."""

import numpy as np


class LambdaRankObjective:
    """LambdaRank objective for learning to rank.

    Optimizes NDCG by computing pairwise lambda gradients.
    Expects y to contain relevance labels and group information
    to be set via set_group().
    """

    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.groups_ = None

    def set_group(self, group):
        """Set query group boundaries.

        Parameters
        ----------
        group : array-like
            Number of samples in each group/query, e.g. [5, 3, 7].
        """
        self.groups_ = np.array(group, dtype=int)

    def init_score(self, y):
        return 0.0

    def gradient(self, y, pred):
        """Compute LambdaRank gradients."""
        n = len(y)
        gradients = np.zeros(n, dtype=np.float64)
        hessians = np.zeros(n, dtype=np.float64)

        if self.groups_ is None:
            self.groups_ = np.array([n])

        start = 0
        for size in self.groups_:
            end = start + size
            g, h = self._lambda_gradient_group(
                y[start:end], pred[start:end])
            gradients[start:end] = g
            hessians[start:end] = h
            start = end

        self._cached_hessians = hessians
        return gradients

    def hessian(self, y, pred):
        if hasattr(self, "_cached_hessians"):
            return self._cached_hessians
        return np.ones_like(y, dtype=np.float64)

    def loss(self, y, pred):
        """Negative mean NDCG across groups."""
        if self.groups_ is None:
            return 0.0
        ndcgs = []
        start = 0
        for size in self.groups_:
            end = start + size
            ndcgs.append(self._ndcg(y[start:end], pred[start:end]))
            start = end
        return -np.mean(ndcgs)

    def transform(self, pred):
        return pred

    def _lambda_gradient_group(self, relevance, scores):
        """Compute lambda gradients for one query group."""
        n = len(relevance)
        gradients = np.zeros(n, dtype=np.float64)
        hessians = np.zeros(n, dtype=np.float64)

        # デルタ NDCG のための理想 DCG
        ideal_dcg = self._dcg(np.sort(relevance)[::-1])
        if ideal_dcg == 0:
            return gradients, np.ones(n, dtype=np.float64)

        sorted_idx = np.argsort(-scores)

        for i in range(n):
            for j in range(i + 1, n):
                si, sj = sorted_idx[i], sorted_idx[j]
                if relevance[si] == relevance[sj]:
                    continue

                # より高い関連性を "正" として確保
                if relevance[si] < relevance[sj]:
                    si, sj = sj, si

                score_diff = self.sigma * (scores[si] - scores[sj])
                rho = 1.0 / (1.0 + np.exp(score_diff))

                # デルタ NDCG
                gain_i = (2.0 ** relevance[si] - 1)
                gain_j = (2.0 ** relevance[sj] - 1)
                disc_i = 1.0 / np.log2(i + 2)
                disc_j = 1.0 / np.log2(j + 2)
                delta_ndcg = abs((gain_i - gain_j) * (disc_i - disc_j)) / ideal_dcg

                lam = rho * delta_ndcg
                gradients[si] -= lam
                gradients[sj] += lam
                hess_val = max(rho * (1 - rho) * delta_ndcg, 1e-7)
                hessians[si] += hess_val
                hessians[sj] += hess_val

        hessians = np.maximum(hessians, 1e-7)
        return gradients, hessians

    def _dcg(self, relevance):
        positions = np.arange(1, len(relevance) + 1)
        return np.sum((2.0 ** relevance - 1) / np.log2(positions + 1))

    def _ndcg(self, relevance, scores):
        ideal = self._dcg(np.sort(relevance)[::-1])
        if ideal == 0:
            return 1.0
        order = np.argsort(-scores)
        actual = self._dcg(relevance[order])
        return actual / ideal
