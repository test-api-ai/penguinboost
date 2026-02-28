"""Monotone constraints for PenguinBoost v2."""

import numpy as np


class MonotoneConstraintChecker:
    """Enforces monotone constraints on tree splits.

    For features with constraint +1, the right child's leaf value must be >= left's.
    For features with constraint -1, the right child's leaf value must be <= left's.

    Parameters
    ----------
    constraints : dict or None
        Mapping of feature index -> constraint direction (+1 or -1).
        Features not in the dict are unconstrained (0).
    """

    def __init__(self, constraints=None):
        self.constraints = constraints or {}

    def is_valid_split(self, feature_idx, left_value, right_value):
        """Check if a split satisfies monotone constraints.

        Parameters
        ----------
        feature_idx : int
            Index of the split feature.
        left_value : float
            Predicted value for the left child.
        right_value : float
            Predicted value for the right child.

        Returns
        -------
        bool
            True if the split is valid (satisfies constraints).
        """
        constraint = self.constraints.get(feature_idx, 0)
        if constraint == 0:
            return True
        if constraint == 1:
            # Increasing: right >= left (higher bin values -> higher predictions)
            return right_value >= left_value
        if constraint == -1:
            # Decreasing: right <= left
            return right_value <= left_value
        return True

    def has_constraints(self):
        """Check if any constraints are defined."""
        return len(self.constraints) > 0
