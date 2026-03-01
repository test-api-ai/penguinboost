"""Multi-Target Auxiliary Learning objective for financial ML.

Addresses high target noise (Numerai-style) by mixing gradients from
multiple related targets.  Trees that fit *all* targets simultaneously
are less likely to overfit the noise of any single one.

Mathematical formulation
------------------------
Given a main target y_main and K auxiliary targets y_1, …, y_K:

    g_i^{mixed} = α · g_i^{main} + (1-α) · (1/K) Σ_k g_i^{aux_k}

where g^{main} and g^{aux_k} are the gradients of the respective
objectives evaluated at the current prediction.

Dynamic alpha schedule (optional)
----------------------------------
In early boosting rounds (k << n_estimators), a lower α gives the model
more flexibility to find a representation shared across all targets.
In later rounds, a higher α focuses on the primary objective.

    α(t) = α_min + (α_max - α_min) · (t / T)^schedule_power
"""

from copy import deepcopy
import numpy as np


class MultiTargetAuxiliaryObjective:
    """Gradient boosting objective that mixes a main and K auxiliary targets.

    Parameters
    ----------
    main_objective : objective object
        Any PenguinBoost objective (SpearmanObjective, MSEObjective, …).
    alpha : float in (0, 1]
        Fixed mixing weight for the main target.  Set ``use_schedule=True``
        to override with a dynamic schedule.
    use_schedule : bool
        If True, linearly ramp alpha from ``alpha_start`` to ``alpha``
        over ``n_estimators`` rounds.
    alpha_start : float
        Initial alpha when ``use_schedule=True``.
    n_estimators : int
        Total planned boosting rounds (needed for schedule).
    schedule_power : float
        Exponent for the schedule ramp (1.0 = linear, 2.0 = quadratic).
    """

    def __init__(self, main_objective, alpha=0.7,
                 use_schedule=False, alpha_start=0.3,
                 n_estimators=100, schedule_power=1.0):
        self.main_objective = main_objective
        self.alpha = alpha
        self.use_schedule = use_schedule
        self.alpha_start = alpha_start
        self.n_estimators = n_estimators
        self.schedule_power = schedule_power

        self._aux_targets = None    # (n_samples, K)
        self._aux_objectives = []   # list of K objectives
        self._current_iter = 0      # updated externally via set_iteration()

    # ── auxiliary target registration ────────────────────────────────────────

    def set_aux_targets(self, Y_aux, aux_objectives=None):
        """Set auxiliary training targets.

        Parameters
        ----------
        Y_aux : np.ndarray of shape (n_samples, K)
            Matrix of K auxiliary targets.
        aux_objectives : list of objective objects or None
            One objective per auxiliary target.  If None, a deep copy of
            ``main_objective`` is used for each auxiliary target.
        """
        self._aux_targets = np.asarray(Y_aux, dtype=np.float64)
        K = self._aux_targets.shape[1]
        if aux_objectives is None:
            self._aux_objectives = [deepcopy(self.main_objective)
                                    for _ in range(K)]
        else:
            if len(aux_objectives) != K:
                raise ValueError(
                    f"aux_objectives length ({len(aux_objectives)}) "
                    f"must equal Y_aux.shape[1] ({K})")
            self._aux_objectives = list(aux_objectives)

    def set_iteration(self, t):
        """Update the current boosting iteration for the schedule."""
        self._current_iter = int(t)

    # ── effective alpha ───────────────────────────────────────────────────────

    def _effective_alpha(self):
        if not self.use_schedule:
            return self.alpha
        T = max(self.n_estimators - 1, 1)
        frac = min(self._current_iter / T, 1.0) ** self.schedule_power
        return self.alpha_start + (self.alpha - self.alpha_start) * frac

    # ── objective interface ───────────────────────────────────────────────────

    def init_score(self, y):
        return self.main_objective.init_score(y)

    def gradient(self, y, pred):
        """Mixed gradient: α·g_main + (1-α)·mean(g_aux_k)."""
        g_main = self.main_objective.gradient(y, pred)

        if self._aux_targets is None or len(self._aux_objectives) == 0:
            return g_main

        K = self._aux_targets.shape[1]
        g_aux = np.zeros_like(g_main)
        for k, obj_k in enumerate(self._aux_objectives):
            g_aux += obj_k.gradient(self._aux_targets[:, k], pred)
        g_aux /= K

        alpha = self._effective_alpha()
        return alpha * g_main + (1.0 - alpha) * g_aux

    def hessian(self, y, pred):
        """Hessian from main objective (auxiliary hessians are not mixed)."""
        return self.main_objective.hessian(y, pred)

    def loss(self, y, pred):
        """Loss of the main objective (for monitoring/early stopping)."""
        return self.main_objective.loss(y, pred)

    # ── convenience ──────────────────────────────────────────────────────────

    def set_era_indices(self, eras):
        """Forward era indices to underlying objectives that support it."""
        if hasattr(self.main_objective, 'set_era_indices'):
            self.main_objective.set_era_indices(eras)
        for obj in self._aux_objectives:
            if hasattr(obj, 'set_era_indices'):
                obj.set_era_indices(eras)
