"""PenguinBoost v3 - AROGB: Adversarial Regularized Ordered Gradient Boosting.

Hybrid gradient boosting fusing LightGBM, CatBoost, and XGBoost techniques
with financial-specific overfitting resistance.

v3 additions:
- Feature neutralization and orthogonal gradient projection
- Era-aware boosting and Sharpe-maximizing objectives
- Spearman / MaxSharpe / FeatureExposurePenalized objectives
"""

from penguinboost.sklearn_api import (
    PenguinBoostClassifier,
    PenguinBoostRegressor,
    PenguinBoostRanker,
    PenguinBoostSurvival,
    PenguinBoostQuantileRegressor,
)

# v3: financial ML utilities
from penguinboost.core.neutralization import (
    FeatureNeutralizer,
    OrthogonalGradientProjector,
)
from penguinboost.core.era_boost import (
    EraBoostingReweighter,
    EraMetrics,
)
from penguinboost.objectives.corr import (
    SpearmanObjective,
    MaxSharpeEraObjective,
    FeatureExposurePenalizedObjective,
)

try:
    from penguinboost._core import set_num_threads, get_num_threads
except ImportError:
    def set_num_threads(n): pass      # noqa: E704
    def get_num_threads(): return 1   # noqa: E704

__version__ = "0.3.1"
__all__ = [
    # sklearn estimators
    "PenguinBoostClassifier",
    "PenguinBoostRegressor",
    "PenguinBoostRanker",
    "PenguinBoostSurvival",
    "PenguinBoostQuantileRegressor",
    # v3: feature neutralization
    "FeatureNeutralizer",
    "OrthogonalGradientProjector",
    # v3: era-aware boosting
    "EraBoostingReweighter",
    "EraMetrics",
    # v3: financial objectives
    "SpearmanObjective",
    "MaxSharpeEraObjective",
    "FeatureExposurePenalizedObjective",
    # thread control
    "set_num_threads",
    "get_num_threads",
]
