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

__version__ = "0.3.0"
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
]
