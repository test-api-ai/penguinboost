"""PenguinBoost - AROGB: Adversarial Regularized Ordered Gradient Boosting.

Hybrid gradient boosting fusing LightGBM, CatBoost, and XGBoost techniques
with financial-specific overfitting resistance.

Features:
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
    NeutralizationAwareObjective,
)
from penguinboost.objectives.regression import (
    AsymmetricHuberObjective,
)
from penguinboost.objectives.multi_target import MultiTargetAuxiliaryObjective
from penguinboost.core.conformal import ConformalPredictor, EraConformalPredictor
from penguinboost.core.sampling import TemporallyWeightedGOSSSampler
from penguinboost.core.regularization import EraAdaptiveGradientClipper
from penguinboost.core.dart import EraAwareDARTManager

try:
    from penguinboost._core import set_num_threads, get_num_threads
except ImportError:
    def set_num_threads(n): pass      # noqa: E704
    def get_num_threads(): return 1   # noqa: E704

__version__ = "0.3.6"
__all__ = [
    # sklearn 推定器
    "PenguinBoostClassifier",
    "PenguinBoostRegressor",
    "PenguinBoostRanker",
    "PenguinBoostSurvival",
    "PenguinBoostQuantileRegressor",
    # 特徴量中立化
    "FeatureNeutralizer",
    "OrthogonalGradientProjector",
    # エラ対応ブースティング
    "EraBoostingReweighter",
    "EraMetrics",
    # 金融向け目的関数
    "SpearmanObjective",
    "MaxSharpeEraObjective",
    "FeatureExposurePenalizedObjective",
    "NeutralizationAwareObjective",
    # 損失関数
    "AsymmetricHuberObjective",
    # マルチターゲット
    "MultiTargetAuxiliaryObjective",
    # 予測区間
    "ConformalPredictor",
    "EraConformalPredictor",
    # サンプリング
    "TemporallyWeightedGOSSSampler",
    # 勾配クリッピング
    "EraAdaptiveGradientClipper",
    # Era-aware DART
    "EraAwareDARTManager",
    # スレッド制御
    "set_num_threads",
    "get_num_threads",
]
