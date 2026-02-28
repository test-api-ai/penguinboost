from penguinboost.objectives.regression import MSEObjective, MAEObjective, HuberObjective
from penguinboost.objectives.classification import BinaryLoglossObjective, SoftmaxObjective
from penguinboost.objectives.ranking import LambdaRankObjective
from penguinboost.objectives.survival import CoxObjective
from penguinboost.objectives.quantile import QuantileObjective, CVaRObjective

OBJECTIVE_REGISTRY = {
    "mse": MSEObjective,
    "mae": MAEObjective,
    "huber": HuberObjective,
    "binary_logloss": BinaryLoglossObjective,
    "softmax": SoftmaxObjective,
    "lambdarank": LambdaRankObjective,
    "cox": CoxObjective,
    "quantile": QuantileObjective,
    "cvar": CVaRObjective,
}
