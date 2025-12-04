"""Cost function module for SeapoPym optimization."""

from .cost_function import CostFunction
from .metric import MetricProtocol, nrmse_std_comparator, rmse_comparator
from .processor import AbstractScoreProcessor, SpatialScoreProcessor, TimeSeriesScoreProcessor
from .protocol import CostFunctionProtocol

__all__ = [
    "AbstractScoreProcessor",
    "CostFunction",
    "CostFunctionProtocol",
    "MetricProtocol",
    "SpatialScoreProcessor",
    "TimeSeriesScoreProcessor",
    "nrmse_std_comparator",
    "rmse_comparator",
]
