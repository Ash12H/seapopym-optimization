from .genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmParameters
from .factory import GeneticAlgorithmFactory
from .evaluation_strategies import (
    EvaluationStrategy,
    SequentialEvaluation,
)

# Import protocols for type checking and runtime validation
from ...protocols import OptimizationAlgorithmProtocol, OptimizationParametersProtocol

__all__ = [
    # Core classes
    "GeneticAlgorithm",
    "GeneticAlgorithmParameters",
    "GeneticAlgorithmFactory",

    # Evaluation strategies
    "EvaluationStrategy",
    "SequentialEvaluation",

    # Protocols
    "OptimizationAlgorithmProtocol",
    "OptimizationParametersProtocol",
]