"""Base classes for genetic algorithms in SeapoPym optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deap import base

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Number


def individual_creator(cost_function_weight: tuple[Number]) -> type:
    """
    Create a custom individual class for DEAP genetic algorithms.

    This individual class inherits from `list` and includes a fitness attribute. It is redefined to work with the
    Dask framework, which does not support the default DEAP individual structure created with `deap.creator.create`.
    """

    class Fitness(base.Fitness):
        """Fitness class to store the fitness of an individual."""

        weights = cost_function_weight

    class Individual(list):
        """Individual class to store the parameters of an individual."""

        def __init__(self: Individual, iterator: Sequence, values: Sequence[Number] = ()) -> None:
            super().__init__(iterator)
            self.fitness = Fitness(values=values)

    return Individual


# Note: ABC classes removed in favor of Protocol-based approach.
# See seapopym_optimization.protocols for OptimizationParametersProtocol and OptimizationAlgorithmProtocol.
