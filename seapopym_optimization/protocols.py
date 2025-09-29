"""Protocols for SeapoPym optimization algorithms and components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from deap import base, tools

    from seapopym_optimization.algorithm.genetic_algorithm.logbook import OptimizationLog
    from seapopym_optimization.functional_group.no_transport_functional_groups import Parameter


@runtime_checkable
class CostFunctionProtocol(Protocol):
    """Protocol for cost functions used in optimization."""

    def generate(self) -> Callable[[Sequence[float]], tuple]:
        """Generate the cost function used for optimization."""
        ...


@runtime_checkable
class OptimizationParametersProtocol(Protocol):
    """Protocol for parameters of an optimization algorithm."""

    def generate_toolbox(self, parameters: Sequence[Parameter], cost_function: CostFunctionProtocol) -> base.Toolbox:
        """Return a DEAP toolbox configured with the necessary optimization algorithm functions."""
        ...


@runtime_checkable
class OptimizationAlgorithmProtocol(Protocol):
    """Protocol for an optimization algorithm implementation."""

    cost_function: CostFunctionProtocol
    constraint: Sequence[ConstraintProtocol] | None

    def optimize(self) -> OptimizationLog:
        """Run the optimization algorithm and return the optimization results as an OptimizationLog."""
        ...


@runtime_checkable
class ConstraintProtocol(Protocol):
    """Protocol for constraints used in optimization algorithms."""

    def generate(self, parameter_names: Sequence[str]) -> tools.DeltaPenalty:
        """Generate the DEAP DeltaPenalty constraint for the optimization algorithm."""
        ...


@runtime_checkable
class ObservationProtocol(Protocol):
    """Protocol for observations used in cost functions."""

    name: str
    observation: object
