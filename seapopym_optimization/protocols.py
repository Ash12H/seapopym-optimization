"""Protocols for SeapoPym optimization algorithms and components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deap import base

    from seapopym_optimization.constraint.base_constraint import AbstractConstraint
    from seapopym_optimization.cost_function.base_cost_function import AbstractCostFunction
    from seapopym_optimization.functional_group.no_transport_functional_groups import Parameter
    from seapopym_optimization.viewer.base_viewer import AbstractViewer


@runtime_checkable
class OptimizationParametersProtocol(Protocol):
    """Protocol for parameters of an optimization algorithm."""

    def generate_toolbox(self, parameters: Sequence[Parameter], cost_function: AbstractCostFunction) -> base.Toolbox:
        """Return a DEAP toolbox configured with the necessary optimization algorithm functions."""
        ...


@runtime_checkable
class OptimizationAlgorithmProtocol(Protocol):
    """Protocol for an optimization algorithm implementation."""

    cost_function: AbstractCostFunction
    constraint: Sequence[AbstractConstraint] | None

    def optimize(self) -> AbstractViewer:
        """Run the optimization algorithm and return a structure containing the results."""
        ...
