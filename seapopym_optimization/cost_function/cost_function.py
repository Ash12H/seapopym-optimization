"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from seapopym.standard.protocols import ForcingParameterProtocol, KernelParameterProtocol

    from seapopym_optimization.configuration_generator.protocols import ConfigurationGeneratorProtocol
    from seapopym_optimization.cost_function.processor import AbstractScoreProcessor
    from seapopym_optimization.functional_group.base_functional_group import AbstractFunctionalGroup, FunctionalGroupSet
    from seapopym_optimization.protocols import ObservationProtocol

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CostFunction:
    """The cost function generator for SeapoPym models."""

    # TODO(Jules): We can gather configuration generators and functional groups in a single object later if needed.
    configuration_generator: ConfigurationGeneratorProtocol
    functional_groups: FunctionalGroupSet[AbstractFunctionalGroup]
    forcing: ForcingParameterProtocol
    kernel: KernelParameterProtocol
    observations: Sequence[ObservationProtocol]  # Can accept any observation implementation
    processor: AbstractScoreProcessor  # Processor for computing scores from state and observations

    def __post_init__(self: CostFunction) -> None:
        """Check types and convert functional groups if necessary."""
        # TODO(Jules): Implement type checking and conversion if necessary

    # NOTE(Jules): Forcing and observations must be passed as parameter of the cost function to be used with Dask
    # and scattered to workers. They cannot be attributes of the class.
    def _cost_function(
        self: CostFunction,
        args: np.ndarray,
        forcing: ForcingParameterProtocol,
        observations: Sequence[ObservationProtocol],
    ) -> tuple:
        configuration = self.configuration_generator.generate(
            functional_group_parameters=self.functional_groups.generate(args),
            forcing_parameters=forcing,
            kernel=self.kernel,
        )

        # Create model from configuration and run it
        with self.configuration_generator.model_class.from_configuration(configuration) as model:
            model.run()
            state = model.state

            # Compute score for each observation
            return tuple(self.processor.process(state, obs) for obs in observations)

    def generate(self: CostFunction) -> Callable[[Sequence[float]], tuple]:
        """Generate the partial cost function used for optimization."""
        return partial(self._cost_function, forcing=self.forcing, observations=self.observations)
