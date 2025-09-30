"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

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
        observations_data: Sequence,
        observations_metadata: Sequence[dict],
    ) -> tuple:
        """
        Evaluate the cost function for given parameters.

        Parameters
        ----------
        args : np.ndarray
            Individual parameters to evaluate
        forcing : ForcingParameterProtocol
            Forcing parameters (resolved from Future if distributed)
        observations_data : Sequence
            List of observation data (resolved from Futures if distributed)
        observations_metadata : Sequence[dict]
            List of observation metadata (name, observation_type)

        Returns
        -------
        tuple
            Fitness values for each observation

        """
        # Reconstruct observation objects from data and metadata
        from seapopym_optimization.observations import TimeSeriesObservation

        observations = [
            TimeSeriesObservation(name=meta["name"], observation=data, observation_type=meta["observation_type"])
            for data, meta in zip(observations_data, observations_metadata, strict=True)
        ]

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
        """Generate the partial cost function used for sequential/parallel optimization."""
        return partial(
            self._cost_function,
            forcing=self.forcing,
            observations_data=[obs.observation for obs in self.observations],
            observations_metadata=[
                {"name": obs.name, "observation_type": obs.observation_type} for obs in self.observations
            ],
        )

    def get_evaluator(self: CostFunction) -> Callable:
        """
        Return the evaluation function to be called on workers.

        This method is used by distributed evaluation strategies to obtain
        the core evaluation function without captured parameters.

        Returns
        -------
        Callable
            Function that takes (args, forcing, observations) and returns fitness tuple

        Examples
        --------
        >>> evaluator = cost_function.get_evaluator()
        >>> fitness = evaluator(args, forcing_data, observations_data)

        """
        return self._cost_function

    def get_distributed_parameters(self: CostFunction) -> dict[str, Any]:
        """
        Return parameters that should be distributed to workers as a dictionary.

        Dask will automatically resolve any Futures contained in this dictionary
        when it's passed as an argument to client.map(). Observations are split
        into data (Futures) and metadata to allow proper resolution.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - 'forcing': ForcingParameter or Future
            - 'observations_data': List of observation data (DataArray or Futures)
            - 'observations_metadata': List of dicts with name and observation_type

        Notes
        -----
        If you subclass CostFunction and add new distributed parameters,
        override this method to include them in the returned dictionary.

        Examples
        --------
        >>> params = cost_function.get_distributed_parameters()
        >>> params['forcing']
        <ForcingParameter or Future>
        >>> params['observations_data']
        [<DataArray or Future>, ...]
        >>> params['observations_metadata']
        [{'name': 'obs1', 'observation_type': <DayCycle.DAY>}, ...]

        See Also
        --------
        get_evaluator : Get the evaluation function to use with these parameters

        """
        return {
            "forcing": self.forcing,
            "observations_data": [obs.observation for obs in self.observations],
            "observations_metadata": [
                {"name": obs.name, "observation_type": obs.observation_type} for obs in self.observations
            ],
        }
