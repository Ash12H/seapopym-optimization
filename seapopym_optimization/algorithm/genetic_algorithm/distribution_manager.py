"""
Distribution manager for Dask-based genetic algorithm optimization.

This module provides the DistributionManager class to handle creation of distributed CostFunction instances.
"""

import logging

from dask.distributed import Future

from seapopym_optimization.cost_function import CostFunction
from seapopym_optimization.observations import TimeSeriesObservation

logger = logging.getLogger(__name__)


class DistributionManager:
    """
    Manager for creating distributed CostFunction instances.

    Provides utilities to create a new CostFunction with Dask Futures
    for forcing and observations data, enabling efficient distributed evaluation.
    """

    @staticmethod
    def create_distributed_cost_function(
        cost_function: CostFunction,
        forcing_future: Future,
        obs_futures: list[Future],
    ) -> CostFunction:
        """
        Create a new CostFunction with distributed data (Futures).

        Parameters
        ----------
        cost_function : CostFunction
            Original cost function with local data
        forcing_future : Future
            Distributed forcing parameters
        obs_futures : list[Future]
            List of distributed observation data

        Returns
        -------
        CostFunction
            New CostFunction instance with Futures

        Raises
        ------
        ValueError
            If number of observation futures doesn't match original observations

        """
        if len(obs_futures) != len(cost_function.observations):
            msg = f"Mismatch: {len(obs_futures)} futures for {len(cost_function.observations)} observations"
            raise ValueError(msg)

        # Create new observation objects with Futures
        distributed_observations = []
        for obs_future, original_obs in zip(obs_futures, cost_function.observations, strict=True):
            # Reconstruct observation with Future data but original metadata
            distributed_obs = TimeSeriesObservation(
                name=original_obs.name,
                observation=obs_future,
                observation_type=original_obs.observation_type,
            )
            distributed_observations.append(distributed_obs)

        # Create new CostFunction with distributed data
        return CostFunction(
            configuration_generator=cost_function.configuration_generator,
            functional_groups=cost_function.functional_groups,
            forcing=forcing_future,
            kernel=cost_function.kernel,
            observations=distributed_observations,
            processor=cost_function.processor,
        )
