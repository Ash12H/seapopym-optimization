"""
Distribution manager for Dask-based genetic algorithm optimization.

This module provides the DistributionManager class to handle data distribution
and the distributed evaluation function for genetic algorithm optimization using Dask.
"""

import logging
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any

from dask.distributed import Future

from seapopym_optimization.configuration_generator import NoTransportConfigurationGenerator
from seapopym_optimization.cost_function import CostFunction
from seapopym_optimization.observations import TimeSeriesObservation
from seapopym_optimization.protocols import CostFunctionProtocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dask.distributed import Client

    from seapopym_optimization.functional_group import FunctionalGroupSet

logger = logging.getLogger(__name__)


def is_distributed(obj: object) -> bool:
    """Check if an object is a Dask distributed Future."""
    return isinstance(obj, Future)


class DistributionManager:
    """
    Manager for distributing heavy data across Dask workers.

    Encapsulates all Dask-related complexity and provides a clean interface
    for data distribution with broadcast=True to prevent memory leaks.

    Attributes
    ----------
    client : Client
        Dask client for distributed computing
    _distributed_data : dict
        Cache of distributed data references
    _original_data : dict
        Cache of original data references for cleanup

    """

    def __init__(self, client: "Client") -> None:
        """
        Initialize the distribution manager.

        Parameters
        ----------
        client : Client
            Dask client for distributed computing

        """
        self.client = client
        self._distributed_data: dict[str, Any] = {}
        self._original_data: dict[str, Any] = {}

    def distribute_forcing(self, forcing_parameters: Any) -> Any:
        """
        Distribute forcing parameters with broadcast=True.

        Parameters
        ----------
        forcing_parameters : Any
            Forcing parameters to distribute

        Returns
        -------
        Any
            Distributed Future object

        Warnings
        --------
        UserWarning
            If forcing parameters are already distributed

        """
        if "forcing" in self._distributed_data:
            warnings.warn("Forcing parameters already distributed", UserWarning, stacklevel=2)
            return self._distributed_data["forcing"]

        if is_distributed(forcing_parameters):
            warnings.warn("Forcing parameters already distributed", UserWarning, stacklevel=2)
            self._distributed_data["forcing"] = forcing_parameters
            return forcing_parameters

        logger.info("Distributing forcing parameters...")
        scattered = self.client.scatter(forcing_parameters, broadcast=True)
        self._distributed_data["forcing"] = scattered
        self._original_data["forcing"] = forcing_parameters

        return scattered

    def distribute_observations(self, observations: "Sequence[TimeSeriesObservation]") -> list[Any]:
        """
        Distribute observations with broadcast=True.

        Parameters
        ----------
        observations : Sequence[TimeSeriesObservation]
            List of observations to distribute

        Returns
        -------
        list[Any]
            List of distributed Future objects

        Warnings
        --------
        UserWarning
            If observations are already distributed

        """
        if "observations" in self._distributed_data:
            warnings.warn("Observations already distributed", UserWarning, stacklevel=2)
            return self._distributed_data["observations"]

        logger.info("Distributing observations...")
        scattered_obs = []

        for i, obs in enumerate(observations):
            if is_distributed(obs.observation):
                warnings.warn(
                    f"observation[{i}] '{obs.name}' already distributed. Skipped.",
                    UserWarning,
                    stacklevel=2,
                )
                scattered_obs.append(obs.observation)
            else:
                logger.info("Distributing observation '%s'...", obs.name)
                scattered = self.client.scatter(obs.observation, broadcast=True)
                scattered_obs.append(scattered)

        self._distributed_data["observations"] = scattered_obs
        self._original_data["observations"] = observations

        return scattered_obs

    def create_distributed_evaluator(self, cost_function: CostFunctionProtocol) -> callable:
        """
        Create an evaluation function using distributed data.

        Parameters
        ----------
        cost_function : CostFunctionProtocol
            Cost function to adapt for distribution

        Returns
        -------
        callable
            Distributed evaluation function

        Raises
        ------
        RuntimeError
            If data is not distributed

        """
        forcing_future = self._distributed_data.get("forcing")
        obs_futures = self._distributed_data.get("observations", [])

        if not forcing_future or not obs_futures:
            msg = "Data not distributed. Call distribute_* methods first."
            raise RuntimeError(msg)

        # TODO(Jules): Cost function use ConfigurationGenerator so the forcing fields are stored inside. To allow partial
        # function with this signature, we need to refactor CostFunction to have forcing_parameters as a direct
        # attribute.

        return partial(
            distributed_evaluate,
            forcing_future,
            obs_futures,
            cost_function.functional_groups,
            cost_function.observations,
        )

    def cleanup(self) -> None:
        """Clean up references to distributed Futures."""
        self._distributed_data.clear()
        self._original_data.clear()


def distributed_evaluate(
    forcing_future: Any,
    observations_futures: list[Any],
    functional_groups: "FunctionalGroupSet",
    original_observations: "Sequence[TimeSeriesObservation]",
    individual_params: list[float],
) -> tuple[float, ...]:
    """
    Distributed evaluation function for genetic algorithm individuals.

    This function is executed on Dask workers. Futures are automatically
    resolved by Dask as direct arguments.

    Parameters
    ----------
    forcing_future : Any
        Forcing parameters (Future resolved automatically by Dask)
    observations_futures : list[Any]
        Observations data (Futures resolved automatically by Dask)
    functional_groups : FunctionalGroupSet
        Functional groups configuration
    original_observations : Sequence[TimeSeriesObservation]
        Original observation objects for metadata
    individual_params : list[float]
        Individual parameters to evaluate

    Returns
    -------
    tuple[float, ...]
        Individual fitness values

    """
    # Import here to avoid circular imports on workers
    # Create temporary model generator with resolved data
    configuration_generator = NoTransportConfigurationGenerator(forcing_parameters=forcing_future)

    # Create temporary observations with resolved data but original metadata
    temp_observations = []
    for obs_data, original_obs in zip(observations_futures, original_observations, strict=True):
        temp_obs = TimeSeriesObservation(
            name=original_obs.name,
            observation=obs_data,
            observation_type=original_obs.observation_type,
        )
        temp_observations.append(temp_obs)

    # Create temporary cost function
    temp_cost_function = CostFunction(
        configuration_generator=configuration_generator,
        observations=temp_observations,
        functional_groups=functional_groups,
    )

    # Evaluate with temporary cost function
    evaluator = temp_cost_function.generate()
    return evaluator(individual_params)
