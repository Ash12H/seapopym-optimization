"""
Factory for creating configured GeneticAlgorithm instances.

This module provides factory methods to simplify the creation
of GeneticAlgorithm instances with different evaluation strategies,
hiding configuration complexity for business users.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dask.distributed import Client, Future

from seapopym_optimization.algorithm.genetic_algorithm.distribution_manager import DistributionManager
from seapopym_optimization.algorithm.genetic_algorithm.evaluation_strategies import (
    DistributedEvaluation,
    ParallelEvaluation,
    SequentialEvaluation,
)
from seapopym_optimization.algorithm.genetic_algorithm.genetic_algorithm import (
    GeneticAlgorithm,
)

if TYPE_CHECKING:
    from seapopym_optimization.algorithm.genetic_algorithm.genetic_algorithm import (
        GeneticAlgorithmParameters,
    )
    from seapopym_optimization.protocols import CostFunctionProtocol

logger = logging.getLogger(__name__)


class GeneticAlgorithmFactory:
    """
    Factory for creating GeneticAlgorithm instances with different configurations.

    This factory simplifies genetic algorithm creation by encapsulating
    the configuration logic for evaluation strategies and distribution.
    """

    @staticmethod
    def create_sequential(
        meta_parameter: GeneticAlgorithmParameters, cost_function: CostFunctionProtocol, **kwargs: Any
    ) -> GeneticAlgorithm:
        """
        Create a GA in sequential mode.

        Simplest evaluation mode, suitable for small populations
        or situations where parallelization is not necessary.

        Parameters
        ----------
        meta_parameter : GeneticAlgorithmParameters
            Genetic algorithm parameters
        cost_function : CostFunctionProtocol
            Cost function to optimize
        **kwargs
            Additional arguments for GeneticAlgorithm

        Returns
        -------
        GeneticAlgorithm
            Instance configured in sequential mode

        Examples
        --------
        >>> ga = GeneticAlgorithmFactory.create_sequential(meta_params, cost_function)
        >>> results = ga.optimize()

        """
        logger.info("Creating genetic algorithm in sequential mode")

        return GeneticAlgorithm(
            meta_parameter=meta_parameter,
            cost_function=cost_function,
            evaluation_strategy=SequentialEvaluation(),
            **kwargs,
        )

    @staticmethod
    def create_parallel(
        meta_parameter: GeneticAlgorithmParameters, cost_function: CostFunctionProtocol, n_jobs: int = -1, **kwargs: Any
    ) -> GeneticAlgorithm:
        """
        Create a GA in parallel mode using multiprocessing.

        Uses ProcessPoolExecutor to evaluate individuals across
        multiple CPU cores for improved performance.

        Parameters
        ----------
        meta_parameter : GeneticAlgorithmParameters
            Genetic algorithm parameters
        cost_function : CostFunctionProtocol
            Cost function to optimize
        n_jobs : int, default=-1
            Number of parallel jobs. If -1, use all available CPUs
        **kwargs
            Additional arguments for GeneticAlgorithm

        Returns
        -------
        GeneticAlgorithm
            Instance configured in parallel mode

        Examples
        --------
        >>> ga = GeneticAlgorithmFactory.create_parallel(meta_params, cost_function, n_jobs=4)
        >>> results = ga.optimize()

        """
        logger.info("Creating genetic algorithm in parallel mode with %d jobs", n_jobs)

        return GeneticAlgorithm(
            meta_parameter=meta_parameter,
            cost_function=cost_function,
            evaluation_strategy=ParallelEvaluation(n_jobs=n_jobs),
            **kwargs,
        )

    @staticmethod
    def create_distributed(
        meta_parameter: GeneticAlgorithmParameters,
        cost_function: CostFunctionProtocol,
        client: Client,
        **kwargs: Any,
    ) -> GeneticAlgorithm:
        """
        Create a GA in distributed mode with Dask.

        Automatically detects if data is already distributed (Futures) and distributes
        if necessary. Uses Dask client.map() with distributed data to evaluate
        individuals across multiple workers efficiently.

        Parameters
        ----------
        meta_parameter : GeneticAlgorithmParameters
            Genetic algorithm parameters
        cost_function : CostFunctionProtocol
            Cost function to optimize
        client : Client
            Dask client for distributed computing
        **kwargs
            Additional arguments for GeneticAlgorithm

        Returns
        -------
        GeneticAlgorithm
            GA instance configured for distributed execution

        Raises
        ------
        TypeError
            If client is not a Dask Client instance
        ValueError
            If observations are partially distributed (inconsistent state)

        Examples
        --------
        >>> from dask.distributed import Client
        >>> client = Client()
        >>> ga = GeneticAlgorithmFactory.create_distributed(
        ...     meta_params, cost_function, client
        ... )
        >>> results = ga.optimize()
        >>> client.close()

        """
        if not isinstance(client, Client):
            msg = "client must be a dask.distributed.Client instance"
            raise TypeError(msg)

        logger.info("Creating genetic algorithm in distributed mode")

        # Check forcing and distribute if necessary
        if isinstance(cost_function.forcing, Future):
            logger.info("Forcing already distributed (Future detected). Using existing Future.")
            forcing_future = cost_function.forcing
        else:
            logger.info("Distributing forcing to Dask workers with broadcast=True...")
            forcing_future = client.scatter(cost_function.forcing, broadcast=True)

        # Check and distribute observations one by one
        obs_futures = []
        for obs in cost_function.observations:
            if isinstance(obs.observation, Future):
                logger.info("Observation '%s' already distributed (Future detected). Using existing Future.", obs.name)
                obs_futures.append(obs.observation)
            else:
                logger.info("Distributing observation '%s' to Dask workers with broadcast=True...", obs.name)
                obs_future = client.scatter(obs.observation, broadcast=True)
                obs_futures.append(obs_future)

        # Create distributed CostFunction
        logger.info("Creating distributed CostFunction with Futures...")
        distributed_cost_function = DistributionManager.create_distributed_cost_function(
            cost_function, forcing_future, obs_futures
        )

        # Create distributed evaluation strategy
        evaluation_strategy = DistributedEvaluation(distributed_cost_function)

        # Create and return GA instance
        return GeneticAlgorithm(
            meta_parameter=meta_parameter,
            cost_function=distributed_cost_function,
            evaluation_strategy=evaluation_strategy,
            **kwargs,
        )
