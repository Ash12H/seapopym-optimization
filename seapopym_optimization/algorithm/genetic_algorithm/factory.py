"""
Factory for creating configured GeneticAlgorithm instances.

This module provides factory methods to simplify the creation
of GeneticAlgorithm instances with different evaluation strategies,
hiding configuration complexity for business users.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dask.distributed import Client

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
        meta_parameter: GeneticAlgorithmParameters, cost_function: CostFunctionProtocol, **kwargs
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
        meta_parameter: GeneticAlgorithmParameters, cost_function: CostFunctionProtocol, n_jobs: int = -1, **kwargs
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
        *,
        auto_distribute: bool = True,
        **kwargs: dict,
    ) -> tuple[GeneticAlgorithm, DistributionManager]:
        """
        Create a GA in distributed mode with Dask.

        Uses Dask client.map() with distributed data to evaluate
        individuals across multiple workers efficiently.

        Parameters
        ----------
        meta_parameter : GeneticAlgorithmParameters
            Genetic algorithm parameters
        cost_function : CostFunctionProtocol
            Cost function to optimize
        client : Client
            Dask client for distributed computing
        auto_distribute : bool, default=True
            If True, automatically distribute heavy data
        **kwargs
            Additional arguments for GeneticAlgorithm

        Returns
        -------
        tuple[GeneticAlgorithm, DistributionManager]
            GA instance and distribution manager

        Raises
        ------
        ImportError
            If Dask is not available

        Examples
        --------
        >>> from dask.distributed import Client
        >>> client = Client()
        >>> ga, dist_manager = GeneticAlgorithmFactory.create_distributed(
        ...     meta_params, cost_function, client, auto_distribute=True
        ... )
        >>> results = ga.optimize()
        >>> dist_manager.cleanup()

        """
        if not isinstance(client, Client):
            msg = "client must be a dask.distributed.Client instance"
            raise TypeError(msg)

        logger.info("Creating genetic algorithm in distributed mode")

        # Create distribution manager
        dist_manager = DistributionManager(client)

        if auto_distribute:
            # Automatically distribute heavy data
            logger.info("Auto-distributing data...")
            dist_manager.distribute_forcing(cost_function.configuration_generator.forcing_parameters)
            dist_manager.distribute_observations(cost_function.observations)

        # Create distributed evaluation strategy
        evaluation_strategy = DistributedEvaluation(dist_manager)

        # Create GA instance
        ga = GeneticAlgorithm(
            meta_parameter=meta_parameter,
            cost_function=cost_function,
            evaluation_strategy=evaluation_strategy,
            **kwargs,
        )

        return ga, dist_manager
