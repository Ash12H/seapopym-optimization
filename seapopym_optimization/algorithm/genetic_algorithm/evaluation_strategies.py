"""
Evaluation strategies for genetic algorithm.

This module defines different evaluation strategies (sequential, parallel, distributed)
using the Strategy pattern, allowing dynamic mode switching without modifying
the business logic of the genetic algorithm.
"""

from __future__ import annotations

import logging
import multiprocessing
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from dask.distributed import Future

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deap import base

    from seapopym_optimization.cost_function import CostFunction

logger = logging.getLogger(__name__)


class EvaluationStrategy(ABC):
    """
    Abstract interface for evaluation strategies.

    The Strategy pattern allows defining a family of evaluation algorithms,
    encapsulating them and making them interchangeable. This allows the genetic
    algorithm to vary independently from the clients that use it.
    """

    @abstractmethod
    def evaluate(self, individuals: Sequence, toolbox: base.Toolbox) -> list:
        """
        Evaluate a list of individuals.

        Parameters
        ----------
        individuals : Sequence
            List of individuals to evaluate
        toolbox : base.Toolbox
            DEAP toolbox with evaluation function

        Returns
        -------
        list
            List of calculated fitness values

        Raises
        ------
        NotImplementedError
            If method is not implemented in derived class

        """

    def __str__(self) -> str:
        """String representation of the strategy."""
        return self.__class__.__name__


class SequentialEvaluation(EvaluationStrategy):
    """
    Classic sequential evaluation strategy.

    Uses Python's standard map() function to evaluate
    individuals one by one sequentially.
    """

    def evaluate(self, individuals: Sequence, toolbox: base.Toolbox) -> list:
        """
        Sequential evaluation with standard map().

        Parameters
        ----------
        individuals : Sequence
            List of individuals to evaluate
        toolbox : base.Toolbox
            DEAP toolbox with evaluation function

        Returns
        -------
        list
            List of calculated fitness values

        """
        logger.debug("Sequential evaluation of %d individuals", len(individuals))
        return list(map(toolbox.evaluate, individuals))


class DistributedEvaluation(EvaluationStrategy):
    """
    Distributed evaluation strategy using Dask.

    Uses Dask client.map() with a distributed CostFunction to evaluate
    individuals across multiple workers efficiently.
    """

    def __init__(self, cost_function: CostFunction) -> None:
        """
        Initialize distributed evaluation strategy.

        Parameters
        ----------
        cost_function : CostFunction
            Cost function with distributed data (Futures)

        Raises
        ------
        TypeError
            If cost function data is not distributed (not Futures)

        """
        # Verify that forcing is distributed
        if not isinstance(cost_function.forcing, Future):
            msg = "CostFunction.forcing must be a Dask Future for distributed evaluation"
            raise TypeError(msg)

        # Verify that all observations are distributed
        if not all(isinstance(obs, Future) for obs in cost_function.observations):
            msg = "All observations must be Dask Futures for distributed evaluation"
            raise TypeError(msg)

        self.cost_function = cost_function
        # Extract client from one of the Futures
        self.client = cost_function.forcing.client

    def evaluate(self, individuals: Sequence, toolbox: base.Toolbox) -> list:  # noqa: ARG002
        """
        Distributed evaluation using client.map() with distributed CostFunction.

        Parameters
        ----------
        individuals : Sequence
            List of individuals to evaluate
        toolbox : base.Toolbox
            DEAP toolbox (not used, kept for interface compatibility)

        Returns
        -------
        list
            List of calculated fitness values

        """
        logger.debug("Distributed evaluation of %d individuals", len(individuals))

        # Generate evaluator from distributed cost function
        evaluator = self.cost_function.generate()

        # Convert individuals to parameter lists
        individual_params = [list(ind) for ind in individuals]

        # Map computation across workers
        # Dask automatically resolves Futures when evaluator is called on workers
        futures = self.client.map(evaluator, individual_params)

        # Gather results
        return self.client.gather(futures)


class ParallelEvaluation(EvaluationStrategy):
    """
    Parallel evaluation strategy using multiprocessing.

    Uses ProcessPoolExecutor to evaluate individuals in parallel
    across multiple CPU cores.
    """

    def __init__(self, n_jobs: int = -1) -> None:
        """
        Initialize parallel evaluation strategy.

        Parameters
        ----------
        n_jobs : int, default=-1
            Number of parallel jobs. If -1, use all available CPUs.

        """
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif n_jobs > 0:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        else:
            msg = "n_jobs must be positive or -1"
            raise ValueError(msg)

    def evaluate(self, individuals: Sequence, toolbox: base.Toolbox) -> list:
        """
        Parallel evaluation using multiprocessing.

        Parameters
        ----------
        individuals : Sequence
            List of individuals to evaluate
        toolbox : base.Toolbox
            DEAP toolbox with evaluation function

        Returns
        -------
        list
            List of calculated fitness values

        """
        logger.debug("Parallel evaluation of %d individuals using %d workers", len(individuals), self.n_jobs)

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(toolbox.evaluate, ind) for ind in individuals]
            return [future.result() for future in futures]
