"""This module contains the main genetic algorithm functions that can be used to optimize the model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from deap import algorithms, base, tools

from seapopym_optimization.algorithm.genetic_algorithm.logbook import OptimizationLog

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Number

    from dask.distributed import Client
    from pandas._typing import FilePath, WriteBuffer

    from seapopym_optimization.functional_group.no_transport_functional_groups import Parameter
    from seapopym_optimization.protocols import ConstraintProtocol, CostFunctionProtocol

logger = logging.getLogger(__name__)


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


@dataclass
class GeneticAlgorithmParameters:
    """
    The structure used to store the genetic algorithm parameters. Can generate the toolbox with default
    parameters.

    Parameters
    ----------
    MUTPB: float
        Represents the probability of mutating an individual. It is recommended to use a value between 0.001 and 0.1.
    ETA: float
        Crowding degree of the mutation. A high eta will produce a mutant resembling its parent, while a small eta will
        produce a solution much more different. It is recommended to use a value between 1 and 20.
    INDPB: float
        Represents the individual probability of mutation for each attribute of the individual. It is recommended to use
        a value between 0.0 and 0.1. If you have a lot of parameters, you can use a 1/len(parameters) value.
    CXPB: float
        Represents the probability of mating two individuals. It is recommended to use a value between 0.5 and 1.0.
    NGEN: int
        Represents the number of generations.
    POP_SIZE: int
        Represents the size of the population.
    cost_function_weight: tuple | float = (-1.0,)
        The weight of the cost function. The default value is (-1.0,) to minimize the cost function.

    """

    ETA: float
    INDPB: float
    CXPB: float
    MUTPB: float
    NGEN: int
    POP_SIZE: int
    TOURNSIZE: int = field(default=3)
    cost_function_weight: tuple[Number] = (-1.0,)

    def __post_init__(self: GeneticAlgorithmParameters) -> None:
        """Check parameters and set default functions for selection, mating, mutation and variation."""
        self.select = tools.selTournament
        self.mate = tools.cxTwoPoint
        self.mutate = tools.mutPolynomialBounded
        self.variation = algorithms.varAnd
        self.cost_function_weight = tuple(
            np.asarray(self.cost_function_weight) / np.sum(np.absolute(self.cost_function_weight))
        )

    def generate_toolbox(
        self: GeneticAlgorithmParameters, parameters: Sequence[Parameter], cost_function: CostFunctionProtocol
    ) -> base.Toolbox:
        """Generate a DEAP toolbox with the necessary functions for the genetic algorithm."""
        toolbox = base.Toolbox()
        Individual = individual_creator(self.cost_function_weight)  # noqa: N806
        toolbox.register("Individual", Individual)

        for param in parameters:
            toolbox.register(param.name, param.init_method, param.lower_bound, param.upper_bound)

        def individual() -> list:
            return Individual([param.init_method(param.lower_bound, param.upper_bound) for param in parameters])

        toolbox.register("population", tools.initRepeat, list, individual)
        toolbox.register("evaluate", cost_function.generate())
        toolbox.register("mate", self.mate)
        low_boundaries = [param.lower_bound for param in parameters]
        up_boundaries = [param.upper_bound for param in parameters]
        toolbox.register("mutate", self.mutate, eta=self.ETA, indpb=self.INDPB, low=low_boundaries, up=up_boundaries)
        toolbox.register("select", self.select, tournsize=self.TOURNSIZE)
        return toolbox


@dataclass
class GeneticAlgorithm:
    """
    Contains the genetic algorithm parameters and the cost function to optimize. By default, the order of
    of the process is SCM: Select, Cross, Mutate.

    Attributes
    ----------
    meta_parameter: GeneticAlgorithmParameters
        The parameters of the genetic algorithm.
    cost_function: CostFunctionProtocol
        The cost function to optimize.
    client: Client | None
        The Dask client to use for parallel computing. If None, the algorithm will run in serial.
    constraint: Sequence[ConstraintProtocol] | None
        The constraints to apply to the individuals. If None, no constraints are applied.
    save: PathLike | None
        The path to save the logbook (in parquet format). If None, the logbook is not saved.

    """

    meta_parameter: GeneticAlgorithmParameters
    cost_function: CostFunctionProtocol
    client: Client | None = None
    constraint: Sequence[ConstraintProtocol] | None = None

    save: FilePath | WriteBuffer[bytes] | None = None
    logbook: OptimizationLog | None = field(default=None, repr=False)
    toolbox: base.Toolbox | None = field(default=None, init=False, repr=False)

    def __post_init__(self: GeneticAlgorithm) -> None:
        """Check parameters."""
        # logbook is now always XarrayLogbook or None

        if self.save is not None:
            self.save = Path(self.save)
            if self.save.exists():
                waring_msg = f"Logbook file {self.save} already exists. It will be overwritten."
                logger.warning(waring_msg)

        ordered_parameters = self.cost_function.functional_groups.unique_functional_groups_parameters_ordered()
        self.toolbox = self.meta_parameter.generate_toolbox(ordered_parameters.values(), self.cost_function)

        if self.constraint is not None:
            for constraint in self.constraint:
                self.toolbox.decorate("evaluate", constraint.generate(list(ordered_parameters.keys())))

        if len(self.meta_parameter.cost_function_weight) != len(self.cost_function.observations):
            msg = (
                "The cost function weight must have the same length as the number of observations. "
                f"Got {len(self.meta_parameter.cost_function_weight)} and {len(self.cost_function.observations)}."
            )
            raise ValueError(msg)

    def update_logbook(self: GeneticAlgorithm, logbook: OptimizationLog) -> None:
        """Update the logbook with the new data and save to disk if a path is provided."""
        if self.logbook is None:
            self.logbook = logbook
        else:
            # Concatenate the entire datasets along the generation dimension
            combined_dataset = xr.concat([self.logbook.dataset, logbook.dataset], dim="generation", join="outer")
            self.logbook = OptimizationLog(combined_dataset)

        if self.save is not None:
            self.logbook.save_netcdf(str(self.save))

    def _evaluate(self: GeneticAlgorithm, individuals: Sequence, generation: int) -> OptimizationLog:
        """Evaluate the cost function of all new individuals and create XarrayLogbook."""

        def update_fitness(individuals: list) -> list:
            known = [ind.fitness.valid for ind in individuals]
            invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
            if self.client is None:
                fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            else:
                futures_results = self.client.map(self.toolbox.evaluate, invalid_ind)
                fitnesses = self.client.gather(futures_results)
            for ind, fit in zip(invalid_ind, fitnesses, strict=True):
                ind.fitness.values = fit
            return known

        known = update_fitness(individuals)

        # Extract parameter values and fitness
        individual_params = [list(ind) for ind in individuals]
        parameter_names = list(
            self.cost_function.functional_groups.unique_functional_groups_parameters_ordered().keys()
        )
        fitness_names = [obs.name for obs in self.cost_function.observations]

        # Extract fitness values
        fitness_values = [tuple(ind.fitness.values) for ind in individuals]

        # Create OptimizationLog
        logbook = OptimizationLog.from_individual(
            generation=generation,
            is_from_previous_generation=known,
            individual=individual_params,
            parameter_names=parameter_names,
            fitness_names=fitness_names,
        )

        # Update fitness values in the logbook
        logbook.update_fitness(generation, list(range(len(individuals))), fitness_values)

        return logbook

    def _initialization(self: GeneticAlgorithm) -> tuple[int, list[list]]:
        """Initialize the genetic algorithm. If a logbook is provided, it will load the last generation."""

        def create_first_generation() -> tuple[Literal[1], list[list]]:
            """Create the first generation (i.e. generation `0`) of individuals."""
            new_generation = 0
            population = self.toolbox.population(n=self.meta_parameter.POP_SIZE)
            logbook = self._evaluate(individuals=population, generation=new_generation)
            self.update_logbook(logbook)
            next_generation = new_generation + 1
            return next_generation, population

        def create_population_from_logbook(generation_data: xr.Dataset) -> list[list]:
            """Create a population from the logbook xarray Dataset."""
            individuals = []
            fitness_data = generation_data["fitness"].data
            param_data = generation_data["parameters"].data

            for ind_idx in range(len(param_data)):
                # Create individual with parameters
                individual_params = param_data[ind_idx].tolist()
                individual = self.toolbox.Individual(individual_params)

                # Set fitness if available
                ind_fitness = fitness_data[ind_idx]
                if not np.any(np.isnan(ind_fitness)):
                    individual.fitness.values = tuple(ind_fitness)

                individuals.append(individual)

            return individuals

        if self.logbook is None:
            return create_first_generation()

        logger.info("OptimizationLog found. Loading last generation.")

        last_generation = max(self.logbook.generations)
        generation_data = self.logbook.sel_generation(last_generation)

        population = create_population_from_logbook(generation_data)

        # Check if re-evaluation is needed
        fitness_data = generation_data["weighted_fitness"]
        if np.any(np.isnan(fitness_data.values)):
            logger.warning("Some individuals in the logbook have no fitness values. Re-evaluating the population.")
            logbook = self._evaluate(population, last_generation)
            # Replace the generation data in the logbook
            self.logbook = None
            self.update_logbook(logbook)

        return last_generation + 1, population

    def optimize(self: GeneticAlgorithm) -> OptimizationLog:
        """This is the main function. Use it to optimize your model."""
        generation_start, population = self._initialization()

        for gen in range(generation_start, self.meta_parameter.NGEN):
            log_message = f"Generation {gen} / {self.meta_parameter.NGEN}."
            logger.info(log_message)
            offspring = self.toolbox.select(population, self.meta_parameter.POP_SIZE)
            offspring = self.meta_parameter.variation(
                offspring, self.toolbox, self.meta_parameter.CXPB, self.meta_parameter.MUTPB
            )
            logbook = self._evaluate(offspring, gen)

            self.update_logbook(logbook)
            population[:] = offspring

        return self.logbook.copy()
