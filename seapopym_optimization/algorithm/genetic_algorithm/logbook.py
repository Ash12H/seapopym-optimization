"""xarray-based Logbook implementation for genetic algorithm optimization results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from seapopym_optimization.functional_group.parameter_initialization import initialize_with_sobol_sampling

if TYPE_CHECKING:
    from collections.abc import Sequence

    from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet


class OptimizationLog:
    """
    xarray-based Logbook for storing genetic algorithm optimization results.

    Provides a more structured and intuitive interface compared to pandas MultiIndex
    for multidimensional optimization data.

    Structure:
    - Dimensions: generation, individual, parameter, objective
    - Data variables: parameters, fitness, weighted_fitness, is_from_previous
    - Coordinates: parameter names, objective names, generation/individual indices
    - Attributes: algorithm metadata, parameter bounds, etc.
    """

    def __init__(self, dataset: xr.Dataset):
        """Initialize XarrayLogbook with an xarray Dataset."""
        self.dataset = dataset

    @classmethod
    def from_individual(
        cls,
        generation: int,
        is_from_previous_generation: list[bool],
        individual: list[list],
        parameter_names: list[str],
        fitness_names: list[str],
        algorithm_metadata: dict | None = None,
    ) -> OptimizationLog:
        """
        Create XarrayLogbook from individual data (equivalent to pandas Logbook.from_individual).

        Parameters
        ----------
        generation : int
            Generation number
        is_from_previous_generation : list[bool]
            Whether each individual comes from previous generation
        individual : list[list]
            List of parameter values for each individual
        parameter_names : list[str]
            Names of parameters
        fitness_names : list[str]
            Names of fitness objectives
        algorithm_metadata : dict, optional
            Additional metadata about the algorithm

        Returns
        -------
        OptimizationLog
            New logbook instance

        """
        n_individuals = len(individual)
        n_parameters = len(parameter_names)
        n_objectives = len(fitness_names)

        # Create parameter data array
        param_data = np.array(individual).reshape(1, n_individuals, n_parameters)
        parameters = xr.DataArray(
            param_data,
            dims=["generation", "individual", "parameter"],
            coords={
                "generation": [generation],
                "individual": range(n_individuals),
                "parameter": parameter_names,
            },
            name="parameters",
        )

        # Create empty fitness arrays
        fitness = xr.DataArray(
            np.full((1, n_individuals, n_objectives), np.nan),
            dims=["generation", "individual", "objective"],
            coords={
                "generation": [generation],
                "individual": range(n_individuals),
                "objective": fitness_names,
            },
            name="fitness",
        )

        weighted_fitness = xr.DataArray(
            np.full((1, n_individuals), np.nan),
            dims=["generation", "individual"],
            coords={
                "generation": [generation],
                "individual": range(n_individuals),
            },
            name="weighted_fitness",
        )

        is_from_previous = xr.DataArray(
            np.array(is_from_previous_generation).reshape(1, n_individuals),
            dims=["generation", "individual"],
            coords={
                "generation": [generation],
                "individual": range(n_individuals),
            },
            name="is_from_previous",
        )

        # Create dataset
        dataset = xr.Dataset(
            {
                "parameters": parameters,
                "fitness": fitness,
                "weighted_fitness": weighted_fitness,
                "is_from_previous": is_from_previous,
            },
            attrs=algorithm_metadata or {},
        )

        return cls(dataset)

    def update_fitness(
        self,
        generation: int,
        individual_indices: list[int],
        fitness_values: list[tuple],
    ) -> None:
        """Update fitness values for specific individuals."""
        for ind_idx, fitness_tuple in zip(individual_indices, fitness_values, strict=True):
            # Update multi-objective fitness
            for obj_idx, fitness_val in enumerate(fitness_tuple):
                obj_name = self.objective_names[obj_idx]
                self.dataset["fitness"].loc[
                    {"generation": generation, "individual": ind_idx, "objective": obj_name}
                ] = fitness_val

            # Update weighted fitness (simple sum for now)
            weighted_val = sum(fitness_tuple) if not any(np.isnan(fitness_tuple)) else np.nan
            self.dataset["weighted_fitness"].loc[{"generation": generation, "individual": ind_idx}] = weighted_val

    @property
    def objective_names(self) -> list[str]:
        """Get objective names."""
        return list(self.dataset.coords["objective"].values)

    @property
    def generations(self) -> list[int]:
        """Get list of generation numbers."""
        return list(self.dataset.coords["generation"].values)

    def sel_generation(self, generation: int) -> xr.Dataset:
        """Select data for a specific generation."""
        return self.dataset.sel(generation=generation)

    def copy(self) -> OptimizationLog:
        """Create a copy of the logbook."""
        return OptimizationLog(self.dataset.copy())

    @classmethod
    def from_sobol_samples(
        cls,
        functional_group_parameters: Sequence | FunctionalGroupSet,
        sample_number: int,
        fitness_names: list[str],
    ) -> OptimizationLog:
        """
        Create OptimizationLog from Sobol samples (equivalent to generate_logbook_with_sobol_sampling).

        Parameters
        ----------
        functional_group_parameters : Sequence[AbstractFunctionalGroup] | FunctionalGroupSet
            Functional group parameters for sampling
        sample_number : int
            N parameter used by the SALib sample_sobol method. The number of generated samples
            is equal to N * (D + 2), where D is the number of parameters.
        fitness_names : list[str]
            Names of fitness objectives
        generation : int, default 0
            Generation number for the samples

        Returns
        -------
        OptimizationLog
            New logbook instance with Sobol samples

        """
        samples = initialize_with_sobol_sampling(functional_group_parameters, sample_number)

        individual_data = [row.tolist() for _, row in samples.iterrows()]
        parameter_names = samples.columns.tolist()
        is_from_previous = [False] * len(samples)

        return cls.from_individual(
            generation=0,
            is_from_previous_generation=is_from_previous,
            individual=individual_data,
            parameter_names=parameter_names,
            fitness_names=fitness_names,
        )

    def save(self, filepath: str, engine: str = "zarr") -> None:
        """Save logbook to NetCDF file."""
        if engine not in ["zarr", "netcdf"]:
            msg = f"Engine should be 'zarr' or 'netcdf', got '{engine}'."
            raise ValueError(msg)

        attrs = {k: str(v) for k, v in self.dataset.attrs.items()}
        dataset_copy = self.dataset.copy()
        dataset_copy.attrs = attrs
        if engine == "zarr":
            dataset_copy.to_zarr(filepath, mode="w")
        else:
            dataset_copy.to_netcdf(filepath, mode="w")

    def __repr__(self) -> str:
        """String representation of the logbook."""
        return self.dataset.__repr__()
