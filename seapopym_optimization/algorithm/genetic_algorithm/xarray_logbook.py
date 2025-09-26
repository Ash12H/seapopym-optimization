"""xarray-based Logbook implementation for genetic algorithm optimization results."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Sequence


class XarrayLogbook:
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
    ) -> XarrayLogbook:
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
        XarrayLogbook
            New logbook instance
        """
        n_individuals = len(individual)
        n_parameters = len(parameter_names)
        n_objectives = len(fitness_names)

        # Create parameter data array
        param_data = np.array(individual).reshape(1, n_individuals, n_parameters)
        parameters = xr.DataArray(
            param_data,
            dims=['generation', 'individual', 'parameter'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
                'parameter': parameter_names,
            },
            name='parameters'
        )

        # Create empty fitness arrays
        fitness = xr.DataArray(
            np.full((1, n_individuals, n_objectives), np.nan),
            dims=['generation', 'individual', 'objective'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
                'objective': fitness_names,
            },
            name='fitness'
        )

        weighted_fitness = xr.DataArray(
            np.full((1, n_individuals), np.nan),
            dims=['generation', 'individual'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
            },
            name='weighted_fitness'
        )

        is_from_previous = xr.DataArray(
            np.array(is_from_previous_generation).reshape(1, n_individuals),
            dims=['generation', 'individual'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
            },
            name='is_from_previous'
        )

        # Create dataset
        dataset = xr.Dataset(
            {
                'parameters': parameters,
                'fitness': fitness,
                'weighted_fitness': weighted_fitness,
                'is_from_previous': is_from_previous,
            },
            attrs=algorithm_metadata or {}
        )

        return cls(dataset)

    def add_generation(
        self,
        generation: int,
        is_from_previous_generation: list[bool],
        individual: list[list],
    ) -> None:
        """Add a new generation to the logbook."""
        n_individuals = len(individual)
        n_parameters = len(self.parameter_names)
        n_objectives = len(self.objective_names)

        # Create new generation data
        new_param_data = np.array(individual).reshape(1, n_individuals, n_parameters)
        new_parameters = xr.DataArray(
            new_param_data,
            dims=['generation', 'individual', 'parameter'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
                'parameter': self.parameter_names,
            }
        )

        new_fitness = xr.DataArray(
            np.full((1, n_individuals, n_objectives), np.nan),
            dims=['generation', 'individual', 'objective'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
                'objective': self.objective_names,
            }
        )

        new_weighted_fitness = xr.DataArray(
            np.full((1, n_individuals), np.nan),
            dims=['generation', 'individual'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
            }
        )

        new_is_from_previous = xr.DataArray(
            np.array(is_from_previous_generation).reshape(1, n_individuals),
            dims=['generation', 'individual'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
            }
        )

        # Concatenate with existing data (use join='outer' for different individual counts)
        self.dataset['parameters'] = xr.concat(
            [self.dataset['parameters'], new_parameters],
            dim='generation',
            join='outer'
        )
        self.dataset['fitness'] = xr.concat(
            [self.dataset['fitness'], new_fitness],
            dim='generation',
            join='outer'
        )
        self.dataset['weighted_fitness'] = xr.concat(
            [self.dataset['weighted_fitness'], new_weighted_fitness],
            dim='generation',
            join='outer'
        )
        self.dataset['is_from_previous'] = xr.concat(
            [self.dataset['is_from_previous'], new_is_from_previous],
            dim='generation',
            join='outer'
        )

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
                self.dataset['fitness'][generation, ind_idx, obj_idx] = fitness_val

            # Update weighted fitness (simple sum for now)
            weighted_val = sum(fitness_tuple) if not any(np.isnan(fitness_tuple)) else np.nan
            self.dataset['weighted_fitness'][generation, ind_idx] = weighted_val

    @property
    def parameter_names(self) -> list[str]:
        """Get parameter names."""
        return list(self.dataset.coords['parameter'].values)

    @property
    def objective_names(self) -> list[str]:
        """Get objective names."""
        return list(self.dataset.coords['objective'].values)

    @property
    def generations(self) -> list[int]:
        """Get list of generation numbers."""
        return list(self.dataset.coords['generation'].values)

    @property
    def n_individuals_per_generation(self) -> dict[int, int]:
        """Get number of individuals per generation."""
        return {
            int(gen): int(self.dataset.sel(generation=gen).dims['individual'])
            for gen in self.generations
        }

    def sel_generation(self, generation: int) -> xr.Dataset:
        """Select data for a specific generation."""
        return self.dataset.sel(generation=generation)

    def sel_parameter(self, parameter: str) -> xr.DataArray:
        """Select data for a specific parameter across all generations."""
        return self.dataset['parameters'].sel(parameter=parameter)

    def copy(self) -> XarrayLogbook:
        """Create a copy of the logbook."""
        return XarrayLogbook(self.dataset.copy())

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame format for compatibility with existing code.

        Returns DataFrame with MultiIndex (generation, individual) and columns
        for parameters, fitness, and weighted_fitness.
        """
        # Convert to DataFrame and restructure
        df = self.dataset.to_dataframe()

        # Flatten parameter columns
        param_df = df['parameters'].unstack('parameter')
        param_df.columns = [f'param_{col}' for col in param_df.columns]

        # Flatten fitness columns
        fitness_df = df['fitness'].unstack('objective')
        fitness_df.columns = [f'fitness_{col}' for col in fitness_df.columns]

        # Combine all data
        result_df = pd.concat([
            param_df,
            fitness_df,
            df[['weighted_fitness', 'is_from_previous']]
        ], axis=1)

        return result_df

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, parameter_names: list[str], objective_names: list[str]) -> XarrayLogbook:
        """
        Create XarrayLogbook from pandas DataFrame (for migration from existing code).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with MultiIndex (generation, individual)
        parameter_names : list[str]
            Names of parameters
        objective_names : list[str]
            Names of objectives
        """
        # Extract generations and individuals from MultiIndex
        generations = df.index.get_level_values('generation').unique().sort_values()
        max_individuals = df.groupby('generation').size().max()

        # Initialize arrays
        n_gen, n_ind, n_param, n_obj = len(generations), max_individuals, len(parameter_names), len(objective_names)

        param_data = np.full((n_gen, n_ind, n_param), np.nan)
        fitness_data = np.full((n_gen, n_ind, n_obj), np.nan)
        weighted_fitness_data = np.full((n_gen, n_ind), np.nan)
        is_from_previous_data = np.full((n_gen, n_ind), False)

        # Fill data from DataFrame
        for gen_idx, gen in enumerate(generations):
            gen_data = df.loc[gen]
            n_gen_individuals = len(gen_data)

            # Parameters
            for param_idx, param in enumerate(parameter_names):
                if f'param_{param}' in gen_data.columns:
                    param_data[gen_idx, :n_gen_individuals, param_idx] = gen_data[f'param_{param}'].values

            # Fitness
            for obj_idx, obj in enumerate(objective_names):
                if f'fitness_{obj}' in gen_data.columns:
                    fitness_data[gen_idx, :n_gen_individuals, obj_idx] = gen_data[f'fitness_{obj}'].values

            # Weighted fitness and previous generation flag
            if 'weighted_fitness' in gen_data.columns:
                weighted_fitness_data[gen_idx, :n_gen_individuals] = gen_data['weighted_fitness'].values
            if 'is_from_previous' in gen_data.columns:
                is_from_previous_data[gen_idx, :n_gen_individuals] = gen_data['is_from_previous'].values

        # Create xarray Dataset
        dataset = xr.Dataset(
            {
                'parameters': (['generation', 'individual', 'parameter'], param_data),
                'fitness': (['generation', 'individual', 'objective'], fitness_data),
                'weighted_fitness': (['generation', 'individual'], weighted_fitness_data),
                'is_from_previous': (['generation', 'individual'], is_from_previous_data),
            },
            coords={
                'generation': generations,
                'individual': range(max_individuals),
                'parameter': parameter_names,
                'objective': objective_names,
            }
        )

        return cls(dataset)

    def save_netcdf(self, filepath: str) -> None:
        """Save logbook to NetCDF file."""
        # Convert complex attributes to strings for NetCDF compatibility
        attrs = {k: str(v) for k, v in self.dataset.attrs.items()}
        dataset_copy = self.dataset.copy()
        dataset_copy.attrs = attrs
        dataset_copy.to_netcdf(filepath)

    @classmethod
    def load_netcdf(cls, filepath: str) -> XarrayLogbook:
        """Load logbook from NetCDF file."""
        dataset = xr.open_dataset(filepath)
        return cls(dataset)

    def __len__(self) -> int:
        """Return total number of individuals across all generations."""
        return int(self.dataset['parameters'].size / len(self.parameter_names))

    def __repr__(self) -> str:
        """String representation of the logbook."""
        n_gen = len(self.generations)
        n_param = len(self.parameter_names)
        n_obj = len(self.objective_names)
        total_individuals = len(self)

        return (
            f"XarrayLogbook:\n"
            f"  Generations: {n_gen}\n"
            f"  Parameters: {n_param} ({', '.join(self.parameter_names)})\n"
            f"  Objectives: {n_obj} ({', '.join(self.objective_names)})\n"
            f"  Total individuals: {total_individuals}\n"
            f"  Dimensions: {dict(self.dataset.dims)}"
        )