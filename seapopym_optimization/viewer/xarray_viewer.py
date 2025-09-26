"""XarrayViewer: xarray-based viewer for optimization results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots
from scipy.stats import entropy
from seapopym.standard.labels import ForcingLabels
from sklearn.preprocessing import QuantileTransformer

from seapopym_optimization.algorithm.genetic_algorithm.xarray_logbook import XarrayLogbook
from seapopym_optimization.cost_function.simple_cost_function import DayCycle, TimeSeriesObservation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Number

    from plotly.graph_objects import Figure

    from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet
    from seapopym_optimization.protocols import ModelGeneratorProtocol


@dataclass
class XarraySimulationManager:
    """Manages model simulations using xarray-based parameter sets."""

    logbook: XarrayLogbook
    model_generator: ModelGeneratorProtocol
    functional_groups: FunctionalGroupSet
    _cache: dict[tuple[int, int], xr.DataArray] = field(default_factory=dict)

    def run_individual(self, generation: int, individual: int) -> xr.DataArray:
        """Run simulation for a specific generation and individual."""
        cache_key = (generation, individual)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get parameters for this individual
        params = self.logbook.dataset['parameters'].sel(
            generation=generation, individual=individual
        ).values

        # Generate and run model
        model = self.model_generator.generate(
            functional_group_names=self.functional_groups.functional_groups_name(),
            functional_group_parameters=self.functional_groups.generate(params),
        )
        model.run()
        result = model.state[ForcingLabels.biomass]

        # Cache result
        self._cache[cache_key] = result
        return result

    def run_best_n(self, n: int) -> xr.DataArray:
        """Run simulations for the n best individuals across all generations."""
        # Get top n individuals
        best_individuals = self._get_best_individuals(n)

        results = []
        for gen, ind in best_individuals:
            result = self.run_individual(gen, ind)
            result = result.expand_dims({'simulation': [f'gen{gen}_ind{ind}']})
            results.append(result)

        return xr.concat(results, dim='simulation')

    def _get_best_individuals(self, n: int) -> list[tuple[int, int]]:
        """Get the n best individuals (generation, individual) based on fitness."""
        # Stack all individuals and sort by weighted_fitness
        fitness_flat = self.logbook.dataset['weighted_fitness'].stack(
            sample=['generation', 'individual']
        ).dropna('sample')

        # Get indices of n best individuals (assuming lower fitness is better)
        best_indices = fitness_flat.argsort()[:n]

        return [
            (int(fitness_flat[idx].generation), int(fitness_flat[idx].individual))
            for idx in best_indices
        ]


@dataclass
class XarrayViewer:
    """
    xarray-based viewer for genetic algorithm optimization results.

    Provides intuitive analysis and visualization of multidimensional optimization data
    using xarray's powerful indexing and computation capabilities.
    """

    logbook: XarrayLogbook
    functional_group_set: FunctionalGroupSet
    model_generator: ModelGeneratorProtocol
    observations: Sequence[TimeSeriesObservation]
    cost_function_weight: tuple[Number]

    simulation_manager: XarraySimulationManager = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the simulation manager."""
        self.simulation_manager = XarraySimulationManager(
            logbook=self.logbook,
            model_generator=self.model_generator,
            functional_groups=self.functional_group_set,
        )

    @classmethod
    def from_optimization_results(
        cls,
        logbook: XarrayLogbook,
        functional_group_set: FunctionalGroupSet,
        model_generator: ModelGeneratorProtocol,
        observations: Sequence[TimeSeriesObservation],
        cost_function_weight: tuple[Number],
    ) -> XarrayViewer:
        """Create XarrayViewer from optimization results."""
        return cls(
            logbook=logbook,
            functional_group_set=functional_group_set,
            model_generator=model_generator,
            observations=observations,
            cost_function_weight=cost_function_weight,
        )

    @property
    def parameter_names(self) -> list[str]:
        """Get parameter names."""
        return self.logbook.parameter_names

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Get parameter bounds from functional groups."""
        return {
            name: (param.lower_bound, param.upper_bound)
            for name, param in self.functional_group_set.unique_functional_groups_parameters_ordered().items()
        }

    def stats(self) -> pd.DataFrame:
        """
        Calculate statistics by generation.

        Returns
        -------
        pd.DataFrame
            Statistics with generations as index and metrics as columns
        """
        fitness_data = self.logbook.dataset['weighted_fitness']
        valid_fitness = fitness_data.where(np.isfinite(fitness_data))

        stats = xr.Dataset({
            'mean': valid_fitness.mean(dim='individual'),
            'std': valid_fitness.std(dim='individual'),
            'min': valid_fitness.min(dim='individual'),
            'max': valid_fitness.max(dim='individual'),
            'count': valid_fitness.count(dim='individual'),
        })

        # Add from_previous_generation ratio
        from_previous_ratio = (
            self.logbook.dataset['is_from_previous']
            .groupby('generation')
            .mean()
        )
        stats['from_previous_generation'] = from_previous_ratio

        return stats.to_dataframe()

    def hall_of_fame(self, n_best: int = 10, drop_duplicates: bool = True) -> XarrayLogbook:
        """
        Get the best n individuals from all generations.

        Parameters
        ----------
        n_best : int
            Number of best individuals to return
        drop_duplicates : bool
            Whether to remove duplicate parameter combinations

        Returns
        -------
        XarrayLogbook
            Logbook containing only the best individuals
        """
        # Stack all individuals
        fitness_flat = self.logbook.dataset['weighted_fitness'].stack(
            sample=['generation', 'individual']
        ).dropna('sample')

        # Get n best individuals (lower fitness is better)
        best_indices = fitness_flat.argsort()[:n_best]

        # Extract best individuals data
        best_generations = [int(fitness_flat[idx].generation) for idx in best_indices]
        best_individuals = [int(fitness_flat[idx].individual) for idx in best_indices]

        # Create new dataset with only best individuals
        best_data = {}
        for var_name in self.logbook.dataset.data_vars:
            var_data = self.logbook.dataset[var_name]
            if 'individual' in var_data.dims:
                # Select best individuals for this variable
                selected_data = []
                for gen, ind in zip(best_generations, best_individuals, strict=True):
                    selected_data.append(var_data.sel(generation=gen, individual=ind))

                # Stack into new array
                new_var = xr.concat(selected_data, dim='best_individual')
                new_var = new_var.assign_coords(best_individual=range(len(selected_data)))
            else:
                new_var = var_data

            best_data[var_name] = new_var

        # Create new coordinates
        new_coords = dict(self.logbook.dataset.coords)
        new_coords['best_individual'] = range(n_best)
        new_coords['generation'] = ('best_individual', best_generations)
        new_coords['individual'] = ('best_individual', best_individuals)

        # Create new dataset
        best_dataset = xr.Dataset(best_data, attrs=self.logbook.dataset.attrs)

        # Handle duplicate removal if requested
        if drop_duplicates:
            best_dataset = self._remove_duplicate_individuals(best_dataset)

        return XarrayLogbook(best_dataset)

    def _remove_duplicate_individuals(self, dataset: xr.Dataset) -> xr.Dataset:
        """Remove individuals with duplicate parameter combinations."""
        # Convert parameters to DataFrame for duplicate detection
        params_df = dataset['parameters'].to_dataframe()['parameters'].unstack('parameter')
        unique_mask = ~params_df.duplicated()

        if not unique_mask.all():
            # Select only unique individuals
            unique_indices = params_df[unique_mask].index
            dataset = dataset.sel(best_individual=unique_indices)

        return dataset

    def get_generation(self, generation: int) -> xr.Dataset:
        """Get data for a specific generation."""
        return self.logbook.dataset.sel(generation=generation)

    def get_parameter_evolution(self, parameter: str) -> xr.DataArray:
        """Get evolution of a specific parameter across generations."""
        return self.logbook.sel_parameter(parameter)

    def fitness_evolution(
        self,
        points: str = "best",
        absolute: bool = False,
        log_y: bool = False
    ) -> Figure:
        """
        Plot fitness evolution across generations.

        Parameters
        ----------
        points : str
            Type of points to show: 'best', 'mean', 'all'
        absolute : bool
            Whether to show absolute values
        log_y : bool
            Whether to use log scale for y-axis
        """
        fig = go.Figure()

        fitness_data = self.logbook.dataset['weighted_fitness']

        if absolute:
            fitness_data = np.abs(fitness_data)

        generations = self.logbook.generations

        if points == "all":
            # Show all points
            for gen in generations:
                gen_fitness = fitness_data.sel(generation=gen).dropna('individual')
                fig.add_trace(go.Scatter(
                    x=[gen] * len(gen_fitness),
                    y=gen_fitness.values,
                    mode='markers',
                    name=f'Gen {gen}',
                    opacity=0.6
                ))
        elif points == "best":
            # Show best individual per generation
            best_fitness = fitness_data.min(dim='individual')
            fig.add_trace(go.Scatter(
                x=generations,
                y=best_fitness.values,
                mode='lines+markers',
                name='Best fitness',
                line=dict(color='red', width=2)
            ))
        elif points == "mean":
            # Show mean fitness per generation
            mean_fitness = fitness_data.mean(dim='individual')
            fig.add_trace(go.Scatter(
                x=generations,
                y=mean_fitness.values,
                mode='lines+markers',
                name='Mean fitness',
                line=dict(color='blue', width=2)
            ))

        fig.update_layout(
            title="Fitness Evolution Across Generations",
            xaxis_title="Generation",
            yaxis_title="Fitness",
            yaxis_type="log" if log_y else "linear"
        )

        return fig

    def parameter_evolution_plot(self, parameter: str) -> Figure:
        """Plot evolution of a specific parameter across generations."""
        param_data = self.get_parameter_evolution(parameter)

        fig = go.Figure()

        # Plot all individuals as light points
        for gen in self.logbook.generations:
            gen_data = param_data.sel(generation=gen).dropna('individual')
            fig.add_trace(go.Scatter(
                x=[gen] * len(gen_data),
                y=gen_data.values,
                mode='markers',
                name=f'Gen {gen}',
                opacity=0.3,
                showlegend=False
            ))

        # Plot mean evolution as a line
        mean_evolution = param_data.mean(dim='individual')
        fig.add_trace(go.Scatter(
            x=self.logbook.generations,
            y=mean_evolution.values,
            mode='lines+markers',
            name=f'Mean {parameter}',
            line=dict(color='red', width=3)
        ))

        # Add parameter bounds if available
        if parameter in self.parameter_bounds:
            lower, upper = self.parameter_bounds[parameter]
            fig.add_hline(y=lower, line_dash="dash", line_color="gray",
                         annotation_text=f"Lower bound: {lower}")
            fig.add_hline(y=upper, line_dash="dash", line_color="gray",
                         annotation_text=f"Upper bound: {upper}")

        fig.update_layout(
            title=f"Evolution of Parameter: {parameter}",
            xaxis_title="Generation",
            yaxis_title=f"{parameter} Value"
        )

        return fig

    def parallel_coordinates(
        self,
        n_best: int = 100,
        reversescale: bool = False,
        unselected_opacity: float = 0.1,
        uniformed: bool = False
    ) -> tuple[Figure, pd.DataFrame]:
        """
        Create parallel coordinates plot of parameter values.

        Parameters
        ----------
        n_best : int
            Number of best individuals to include
        reversescale : bool
            Whether to reverse color scale
        unselected_opacity : float
            Opacity for unselected lines
        uniformed : bool
            Whether to normalize parameters to [0,1] range

        Returns
        -------
        tuple[Figure, pd.DataFrame]
            Plotly figure and DataFrame used for plotting
        """
        # Get best individuals
        hall_of_fame = self.hall_of_fame(n_best=n_best)

        # Convert to DataFrame
        df = hall_of_fame.to_pandas()

        # Prepare data for parallel coordinates
        param_cols = [f'param_{param}' for param in self.parameter_names]
        plot_data = df[param_cols + ['weighted_fitness']].copy()

        # Normalize parameters if requested
        if uniformed:
            scaler = QuantileTransformer(n_quantiles=min(1000, len(plot_data)))
            plot_data[param_cols] = scaler.fit_transform(plot_data[param_cols])

        # Create parallel coordinates plot
        dimensions = []
        for param in self.parameter_names:
            col_name = f'param_{param}'
            dimensions.append(dict(
                label=param,
                values=plot_data[col_name],
                range=[plot_data[col_name].min(), plot_data[col_name].max()]
            ))

        # Add fitness dimension
        dimensions.append(dict(
            label='Fitness',
            values=plot_data['weighted_fitness'],
            range=[plot_data['weighted_fitness'].min(), plot_data['weighted_fitness'].max()]
        ))

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=plot_data['weighted_fitness'],
                colorscale='Viridis' if not reversescale else 'Viridis_r',
                showscale=True,
                colorbar=dict(title="Fitness")
            ),
            dimensions=dimensions,
            unselected=dict(opacity=unselected_opacity)
        ))

        fig.update_layout(
            title=f"Parallel Coordinates Plot - Top {n_best} Individuals",
            height=600
        )

        return fig, plot_data

    def box_plot(self, generation: int, n_best: int = 50) -> Figure:
        """
        Create box plot of parameter distributions for a specific generation.

        Parameters
        ----------
        generation : int
            Generation to analyze
        n_best : int
            Number of best individuals to include

        Returns
        -------
        Figure
            Plotly box plot figure
        """
        # Get data for specific generation
        gen_data = self.get_generation(generation)

        # Sort by fitness and take n_best
        fitness = gen_data['weighted_fitness'].dropna('individual')
        best_indices = fitness.argsort()[:n_best]

        best_params = gen_data['parameters'].isel(individual=best_indices)

        fig = go.Figure()

        for param in self.parameter_names:
            param_values = best_params.sel(parameter=param).values
            fig.add_trace(go.Box(
                y=param_values,
                name=param,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))

        fig.update_layout(
            title=f"Parameter Distribution - Generation {generation} (Top {n_best})",
            yaxis_title="Parameter Value",
            xaxis_title="Parameters"
        )

        return fig

    def shannon_entropy(self, bins: int = 1000) -> pd.DataFrame:
        """
        Calculate Shannon entropy for each parameter across generations.

        Parameters
        ----------
        bins : int
            Number of bins for histogram calculation

        Returns
        -------
        pd.DataFrame
            Shannon entropy values by generation and parameter
        """
        entropy_data = []

        for gen in self.logbook.generations:
            gen_data = self.get_generation(gen)
            gen_entropy = {}

            for param in self.parameter_names:
                param_values = gen_data['parameters'].sel(parameter=param).dropna('individual').values

                if len(param_values) > 1:
                    # Calculate histogram
                    hist, _ = np.histogram(param_values, bins=bins, density=True)
                    hist = hist[hist > 0]  # Remove zero bins

                    # Calculate entropy
                    param_entropy = entropy(hist, base=2)
                else:
                    param_entropy = 0.0

                gen_entropy[param] = param_entropy

            gen_entropy['generation'] = gen
            entropy_data.append(gen_entropy)

        return pd.DataFrame(entropy_data).set_index('generation')

    def time_series(self, n_simulations: int = 10) -> list[Figure]:
        """
        Generate time series plots for the best n individuals.

        Parameters
        ----------
        n_simulations : int
            Number of best individuals to simulate and plot

        Returns
        -------
        list[Figure]
            List of plotly figures showing time series for each observation
        """
        # Run simulations for best individuals
        simulation_results = self.simulation_manager.run_best_n(n_simulations)

        figures = []

        for obs in self.observations:
            fig = go.Figure()

            # Plot each simulation
            for sim_idx in range(n_simulations):
                sim_data = simulation_results.isel(simulation=sim_idx)
                sim_name = simulation_results.coords['simulation'][sim_idx].values

                # Match observation dimensions and plot
                if hasattr(obs.observation, 'time'):
                    fig.add_trace(go.Scatter(
                        x=sim_data.time,
                        y=sim_data.values.flatten(),
                        mode='lines',
                        name=f'Simulation {sim_name}',
                        opacity=0.7
                    ))

            # Add observed data
            if hasattr(obs.observation, 'time'):
                fig.add_trace(go.Scatter(
                    x=obs.observation.time,
                    y=obs.observation.values.flatten(),
                    mode='lines',
                    name='Observed',
                    line=dict(color='red', width=3)
                ))

            fig.update_layout(
                title=f"Time Series Comparison - {obs.name}",
                xaxis_title="Time",
                yaxis_title=obs.name
            )

            figures.append(fig)

        return figures

    def save_results(self, filepath: str) -> None:
        """Save optimization results to NetCDF file."""
        self.logbook.save_netcdf(filepath)

    @classmethod
    def load_results(
        cls,
        filepath: str,
        functional_group_set: FunctionalGroupSet,
        model_generator: ModelGeneratorProtocol,
        observations: Sequence[TimeSeriesObservation],
        cost_function_weight: tuple[Number],
    ) -> XarrayViewer:
        """Load optimization results from NetCDF file."""
        logbook = XarrayLogbook.load_netcdf(filepath)
        return cls(
            logbook=logbook,
            functional_group_set=functional_group_set,
            model_generator=model_generator,
            observations=observations,
            cost_function_weight=cost_function_weight,
        )