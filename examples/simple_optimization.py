"""Simple optimization example using NoTransport model with synthetic data."""
# ruff: noqa: T201, PLR0915, PD011, NPY002

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from seapopym.configuration.no_transport import ForcingParameter, ForcingUnit, KernelParameter
from seapopym.model import NoTransportModel

from seapopym_optimization.algorithm.genetic_algorithm.factory import GeneticAlgorithmFactory
from seapopym_optimization.algorithm.genetic_algorithm.genetic_algorithm import GeneticAlgorithmParameters
from seapopym_optimization.algorithm.genetic_algorithm.logbook import OptimizationLog
from seapopym_optimization.configuration_generator.no_transport_configuration_generator import (
    NoTransportConfigurationGenerator,
)
from seapopym_optimization.cost_function import TimeSeriesScoreProcessor
from seapopym_optimization.cost_function.cost_function import CostFunction
from seapopym_optimization.cost_function.metric import rmse_comparator
from seapopym_optimization.functional_group import NoTransportFunctionalGroup, Parameter
from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet
from seapopym_optimization.functional_group.parameter_initialization import random_uniform_exclusive
from seapopym_optimization.observations.observation import DayCycle
from seapopym_optimization.observations.time_serie import TimeSeriesObservation

# Configure logging
logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("seapopym_optimization")
logger.setLevel(logging.INFO)


def main() -> None:
    """Run a simple optimization example."""
    # ======================
    # 1. Generate synthetic forcing data
    # ======================
    print("\n=== Generating synthetic forcing data ===")
    nb_days_by_year = 365
    nb_years = 2

    # Temperature: sinusoidal pattern
    temperature = xr.DataArray(
        data=(np.sin(np.linspace(0, (2 * np.pi) * nb_years, nb_days_by_year * nb_years)) * 5 + 20).reshape(
            (nb_days_by_year * nb_years, 1, 1, 1)
        ),
        dims=["time", "latitude", "longitude", "depth"],
        coords={
            "time": pd.date_range("2023-01-01", periods=nb_days_by_year * nb_years, freq="D"),
            "latitude": [0],
            "longitude": [0],
            "depth": [0],
        },
        name="temperature",
        attrs={
            "units": "Celsius",
            "long_name": "Sea surface temperature",
            "standard_name": "sea_surface_temperature",
        },
    )

    # Primary production: random + cosine pattern
    primary_production = xr.DataArray(
        data=(
            (
                np.random.rand(nb_days_by_year * nb_years).reshape((nb_days_by_year * nb_years, 1, 1))
                + (np.cos(np.linspace(0, np.pi * nb_years, nb_days_by_year * nb_years))).reshape(
                    (nb_days_by_year * nb_years, 1, 1)
                )
            )
            + 2
        )
        / 100,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": pd.date_range("2023-01-01", periods=nb_days_by_year * nb_years, freq="D"),
            "latitude": [0],
            "longitude": [0],
        },
        name="primary_production",
        attrs={
            "units": "kg/m^2/day",
            "long_name": "Primary production",
            "standard_name": "primary_production",
        },
    )

    # Set axis attributes
    temperature.time.attrs = {"axis": "T"}
    primary_production.time.attrs = {"axis": "T"}
    temperature.latitude.attrs = {"axis": "Y"}
    primary_production.latitude.attrs = {"axis": "Y"}
    temperature.longitude.attrs = {"axis": "X", "unit": "degrees_east"}
    primary_production.longitude.attrs = {"axis": "X"}
    temperature.depth.attrs = {"axis": "Z"}

    forcing_parameter = ForcingParameter(
        temperature=ForcingUnit(forcing=temperature),
        primary_production=ForcingUnit(forcing=primary_production),
    )

    print(f"Temperature range: {temperature.min().values:.2f} - {temperature.max().values:.2f} °C")
    print(f"Primary production range: {primary_production.min().values:.4f} - {primary_production.max().values:.4f}")

    # ======================
    # 2. Generate synthetic observations
    # ======================
    print("\n=== Generating synthetic observations ===")

    # Create a configuration generator
    configuration_generator = NoTransportConfigurationGenerator()

    # Run a model with known parameters to generate synthetic observations
    initial_config = configuration_generator.generate(
        functional_group_parameters=[
            NoTransportFunctionalGroup(
                name="Zooplankton",
                day_layer=0,
                night_layer=0,
                energy_transfert=0.1668,
                gamma_tr=-0.11,
                tr_0=10.38,
                gamma_lambda_temperature=0.15,
                lambda_temperature_0=1 / 150,
            )
        ],
        forcing_parameters=forcing_parameter,
        kernel=KernelParameter(),
    )

    with NoTransportModel.from_configuration(initial_config) as initial_model:
        initial_model.run()
        observed_biomass = initial_model.state.biomass

    # Prepare observation data
    observed_biomass = (
        observed_biomass.expand_dims({"layer": [0]}).isel(functional_group=0).drop_vars(["functional_group"])
    )
    observed_biomass.layer.attrs = {"axis": "Z"}

    print(f"Observed biomass range: {observed_biomass.min().values:.2e} - {observed_biomass.max().values:.2e}")

    # Create observation object
    observation = TimeSeriesObservation(
        name="Zooplankton Biomass", observation=observed_biomass, observation_type=DayCycle.DAY
    )

    # ======================
    # 3. Define functional groups for optimization
    # ======================
    print("\n=== Defining functional groups ===")

    functional_groups = [
        NoTransportFunctionalGroup(
            name="Zooplankton",
            day_layer=0,
            night_layer=0,
            energy_transfert=Parameter("D1N1_energy_transfert", 0.001, 0.3, init_method=random_uniform_exclusive),
            gamma_tr=Parameter("D1N1_gamma_tr", -0.3, -0.001, init_method=random_uniform_exclusive),
            tr_0=Parameter("D1N1_tr_0", 0, 30, init_method=random_uniform_exclusive),
            gamma_lambda_temperature=Parameter(
                "D1N1_gamma_lambda_temperature", 1 / 300, 1, init_method=random_uniform_exclusive
            ),
            lambda_temperature_0=Parameter("D1N1_lambda_temperature_0", 0, 0.3, init_method=random_uniform_exclusive),
        ),
    ]

    fg_set = FunctionalGroupSet(functional_groups=functional_groups)

    # ======================
    # 4. Create cost function with processor
    # ======================
    print("\n=== Creating cost function ===")

    # Create processor with RMSE metric
    processor = TimeSeriesScoreProcessor(comparator=rmse_comparator)

    # Create cost function
    cost_function = CostFunction(
        configuration_generator=configuration_generator,
        functional_groups=fg_set,
        forcing=forcing_parameter,
        kernel=KernelParameter(),
        observations=[observation],
        processor=processor,
    )

    print("Cost function created successfully")

    # ======================
    # 5. Initialize logbook with Sobol samples
    # ======================
    print("\n=== Initializing optimization logbook ===")

    logbook = OptimizationLog.from_sobol_samples(fg_set, sample_number=2, fitness_names=["Zooplankton Biomass"])

    # ======================
    # 6. Configure and run genetic algorithm
    # ======================
    print("\n=== Configuring genetic algorithm ===")

    metaparam = GeneticAlgorithmParameters(
        ETA=20,
        INDPB=0.2,
        CXPB=0.7,
        MUTPB=1,
        NGEN=5,
        POP_SIZE=10,
        cost_function_weight=(-1,),  # Minimize RMSE
    )

    print(f"Population size: {metaparam.POP_SIZE}")
    print(f"Number of generations: {metaparam.NGEN}")
    print(f"Crossover probability: {metaparam.CXPB}")
    print(f"Mutation probability: {metaparam.MUTPB}")

    # Initialize Dask client
    print("\n=== Starting Dask client ===")
    client = Client()
    print(f"Dask dashboard available at: {client.dashboard_link}")

    # Create distributed genetic algorithm
    # Factory automatically detects and distributes data if needed
    genetic_algorithm = GeneticAlgorithmFactory.create_distributed(
        meta_parameter=metaparam,
        cost_function=cost_function,
        client=client,
        logbook=logbook,
    )

    # ======================
    # 7. Run optimization
    # ======================
    print("\n=== Running optimization ===")
    print("This may take a few minutes...")

    optimization_results = genetic_algorithm.optimize()

    print("\n=== Optimization completed ===")
    print(f"Best fitness: {optimization_results.dataset.weighted_fitness.min().values:.6e}")
    print("\nOptimization results summary:")
    print(optimization_results)

    # Clean up
    client.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
