import logging

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from seapopym.configuration.no_transport import ForcingParameter, ForcingUnit

from seapopym_optimization.algorithm.genetic_algorithm.genetic_algorithm import (
    GeneticAlgorithm,
    GeneticAlgorithmParameters,
)
from seapopym_optimization.algorithm.genetic_algorithm.logbook import OptimizationLog
from seapopym_optimization.cost_function.cost_function import CostFunction, DayCycle, TimeSeriesObservation
from seapopym_optimization.functional_group import NoTransportFunctionalGroup, Parameter
from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet
from seapopym_optimization.functional_group.parameter_initialization import random_uniform_exclusive
from seapopym_optimization.model_generator import NoTransportModelGenerator

logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("seapopym_optimization")


def create_functional_groups():
    """Create functional groups for optimization."""
    functional_groups = [
        NoTransportFunctionalGroup(
            name="Zooplankton",
            day_layer=0,
            night_layer=0,
            energy_transfert=Parameter("D1N1_energy_transfert energy_transfert", 0.001, 0.3, random_uniform_exclusive),
            gamma_tr=Parameter("D1N1_gamma_tr", -0.3, -0.001, random_uniform_exclusive),
            tr_0=Parameter("D1N1_tr_0", 0, 30, random_uniform_exclusive),
            gamma_lambda_temperature=Parameter("D1N1_gamma_lambda_temperature", 1 / 300, 1, random_uniform_exclusive),
            lambda_temperature_0=Parameter("D1N1_lambda_temperature_0", 0, 0.3, random_uniform_exclusive),
        ),
    ]
    return FunctionalGroupSet(functional_groups=functional_groups)


def create_forcing_data(nb_days_by_year=365, nb_years=2):
    """Create synthetic forcing data for temperature and primary production."""
    time_coords = pd.date_range("2023-01-01", periods=nb_days_by_year * nb_years, freq="D")

    # Temperature data
    temp_data = (np.sin(np.linspace(0, (2 * np.pi) * nb_years, nb_days_by_year * nb_years)) * 5 + 20).reshape(
        (nb_days_by_year * nb_years, 1, 1, 1)
    )
    temperature = xr.DataArray(
        data=temp_data,
        dims=["time", "latitude", "longitude", "depth"],
        coords={
            "time": time_coords,
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

    # Primary production data
    pp_data = (
        (
            np.random.rand(nb_days_by_year * nb_years).reshape((nb_days_by_year * nb_years, 1, 1))
            + (np.cos(np.linspace(0, np.pi * nb_years, nb_days_by_year * nb_years))).reshape(
                (nb_days_by_year * nb_years, 1, 1)
            )
        )
        + 2
    ) / 100

    primary_production = xr.DataArray(
        data=pp_data,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": time_coords,
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

    # Set coordinate attributes
    for data_array in [temperature, primary_production]:
        data_array.time.attrs = {"axis": "T"}
        data_array.latitude.attrs = {"axis": "Y"}
        data_array.longitude.attrs = {"axis": "X"}

    temperature.longitude.attrs["unit"] = "degrees_east"
    temperature.depth.attrs = {"axis": "Z"}

    return temperature, primary_production


def create_forcing_parameter(temperature, primary_production):
    """Create forcing parameter from temperature and primary production data."""
    return ForcingParameter(
        temperature=ForcingUnit(forcing=temperature),
        primary_production=ForcingUnit(forcing=primary_production),
    )


def generate_observed_biomass(forcing_parameter):
    """Generate observed biomass data for optimization target."""
    model_generator = NoTransportModelGenerator(forcing_parameters=forcing_parameter)

    initial_model = model_generator.generate(
        functional_group_parameters=[
            {
                "energy_transfert": 0.1668,
                "day_layer": 0,
                "night_layer": 0,
                "gamma_tr": -0.11,
                "tr_0": 10.38,
                "gamma_lambda_temperature": 0.15,
                "lambda_temperature_0": 1 / 150,
            }
        ],
        functional_group_names=["Zooplankton"],
    )
    initial_model.run()
    observed_biomass = initial_model.state.biomass
    observed_biomass = (
        observed_biomass.expand_dims({"layer": [0]}).isel(functional_group=0).drop_vars(["functional_group"])
    )
    observed_biomass.layer.attrs = {"axis": "Z"}
    return observed_biomass, model_generator


def run_optimization():
    """Main function to run the genetic algorithm optimization."""
    # Create functional groups
    fg_set = create_functional_groups()

    # Create forcing data
    temperature, primary_production = create_forcing_data()
    forcing_parameter = create_forcing_parameter(temperature, primary_production)

    # Generate observed biomass
    observed_biomass, model_generator = generate_observed_biomass(forcing_parameter)

    # Create observation and cost function
    observation = TimeSeriesObservation(
        name="Zooplankton Biomass", observation=observed_biomass, observation_type=DayCycle.DAY
    )
    cost_function = CostFunction(model_generator=model_generator, observations=[observation], functional_groups=fg_set)

    # Create logbook and genetic algorithm
    logbook = OptimizationLog.from_sobol_samples(
        fg_set, sample_number=2, fitness_names=["Zooplankton Biomass"], generation=0
    )
    metaparam = GeneticAlgorithmParameters(
        ETA=20, INDPB=0.2, CXPB=0.7, MUTPB=1, NGEN=5, POP_SIZE=10, cost_function_weight=(-1,)
    )
    genetic_algorithm = GeneticAlgorithm(
        meta_parameter=metaparam, cost_function=cost_function, client=Client(), logbook=logbook
    )

    # Run optimization
    optimization_results = genetic_algorithm.optimize()
    return optimization_results


if __name__ == "__main__":
    results = run_optimization()
    logger.info("Optimization completed successfully")
