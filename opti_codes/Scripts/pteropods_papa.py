import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client

from seapopym.configuration.acidity import (
    ForcingParameter,
)

from seapopym.configuration.acidity_bed import (
    AcidityBedConfiguration,
    FunctionalGroupParameter,
    FunctionalGroupUnit,
    FunctionalTypeParameter,
)

from seapopym.model import AcidityBedModel

from seapopym.standard.coordinate_authority import (
    create_latitude_coordinate,
    create_layer_coordinate,
    create_longitude_coordinate,
    create_time_coordinate,
)
from seapopym.standard.units import StandardUnitsLabels

from seapopym.configuration.no_transport import ForcingUnit, KernelParameter

from seapopym_optimization.algorithm.genetic_algorithm.factory import GeneticAlgorithmFactory
from seapopym_optimization.algorithm.genetic_algorithm.genetic_algorithm import GeneticAlgorithmParameters
from seapopym_optimization.algorithm.genetic_algorithm.logbook import Logbook
from seapopym_optimization.configuration_generator.pteropods_bed_configuration_generator import (
    PteropodsBedConfigurationGenerator,
)
from seapopym_optimization.algorithm.genetic_algorithm.logbook import Logbook, LogbookCategory, LogbookIndex
from seapopym_optimization.cost_function import TimeSeriesScoreProcessor
from seapopym_optimization.cost_function.cost_function import CostFunction
from seapopym_optimization.cost_function.metric import rmse_comparator
from seapopym_optimization.functional_group import PteropodBedFunctionalGroup, Parameter
from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet
from seapopym_optimization.functional_group.parameter_initialization import random_uniform_exclusive
from seapopym_optimization.observations.observation import DayCycle
from seapopym_optimization.observations.time_serie import TimeSeriesObservation

logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("seapopym_optimization")
logger.setLevel(logging.INFO)


def main():
    client = Client(memory_limit="10GB", n_workers=4)

    data_path = "/home/salbernhe/workspace/pteropod/data/papa_forcings/"
    aragonite_file = data_path + "daily_interp_aragonite_papa_1998_2020.nc"
    temperature_file = data_path + "daily_temp_papa_1998_2020.nc"
    npp_file = data_path + "daily_pp_papa_1998_2020.nc"

    data_aragonite = xr.open_dataset(aragonite_file)
    data_temperature = xr.open_dataset(temperature_file)
    data_npp = xr.open_dataset(npp_file)

    infos_forcings = [
        {
            "Variable": "Aragonite",
            "Shape": data_aragonite["omega_ar"].shape,
            "Unique values": len(np.unique(data_aragonite["omega_ar"])),
            "NaNs": int(np.isnan(data_aragonite["omega_ar"]).sum()),
        },
        {
            "Variable": "Temperature",
            "Shape": data_temperature["T"].shape,
            "Unique values": len(np.unique(data_temperature["T"])),
            "NaNs": int(np.isnan(data_temperature["T"]).sum()),
        },
        {
            "Variable": "NPP",
            "Shape": data_npp["npp"].shape,
            "Unique values": len(np.unique(data_npp["npp"])),
            "NaNs": int(np.isnan(data_npp["npp"]).sum()),
        },
    ]

    temperature = xr.DataArray(
        dims=["T", "Y", "X", "Z"],
        coords={
            "T": create_time_coordinate(pd.to_datetime(data_temperature["time"].values)),
            "Y": create_latitude_coordinate([0]),
            "X": create_longitude_coordinate([0]),
            "Z": create_layer_coordinate([0]),
        },
        attrs={"units": StandardUnitsLabels.temperature},
        data=data_temperature["T"].values[:, np.newaxis, np.newaxis, np.newaxis],
    )

    acidity = xr.DataArray(
        dims=["T", "Y", "X", "Z"],
        coords={
            "T": create_time_coordinate(pd.to_datetime(data_aragonite["time"].values)),
            "Y": create_latitude_coordinate([0]),
            "X": create_longitude_coordinate([0]),
            "Z": create_layer_coordinate([0]),
        },
        attrs={"units": StandardUnitsLabels.acidity},
        data=data_aragonite["omega_ar"].values[:, np.newaxis, np.newaxis, np.newaxis],
    )

    primary_production = xr.DataArray(
        dims=["T", "Y", "X"],
        coords={
            "T": create_time_coordinate(pd.to_datetime(data_npp["time"].values)),
            "Y": create_latitude_coordinate([0]),
            "X": create_longitude_coordinate([0]),
        },
        attrs={"units": "mg/m2/day"},
        data=data_npp["npp"].values[:, np.newaxis, np.newaxis],
    )

    dataset = xr.Dataset({"temperature": temperature, "primary_production": primary_production, "acidity": acidity})

    dataset = dataset.dropna("T")  # drop les nan à chaque T (en l'occurence acidité)

    dataset = dataset.resample(T="1D").mean().interpolate_na(dim="T")

    forcing_parameter = ForcingParameter(
        temperature=ForcingUnit(forcing=dataset.temperature),
        primary_production=ForcingUnit(forcing=dataset.primary_production),
        acidity=ForcingUnit(forcing=dataset.acidity),
    )

    pteropod_file = (
        "/home/salbernhe/workspace/pteropod/data/pteropod_data/time_serie_pter_papa_1998_2020_clean_k_075.nc"
    )

    data_pteropod = xr.open_dataset(pteropod_file)

    observed_biomass = data_pteropod["pteropod_biomass_gm2"]

    observed_biomass = observed_biomass.expand_dims({"Z": [0], "Y": [0], "X": [0]})

    observed_biomass = observed_biomass.rename({"time": "T"})

    observed_biomass.name = "biomass"
    observed_biomass.attrs = {"units": "g/m2"}

    observation = TimeSeriesObservation(
        name="Pteropods Biomass", observation=observed_biomass, observation_type=DayCycle.DAY
    )

    epsilon = np.finfo(float).eps
    functional_groups = [
        PteropodBedFunctionalGroup(
            name="Pteropods",
            day_layer=0,
            night_layer=0,
            energy_transfert=Parameter("D1N1_energy_transfert", epsilon, 0.5, init_method=random_uniform_exclusive),
            lambda_0=-19.4,  # Bed et al.
            gamma_lambda_temperature=11.5,  # Bed et al.
            gamma_lambda_acidity=-32.7,  # Bed et al.
            survival_rate_0=13.49,  # Bed et al.
            gamma_survival_rate_temperature=-2.475,  # Bed et al.
            gamma_survival_rate_acidity=10.10,  # Bed et al.
            tr_0=Parameter("D1N1_tr_0", epsilon, 100, init_method=random_uniform_exclusive),
            gamma_tr=Parameter("D1N1_gamma_tr", -0.5, -epsilon, init_method=random_uniform_exclusive),
        ),
    ]

    fg_set = FunctionalGroupSet(functional_groups=functional_groups)

    # Create processor with RMSE metric
    processor = TimeSeriesScoreProcessor(comparator=rmse_comparator)

    # Create cost function
    cost_function = CostFunction(
        configuration_generator=PteropodsBedConfigurationGenerator(AcidityBedModel),
        functional_groups=fg_set,
        forcing=forcing_parameter,
        kernel=KernelParameter(),
        observations=[observation],
        processor=processor,
    )

    logbook = Logbook.from_sobol_samples(fg_set, sample_number=2048, fitness_names=["Pteropods Biomass"])

    metaparam = GeneticAlgorithmParameters(
        ETA=20,
        INDPB=0.33,
        CXPB=0.9,
        MUTPB=1,
        NGEN=10,
        POP_SIZE=10000,
        cost_function_weight=(-1,),  # Minimize RMSE
    )

    genetic_algorithm = GeneticAlgorithmFactory.create_distributed(
        meta_parameter=metaparam,
        cost_function=cost_function,
        client=client,
        logbook=logbook,
    )

    optimization_results = genetic_algorithm.optimize()

    best_idx = optimization_results[(LogbookCategory.WEIGHTED_FITNESS, LogbookCategory.WEIGHTED_FITNESS)].idxmax()
    best_individual = optimization_results.loc[best_idx]
    best_params = best_individual[LogbookCategory.PARAMETER]

    # Create functional group with optimized parameters
    optimized_fg = PteropodBedFunctionalGroup(
        name="Pteropods",
        day_layer=0,
        night_layer=0,
        energy_transfert=best_params["D1N1_energy_transfert"],
        lambda_0=-19.4,  # Bed et al.
        gamma_lambda_temperature=11.5,  # Bed et al.
        gamma_lambda_acidity=-32.7,  # Bed et al.
        survival_rate_0=13.49,  # Bed et al.
        gamma_survival_rate_temperature=-2.475,  # Bed et al.
        gamma_survival_rate_acidity=10.10,  # Bed et al.
        gamma_tr=best_params["D1N1_gamma_tr"],
        tr_0=best_params["D1N1_tr_0"],
    )

    # Generate configuration with optimized parameters
    optimized_config = PteropodsBedConfigurationGenerator(AcidityBedModel).generate(
        functional_group_parameters=[optimized_fg],
        forcing_parameters=forcing_parameter,
        kernel=KernelParameter(),
    )

    # Run model with optimized parameters
    with AcidityBedModel.from_configuration(optimized_config) as optimized_model:
        optimized_model.run()
        optimized_biomass = optimized_model.state.biomass

    optimized_biomass = (
        optimized_biomass.expand_dims({"layer": [0]}).isel(functional_group=0).drop_vars(["functional_group"])
    )
    optimized_biomass = optimized_biomass.pint.quantify().pint.to("gram/m2")

    optimized_biomass = optimized_biomass.pint.dequantify()

    optimized_biomass.to_netcdf("/home/salbernhe/workspace/pteropod/simulations_outputs/opti_biomass_papa_pytest.nc")

    print("Netcdf done")


if __name__ == "__main__":
    main()
