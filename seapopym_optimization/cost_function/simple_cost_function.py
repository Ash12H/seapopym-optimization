"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from functools import partial
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from pandas.tseries.frequencies import to_offset
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels

from seapopym_optimization.cost_function.base_cost_function import AbstractCostFunction
from seapopym_optimization.cost_function.base_observation import AbstractObservation, DayCycle

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TimeSeriesObservation(AbstractObservation):
    """
    The structure used to store the observations as a time series.

    Meaning that the observation is a time series of biomass values at a given location and layer.
    """

    name: str
    observation: xr.DataArray
    observation_type: DayCycle = DayCycle.DAY
    observation_interval: pd.offsets.BaseOffset | None = "1D"

    def __post_init__(self: TimeSeriesObservation) -> None:
        """Check that the observation data is complient with the format of the predicted biomass."""
        if not isinstance(self.observation, xr.DataArray):
            msg = "Observation must be an xarray DataArray."
            raise TypeError(msg)

        for coord in ["T", "X", "Y", "Z"]:
            if coord not in self.observation.cf.coords:
                msg = f"Coordinate {coord} must be in the observation Dataset."
                raise ValueError(msg)

        for coord in [CoordinatesLabels.X, CoordinatesLabels.Y, CoordinatesLabels.Z]:
            if self.observation.cf.coords[coord].data.size != 1:
                msg = (
                    f"Multiple {coord} coordinates found in the observation Dataset. "
                    "The observation must be a time series with a single X, Y and Z (i.e. Seapodym layer) coordinate."
                )
                raise NotImplementedError(msg)

        try:
            self.observation = self.observation.pint.quantify().pint.to(StandardUnitsLabels.biomass).pint.dequantify()
        except Exception as e:
            msg = (
                f"At least one variable is not convertible to {StandardUnitsLabels.biomass}, which is the unit of the "
                "predicted biomass."
            )
            raise ValueError(msg) from e

        if not isinstance(self.observation_interval, (pd.offsets.BaseOffset, type(None))):
            self.observation_interval = to_offset(self.observation_interval)

        if self.observation_interval is not None:
            self.observation = self.resample_data_by_observation_interval(self.observation)

    def resample_data_by_observation_interval(self: TimeSeriesObservation, data: xr.DataArray) -> xr.DataArray:
        """Resample the data according to the observation type."""
        return data.cf.resample(T=self.observation_interval).mean().cf.dropna("T", how="all")


def aggregate_biomass_by_layer(
    data: xr.DataArray,
    position: Sequence[int],
    name: str,
    layer_coordinates: Sequence[int],
    layer_coordinates_name: str = "layer",
) -> xr.DataArray:
    """Aggregate biomass data by layer coordinates."""
    layer_coord = xr.DataArray(
        np.asarray(position),
        dims=[CoordinatesLabels.functional_group],
        coords={CoordinatesLabels.functional_group: data[CoordinatesLabels.functional_group].data},
        name=layer_coordinates_name,
        attrs={"axis": "Z"},
    )
    return (
        data.assign_coords({layer_coordinates_name: layer_coord})
        .groupby(layer_coordinates_name)
        .sum(dim=CoordinatesLabels.functional_group)
        .reindex({layer_coordinates_name: layer_coordinates})
        .fillna(0)
        .rename(name)
    )


def root_mean_square_error(
    pred: xr.DataArray,
    obs: xr.DataArray,
    *,
    root: bool,
    centered: bool,
    normalized: bool,
) -> float:
    """Mean square error applied to xr.DataArray."""
    if centered:
        cost = float(((pred - pred.mean()) - (obs - obs.mean())) ** 2).mean()
    else:
        cost = float(((obs - pred) ** 2).mean())

    if root:
        cost = np.sqrt(cost)

    if normalized:
        cost /= float(obs.std())

    if not np.isfinite(cost):
        msg = (
            "Nan value in cost function. The observation cannot be compared to the prediction. Verify that "
            "coordinates are fitting both in space and time."
        )
        raise ValueError(msg)

    return cost


@dataclass(kw_only=True)
class SimpleCostFunction(AbstractCostFunction):
    """
    Generator of the cost function for the 'SeapoPym No Transport' model.

    Attributes
    ----------
    functional_groups: Sequence[FunctionalGroupOptimizeNoTransport]
        The list of functional groups.
    forcing_parameters : ForcingParameters
        Forcing parameters.
    observations : Sequence[Observation]
        Observations.
    evaluation_function : callable[xr.DataArray, xr.DataArray] -> xr.DataArray
        The evaluation function to use for the cost function. Defaults to root mean square error without normalization
        nor centering. Be careful, the 1st argument is the prediction and the 2nd is the observation.

    """

    observations: Sequence[TimeSeriesObservation]  # TODO(Jules): Should accept spatial observations?
    evaluation_function: callable[[xr.DataArray, xr.DataArray], xr.DataArray] = field(
        default=partial(root_mean_square_error, root=True, centered=False, normalized=False)
    )
    resample_prediction: bool = True

    def _cost_function(self: SimpleCostFunction, args: np.ndarray) -> tuple:
        model = self.model_generator.generate(
            functional_group_names=self.functional_groups.functional_groups_name(),
            functional_group_parameters=self.functional_groups.generate(args),
        )

        model.run()

        predicted_biomass = model.state[ForcingLabels.biomass]

        biomass_day = aggregate_biomass_by_layer(
            data=predicted_biomass,
            position=model.state[ConfigurationLabels.day_layer].data,
            name=DayCycle.DAY,
            layer_coordinates=model.state.cf[CoordinatesLabels.Z].data,  # TODO(Jules): layer_coordinates ?
        )
        biomass_night = aggregate_biomass_by_layer(
            data=predicted_biomass,
            position=model.state[ConfigurationLabels.night_layer].data,
            name=DayCycle.NIGHT,
            layer_coordinates=model.state.cf[CoordinatesLabels.Z].data,
        )

        def evaluate_observation(prediction: xr.DataArray, observation: xr.DataArray) -> xr.DataArray:
            if self.resample_prediction:
                prediction = observation.resample_data_by_observation_interval(prediction)
            return self.evaluation_function(prediction, observation)

        return tuple(
            evaluate_observation(
                prediction=(biomass_day if obs.observation_type == DayCycle.DAY else biomass_night),
                observation=obs.observation,
            )
            for obs in self.observations
        )
