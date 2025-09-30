"""Observation processing components for cost function evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

from seapopym_optimization.observations.observation import DayCycle

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Number

    from seapopym.standard import SeapopymState

    from seapopym_optimization.cost_function.metric import MetricProtocol
    from seapopym_optimization.observations.time_serie import TimeSeriesObservation
    from seapopym_optimization.protocols import ObservationProtocol


# NOTE(Jules): This function will be used in the future to aggregate biomass by layer so we can compute score for
# spatial observations.
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


class AbstractScoreProcessor(ABC):
    """Abstract class for processing model state and observations to return a score."""

    def __init__(self, comparator: MetricProtocol[xr.DataArray, ObservationProtocol]) -> None:
        """Initialize with a comparator metric."""
        self.comparator = comparator

    @abstractmethod
    def process(self, state: SeapopymState, observation: ObservationProtocol) -> Number:
        """Process model state and observation to return a score."""


class TimeSeriesScoreProcessor(AbstractScoreProcessor):
    """Processes observations in time series format by applying preprocessing and comparison metrics."""

    def _pre_process_prediction(
        self, prediction: xr.DataArray, observation: TimeSeriesObservation, fg_positions: Sequence[int]
    ) -> xr.DataArray:
        """Pre-process prediction to match observation dimensions."""
        prediction = prediction.pint.quantify().pint.to(observation.observation.units).pint.dequantify()
        selected = prediction.sel(
            {
                CoordinatesLabels.functional_group: fg_positions,
                CoordinatesLabels.time: observation.observation[CoordinatesLabels.time],
                CoordinatesLabels.X: observation.observation[CoordinatesLabels.X],
                CoordinatesLabels.Y: observation.observation[CoordinatesLabels.Y],
            },
        )
        # Sum over functional_group dimension, squeeze size-1 dimensions
        summed = selected.sum(CoordinatesLabels.functional_group)
        return summed.squeeze()

    def process(self, state: SeapopymState, observation: TimeSeriesObservation) -> Number:
        """Compare prediction with observation by applying the comparator. Can pre-process data if needed."""
        if observation.observation_type is DayCycle.DAY:
            positions = state[ConfigurationLabels.day_layer]
        elif observation.observation_type is DayCycle.NIGHT:
            positions = state[ConfigurationLabels.night_layer]
        else:
            msg = f"Unknown observation type: {observation.observation_type}"
            raise ValueError(msg)

        prediction = self._pre_process_prediction(state[ForcingLabels.biomass], observation, positions)
        return self.comparator(prediction, observation)
