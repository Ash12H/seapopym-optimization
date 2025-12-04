"""This module contains the cost function used to optimize the parameters of the SeapoPym model version adapted to pteropods from Bednarsek equations, and considering Berverton Holt density dependance."""

from __future__ import annotations

from dataclasses import dataclass

from seapopym_optimization.functional_group.base_functional_group import AbstractFunctionalGroup, Parameter


@dataclass
class PteropodBedBHFunctionalGroup(AbstractFunctionalGroup):
    """The parameters of a functional group as they are defined in the SeapoPym pteropod Bednarsek model, and considering Berverton Holt density dependance."""

    day_layer: float | Parameter
    night_layer: float | Parameter
    energy_transfert: float | Parameter
    lambda_0: float | Parameter
    gamma_lambda_temperature: float | Parameter
    gamma_lambda_acidity: float | Parameter
    survival_rate_0: float | Parameter
    gamma_survival_rate_temperature: float | Parameter
    gamma_survival_rate_acidity: float | Parameter
    tr_0: float | Parameter
    gamma_tr: float | Parameter
    density_dependance_parameter: float | Parameter
