"""This module contains the cost function used to optimize the parameters of the SeapoPym model version adapted to pteropods from Bednarsek equations, and considering Berverton Holt density dependance."""

from __future__ import annotations

from dataclasses import dataclass

from seapopym_optimization.functional_group import PteropodBedFunctionalGroup, Parameter


@dataclass
class PteropodBedBHFunctionalGroup(PteropodBedFunctionalGroup):
    """The parameters of a functional group as they are defined in the SeapoPym pteropod Bednarsek model, and considering Berverton Holt density dependance."""

    density_dependance_parameter: float | Parameter
