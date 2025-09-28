"""
Stratégies d'évaluation pour l'algorithme génétique.

Ce module définit différentes stratégies d'évaluation (séquentielle, parallèle, distribuée)
selon le pattern Strategy, permettant de changer dynamiquement le mode d'exécution
sans modifier la logique métier de l'algorithme génétique.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deap import base

logger = logging.getLogger(__name__)


class EvaluationStrategy(ABC):
    """
    Interface abstraite pour les stratégies d'évaluation.

    Le pattern Strategy permet de définir une famille d'algorithmes d'évaluation,
    de les encapsuler et de les rendre interchangeables. Cela permet à l'algorithme
    génétique de varier indépendamment des clients qui l'utilisent.
    """

    @abstractmethod
    def evaluate(self, individuals: Sequence, toolbox: base.Toolbox) -> list:
        """
        Évalue une liste d'individus.

        Parameters
        ----------
        individuals : Sequence
            Liste des individus à évaluer
        toolbox : base.Toolbox
            Toolbox DEAP avec la fonction d'évaluation

        Returns
        -------
        list
            Liste des fitness calculées

        Raises
        ------
        NotImplementedError
            Si la méthode n'est pas implémentée dans la classe dérivée

        """

    def __str__(self) -> str:
        """Représentation string de la stratégie."""
        return self.__class__.__name__


class SequentialEvaluation(EvaluationStrategy):
    """
    Stratégie d'évaluation séquentielle classique.

    Utilise la fonction map() standard de Python pour évaluer
    les individus un par un de manière séquentielle.
    """

    def evaluate(self, individuals: Sequence, toolbox: base.Toolbox) -> list:
        """
        Évaluation séquentielle avec map() standard.

        Parameters
        ----------
        individuals : Sequence
            Liste des individus à évaluer
        toolbox : base.Toolbox
            Toolbox DEAP avec la fonction d'évaluation

        Returns
        -------
        list
            Liste des fitness calculées

        """
        logger.debug("Évaluation séquentielle de %d individus", len(individuals))
        return list(map(toolbox.evaluate, individuals))






