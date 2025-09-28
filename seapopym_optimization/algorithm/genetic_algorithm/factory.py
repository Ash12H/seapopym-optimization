"""
Factory pour créer des instances GeneticAlgorithm configurées.

Ce module fournit des factory methods pour simplifier la création
d'instances GeneticAlgorithm avec différentes stratégies d'évaluation,
cachant la complexité de configuration pour les utilisateurs métier.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seapopym_optimization.algorithm.genetic_algorithm.genetic_algorithm import (
        GeneticAlgorithm,
        GeneticAlgorithmParameters,
    )
    from seapopym_optimization.protocols import CostFunctionProtocol

logger = logging.getLogger(__name__)


class GeneticAlgorithmFactory:
    """
    Factory pour créer des instances GeneticAlgorithm avec différentes configurations.

    Cette factory simplifie la création d'algorithmes génétiques en encapsulant
    la logique de configuration des stratégies d'évaluation et de distribution.
    """

    @staticmethod
    def create_sequential(
        meta_parameter: GeneticAlgorithmParameters,
        cost_function: CostFunctionProtocol,
        **kwargs
    ) -> GeneticAlgorithm:
        """
        Crée un GA en mode séquentiel.

        Mode d'évaluation le plus simple, adaptés aux petites populations
        ou aux situations où la parallélisation n'est pas nécessaire.

        Parameters
        ----------
        meta_parameter : GeneticAlgorithmParameters
            Paramètres de l'algorithme génétique
        cost_function : CostFunctionProtocol
            Fonction de coût à optimiser
        **kwargs
            Arguments supplémentaires pour GeneticAlgorithm

        Returns
        -------
        GeneticAlgorithm
            Instance configurée en mode séquentiel

        Examples
        --------
        >>> ga = GeneticAlgorithmFactory.create_sequential(meta_params, cost_function)
        >>> results = ga.optimize()

        """
        from seapopym_optimization.algorithm.genetic_algorithm.evaluation_strategies import SequentialEvaluation
        from seapopym_optimization.algorithm.genetic_algorithm.genetic_algorithm import GeneticAlgorithm

        logger.info("Création d'un algorithme génétique en mode séquentiel")

        return GeneticAlgorithm(
            meta_parameter=meta_parameter,
            cost_function=cost_function,
            evaluation_strategy=SequentialEvaluation(),
            **kwargs
        )




