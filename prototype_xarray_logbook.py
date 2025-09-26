"""Prototype d'un Logbook basé sur xarray.Dataset au lieu de pandas.DataFrame"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class XarrayLogbook:
    """
    Prototype de Logbook utilisant xarray.Dataset pour stocker les résultats d'optimisation.

    Structure des données:
    - Dimensions: generation, individual, parameter, objective
    - Variables: parameters, fitness, weighted_fitness
    - Coordonnées: noms des paramètres, objectifs, etc.
    - Attributs: métadonnées (bounds, algorithme, etc.)
    """

    def __init__(
        self,
        parameters: xr.DataArray,
        fitness: xr.DataArray | None = None,
        weighted_fitness: xr.DataArray | None = None,
        attrs: dict | None = None,
    ):
        """Initialiser le logbook xarray."""
        self.dataset = xr.Dataset(
            {
                'parameters': parameters,
                'fitness': fitness if fitness is not None else self._create_empty_fitness(parameters),
                'weighted_fitness': weighted_fitness if weighted_fitness is not None else self._create_empty_weighted_fitness(parameters),
            },
            attrs=attrs or {}
        )

    def _create_empty_fitness(self, parameters: xr.DataArray) -> xr.DataArray:
        """Créer un array vide pour les fitness."""
        fitness_shape = list(parameters.shape[:-1]) + [1]  # Remove last dim, add objective
        fitness_dims = list(parameters.dims[:-1]) + ['objective']
        fitness_coords = {dim: parameters.coords[dim] for dim in parameters.dims[:-1]}
        fitness_coords['objective'] = [0]  # Single objective for now

        return xr.DataArray(
            np.full(fitness_shape, np.nan, dtype=float),
            dims=fitness_dims,
            coords=fitness_coords
        )

    def _create_empty_weighted_fitness(self, parameters: xr.DataArray) -> xr.DataArray:
        """Créer un array vide pour les weighted_fitness."""
        return xr.full_like(
            parameters.isel(parameter=0),  # Drop parameter dimension
            np.nan,
            dtype=float
        )

    @classmethod
    def from_individual(
        cls,
        generation: int,
        is_from_previous_generation: list[bool],
        individual: list[list],
        parameter_names: list[str],
        fitness_name: list[str],
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> XarrayLogbook:
        """
        Créer un XarrayLogbook à partir d'une liste d'individus.

        Equivalent de la méthode from_individual du Logbook pandas.
        """
        n_individuals = len(individual)
        n_parameters = len(parameter_names)
        n_objectives = len(fitness_name)

        # Créer les données des paramètres
        param_data = np.array(individual).reshape(1, n_individuals, n_parameters)

        parameters = xr.DataArray(
            param_data,
            dims=['generation', 'individual', 'parameter'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
                'parameter': parameter_names,
                'is_from_previous_generation': (['generation', 'individual'],
                                              np.array(is_from_previous_generation).reshape(1, n_individuals))
            },
            name='parameters'
        )

        # Ajouter les métadonnées
        attrs = {
            'algorithm': 'SimpleGeneticAlgorithm',
            'parameter_names': parameter_names,
            'objective_names': fitness_name,
            'created_from_generation': generation,
        }

        if parameter_bounds:
            attrs['parameter_bounds'] = parameter_bounds

        return cls(parameters, attrs=attrs)

    def add_generation(
        self,
        generation: int,
        is_from_previous_generation: list[bool],
        individual: list[list],
    ) -> None:
        """Ajouter une nouvelle génération au logbook."""
        n_individuals = len(individual)
        n_parameters = len(self.parameter_names)

        # Créer les nouvelles données
        new_param_data = np.array(individual).reshape(1, n_individuals, n_parameters)

        new_parameters = xr.DataArray(
            new_param_data,
            dims=['generation', 'individual', 'parameter'],
            coords={
                'generation': [generation],
                'individual': range(n_individuals),
                'parameter': self.parameter_names,
                'is_from_previous_generation': (['generation', 'individual'],
                                              np.array(is_from_previous_generation).reshape(1, n_individuals))
            }
        )

        # Concaténer avec les données existantes
        self.dataset['parameters'] = xr.concat(
            [self.dataset['parameters'], new_parameters],
            dim='generation',
            join='outer'  # Permettre différentes tailles d'individus
        )

        # Étendre fitness et weighted_fitness
        new_fitness = self._create_empty_fitness(new_parameters)
        new_weighted_fitness = self._create_empty_weighted_fitness(new_parameters)

        self.dataset['fitness'] = xr.concat([self.dataset['fitness'], new_fitness], dim='generation', join='outer')
        self.dataset['weighted_fitness'] = xr.concat([self.dataset['weighted_fitness'], new_weighted_fitness], dim='generation', join='outer')

    def update_fitness(self, generation: int, individual_indices: list[int], fitness_values: list[tuple]) -> None:
        """Mettre à jour les valeurs de fitness pour des individus spécifiques."""
        for i, fitness_tuple in zip(individual_indices, fitness_values):
            # Pour fitness multi-objectifs, assigner chaque valeur
            for obj_idx, value in enumerate(fitness_tuple):
                self.dataset['fitness'].loc[dict(generation=generation, individual=i, objective=obj_idx)] = value

            # Calculer weighted_fitness (exemple simple: somme pondérée)
            weighted = sum(fitness_tuple) if not np.any(np.isnan(fitness_tuple)) else np.nan
            self.dataset['weighted_fitness'].loc[dict(generation=generation, individual=i)] = weighted

    @property
    def parameter_names(self) -> list[str]:
        """Noms des paramètres."""
        return list(self.dataset.coords['parameter'].values)

    @property
    def objective_names(self) -> list[str]:
        """Noms des objectifs."""
        return list(self.dataset.coords.get('objective', []))

    @property
    def generations(self) -> list[int]:
        """Liste des générations."""
        return list(self.dataset.coords['generation'].values)

    def stats(self) -> pd.DataFrame:
        """
        Calculer les statistiques par génération (équivalent à compute_stats).

        Exemple des capacités de xarray pour les opérations statistiques.
        """
        # Utiliser xarray pour calculer des stats sur weighted_fitness
        weights = self.dataset['weighted_fitness']

        # Filtrer les valeurs finies
        valid_weights = weights.where(np.isfinite(weights))

        # Calculer stats par génération
        stats = xr.Dataset({
            'mean': valid_weights.mean(dim='individual'),
            'std': valid_weights.std(dim='individual'),
            'min': valid_weights.min(dim='individual'),
            'max': valid_weights.max(dim='individual'),
            'count': valid_weights.count(dim='individual'),
        })

        # Convertir en DataFrame pour compatibilité
        return stats.to_dataframe()

    def hall_of_fame(self, n_best: int = 10) -> xr.Dataset:
        """
        Récupérer les meilleurs individus de toutes les générations.

        Exemple des capacités de slicing/sorting de xarray.
        """
        # Stack all individuals across generations
        stacked = self.dataset.stack(all_individuals=['generation', 'individual'])

        # Sort by weighted_fitness (descending for maximization)
        sorted_indices = stacked['weighted_fitness'].argsort()[-n_best:]

        return stacked.isel(all_individuals=sorted_indices)

    def get_generation(self, gen: int) -> xr.Dataset:
        """Récupérer une génération spécifique."""
        return self.dataset.sel(generation=gen)

    def get_parameter_evolution(self, param_name: str) -> xr.DataArray:
        """Récupérer l'évolution d'un paramètre à travers les générations."""
        return self.dataset['parameters'].sel(parameter=param_name)

    def copy(self) -> XarrayLogbook:
        """Créer une copie du logbook."""
        return XarrayLogbook(
            parameters=self.dataset['parameters'].copy(),
            fitness=self.dataset['fitness'].copy(),
            weighted_fitness=self.dataset['weighted_fitness'].copy(),
            attrs=self.dataset.attrs.copy()
        )

    def to_pandas(self) -> pd.DataFrame:
        """
        Convertir vers le format pandas MultiIndex pour compatibilité.

        Permet une migration progressive.
        """
        # Stack toutes les dimensions non-coordonnées
        stacked = self.dataset.stack(sample=['generation', 'individual'])

        # Convertir en DataFrame
        df = stacked.to_dataframe()

        # Restructurer pour correspondre au format Logbook original
        # (Nécessiterait plus de travail pour une correspondance exacte)
        return df


def demo_usage():
    """Démonstration de l'usage du XarrayLogbook."""

    # Paramètres d'exemple
    parameter_names = ['param_A', 'param_B', 'param_C']
    fitness_names = ['objective_1']
    parameter_bounds = {
        'param_A': (0.0, 1.0),
        'param_B': (-1.0, 1.0),
        'param_C': (0.0, 10.0)
    }

    # Créer des individus d'exemple
    individuals = [
        [0.5, 0.2, 5.0],
        [0.8, -0.5, 3.2],
        [0.1, 0.9, 8.1]
    ]
    is_from_previous = [False, False, False]

    # Créer le logbook
    logbook = XarrayLogbook.from_individual(
        generation=0,
        is_from_previous_generation=is_from_previous,
        individual=individuals,
        parameter_names=parameter_names,
        fitness_name=fitness_names,
        parameter_bounds=parameter_bounds
    )

    print("=== Structure du Dataset ===")
    print(logbook.dataset)

    print("\n=== Métadonnées ===")
    print(logbook.dataset.attrs)

    # Ajouter une génération
    new_individuals = [
        [0.6, 0.3, 4.8],
        [0.7, -0.2, 3.5],
    ]
    logbook.add_generation(
        generation=1,
        is_from_previous_generation=[True, True],
        individual=new_individuals
    )

    # Mettre à jour les fitness
    logbook.update_fitness(
        generation=0,
        individual_indices=[0, 1, 2],
        fitness_values=[(0.8,), (0.6,), (0.9,)]
    )

    print("\n=== Après ajout génération 1 ===")
    print(logbook.dataset)

    print("\n=== Stats par génération ===")
    print(logbook.stats())

    print("\n=== Hall of Fame ===")
    print(logbook.hall_of_fame(n_best=3))

    print("\n=== Évolution param_A ===")
    param_evolution = logbook.get_parameter_evolution('param_A')
    print(param_evolution)

    return logbook


if __name__ == "__main__":
    demo_usage()