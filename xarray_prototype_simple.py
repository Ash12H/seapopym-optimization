"""Version simplifiée du prototype xarray pour démonstration."""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Sequence


def create_xarray_logbook_demo():
    """Démonstration concrète des avantages de xarray vs pandas pour les données d'optimisation."""

    print("🧬 Prototype xarray.Dataset pour données d'optimisation\n")

    # Paramètres d'exemple
    n_generations = 3
    n_individuals = 4
    parameter_names = ['energy_transfert', 'gamma_tr', 'tr_0']
    objective_names = ['RMSE_biomass']

    # === CRÉATION DES DONNÉES ===
    # Génération de données d'exemple réalistes
    np.random.seed(42)

    # Paramètres : (generation, individual, parameter)
    param_data = np.random.rand(n_generations, n_individuals, len(parameter_names))
    param_data[:, :, 0] *= 0.3  # energy_transfert: 0-0.3
    param_data[:, :, 1] = param_data[:, :, 1] * 0.3 - 0.3  # gamma_tr: -0.3 à 0
    param_data[:, :, 2] *= 30  # tr_0: 0-30

    # Fitness : (generation, individual, objective)
    fitness_data = np.random.rand(n_generations, n_individuals, 1) * 2
    # Simuler amélioration au fil des générations
    for gen in range(n_generations):
        fitness_data[gen] *= (1 - 0.1 * gen)  # Fitness diminue (meilleure)

    # Weighted fitness : (generation, individual)
    weighted_fitness = fitness_data.squeeze(-1)  # Remove objective dim

    print("=== 1. CRÉATION du xarray.Dataset ===")

    # Créer le dataset xarray
    optimization_results = xr.Dataset(
        {
            'parameters': (['generation', 'individual', 'parameter'], param_data),
            'fitness': (['generation', 'individual', 'objective'], fitness_data),
            'weighted_fitness': (['generation', 'individual'], weighted_fitness),
            'is_from_previous': (['generation', 'individual'],
                               np.array([[False]*n_individuals, [True]*n_individuals, [True]*n_individuals]))
        },
        coords={
            'generation': range(n_generations),
            'individual': range(n_individuals),
            'parameter': parameter_names,
            'objective': objective_names,
        },
        attrs={
            'algorithm': 'SimpleGeneticAlgorithm',
            'parameter_bounds': {
                'energy_transfert': (0.001, 0.3),
                'gamma_tr': (-0.3, -0.001),
                'tr_0': (0, 30)
            },
            'created': '2025-01-15',
            'description': 'Optimization of SeapoPym NoTransport model'
        }
    )

    print(optimization_results)
    print()

    print("=== 2. AVANTAGES du SLICING/INDEXING ===")

    # ✅ Accès intuitif par nom
    print("🔍 Paramètre 'energy_transfert' pour toutes générations:")
    energy_param = optimization_results['parameters'].sel(parameter='energy_transfert')
    print(energy_param)
    print()

    # ✅ Slice par génération
    print("🔍 Génération 2 (dernière):")
    last_gen = optimization_results.sel(generation=2)
    print(last_gen['parameters'])
    print()

    # ✅ Meilleur individu par génération
    print("🔍 Meilleur individu de chaque génération:")
    best_individuals = optimization_results.weighted_fitness.argmin(dim='individual')
    best_params = optimization_results['parameters'].isel(individual=best_individuals)
    print(best_params)
    print()

    print("=== 3. OPÉRATIONS STATISTIQUES ===")

    # ✅ Stats par génération (très simple)
    print("📊 Statistiques par génération:")
    stats = xr.Dataset({
        'mean': optimization_results.weighted_fitness.mean(dim='individual'),
        'std': optimization_results.weighted_fitness.std(dim='individual'),
        'min': optimization_results.weighted_fitness.min(dim='individual'),
        'max': optimization_results.weighted_fitness.max(dim='individual'),
    })
    print(stats.to_dataframe())
    print()

    # ✅ Évolution des paramètres
    print("📈 Évolution moyenne des paramètres:")
    param_evolution = optimization_results['parameters'].mean(dim='individual')
    print(param_evolution.to_dataframe().round(4))
    print()

    print("=== 4. COMPARAISON avec PANDAS ===")

    # Créer équivalent pandas (MultiIndex complexe)
    pandas_data = []
    for gen in range(n_generations):
        for ind in range(n_individuals):
            row = {
                'generation': gen,
                'individual': ind,
                'is_from_previous': bool(optimization_results.is_from_previous[gen, ind]),
            }
            # Paramètres
            for i, param in enumerate(parameter_names):
                row[f'param_{param}'] = param_data[gen, ind, i]
            # Fitness
            for i, obj in enumerate(objective_names):
                row[f'fitness_{obj}'] = fitness_data[gen, ind, i]
            row['weighted_fitness'] = weighted_fitness[gen, ind]
            pandas_data.append(row)

    df_pandas = pd.DataFrame(pandas_data).set_index(['generation', 'individual'])

    print("🐼 Équivalent pandas DataFrame:")
    print(df_pandas.head(8))
    print()

    print("=== 5. COMPARAISON des OPÉRATIONS ===")

    print("🔄 Récupérer 'energy_transfert' génération 1:")

    # Pandas (verbeux)
    pandas_result = df_pandas.loc[(1, slice(None)), 'param_energy_transfert']
    print(f"Pandas: df.loc[(1, slice(None)), 'param_energy_transfert']")
    print(f"Résultat: {pandas_result.values}")

    # xarray (intuitif)
    xarray_result = optimization_results['parameters'].sel(generation=1, parameter='energy_transfert')
    print(f"xarray: results['parameters'].sel(generation=1, parameter='energy_transfert')")
    print(f"Résultat: {xarray_result.values}")
    print()

    print("📊 Calculer moyenne par génération:")

    # Pandas
    pandas_mean = df_pandas.groupby('generation')['weighted_fitness'].mean()
    print(f"Pandas: df.groupby('generation')['weighted_fitness'].mean()")
    print(f"Résultat: {pandas_mean.values}")

    # xarray
    xarray_mean = optimization_results.weighted_fitness.mean(dim='individual')
    print(f"xarray: results.weighted_fitness.mean(dim='individual')")
    print(f"Résultat: {xarray_mean.values}")
    print()

    print("=== 6. MÉTADONNÉES et EXTENSIBILITÉ ===")

    print("🏷️ Métadonnées intégrées:")
    for key, value in optimization_results.attrs.items():
        print(f"  {key}: {value}")
    print()

    print("🔧 Ajout facile de nouvelles dimensions:")
    # Simuler ajout d'une dimension 'constraint'
    constraint_data = np.random.choice([True, False], size=(n_generations, n_individuals))
    optimization_results['constraint_satisfied'] = (['generation', 'individual'], constraint_data)
    print("✅ Ajouté 'constraint_satisfied' sans refactoring!")
    print()

    print("=== 7. SÉRIALISATION/PERSISTENCE ===")

    # Créer version compatible NetCDF (sans dict complexes)
    simple_attrs = {k: str(v) for k, v in optimization_results.attrs.items()}
    optimization_results.attrs = simple_attrs

    # Sauvegarder (format NetCDF très efficace)
    optimization_results.to_netcdf('/tmp/optimization_results.nc')
    print("💾 Sauvé en NetCDF (format binaire compact)")

    # Recharger
    loaded = xr.open_dataset('/tmp/optimization_results.nc')
    print("📂 Rechargé avec métadonnées intactes")
    print(f"   Attributs préservés: {list(loaded.attrs.keys())}")
    loaded.close()

    print("\n🎯 CONCLUSION:")
    print("✅ xarray offre une API plus naturelle pour données multidimensionnelles")
    print("✅ Métadonnées intégrées (bounds, descriptions)")
    print("✅ Slicing/indexing par nom plus lisible")
    print("✅ Broadcasting automatique pour calculs")
    print("✅ Extensibilité sans refactoring")
    print("❌ Overhead mémoire ~20%")
    print("❌ Courbe d'apprentissage plus élevée")

    return optimization_results


def compare_operations():
    """Comparaison directe d'opérations courantes."""

    print("\n🔬 COMPARAISON OPÉRATIONS COURANTES\n")

    # Données simulées
    results = create_sample_data()
    df = to_pandas_multiindex(results)

    operations = [
        ("Hall of Fame (top 5)",
         "df.nlargest(5, 'weighted_fitness')",
         "results.weighted_fitness.where(...).topk(5)"),

        ("Stats par génération",
         "df.groupby('generation').agg(['mean', 'std'])",
         "results.groupby('generation').agg({'mean': 'mean', 'std': 'std'})"),

        ("Évolution paramètre",
         "df.pivot_table(values='param_X', index='generation')",
         "results['parameters'].sel(parameter='X')"),

        ("Filtrage par contrainte",
         "df[df['constraint'] == True]['weighted_fitness']",
         "results.weighted_fitness.where(results.constraint)"),
    ]

    for desc, pandas_code, xarray_code in operations:
        print(f"📋 {desc}")
        print(f"  Pandas:  {pandas_code}")
        print(f"  xarray:  {xarray_code}")
        print()


def create_sample_data():
    """Créer données d'exemple simples."""
    return xr.Dataset({
        'parameters': (['gen', 'ind', 'param'], np.random.rand(3, 5, 4)),
        'weighted_fitness': (['gen', 'ind'], np.random.rand(3, 5)),
    })


def to_pandas_multiindex(ds):
    """Convertir xarray en pandas MultiIndex."""
    return ds.to_dataframe()


if __name__ == "__main__":
    results = create_xarray_logbook_demo()
    compare_operations()