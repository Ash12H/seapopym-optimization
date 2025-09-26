"""Analyse de l'impact du passage à xarray sur les viewers existants."""

import pandas as pd
import xarray as xr
import numpy as np


def analyze_simpleviewer_methods():
    """Analyser l'impact sur les méthodes de SimpleViewer."""

    print("🔍 ANALYSE D'IMPACT - SimpleViewer avec xarray\n")

    # Créer données d'exemple
    xr_data = create_sample_xarray()
    pd_data = xr_data.to_dataframe().reset_index()

    print("=== MÉTHODES SIMPLEVIEWER IMPACTÉES ===\n")

    methods_analysis = [
        {
            "method": "stats()",
            "current_pandas": "df.groupby('generation')['weighted_fitness'].agg(['mean', 'std', 'min', 'max'])",
            "xarray_equivalent": "ds.weighted_fitness.groupby('generation').agg({'mean': 'mean', 'std': 'std'})",
            "difficulty": "⭐ Facile",
            "benefits": "Plus lisible, métadonnées préservées"
        },
        {
            "method": "hall_of_fame()",
            "current_pandas": "df.nlargest(n, 'weighted_fitness')",
            "xarray_equivalent": "ds.weighted_fitness.sortby(ds.weighted_fitness).isel(sample=-n:)",
            "difficulty": "⭐⭐ Moyen",
            "benefits": "Accès direct aux coordonnées, slicing par nom"
        },
        {
            "method": "get_generation(gen)",
            "current_pandas": "df.loc[df['generation'] == gen]",
            "xarray_equivalent": "ds.sel(generation=gen)",
            "difficulty": "⭐ Facile",
            "benefits": "Syntaxe plus claire, plus performant"
        },
        {
            "method": "fitness_evolution()",
            "current_pandas": "Complex MultiIndex manipulation",
            "xarray_equivalent": "ds.weighted_fitness.plot(x='generation')",
            "difficulty": "⭐ Facile",
            "benefits": "Plot intégré, dimensions automatiques"
        },
        {
            "method": "parameter_evolution(param_name)",
            "current_pandas": "df.pivot_table(values=f'param_{param_name}', index='generation')",
            "xarray_equivalent": "ds.parameters.sel(parameter=param_name)",
            "difficulty": "⭐ Facile",
            "benefits": "Accès direct par nom, plus intuitif"
        },
        {
            "method": "parallel_coordinates()",
            "current_pandas": "Complex DataFrame manipulation + plotly",
            "xarray_equivalent": "ds.parameters.to_dataframe() + plotly",
            "difficulty": "⭐⭐ Moyen",
            "benefits": "Conversion facile vers DataFrame quand nécessaire"
        },
        {
            "method": "box_plot()",
            "current_pandas": "DataFrame groupby operations",
            "xarray_equivalent": "ds.parameters.groupby('generation').quantile([0.25, 0.5, 0.75])",
            "difficulty": "⭐⭐ Moyen",
            "benefits": "Opérations statistiques plus naturelles"
        },
        {
            "method": "shannon_entropy()",
            "current_pandas": "Manual calculation on DataFrame",
            "xarray_equivalent": "Manual calculation on DataArray",
            "difficulty": "⭐ Facile",
            "benefits": "Même logique, données mieux structurées"
        }
    ]

    for method in methods_analysis:
        print(f"📋 {method['method']}")
        print(f"   Pandas actuel: {method['current_pandas']}")
        print(f"   xarray équivalent: {method['xarray_equivalent']}")
        print(f"   Difficulté: {method['difficulty']}")
        print(f"   Bénéfices: {method['benefits']}")
        print()

    return methods_analysis


def demonstrate_viewer_methods():
    """Démontrer quelques méthodes viewer avec xarray."""

    print("=== DÉMONSTRATIONS PRATIQUES ===\n")

    # Créer données d'exemple
    ds = create_sample_xarray()

    print("🎯 Exemple 1: stats() par génération")
    print("Xarray:")
    stats_xr = xr.Dataset({
        'mean': ds.weighted_fitness.groupby('generation').mean(),
        'std': ds.weighted_fitness.groupby('generation').std(),
        'min': ds.weighted_fitness.groupby('generation').min(),
        'max': ds.weighted_fitness.groupby('generation').max(),
    })
    print(stats_xr.to_dataframe())
    print()

    print("🎯 Exemple 2: get_parameter_evolution()")
    print("Évolution du paramètre 'param_A':")
    param_evolution = ds.parameters.sel(parameter='param_A')
    print(param_evolution)
    print()

    print("🎯 Exemple 3: hall_of_fame() - Top 3")
    # Approche simplifiée pour hall of fame
    fitness_flat = ds.weighted_fitness.stack(sample=['generation', 'individual'])
    top_3_indices = fitness_flat.argsort()[-3:]
    top_3_coords = [(int(fitness_flat[i].generation), int(fitness_flat[i].individual)) for i in top_3_indices]
    print("Top 3 individus:")
    for i, (gen, ind) in enumerate(top_3_coords):
        fitness_val = ds.weighted_fitness[gen, ind].values
        print(f"  #{i+1}: Gen {gen}, Ind {ind}, Fitness: {fitness_val:.3f}")
    print()

    print("🎯 Exemple 4: Filtrage par génération")
    last_gen = ds.sel(generation=2)
    print(f"Dernière génération - Fitness moyenne: {last_gen.weighted_fitness.mean().values:.3f}")
    print()


def migration_strategy():
    """Proposer une stratégie de migration."""

    print("=== STRATÉGIE DE MIGRATION ===\n")

    strategies = [
        {
            "approach": "1. Migration Progressive",
            "description": [
                "• Garder interface pandas pour SimpleViewer",
                "• Ajouter méthodes .to_xarray() et .from_xarray()",
                "• Migrer méthode par méthode",
                "• Tests de non-régression"
            ],
            "pros": "Risque faible, compatibilité préservée",
            "cons": "Plus lent, code dupliqué temporairement"
        },
        {
            "approach": "2. Réécriture Complète",
            "description": [
                "• Nouveau XarrayViewer parallèle à SimpleViewer",
                "• Port de toutes les méthodes",
                "• Dépréciation progressive de SimpleViewer",
                "• Migration des notebooks/tests"
            ],
            "pros": "Architecture cohérente, bénéfices immédiats",
            "cons": "Plus de travail initial"
        },
        {
            "approach": "3. Couche d'Adaptation",
            "description": [
                "• Classe wrapper qui accepte les deux formats",
                "• Conversion automatique pandas <-> xarray",
                "• Interface unifiée",
                "• Optimisations internes avec xarray"
            ],
            "pros": "Flexibilité maximale, migration douce",
            "cons": "Complexité de maintenance"
        }
    ]

    for strategy in strategies:
        print(f"📋 {strategy['approach']}")
        for desc in strategy['description']:
            print(f"   {desc}")
        print(f"   ✅ Avantages: {strategy['pros']}")
        print(f"   ❌ Inconvénients: {strategy['cons']}")
        print()

    print("🎯 RECOMMANDATION:")
    print("   Approche 2 (Réécriture Complète) est recommandée car:")
    print("   • Les bénéfices de xarray sont substantiels")
    print("   • L'API sera plus cohérente et extensible")
    print("   • Évite la dette technique d'une couche d'adaptation")
    print("   • Le SimpleViewer actuel peut rester pour compatibilité")
    print()


def create_sample_xarray():
    """Créer des données xarray d'exemple."""
    np.random.seed(42)

    n_gen, n_ind, n_param = 3, 4, 3
    param_names = ['param_A', 'param_B', 'param_C']

    # Paramètres
    params = np.random.rand(n_gen, n_ind, n_param)

    # Fitness (plus bas = meilleur)
    fitness = np.random.rand(n_gen, n_ind) * 2
    for gen in range(n_gen):
        fitness[gen] *= (1 - 0.1 * gen)  # Amélioration au fil du temps

    ds = xr.Dataset(
        {
            'parameters': (['generation', 'individual', 'parameter'], params),
            'weighted_fitness': (['generation', 'individual'], fitness),
        },
        coords={
            'generation': range(n_gen),
            'individual': range(n_ind),
            'parameter': param_names,
        }
    )

    return ds


def performance_comparison():
    """Comparer les performances pandas vs xarray."""

    print("=== COMPARAISON PERFORMANCES ===\n")

    import time

    # Créer données de taille réaliste
    n_gen, n_ind, n_param = 50, 300, 10
    np.random.seed(42)

    # xarray dataset
    params = np.random.rand(n_gen, n_ind, n_param)
    fitness = np.random.rand(n_gen, n_ind)
    param_names = [f'param_{i}' for i in range(n_param)]

    ds = xr.Dataset({
        'parameters': (['generation', 'individual', 'parameter'], params),
        'weighted_fitness': (['generation', 'individual'], fitness),
    }, coords={
        'generation': range(n_gen),
        'individual': range(n_ind),
        'parameter': param_names,
    })

    # pandas equivalent
    df_data = []
    for gen in range(n_gen):
        for ind in range(n_ind):
            row = {'generation': gen, 'individual': ind, 'weighted_fitness': fitness[gen, ind]}
            for p, param_name in enumerate(param_names):
                row[f'param_{param_name}'] = params[gen, ind, p]
            df_data.append(row)
    df = pd.DataFrame(df_data)

    print(f"📊 Taille des données: {n_gen} générations × {n_ind} individus × {n_param} paramètres")
    print(f"   Total: {n_gen * n_ind:,} échantillons")
    print()

    # Test 1: Stats par génération
    print("⏱️  Test 1: Statistiques par génération")

    start = time.time()
    for _ in range(100):
        pandas_stats = df.groupby('generation')['weighted_fitness'].agg(['mean', 'std'])
    pandas_time = time.time() - start

    start = time.time()
    for _ in range(100):
        xarray_stats = xr.Dataset({
            'mean': ds.weighted_fitness.groupby('generation').mean(),
            'std': ds.weighted_fitness.groupby('generation').std(),
        })
    xarray_time = time.time() - start

    print(f"   Pandas: {pandas_time:.4f}s")
    print(f"   xarray: {xarray_time:.4f}s")
    print(f"   Ratio: {xarray_time/pandas_time:.2f}x")
    print()

    # Test 2: Accès paramètre spécifique
    print("⏱️  Test 2: Accès paramètre spécifique")

    start = time.time()
    for _ in range(1000):
        pandas_param = df['param_param_5']
    pandas_time = time.time() - start

    start = time.time()
    for _ in range(1000):
        xarray_param = ds.parameters.sel(parameter='param_5')
    xarray_time = time.time() - start

    print(f"   Pandas: {pandas_time:.4f}s")
    print(f"   xarray: {xarray_time:.4f}s")
    print(f"   Ratio: {xarray_time/pandas_time:.2f}x")
    print()

    print("💾 Taille mémoire:")
    print(f"   xarray Dataset: {ds.nbytes:,} bytes")
    print(f"   pandas DataFrame: {df.memory_usage(deep=True).sum():,} bytes")
    print(f"   Ratio: {df.memory_usage(deep=True).sum()/ds.nbytes:.2f}x")


if __name__ == "__main__":
    analyze_simpleviewer_methods()
    demonstrate_viewer_methods()
    migration_strategy()
    performance_comparison()