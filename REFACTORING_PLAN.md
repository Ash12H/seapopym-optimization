# Plan de Refactorisation - Distribution Parallèle

## Objectif

Résoudre les fuites mémoire lors de l'optimisation parallèle en distribuant efficacement les données lourdes (`ForcingParameter` et observations) avec `client.scatter(..., broadcast=True)`.

## Architecture Cible

### Principe Simple
- **Client fourni** → Mode parallèle avec possibilité de distribution
- **Client = None** → Mode séquentiel (aucun changement)

### Logique de Distribution
1. **Validation dans `__post_init__`** : Warning si client fourni mais données non distribuées
2. **Méthode `distribute_data()`** : Distribution automatique pour l'utilisateur
3. **Documentation** : Bonnes pratiques avec `broadcast=True`

## Modifications Requises

### 1. Utilitaires de Détection (`genetic_algorithm.py`)

```python
def is_distributed(obj) -> bool:
    """Vérifie si un objet est une Future Dask distribuée."""
    return isinstance(obj, Future)

def has_distributed_data(model_generator, observations) -> bool:
    """Vérifie si les données lourdes sont déjà distribuées."""
    forcing_distributed = is_distributed(model_generator.forcing_parameters)
    obs_distributed = any(is_distributed(obs.observation) for obs in observations)
    return forcing_distributed or obs_distributed
```

### 2. Validation dans `__post_init__`

**Fichier**: `seapopym_optimization/algorithm/genetic_algorithm/genetic_algorithm.py`
**Ligne**: Après la logique existante dans `__post_init__`

```python
def __post_init__(self: GeneticAlgorithm) -> None:
    # ... logique existante ...

    # NOUVELLE VALIDATION DE DISTRIBUTION
    if self.client is not None:
        if not has_distributed_data(self.cost_function.model_generator, self.cost_function.observations):
            warnings.warn(
                "Client Dask fourni mais données non distribuées. "
                "Cela peut causer des fuites mémoire lors de l'optimisation parallèle. "
                "Utilisez ga.distribute_data() ou distribuez manuellement avec client.scatter(..., broadcast=True)",
                UserWarning,
                stacklevel=2
            )
```

### 3. Méthode `distribute_data()`

**Fichier**: `seapopym_optimization/algorithm/genetic_algorithm/genetic_algorithm.py`
**Emplacement**: Nouvelle méthode dans la classe `GeneticAlgorithm`

```python
def distribute_data(self: GeneticAlgorithm) -> dict[str, bool]:
    """
    Distribue les données lourdes sur les workers Dask avec broadcast=True.

    Returns
    -------
    dict[str, bool]
        Statut de distribution pour chaque type de donnée

    Raises
    ------
    RuntimeError
        Si aucun client Dask n'est disponible
    """
    if self.client is None:
        raise RuntimeError("Aucun client Dask disponible. Distribution impossible.")

    results = {'forcing_parameters': False, 'observations': False}

    # Distribuer forcing_parameters
    if is_distributed(self.cost_function.model_generator.forcing_parameters):
        warnings.warn("forcing_parameters déjà distribué. Ignoré.", UserWarning, stacklevel=2)
        results['forcing_parameters'] = True
    else:
        logger.info("Distribution du forcing_parameters...")
        scattered_forcing = self.client.scatter(
            self.cost_function.model_generator.forcing_parameters,
            broadcast=True
        )
        self.cost_function.model_generator.forcing_parameters = scattered_forcing
        results['forcing_parameters'] = True

    # Distribuer observations
    for i, obs in enumerate(self.cost_function.observations):
        if is_distributed(obs.observation):
            warnings.warn(f"observation[{i}] '{obs.name}' déjà distribuée. Ignorée.", UserWarning, stacklevel=2)
        else:
            logger.info(f"Distribution de l'observation '{obs.name}'...")
            scattered_obs = self.client.scatter(obs.observation, broadcast=True)
            obs.observation = scattered_obs

    results['observations'] = True
    logger.info("Distribution terminée avec succès.")
    return results
```

### 4. Imports Requis

**Fichier**: `seapopym_optimization/algorithm/genetic_algorithm/genetic_algorithm.py`
**Ajouter**:

```python
import warnings
from dask.distributed import Future
```

### 5. Documentation de la Classe

**Fichier**: `seapopym_optimization/algorithm/genetic_algorithm/genetic_algorithm.py`
**Mettre à jour**: Docstring de la classe `GeneticAlgorithm`

```python
class GeneticAlgorithm:
    """
    Algorithme génétique pour l'optimisation de modèles SeapoPym.

    Optimisation parallèle avec Dask
    --------------------------------
    Pour éviter les fuites mémoire lors de l'optimisation parallèle, les données lourdes
    (forcing_parameters, observations) doivent être distribuées sur les workers avec
    broadcast=True.

    Méthode 1 - Distribution automatique:
        >>> ga = GeneticAlgorithm(client=client, cost_function=cost_function)
        >>> ga.distribute_data()  # Distribue automatiquement avec broadcast=True
        >>> results = ga.optimize()

    Méthode 2 - Distribution manuelle:
        >>> scattered_forcing = client.scatter(forcing_parameters, broadcast=True)
        >>> scattered_obs = client.scatter(observation.observation, broadcast=True)
        >>> # Remplacer dans les objets avant création du GeneticAlgorithm
        >>> ga = GeneticAlgorithm(client=client, cost_function=cost_function)
        >>> results = ga.optimize()

    Note importante: broadcast=True est essentiel pour éviter la sérialisation
    répétée des données lourdes vers chaque worker.
    """
```

## Tests à Créer

### 1. Test de Détection
```python
def test_is_distributed():
    # Test avec Future vs objet local

def test_has_distributed_data():
    # Test avec différentes combinaisons de données distribuées/locales
```

### 2. Test de Validation
```python
def test_post_init_warning():
    # Vérifier que le warning est émis quand client fourni mais données locales

def test_post_init_no_warning():
    # Vérifier qu'aucun warning si pas de client ou données distribuées
```

### 3. Test de Distribution
```python
def test_distribute_data_success():
    # Test distribution normale

def test_distribute_data_already_distributed():
    # Test avec données déjà distribuées (warnings)

def test_distribute_data_no_client():
    # Test erreur si pas de client
```

## Exemple d'Usage Final

```python
from dask.distributed import Client
from seapopym_optimization.algorithm.genetic_algorithm import GeneticAlgorithm

# Créer l'algorithme avec client
client = Client()
ga = GeneticAlgorithm(client=client, cost_function=cost_function)
# Warning: "Client Dask fourni mais données non distribuées..."

# Distribuer automatiquement les données
status = ga.distribute_data()
print(status)  # {'forcing_parameters': True, 'observations': True}

# Optimiser sans fuite mémoire
results = ga.optimize()
```

## Ordre d'Implémentation

1. **Utilitaires de détection** (`is_distributed`, `has_distributed_data`)
2. **Validation dans `__post_init__`** (warning simple)
3. **Méthode `distribute_data()`** (fonctionnalité principale)
4. **Documentation mise à jour** (docstring classe)
5. **Tests unitaires** (validation du comportement)
6. **Test d'intégration** (exemple notebook)

## Points d'Attention

- **Import Future** : Vérifier compatibilité avec version Dask utilisée
- **Mutation d'objets** : `distribute_data()` modifie les objets en place
- **Gestion d'erreur** : Client Dask peut planter, gérer les exceptions
- **Performance** : `client.scatter(..., broadcast=True)` est bloquant
- **Logging** : Messages informatifs pour l'utilisateur

## Validation

### Tests Manuels
1. Créer GA avec client mais données locales → Warning attendu
2. Appeler `distribute_data()` → Distribution réussie
3. Optimiser → Pas de fuite mémoire
4. Créer GA sans client → Pas de warning

### Tests de Performance
1. Comparer temps d'exécution avant/après distribution
2. Vérifier usage mémoire sur workers
3. Confirmer que `broadcast=True` évite re-sérialisation