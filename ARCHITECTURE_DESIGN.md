# Architecture Design - Refactoring GeneticAlgorithm

## 🎯 Objectifs

Refactorer `GeneticAlgorithm` pour :
- **Séparer les responsabilités** (GA logique vs Distribution vs Évaluation)
- **Améliorer la lisibilité** pour les utilisateurs métier
- **Faciliter la testabilité** avec isolation des composants
- **Permettre l'extensibilité** vers d'autres backends de calcul

## 🔍 Problème Actuel

La classe `GeneticAlgorithm` actuelle viole le **Single Responsibility Principle** :

```python
class GeneticAlgorithm:
    # ✓ Responsabilité: Logique GA
    def optimize(self):
        for generation in range(self.NGEN):
            # ... logique GA ...

    # ❌ Responsabilité: Distribution Dask
    def distribute_data(self):
        self._distributed_forcing = client.scatter(...)

    # ❌ Responsabilité: Stratégie d'évaluation
    def _evaluate(self):
        if self.client is None:
            # Mode séquentiel
        else:
            # Mode parallèle
```

**Conséquences** :
- Code complexe et difficile à maintenir
- Tests difficiles (couplage fort)
- Extensibilité limitée (nouveaux backends = refactoring massif)
- Lisibilité réduite pour les utilisateurs métier

## 🏗️ Nouvelle Architecture : Composition + Strategy Pattern

### 1. Gestionnaire de Distribution

```python
class DistributionManager:
    """
    Responsabilité: Gestion de la distribution des données avec Dask.
    Encapsule toute la complexité liée à Dask.
    """

    def __init__(self, client: Client):
        self.client = client
        self._distributed_data = {}
        self._original_data = {}

    def distribute_forcing(self, forcing_parameters) -> object:
        """
        Distribue les paramètres de forçage avec broadcast=True.

        Returns
        -------
        object
            Future Dask distribuée
        """
        if 'forcing' in self._distributed_data:
            warnings.warn("Forcing parameters déjà distribués", UserWarning)
            return self._distributed_data['forcing']

        logger.info("Distribution des paramètres de forçage...")
        scattered = self.client.scatter(forcing_parameters, broadcast=True)
        self._distributed_data['forcing'] = scattered
        self._original_data['forcing'] = forcing_parameters

        return scattered

    def distribute_observations(self, observations) -> list:
        """
        Distribue les observations avec broadcast=True.

        Parameters
        ----------
        observations : list[ObservationProtocol]
            Liste des observations à distribuer

        Returns
        -------
        list[object]
            Liste des Futures Dask distribuées
        """
        if 'observations' in self._distributed_data:
            warnings.warn("Observations déjà distribuées", UserWarning)
            return self._distributed_data['observations']

        logger.info("Distribution des observations...")
        scattered_obs = []
        for obs in observations:
            scattered = self.client.scatter(obs.observation, broadcast=True)
            scattered_obs.append(scattered)

        self._distributed_data['observations'] = scattered_obs
        self._original_data['observations'] = observations

        return scattered_obs

    def create_distributed_evaluator(self, cost_function) -> callable:
        """
        Crée une fonction d'évaluation utilisant les données distribuées.

        Parameters
        ----------
        cost_function : CostFunctionProtocol
            Fonction de coût à adapter pour la distribution

        Returns
        -------
        callable
            Fonction d'évaluation distribuée
        """
        forcing_future = self._distributed_data.get('forcing')
        obs_futures = self._distributed_data.get('observations', [])

        if not forcing_future or not obs_futures:
            raise RuntimeError("Données non distribuées. Appelez distribute_* d'abord.")

        return partial(
            distributed_evaluate,
            forcing_future,
            obs_futures,
            cost_function.functional_groups,
            cost_function.evaluation_function
        )

    def cleanup(self):
        """Nettoie les références aux Futures distribuées."""
        self._distributed_data.clear()
        self._original_data.clear()


def distributed_evaluate(forcing_future, observations_futures, functional_groups,
                        evaluation_function, individual_params):
    """
    Fonction d'évaluation distribuée.
    Les Futures sont résolues automatiquement par Dask comme arguments directs.

    Parameters
    ----------
    forcing_future : ForcingParameter
        Paramètres de forçage (Future résolue automatiquement)
    observations_futures : list[xr.DataArray]
        Observations (Futures résolues automatiquement)
    functional_groups : FunctionalGroupSet
        Configuration des groupes fonctionnels
    evaluation_function : callable
        Fonction d'évaluation originale
    individual_params : list[float]
        Paramètres de l'individu à évaluer

    Returns
    -------
    tuple
        Fitness de l'individu
    """
    # Créer un model_generator temporaire avec les données résolues
    from seapopym_optimization.model_generator import NoTransportModelGenerator
    temp_model_generator = NoTransportModelGenerator(forcing_parameters=forcing_future)

    # Créer des observations temporaires avec les données résolues
    temp_observations = []
    for obs_data in observations_futures:
        # Utiliser le type original mais avec les données résolues
        # Note: Nécessite accès aux métadonnées originales
        temp_obs = create_observation_from_data(obs_data)
        temp_observations.append(temp_obs)

    # Créer une cost_function temporaire
    from seapopym_optimization.cost_function.cost_function import CostFunction
    temp_cost_function = CostFunction(
        model_generator=temp_model_generator,
        observations=temp_observations,
        functional_groups=functional_groups,
        evaluation_function=evaluation_function
    )

    # Évaluer avec la cost function temporaire
    return temp_cost_function.generate()(individual_params)
```

### 2. Stratégies d'Évaluation

```python
from abc import ABC, abstractmethod

class EvaluationStrategy(ABC):
    """Interface pour les différentes stratégies d'évaluation."""

    @abstractmethod
    def evaluate(self, individuals: list, toolbox: base.Toolbox) -> list:
        """
        Évalue une liste d'individus.

        Parameters
        ----------
        individuals : list
            Liste des individus à évaluer
        toolbox : base.Toolbox
            Toolbox DEAP avec la fonction d'évaluation

        Returns
        -------
        list
            Liste des fitness calculées
        """
        pass


class SequentialEvaluation(EvaluationStrategy):
    """Stratégie d'évaluation séquentielle classique."""

    def evaluate(self, individuals: list, toolbox: base.Toolbox) -> list:
        """Évaluation séquentielle avec map() standard."""
        return list(map(toolbox.evaluate, individuals))


class DistributedEvaluation(EvaluationStrategy):
    """Stratégie d'évaluation distribuée avec Dask."""

    def __init__(self, distribution_manager: DistributionManager):
        self.distribution_manager = distribution_manager

    def evaluate(self, individuals: list, toolbox: base.Toolbox) -> list:
        """
        Évaluation distribuée utilisant client.map() avec données pré-distribuées.
        """
        # Créer la fonction d'évaluation distribuée
        distributed_evaluator = self.distribution_manager.create_distributed_evaluator(
            toolbox.cost_function
        )

        # Mapper sur les workers avec les Futures comme arguments directs
        individual_params = [list(ind) for ind in individuals]
        futures = self.distribution_manager.client.map(
            distributed_evaluator,
            individual_params
        )

        # Collecter les résultats
        return self.distribution_manager.client.gather(futures)


class ParallelEvaluation(EvaluationStrategy):
    """Stratégie d'évaluation parallèle classique (multiprocessing)."""

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    def evaluate(self, individuals: list, toolbox: base.Toolbox) -> list:
        """Évaluation parallèle avec multiprocessing."""
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(toolbox.evaluate, ind) for ind in individuals]
            return [future.result() for future in futures]
```

### 3. Classe GeneticAlgorithm Simplifiée

```python
@dataclass
class GeneticAlgorithm:
    """
    Algorithme génétique focalisé sur la logique métier.

    Délègue l'évaluation à une stratégie configurable, permettant
    différents modes d'exécution sans complexifier la logique principale.
    """

    meta_parameter: GeneticAlgorithmParameters
    cost_function: CostFunctionProtocol
    evaluation_strategy: EvaluationStrategy = field(default_factory=SequentialEvaluation)
    constraint: Sequence[ConstraintProtocol] | None = None
    save: FilePath | WriteBuffer[bytes] | None = None
    logbook: OptimizationLog | None = field(default=None, repr=False)
    toolbox: base.Toolbox | None = field(default=None, init=False, repr=False)

    def __post_init__(self: GeneticAlgorithm) -> None:
        """Initialisation focalisée sur la logique GA."""
        # Configuration du logbook
        if self.save is not None:
            self.save = Path(self.save)
            if self.save.exists():
                logger.warning(f"Logbook file {self.save} already exists. It will be overwritten.")

        # Génération du toolbox
        ordered_parameters = self.cost_function.functional_groups.unique_functional_groups_parameters_ordered()
        self.toolbox = self.meta_parameter.generate_toolbox(ordered_parameters.values(), self.cost_function)

        # Application des contraintes
        if self.constraint is not None:
            for constraint in self.constraint:
                self.toolbox.decorate("evaluate", constraint.generate(list(ordered_parameters.keys())))

        # Validation des poids
        if len(self.meta_parameter.cost_function_weight) != len(self.cost_function.observations):
            msg = (
                "The cost function weight must have the same length as the number of observations. "
                f"Got {len(self.meta_parameter.cost_function_weight)} and {len(self.cost_function.observations)}."
            )
            raise ValueError(msg)

    def _evaluate(self: GeneticAlgorithm, individuals: Sequence, generation: int) -> OptimizationLog:
        """
        Évalue les individus en déléguant à la stratégie d'évaluation.
        Logique simplifiée et focalisée.
        """
        def update_fitness(individuals: list) -> list:
            known = [ind.fitness.valid for ind in individuals]
            invalid_ind = [ind for ind in individuals if not ind.fitness.valid]

            if invalid_ind:
                # Délégation à la stratégie d'évaluation
                fitnesses = self.evaluation_strategy.evaluate(invalid_ind, self.toolbox)

                for ind, fit in zip(invalid_ind, fitnesses, strict=True):
                    ind.fitness.values = fit

            return known

        known = update_fitness(individuals)

        # Création du logbook (logique inchangée)
        individual_params = [list(ind) for ind in individuals]
        parameter_names = list(
            self.cost_function.functional_groups.unique_functional_groups_parameters_ordered().keys()
        )
        fitness_names = [obs.name for obs in self.cost_function.observations]
        fitness_values = [tuple(ind.fitness.values) for ind in individuals]

        logbook = OptimizationLog.from_individual(
            generation=generation,
            is_from_previous_generation=known,
            individual=individual_params,
            parameter_names=parameter_names,
            fitness_names=fitness_names,
        )

        logbook.update_fitness(generation, list(range(len(individuals))), fitness_values)
        return logbook

    def optimize(self: GeneticAlgorithm) -> OptimizationLog:
        """
        Logique d'optimisation pure, sans préoccupation de distribution.
        """
        generation_start, population = self._initialization()

        for gen in range(generation_start, self.meta_parameter.NGEN):
            logger.info(f"Generation {gen} / {self.meta_parameter.NGEN}.")

            # Sélection, croisement, mutation (logique GA standard)
            offspring = self.toolbox.select(population, self.meta_parameter.POP_SIZE)
            offspring = self.meta_parameter.variation(
                offspring, self.toolbox, self.meta_parameter.CXPB, self.meta_parameter.MUTPB
            )

            # Évaluation déléguée à la stratégie
            logbook = self._evaluate(offspring, gen)

            self.update_logbook(logbook)
            population[:] = offspring

        return self.logbook.copy()
```

### 4. Factory Pattern pour Simplifier l'Usage

```python
class GeneticAlgorithmFactory:
    """Factory pour créer des instances GeneticAlgorithm configurées."""

    @staticmethod
    def create_sequential(meta_parameter: GeneticAlgorithmParameters,
                         cost_function: CostFunctionProtocol,
                         **kwargs) -> GeneticAlgorithm:
        """Crée un GA en mode séquentiel."""
        return GeneticAlgorithm(
            meta_parameter=meta_parameter,
            cost_function=cost_function,
            evaluation_strategy=SequentialEvaluation(),
            **kwargs
        )

    @staticmethod
    def create_parallel(meta_parameter: GeneticAlgorithmParameters,
                       cost_function: CostFunctionProtocol,
                       n_jobs: int = -1,
                       **kwargs) -> GeneticAlgorithm:
        """Crée un GA en mode parallèle multiprocessing."""
        return GeneticAlgorithm(
            meta_parameter=meta_parameter,
            cost_function=cost_function,
            evaluation_strategy=ParallelEvaluation(n_jobs=n_jobs),
            **kwargs
        )

    @staticmethod
    def create_distributed(meta_parameter: GeneticAlgorithmParameters,
                          cost_function: CostFunctionProtocol,
                          client: Client,
                          auto_distribute: bool = True,
                          **kwargs) -> tuple[GeneticAlgorithm, DistributionManager]:
        """
        Crée un GA en mode distribué avec Dask.

        Parameters
        ----------
        auto_distribute : bool
            Si True, distribue automatiquement les données lourdes

        Returns
        -------
        tuple[GeneticAlgorithm, DistributionManager]
            L'instance GA et le gestionnaire de distribution
        """
        # Créer le gestionnaire de distribution
        dist_manager = DistributionManager(client)

        if auto_distribute:
            # Distribution automatique des données lourdes
            dist_manager.distribute_forcing(cost_function.model_generator.forcing_parameters)
            dist_manager.distribute_observations(cost_function.observations)

        # Créer la stratégie d'évaluation distribuée
        evaluation_strategy = DistributedEvaluation(dist_manager)

        # Créer l'instance GA
        ga = GeneticAlgorithm(
            meta_parameter=meta_parameter,
            cost_function=cost_function,
            evaluation_strategy=evaluation_strategy,
            **kwargs
        )

        return ga, dist_manager
```

## 📋 Usage Simplifié

### Mode Séquentiel (inchangé)
```python
ga = GeneticAlgorithmFactory.create_sequential(meta_params, cost_function)
results = ga.optimize()
```

### Mode Parallèle Classique
```python
ga = GeneticAlgorithmFactory.create_parallel(meta_params, cost_function, n_jobs=4)
results = ga.optimize()
```

### Mode Distribué Automatique
```python
client = Client()
ga, dist_manager = GeneticAlgorithmFactory.create_distributed(
    meta_params, cost_function, client, auto_distribute=True
)
results = ga.optimize()
dist_manager.cleanup()
```

### Mode Distribué Manuel (contrôle fin)
```python
client = Client()
dist_manager = DistributionManager(client)

# Distribution sélective
dist_manager.distribute_forcing(forcing_params)
dist_manager.distribute_observations(observations)

evaluation_strategy = DistributedEvaluation(dist_manager)
ga = GeneticAlgorithm(meta_params, cost_function, evaluation_strategy)
results = ga.optimize()
```

## 🧪 Structure de Tests

```python
# tests/test_distribution_manager.py
class TestDistributionManager:
    def test_distribute_forcing(self):
        # Test distribution des forcing parameters

    def test_distribute_observations(self):
        # Test distribution des observations

    def test_create_distributed_evaluator(self):
        # Test création de l'évaluateur distribué

    def test_cleanup(self):
        # Test nettoyage des ressources

# tests/test_evaluation_strategies.py
class TestEvaluationStrategies:
    def test_sequential_evaluation(self):
        # Test évaluation séquentielle

    def test_distributed_evaluation(self):
        # Test évaluation distribuée

    def test_parallel_evaluation(self):
        # Test évaluation parallèle multiprocessing

# tests/test_genetic_algorithm.py
class TestGeneticAlgorithm:
    def test_optimization_logic(self):
        # Test logique GA avec mock evaluation strategy

    def test_strategy_injection(self):
        # Test injection de différentes stratégies

# tests/test_factory.py
class TestGeneticAlgorithmFactory:
    def test_create_sequential(self):
        # Test factory séquentiel

    def test_create_distributed(self):
        # Test factory distribué

# tests/integration/
test_memory_usage.py      # Tests de consommation mémoire
test_performance.py       # Tests de performance
test_notebook_examples.py # Tests sur les exemples notebooks
```

## ✅ Avantages de cette Architecture

### 🎯 **Séparation des Responsabilités**
- `GeneticAlgorithm` → **Logique métier GA pure**
- `DistributionManager` → **Gestion Dask isolée**
- `EvaluationStrategy` → **Modes d'exécution modulaires**

### 📖 **Lisibilité Utilisateur**
- **Focus métier** : Les utilisateurs voient la logique GA principale
- **Distribution optionnelle** : Complexité cachée quand non utilisée
- **Configuration explicite** : Pas de magie noire

### 🧪 **Testabilité Maximale**
- **Tests isolés** par responsabilité
- **Mocks faciles** avec injection de stratégies
- **Tests d'intégration** ciblés

### 🚀 **Extensibilité Future**
- **Nouveaux backends** : Ray, MPI, Cloud APIs
- **Nouvelles stratégies** : GPU computing, edge computing
- **Nouvelles optimisations** : Adaptive scheduling, load balancing

### 🔄 **Rétrocompatibilité**
- **API existante** préservée via factory methods
- **Migration progressive** possible
- **Notebooks existants** fonctionnent sans modification

## 🎛️ Plan de Migration

### Phase 1 : Implémentation Base
1. Créer `DistributionManager` et `EvaluationStrategy`
2. Implémenter `SequentialEvaluation` et `DistributedEvaluation`
3. Tests unitaires complets

### Phase 2 : Refactoring GeneticAlgorithm
1. Simplifier la classe principale
2. Injection de stratégie d'évaluation
3. Tests d'intégration

### Phase 3 : Factory et Documentation
1. Créer `GeneticAlgorithmFactory`
2. Mettre à jour la documentation
3. Exemples d'usage

### Phase 4 : Migration et Optimisation
1. Migrer les notebooks existants
2. Tests de performance
3. Optimisations spécifiques

Cette architecture rend le code **beaucoup plus maintenable**, **extensible** et **compréhensible** pour tous les types d'utilisateurs !