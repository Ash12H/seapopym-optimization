# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seapopym-optimisation is a Python package for optimizing SeapoPym spatial ecological models using genetic algorithms. The project focuses on model calibration, parameter set validation, and comparison between simulations and observations using various cost functions.

## Development Commands

### Environment Setup
- Install dependencies: `poetry install`
- Activate shell: `poetry shell`

### Code Quality
- Lint code: `make lint` or `poetry run ruff check ./seapopym_optimization`
- Format code: `make format` or `poetry run ruff format ./seapopym_optimization`

### Testing
- Run tests: `poetry run pytest test/`
- Run specific test: `poetry run pytest test/test_[filename].py`

### Documentation
- Build docs: `make doc` (requires pandoc for notebook conversion)
- Export requirements: `poetry export -f requirements.txt --with doc --output docs/requirements.txt`

### Publishing
- Test publish: `make publish_test`
- Production publish: `make publish`

## Architecture Overview

The codebase follows a modular architecture with clear separation of concerns:

### Core Modules

1. **cost_function/**: Cost function implementations for optimization
   - `base_cost_function.py`: Abstract base class defining cost function interface
   - `simple_cost_function.py`: Basic RMSE-based cost functions
   - `seasonality_cost_function.py`: Advanced cost functions with seasonal decomposition
   - `error_weighted_cost_function.py`: Cost functions with error weighting

2. **genetic_algorithm/**: DEAP-based genetic algorithm implementations
   - `base_genetic_algorithm.py`: Abstract GA base class with custom individual creator for Dask compatibility
   - `simple_genetic_algorithm.py`: Standard GA implementation
   - `simple_logbook.py`: Optimization logging utilities

3. **functional_group/**: Parameter management and initialization
   - `base_functional_group.py`: Core parameter set management
   - `parameter_initialization.py`: Parameter initialization strategies
   - `sobol_initialization.py`: Sobol sequence-based parameter initialization

4. **model_generator/**: SeapoPym model creation and configuration
   - `base_model_generator.py`: Abstract model generator interface
   - `no_transport_model_generator.py`: Model generator for no-transport scenarios
   - `acidity_model_generator.py`: Specialized generator for acidity models

5. **constraint/**: Parameter constraint implementations
   - `base_constraint.py`: Abstract constraint interface
   - `energy_transfert_constraint.py`: Energy transfer constraints

6. **viewer/**: Visualization and analysis tools
   - `simple_viewer.py`: Primary visualization utilities for optimization results
   - `taylor_diagram.py`: Taylor diagram plotting for model evaluation

### Key Dependencies

- **seapopym**: Core SeapoPym model library (^0.0.2.5.1)
- **deap**: Genetic algorithm framework (^1.4.3)
- **dask/distributed**: Parallel computing (^2025.9.0)
- **scipy/scikit-learn**: Scientific computing and machine learning
- **plotly**: Interactive visualizations

## Development Workflow

1. Create feature branches from `dev` branch
2. Merge completed features back to `dev`
3. Use Poetry for dependency management
4. Run linting and formatting before commits
5. Ensure tests pass before merging

## Code Conventions

- Line length: 120 characters (configured in .ruff.toml)
- Use Ruff for linting and formatting (replaces Black)
- Follow dataclass patterns for configuration objects
- Use abstract base classes for extensible components
- Type hints are mandatory (checked by Ruff)
- Docstrings required for public APIs

## SeapoPym 0.0.2.5.1 Migration Notes

### Breaking Changes Applied
- **Protocols over ABC**: SeapoPym now uses `typing.Protocol` instead of Abstract Base Classes
- **Configuration Updates**:
  - `AbstractEnvironmentParameter` removed (functionality integrated into configurations)
  - Import from `seapopym.standard.protocols` for type hints
  - `NoTransportConfiguration` and `AcidityConfiguration` no longer take `environment` parameter
- **Type System**: Model generators now use `ModelProtocol`, `ForcingParameterProtocol`, `KernelParameterProtocol`

### Compatibility
- **Maintained**: All existing optimization workflows remain functional
- **Enhanced**: Better type safety through Protocol-based interfaces
- **Improved**: Duck typing allows more flexible implementations