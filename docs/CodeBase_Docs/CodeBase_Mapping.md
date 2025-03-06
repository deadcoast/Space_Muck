# CodeBase Mapping

This document provides a navigation guide to the Space Muck codebase, helping to locate files and understand their purpose.

## Table of Contents

1. [Source Code](#source-code)
2. [Tests](#tests)
3. [Benchmarks](#benchmarks)
4. [Documentation](#documentation)

## Source Code

### Core Entities

- `/src/entities/base_entity.py`: Base class for all entities in the Space Muck system
- `/src/entities/base_generator.py`: Base generator class for procedural content generation
- `/src/entities/miner_entity.py`: Symbiotic mining races that evolve in the asteroid field

### Generators

- `/src/generators/asteroid_field.py`: Manages the asteroid field grid and symbiote entities with procedural generation
- `/src/generators/asteroid_generator.py`: Specialized generator for asteroid fields with various patterns
- `/src/generators/procedural_generator.py`: Base procedural generation utilities
- `/src/generators/symbiote_evolution_generator.py`: Handles symbiote race evolution

### Systems

- `/src/systems/fleet_manager.py`: Manages ship fleets, formations, combat, and resource distribution
- `/src/systems/trading_system.py`: Handles trading, economy, and market interactions
- `/src/systems/combat_system.py`: Manages combat mechanics and interactions
- `/src/systems/encounter_generator.py`: Generates random encounters and events

### Core Game

- `/src/main.py`: Main game entry point and game loop implementation

### Utilities

- `/src/utils/gpu_utils.py`: GPU acceleration utilities for various computational tasks
- `/src/utils/noise_generator.py`: Noise generation utilities for procedural generation
- `/src/utils/value_generator.py`: Value distribution generation utilities
- `/src/utils/value_generator_gpu.py`: GPU-accelerated value distribution generation
- `/src/utils/cellular_automaton_utils.py`: Cellular automaton utilities for terrain generation

## Tests

### Unit Tests

- `/src/tests/test_base_entity.py`: Tests for the base entity class
  - Tests entity initialization and properties
  - Tests entity serialization and deserialization
  - Tests entity validation and error handling
  - Tests entity movement and positioning
  - Tests entity state management

- `/src/tests/test_gpu_utils.py`: Tests for GPU acceleration utilities
  - Tests GPU availability detection
  - Tests memory transfer between CPU and GPU
  - Tests GPU-accelerated operations
  - Tests fallback mechanisms when GPU is unavailable

- `/src/tests/test_base_generator.py`: Tests for the base generator class
  - Tests generator initialization and properties
  - Tests noise generation functionality
  - Tests cellular automaton operations
  - Tests caching mechanisms
  - Tests value distribution generation

- `/src/tests/test_fleet_manager.py`: Tests for the fleet management system
  - Tests fleet creation and management
  - Tests combat mechanics and damage distribution
  - Tests formation-based tactics
  - Tests resource allocation and distribution
  - Tests fleet serialization and deserialization

- `/src/tests/test_symbiote_evolution_generator.py`: Comprehensive tests for the symbiote evolution generator
  - Tests initialization parameters and default values
  - Tests evolution algorithm integration
  - Tests mineral distribution and consumption
  - Tests mutation mapping and base genome generation
  - Tests serialization (to_dict/from_dict)
  - Tests colony generation
  - Tests evolution simulation over time with detailed history verification
  - Tests mineral consumption impact on population and mutations
  - Tests visualization capabilities (with matplotlib dependency checks)
  - Includes robust error handling and dependency verification
  - Features a comprehensive test runner for standalone execution

- `/src/tests/test_asteroid_generator.py`: Comprehensive tests for the asteroid generator
  - Tests initialization parameters and default values
  - Tests asteroid field generation with various patterns (belt, cluster, mixed)
  - Tests pattern generation with different parameters and noise scales
  - Tests mineral distribution generation with multiple distribution types
  - Tests rare resource generation with configurable probabilities
  - Tests energy field generation with different energy types (standard, radiation, plasma)
  - Tests asteroid field serialization and object creation
  - Tests performance benchmarking and caching verification
  - Tests visualization capabilities (with matplotlib dependency checks)
  - Includes robust error handling and dependency verification
  - Features a comprehensive test runner for standalone execution

- `/src/tests/test_procedural_generator.py`: Tests for procedural generation utilities
  - Tests procedural algorithm selection
  - Tests parameter validation
  - Tests generation consistency with same seed
  - Tests different generation patterns
  - Tests integration with other generator components

### Test Utilities

- `/src/tests/test_imports.py`: Comprehensive import verification tool
  - Tests core module imports
  - Tests optional dependency detection
  - Tests basic functionality of key components
  - Provides troubleshooting information for import issues

- `/src/tests/directory_tree.py`: Utilities for generating directory structure representations
  - Supports custom depth and exclusion patterns
  - Provides formatted output for documentation

### Test Tools

- `/src/tests/tools/check_linting.py`: Linting verification tools
  - Runs flake8, pylint, ruff, and sourcery
  - Reports code quality issues

- `/src/tests/tools/integration_tests.py`: High-level integration tests
  - Tests interactions between multiple components
  - Tests end-to-end workflows

- `/src/tests/tools/test_combat_system.py`: Combat system tests
  - Tests combat mechanics
  - Tests damage calculation
  - Tests combat resolution

- `/src/tests/tools/test_dependency_config.py`: Tests for dependency configuration
  - Tests optional dependency handling
  - Tests fallback mechanisms

- `/src/tests/README.md`: Documentation for the test suite

## Benchmarks

- `/src/tests/bunchmarks/benchmark_base_generator.py`: Benchmarks for the base generator performance
- `/src/tests/bunchmarks/benchmark_procedural_generation.py`: Benchmarks for procedural generation algorithms
- `/src/tests/bunchmarks/benchmark_comprehensive_gpu.py`: Comprehensive GPU acceleration benchmarks

## Documentation

- `/docs/CodeBase_Docs/CodeBase_Architecture.md`: Architecture documentation
- `/docs/CodeBase_Docs/CodeBase_Mapping.md`: This file - codebase navigation
- `/docs/CodeBase_Docs/CodeBase_Error_Fixes.md`: Common errors and their solutions
- `/docs/CodeBase_Docs/CodeBase_Scratchpad.md`: Development notes and task tracking
