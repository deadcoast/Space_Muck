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

### Utilities

- `/src/utils/gpu_utils.py`: GPU acceleration utilities for various computational tasks
- `/src/utils/noise_generator.py`: Noise generation utilities for procedural generation
- `/src/utils/value_generator.py`: Value distribution generation utilities
- `/src/utils/value_generator_gpu.py`: GPU-accelerated value distribution generation
- `/src/utils/cellular_automaton_utils.py`: Cellular automaton utilities for terrain generation

## Tests

### Unit Tests

- `/src/tests/test_base_entity.py`: Tests for the base entity class
- `/src/tests/test_gpu_utils.py`: Tests for GPU acceleration utilities

### Benchmarks

- `/src/tests/bunchmarks/benchmark_comprehensive_gpu.py`: Comprehensive benchmarking for GPU-accelerated operations
- `/src/tests/test_base_generator.py`: Tests for the base generator class
- `/src/tests/test_fleet_manager.py`: Tests for the fleet management system

## Benchmarks

- `/src/tests/bunchmarks/benchmark_base_generator.py`: Benchmarks for the base generator performance
- `/src/tests/bunchmarks/benchmark_procedural_generation.py`: Benchmarks for procedural generation algorithms
- `/src/tests/bunchmarks/benchmark_comprehensive_gpu.py`: Comprehensive GPU acceleration benchmarks

## Documentation

- `/docs/CodeBase_Docs/CodeBase_Architecture.md`: Architecture documentation
- `/docs/CodeBase_Docs/CodeBase_Mapping.md`: This file - codebase navigation
- `/docs/CodeBase_Docs/CodeBase_Error_Fixes.md`: Common errors and their solutions
- `/docs/CodeBase_Docs/CodeBase_Scratchpad.md`: Development notes and task tracking
