# Space Muck Tests

This directory contains all test files for the Space Muck project. Tests are organized into the following categories:

- **Main directory**: Unit tests for core components
- **benchmarks/**: Performance benchmarks for various components
- **tools/**: Testing utilities and specialized test cases
- **verifications/**: Verification scripts for game components

## Test Organization

### Main Directory
Contains unit tests for core components of the Space Muck project. These tests verify the basic functionality of individual components and ensure they work correctly in isolation.

### Benchmarks
Contains performance benchmark scripts for measuring the efficiency of various components, particularly those related to procedural generation, GPU acceleration, and parallel processing.

### Tools
Contains utility scripts for testing purposes, including:
- Linting checks
- Integration tests
- Performance tests
- Regression tests

### Verifications
Contains verification scripts that validate component behavior in realistic scenarios. Unlike unit tests, these scripts provide more detailed output and test components in a more integrated fashion.

## Running Tests

### Run all tests
```bash
# From project root
python -m pytest src/tests

# With coverage report
python -m pytest src/tests --cov=src
```

### Run specific test file
```bash
# Example: Run base generator tests
python -m pytest src/tests/test_base_generator.py
```

### Run tests by category
```bash
# Run all GPU-related tests
python -m pytest src/tests/test_gpu_*.py

# Run all entity tests
python -m pytest src/tests/test_*entity*.py
```

## Import Testing

The main import test file (`test_imports.py`) serves as a comprehensive import verification tool. It combines functionality from multiple import test files and verifies that imports are working correctly across the codebase.

```bash
# Run the comprehensive import test
python -m pytest src/tests/test_imports.py

# Run the import test with detailed output
python src/tests/test_imports.py
```

## Quick Reference

| Component | Test Command |
|-----------|--------------|
| Asteroid Generator | `python -m pytest src/tests/test_asteroid_generator.py` |
| Base Entity | `python -m pytest src/tests/test_base_entity.py` |
| Base Generator | `python -m pytest src/tests/test_base_generator.py` |
| Cellular Automaton | `python -m pytest src/tests/test_cellular_automaton_utils.py` |
| Fleet | `python -m pytest src/tests/test_fleet.py` |
| Fleet Manager | `python -m pytest src/tests/test_fleet_manager.py` |
| GPU Acceleration | `python -m pytest src/tests/test_gpu_acceleration.py` |
| GPU Clustering | `python -m pytest src/tests/test_gpu_clustering.py` |
| GPU Utils | `python -m pytest src/tests/test_gpu_utils.py` |
| Imports | `python -m pytest src/tests/test_imports.py` |
| Miner Entity | `python -m pytest src/tests/test_miner_entity.py` |
| Pattern Generator | `python -m pytest src/tests/test_pattern_generator.py` |
| Player | `python -m pytest src/tests/test_player.py` |
| Procedural Generator | `python -m pytest src/tests/test_procedural_generator.py` |
| Symbiote Evolution | `python -m pytest src/tests/test_symbiote_evolution_generator.py` |
| Trading System | `python -m pytest src/tests/test_trading_system.py` |
| Value Generator | `python -m pytest src/tests/test_value_generator.py` |
| Value Generator GPU | `python -m pytest src/tests/test_value_generator_gpu.py` |
| Visualization | `python -m pytest src/tests/test_visualization.py` |

## Utility Tools

- **directory_tree.py**: Generate a directory tree structure
  ```bash
  # Show structure of tests directory
  python directory_tree.py --use-ignore
  
  # Show structure of a custom directory
  python directory_tree.py -c /path/to/directory --use-ignore
  ```
