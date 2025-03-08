# CodeBase Architecture

This document outlines the architecture, best practices, and design patterns used in the Space Muck project.

## Project Structure

Space Muck is organized into the following main components:

- **Entities**: Core classes that represent objects in the system
- **Utils**: Utility functions and helper classes
- **Tests**: Unit tests and benchmarks
- **Documentation**: Project documentation

## Best Practices

### Code Quality

1. Follow PEP 8 style guidelines
2. Use type hints for better code readability and IDE support
3. Keep functions small and focused on a single responsibility
4. Document all public functions, classes, and modules
5. Use consistent naming conventions
6. Follow standardized import structure (see Import Structure section below)

### Function Naming Conventions

1. **Descriptive Function Names**: Use descriptive names that clearly indicate the function's purpose
   - Example: `_generate_fallback_noise` instead of `_extracted_from_apply_noise_generation_gpu_49`
   - Example: `_calculate_territory_metrics` instead of `_territory_handler`

2. **Handler Functions**: Use the pattern `_handle_X` for functions that handle specific cases or errors
   - Example: `_handle_benchmark_failure` for handling benchmark failures
   - Example: `_handle_bandwidth_benchmark_failure` for handling bandwidth benchmark failures

3. **Parameter Naming**: Use descriptive parameter names instead of generic names like `arg0`, `arg1`
   - Example: `results_dict` instead of `arg0`
   - Example: `operation_prefix` instead of `arg2`

4. **Docstrings**: All functions should have proper docstrings with Args and Returns sections
   ```python
   def function_name(param1, param2):
       """Short description of what the function does.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
       """
   ```

### Import Structure

1. **Import Organization**: Organize imports in the following order with a blank line between each group:
   ```python
   # Standard library imports
   import logging
   import math
   import random
   from typing import Dict, List, Optional
   
   # Third-party library imports
   import numpy as np
   import pygame
   import scipy.ndimage as ndimage
   
   # Local application imports
   from config import GRID_WIDTH, GRID_HEIGHT
   from entities.base_entity import BaseEntity
   from utils.noise_generator import NoiseGenerator
   ```

2. **Relative Imports**: Use relative imports for local application modules instead of absolute imports with 'src' prefix:
   - Correct: `from entities.base_entity import BaseEntity`
   - Incorrect: `from src.entities.base_entity import BaseEntity`

3. **Optional Dependencies**: Handle optional dependencies with try/except blocks:
   ```python
   # Optional dependencies
   try:
       from perlin_noise import PerlinNoise
       PERLIN_AVAILABLE = True
   except ImportError:
       PERLIN_AVAILABLE = False
       logging.warning("PerlinNoise package is not available. Using fallback noise generator.")
   ```

4. **Duplicate Imports**: Avoid duplicate imports of the same module. Check if a module is already imported before trying to import it again:
   ```python
   # Check if scipy is available for optimized cellular automaton
   SCIPY_AVAILABLE = 'signal' in dir(scipy)
   if not SCIPY_AVAILABLE:
       logging.warning("SciPy signal module not available. Using manual implementation for cellular automaton.")
   ```

### GPU Acceleration

1. Always provide a CPU fallback for GPU-accelerated functions
2. Handle optional dependencies gracefully using `importlib.util.find_spec()`
3. Use conditional imports for GPU libraries (cupy, torch)
4. Implement proper error handling for GPU operations
5. Benchmark both CPU and GPU implementations to ensure performance gains

### Testing

1. Maintain 100% test pass rate
2. Use mocking for external dependencies
3. Write deterministic tests that don't rely on implementation details
4. Test both success and error cases
5. Include performance benchmarks for critical operations

## Design Patterns

### Factory Pattern

Used for creating different implementations of generators and processors.

### Strategy Pattern

Used for swappable algorithms in procedural generation.

### Observer Pattern

Used for event handling and notifications.

## Performance Optimization

1. Use vectorized operations with NumPy where possible
2. Implement GPU acceleration for computationally intensive tasks
3. Use caching mechanisms for noise generation
4. Implement parallel processing for independent operations
5. Profile and optimize critical code paths

## Error Handling

1. Use specific exception types
2. Provide meaningful error messages
3. Log errors with appropriate context
4. Implement graceful degradation for optional features
5. Validate inputs to prevent downstream errors

### Import Structure

1. **Module Structure**:
   - The project uses a package-based structure with `src` as the root package
   - Main modules are organized in subdirectories: `entities`, `generators`, `utils`, etc.

2. **Import Paths**:
   - For imports within the same directory, use relative imports:
     ```python
     from .base_generator import BaseGenerator
     ```
   - For imports from other modules, use absolute imports with the `src` prefix:
     ```python
     from src.utils.noise_generator import NoiseGenerator
     ```
   - For imports in test files, ensure the proper path is added to sys.path:
     ```python
     import os
     import sys
     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
     from src.generators.asteroid_field import AsteroidField
     ```

3. **Import Order**:
   - Standard library imports first
   - Third-party library imports second
   - Local application imports last
   - Each group separated by a blank line

4. **Module Location Changes**:
   - The `asteroid_field.py` and related modules are now in the `src/generators` directory
   - Previously, some tests were looking for these modules in `src/world`
   - All import paths have been updated to reflect the correct locations

5. **Import Optimization**:
   - Only import what you need; avoid wildcard imports (`from module import *`)
   - Remove unused imports to prevent linting errors
   - Comment out potentially useful imports for future use instead of removing them completely:
     ```python
     # from collections import defaultdict  # Uncomment if needed in the future
     ```
   - For typing imports, only include what's actually used in type annotations

### Linting Practices

1. **Linting Tools**:
   - We use Ruff as our primary linter for its speed and comprehensive rule set
   - Additional tools include flake8, pylint, and sourcery for more in-depth analysis

2. **Common Linting Issues and Solutions**:

   - **Unused Imports**:
     - Remove unused imports or comment them out if they might be needed later
     - Example: `# from collections import defaultdict  # Uncomment if needed in the future`

   - **Unused Variables**:
     - Remove variable assignments if the variables are not used
     - If the expression has side effects, remove just the assignment part
     - Example: Change `ships_by_priority = sorted(...)` to just `sorted(...)`

   - **Unused Function Parameters**:
     - If parameters are truly unused, prefix them with underscore: `def func(_unused_param)`
     - If they might be used in the future, add a comment explaining why they're kept

   - **Code Style Inconsistencies**:
     - Follow PEP 8 for consistent indentation, line length, and spacing
     - Use consistent naming conventions throughout the codebase

3. **Linting Configuration**:
   - Aim to have zero linting errors in the codebase
   - Configure linting tools to run automatically during development
   - Use inline comments to disable specific linting rules when necessary (sparingly):
     ```python
     # noqa: F401  # Import needed for type checking only
     ```
