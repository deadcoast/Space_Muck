# CodeBase Error Fixes

This document logs common issues encountered during development and their solutions to avoid repeating the same mistakes.

## Table of Contents

1. [Import Issues](#import-issues)
2. [Linting Issues](#linting-issues)
3. [GPU Acceleration Issues](#gpu-acceleration-issues)

## Import Issues

### Import Structure Inconsistencies

**Issue**: Inconsistent import patterns across the codebase, including:
1. Mixing absolute imports with 'src' prefix and relative imports
2. Duplicate imports of the same module
3. Inconsistent import organization

**Solution**: Standardize import structure following these guidelines:

1. **Organize imports** in a consistent order:
   ```python
   # Standard library imports
   import logging
   import math
   
   # Third-party library imports
   import numpy as np
   import pygame
   
   # Local application imports
   from entities.base_entity import BaseEntity
   ```

2. **Use relative imports** for local application modules:
   ```python
   # Incorrect - using absolute imports with 'src' prefix
   from src.utils.gpu_utils import apply_noise_generation_gpu
   
   # Correct - using relative imports
   from utils.gpu_utils import apply_noise_generation_gpu
   ```

3. **Avoid duplicate imports** by checking if a module is already imported:
   ```python
   # Incorrect - duplicate import
   import scipy
   import scipy.signal
   
   # Correct - single import
   import scipy
   from scipy import signal
   ```

### Test Import Path Issues

**Issue**: Test files were importing modules from incorrect paths, such as:
1. Importing from `src.world.asteroid_field` when the module is actually in `src.generators.asteroid_field`
2. Importing from `src.world.procedural_generation` when the module is actually in `src.generators.asteroid_generator`

**Solution**: Update import paths in test files to match the actual module structure:

1. **Update import statements** in test files:
   ```python
   # Incorrect - importing from non-existent path
   from src.world.asteroid_field import AsteroidField
   
   # Correct - importing from actual module location
   from src.generators.asteroid_field import AsteroidField
   ```

2. **Update function references** when module structure changes:
   ```python
   # Incorrect - using function from non-existent module
   field_data = generate_field(width, height)
   
   # Correct - using class method from actual module
   generator = AsteroidGenerator(width=width, height=height)
   asteroid_grid, metadata = generator.generate_field()
   ```

3. **Ensure proper path setup** in test files:
   ```python
   import os
   import sys
   
   # Add the src directory to the path
   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
   ```

### Optional Dependencies

**Issue**: Missing imports for optional dependencies like `cupy` and `torch` causing linting errors.

**Solution**: Use a robust pattern for handling optional dependencies with proper type checking:

```python
import importlib.util
from typing import TYPE_CHECKING

# Check for optional dependencies
CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# For type checking only - these imports are not executed at runtime
if TYPE_CHECKING:
    try:
        import cupy  # type: ignore
        import torch  # type: ignore
    except ImportError:
        pass

# Import optional dependencies at runtime
cp = None  # Define cp at module level
torch = None  # Define torch at module level

if CUPY_AVAILABLE:
    try:
        import cupy as cp
    except ImportError:
        pass

if TORCH_AVAILABLE:
    try:
        import torch
    except ImportError:
        pass
```

Then use these flags to conditionally execute code:

```python
if backend in ["cuda", "cupy"] and CUPY_AVAILABLE:
    # CUDA/CuPy specific code
elif backend == "mps" and TORCH_AVAILABLE:
    # MPS/PyTorch specific code
```

This approach:
1. Uses `importlib.util.find_spec()` to check if the dependency is installed
2. Defines module-level variables to prevent undefined variable errors
3. Uses `TYPE_CHECKING` with try/except for type hints without runtime errors
4. Adds extra error handling with try/except during imports
5. Avoids unused import warnings by not importing unused typing modules

## Linting Issues

### Unused Imports

**Issue**: Unused imports causing linting warnings.

**Solution**: Remove unused imports or use them if they're necessary. Common unused imports found:
- `argparse` when not using command-line arguments
- `from typing import Tuple, Optional` when not using these types
- Specific GPU utility imports when not using them

### Unused Variables

**Issue**: Variables assigned but never used.

**Solution**: Remove the variable assignment if the value isn't needed, or use the variable:

```python
# Before
start_time = time.time()
result = func(*args, **kwargs)  # result is never used
end_time = time.time()

# After
start_time = time.time()
func(*args, **kwargs)  # Don't assign if not using the result
end_time = time.time()
```

## GPU Acceleration Issues

### Handling Backend Failures

**Issue**: Repetitive code for handling failed backends in benchmark functions.

**Solution**: Extract common patterns into helper functions:

```python
def _time_handler(results, backend, prefix):
    results["times"][backend].append(float("inf"))
    results["speedup"][backend].append(0.0)
    logging.info(f"{prefix}{backend}: Failed")

def _bandwidth_handler(results, backend, prefix):
    results["times"][backend].append(float("inf"))
    results["bandwidth"][backend].append(0.0)
    logging.info(f"{prefix}{backend}: Failed")
```

## Code Quality Issues

### Poorly Named Functions

**Issue**: Functions with generic extracted names (e.g., `_extracted_from_function_name_123`) that lack descriptive meaning.

**Solution**: Rename these functions with descriptive names that indicate their purpose and add proper docstrings:

```python
# Before
# TODO Rename this here and in `apply_noise_generation_gpu`
def _extracted_from_apply_noise_generation_gpu_49(seed, height, width):
    logging.warning("Noise generator not available. Using random noise.")
    rng = np.random.RandomState(seed)
    return rng.random((height, width))

# After
def _generate_fallback_noise(seed, height, width):
    """Generate random noise as a fallback when noise generators are unavailable.
    
    Args:
        seed: Random seed for reproducibility
        height: Height of the noise grid
        width: Width of the noise grid
        
    Returns:
        np.ndarray: Random noise grid
    """
    logging.warning("Noise generator not available. Using random noise.")
    rng = np.random.RandomState(seed)
    return rng.random((height, width))
```

Also update all references to the renamed function:

```python
# Before
return _extracted_from_apply_noise_generation_gpu_49(seed, height, width)

# After
return _generate_fallback_noise(seed, height, width)
```

### Monolithic Methods

**Issue**: Large methods that handle multiple responsibilities, making them difficult to understand, test, and maintain.

**Solution**: Refactor large methods into smaller, more focused methods with clear responsibilities:

```python
# Before: Single method handling multiple generator types
def _asteroid_handler(self):
    # Common parameters setup
    # ...
    
    if ASTEROID_GENERATOR_AVAILABLE:
        # Optimized generator logic
        # ...
    else:
        # Procedural generator logic
        # ...
    
    # Common post-processing
    # ...

# After: Split into multiple methods with clear responsibilities
def _asteroid_handler(self) -> None:
    """Generate asteroid field using available generators."""
    # Common parameters setup
    # ...
    
    try:
        if ASTEROID_GENERATOR_AVAILABLE:
            self._generate_asteroid_field_with_optimized_generator(common_params, seed)
        else:
            self._generate_asteroid_field_with_procedural_generator(common_params, seed)
        
        # Common post-processing
        # ...
    except Exception as e:
        # Error handling
        # ...

def _generate_asteroid_field_with_optimized_generator(self, common_params: dict, seed: int) -> None:
    """Generate asteroid field using the optimized AsteroidGenerator."""
    # Optimized generator logic
    # ...

def _generate_asteroid_field_with_procedural_generator(self, common_params: dict, seed: int) -> None:
    """Generate asteroid field using the ProceduralGenerator."""
    # Procedural generator logic
    # ...
```

This approach:
1. Improves readability by breaking down complex logic into smaller, focused methods
2. Enhances maintainability by isolating different generator implementations
3. Makes testing easier by allowing each method to be tested independently
4. Provides clear documentation through method names and docstrings
5. Centralizes error handling in the parent method
