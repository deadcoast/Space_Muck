# CodeBase Error Fixes

This document logs common issues encountered during development and their solutions to avoid repeating the same mistakes.

## Table of Contents

1. [Import Issues](#import-issues)
2. [Linting Issues](#linting-issues)
3. [GPU Acceleration Issues](#gpu-acceleration-issues)

## Import Issues

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
