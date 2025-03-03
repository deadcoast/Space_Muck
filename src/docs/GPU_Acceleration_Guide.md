# GPU Acceleration Integration Guide

## Overview

This guide explains how to integrate GPU acceleration into Space Muck components using the `gpu_utils` module. GPU acceleration can significantly improve performance for computationally intensive operations, especially with large datasets.

## Prerequisites

The GPU acceleration system has the following optional dependencies:

- **Numba**: For CUDA-based acceleration on NVIDIA GPUs
- **CuPy**: For more general GPU acceleration with a NumPy-compatible API

These dependencies are optional - the system will automatically fall back to CPU implementations if they are not available.

## Installation

To enable GPU acceleration, install the optional dependencies:

```bash
# For NVIDIA CUDA support via Numba
pip install numba

# For more general GPU support via CuPy
pip install cupy
```

## Basic Usage

### Checking GPU Availability

Before using GPU acceleration, check if it's available:

```python
from src.utils.gpu_utils import is_gpu_available, get_available_backends

# Check if any GPU acceleration is available
if is_gpu_available():
    print("GPU acceleration is available!")
    
    # Get list of available backends
    backends = get_available_backends()
    print(f"Available backends: {backends}")
else:
    print("GPU acceleration is not available. Using CPU fallback.")
```

### Using GPU-Accelerated Functions

The module provides GPU-accelerated versions of common operations:

```python
import numpy as np
from src.utils.gpu_utils import apply_cellular_automaton_gpu, apply_noise_generation_gpu

# Create a grid
grid = np.random.randint(0, 2, size=(512, 512))

# Apply GPU-accelerated cellular automaton
next_gen = apply_cellular_automaton_gpu(
    grid, 
    rule_func="conway",  # Predefined rule or custom function
    fallback_to_cpu=True  # Use CPU if GPU is unavailable
)

# Generate noise using GPU acceleration
noise = apply_noise_generation_gpu(
    shape=(1024, 1024),
    scale=0.1,
    octaves=6,
    fallback_to_cpu=True
)
```

### Manual Data Transfer

For more complex operations, you can manually transfer data to and from the GPU:

```python
import numpy as np
from src.utils.gpu_utils import to_gpu, to_cpu, is_gpu_available

# Only proceed with GPU operations if available
if is_gpu_available():
    # Create a NumPy array
    cpu_array = np.random.random((1000, 1000))
    
    # Transfer to GPU
    gpu_array = to_gpu(cpu_array)
    
    # Perform operations on the GPU array
    # (The exact operations depend on the backend)
    
    # Transfer back to CPU
    result_cpu = to_cpu(gpu_array)
else:
    # Fallback to CPU implementation
    pass
```

## Integration Examples

### Integrating with Cellular Automaton

```python
from src.utils.gpu_utils import apply_cellular_automaton_gpu
from src.generators.cellular_automaton import apply_cellular_automaton

def generate_cave_system(width, height, iterations):
    # Initialize grid
    grid = initialize_random_grid(width, height)
    
    # Use GPU acceleration for large grids
    if width * height >= 65536:  # 256x256 or larger
        for _ in range(iterations):
            grid = apply_cellular_automaton_gpu(
                grid, 
                rule_func="cave_generation",
                fallback_to_cpu=True
            )
    else:
        # Use CPU implementation for smaller grids
        for _ in range(iterations):
            grid = apply_cellular_automaton(grid, rule="cave_generation")
            
    return grid
```

### Integrating with Noise Generation

```python
from src.utils.gpu_utils import apply_noise_generation_gpu
from src.generators.noise_generator import generate_perlin_noise

def generate_terrain(width, height, scale):
    # Use GPU acceleration for large terrain
    if width * height >= 262144:  # 512x512 or larger
        return apply_noise_generation_gpu(
            shape=(height, width),
            scale=scale,
            octaves=6,
            fallback_to_cpu=True
        )
    else:
        # Use CPU implementation for smaller terrain
        return generate_perlin_noise(width, height, scale, octaves=6)
```

## Performance Considerations

1. **Grid Size Threshold**: GPU acceleration typically provides benefits for larger grids (>256x256). For smaller grids, the CPU implementation may be faster due to memory transfer overhead.

2. **Batch Processing**: When processing multiple grids, batch them together when possible to maximize GPU utilization.

3. **Memory Management**: Minimize transfers between CPU and GPU memory, as these transfers can be expensive.

4. **Backend Selection**: Different backends may perform better for different operations. Use the benchmarking tools to determine the optimal backend for your specific use case.

## Benchmarking

Use the benchmarking tools to measure performance:

```python
from src.tests.benchmark_gpu_acceleration import run_benchmark

# Run benchmark for cellular automaton
results_ca = run_benchmark(
    operation="cellular_automaton",
    grid_sizes=[128, 256, 512, 1024],
    iterations=10
)

# Run benchmark for noise generation
results_noise = run_benchmark(
    operation="noise_generation",
    grid_sizes=[128, 256, 512, 1024],
    iterations=5
)

# Generate performance visualization
from src.tests.benchmark_gpu_acceleration import plot_benchmark_results
plot_benchmark_results(results_ca, "cellular_automaton_benchmark.png")
plot_benchmark_results(results_noise, "noise_generation_benchmark.png")
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**: Ensure that your GPU drivers are up-to-date and that the required libraries (Numba, CuPy) are properly installed.

2. **Out of Memory Errors**: Reduce the size of your data or process it in smaller chunks.

3. **Inconsistent Results**: Ensure that you're using the same random seed for both CPU and GPU implementations when comparing results.

4. **Performance Not Improved**: For small datasets, the overhead of transferring data to the GPU may outweigh the benefits. Use the benchmarking tools to determine the optimal approach for your specific use case.

### Debugging

Enable debug logging to get more information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom GPU-Accelerated Functions

You can create custom GPU-accelerated functions:

```python
from src.utils.gpu_utils import to_gpu, to_cpu, is_gpu_available

def my_custom_gpu_function(data, param1, param2, fallback_to_cpu=True):
    """
    A custom GPU-accelerated function.
    
    Args:
        data: Input data as NumPy array
        param1, param2: Function parameters
        fallback_to_cpu: Whether to fall back to CPU if GPU is unavailable
        
    Returns:
        Processed data as NumPy array
    """
    if is_gpu_available():
        try:
            # Transfer data to GPU
            gpu_data = to_gpu(data)
            
            # Perform GPU-accelerated operations
            # ...
            
            # Transfer result back to CPU
            result = to_cpu(gpu_result)
            return result
        except Exception as e:
            if not fallback_to_cpu:
                raise e
            # Log the error and fall back to CPU
            logging.warning(f"GPU acceleration failed: {e}. Falling back to CPU.")
    
    if not is_gpu_available() or fallback_to_cpu:
        # CPU implementation
        # ...
        return cpu_result
    
    raise RuntimeError("GPU acceleration failed and fallback to CPU was disabled.")
```

## Conclusion

GPU acceleration can significantly improve performance for computationally intensive operations in Space Muck. By following this guide, you can integrate GPU acceleration into your components while maintaining compatibility with systems that don't have GPU support.

Remember to always provide a CPU fallback for systems without GPU support, and use the benchmarking tools to determine when GPU acceleration is beneficial for your specific use case.
