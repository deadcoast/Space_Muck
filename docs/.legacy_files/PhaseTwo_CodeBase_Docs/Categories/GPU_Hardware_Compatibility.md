# GPU Hardware Compatibility Guide

## Overview

This document outlines hardware requirements and compatibility considerations for using GPU acceleration features in Space Muck. The GPU acceleration system is designed to be flexible and adaptable to different hardware configurations, with graceful fallbacks for systems without GPU support.

## Supported Hardware

### NVIDIA GPUs (via CUDA)

For optimal performance with CUDA-based acceleration (Numba backend):

| GPU Series | Minimum CUDA Compute Capability | Recommended Models |
|------------|--------------------------------|-------------------|
| GeForce    | 3.0+ | GTX 1060 or newer |
| RTX        | 5.0+ | Any RTX series |
| Quadro     | 3.0+ | P2000 or newer |
| Tesla      | 3.0+ | Any Tesla series |

**Required Software:**
- CUDA Toolkit 10.0 or newer
- Compatible NVIDIA drivers

### AMD GPUs (via ROCm/HIP with CuPy)

For AMD GPU support via CuPy with ROCm backend:

| GPU Series | Minimum Requirements | Recommended Models |
|------------|----------------------|-------------------|
| Radeon RX  | GCN 3rd gen or newer | RX 580 or newer |
| Radeon Pro | GCN 3rd gen or newer | WX 5100 or newer |
| Instinct   | Any | Any Instinct series |

**Required Software:**
- ROCm 4.0 or newer
- Compatible AMD drivers

### Intel GPUs (via oneAPI with CuPy)

Limited support for Intel GPUs via CuPy with oneAPI backend:

| GPU Series | Minimum Requirements | Recommended Models |
|------------|----------------------|-------------------|
| Iris Xe    | Gen12 or newer | Any Iris Xe |
| Arc        | Any | Any Arc series |

**Required Software:**
- Intel oneAPI Base Toolkit
- Compatible Intel drivers

## Memory Requirements

| Grid Size | Minimum VRAM | Recommended VRAM |
|-----------|--------------|------------------|
| 256x256   | 1 GB         | 2 GB             |
| 512x512   | 2 GB         | 4 GB             |
| 1024x1024 | 4 GB         | 8 GB             |
| 2048x2048 | 8 GB         | 16 GB            |

## CPU Fallback Performance

When GPU acceleration is unavailable, the system falls back to CPU implementations. Here are approximate performance expectations for CPU fallback:

| CPU Type | Relative Performance | Notes |
|----------|----------------------|-------|
| Modern Desktop (8+ cores) | 20-40% of GPU speed | Good fallback option |
| Mid-range Laptop (4-6 cores) | 10-20% of GPU speed | Acceptable for medium grids |
| Older/Low-end (2-4 cores) | 5-10% of GPU speed | May struggle with large grids |

## Compatibility Matrix

| Feature | NVIDIA (CUDA) | AMD (ROCm) | Intel (oneAPI) | CPU Fallback |
|---------|---------------|------------|----------------|--------------|
| Cellular Automaton | Full support | Full support | Partial support | Full support |
| Noise Generation | Full support | Full support | Partial support | Full support |
| Clustering Algorithms | Full support | Partial support | Limited support | Full support |
| Memory Efficiency | Excellent | Good | Fair | N/A |
| Setup Complexity | Moderate | High | High | None |

## Installation Requirements

### NVIDIA CUDA Setup

```bash
# Install CUDA Toolkit (Linux example)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# Install Python dependencies
pip install numba cupy-cuda11x  # Replace with appropriate CUDA version
```

### AMD ROCm Setup

```bash
# Install ROCm (Linux example)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev

# Install Python dependencies
pip install cupy-rocm
```

### Intel oneAPI Setup

```bash
# Install oneAPI (Linux example)
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-base

# Install Python dependencies
pip install cupy-intel
```

## Troubleshooting Common Issues

### NVIDIA GPUs

1. **CUDA Not Found**
   - Ensure CUDA Toolkit is properly installed
   - Check that PATH and LD_LIBRARY_PATH include CUDA directories
   - Verify that `nvcc --version` returns the correct version

2. **Incompatible CUDA Version**
   - Make sure the installed CuPy version matches your CUDA version
   - Use `pip install cupy-cuda11x` where x matches your CUDA version

3. **Driver Version Mismatch**
   - Update GPU drivers to version compatible with CUDA Toolkit

### AMD GPUs

1. **ROCm Not Found**
   - Ensure ROCm is properly installed
   - Check that PATH includes ROCm directories
   - Verify that `rocm-smi` works correctly

2. **HIP Compilation Errors**
   - Make sure your GPU is supported by ROCm
   - Check for ROCm version compatibility with CuPy

### General Issues

1. **Out of Memory Errors**
   - Reduce grid size or batch size
   - Close other GPU-intensive applications
   - Consider using CPU fallback for very large operations

2. **Performance Lower Than Expected**
   - Check for thermal throttling
   - Ensure no other processes are using the GPU
   - Try different batch sizes to find optimal performance

3. **Inconsistent Results Between CPU and GPU**
   - Ensure random seeds are set consistently
   - Check for floating-point precision differences
   - Verify boundary condition handling

## Benchmarking Your Hardware

To evaluate your specific hardware configuration:

```python
from src.tests.benchmark_gpu_acceleration import run_comprehensive_benchmark

# Run comprehensive benchmark across operations and grid sizes
results = run_comprehensive_benchmark(
    operations=["cellular_automaton", "noise_generation"],
    grid_sizes=[128, 256, 512, 1024],
    iterations=5,
    generate_plots=True
)

# Results will be saved to benchmark_results directory
```

## Conclusion

The GPU acceleration system in Space Muck is designed to work across a wide range of hardware configurations, with automatic fallbacks to ensure compatibility. For optimal performance, an NVIDIA GPU with CUDA support is recommended, but the system will adapt to available hardware capabilities.

When deploying on systems with limited or no GPU support, consider adjusting grid sizes and operation complexity to maintain acceptable performance.
