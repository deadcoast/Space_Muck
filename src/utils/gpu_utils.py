#!/usr/bin/env python3
"""
GPU Acceleration utilities for Space Muck.

This module provides GPU-accelerated implementations of common operations
used throughout the codebase, with fallback mechanisms for systems without
GPU support.
"""

import contextlib
import importlib.util

# Standard library imports
import itertools
import logging
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Union

# Third-party library imports
import numpy as np
from numpy.random import PCG64, Generator

# Initialize random number generator with a fixed seed for reproducibility
rng = Generator(PCG64(42))

# Check for optional dependencies
NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None

# For type checking only
if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        import cupy as cp  # type: ignore
        import numba  # type: ignore
        from numba import cuda  # type: ignore
# Import Numba at runtime if available
if NUMBA_AVAILABLE:
    try:
        import numba
        from numba import cuda

        if CUDA_AVAILABLE := cuda.is_available():
            logging.info("CUDA is available for GPU acceleration")
        else:
            logging.info("Numba is available but CUDA is not detected")
    except ImportError:
        NUMBA_AVAILABLE = False
        CUDA_AVAILABLE = False
        logging.warning("Error importing Numba, GPU acceleration disabled")
else:
    CUDA_AVAILABLE = False

# Import CuPy at runtime if available
cp = None  # Define at module level to prevent undefined variable errors
if CUPY_AVAILABLE:
    try:
        import cupy as cp

        logging.info("CuPy is available for GPU acceleration")
    except ImportError:
        CUPY_AVAILABLE = False
        logging.warning("Error importing CuPy, some GPU operations will be disabled")
    logging.warning("Numba not available. GPU acceleration will be disabled.")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    logging.info("CuPy is available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. Some GPU operations will be disabled.")

# Try to import PyTorch with MPS support for macOS
try:
    import torch

    TORCH_AVAILABLE = True
    if (
        MPS_AVAILABLE := hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        logging.info("PyTorch with MPS is available for GPU acceleration on macOS")
        # Create MPS device
        mps_device = torch.device("mps")
    else:
        logging.info("PyTorch is available but MPS is not detected")

except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    logging.warning("PyTorch not available. macOS GPU acceleration will be disabled.")

# Check for metalgpu availability without importing it directly to avoid lint errors
# The actual import will happen in the specific functions that use it
METALGPU_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("metalgpu") is not None:
        METALGPU_AVAILABLE = True
        logging.info("metalgpu is available for GPU acceleration on macOS")
    else:
        logging.warning(
            "metalgpu not available. Alternative macOS GPU acceleration will be disabled."
        )
except Exception as e:
    logging.warning(f"Error checking for metalgpu: {e}")


def is_gpu_available() -> bool:
    """
    Check if any GPU acceleration is available.

    Returns:
        bool: True if any GPU acceleration is available
    """
    return CUDA_AVAILABLE or CUPY_AVAILABLE or MPS_AVAILABLE or METALGPU_AVAILABLE


def get_available_backends() -> List[str]:
    """
    Get a list of available GPU backends.

    Returns:
        List[str]: List of available backends
    """
    backends = []
    if CUDA_AVAILABLE:
        backends.append("cuda")
    if CUPY_AVAILABLE:
        backends.append("cupy")
    if NUMBA_AVAILABLE:
        backends.append("numba")
    if MPS_AVAILABLE:
        backends.append("mps")
    if METALGPU_AVAILABLE:
        backends.append("metalgpu")
    if not backends:
        backends.append("cpu")
    return backends


def to_gpu(array: np.ndarray) -> Union[np.ndarray, Any]:
    """
    Transfer a numpy array to the GPU if available.

    Args:
        array: Input numpy array

    Returns:
        Union[np.ndarray, Any]: Array on GPU if available, otherwise original array
    """
    # Try CuPy first (NVIDIA/AMD GPUs)
    if CUPY_AVAILABLE:
        return cp.asarray(array)

    # Try PyTorch MPS for macOS
    if MPS_AVAILABLE and TORCH_AVAILABLE:
        try:
            # Convert numpy array to PyTorch tensor and move to MPS device
            return torch.from_numpy(array).to(device=mps_device)
        except Exception as e:
            logging.warning(f"Failed to transfer array to MPS: {e}")

    # If no GPU transfer was successful, return the original array
    return array


def to_cpu(array: Any) -> np.ndarray:
    """
    Transfer an array from GPU back to CPU if necessary.

    Args:
        array: Input array (either on GPU or CPU)

    Returns:
        np.ndarray: Array on CPU
    """
    # Handle CuPy arrays
    if CUPY_AVAILABLE and cp is not None and isinstance(array, cp.ndarray):
        return array.get()

    # Handle PyTorch tensors (including those on MPS device)
    if TORCH_AVAILABLE and isinstance(array, torch.Tensor):
        try:
            # Move tensor to CPU and convert to numpy
            return array.detach().cpu().numpy()
        except Exception as e:
            logging.warning(f"Failed to transfer tensor from GPU to CPU: {e}")
            # If conversion fails, try to get the tensor values directly
            if hasattr(array, "numpy"):
                return array.numpy()

    # If it's already a numpy array or another type, return as is
    return array


if NUMBA_AVAILABLE and CUDA_AVAILABLE:

    @cuda.jit
    def _apply_cellular_automaton_kernel(
        grid, new_grid, width, height, birth_set, survival_set
    ):
        """CUDA kernel for cellular automaton.

        This kernel counts the neighbors for each cell and applies the cellular automaton rules.
        The kernel is optimized to reduce cognitive complexity while maintaining CUDA compatibility.

        Args:
            grid: Current grid state
            new_grid: New grid state to update
            width: Width of the grid
            height: Height of the grid
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
        """
        # Get current thread coordinates
        x, y = cuda.grid(2)

        # Skip if outside grid bounds
        if x >= width or y >= height:
            return

        # Initialize neighbor count
        neighbors = 0

        # Calculate row and column indices with wrapping
        # This reduces cognitive complexity by pre-calculating all indices
        prev_row = (y - 1) % height
        next_row = (y + 1) % height
        prev_col = (x - 1) % width
        next_col = (x + 1) % width

        # Count neighbors (unrolled for CUDA optimization)
        # Top row
        neighbors += grid[prev_row, prev_col] > 0
        neighbors += grid[prev_row, x] > 0
        neighbors += grid[prev_row, next_col] > 0

        # Middle row (excluding center)
        neighbors += grid[y, prev_col] > 0
        neighbors += grid[y, next_col] > 0

        # Bottom row
        neighbors += grid[next_row, prev_col] > 0
        neighbors += grid[next_row, x] > 0
        neighbors += grid[next_row, next_col] > 0

        # Apply cellular automaton rules
        # Check if cell is currently alive
        if grid[y, x] > 0:
            # Apply survival rule
            new_grid[y, x] = 1 if neighbors in survival_set else 0
        else:
            # Apply birth rule
            new_grid[y, x] = 1 if neighbors in birth_set else 0


def _determine_backend_for_cellular_automaton(backend: str) -> str:
    """
    Determine the appropriate backend for cellular automaton based on availability.

    Args:
        backend: Requested backend ('cuda', 'cupy', 'mps', 'metalgpu', 'auto')

    Returns:
        str: Selected backend to use
    """
    if backend != "auto":
        return backend

    if CUDA_AVAILABLE:
        return "cuda"
    elif CUPY_AVAILABLE:
        return "cupy"
    else:
        return "cpu"


def _apply_cellular_automaton_cpu(
    grid: np.ndarray,
    birth_set: Set[int],
    survival_set: Set[int],
    iterations: int,
    wrap: bool,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Apply cellular automaton rules using CPU implementation.

    Args:
        grid: Input grid to evolve
        birth_set: Set of neighbor counts that cause cell birth
        survival_set: Set of neighbor counts that allow cell survival
        iterations: Number of iterations to perform
        wrap: Whether to wrap around grid edges
        width: Width of the grid
        height: Height of the grid

    Returns:
        np.ndarray: Evolved grid
    """
    from utils.cellular_automaton_utils import apply_cellular_automaton

    return apply_cellular_automaton(
        grid, birth_set, survival_set, iterations, wrap, width, height
    )


def _apply_cellular_automaton_cuda(
    grid: np.ndarray,
    birth_list: List[int],
    survival_list: List[int],
    iterations: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Apply cellular automaton rules using CUDA implementation.

    Args:
        grid: Input grid to evolve
        birth_list: List of neighbor counts that cause cell birth
        survival_list: List of neighbor counts that allow cell survival
        iterations: Number of iterations to perform
        width: Width of the grid
        height: Height of the grid

    Returns:
        np.ndarray: Evolved grid
    """
    # Convert to binary grid
    binary_grid = (grid > 0).astype(np.int8)
    d_grid = binary_grid.copy()

    # Configure CUDA grid
    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Create device arrays for birth and survival sets
    d_birth_set = numba.cuda.to_device(np.array(birth_list, dtype=np.int8))
    d_survival_set = numba.cuda.to_device(np.array(survival_list, dtype=np.int8))

    for _ in range(iterations):
        d_new_grid = np.zeros_like(d_grid)
        _apply_cellular_automaton_kernel[blockspergrid, threadsperblock](
            d_grid, d_new_grid, width, height, d_birth_set, d_survival_set
        )
        d_grid = d_new_grid

    # Preserve original values where cells are alive
    return grid * d_grid


def _count_neighbors_cupy(d_grid: Any, wrap: bool) -> Any:
    """
    Count neighbors for each cell using CuPy.

    Args:
        d_grid: Grid on GPU
        wrap: Whether to wrap around grid edges

    Returns:
        Any: Neighbor count for each cell (cp.ndarray when CuPy is available)
    """
    neighbors_kernel = cp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    if wrap:
        return cp.fft.ifft2(
            cp.fft.fft2(d_grid) * cp.fft.fft2(neighbors_kernel, d_grid.shape)
        ).real.astype(cp.int8)
    neighbor_count = cp.zeros_like(d_grid)
    for dy, dx in itertools.product(range(-1, 2), range(-1, 2)):
        if dx == 0 and dy == 0:
            continue
        neighbor_count += cp.roll(d_grid, (dy, dx), axis=(0, 1))
    return neighbor_count


def _apply_cellular_automaton_rules_cupy(
    d_grid: Any,
    neighbor_count: Any,
    birth_list: List[int],
    survival_list: List[int],
) -> Any:
    """
    Apply cellular automaton rules using CuPy.

    Args:
        d_grid: Current grid state on GPU
        neighbor_count: Neighbor count for each cell
        birth_list: List of neighbor counts that cause cell birth
        survival_list: List of neighbor counts that allow cell survival

    Returns:
        Any: New grid state (cp.ndarray when CuPy is available)
    """
    new_grid = cp.zeros_like(d_grid)

    # Apply survival rules
    for n in survival_list:
        new_grid |= (neighbor_count == n) & (d_grid > 0)

    # Apply birth rules
    for n in birth_list:
        new_grid |= (neighbor_count == n) & (d_grid == 0)

    return new_grid


def _apply_cellular_automaton_cupy(
    grid: np.ndarray,
    birth_list: List[int],
    survival_list: List[int],
    iterations: int,
    wrap: bool,
) -> np.ndarray:
    """
    Apply cellular automaton rules using CuPy implementation.

    Args:
        grid: Input grid to evolve
        birth_list: List of neighbor counts that cause cell birth
        survival_list: List of neighbor counts that allow cell survival
        iterations: Number of iterations to perform
        wrap: Whether to wrap around grid edges

    Returns:
        np.ndarray: Evolved grid
    """
    # Transfer to GPU
    d_grid = cp.asarray((grid > 0).astype(np.int8))

    for _ in range(iterations):
        # Count neighbors
        neighbor_count = _count_neighbors_cupy(d_grid, wrap)

        # Apply rules
        d_grid = _apply_cellular_automaton_rules_cupy(
            d_grid, neighbor_count, birth_list, survival_list
        )

    # Transfer back to CPU and preserve original values
    result_grid = cp.asnumpy(d_grid)
    return grid * result_grid


def apply_cellular_automaton_gpu(
    grid: np.ndarray,
    birth_set: Set[int] = None,
    survival_set: Set[int] = None,
    iterations: int = 3,
    wrap: bool = True,
    backend: str = "auto",
) -> np.ndarray:
    """
    Apply cellular automaton rules using GPU acceleration if available.

    Args:
        grid: Input grid to evolve
        birth_set: Set of neighbor counts that cause cell birth
        survival_set: Set of neighbor counts that allow cell survival
        iterations: Number of iterations to perform
        wrap: Whether to wrap around grid edges
        backend: GPU backend to use ('cuda', 'cupy', 'mps', 'metalgpu', 'auto')

    Returns:
        np.ndarray: Evolved grid
    """
    # Set default values
    if birth_set is None:
        birth_set = {3}
    if survival_set is None:
        survival_set = {2, 3}
    height, width = grid.shape

    # Convert sets to lists for GPU compatibility
    birth_list = list(birth_set)
    survival_list = list(survival_set)

    # Determine backend to use
    backend = _determine_backend_for_cellular_automaton(backend)

    # Use CPU implementation if no GPU is available or requested
    if backend == "cpu" or (
        not CUDA_AVAILABLE
        and not CUPY_AVAILABLE
        and not MPS_AVAILABLE
        and not METALGPU_AVAILABLE
    ):
        return _apply_cellular_automaton_cpu(
            grid, birth_set, survival_set, iterations, wrap, width, height
        )

    # CUDA implementation using Numba
    if backend == "cuda" and CUDA_AVAILABLE:
        return _apply_cellular_automaton_cuda(
            grid, birth_list, survival_list, iterations, width, height
        )

    # CuPy implementation
    if backend == "cupy" and CUPY_AVAILABLE:
        return _apply_cellular_automaton_cupy(
            grid, birth_list, survival_list, iterations, wrap
        )

    # Fallback to CPU implementation
    return _apply_cellular_automaton_cpu(
        grid, birth_set, survival_set, iterations, wrap, width, height
    )


def _determine_backend_for_noise_generation(backend: str) -> str:
    """
    Determine the appropriate backend for noise generation based on availability.

    Args:
        backend: Requested backend ('cuda', 'cupy', 'mps', 'metalgpu', 'auto')

    Returns:
        str: Selected backend to use
    """
    if backend != "auto":
        return backend

    if CUPY_AVAILABLE:
        return "cupy"
    elif CUDA_AVAILABLE:
        return "cuda"
    elif MPS_AVAILABLE:
        return "mps"
    elif METALGPU_AVAILABLE:
        return "metalgpu"
    else:
        return "cpu"


def _generate_noise_cpu(
    width: int,
    height: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: Optional[int],
) -> np.ndarray:
    """
    Generate Perlin noise using CPU implementation.

    Args:
        width: Width of the noise grid
        height: Height of the noise grid
        scale: Scale of the noise
        octaves: Number of octaves for the noise
        persistence: Persistence parameter
        lacunarity: Lacunarity parameter
        seed: Random seed

    Returns:
        np.ndarray: Generated noise grid
    """
    try:
        from utils.noise_generator import generate_perlin_noise

        return generate_perlin_noise(
            width, height, scale, octaves, persistence, lacunarity, seed
        )
    except ImportError:
        return _generate_fallback_noise(seed, height, width)


def _generate_noise_layer_mps(
    height: int, width: int, frequency: float, device: torch.device
) -> torch.Tensor:
    """
    Generate a single noise layer using MPS with blur-based smoothing.

    Args:
        height: Height of the noise grid
        width: Width of the noise grid
        frequency: Frequency for this octave
        device: MPS device to use

    Returns:
        torch.Tensor: Generated noise layer
    """
    # Generate base noise
    noise_layer = torch.rand((height, width), dtype=torch.float32, device=device)

    # Apply frequency domain filtering (simplified for MPS)
    # We use a Gaussian blur as a substitute for frequency domain filtering
    # since FFT operations might not be fully optimized on MPS
    blur_size = int(1.0 / frequency)
    if blur_size > 0:
        # Ensure blur_size is odd
        blur_size = blur_size * 2 + 1 if blur_size > 0 else 3
        # Apply blur using average pooling with appropriate padding
        padded = torch.nn.functional.pad(
            noise_layer.unsqueeze(0).unsqueeze(0),
            (
                blur_size // 2,
                blur_size // 2,
                blur_size // 2,
                blur_size // 2,
            ),
            mode="reflect",
        )
        noise_layer = (
            torch.nn.functional.avg_pool2d(padded, blur_size, stride=1)
            .squeeze(0)
            .squeeze(0)
        )

    # Normalize
    min_val = torch.min(noise_layer)
    max_val = torch.max(noise_layer)
    noise_layer = (noise_layer - min_val) / (max_val - min_val + 1e-8)

    return noise_layer


def _generate_noise_mps(
    width: int,
    height: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: Optional[int],
) -> np.ndarray:
    """
    Generate Perlin noise using MPS implementation.

    Args:
        width: Width of the noise grid
        height: Height of the noise grid
        scale: Scale of the noise
        octaves: Number of octaves for the noise
        persistence: Persistence parameter
        lacunarity: Lacunarity parameter
        seed: Random seed

    Returns:
        np.ndarray: Generated noise grid
    """
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.zeros((height, width), dtype=torch.float32, device=mps_device)

    # Generate noise with multiple octaves
    max_amplitude = 0
    amplitude = 1.0
    frequency = scale

    for _ in range(octaves):
        # Generate and process noise layer
        noise_layer = _generate_noise_layer_mps(height, width, frequency, mps_device)

        # Add to final noise
        noise += amplitude * noise_layer

        # Update parameters for next octave
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize the final noise
    noise /= max_amplitude

    # Transfer back to CPU
    return to_cpu(noise)


def _generate_noise_metalgpu(
    width: int,
    height: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: Optional[int],
) -> np.ndarray:
    """
    Generate Perlin noise using Metal GPU implementation.

    Args:
        width: Width of the noise grid
        height: Height of the noise grid
        scale: Scale of the noise
        octaves: Number of octaves for the noise
        persistence: Persistence parameter
        lacunarity: Lacunarity parameter
        seed: Random seed

    Returns:
        np.ndarray: Generated noise grid
    """
    # Import metalgpu here to avoid lint errors when it's not used
    import metalgpu

    # Log that we're using metalgpu for noise generation
    logging.info(
        f"Using metalgpu version {metalgpu.__version__ if hasattr(metalgpu, '__version__') else 'unknown'} for noise generation"
    )

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        # Also set metalgpu seed if available
        if hasattr(metalgpu, "set_seed"):
            metalgpu.set_seed(seed)

    return metalgpu.generate_perlin_noise(
        width=width,
        height=height,
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        seed=seed if seed is not None else rng.integers(0, 100000),
    )


def _generate_noise_layer_cupy(height: int, width: int, frequency: float) -> Any:
    """
    Generate a single noise layer using CuPy with FFT-based smoothing.

    Args:
        height: Height of the noise grid
        width: Width of the noise grid
        frequency: Frequency for this octave

    Returns:
        Any: Generated noise layer (cp.ndarray when CuPy is available)
    """
    # Generate base noise
    noise_layer = cp.random.random((height, width)).astype(cp.float32)

    # Apply FFT-based smoothing
    fft = cp.fft.rfft2(noise_layer)

    # Create frequency filter
    freq_x = cp.fft.fftfreq(width)[: width // 2 + 1]
    freq_y = cp.fft.fftfreq(height)

    # Create distance matrix
    dist = cp.sqrt((freq_y[:, cp.newaxis]) ** 2 + (freq_x[cp.newaxis, :]) ** 2)

    # Apply frequency filter
    fft *= cp.exp(-dist * frequency * 10)

    # Apply inverse FFT
    noise_layer = cp.fft.irfft2(fft, s=(height, width))

    # Normalize
    noise_layer = (noise_layer - cp.min(noise_layer)) / (
        cp.max(noise_layer) - cp.min(noise_layer)
    )

    return noise_layer


def _generate_noise_cupy(
    width: int,
    height: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: Optional[int],
) -> np.ndarray:
    """
    Generate Perlin noise using CuPy implementation.

    Args:
        width: Width of the noise grid
        height: Height of the noise grid
        scale: Scale of the noise
        octaves: Number of octaves for the noise
        persistence: Persistence parameter
        lacunarity: Lacunarity parameter
        seed: Random seed

    Returns:
        np.ndarray: Generated noise grid
    """
    if seed is not None:
        cp.random.seed(seed)

    noise = cp.zeros((height, width), dtype=cp.float32)

    # Generate noise with multiple octaves
    max_amplitude = 0
    amplitude = 1.0
    frequency = scale

    for _ in range(octaves):
        # Generate and process noise layer
        noise_layer = _generate_noise_layer_cupy(height, width, frequency)

        # Add to final noise
        noise += amplitude * noise_layer

        # Update parameters for next octave
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize the final noise
    noise /= max_amplitude

    # Transfer back to CPU
    return cp.asnumpy(noise)


def apply_noise_generation_gpu(
    width: int,
    height: int,
    scale: float = 0.1,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: Optional[int] = None,
    backend: str = "auto",
) -> np.ndarray:
    """
    Generate Perlin noise using GPU acceleration if available.

    Args:
        width: Width of the noise grid
        height: Height of the noise grid
        scale: Scale of the noise
        octaves: Number of octaves for the noise
        persistence: Persistence parameter
        lacunarity: Lacunarity parameter
        seed: Random seed
        backend: GPU backend to use ('cuda', 'cupy', 'mps', 'metalgpu', 'auto')

    Returns:
        np.ndarray: Generated noise grid
    """
    # Determine backend to use
    backend = _determine_backend_for_noise_generation(backend)

    # Use CPU implementation if no GPU is available or requested
    if backend == "cpu" or (
        not CUDA_AVAILABLE
        and not CUPY_AVAILABLE
        and not MPS_AVAILABLE
        and not METALGPU_AVAILABLE
    ):
        return _generate_noise_cpu(
            width, height, scale, octaves, persistence, lacunarity, seed
        )

    # CuPy implementation
    if backend == "cupy" and CUPY_AVAILABLE:
        return _generate_noise_cupy(
            width, height, scale, octaves, persistence, lacunarity, seed
        )

    # MPS implementation (for macOS with Apple Silicon or AMD GPUs)
    if backend == "mps" and MPS_AVAILABLE and TORCH_AVAILABLE:
        try:
            return _generate_noise_mps(
                width, height, scale, octaves, persistence, lacunarity, seed
            )
        except Exception as e:
            logging.warning(
                f"MPS noise generation failed: {e}. Falling back to CPU implementation."
            )

    # Metal implementation for macOS using metalgpu for more advanced Metal API usage
    if backend == "metalgpu" and METALGPU_AVAILABLE:
        try:
            return _generate_noise_metalgpu(
                width, height, scale, octaves, persistence, lacunarity, seed
            )
        except Exception as e:
            logging.warning(
                f"Metal GPU noise generation failed: {e}. Falling back to CPU implementation."
            )

    # Fallback to CPU implementation if all GPU methods failed
    return _generate_noise_cpu(
        width, height, scale, octaves, persistence, lacunarity, seed
    )


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
    random_gen = Generator(PCG64(seed if seed is not None else 42))
    return random_gen.random((height, width))


def _initialize_centroids_cpu(
    data: np.ndarray, n_samples: int, n_clusters: int, random_gen: Generator
) -> np.ndarray:
    """
    Initialize centroids randomly for CPU implementation.

    Args:
        data: Input data points
        n_samples: Number of samples in data
        n_clusters: Number of clusters to form
        random_gen: Random number generator

    Returns:
        np.ndarray: Initial centroids
    """
    idx = random_gen.choice(n_samples, n_clusters, replace=False)
    return data[idx].copy()


def _compute_distances_cpu(
    data: np.ndarray, centroids: np.ndarray, n_samples: int, n_clusters: int
) -> np.ndarray:
    """
    Compute distances from each point to each centroid.

    Args:
        data: Input data points
        centroids: Current centroids
        n_samples: Number of samples in data
        n_clusters: Number of clusters

    Returns:
        np.ndarray: Distances matrix
    """
    distances = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)
    return distances


def _update_centroids_cpu(
    data: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    n_features: int,
    n_samples: int,
    random_gen: Generator,
) -> np.ndarray:
    """
    Update centroids based on current assignments.

    Args:
        data: Input data points
        labels: Current cluster assignments
        n_clusters: Number of clusters
        n_features: Number of features in data
        n_samples: Number of samples in data
        random_gen: Random number generator

    Returns:
        np.ndarray: Updated centroids
    """
    new_centroids = np.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        if np.sum(labels == i) > 0:
            new_centroids[i] = np.mean(data[labels == i], axis=0)
        else:
            # If a cluster is empty, reinitialize it
            new_centroids[i] = data[random_gen.choice(n_samples)]
    return new_centroids


def _apply_kmeans_cpu(
    data: np.ndarray,
    n_clusters: int,
    max_iterations: int,
    tolerance: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply K-means clustering using CPU implementation.

    Args:
        data: Input data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        seed: Random seed

    Returns:
        Tuple[np.ndarray, np.ndarray]: (cluster_centers, labels)
    """
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iterations,
            tol=tolerance,
            random_state=seed,
            n_init=10,
        )
        labels = kmeans.fit_predict(data)
        return kmeans.cluster_centers_, labels
    except ImportError:
        logging.warning(
            "scikit-learn not available. Using simple K-means implementation."
        )
        # Simple K-means implementation
        n_samples, n_features = data.shape
        random_gen = Generator(PCG64(seed if seed is not None else 42))

        # Initialize centroids randomly
        centroids = _initialize_centroids_cpu(data, n_samples, n_clusters, random_gen)

        for _ in range(max_iterations):
            # Compute distances and assign points to nearest centroid
            distances = _compute_distances_cpu(data, centroids, n_samples, n_clusters)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = _update_centroids_cpu(
                data, labels, n_clusters, n_features, n_samples, random_gen
            )

            # Check for convergence
            if np.sum((new_centroids - centroids) ** 2) < tolerance:
                break

            centroids = new_centroids

        return centroids, labels


def _apply_kmeans_cupy(
    data: np.ndarray,
    n_clusters: int,
    max_iterations: int,
    tolerance: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply K-means clustering using CuPy implementation.

    Args:
        data: Input data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        seed: Random seed

    Returns:
        Tuple[np.ndarray, np.ndarray]: (cluster_centers, labels)
    """
    n_samples, n_features = data.shape

    # Set random seed
    if seed is not None:
        cp.random.seed(seed)

    # Transfer data to GPU
    d_data = cp.asarray(data)

    # Initialize centroids randomly
    idx = cp.random.choice(n_samples, n_clusters, replace=False)
    centroids = d_data[idx].copy()

    # K-means iterations
    for _ in range(max_iterations):
        # Compute distances to centroids
        distances = cp.zeros((n_samples, n_clusters), dtype=cp.float32)
        for i in range(n_clusters):
            # Vectorized distance calculation
            diff = d_data - centroids[i]
            distances[:, i] = cp.sum(diff * diff, axis=1)

        # Assign points to nearest centroid
        labels = cp.argmin(distances, axis=1)

        # Update centroids
        new_centroids = cp.zeros((n_clusters, n_features), dtype=cp.float32)
        for i in range(n_clusters):
            cluster_points = d_data[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cp.mean(cluster_points, axis=0)
            else:
                # If a cluster is empty, reinitialize it
                new_centroids[i] = d_data[cp.random.choice(n_samples)]

        # Check for convergence
        if cp.sum((new_centroids - centroids) ** 2) < tolerance:
            break

        centroids = new_centroids

    # Transfer results back to CPU
    return cp.asnumpy(centroids), cp.asnumpy(labels)


def _initialize_centroids_cuda(
    data: np.ndarray, n_samples: int, n_clusters: int, random_gen: Generator
) -> Tuple[np.ndarray, Any]:
    """
    Initialize centroids randomly for CUDA implementation.

    Args:
        data: Input data points
        n_samples: Number of samples in data
        n_clusters: Number of clusters to form
        random_gen: Random number generator

    Returns:
        Tuple[np.ndarray, Any]: (CPU centroids, GPU centroids)
    """
    idx = random_gen.choice(n_samples, n_clusters, replace=False)
    centroids = data[idx].copy()
    d_centroids = cuda.to_device(centroids)
    return centroids, d_centroids


def _setup_cuda_environment(
    data: np.ndarray, n_samples: int, n_clusters: int
) -> Tuple[Any, Any, int, int]:
    """
    Set up CUDA environment for K-means.

    Args:
        data: Input data points
        n_samples: Number of samples in data
        n_clusters: Number of clusters

    Returns:
        Tuple: (d_data, d_distances, threadsperblock, blockspergrid)
    """
    # Transfer data to device
    d_data = cuda.to_device(data)

    # Prepare distance matrix
    distances_array = np.zeros((n_samples, n_clusters), dtype=np.float32)
    d_distances = cuda.to_device(distances_array)

    # Configure CUDA grid
    threadsperblock = 256
    blockspergrid = (n_samples + threadsperblock - 1) // threadsperblock

    return d_data, d_distances, threadsperblock, blockspergrid


def _compute_cuda_distances(
    d_data: Any,
    d_centroids: Any,
    d_distances: Any,
    blockspergrid: int,
    threadsperblock: int,
    kernel_func,
) -> np.ndarray:
    """
    Compute distances using CUDA kernel and return labels.

    Args:
        d_data: Device data points
        d_centroids: Device centroids
        d_distances: Device distances matrix
        blockspergrid: CUDA grid size
        threadsperblock: CUDA block size
        kernel_func: CUDA kernel function

    Returns:
        np.ndarray: Labels array
    """
    # Compute distances
    kernel_func[blockspergrid, threadsperblock](d_data, d_centroids, d_distances)

    # Copy distances back to host
    distances = d_distances.copy_to_host()

    # Find nearest centroid for each point
    return np.argmin(distances, axis=1)


def _apply_kmeans_cuda(
    data: np.ndarray,
    n_clusters: int,
    max_iterations: int,
    tolerance: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply K-means clustering using CUDA implementation.

    Args:
        data: Input data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        seed: Random seed

    Returns:
        Tuple[np.ndarray, np.ndarray]: (cluster_centers, labels)
    """
    n_samples, n_features = data.shape

    # Define CUDA kernels for K-means
    @cuda.jit
    def _compute_distances_kernel(data, centroids, distances):
        """Compute distances from each point to each centroid."""
        point_idx = cuda.grid(1)

        if point_idx < data.shape[0]:
            for c in range(centroids.shape[0]):
                dist = 0.0
                for f in range(data.shape[1]):
                    diff = data[point_idx, f] - centroids[c, f]
                    dist += diff * diff
                distances[point_idx, c] = dist

    # Initialize random number generator
    random_gen = Generator(PCG64(seed if seed is not None else 42))

    # Set up CUDA environment
    d_data, d_distances, threadsperblock, blockspergrid = _setup_cuda_environment(
        data, n_samples, n_clusters
    )

    # Initialize centroids
    centroids, d_centroids = _initialize_centroids_cuda(
        data, n_samples, n_clusters, random_gen
    )

    # K-means iterations
    for _ in range(max_iterations):
        # Compute distances and get labels
        labels = _compute_cuda_distances(
            d_data,
            d_centroids,
            d_distances,
            blockspergrid,
            threadsperblock,
            _compute_distances_kernel,
        )

        # Update centroids
        new_centroids = _update_centroids_cpu(
            data, labels, n_clusters, n_features, n_samples, random_gen
        )

        # Check for convergence
        if np.sum((new_centroids - centroids) ** 2) < tolerance:
            break

        centroids = new_centroids
        d_centroids = cuda.to_device(centroids)

    return centroids, labels


def _initialize_mps_environment(seed: Optional[int]) -> torch.device:
    """
    Initialize MPS environment and set random seed.

    Args:
        seed: Random seed

    Returns:
        torch.device: MPS device
    """
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create MPS device
    return torch.device("mps")


def _initialize_mps_centroids(
    data: torch.Tensor, n_samples: int, n_clusters: int
) -> torch.Tensor:
    """
    Initialize centroids randomly for MPS implementation.

    Args:
        data: Input data points on MPS device
        n_samples: Number of samples in data
        n_clusters: Number of clusters to form

    Returns:
        torch.Tensor: Initial centroids
    """
    idx = torch.randperm(n_samples, device=data.device)[:n_clusters]
    return data[idx].clone()


def _compute_mps_distances(
    data: torch.Tensor, centroids: torch.Tensor, n_samples: int, n_clusters: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distances from each point to each centroid and assign labels.

    Args:
        data: Input data points on MPS device
        centroids: Current centroids
        n_samples: Number of samples
        n_clusters: Number of clusters

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (distances, labels)
    """
    # Compute distances to centroids
    distances = torch.zeros(
        (n_samples, n_clusters), dtype=torch.float32, device=data.device
    )
    for i in range(n_clusters):
        # Vectorized distance calculation
        diff = data - centroids[i]
        distances[:, i] = torch.sum(diff * diff, dim=1)

    # Assign points to nearest centroid
    labels = torch.argmin(distances, dim=1)

    return distances, labels


def _update_mps_centroids(
    data: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
    n_features: int,
    n_samples: int,
) -> torch.Tensor:
    """
    Update centroids based on assigned labels.

    Args:
        data: Input data points on MPS device
        labels: Cluster assignments
        n_clusters: Number of clusters
        n_features: Number of features
        n_samples: Number of samples

    Returns:
        torch.Tensor: Updated centroids
    """
    new_centroids = torch.zeros(
        (n_clusters, n_features), dtype=torch.float32, device=data.device
    )
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = torch.mean(cluster_points, dim=0)
        else:
            # If a cluster is empty, reinitialize it
            new_centroids[i] = data[
                torch.randint(0, n_samples, (1,), device=data.device)
            ]

    return new_centroids


def _apply_kmeans_mps(
    data: np.ndarray,
    n_clusters: int,
    max_iterations: int,
    tolerance: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply K-means clustering using MPS (Metal Performance Shaders) implementation.

    Args:
        data: Input data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        seed: Random seed

    Returns:
        Tuple[np.ndarray, np.ndarray]: (cluster_centers, labels)
    """
    n_samples, n_features = data.shape

    try:
        # Initialize MPS environment
        mps_device = _initialize_mps_environment(seed)

        # Create MPS tensor
        mps_data = torch.tensor(data, dtype=torch.float32, device=mps_device)

        # Initialize centroids randomly
        centroids = _initialize_mps_centroids(mps_data, n_samples, n_clusters)

        # K-means iterations
        for _ in range(max_iterations):
            # Compute distances and get labels
            _, labels = _compute_mps_distances(
                mps_data, centroids, n_samples, n_clusters
            )

            # Update centroids
            new_centroids = _update_mps_centroids(
                mps_data, labels, n_clusters, n_features, n_samples
            )

            # Check for convergence
            if torch.sum((new_centroids - centroids) ** 2) < tolerance:
                break

            centroids = new_centroids

        # Transfer results back to CPU
        centroids = to_cpu(centroids)
        labels = to_cpu(labels)

        return centroids, labels
    except Exception as e:
        logging.warning(f"Error in MPS implementation: {e}")
        return np.zeros((n_clusters, n_features)), np.full(n_samples, -1)


def _initialize_metalgpu_environment(seed: Optional[int]) -> Tuple[Any, torch.device]:
    """
    Initialize MetalGPU environment and set random seed.

    Args:
        seed: Random seed

    Returns:
        Tuple[Any, torch.device]: (metalgpu module, MPS device)
    """
    # Import metalgpu here to avoid lint errors when it's not used
    import metalgpu

    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        # Also set metalgpu seed if available
        if hasattr(metalgpu, "set_seed"):
            metalgpu.set_seed(seed)

    # Create MPS device
    mps_device = torch.device("mps")

    return metalgpu, mps_device


def _apply_kmeans_metalgpu(
    data: np.ndarray,
    n_clusters: int,
    max_iterations: int,
    tolerance: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply K-means clustering using MetalGPU implementation.

    Args:
        data: Input data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        seed: Random seed

    Returns:
        Tuple[np.ndarray, np.ndarray]: (cluster_centers, labels)
    """
    n_samples, n_features = data.shape

    try:
        # Initialize MetalGPU environment
        _, mps_device = _initialize_metalgpu_environment(seed)

        # Transfer data to MPS device
        d_data = torch.tensor(data, dtype=torch.float32, device=mps_device)

        # Initialize centroids randomly
        centroids = _initialize_mps_centroids(d_data, n_samples, n_clusters)

        # K-means iterations
        for _ in range(max_iterations):
            # Compute distances and get labels
            _, labels = _compute_mps_distances(d_data, centroids, n_samples, n_clusters)

            # Update centroids
            new_centroids = _update_mps_centroids(
                d_data, labels, n_clusters, n_features, n_samples
            )

            # Check for convergence
            if torch.sum((new_centroids - centroids) ** 2) < tolerance:
                break

            centroids = new_centroids

        # Transfer results back to CPU
        centroids = to_cpu(centroids)
        labels = to_cpu(labels)

        return centroids, labels
    except Exception as e:
        logging.warning(f"Error in MetalGPU implementation: {e}")
        return np.zeros((n_clusters, n_features)), np.full(n_samples, -1)


def _select_kmeans_backend(backend: str) -> str:
    """
    Select the appropriate backend for K-means clustering.

    Args:
        backend: Requested backend ('cuda', 'cupy', 'mps', 'metalgpu', 'auto')

    Returns:
        str: Selected backend
    """
    if backend == "auto":
        if CUPY_AVAILABLE:
            backend = "cupy"
        elif CUDA_AVAILABLE:
            backend = "cuda"
        elif MPS_AVAILABLE:
            backend = "mps"
        elif METALGPU_AVAILABLE:
            backend = "metalgpu"
        else:
            backend = "cpu"

    return backend


def apply_kmeans_clustering_gpu(
    data: np.ndarray,
    n_clusters: int = 5,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    seed: Optional[int] = None,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply K-means clustering using GPU acceleration if available.

    Args:
        data: Input data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        seed: Random seed
        backend: GPU backend to use ('cuda', 'cupy', 'mps', 'metalgpu', 'auto')

    Returns:
        Tuple[np.ndarray, np.ndarray]: (cluster_centers, labels)
    """
    # Select the appropriate backend
    backend = _select_kmeans_backend(backend)

    # Use the appropriate implementation based on the selected backend
    if backend == "cpu" or (
        not CUDA_AVAILABLE
        and not CUPY_AVAILABLE
        and not MPS_AVAILABLE
        and not METALGPU_AVAILABLE
    ):
        return _apply_kmeans_cpu(data, n_clusters, max_iterations, tolerance, seed)

    # CuPy implementation
    if backend == "cupy" and CUPY_AVAILABLE:
        return _apply_kmeans_cupy(data, n_clusters, max_iterations, tolerance, seed)

    # CUDA implementation using Numba
    if backend == "cuda" and CUDA_AVAILABLE and NUMBA_AVAILABLE:
        return _apply_kmeans_cuda(data, n_clusters, max_iterations, tolerance, seed)

    # MPS implementation for macOS
    if backend == "mps" and MPS_AVAILABLE:
        return _apply_kmeans_mps(data, n_clusters, max_iterations, tolerance, seed)

    # Metal implementation for macOS
    if backend == "metalgpu" and METALGPU_AVAILABLE:
        return _apply_kmeans_metalgpu(data, n_clusters, max_iterations, tolerance, seed)

    # Fallback to CPU implementation if requested backend is not available
    logging.warning(
        f"Requested backend '{backend}' not available. Falling back to CPU implementation."
    )
    return _apply_kmeans_cpu(data, n_clusters, max_iterations, tolerance, seed)


# Define constant for duplicate warning message
SCIKIT_DBSCAN_WARNING = (
    "scikit-learn not available. DBSCAN requires scikit-learn or GPU."
)


def _apply_dbscan_cpu(
    data: np.ndarray, eps: float, min_samples: int, n_samples: int
) -> np.ndarray:
    """Apply DBSCAN clustering using scikit-learn CPU implementation.

    Args:
        data: Input data points
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN
        n_samples: Number of samples

    Returns:
        np.ndarray: Cluster labels
    """
    try:
        from sklearn.cluster import DBSCAN

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(data)
    except ImportError:
        logging.warning(SCIKIT_DBSCAN_WARNING)
        # Return all points as noise if scikit-learn is not available
        return np.full(n_samples, -1)


def _apply_dbscan_mps(data: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Apply DBSCAN clustering using MPS (Metal Performance Shaders).

    Args:
        data: Input data points
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN

    Returns:
        np.ndarray: Cluster labels
    """
    try:
        # Transfer data to MPS device
        d_data = torch.tensor(data, dtype=torch.float32, device=mps_device)
        data_shape = data.shape
        sample_count, n_features = data_shape

        # Compute pairwise distances - using n_features in the calculation
        distances = torch.zeros(
            (sample_count, sample_count), dtype=torch.float32, device=mps_device
        )

        # Using batch matrix operations to utilize n_features dimension more efficiently
        for i in range(sample_count):
            # For each sample, compute distance to all other samples
            diff = d_data - d_data[i].view(1, n_features)
            # Square differences and sum across feature dimension
            distances[i] = torch.sqrt(torch.sum(diff * diff, dim=1))

        # Find neighbors
        neighbors = distances <= eps

        # Count neighbors
        neighbor_counts = torch.sum(neighbors, dim=1)

        # Find core points
        core_points = neighbor_counts >= min_samples

        # Initialize labels
        labels = torch.full((sample_count,), -1, dtype=torch.int32, device=mps_device)

        # Cluster ID
        cluster_id = 0

        # Process each point
        for i in range(sample_count):
            # Skip non-core points or already labeled points
            if not core_points[i] or labels[i] != -1:
                continue

            # Start a new cluster
            labels[i] = cluster_id

            # Find neighbors to process
            to_process = torch.where(neighbors[i] & (labels == -1))[0]

            # Process neighbors
            while len(to_process) > 0:
                j = to_process[0]
                to_process = to_process[1:]

                labels[j] = cluster_id

                # If core point, add its neighbors
                if core_points[j]:
                    new_neighbors = torch.where(neighbors[j] & (labels == -1))[0]
                    to_process = torch.cat([to_process, new_neighbors])

            cluster_id += 1

        return to_cpu(labels)

    except Exception as e:
        logging.warning(f"MPS DBSCAN implementation failed: {e}. Falling back to CPU.")
        return None


def _apply_dbscan_metalgpu(
    data: np.ndarray, eps: float, min_samples: int, _: int
) -> np.ndarray:
    """Apply DBSCAN clustering using the metalgpu library.

    Args:
        data: Input data points
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN
        _: Unused parameter (number of samples) to maintain consistent interface

    Returns:
        np.ndarray: Cluster labels
    """
    try:
        # Import metalgpu here to avoid lint errors when it's not used
        import metalgpu

        # Log that we're using metalgpu for DBSCAN
        version_str = "unknown"
        if hasattr(metalgpu, "__version__"):
            version_str = metalgpu.__version__

        logging.info(f"Using metalgpu version {version_str} for DBSCAN clustering")

        # Use metalgpu for DBSCAN clustering
        return metalgpu.dbscan_clustering(data=data, eps=eps, min_samples=min_samples)

    except Exception as e:
        logging.warning(
            f"Metal GPU DBSCAN implementation failed: {e}. Falling back to CPU."
        )
        return None


def _find_and_process_cluster_cupy(
    index: int, neighbors: Any, core_points: Any, labels: Any, cluster_id: int
) -> None:
    """Find and process a cluster from a core point using CuPy.

    Args:
        index: Index of the core point to start from
        neighbors: Matrix of neighbor relationships
        core_points: Array indicating core points
        labels: Array of cluster labels to update
        cluster_id: Current cluster ID to assign
    """
    # Start a new cluster
    labels[index] = cluster_id

    # Find all points reachable from this core point
    stack = [index]
    while stack:
        current = stack.pop()

        # Get neighbors
        current_neighbors = cp.where(neighbors[current])[0]

        # Process neighbors
        for neighbor in current_neighbors:
            # If not yet assigned to a cluster
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id

                # If it's a core point, add to stack for further expansion
                if core_points[neighbor]:
                    stack.append(neighbor)


def _compute_distances_cupy(d_data: Any, n_samples: int) -> Any:
    """Compute pairwise distances between points using CuPy.

    Args:
        d_data: CuPy array of data points
        n_samples: Number of samples

    Returns:
        CuPy array of pairwise distances
    """
    distances = cp.zeros((n_samples, n_samples), dtype=cp.float32)
    for i in range(n_samples):
        diff = d_data - d_data[i]
        distances[i] = cp.sum(diff * diff, axis=1)
    return distances


def _apply_dbscan_cupy(
    data: np.ndarray, eps: float, min_samples: int, n_samples: int
) -> np.ndarray:
    """Apply DBSCAN clustering using CuPy.

    Args:
        data: Input data points
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN
        n_samples: Number of samples

    Returns:
        np.ndarray: Cluster labels
    """
    # Transfer data to GPU
    d_data = cp.asarray(data)

    # Compute pairwise distances
    distances = _compute_distances_cupy(d_data, n_samples)

    # Find neighbors
    neighbors = distances <= eps**2

    # Count neighbors
    n_neighbors = cp.sum(neighbors, axis=1)

    # Find core points
    core_points = n_neighbors >= min_samples

    # Initialize labels
    labels = cp.full(n_samples, -1, dtype=cp.int32)

    # Cluster ID counter
    cluster_id = 0

    # Perform clustering
    for i in range(n_samples):
        # Skip non-core points or already labeled points
        if not core_points[i] or labels[i] != -1:
            continue

        # Process cluster starting from this core point
        _find_and_process_cluster_cupy(i, neighbors, core_points, labels, cluster_id)

        # Move to next cluster
        cluster_id += 1

    # Transfer results back to CPU
    return cp.asnumpy(labels)


def _setup_cuda_distance_computation(data: np.ndarray, n_samples: int) -> tuple:
    """Set up CUDA environment for distance computation.

    Args:
        data: Input data points
        n_samples: Number of samples

    Returns:
        tuple: (d_data, d_distances, threadsperblock, blockspergrid)
    """
    # Transfer data to device
    d_data = cuda.to_device(data)

    # Prepare distance matrix
    distances = np.zeros((n_samples, n_samples), dtype=np.float32)
    d_distances = cuda.to_device(distances)

    # Configure CUDA grid for distance computation
    threadsperblock = (16, 16)
    blockspergrid_x = (n_samples + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n_samples + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    return d_data, d_distances, threadsperblock, blockspergrid


def _find_and_process_cluster_cuda(
    index: int,
    neighbors: np.ndarray,
    core_points: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
) -> None:
    """Find and process a cluster from a core point using CUDA.

    Args:
        index: Index of the core point to start from
        neighbors: Matrix of neighbor relationships
        core_points: Array indicating core points
        labels: Array of cluster labels to update
        cluster_id: Current cluster ID to assign
    """
    # Start a new cluster
    labels[index] = cluster_id

    # Find all points reachable from this core point
    stack = [index]
    while stack:
        current = stack.pop()

        # Get neighbors
        current_neighbors = np.nonzero(neighbors[current])[0]

        # Process neighbors
        for neighbor in current_neighbors:
            # If not yet assigned to a cluster
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id

                # If it's a core point, add to stack for further expansion
                if core_points[neighbor]:
                    stack.append(neighbor)


def _apply_dbscan_cuda(
    data: np.ndarray, eps: float, min_samples: int, n_samples: int
) -> np.ndarray:
    """Apply DBSCAN clustering using CUDA.

    Args:
        data: Input data points
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN
        n_samples: Number of samples

    Returns:
        np.ndarray: Cluster labels
    """

    # Define CUDA kernels for DBSCAN
    @cuda.jit
    def _compute_distance_matrix_kernel(data, distances):
        """Compute pairwise distances between all points."""
        i, j = cuda.grid(2)

        if i < data.shape[0] and j < data.shape[0]:
            dist = 0.0
            for f in range(data.shape[1]):  # Iterate over n_features
                diff = data[i, f] - data[j, f]
                dist += diff * diff
            distances[i, j] = dist

    # Set up CUDA environment
    d_data, d_distances, threadsperblock, blockspergrid = (
        _setup_cuda_distance_computation(data, n_samples)
    )

    # Compute distances
    _compute_distance_matrix_kernel[blockspergrid, threadsperblock](d_data, d_distances)

    # Copy distances back to host
    distances = d_distances.copy_to_host()

    # Find neighbors and core points
    neighbors = distances <= eps**2
    n_neighbors = np.sum(neighbors, axis=1)
    core_points = n_neighbors >= min_samples

    # Initialize labels
    labels = np.full(n_samples, -1, dtype=np.int32)
    cluster_id = 0

    # Perform clustering
    for i in range(n_samples):
        # Skip non-core points or already labeled points
        if not core_points[i] or labels[i] != -1:
            continue

        # Process cluster starting from this core point
        _find_and_process_cluster_cuda(i, neighbors, core_points, labels, cluster_id)

        # Move to next cluster
        cluster_id += 1

    return labels


def _select_dbscan_backend(backend: str) -> str:
    """Select appropriate backend for DBSCAN clustering based on availability.

    Args:
        backend: Requested backend ('cuda', 'cupy', 'mps', 'metalgpu', 'auto')

    Returns:
        str: Selected backend name
    """
    if backend != "auto":
        return backend

    # Auto-select based on availability
    if CUPY_AVAILABLE:
        return "cupy"
    elif CUDA_AVAILABLE:
        return "cuda"
    elif MPS_AVAILABLE:
        return "mps"
    elif METALGPU_AVAILABLE:
        return "metalgpu"
    else:
        return "cpu"


def _is_gpu_available() -> bool:
    """Check if any GPU backend is available for DBSCAN.

    Returns:
        bool: True if at least one GPU backend is available
    """
    return any([CUDA_AVAILABLE, CUPY_AVAILABLE, MPS_AVAILABLE, METALGPU_AVAILABLE])


def _try_gpu_backend(
    backend: str, data: np.ndarray, eps: float, min_samples: int, n_samples: int
) -> Optional[np.ndarray]:
    """Try to run DBSCAN using the specified GPU backend.

    Args:
        backend: Backend to use
        data: Input data points
        eps: Epsilon parameter
        min_samples: Minimum samples parameter
        n_samples: Number of samples

    Returns:
        Optional[np.ndarray]: Cluster labels if successful, None if failed
    """
    # Select the appropriate implementation based on backend
    if backend == "mps" and MPS_AVAILABLE:
        return _apply_dbscan_mps(data, eps, min_samples)

    if backend == "metalgpu" and METALGPU_AVAILABLE:
        return _apply_dbscan_metalgpu(data, eps, min_samples, n_samples)

    if backend == "cupy" and CUPY_AVAILABLE:
        return _apply_dbscan_cupy(data, eps, min_samples, n_samples)

    if backend == "cuda" and CUDA_AVAILABLE and NUMBA_AVAILABLE:
        return _apply_dbscan_cuda(data, eps, min_samples, n_samples)

    return None


def apply_dbscan_clustering_gpu(
    data: np.ndarray, eps: float = 0.5, min_samples: int = 5, backend: str = "auto"
) -> np.ndarray:
    """
    Apply DBSCAN clustering using GPU acceleration if available.

    Args:
        data: Input data points, shape (n_samples, n_features)
        eps: The maximum distance between two samples for one to be considered
             as in the neighborhood of the other
        min_samples: The number of samples in a neighborhood for a point
                     to be considered as a core point
        backend: GPU backend to use ('cuda', 'cupy', 'mps', 'metalgpu', 'auto')

    Returns:
        np.ndarray: Cluster labels for each point. Noisy samples are labeled as -1.
    """
    n_samples = data.shape[0]

    # Select appropriate backend
    selected_backend = _select_dbscan_backend(backend)

    # Use CPU if requested or if no GPU available
    if selected_backend == "cpu" or not _is_gpu_available():
        return _apply_dbscan_cpu(data, eps, min_samples, n_samples)

    # Try the selected GPU backend
    result = _try_gpu_backend(selected_backend, data, eps, min_samples, n_samples)
    if result is not None:
        return result

    # Fallback to CPU implementation
    return _apply_dbscan_cpu(data, eps, min_samples, n_samples)
