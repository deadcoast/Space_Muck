#!/usr/bin/env python3
"""
GPU-accelerated Value Generator utilities for Space Muck.

This module provides GPU-accelerated implementations of value generation functions
that can be used by different generator classes to improve performance.
"""

# Standard library imports
import logging
import math

import numpy as np

# Third-party library imports
from numpy.random import PCG64, Generator

# Local application imports
from utils.value_generator import add_value_clusters, generate_value_distribution

# Standard library imports

# Typing imports removed (unused)

# Third-party library imports

# Local imports
# GPU utilities imports removed (unused)

# Initialize random number generator with a fixed seed for reproducibility
rng = Generator(PCG64(42))

# Optional dependencies
try:
    from numba import cuda

    NUMBA_AVAILABLE = True
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    NUMBA_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def _select_backend(backend: str) -> str:
    """
    Select the appropriate backend based on availability.

    Args:
        backend: Requested backend ('cuda', 'cupy', 'auto', 'cpu')

    Returns:
        str: Selected backend based on availability
    """
    if backend == "auto":
        if CUDA_AVAILABLE:
            return "cuda"
        elif CUPY_AVAILABLE:
            return "cupy"
        else:
            return "cpu"
    return backend


def _apply_value_distribution_cupy(
    grid: np.ndarray,
    base_grid: np.ndarray,
    value_mean: float,
    value_stddev: float,
    min_value: int,
) -> np.ndarray:
    """
    Apply value distribution using CuPy backend.

    Args:
        grid: Binary grid indicating entity presence
        base_grid: Noise grid for value distribution
        value_mean: Mean value for resources
        value_stddev: Standard deviation for resource values
        min_value: Minimum value for non-zero cells

    Returns:
        np.ndarray: Grid with resource values
    """
    # Transfer data to GPU
    grid_gpu = cp.asarray(grid)
    base_grid_gpu = cp.asarray(base_grid)

    # Scale noise to desired mean and standard deviation
    value_noise = base_grid_gpu * value_stddev + value_mean

    # Apply values only to non-zero cells
    value_grid = grid_gpu * value_noise.astype(cp.int32)

    # Ensure minimum value for non-zero cells
    mask = value_grid > 0
    value_grid[mask] = cp.maximum(value_grid[mask], min_value)

    # Transfer back to CPU
    return cp.asnumpy(value_grid)


def _apply_value_distribution_cuda(
    grid: np.ndarray,
    base_grid: np.ndarray,
    value_mean: float,
    value_stddev: float,
    min_value: int,
) -> np.ndarray:
    """
    Apply value distribution using CUDA backend.

    Args:
        grid: Binary grid indicating entity presence
        base_grid: Noise grid for value distribution
        value_mean: Mean value for resources
        value_stddev: Standard deviation for resource values
        min_value: Minimum value for non-zero cells

    Returns:
        np.ndarray: Grid with resource values
    """

    @cuda.jit
    def value_distribution_kernel(
        grid, base_grid, value_grid, value_mean, value_stddev, min_value
    ):
        x, y = cuda.grid(2)
        if x < grid.shape[1] and y < grid.shape[0] and grid[y, x] > 0:
            value = base_grid[y, x] * value_stddev + value_mean
            value_grid[y, x] = max(int(value), min_value)

    # Prepare data
    value_grid = np.zeros_like(grid, dtype=np.int32)

    # Configure CUDA grid
    threadsperblock = (16, 16)
    blockspergrid_x = (grid.shape[1] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (grid.shape[0] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Execute kernel
    value_distribution_kernel[blockspergrid, threadsperblock](
        grid, base_grid, value_grid, value_mean, value_stddev, min_value
    )

    return value_grid


def generate_value_distribution_gpu(
    grid: np.ndarray,
    base_grid: np.ndarray,
    value_mean: float = 5.0,
    value_stddev: float = 2.0,
    min_value: int = 1,
    backend: str = "auto",
) -> np.ndarray:
    """
    Generate resource values for a grid based on a noise pattern using GPU acceleration.

    Args:
        grid: Binary grid indicating entity presence
        base_grid: Noise grid for value distribution
        value_mean: Mean value for resources
        value_stddev: Standard deviation for resource values
        min_value: Minimum value for non-zero cells
        backend: GPU backend to use ('cuda', 'cupy', 'auto')

    Returns:
        np.ndarray: Grid with resource values
    """
    # Select backend based on availability
    backend = _select_backend(backend)

    # Use CPU implementation if no GPU is available or requested
    if backend == "cpu" or (not CUDA_AVAILABLE and not CUPY_AVAILABLE):
        return generate_value_distribution(
            grid, base_grid, value_mean, value_stddev, min_value
        )

    # GPU implementation
    try:
        if backend == "cupy" and CUPY_AVAILABLE:
            return _apply_value_distribution_cupy(
                grid, base_grid, value_mean, value_stddev, min_value
            )
        elif backend == "cuda" and CUDA_AVAILABLE:
            return _apply_value_distribution_cuda(
                grid, base_grid, value_mean, value_stddev, min_value
            )

    except Exception as e:
        logging.warning(
            f"GPU value distribution failed: {str(e)}. Falling back to CPU implementation."
        )
        return generate_value_distribution(
            grid, base_grid, value_mean, value_stddev, min_value
        )


def _get_cluster_centers(value_grid: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Find valid cluster centers from non-zero cells in the grid.

    Args:
        value_grid: Grid with entity values
        num_clusters: Number of high-value clusters to create

    Returns:
        np.ndarray: Selected cluster centers or empty array if none available
    """
    # Find cells with entities
    entity_cells = np.argwhere(value_grid > 0)

    if len(entity_cells) == 0:
        return np.array([])

    # Adjust number of clusters if needed
    if len(entity_cells) < num_clusters:
        num_clusters = len(entity_cells)

    # Select random cluster centers
    center_indices = rng.choice(len(entity_cells), size=num_clusters, replace=False)
    return entity_cells[center_indices]


def _get_subgrid_boundaries(
    center_y: int, center_x: int, radius: int, height: int, width: int
):
    """
    Calculate the boundaries of a subgrid around a center point.

    Args:
        center_y: Y-coordinate of center
        center_x: X-coordinate of center
        radius: Radius around center
        height: Grid height
        width: Grid width

    Returns:
        tuple: (y_min, y_max, x_min, x_max) boundaries
    """
    y_min = max(0, center_y - radius)
    y_max = min(height, center_y + radius + 1)
    x_min = max(0, center_x - radius)
    x_max = min(width, center_x + radius + 1)

    return y_min, y_max, x_min, x_max


def _apply_value_clusters_cupy(
    value_grid: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_radius: int,
    cluster_value_multiplier: float,
) -> np.ndarray:
    """
    Apply value clusters using the CuPy backend.

    Args:
        value_grid: Grid with entity values
        cluster_centers: Centers of clusters to create
        cluster_radius: Radius of each cluster
        cluster_value_multiplier: Multiplier for values in clusters

    Returns:
        np.ndarray: Grid with value clusters added
    """
    height, width = value_grid.shape

    # Transfer data to GPU
    value_grid_gpu = cp.asarray(value_grid)
    result_grid_gpu = cp.copy(value_grid_gpu)

    # Process each cluster
    for center_y, center_x in cluster_centers:
        # Get subgrid boundaries
        y_min, y_max, x_min, x_max = _get_subgrid_boundaries(
            center_y, center_x, cluster_radius, height, width
        )

        # Create coordinate grids for vectorized distance calculation
        y_coords, x_coords = cp.mgrid[y_min:y_max, x_min:x_max]

        # Calculate distances from center
        distances = cp.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

        # Create mask for cells within radius and with values
        mask = (distances <= cluster_radius) & (
            value_grid_gpu[y_min:y_max, x_min:x_max] > 0
        )

        # Calculate falloff and multipliers
        falloff = 1.0 - (distances / cluster_radius)
        multipliers = 1.0 + (cluster_value_multiplier - 1.0) * falloff

        # Apply multipliers to masked cells
        result_grid_gpu[y_min:y_max, x_min:x_max][mask] = (
            value_grid_gpu[y_min:y_max, x_min:x_max][mask] * multipliers[mask]
        ).astype(cp.int32)

    # Transfer back to CPU
    return cp.asnumpy(result_grid_gpu)


def _apply_value_clusters_cuda(
    value_grid: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_radius: int,
    cluster_value_multiplier: float,
) -> np.ndarray:
    """
    Apply value clusters using the CUDA backend.

    Args:
        value_grid: Grid with entity values
        cluster_centers: Centers of clusters to create
        cluster_radius: Radius of each cluster
        cluster_value_multiplier: Multiplier for values in clusters

    Returns:
        np.ndarray: Grid with value clusters added
    """
    height, width = value_grid.shape

    @cuda.jit
    def apply_cluster_kernel(
        value_grid, result_grid, centers, radius, multiplier, width, height
    ):
        x, y = cuda.grid(2)

        if x < width and y < height and value_grid[y, x] > 0:
            # Check against all cluster centers
            for i in range(centers.shape[0]):
                center_y = centers[i, 0]
                center_x = centers[i, 1]

                # Calculate distance
                distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                # Apply multiplier with falloff if within radius
                if distance <= radius:
                    falloff = 1.0 - (distance / radius)
                    current_multiplier = 1.0 + (multiplier - 1.0) * falloff
                    result_grid[y, x] = int(value_grid[y, x] * current_multiplier)

    # Prepare data
    result_grid = value_grid.copy()

    # Configure CUDA grid
    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Execute kernel
    apply_cluster_kernel[blockspergrid, threadsperblock](
        value_grid,
        result_grid,
        cluster_centers,
        cluster_radius,
        cluster_value_multiplier,
        width,
        height,
    )

    return result_grid


def add_value_clusters_gpu(
    value_grid: np.ndarray,
    num_clusters: int = 5,
    cluster_radius: int = 10,
    cluster_value_multiplier: float = 2.0,
    backend: str = "auto",
) -> np.ndarray:
    """
    Add value clusters to a grid using GPU acceleration.

    Args:
        value_grid: Grid with entity values
        num_clusters: Number of high-value clusters to create
        cluster_radius: Radius of each cluster
        cluster_value_multiplier: Multiplier for values in clusters
        backend: GPU backend to use ('cuda', 'cupy', 'auto')

    Returns:
        np.ndarray: Grid with value clusters added
    """
    # Select backend based on availability
    backend = _select_backend(backend)

    # Use CPU implementation if no GPU is available or requested
    if backend == "cpu" or (not CUDA_AVAILABLE and not CUPY_AVAILABLE):
        return add_value_clusters(
            value_grid=value_grid,
            binary_grid=None,
            num_clusters=num_clusters,
            cluster_radius=cluster_radius,
            cluster_value_multiplier=cluster_value_multiplier,
        )

    # Get cluster centers
    cluster_centers = _get_cluster_centers(value_grid, num_clusters)

    # If no valid centers found, return a copy of the original grid
    if len(cluster_centers) == 0:
        return value_grid.copy()

    # GPU implementation
    try:
        if backend == "cupy" and CUPY_AVAILABLE:
            return _apply_value_clusters_cupy(
                value_grid, cluster_centers, cluster_radius, cluster_value_multiplier
            )
        elif backend == "cuda" and CUDA_AVAILABLE:
            return _apply_value_clusters_cuda(
                value_grid, cluster_centers, cluster_radius, cluster_value_multiplier
            )

    except Exception as e:
        logging.warning(
            f"GPU value clustering failed: {str(e)}. Falling back to CPU implementation."
        )
        return add_value_clusters(
            value_grid,
            num_clusters,
            cluster_radius,
            cluster_value_multiplier=cluster_value_multiplier,
        )
