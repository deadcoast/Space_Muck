#!/usr/bin/env python3
"""
GPU-accelerated Value Generator utilities for Space Muck.

This module provides GPU-accelerated implementations of value generation functions
that can be used by different generator classes to improve performance.
"""

# Standard library imports
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

# Third-party library imports
import numpy as np

# Local imports
from src.utils.gpu_utils import (
    is_gpu_available,
    get_available_backends,
    to_gpu,
    to_cpu,
)
from src.utils.value_generator import (
    generate_value_distribution,
    add_value_clusters,
    generate_rare_resource_distribution,
)

# Optional dependencies
try:
    import numba
    from numba import cuda, float32, int32

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
    # Choose backend
    if backend == "auto":
        if CUDA_AVAILABLE:
            backend = "cuda"
        elif CUPY_AVAILABLE:
            backend = "cupy"
        else:
            backend = "cpu"

    # Use CPU implementation if no GPU is available or requested
    if backend == "cpu" or (not CUDA_AVAILABLE and not CUPY_AVAILABLE):
        return generate_value_distribution(
            grid, base_grid, value_mean, value_stddev, min_value
        )

    # GPU implementation
    try:
        if backend == "cupy" and CUPY_AVAILABLE:
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

        elif backend == "cuda" and CUDA_AVAILABLE:
            # For CUDA, we'll implement a kernel
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
            blockspergrid_x = (
                grid.shape[1] + threadsperblock[0] - 1
            ) // threadsperblock[0]
            blockspergrid_y = (
                grid.shape[0] + threadsperblock[1] - 1
            ) // threadsperblock[1]
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            # Execute kernel
            value_distribution_kernel[blockspergrid, threadsperblock](
                grid, base_grid, value_grid, value_mean, value_stddev, min_value
            )

            return value_grid

    except Exception as e:
        logging.warning(
            f"GPU value distribution failed: {str(e)}. Falling back to CPU implementation."
        )
        return generate_value_distribution(
            grid, base_grid, value_mean, value_stddev, min_value
        )


def add_value_clusters_gpu(
    value_grid: np.ndarray,
    num_clusters: int = 5,
    cluster_radius: int = 10,
    value_multiplier: float = 2.0,
    backend: str = "auto",
) -> np.ndarray:
    """
    Add value clusters to a grid using GPU acceleration.

    Args:
        value_grid: Grid with entity values
        num_clusters: Number of high-value clusters to create
        cluster_radius: Radius of each cluster
        value_multiplier: Multiplier for values in clusters
        backend: GPU backend to use ('cuda', 'cupy', 'auto')

    Returns:
        np.ndarray: Grid with value clusters added
    """
    # Choose backend
    if backend == "auto":
        if CUDA_AVAILABLE:
            backend = "cuda"
        elif CUPY_AVAILABLE:
            backend = "cupy"
        else:
            backend = "cpu"

    # Use CPU implementation if no GPU is available or requested
    if backend == "cpu" or (not CUDA_AVAILABLE and not CUPY_AVAILABLE):
        return add_value_clusters(
            value_grid, num_clusters, cluster_radius, value_multiplier
        )

    height, width = value_grid.shape

    # Find cells with entities
    entity_cells = np.argwhere(value_grid > 0)

    if len(entity_cells) == 0:
        return value_grid.copy()

    # Select random cluster centers
    if len(entity_cells) < num_clusters:
        num_clusters = len(entity_cells)

    center_indices = np.random.choice(
        len(entity_cells), size=num_clusters, replace=False
    )
    cluster_centers = entity_cells[center_indices]

    # GPU implementation
    try:
        if backend == "cupy" and CUPY_AVAILABLE:
            # Transfer data to GPU
            value_grid_gpu = cp.asarray(value_grid)
            result_grid_gpu = cp.copy(value_grid_gpu)

            # Process each cluster
            for center_y, center_x in cluster_centers:
                # Create coordinate grids for vectorized distance calculation
                y_coords, x_coords = cp.mgrid[
                    max(0, center_y - cluster_radius) : min(
                        height, center_y + cluster_radius + 1
                    ),
                    max(0, center_x - cluster_radius) : min(
                        width, center_x + cluster_radius + 1
                    ),
                ]

                # Calculate distances from center
                distances = cp.sqrt(
                    (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
                )

                # Create mask for cells within radius and with values
                mask = (distances <= cluster_radius) & (
                    value_grid_gpu[
                        max(0, center_y - cluster_radius) : min(
                            height, center_y + cluster_radius + 1
                        ),
                        max(0, center_x - cluster_radius) : min(
                            width, center_x + cluster_radius + 1
                        ),
                    ]
                    > 0
                )

                # Calculate falloff and multipliers
                falloff = 1.0 - (distances / cluster_radius)
                multipliers = 1.0 + (value_multiplier - 1.0) * falloff

                # Apply multipliers to masked cells
                result_grid_gpu[
                    max(0, center_y - cluster_radius) : min(
                        height, center_y + cluster_radius + 1
                    ),
                    max(0, center_x - cluster_radius) : min(
                        width, center_x + cluster_radius + 1
                    ),
                ][mask] = (
                    value_grid_gpu[
                        max(0, center_y - cluster_radius) : min(
                            height, center_y + cluster_radius + 1
                        ),
                        max(0, center_x - cluster_radius) : min(
                            width, center_x + cluster_radius + 1
                        ),
                    ][mask]
                    * multipliers[mask]
                ).astype(
                    cp.int32
                )

            # Transfer back to CPU
            return cp.asnumpy(result_grid_gpu)

        elif backend == "cuda" and CUDA_AVAILABLE:
            # For CUDA, we'll implement a kernel
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
                            result_grid[y, x] = int(
                                value_grid[y, x] * current_multiplier
                            )

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
                value_multiplier,
                width,
                height,
            )

            return result_grid

    except Exception as e:
        logging.warning(
            f"GPU value clustering failed: {str(e)}. Falling back to CPU implementation."
        )
        return add_value_clusters(
            value_grid, num_clusters, cluster_radius, value_multiplier
        )
