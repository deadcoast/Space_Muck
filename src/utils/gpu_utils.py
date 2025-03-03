#!/usr/bin/env python3
"""
GPU Acceleration utilities for Space Muck.

This module provides GPU-accelerated implementations of common operations
used throughout the codebase, with fallback mechanisms for systems without
GPU support.
"""

# Standard library imports
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Third-party library imports
import numpy as np

# GPU acceleration libraries - try to import with fallbacks
try:
    import numba
    from numba import cuda

    NUMBA_AVAILABLE = True
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        logging.info("CUDA is available for GPU acceleration")
    else:
        logging.info("Numba is available but CUDA is not detected")
except ImportError:
    NUMBA_AVAILABLE = False
    CUDA_AVAILABLE = False
    logging.warning("Numba not available. GPU acceleration will be disabled.")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    logging.info("CuPy is available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. Some GPU operations will be disabled.")


def is_gpu_available() -> bool:
    """
    Check if any GPU acceleration is available.

    Returns:
        bool: True if any GPU acceleration is available
    """
    return CUDA_AVAILABLE or CUPY_AVAILABLE


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
    if CUPY_AVAILABLE:
        return cp.asarray(array)
    return array


def to_cpu(array: Any) -> np.ndarray:
    """
    Transfer an array from GPU back to CPU if necessary.

    Args:
        array: Input array (either on GPU or CPU)

    Returns:
        np.ndarray: Array on CPU
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return array.get()
    return array


if NUMBA_AVAILABLE and CUDA_AVAILABLE:

    @cuda.jit
    def _apply_cellular_automaton_kernel(
        grid, new_grid, width, height, birth_set, survival_set
    ):
        """CUDA kernel for cellular automaton."""
        x, y = cuda.grid(2)

        if x < width and y < height:
            # Count live neighbors
            neighbors = 0

            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = (x + dx) % width, (y + dy) % height
                    if grid[ny, nx] > 0:
                        neighbors += 1

            # Apply rules
            if grid[y, x] > 0:
                # Cell is alive
                new_grid[y, x] = 1 if neighbors in survival_set else 0
            else:
                # Cell is dead
                new_grid[y, x] = 1 if neighbors in birth_set else 0


def apply_cellular_automaton_gpu(
    grid: np.ndarray,
    birth_set: Set[int] = {3},
    survival_set: Set[int] = {2, 3},
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
        backend: GPU backend to use ('cuda', 'cupy', 'auto')

    Returns:
        np.ndarray: Evolved grid
    """
    height, width = grid.shape

    # Convert sets to lists for GPU compatibility
    birth_list = list(birth_set)
    survival_list = list(survival_set)

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
        from src.utils.cellular_automaton_utils import apply_cellular_automaton

        return apply_cellular_automaton(
            grid, birth_set, survival_set, iterations, wrap, width, height
        )

    # CUDA implementation using Numba
    if backend == "cuda" and CUDA_AVAILABLE:
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

    # CuPy implementation
    if backend == "cupy" and CUPY_AVAILABLE:
        # Transfer to GPU
        d_grid = cp.asarray((grid > 0).astype(np.int8))

        for _ in range(iterations):
            # Count neighbors using convolution
            neighbors_kernel = cp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            if wrap:
                neighbor_count = cp.fft.ifft2(
                    cp.fft.fft2(d_grid) * cp.fft.fft2(neighbors_kernel, d_grid.shape)
                ).real.astype(cp.int8)
            else:
                neighbor_count = cp.zeros_like(d_grid)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        neighbor_count += cp.roll(d_grid, (dy, dx), axis=(0, 1))

            # Apply rules
            new_grid = cp.zeros_like(d_grid)

            # Apply survival rules
            for n in survival_list:
                new_grid |= (neighbor_count == n) & (d_grid > 0)

            # Apply birth rules
            for n in birth_list:
                new_grid |= (neighbor_count == n) & (d_grid == 0)

            d_grid = new_grid

        # Transfer back to CPU and preserve original values
        result_grid = cp.asnumpy(d_grid)
        return grid * result_grid

    # Fallback to CPU implementation
    from src.utils.cellular_automaton_utils import apply_cellular_automaton

    return apply_cellular_automaton(
        grid, birth_set, survival_set, iterations, wrap, width, height
    )


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
        backend: GPU backend to use ('cuda', 'cupy', 'auto')

    Returns:
        np.ndarray: Generated noise grid
    """
    # Choose backend
    if backend == "auto":
        if CUPY_AVAILABLE:
            backend = "cupy"
        elif CUDA_AVAILABLE:
            backend = "cuda"
        else:
            backend = "cpu"

    # Use CPU implementation if no GPU is available or requested
    if backend == "cpu" or (not CUDA_AVAILABLE and not CUPY_AVAILABLE):
        try:
            from src.utils.noise_generator import generate_perlin_noise

            return generate_perlin_noise(
                width, height, scale, octaves, persistence, lacunarity, seed
            )
        except ImportError:
            logging.warning("Noise generator not available. Using random noise.")
            rng = np.random.RandomState(seed)
            return rng.random((height, width))

    # CuPy implementation
    if backend == "cupy" and CUPY_AVAILABLE:
        if seed is not None:
            cp.random.seed(seed)

        noise = cp.zeros((height, width), dtype=cp.float32)

        # Generate noise with multiple octaves
        max_amplitude = 0
        amplitude = 1.0
        frequency = scale

        for _ in range(octaves):
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

    # CUDA implementation not provided for noise generation due to complexity
    # Fallback to CPU implementation
    try:
        from src.utils.noise_generator import generate_perlin_noise

        return generate_perlin_noise(
            width, height, scale, octaves, persistence, lacunarity, seed
        )
    except ImportError:
        logging.warning("Noise generator not available. Using random noise.")
        rng = np.random.RandomState(seed)
        return rng.random((height, width))


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
        backend: GPU backend to use ('cuda', 'cupy', 'auto')

    Returns:
        Tuple[np.ndarray, np.ndarray]: (cluster_centers, labels)
    """
    n_samples, n_features = data.shape

    # Choose backend
    if backend == "auto":
        if CUPY_AVAILABLE:
            backend = "cupy"
        elif CUDA_AVAILABLE:
            backend = "cuda"
        else:
            backend = "cpu"

    # Use CPU implementation if no GPU is available or requested
    if backend == "cpu" or (not CUDA_AVAILABLE and not CUPY_AVAILABLE):
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
            rng = np.random.RandomState(seed)
            # Initialize centroids randomly
            idx = rng.choice(n_samples, n_clusters, replace=False)
            centroids = data[idx].copy()

            for _ in range(max_iterations):
                # Assign points to nearest centroid
                distances = np.zeros((n_samples, n_clusters))
                for i in range(n_clusters):
                    distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)
                labels = np.argmin(distances, axis=1)

                # Update centroids
                new_centroids = np.zeros((n_clusters, n_features))
                for i in range(n_clusters):
                    if np.sum(labels == i) > 0:
                        new_centroids[i] = np.mean(data[labels == i], axis=0)
                    else:
                        # If a cluster is empty, reinitialize it
                        new_centroids[i] = data[rng.choice(n_samples)]

                # Check for convergence
                if np.sum((new_centroids - centroids) ** 2) < tolerance:
                    break

                centroids = new_centroids

            return centroids, labels

    # CuPy implementation
    if backend == "cupy" and CUPY_AVAILABLE:
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

    # CUDA implementation using Numba
    if backend == "cuda" and CUDA_AVAILABLE and NUMBA_AVAILABLE:
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

        # Transfer data to device
        d_data = cuda.to_device(data)

        # Initialize centroids randomly
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_samples, n_clusters, replace=False)
        centroids = data[idx].copy()
        d_centroids = cuda.to_device(centroids)

        # Prepare distance matrix and labels array
        distances = np.zeros((n_samples, n_clusters), dtype=np.float32)
        d_distances = cuda.to_device(distances)

        # Configure CUDA grid
        threadsperblock = 256
        blockspergrid = (n_samples + threadsperblock - 1) // threadsperblock

        # K-means iterations
        for _ in range(max_iterations):
            # Compute distances
            _compute_distances_kernel[blockspergrid, threadsperblock](
                d_data, d_centroids, d_distances
            )

            # Copy distances back to host
            distances = d_distances.copy_to_host()

            # Find nearest centroid for each point
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros((n_clusters, n_features), dtype=np.float32)
            for i in range(n_clusters):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    # If a cluster is empty, reinitialize it
                    new_centroids[i] = data[rng.choice(n_samples)]

            # Check for convergence
            if np.sum((new_centroids - centroids) ** 2) < tolerance:
                break

            centroids = new_centroids
            d_centroids = cuda.to_device(centroids)

        return centroids, labels

    # If we reach here, fallback to CPU implementation
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
        # Simple K-means implementation (same as above)
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_samples, n_clusters, replace=False)
        centroids = data[idx].copy()

        for _ in range(max_iterations):
            distances = np.zeros((n_samples, n_clusters))
            for i in range(n_clusters):
                distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros((n_clusters, n_features))
            for i in range(n_clusters):
                if np.sum(labels == i) > 0:
                    new_centroids[i] = np.mean(data[labels == i], axis=0)
                else:
                    new_centroids[i] = data[rng.choice(n_samples)]

            if np.sum((new_centroids - centroids) ** 2) < tolerance:
                break

            centroids = new_centroids

        return centroids, labels


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
        backend: GPU backend to use ('cuda', 'cupy', 'auto')

    Returns:
        np.ndarray: Cluster labels for each point. Noisy samples are labeled as -1.
    """
    n_samples = data.shape[0]

    # Choose backend
    if backend == "auto":
        if CUPY_AVAILABLE:
            backend = "cupy"
        elif CUDA_AVAILABLE:
            backend = "cuda"
        else:
            backend = "cpu"

    # Use CPU implementation if no GPU is available or requested
    if backend == "cpu" or (not CUDA_AVAILABLE and not CUPY_AVAILABLE):
        try:
            from sklearn.cluster import DBSCAN

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            return dbscan.fit_predict(data)
        except ImportError:
            logging.warning(
                "scikit-learn not available. DBSCAN requires scikit-learn or GPU."
            )
            # Return all points as noise if scikit-learn is not available
            return np.full(n_samples, -1)

    # CuPy implementation
    if backend == "cupy" and CUPY_AVAILABLE:
        # Transfer data to GPU
        d_data = cp.asarray(data)

        # Compute pairwise distances
        distances = cp.zeros((n_samples, n_samples), dtype=cp.float32)
        for i in range(n_samples):
            diff = d_data - d_data[i]
            distances[i] = cp.sum(diff * diff, axis=1)

        # Find neighbors
        neighbors = distances <= eps * eps

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
            if not core_points[i] or labels[i] != -1:
                continue

            # Start a new cluster
            labels[i] = cluster_id

            # Find all points reachable from this core point
            stack = [i]
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

            # Move to next cluster
            cluster_id += 1

        # Transfer results back to CPU
        return cp.asnumpy(labels)

    # CUDA implementation using Numba
    if backend == "cuda" and CUDA_AVAILABLE and NUMBA_AVAILABLE:
        # Define CUDA kernels for DBSCAN
        @cuda.jit
        def _compute_distance_matrix_kernel(data, distances):
            """Compute pairwise distances between all points."""
            i, j = cuda.grid(2)

            if i < data.shape[0] and j < data.shape[0]:
                dist = 0.0
                for f in range(data.shape[1]):
                    diff = data[i, f] - data[j, f]
                    dist += diff * diff
                distances[i, j] = dist

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

        # Compute distances
        _compute_distance_matrix_kernel[blockspergrid, threadsperblock](
            d_data, d_distances
        )

        # Copy distances back to host
        distances = d_distances.copy_to_host()

        # Find neighbors
        neighbors = distances <= eps * eps

        # Count neighbors
        n_neighbors = np.sum(neighbors, axis=1)

        # Find core points
        core_points = n_neighbors >= min_samples

        # Initialize labels
        labels = np.full(n_samples, -1, dtype=np.int32)

        # Cluster ID counter
        cluster_id = 0

        # Perform clustering
        for i in range(n_samples):
            if not core_points[i] or labels[i] != -1:
                continue

            # Start a new cluster
            labels[i] = cluster_id

            # Find all points reachable from this core point
            stack = [i]
            while stack:
                current = stack.pop()

                # Get neighbors
                current_neighbors = np.where(neighbors[current])[0]

                # Process neighbors
                for neighbor in current_neighbors:
                    # If not yet assigned to a cluster
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id

                        # If it's a core point, add to stack for further expansion
                        if core_points[neighbor]:
                            stack.append(neighbor)

            # Move to next cluster
            cluster_id += 1

        return labels

    # If we reach here, fallback to CPU implementation
    try:
        from sklearn.cluster import DBSCAN

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(data)
    except ImportError:
        logging.warning(
            "scikit-learn not available. DBSCAN requires scikit-learn or GPU."
        )
        # Return all points as noise if scikit-learn is not available
        return np.full(n_samples, -1)
