#!/usr/bin/env python3
"""
Comprehensive GPU Benchmarking Script for Space Muck.

This script provides a unified framework for benchmarking all GPU-accelerated operations
across different backends, grid sizes, and configurations. It generates detailed
performance metrics and visualizations to help optimize GPU acceleration usage.
"""

# Standard library imports
import logging
import os
import sys
import time

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np

# Local application imports
from entities.base_generator import BaseGenerator  # noqa: E402
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from utils.cellular_automaton_utils import apply_cellular_automaton  # noqa: E402
from utils.gpu_utils import (  # noqa: E402
from utils.value_generator import (  # noqa: E402
from utils.value_generator_gpu import (  # noqa: E402
import contextlib
import importlib.util
import platform

# contextlib import removed (no longer needed)

# For type checking only - these imports are not executed at runtime
if TYPE_CHECKING:
    # import cupy  # Uncomment if needed in the future
    import torch

# Note: This benchmarking script requires optional dependencies:
# - cupy: For CUDA-based GPU acceleration
# - torch: For PyTorch-based GPU acceleration
#
# If these dependencies are not available, the script will still run
# but will skip the corresponding GPU backends.

# Check for optional dependencies
CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if not CUPY_AVAILABLE:
    logging.warning(
        "CuPy not found. CUDA backend will be unavailable for benchmarking."
    )
    logging.warning(
        "To install: pip install cupy-cuda11x (replace with your CUDA version)"
    )

if not TORCH_AVAILABLE:
    logging.warning(
        "PyTorch not found. PyTorch backend will be unavailable for benchmarking."
    )
    logging.warning("To install: pip install torch")

# Set up dummy modules to prevent errors when dependencies are missing
class DummyModule:
    """Dummy module for when optional dependencies are not available.

    This class acts as a placeholder that returns itself for any attribute
    access or function call, allowing code to run without raising errors
    when optional dependencies are missing.
    """

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

# Import optional dependencies with proper error handling
cp = DummyModule()  # Default to dummy module
if CUPY_AVAILABLE:
    try:
        import cupy as cp  # type: ignore

        logging.info("Successfully imported CuPy for CUDA acceleration")
    except ImportError as e:
        logging.warning(f"Failed to import CuPy despite being found: {e}")
        CUPY_AVAILABLE = False

torch = DummyModule()  # Default to dummy module
if TORCH_AVAILABLE:
    try:
        import torch  # type: ignore

        logging.info(
            f"Successfully imported PyTorch {torch.__version__} for GPU acceleration"
        )
    except ImportError as e:
        logging.warning(f"Failed to import PyTorch despite being found: {e}")
        TORCH_AVAILABLE = False
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add the parent directory to the path so we can import our modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import all modules at the top

    get_available_backends,
    apply_cellular_automaton_gpu,
    apply_noise_generation_gpu,
    apply_kmeans_clustering_gpu,
    apply_dbscan_clustering_gpu,
)

    generate_value_distribution,
    add_value_clusters,
)

    generate_value_distribution_gpu,
    add_value_clusters_gpu,
)

# Define terrain generation functions
def generate_terrain_cpu(width, height):
    """Generate terrain using CPU implementation."""
    # Use value_generator for CPU implementation
    # Create base grid and grid for terrain generation
    base_grid = np.random.rand(height, width)  # Simplified for benchmark
    grid = np.ones((height, width))  # All cells active for terrain
    return generate_value_distribution(grid, base_grid)

def generate_terrain_gpu(width, height, backend="cuda"):
    """Generate terrain using GPU implementation."""
    # Use value_generator_gpu for GPU implementation
    # Create base grid and grid for terrain generation
    base_grid = np.random.rand(height, width)  # Simplified for benchmark
    grid = np.ones((height, width))  # All cells active for terrain
    return generate_value_distribution_gpu(grid, base_grid, backend=backend)

# Define resource distribution functions
def generate_resource_distribution_cpu(width, height):
    """Generate resource distribution using CPU implementation."""
    # Use value_generator for CPU implementation
    # Create base grid and value grid
    base_grid = np.random.rand(height, width)  # Simplified for benchmark
    grid = np.random.choice([0, 1], size=(height, width), p=[0.3, 0.7])  # Binary grid
    value_grid = generate_value_distribution(grid, base_grid)
    return add_value_clusters(value_grid, num_clusters=10, cluster_value_multiplier=1.5)

def generate_resource_distribution_gpu(width, height, backend="cuda"):
    """Generate resource distribution using GPU implementation."""
    # Use value_generator_gpu for GPU implementation
    # Create base grid and value grid
    base_grid = np.random.rand(height, width)  # Simplified for benchmark
    grid = np.random.choice([0, 1], size=(height, width), p=[0.3, 0.7])  # Binary grid
    value_grid = generate_value_distribution_gpu(grid, base_grid, backend=backend)
    return add_value_clusters_gpu(
        value_grid, num_clusters=10, cluster_value_multiplier=1.5, backend=backend
    )

def benchmark_cellular_automaton(
    grid_sizes: List[int], iterations: int = 5, repetitions: int = 3
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark cellular automaton performance across different backends.

    Args:
        grid_sizes: List of grid sizes to test
        iterations: Number of cellular automaton iterations
        repetitions: Number of times to repeat each test for averaging

    Returns:
        Dict: Nested dictionary with results organized by operation, backend, and grid size
    """
    # Get available backends
    backends = get_available_backends()
    if "cpu" not in backends:
        backends.append("cpu")

    # Initialize results dictionary
    results = {
        "operation": "cellular_automaton",
        "grid_sizes": grid_sizes,
        "times": {backend: [] for backend in backends},
        "memory": {backend: [] for backend in backends},
        "speedup": {backend: [] for backend in backends},
    }

    # Run benchmarks for each grid size
    for size in grid_sizes:
        logging.info(f"Benchmarking cellular automaton on grid size {size}x{size}...")

        # Create a random grid
        grid = np.random.randint(0, 2, (size, size), dtype=np.int8)

        # Track CPU time for speedup calculations
        cpu_time = 0

        # Benchmark each backend
        for backend in backends:
            times = []

            for _ in range(repetitions):
                # Benchmark CPU implementation
                start_time = time.time()
                if backend == "cpu":
                    apply_cellular_automaton(grid, iterations=iterations)
                else:
                    apply_cellular_automaton_gpu(
                        grid, backend=backend, iterations=iterations
                    )
                end_time = time.time()
                times.append(end_time - start_time)

            # Record average time
            avg_time = sum(times) / len(times)
            results["times"][backend].append(avg_time)

            # Store CPU time for speedup calculations
            if backend == "cpu":
                cpu_time = avg_time

            # Calculate speedup relative to CPU
            if backend != "cpu" and cpu_time > 0:
                speedup = cpu_time / avg_time
                results["speedup"][backend].append(speedup)
                logging.info(
                    f"  {backend}: {avg_time:.4f} seconds (speedup: {speedup:.2f}x)"
                )
            else:
                results["speedup"][backend].append(1.0)  # CPU speedup is 1.0
                logging.info(f"  {backend}: {avg_time:.4f} seconds")

    return results

def benchmark_clustering(
    data_sizes: List[int], n_clusters: int = 5, repetitions: int = 3
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark clustering algorithms performance across different backends.

    Args:
        data_sizes: List of data sizes to test (number of points)
        n_clusters: Number of clusters for K-means
        repetitions: Number of times to repeat each test for averaging

    Returns:
        Dict: Nested dictionary with results organized by operation, backend, and data size
    """
    # Get available backends
    backends = get_available_backends()
    if "cpu" not in backends:
        backends.append("cpu")

    # Initialize results dictionary for K-means
    kmeans_results = {
        "operation": "kmeans_clustering",
        "data_sizes": data_sizes,
        "times": {backend: [] for backend in backends},
        "memory": {backend: [] for backend in backends},
        "speedup": {backend: [] for backend in backends},
    }

    # Initialize results dictionary for DBSCAN
    dbscan_results = {
        "operation": "dbscan_clustering",
        "data_sizes": data_sizes,
        "times": {backend: [] for backend in backends},
        "memory": {backend: [] for backend in backends},
        "speedup": {backend: [] for backend in backends},
    }

    # Run benchmarks for each data size
    for size in data_sizes:
        logging.info(f"Benchmarking clustering algorithms on data size {size}...")

        # Create random data points
        data = np.random.random((size, 2))

        # Track CPU times for speedup calculations
        kmeans_cpu_time = 0
        dbscan_cpu_time = 0

        # Benchmark each backend
        for backend in backends:
            # K-means benchmarking
            kmeans_times = []
            for _ in range(repetitions):
                try:
                    start_time = time.time()

                    if backend == "cpu":
                        # Use sklearn or a CPU implementation
                        from sklearn.cluster import KMeans

                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        labels = kmeans.fit_predict(data)
                        centroids = kmeans.cluster_centers_
                    else:
                        # Use GPU implementation
                        centroids, labels = apply_kmeans_clustering_gpu(
                            data=data, n_clusters=n_clusters, backend=backend
                        )

                    end_time = time.time()
                    kmeans_times.append(end_time - start_time)
                except Exception as e:
                    logging.warning(
                        f"K-means clustering failed with backend {backend}: {e}"
                    )
                    kmeans_times.append(float("inf"))

            # Record average time for K-means
            if kmeans_times and any(t != float("inf") for t in kmeans_times):
                valid_times = [t for t in kmeans_times if t != float("inf")]
                avg_time = (
                    sum(valid_times) / len(valid_times) if valid_times else float("inf")
                )
                kmeans_results["times"][backend].append(avg_time)

                # Store CPU time for speedup calculations
                if backend == "cpu":
                    kmeans_cpu_time = avg_time

                # Calculate speedup relative to CPU
                if (
                    backend != "cpu"
                    and kmeans_cpu_time > 0
                    and avg_time != float("inf")
                ):
                    speedup = kmeans_cpu_time / avg_time
                    kmeans_results["speedup"][backend].append(speedup)
                    logging.info(
                        f"  K-means {backend}: {avg_time:.4f} seconds (speedup: {speedup:.2f}x)"
                    )
                else:
                    kmeans_results["speedup"][backend].append(1.0)  # CPU speedup is 1.0
                    logging.info(f"  K-means {backend}: {avg_time:.4f} seconds")
            else:
                _speed_handler(kmeans_results, backend, "  K-means ")
            # DBSCAN benchmarking
            dbscan_times = []
            for _ in range(repetitions):
                try:
                    start_time = time.time()

                    if backend == "cpu":
                        # Use sklearn for CPU implementation
                        from sklearn.cluster import DBSCAN

                        dbscan = DBSCAN(eps=0.3, min_samples=5)
                        dbscan.fit_predict(data)
                    else:
                        # Use GPU implementation
                        apply_dbscan_clustering_gpu(
                            data=data, eps=0.3, min_samples=5, backend=backend
                        )

                    end_time = time.time()
                    dbscan_times.append(end_time - start_time)
                except Exception as e:
                    logging.warning(
                        f"DBSCAN clustering failed with backend {backend}: {e}"
                    )
                    dbscan_times.append(float("inf"))

            # Record average time for DBSCAN
            if dbscan_times and any(t != float("inf") for t in dbscan_times):
                valid_times = [t for t in dbscan_times if t != float("inf")]
                avg_time = (
                    sum(valid_times) / len(valid_times) if valid_times else float("inf")
                )
                dbscan_results["times"][backend].append(avg_time)

                # Store CPU time for speedup calculations
                if backend == "cpu":
                    dbscan_cpu_time = avg_time

                # Calculate speedup relative to CPU
                if (
                    backend != "cpu"
                    and dbscan_cpu_time > 0
                    and avg_time != float("inf")
                ):
                    speedup = dbscan_cpu_time / avg_time
                    dbscan_results["speedup"][backend].append(speedup)
                    logging.info(
                        f"  DBSCAN {backend}: {avg_time:.4f} seconds (speedup: {speedup:.2f}x)"
                    )
                else:
                    dbscan_results["speedup"][backend].append(1.0)  # CPU speedup is 1.0
                    logging.info(f"  DBSCAN {backend}: {avg_time:.4f} seconds")
            else:
                _speed_handler(dbscan_results, backend, "  DBSCAN ")
    return {"kmeans": kmeans_results, "dbscan": dbscan_results}

def _speed_handler(arg0, backend, arg2):
    arg0["times"][backend].append(float("inf"))
    arg0["speedup"][backend].append(0.0)
    logging.info(f"{arg2}{backend}: Failed")

def benchmark_value_generation(
    grid_sizes: List[int], repetitions: int = 3
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark value generation algorithms performance across different backends.
    This includes terrain height maps, resource distribution, and other value-based
    procedural generation techniques.

    Args:
        grid_sizes: List of grid sizes to test (width x height)
        repetitions: Number of times to repeat each test for averaging

    Returns:
        Dict: Nested dictionary with results organized by operation, backend, and grid size
    """
    # Get available backends
    backends = get_available_backends()
    if "cpu" not in backends:
        backends.append("cpu")

    # Initialize results dictionary for terrain generation
    terrain_results = {
        "operation": "terrain_generation",
        "grid_sizes": grid_sizes,
        "times": {backend: [] for backend in backends},
        "memory": {backend: [] for backend in backends},
        "speedup": {backend: [] for backend in backends},
    }

    # Initialize results dictionary for resource distribution
    resource_results = {
        "operation": "resource_distribution",
        "grid_sizes": grid_sizes,
        "times": {backend: [] for backend in backends},
        "memory": {backend: [] for backend in backends},
        "speedup": {backend: [] for backend in backends},
    }

    # Run benchmarks for each grid size
    for size in grid_sizes:
        logging.info(f"Benchmarking value generation on grid size {size}x{size}...")

        # Track CPU times for speedup calculations
        terrain_cpu_time = 0
        resource_cpu_time = 0

        # Benchmark each backend
        for backend in backends:
            # Terrain generation benchmarking
            terrain_times = []
            for _ in range(repetitions):
                try:
                    start_time = time.time()

                    if backend == "cpu":
                        # Use CPU implementation for terrain generation
                        generate_terrain_cpu(size, size)
                    else:
                        # Use GPU implementation for terrain generation
                        generate_terrain_gpu(size, size, backend=backend)

                    end_time = time.time()
                    terrain_times.append(end_time - start_time)
                except Exception as e:
                    logging.warning(
                        f"Terrain generation failed with backend {backend}: {e}"
                    )
                    terrain_times.append(float("inf"))

            # Record average time for terrain generation
            if terrain_times and any(t != float("inf") for t in terrain_times):
                valid_times = [t for t in terrain_times if t != float("inf")]
                avg_time = (
                    sum(valid_times) / len(valid_times) if valid_times else float("inf")
                )
                terrain_results["times"][backend].append(avg_time)

                # Store CPU time for speedup calculations
                if backend == "cpu":
                    terrain_cpu_time = avg_time

                # Calculate speedup relative to CPU
                if (
                    backend != "cpu"
                    and terrain_cpu_time > 0
                    and avg_time != float("inf")
                ):
                    speedup = terrain_cpu_time / avg_time
                    terrain_results["speedup"][backend].append(speedup)
                    logging.info(
                        f"  Terrain {backend}: {avg_time:.4f} seconds (speedup: {speedup:.2f}x)"
                    )
                else:
                    terrain_results["speedup"][backend].append(
                        1.0
                    )  # CPU speedup is 1.0
                    logging.info(f"  Terrain {backend}: {avg_time:.4f} seconds")
            else:
                _handle_benchmark_failure(terrain_results, backend, "  Terrain ")
            # Resource distribution benchmarking
            resource_times = []
            for _ in range(repetitions):
                try:
                    start_time = time.time()

                    if backend == "cpu":
                        # Use CPU implementation for resource distribution
                        generate_resource_distribution_cpu(size, size)
                    else:
                        # Use GPU implementation for resource distribution
                        generate_resource_distribution_gpu(size, size, backend=backend)

                    end_time = time.time()
                    resource_times.append(end_time - start_time)
                except Exception as e:
                    logging.warning(
                        f"Resource distribution failed with backend {backend}: {e}"
                    )
                    resource_times.append(float("inf"))

            # Record average time for resource distribution
            if resource_times and any(t != float("inf") for t in resource_times):
                valid_times = [t for t in resource_times if t != float("inf")]
                avg_time = (
                    sum(valid_times) / len(valid_times) if valid_times else float("inf")
                )
                resource_results["times"][backend].append(avg_time)

                # Store CPU time for speedup calculations
                if backend == "cpu":
                    resource_cpu_time = avg_time

                # Calculate speedup relative to CPU
                if (
                    backend != "cpu"
                    and resource_cpu_time > 0
                    and avg_time != float("inf")
                ):
                    speedup = resource_cpu_time / avg_time
                    resource_results["speedup"][backend].append(speedup)
                    logging.info(
                        f"  Resources {backend}: {avg_time:.4f} seconds (speedup: {speedup:.2f}x)"
                    )
                else:
                    resource_results["speedup"][backend].append(
                        1.0
                    )  # CPU speedup is 1.0
                    logging.info(f"  Resources {backend}: {avg_time:.4f} seconds")
            else:
                _handle_benchmark_failure(resource_results, backend, "  Resources ")
    return {"terrain": terrain_results, "resources": resource_results}

def _handle_benchmark_failure(results_dict, backend, operation_prefix):
    """Handle benchmark failure by setting appropriate values in the results dictionary.

    Args:
        results_dict: Dictionary containing benchmark results
        backend: The backend that failed (e.g., 'cuda', 'cupy')
        operation_prefix: Prefix string for logging the operation type

    Returns:
        None: Updates the results_dict in-place
    """
    results_dict["times"][backend].append(float("inf"))
    results_dict["speedup"][backend].append(0.0)
    logging.info(f"{operation_prefix}{backend}: Failed")

def benchmark_memory_transfer(
    data_sizes: List[int], repetitions: int = 5
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark memory transfer overhead between CPU and GPU for different data sizes.
    This is crucial for understanding when GPU acceleration is beneficial vs. when
    the transfer overhead negates the computational benefits.

    Args:
        data_sizes: List of data sizes to test (in elements)
        repetitions: Number of times to repeat each test for averaging

    Returns:
        Dict: Nested dictionary with results organized by operation, backend, and data size
    """
    # Get available backends (excluding CPU as this is specifically for GPU transfer)
    backends = get_available_backends()
    backends = [b for b in backends if b != "cpu"]

    if not backends:
        logging.warning("No GPU backends available for memory transfer benchmark")
        return {}

    # Initialize results dictionary for host to device transfer
    h2d_results = {
        "operation": "host_to_device_transfer",
        "data_sizes": data_sizes,
        "times": {backend: [] for backend in backends},
        "bandwidth": {backend: [] for backend in backends},
    }

    # Initialize results dictionary for device to host transfer
    d2h_results = {
        "operation": "device_to_host_transfer",
        "data_sizes": data_sizes,
        "times": {backend: [] for backend in backends},
        "bandwidth": {backend: [] for backend in backends},
    }

    # Run benchmarks for each data size
    for size in data_sizes:
        logging.info(f"Benchmarking memory transfer for data size {size} elements...")

        # Create random data array
        data = np.random.random(size).astype(np.float32)
        data_size_bytes = data.nbytes

        # Benchmark each backend
        for backend in backends:
            # Host to device transfer benchmarking
            h2d_times = []
            for _ in range(repetitions):
                try:
                    if backend in ["cuda", "cupy"] and CUPY_AVAILABLE:
                        # Ensure we're starting from CPU memory
                        cpu_data = np.array(data)
                        # Time the transfer
                        start_time = time.time()
                        gpu_data = cp.array(cpu_data)
                        # Force synchronization to ensure transfer is complete
                        cp.cuda.stream.get_current_stream().synchronize()
                        end_time = time.time()
                    elif backend == "mps" and TORCH_AVAILABLE:
                        # Ensure we're starting from CPU memory
                        cpu_data = torch.tensor(data)
                        # Time the transfer
                        start_time = time.time()
                        gpu_data = cpu_data.to("mps")
                        # Force synchronization
                        torch.mps.synchronize()
                        end_time = time.time()
                    else:
                        # Unknown backend
                        raise ValueError(
                            f"Unsupported backend for memory transfer: {backend}"
                        )

                    h2d_times.append(end_time - start_time)
                except Exception as e:
                    logging.warning(
                        f"Host to device transfer failed with backend {backend}: {e}"
                    )
                    h2d_times.append(float("inf"))

            # Record average time and calculate bandwidth for host to device
            if h2d_times and any(t != float("inf") for t in h2d_times):
                valid_times = [t for t in h2d_times if t != float("inf")]
                avg_time = (
                    sum(valid_times) / len(valid_times) if valid_times else float("inf")
                )
                h2d_results["times"][backend].append(avg_time)

                # Calculate bandwidth in GB/s
                if avg_time > 0:
                    bandwidth = (data_size_bytes / (1024**3)) / avg_time  # GB/s
                    h2d_results["bandwidth"][backend].append(bandwidth)
                    logging.info(
                        f"  H2D {backend}: {avg_time:.6f} seconds ({bandwidth:.2f} GB/s)"
                    )
                else:
                    h2d_results["bandwidth"][backend].append(0.0)
                    logging.info(f"  H2D {backend}: {avg_time:.6f} seconds")
            else:
                _handle_bandwidth_benchmark_failure(h2d_results, backend, "  H2D ")
            # Device to host transfer benchmarking
            d2h_times = []
            for _ in range(repetitions):
                try:
                    if backend in ["cuda", "cupy"] and CUPY_AVAILABLE:
                        # Create data on GPU
                        gpu_data = cp.random.random(size).astype(cp.float32)
                        # Time the transfer
                        start_time = time.time()
                        cpu_data = cp.asnumpy(gpu_data)
                        end_time = time.time()
                    elif backend == "mps" and TORCH_AVAILABLE:
                        # Create data on GPU
                        gpu_data = torch.rand(size, device="mps")
                        # Time the transfer
                        start_time = time.time()
                        cpu_data = gpu_data.to("cpu")
                        # Force synchronization
                        torch.mps.synchronize()
                        end_time = time.time()
                    else:
                        # Unknown backend
                        raise ValueError(
                            f"Unsupported backend for memory transfer: {backend}"
                        )

                    d2h_times.append(end_time - start_time)
                except Exception as e:
                    logging.warning(
                        f"Device to host transfer failed with backend {backend}: {e}"
                    )
                    d2h_times.append(float("inf"))

            # Record average time and calculate bandwidth for device to host
            if d2h_times and any(t != float("inf") for t in d2h_times):
                valid_times = [t for t in d2h_times if t != float("inf")]
                avg_time = (
                    sum(valid_times) / len(valid_times) if valid_times else float("inf")
                )
                d2h_results["times"][backend].append(avg_time)

                # Calculate bandwidth in GB/s
                if avg_time > 0:
                    bandwidth = (data_size_bytes / (1024**3)) / avg_time  # GB/s
                    d2h_results["bandwidth"][backend].append(bandwidth)
                    logging.info(
                        f"  D2H {backend}: {avg_time:.6f} seconds ({bandwidth:.2f} GB/s)"
                    )
                else:
                    d2h_results["bandwidth"][backend].append(0.0)
                    logging.info(f"  D2H {backend}: {avg_time:.6f} seconds")
            else:
                _handle_bandwidth_benchmark_failure(d2h_results, backend, "  D2H ")
    return {"host_to_device": h2d_results, "device_to_host": d2h_results}

def _handle_bandwidth_benchmark_failure(results_dict, backend, operation_prefix):
    """Handle bandwidth benchmark failure by setting appropriate values in the results dictionary.

    Args:
        results_dict: Dictionary containing benchmark results
        backend: The backend that failed (e.g., 'cuda', 'cupy')
        operation_prefix: Prefix string for logging the operation type

    Returns:
        None: Updates the results_dict in-place
    """
    results_dict["times"][backend].append(float("inf"))
    results_dict["bandwidth"][backend].append(0.0)
    logging.info(f"{operation_prefix}{backend}: Failed")

def visualize_benchmark_results(
    results: Dict[str, Any], title: str, output_dir: str = "./benchmark_results"
) -> None:
    """
    Visualize benchmark results with plots for execution time and speedup.

    Args:
        results: Dictionary containing benchmark results
        title: Title for the plots
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each operation in the results
    for op_name, op_results in results.items():
        if "operation" not in op_results:
            # This is a nested dictionary with multiple operations
            for sub_op_name, sub_op_results in op_results.items():
                visualize_single_operation(
                    sub_op_results, f"{title} - {op_name} - {sub_op_name}", output_dir
                )
        else:
            # This is a single operation result
            visualize_single_operation(op_results, f"{title} - {op_name}", output_dir)

def visualize_single_operation(
    op_results: Dict[str, Any], title: str, output_dir: str
) -> None:
    """
    Visualize results for a single operation.

    Args:
        op_results: Dictionary containing operation results
        title: Title for the plots
        output_dir: Directory to save the plots
    """
    # Extract data
    operation = op_results.get("operation", "unknown")

    # Determine x-axis values and label
    x_values, x_label = _get_x_axis_data(op_results, operation)
    if not x_values:  # Skip if no x-axis data found
        return

    # Create different plot types based on available data
    if "times" in op_results:
        _create_time_plot(op_results, x_values, x_label, title, operation, output_dir)

    if "speedup" in op_results:
        _create_speedup_plot(
            op_results, x_values, x_label, title, operation, output_dir
        )

    if "bandwidth" in op_results:
        _create_bandwidth_plot(
            op_results, x_values, x_label, title, operation, output_dir
        )

def _get_x_axis_data(
    op_results: Dict[str, Any], operation: str
) -> Tuple[List[int], str]:
    """Extract x-axis data and label from operation results."""
    if "grid_sizes" in op_results:
        return op_results["grid_sizes"], "Grid Size"
    elif "data_sizes" in op_results:
        return op_results["data_sizes"], "Data Size"
    else:
        logging.warning(
            f"No size information found for {operation}, skipping visualization"
        )
        return [], ""

def _set_log_scale_if_needed(x_values: List[int]) -> None:
    """Set x-axis to log scale if the range is large enough."""
    if len(x_values) > 1 and max(x_values) / min(x_values) > 100:
        plt.xscale("log")

def _filter_valid_data(
    values: List[float], x_values: List[int], filter_func=None
) -> Tuple[List[int], List[float]]:
    """Filter out invalid data points and return valid x and y values."""
    if filter_func is None:
        # Default filter: keep non-infinite values
        def default_filter(v):
            return v != float("inf")

        filter_func = default_filter

    valid_indices = [i for i, v in enumerate(values) if filter_func(v)]
    valid_x = [x_values[i] for i in valid_indices]
    valid_y = [values[i] for i in valid_indices]
    return valid_x, valid_y

def _create_time_plot(
    op_results: Dict[str, Any],
    x_values: List[int],
    x_label: str,
    title: str,
    operation: str,
    output_dir: str,
) -> None:
    """Create execution time plot."""
    plt.figure(figsize=(12, 8))

    # Plot data for each backend
    for backend, times in op_results["times"].items():
        if not times or all(t == float("inf") for t in times):
            continue

        valid_x, valid_times = _filter_valid_data(times, x_values)
        plt.plot(valid_x, valid_times, marker="o", label=f"{backend}")

    # Configure plot
    plt.title(f"{title} - Execution Time")
    plt.xlabel(x_label)
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.legend()

    # Set x-axis scale
    _set_log_scale_if_needed(x_values)

    # Set y-axis to log scale if the range is large enough
    with contextlib.suppress(ValueError, ZeroDivisionError):
        should_use_log_scale = False
        for times in op_results["times"].values():
            if not times:
                continue

            valid_times = [t for t in times if t != float("inf") and t > 0]
            if len(valid_times) >= 2:
                time_range = max(valid_times) / min(valid_times)
                if time_range > 100:
                    should_use_log_scale = True
                    break

        if should_use_log_scale:
            plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{operation}_time.png"))
    plt.close()

def _create_speedup_plot(
    op_results: Dict[str, Any],
    x_values: List[int],
    x_label: str,
    title: str,
    operation: str,
    output_dir: str,
) -> None:
    """Create speedup comparison plot."""
    plt.figure(figsize=(12, 8))

    # Define filter function outside the loop
    def positive_filter(s):
        return s > 0

    # Plot data for each backend
    for backend, speedups in op_results["speedup"].items():
        if backend == "cpu" or not speedups or all(s <= 0 for s in speedups):
            continue

        valid_x, valid_speedups = _filter_valid_data(
            speedups, x_values, positive_filter
        )
        plt.plot(valid_x, valid_speedups, marker="o", label=f"{backend}")

    # Configure plot
    plt.title(f"{title} - Speedup vs CPU")
    plt.xlabel(x_label)
    plt.ylabel("Speedup Factor (x)")
    plt.grid(True)
    plt.legend()
    plt.axhline(y=1, color="r", linestyle="--", label="CPU Baseline")

    # Set x-axis scale
    _set_log_scale_if_needed(x_values)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{operation}_speedup.png"))
    plt.close()

def _create_bandwidth_plot(
    op_results: Dict[str, Any],
    x_values: List[int],
    x_label: str,
    title: str,
    operation: str,
    output_dir: str,
) -> None:
    """Create bandwidth plot for memory transfer benchmarks."""
    plt.figure(figsize=(12, 8))

    # Define filter function outside the loop
    def positive_filter(b):
        return b > 0

    # Plot data for each backend
    for backend, bandwidths in op_results["bandwidth"].items():
        if not bandwidths or all(b <= 0 for b in bandwidths):
            continue

        valid_x, valid_bandwidths = _filter_valid_data(
            bandwidths, x_values, positive_filter
        )
        plt.plot(valid_x, valid_bandwidths, marker="o", label=f"{backend}")

    # Configure plot
    plt.title(f"{title} - Memory Bandwidth")
    plt.xlabel(x_label)
    plt.ylabel("Bandwidth (GB/s)")
    plt.grid(True)
    plt.legend()

    # Set x-axis scale
    _set_log_scale_if_needed(x_values)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{operation}_bandwidth.png"))
    plt.close()

def run_all_benchmarks(
    output_dir: str = "./benchmark_results",
    small_grid_sizes: Optional[List[int]] = None,
    large_data_sizes: Optional[List[int]] = None,
    repetitions: int = 3,
) -> None:
    """
    Run all benchmark functions and visualize results.

    Args:
        output_dir: Directory to save benchmark results and visualizations
        small_grid_sizes: Grid sizes for 2D operations (cellular automaton, noise)
        large_data_sizes: Data sizes for 1D operations (clustering, memory transfer)
        repetitions: Number of repetitions for each benchmark
    """
    if small_grid_sizes is None:
        small_grid_sizes = [32, 64, 128, 256, 512, 1024]
    if large_data_sizes is None:
        large_data_sizes = [1000, 10000, 100000, 1000000]
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging to file and console
    log_file = os.path.join(output_dir, "benchmark_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

    # Log system information
    logging.info(f"System: {platform.system()} {platform.release()}")
    logging.info(f"Python: {platform.python_version()}")
    logging.info(f"Available backends: {get_available_backends()}")

    # Run cellular automaton benchmarks
    logging.info("\n===== CELLULAR AUTOMATON BENCHMARKS =====")
    ca_results = benchmark_cellular_automaton(
        grid_sizes=small_grid_sizes, iterations=5, repetitions=repetitions
    )
    visualize_benchmark_results(ca_results, "Cellular Automaton", output_dir)

    # Run noise generation benchmarks
    logging.info("\n===== NOISE GENERATION BENCHMARKS =====")
    noise_results = benchmark_noise_generation(
        grid_sizes=small_grid_sizes, octaves=4, repetitions=repetitions
    )
    visualize_benchmark_results(noise_results, "Noise Generation", output_dir)

    # Run clustering benchmarks
    logging.info("\n===== CLUSTERING BENCHMARKS =====")
    clustering_results = benchmark_clustering(
        data_sizes=large_data_sizes[:3],  # Use smaller subset for clustering
        n_clusters=5,
        repetitions=repetitions,
    )
    visualize_benchmark_results(clustering_results, "Clustering", output_dir)

    # Run value generation benchmarks
    logging.info("\n===== VALUE GENERATION BENCHMARKS =====")
    value_results = benchmark_value_generation(
        grid_sizes=small_grid_sizes, repetitions=repetitions
    )
    visualize_benchmark_results(value_results, "Value Generation", output_dir)

    # Run memory transfer benchmarks
    logging.info("\n===== MEMORY TRANSFER BENCHMARKS =====")
    memory_results = benchmark_memory_transfer(
        data_sizes=large_data_sizes, repetitions=repetitions
    )
    visualize_benchmark_results(memory_results, "Memory Transfer", output_dir)

    logging.info("\n===== ALL BENCHMARKS COMPLETED =====")
    logging.info(f"Results saved to {output_dir}")

def benchmark_noise_generation(
    grid_sizes: List[int], octaves: int = 4, repetitions: int = 3
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark noise generation performance across different backends.

    Args:
        grid_sizes: List of grid sizes to test
        octaves: Number of octaves for noise generation
        repetitions: Number of times to repeat each test for averaging

    Returns:
        Dict: Nested dictionary with results organized by operation, backend, and grid size
    """
    # Get available backends
    backends = get_available_backends()
    if "cpu" not in backends:
        backends.append("cpu")

    # Initialize results dictionary
    results = {
        "operation": "noise_generation",
        "grid_sizes": grid_sizes,
        "times": {backend: [] for backend in backends},
        "memory": {backend: [] for backend in backends},
        "speedup": {backend: [] for backend in backends},
    }

    # Run benchmarks for each grid size
    for size in grid_sizes:
        logging.info(f"Benchmarking noise generation on grid size {size}x{size}...")

        # Track CPU time for speedup calculations
        cpu_time = 0

        # Benchmark each backend
        for backend in backends:
            times = []

            for _ in range(repetitions):
                start_time = time.time()

                if backend == "cpu":
                    # Use BaseGenerator with GPU disabled for CPU benchmark
                    generator = BaseGenerator(width=size, height=size, use_gpu=False)
                    generator.generate_noise_layer(scale=0.1)
                else:
                    # Use apply_noise_generation_gpu directly
                    apply_noise_generation_gpu(
                        width=size,
                        height=size,
                        scale=0.1,
                        octaves=octaves,
                        backend=backend,
                    )

                end_time = time.time()
                times.append(end_time - start_time)

            # Record average time
            avg_time = sum(times) / len(times)
            results["times"][backend].append(avg_time)

            # Store CPU time for speedup calculations
            if backend == "cpu":
                cpu_time = avg_time

            # Calculate speedup relative to CPU
            if backend != "cpu" and cpu_time > 0:
                speedup = cpu_time / avg_time
                results["speedup"][backend].append(speedup)
                logging.info(
                    f"  {backend}: {avg_time:.4f} seconds (speedup: {speedup:.2f}x)"
                )
            else:
                results["speedup"][backend].append(1.0)  # CPU speedup is 1.0
                logging.info(f"  {backend}: {avg_time:.4f} seconds")

    return results
