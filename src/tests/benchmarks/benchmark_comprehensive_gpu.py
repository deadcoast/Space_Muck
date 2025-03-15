#!/usr/bin/env python3
"""
Comprehensive GPU Benchmarking Script for Space Muck.

This script provides a unified framework for benchmarking all GPU-accelerated operations
across different backends, grid sizes, and configurations. It generates detailed
performance metrics and visualizations to help optimize GPU acceleration usage.
"""

import contextlib
import importlib.util

# Standard library imports
import logging
import os
import platform
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np

# Local application imports
from entities.base_generator import BaseGenerator  # noqa: E402
from utils.cellular_automaton_utils import apply_cellular_automaton  # noqa: E402
from utils.gpu_utils import (
    apply_cellular_automaton_gpu,
    apply_dbscan_clustering_gpu,
    apply_kmeans_clustering_gpu,
    apply_noise_generation_gpu,
    get_available_backends,
)
from utils.value_generator import (
    add_value_clusters,
    generate_value_distribution,
)
from utils.value_generator_gpu import (
    add_value_clusters_gpu,
    generate_value_distribution_gpu,
)

rng = np.random.default_rng(seed=42)

# For type checking only - these imports are not executed at runtime
if TYPE_CHECKING:
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


# Define terrain generation functions
def generate_terrain_cpu(width, height):
    """Generate terrain using CPU implementation."""
    # Use value_generator for CPU implementation
    # Create base grid and grid for terrain generation
    base_grid = rng.random((height, width))  # Simplified for benchmark
    grid = np.ones((height, width))  # All cells active for terrain
    return generate_value_distribution(grid, base_grid)


def generate_terrain_gpu(width, height, backend="cuda"):
    """Generate terrain using GPU implementation."""
    # Use value_generator_gpu for GPU implementation
    # Create base grid and grid for terrain generation
    base_grid = rng.random((height, width))  # Simplified for benchmark
    grid = np.ones((height, width))  # All cells active for terrain
    return generate_value_distribution_gpu(grid, base_grid, backend=backend)


# Define resource distribution functions
def generate_resource_distribution_cpu(width, height):
    """Generate resource distribution using CPU implementation."""
    # Use value_generator for CPU implementation
    # Create base grid and value grid
    base_grid = rng.random((height, width))  # Simplified for benchmark
    grid = rng.choice([0, 1], size=(height, width), p=[0.3, 0.7])  # Binary grid
    value_grid = generate_value_distribution(grid, base_grid)
    return add_value_clusters(value_grid, num_clusters=10, cluster_value_multiplier=1.5)


def generate_resource_distribution_gpu(width, height, backend="cuda"):
    """Generate resource distribution using GPU implementation."""
    # Use value_generator_gpu for GPU implementation
    # Create base grid and value grid
    base_grid = rng.random((height, width))  # Simplified for benchmark
    grid = rng.choice([0, 1], size=(height, width), p=[0.3, 0.7])  # Binary grid
    value_grid = generate_value_distribution_gpu(grid, base_grid, backend=backend)
    return add_value_clusters_gpu(
        value_grid, num_clusters=10, cluster_value_multiplier=1.5, backend=backend
    )


def _initialize_benchmark_results(
    backends: List[str], grid_sizes: List[int]
) -> Dict[str, Dict[str, List[float]]]:
    """Initialize the results dictionary for benchmarking.

    Args:
        backends: List of backends to benchmark
        grid_sizes: List of grid sizes to test

    Returns:
        Dict: Initialized results dictionary
    """
    return {
        "operation": "cellular_automaton",
        "grid_sizes": grid_sizes,
        "times": {backend: [] for backend in backends},
        "memory": {backend: [] for backend in backends},
        "speedup": {backend: [] for backend in backends},
    }


def _run_cellular_automaton(grid: np.ndarray, backend: str, iterations: int) -> None:
    """Run cellular automaton using the specified backend.

    Args:
        grid: Input grid for cellular automaton
        backend: Backend to use ("cpu" or GPU backend)
        iterations: Number of iterations to run
    """
    if backend == "cpu":
        apply_cellular_automaton(grid, iterations=iterations)
    else:
        apply_cellular_automaton_gpu(grid, backend=backend, iterations=iterations)


def _measure_execution_time(func, *args, **kwargs) -> float:
    """Measure the execution time of a function.

    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        float: Execution time in seconds
    """
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time


def _calculate_and_log_speedup(
    results: Dict, backend: str, avg_time: float, cpu_time: float
) -> None:
    """Calculate speedup relative to CPU and log results.

    Args:
        results: Results dictionary to update
        backend: Current backend being benchmarked
        avg_time: Average execution time for the current backend
        cpu_time: CPU execution time for comparison
    """
    if backend != "cpu" and cpu_time > 0:
        speedup = cpu_time / avg_time
        results["speedup"][backend].append(speedup)
        logging.info(f"  {backend}: {avg_time:.4f} seconds (speedup: {speedup:.2f}x)")
    else:
        results["speedup"][backend].append(1.0)  # CPU speedup is 1.0
        logging.info(f"  {backend}: {avg_time:.4f} seconds")


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
    results = _initialize_benchmark_results(backends, grid_sizes)

    # Run benchmarks for each grid size
    for size in grid_sizes:
        logging.info(f"Benchmarking cellular automaton on grid size {size}x{size}...")

        # Create a random grid
        grid = rng.integers(0, 2, size=(size, size), dtype=np.int8)

        # Track CPU time for speedup calculations
        cpu_time = 0

        # Benchmark each backend
        for backend_index, backend in enumerate(backends):
            times = []

            for _ in range(repetitions):
                # Measure execution time
                execution_time = _measure_execution_time(
                    _run_cellular_automaton, grid, backend, iterations
                )
                times.append(execution_time)

            # Record average time
            avg_time = sum(times) / len(times)
            results["times"][backend].append(avg_time)

            # Store CPU time for speedup calculations
            if backend == "cpu":
                cpu_time = avg_time

            # Calculate speedup relative to CPU and log results
            _calculate_and_log_speedup(results, backend, avg_time, cpu_time)

    return results


def _initialize_clustering_results(
    algorithm_name: str, backends: List[str], data_sizes: List[int]
) -> Dict:
    """Initialize results dictionary for clustering benchmarks.

    Args:
        algorithm_name: Name of the clustering algorithm
        backends: List of backends to benchmark
        data_sizes: List of data sizes to test

    Returns:
        Dict: Initialized results dictionary
    """
    return {
        "operation": f"{algorithm_name}_clustering",
        "data_sizes": data_sizes,
        "times": {backend: [] for backend in backends},
        "memory": {backend: [] for backend in backends},
        "speedup": {backend: [] for backend in backends},
    }


def _run_kmeans_clustering(data: np.ndarray, n_clusters: int, backend: str) -> None:
    """Run K-means clustering using the specified backend.

    Args:
        data: Input data for clustering
        n_clusters: Number of clusters
        backend: Backend to use ("cpu" or GPU backend)
    """
    if backend == "cpu":
        # Use sklearn for CPU implementation
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        _ = kmeans.fit_predict(data)
        _ = kmeans.cluster_centers_
    else:
        # Use GPU implementation
        _, _ = apply_kmeans_clustering_gpu(
            data=data, n_clusters=n_clusters, backend=backend
        )


def _run_dbscan_clustering(data: np.ndarray, backend: str) -> None:
    """Run DBSCAN clustering using the specified backend.

    Args:
        data: Input data for clustering
        backend: Backend to use ("cpu" or GPU backend)
    """
    if backend == "cpu":
        # Use sklearn for CPU implementation
        from sklearn.cluster import DBSCAN

        dbscan = DBSCAN(eps=0.3, min_samples=5)
        dbscan.fit_predict(data)
    else:
        # Use GPU implementation
        apply_dbscan_clustering_gpu(data=data, eps=0.3, min_samples=5, backend=backend)


def _calculate_average_time(times: List[float]) -> float:
    """Calculate average execution time, filtering out failed runs.

    Args:
        times: List of execution times, with float("inf") representing failed runs

    Returns:
        float: Average execution time or float("inf") if all runs failed
    """
    if not times or all(t == float("inf") for t in times):
        return float("inf")

    valid_times = [t for t in times if t != float("inf")]
    return sum(valid_times) / len(valid_times) if valid_times else float("inf")


def _process_benchmark_results(
    results: Dict, backend: str, avg_time: float, cpu_time: float, algorithm_name: str
) -> None:
    """Process benchmark results and calculate speedups.

    Args:
        results: Results dictionary to update
        backend: Current backend being benchmarked
        avg_time: Average execution time for the current backend
        cpu_time: CPU execution time for comparison
        algorithm_name: Name of the algorithm (for logging)
    """
    results["times"][backend].append(avg_time)

    # Calculate speedup relative to CPU
    if backend != "cpu" and cpu_time > 0 and avg_time != float("inf"):
        speedup = cpu_time / avg_time
        results["speedup"][backend].append(speedup)
        logging.info(
            f"  {algorithm_name} {backend}: {avg_time:.4f} seconds (speedup: {speedup:.2f}x)"
        )
    else:
        results["speedup"][backend].append(1.0)  # CPU speedup is 1.0
        logging.info(f"  {algorithm_name} {backend}: {avg_time:.4f} seconds")


def _benchmark_algorithm(
    data: np.ndarray,
    backends: List[str],
    repetitions: int,
    algorithm_func: Callable,
    algorithm_name: str,
    results: Dict,
    **kwargs,
) -> float:
    """Benchmark a clustering algorithm across different backends.

    Args:
        data: Input data for clustering
        backends: List of backends to benchmark
        repetitions: Number of times to repeat each test for averaging
        algorithm_func: Function to run the clustering algorithm
        algorithm_name: Name of the algorithm (for logging)
        results: Results dictionary to update
        **kwargs: Additional arguments to pass to algorithm_func

    Returns:
        float: CPU execution time for speedup calculations
    """
    cpu_time = 0

    for backend in backends:
        times = []

        for _ in range(repetitions):
            try:
                start_time = time.time()
                algorithm_func(data, backend=backend, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logging.warning(
                    f"{algorithm_name} clustering failed with backend {backend}: {e}"
                )
                times.append(float("inf"))

        avg_time = _calculate_average_time(times)

        if avg_time != float("inf"):
            # Store CPU time for speedup calculations
            if backend == "cpu":
                cpu_time = avg_time

            _process_benchmark_results(
                results, backend, avg_time, cpu_time, algorithm_name
            )
        else:
            _handle_failed_benchmark(results, backend, f"  {algorithm_name} ")

    return cpu_time


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

    # Initialize results dictionaries
    kmeans_results = _initialize_clustering_results("kmeans", backends, data_sizes)
    dbscan_results = _initialize_clustering_results("dbscan", backends, data_sizes)

    # Run benchmarks for each data size
    for size in data_sizes:
        logging.info(f"Benchmarking clustering algorithms on data size {size}...")

        # Create random data points
        data = rng.random((size, 2))

        # Benchmark K-means clustering
        _benchmark_algorithm(
            data=data,
            backends=backends,
            repetitions=repetitions,
            algorithm_func=_run_kmeans_clustering,
            algorithm_name="K-means",
            results=kmeans_results,
            n_clusters=n_clusters,
        )

        # Benchmark DBSCAN clustering
        _benchmark_algorithm(
            data=data,
            backends=backends,
            repetitions=repetitions,
            algorithm_func=_run_dbscan_clustering,
            algorithm_name="DBSCAN",
            results=dbscan_results,
        )

    return {"kmeans": kmeans_results, "dbscan": dbscan_results}


def _handle_failed_benchmark(results: Dict, backend: str, algorithm_name: str) -> None:
    """Handle the case when a benchmark fails.

    Args:
        results: Results dictionary to update
        backend: Current backend being benchmarked
        algorithm_name: Name of the algorithm (for logging)
    """
    results["times"][backend].append(float("inf"))
    results["speedup"][backend].append(0.0)
    logging.info(f"{algorithm_name}{backend}: Failed")


def _run_terrain_generation(size: int, backend: str) -> None:
    """Run terrain generation using the specified backend.

    Args:
        size: Grid size for terrain generation
        backend: Backend to use ("cpu" or GPU backend)
    """
    if backend == "cpu":
        # Use CPU implementation for terrain generation
        generate_terrain_cpu(size, size)
    else:
        # Use GPU implementation for terrain generation
        generate_terrain_gpu(size, size, backend=backend)


def _run_resource_distribution(size: int, backend: str) -> None:
    """Run resource distribution using the specified backend.

    Args:
        size: Grid size for resource distribution
        backend: Backend to use ("cpu" or GPU backend)
    """
    if backend == "cpu":
        # Use CPU implementation for resource distribution
        generate_resource_distribution_cpu(size, size)
    else:
        # Use GPU implementation for resource distribution
        generate_resource_distribution_gpu(size, size, backend=backend)


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

    # Initialize results dictionaries
    terrain_results = _initialize_clustering_results("terrain", backends, grid_sizes)
    resource_results = _initialize_clustering_results("resource", backends, grid_sizes)

    # Override operation names to be more specific
    terrain_results["operation"] = "terrain_generation"
    resource_results["operation"] = "resource_distribution"

    # Run benchmarks for each grid size
    for size in grid_sizes:
        logging.info(f"Benchmarking value generation on grid size {size}x{size}...")

        # Benchmark terrain generation
        _benchmark_algorithm(
            data=size,  # Just passing the size as data
            backends=backends,
            repetitions=repetitions,
            algorithm_func=_run_terrain_generation,
            algorithm_name="Terrain",
            results=terrain_results,
        )

        # Benchmark resource distribution
        _benchmark_algorithm(
            data=size,  # Just passing the size as data
            backends=backends,
            repetitions=repetitions,
            algorithm_func=_run_resource_distribution,
            algorithm_name="Resources",
            results=resource_results,
        )

    return {"terrain": terrain_results, "resources": resource_results}


# This function is replaced by _handle_failed_benchmark


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

    # Initialize results dictionaries
    h2d_results = _initialize_bandwidth_results(
        "host_to_device_transfer", backends, data_sizes
    )
    d2h_results = _initialize_bandwidth_results(
        "device_to_host_transfer", backends, data_sizes
    )

    # Run benchmarks for each data size
    for size in data_sizes:
        logging.info(f"Benchmarking memory transfer for data size {size} elements...")

        # Create random data array
        data = rng.random(size).astype(np.float32)
        data_size_bytes = data.nbytes

        # Benchmark each backend
        for backend in backends:
            # Host to device transfer benchmarking
            h2d_times = [
                _run_host_to_device_transfer(data, backend) for _ in range(repetitions)
            ]
            _process_bandwidth_results(
                h2d_results, backend, h2d_times, data_size_bytes, "H2D"
            )

            # Device to host transfer benchmarking
            d2h_times = [
                _run_device_to_host_transfer(size, backend) for _ in range(repetitions)
            ]
            _process_bandwidth_results(
                d2h_results, backend, d2h_times, data_size_bytes, "D2H"
            )

    return {"host_to_device": h2d_results, "device_to_host": d2h_results}


def _initialize_bandwidth_results(
    operation_name: str, backends: List[str], data_sizes: List[int]
) -> Dict:
    """Initialize results dictionary for bandwidth benchmarks.

    Args:
        operation_name: Name of the transfer operation
        backends: List of backends to benchmark
        data_sizes: List of data sizes to test

    Returns:
        Dict: Initialized results dictionary
    """
    return {
        "operation": operation_name,
        "data_sizes": data_sizes,
        "times": {backend: [] for backend in backends},
        "bandwidth": {backend: [] for backend in backends},
    }


def _run_host_to_device_transfer(data: np.ndarray, backend: str) -> float:
    """Run host to device memory transfer using the specified backend.

    Args:
        data: Input data to transfer
        backend: Backend to use ("cuda", "cupy", or "mps")

    Returns:
        float: Transfer time in seconds or float("inf") if failed
    """
    try:
        if backend in ["cuda", "cupy"] and CUPY_AVAILABLE:
            # Ensure we're starting from CPU memory
            cpu_data = np.array(data)
            # Time the transfer
            start_time = time.time()
            _ = cp.array(
                cpu_data
            )  # We don't need to store the result, just measure time
            # Force synchronization to ensure transfer is complete
            cp.cuda.stream.get_current_stream().synchronize()
            end_time = time.time()
        elif backend == "mps" and TORCH_AVAILABLE:
            # Ensure we're starting from CPU memory
            cpu_data = torch.tensor(data)
            # Time the transfer
            start_time = time.time()
            _ = cpu_data.to(
                "mps"
            )  # We don't need to store the result, just measure time
            # Force synchronization
            torch.mps.synchronize()
            end_time = time.time()
        else:
            # Unknown backend
            raise ValueError(f"Unsupported backend for memory transfer: {backend}")

        return end_time - start_time
    except Exception as e:
        logging.warning(f"Host to device transfer failed with backend {backend}: {e}")
        return float("inf")


def _run_device_to_host_transfer(size: int, backend: str) -> float:
    """Run device to host memory transfer using the specified backend.

    Args:
        size: Size of data to transfer
        backend: Backend to use ("cuda", "cupy", or "mps")

    Returns:
        float: Transfer time in seconds or float("inf") if failed
    """
    try:
        if backend in ["cuda", "cupy"] and CUPY_AVAILABLE:
            # Create data on GPU
            gpu_data = cp.random.random(size).astype(cp.float32)
            # Time the transfer
            start_time = time.time()
            _ = cp.asnumpy(
                gpu_data
            )  # We don't need to store the result, just measure time
            end_time = time.time()
        elif backend == "mps" and TORCH_AVAILABLE:
            # Create data on GPU
            gpu_data = torch.rand(size, device="mps")
            # Time the transfer
            start_time = time.time()
            _ = gpu_data.to(
                "cpu"
            )  # We don't need to store the result, just measure time
            # Force synchronization
            torch.mps.synchronize()
            end_time = time.time()
        else:
            # Unknown backend
            raise ValueError(f"Unsupported backend for memory transfer: {backend}")

        return end_time - start_time
    except Exception as e:
        logging.warning(f"Device to host transfer failed with backend {backend}: {e}")
        return float("inf")


def _process_bandwidth_results(
    results: Dict,
    backend: str,
    times: List[float],
    data_size_bytes: int,
    direction_prefix: str,
) -> None:
    """Process bandwidth benchmark results and calculate bandwidth.

    Args:
        results: Results dictionary to update
        backend: Current backend being benchmarked
        times: List of execution times
        data_size_bytes: Size of data in bytes
        direction_prefix: Prefix string for logging the direction (H2D or D2H)
    """
    if times and any(t != float("inf") for t in times):
        avg_time = _calculate_average_time(times)
        results["times"][backend].append(avg_time)

        # Calculate bandwidth in GB/s
        if avg_time > 0:
            bandwidth = (data_size_bytes / (1024**3)) / avg_time  # GB/s
            results["bandwidth"][backend].append(bandwidth)
            logging.info(
                f"  {direction_prefix} {backend}: {avg_time:.6f} seconds ({bandwidth:.2f} GB/s)"
            )
        else:
            results["bandwidth"][backend].append(0.0)
            logging.info(f"  {direction_prefix} {backend}: {avg_time:.6f} seconds")
    else:
        _handle_bandwidth_benchmark_failure(results, backend, f"  {direction_prefix} ")


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


def _initialize_noise_benchmark_results(backends, grid_sizes):
    """Initialize the results dictionary for noise benchmark.

    Args:
        backends: List of available backends
        grid_sizes: List of grid sizes to test

    Returns:
        Dict: Initialized results dictionary
    """
    return {
        "operation": "noise_generation",
        "grid_sizes": grid_sizes,
        "times": {backend: [] for backend in backends},
        "memory": {backend: [] for backend in backends},
        "speedup": {backend: [] for backend in backends},
    }


def _run_noise_benchmark_iteration(backend, size, octaves):
    """Run a single noise benchmark iteration.

    Args:
        backend: The backend to use (cpu or gpu backend name)
        size: Grid size
        octaves: Number of octaves for noise generation

    Returns:
        float: Time taken for the operation
    """
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

    return time.time() - start_time


def _record_benchmark_results(results, backend, avg_time, cpu_time):
    """Record benchmark results and calculate speedup.

    Args:
        results: Results dictionary
        backend: Current backend
        avg_time: Average time for the current backend
        cpu_time: CPU time for speedup calculation

    Returns:
        float: CPU time (unchanged if backend is not CPU, updated if it is)
    """
    # Calculate and log speedup relative to CPU
    if backend != "cpu" and cpu_time > 0:
        speedup = cpu_time / avg_time
        results["speedup"][backend].append(speedup)
        logging.info(f"  {backend}: {avg_time:.4f} seconds (speedup: {speedup:.2f}x)")
    else:
        results["speedup"][backend].append(1.0)  # CPU speedup is 1.0
        logging.info(f"  {backend}: {avg_time:.4f} seconds")

    # Return updated CPU time if this is the CPU backend, otherwise return unchanged
    return avg_time if backend == "cpu" else cpu_time


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
    results = _initialize_noise_benchmark_results(backends, grid_sizes)

    # Run benchmarks for each grid size
    for size in grid_sizes:
        logging.info(f"Benchmarking noise generation on grid size {size}x{size}...")
        cpu_time = 0  # Track CPU time for speedup calculations

        # Benchmark each backend
        for backend in backends:
            # Run multiple iterations and collect times
            times = [
                _run_noise_benchmark_iteration(backend, size, octaves)
                for _ in range(repetitions)
            ]

            # Record average time
            avg_time = sum(times) / len(times)
            results["times"][backend].append(avg_time)

            # Record results and update CPU time if needed
            cpu_time = _record_benchmark_results(results, backend, avg_time, cpu_time)

    return results
