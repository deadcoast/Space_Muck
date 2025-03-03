#!/usr/bin/env python3
"""
Comprehensive GPU Benchmarking Script for Space Muck.

This script provides a unified framework for benchmarking all GPU-accelerated operations
across different backends, grid sizes, and configurations. It generates detailed
performance metrics and visualizations to help optimize GPU acceleration usage.
"""

import os
import sys
import time
import logging
import argparse
import platform
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GPU utilities
from src.utils.gpu_utils import (
    is_gpu_available,
    get_available_backends,
    to_gpu,
    to_cpu,
    apply_cellular_automaton_gpu,
    apply_noise_generation_gpu,
    apply_kmeans_clustering_gpu,
    apply_dbscan_clustering_gpu,
    CUDA_AVAILABLE,
    CUPY_AVAILABLE,
    MPS_AVAILABLE,
    METALGPU_AVAILABLE,
)

# Import CPU implementations for comparison
from src.utils.cellular_automaton_utils import apply_cellular_automaton
from src.utils.value_generator import (
    generate_value_distribution,
    add_value_clusters,
)
from src.utils.value_generator_gpu import (
    generate_value_distribution_gpu,
    add_value_clusters_gpu,
)

# Import BaseGenerator for integration testing
from src.entities.base_generator import BaseGenerator


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
                    result = apply_cellular_automaton(grid, iterations=iterations)
                else:
                    result = apply_cellular_automaton_gpu(
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
                kmeans_results["times"][backend].append(float("inf"))
                kmeans_results["speedup"][backend].append(0.0)
                logging.info(f"  K-means {backend}: Failed")

            # DBSCAN benchmarking
            dbscan_times = []
            for _ in range(repetitions):
                try:
                    start_time = time.time()

                    if backend == "cpu":
                        # Use sklearn for CPU implementation
                        from sklearn.cluster import DBSCAN

                        dbscan = DBSCAN(eps=0.3, min_samples=5)
                        labels = dbscan.fit_predict(data)
                    else:
                        # Use GPU implementation
                        labels = apply_dbscan_clustering_gpu(
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
                dbscan_results["times"][backend].append(float("inf"))
                dbscan_results["speedup"][backend].append(0.0)
                logging.info(f"  DBSCAN {backend}: Failed")

    return {"kmeans": kmeans_results, "dbscan": dbscan_results}


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
                        terrain = generate_terrain_cpu(size, size)
                    else:
                        # Use GPU implementation for terrain generation
                        terrain = generate_terrain_gpu(size, size, backend=backend)

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
                terrain_results["times"][backend].append(float("inf"))
                terrain_results["speedup"][backend].append(0.0)
                logging.info(f"  Terrain {backend}: Failed")

            # Resource distribution benchmarking
            resource_times = []
            for _ in range(repetitions):
                try:
                    start_time = time.time()

                    if backend == "cpu":
                        # Use CPU implementation for resource distribution
                        resources = generate_resource_distribution_cpu(size, size)
                    else:
                        # Use GPU implementation for resource distribution
                        resources = generate_resource_distribution_gpu(
                            size, size, backend=backend
                        )

                    end_time = time.time()
                    resource_times.append(end_time - start_time)
                except Exception as e:
                    logging.warning(
                        f"Resource distribution failed with backend {backend}: {e}"
                    )
                    resource_times.append(float("inf"))

            # Record average time for resource distribution
            if resource_times and not all(t == float("inf") for t in resource_times):
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
                resource_results["times"][backend].append(float("inf"))
                resource_results["speedup"][backend].append(0.0)
                logging.info(f"  Resources {backend}: Failed")

    return {"terrain": terrain_results, "resources": resource_results}


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
                    if backend in ["cuda", "cupy"]:
                        import cupy as cp

                        # Ensure we're starting from CPU memory
                        cpu_data = np.array(data)
                        # Time the transfer
                        start_time = time.time()
                        gpu_data = cp.array(cpu_data)
                        # Force synchronization to ensure transfer is complete
                        cp.cuda.stream.get_current_stream().synchronize()
                        end_time = time.time()
                    elif backend == "mps":
                        import torch

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
                h2d_results["times"][backend].append(float("inf"))
                h2d_results["bandwidth"][backend].append(0.0)
                logging.info(f"  H2D {backend}: Failed")

            # Device to host transfer benchmarking
            d2h_times = []
            for _ in range(repetitions):
                try:
                    if backend in ["cuda", "cupy"]:
                        import cupy as cp

                        # Create data on GPU
                        gpu_data = cp.random.random(size).astype(cp.float32)
                        # Time the transfer
                        start_time = time.time()
                        cpu_data = cp.asnumpy(gpu_data)
                        end_time = time.time()
                    elif backend == "mps":
                        import torch

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
                d2h_results["times"][backend].append(float("inf"))
                d2h_results["bandwidth"][backend].append(0.0)
                logging.info(f"  D2H {backend}: Failed")

    return {"host_to_device": h2d_results, "device_to_host": d2h_results}


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
    if "grid_sizes" in op_results:
        x_values = op_results["grid_sizes"]
        x_label = "Grid Size"
    elif "data_sizes" in op_results:
        x_values = op_results["data_sizes"]
        x_label = "Data Size"
    else:
        logging.warning(
            f"No size information found for {operation}, skipping visualization"
        )
        return

    # Create execution time plot
    if "times" in op_results:
        plt.figure(figsize=(12, 8))
        for backend, times in op_results["times"].items():
            if times and any(t != float("inf") for t in times):
                valid_indices = [i for i, t in enumerate(times) if t != float("inf")]
                valid_x = [x_values[i] for i in valid_indices]
                valid_times = [times[i] for i in valid_indices]
                plt.plot(valid_x, valid_times, marker="o", label=f"{backend}")

        plt.title(f"{title} - Execution Time")
        plt.xlabel(x_label)
        plt.ylabel("Time (seconds)")
        plt.grid(True)
        plt.legend()
        (
            plt.xscale("log")
            if len(x_values) > 1 and max(x_values) / min(x_values) > 100
            else None
        )
        (
            plt.yscale("log")
            if op_results["times"]
            and any(
                len(times) > 0
                and max(t for t in times if t != float("inf"))
                / min(t for t in times if t != float("inf") and t > 0)
                > 100
                for times in op_results["times"].values()
                if times
            )
            else None
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{operation}_time.png"))
        plt.close()

    # Create speedup plot
    if "speedup" in op_results:
        plt.figure(figsize=(12, 8))
        for backend, speedups in op_results["speedup"].items():
            if backend != "cpu" and speedups and any(s != 0 for s in speedups):
                valid_indices = [i for i, s in enumerate(speedups) if s > 0]
                valid_x = [x_values[i] for i in valid_indices]
                valid_speedups = [speedups[i] for i in valid_indices]
                plt.plot(valid_x, valid_speedups, marker="o", label=f"{backend}")

        plt.title(f"{title} - Speedup vs CPU")
        plt.xlabel(x_label)
        plt.ylabel("Speedup Factor (x)")
        plt.grid(True)
        plt.legend()
        (
            plt.xscale("log")
            if len(x_values) > 1 and max(x_values) / min(x_values) > 100
            else None
        )
        plt.axhline(y=1, color="r", linestyle="--", label="CPU Baseline")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{operation}_speedup.png"))
        plt.close()

    # Create bandwidth plot for memory transfer benchmarks
    if "bandwidth" in op_results:
        plt.figure(figsize=(12, 8))
        for backend, bandwidths in op_results["bandwidth"].items():
            if bandwidths and any(b != 0 for b in bandwidths):
                valid_indices = [i for i, b in enumerate(bandwidths) if b > 0]
                valid_x = [x_values[i] for i in valid_indices]
                valid_bandwidths = [bandwidths[i] for i in valid_indices]
                plt.plot(valid_x, valid_bandwidths, marker="o", label=f"{backend}")

        plt.title(f"{title} - Memory Bandwidth")
        plt.xlabel(x_label)
        plt.ylabel("Bandwidth (GB/s)")
        plt.grid(True)
        plt.legend()
        (
            plt.xscale("log")
            if len(x_values) > 1 and max(x_values) / min(x_values) > 100
            else None
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{operation}_bandwidth.png"))
        plt.close()


def run_all_benchmarks(output_dir: str = "./benchmark_results", small_grid_sizes: List[int] = None, large_data_sizes: List[int] = None, repetitions: int = 3) -> None:
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
                    result = generator.generate_noise_layer(scale=0.1)
                else:
                    # Use apply_noise_generation_gpu directly
                    result = apply_noise_generation_gpu(
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
