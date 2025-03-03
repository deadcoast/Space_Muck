#!/usr/bin/env python3
"""
Comprehensive benchmark script for procedural generation in Space Muck.

This script measures the performance of various procedural generation components
across different grid sizes, configurations, and hardware acceleration options.
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any
import logging
import multiprocessing

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the classes to benchmark
from entities.base_generator import BaseGenerator
from utils.noise_generator import get_noise_generator
from utils.gpu_utils import (
    is_gpu_available,
    get_available_backends,
    apply_cellular_automaton_gpu,
    apply_noise_generation_gpu,
)
from utils.cellular_automaton_utils import apply_cellular_automaton

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def time_function(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    """
    Measure the execution time of a function.

    Args:
        func: Function to time
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple[float, Any]: Execution time in seconds and function result
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def benchmark_complete_generation(
    generator_class: type,
    grid_sizes: List[int],
    repetitions: int = 3,
    use_gpu: bool = True,
) -> Dict[str, List[float]]:
    """
    Benchmark the complete generation process.

    Args:
        generator_class: Generator class to benchmark
        grid_sizes: List of grid sizes to test
        repetitions: Number of times to repeat each test for averaging
        use_gpu: Whether to use GPU acceleration if available

    Returns:
        Dict[str, List[float]]: Dictionary of execution times for different phases
    """
    results = {
        "grid_sizes": grid_sizes,
        "noise_generation": [],
        "cellular_automaton": [],
        "clustering": [],
        "total": [],
    }

    for size in grid_sizes:
        logging.info(f"Benchmarking grid size {size}x{size}...")
        
        noise_times = []
        ca_times = []
        cluster_times = []
        total_times = []
        
        for _ in range(repetitions):
            # Create a generator with the specified size
            generator = generator_class(width=size, height=size)
            
            # Benchmark the complete generation process
            start_total = time.time()
            
            # Noise generation
            start_time = time.time()
            noise_grid = generator.generate_noise_layer(noise_type="medium", scale=0.1)
            noise_time = time.time() - start_time
            noise_times.append(noise_time)
            
            # Cellular automaton
            start_time = time.time()
            ca_grid = generator.apply_cellular_automaton(
                grid=noise_grid > 0.5,
                birth_set={3},
                survival_set={2, 3},
                iterations=3,
                wrap=True,
            )
            ca_time = time.time() - start_time
            ca_times.append(ca_time)
            
            # Clustering
            start_time = time.time()
            final_grid = generator.create_clusters(
                grid=ca_grid,
                num_clusters=5,
                cluster_value_multiplier=2.0,
            )
            cluster_time = time.time() - start_time
            cluster_times.append(cluster_time)
            
            total_time = time.time() - start_total
            total_times.append(total_time)
        
        # Record average times
        results["noise_generation"].append(sum(noise_times) / repetitions)
        results["cellular_automaton"].append(sum(ca_times) / repetitions)
        results["clustering"].append(sum(cluster_times) / repetitions)
        results["total"].append(sum(total_times) / repetitions)
        
        logging.info(f"  Noise Generation: {results['noise_generation'][-1]:.4f} seconds")
        logging.info(f"  Cellular Automaton: {results['cellular_automaton'][-1]:.4f} seconds")
        logging.info(f"  Clustering: {results['clustering'][-1]:.4f} seconds")
        logging.info(f"  Total: {results['total'][-1]:.4f} seconds")
    
    return results


def benchmark_noise_generation(
    grid_sizes: List[int],
    repetitions: int = 3,
    backends: List[str] = None,
) -> Dict[str, List[float]]:
    """
    Benchmark noise generation across different backends.

    Args:
        grid_sizes: List of grid sizes to test
        repetitions: Number of times to repeat each test for averaging
        backends: List of backends to test (None for all available)

    Returns:
        Dict[str, List[float]]: Dictionary mapping backend names to execution times
    """
    # Get available backends if not specified
    if backends is None:
        backends = get_available_backends()
        if "cpu" not in backends:
            backends.append("cpu")
    
    # Initialize results dictionary
    results = {backend: [] for backend in backends}
    results["grid_sizes"] = grid_sizes
    
    # Run benchmarks for each grid size
    for size in grid_sizes:
        logging.info(f"Benchmarking noise generation for size {size}x{size}...")
        
        # Benchmark each backend
        for backend in backends:
            times = []
            
            for _ in range(repetitions):
                start_time = time.time()
                if backend == "cpu":
                    # Use standard noise generator
                    noise_gen = get_noise_generator()
                    noise_gen.generate_noise(size, size, scale=0.1, octaves=5)
                else:
                    # Use GPU-accelerated noise generation
                    apply_noise_generation_gpu(size, size, scale=0.1, octaves=5, backend=backend)
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Record average time
            avg_time = sum(times) / len(times)
            results[backend].append(avg_time)
            logging.info(f"  {backend}: {avg_time:.4f} seconds")
    
    return results


def benchmark_cellular_automaton(
    grid_sizes: List[int],
    repetitions: int = 3,
    backends: List[str] = None,
) -> Dict[str, List[float]]:
    """
    Benchmark cellular automaton across different backends.

    Args:
        grid_sizes: List of grid sizes to test
        repetitions: Number of times to repeat each test for averaging
        backends: List of backends to test (None for all available)

    Returns:
        Dict[str, List[float]]: Dictionary mapping backend names to execution times
    """
    # Get available backends if not specified
    if backends is None:
        backends = get_available_backends()
        if "cpu" not in backends:
            backends.append("cpu")
    
    # Initialize results dictionary
    results = {backend: [] for backend in backends}
    results["grid_sizes"] = grid_sizes
    
    # Run benchmarks for each grid size
    for size in grid_sizes:
        logging.info(f"Benchmarking cellular automaton for size {size}x{size}...")
        
        # Create a random grid
        grid = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
        
        # Benchmark each backend
        for backend in backends:
            times = []
            
            for _ in range(repetitions):
                if backend == "cpu":
                    # Benchmark CPU implementation
                    start_time = time.time()
                    apply_cellular_automaton(grid, iterations=3)
                    end_time = time.time()
                else:
                    # Benchmark GPU implementation
                    start_time = time.time()
                    apply_cellular_automaton_gpu(grid, backend=backend, iterations=3)
                    end_time = time.time()
                
                times.append(end_time - start_time)
            
            # Record average time
            avg_time = sum(times) / len(times)
            results[backend].append(avg_time)
            logging.info(f"  {backend}: {avg_time:.4f} seconds")
    
    return results


def benchmark_clustering(
    grid_sizes: List[int],
    repetitions: int = 3,
) -> Dict[str, List[float]]:
    """
    Benchmark clustering performance.

    Args:
        grid_sizes: List of grid sizes to test
        repetitions: Number of times to repeat each test for averaging

    Returns:
        Dict[str, List[float]]: Dictionary of execution times
    """
    # Initialize results dictionary
    results = {
        "grid_sizes": grid_sizes,
        "sequential": [],
        "parallel": [],
    }
    
    # Run benchmarks for each grid size
    for size in grid_sizes:
        logging.info(f"Benchmarking clustering for size {size}x{size}...")
        
        # Create a generator with the specified size
        generator = BaseGenerator(width=size, height=size)
        
        # Create a random grid
        grid = np.random.random((size, size))
        
        # Benchmark sequential clustering
        seq_times = []
        for _ in range(repetitions):
            # Set a very high threshold to force sequential processing
            generator._parallel_clustering_threshold = size * size * 10
            start_time = time.time()
            generator.create_clusters(grid, num_clusters=5, cluster_value_multiplier=2.0)
            end_time = time.time()
            seq_times.append(end_time - start_time)
        
        avg_seq_time = sum(seq_times) / len(seq_times)
        results["sequential"].append(avg_seq_time)
        logging.info(f"  Sequential: {avg_seq_time:.4f} seconds")
        
        # Benchmark parallel clustering
        par_times = []
        for _ in range(repetitions):
            # Set a very low threshold to force parallel processing
            generator._parallel_clustering_threshold = 1
            start_time = time.time()
            generator.create_clusters(grid, num_clusters=5, cluster_value_multiplier=2.0)
            end_time = time.time()
            par_times.append(end_time - start_time)
        
        avg_par_time = sum(par_times) / len(par_times)
        results["parallel"].append(avg_par_time)
        logging.info(f"  Parallel: {avg_par_time:.4f} seconds")
    
    return results


def plot_results(
    results: Dict[str, Dict[str, List[float]]],
    output_file: str = "procedural_generation_benchmark.png",
) -> None:
    """
    Plot benchmark results.

    Args:
        results: Dictionary of benchmark results
        output_file: Output file path for the plot
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot complete generation results
    if "complete" in results:
        complete_results = results["complete"]
        grid_sizes = complete_results["grid_sizes"]
        
        axes[0, 0].plot(grid_sizes, complete_results["noise_generation"], marker="o", label="Noise Generation")
        axes[0, 0].plot(grid_sizes, complete_results["cellular_automaton"], marker="s", label="Cellular Automaton")
        axes[0, 0].plot(grid_sizes, complete_results["clustering"], marker="^", label="Clustering")
        axes[0, 0].plot(grid_sizes, complete_results["total"], marker="*", label="Total")
        
        axes[0, 0].set_title("Complete Generation Performance")
        axes[0, 0].set_xlabel("Grid Size")
        axes[0, 0].set_ylabel("Execution Time (seconds)")
        axes[0, 0].set_xscale("log", base=2)
        axes[0, 0].set_yscale("log")
        axes[0, 0].grid(True)
        axes[0, 0].legend()
    
    # Plot noise generation results
    if "noise" in results:
        noise_results = results["noise"]
        grid_sizes = noise_results["grid_sizes"]
        
        for backend, times in noise_results.items():
            if backend != "grid_sizes":
                axes[0, 1].plot(grid_sizes, times, marker="o", label=backend)
        
        axes[0, 1].set_title("Noise Generation Performance")
        axes[0, 1].set_xlabel("Grid Size")
        axes[0, 1].set_ylabel("Execution Time (seconds)")
        axes[0, 1].set_xscale("log", base=2)
        axes[0, 1].set_yscale("log")
        axes[0, 1].grid(True)
        axes[0, 1].legend()
    
    # Plot cellular automaton results
    if "cellular_automaton" in results:
        ca_results = results["cellular_automaton"]
        grid_sizes = ca_results["grid_sizes"]
        
        for backend, times in ca_results.items():
            if backend != "grid_sizes":
                axes[1, 0].plot(grid_sizes, times, marker="o", label=backend)
        
        axes[1, 0].set_title("Cellular Automaton Performance")
        axes[1, 0].set_xlabel("Grid Size")
        axes[1, 0].set_ylabel("Execution Time (seconds)")
        axes[1, 0].set_xscale("log", base=2)
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    
    # Plot clustering results
    if "clustering" in results:
        cluster_results = results["clustering"]
        grid_sizes = cluster_results["grid_sizes"]
        
        axes[1, 1].plot(grid_sizes, cluster_results["sequential"], marker="o", label="Sequential")
        axes[1, 1].plot(grid_sizes, cluster_results["parallel"], marker="s", label="Parallel")
        
        axes[1, 1].set_title("Clustering Performance")
        axes[1, 1].set_xlabel("Grid Size")
        axes[1, 1].set_ylabel("Execution Time (seconds)")
        axes[1, 1].set_xscale("log", base=2)
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    logging.info(f"Results saved to {output_file}")


def main():
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description="Benchmark procedural generation")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="Grid sizes to benchmark",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions for each test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="procedural_generation_benchmark.png",
        help="Output file for benchmark results plot",
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=["complete", "noise", "cellular_automaton", "clustering"],
        help="Components to benchmark",
    )
    args = parser.parse_args()
    
    logging.info("Procedural Generation Benchmark")
    logging.info("==============================")
    logging.info(f"Available backends: {get_available_backends()}")
    logging.info(f"GPU available: {is_gpu_available()}")
    logging.info(f"CPU cores: {multiprocessing.cpu_count()}")
    logging.info("")
    
    # Run benchmarks
    results = {}
    
    if "complete" in args.components:
        logging.info("Benchmarking complete generation process...")
        results["complete"] = benchmark_complete_generation(
            BaseGenerator,
            args.sizes,
            repetitions=args.repetitions,
        )
    
    if "noise" in args.components:
        logging.info("Benchmarking noise generation...")
        results["noise"] = benchmark_noise_generation(
            args.sizes,
            repetitions=args.repetitions,
        )
    
    if "cellular_automaton" in args.components:
        logging.info("Benchmarking cellular automaton...")
        results["cellular_automaton"] = benchmark_cellular_automaton(
            args.sizes,
            repetitions=args.repetitions,
        )
    
    if "clustering" in args.components:
        logging.info("Benchmarking clustering...")
        results["clustering"] = benchmark_clustering(
            args.sizes,
            repetitions=args.repetitions,
        )
    
    # Plot results
    plot_results(results, args.output)


if __name__ == "__main__":
    main()
