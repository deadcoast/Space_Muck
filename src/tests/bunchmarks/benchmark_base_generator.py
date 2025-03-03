#!/usr/bin/env python3
"""
Benchmark script for BaseGenerator performance testing.

This script measures the performance of the BaseGenerator's refactored methods
and compares them with the original implementations.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Callable
import logging

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the classes to benchmark
from entities.base_generator import BaseGenerator
from utils.noise_generator import NoiseGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def time_function(func: Callable, *args, **kwargs) -> float:
    """
    Measure the execution time of a function.

    Args:
        func: Function to time
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        float: Execution time in seconds
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time


def benchmark_cellular_automaton(
    generator: BaseGenerator, grid_sizes: List[int], iterations: int = 3
) -> Dict[str, List[float]]:
    """
    Benchmark the apply_cellular_automaton method.

    Args:
        generator: BaseGenerator instance
        grid_sizes: List of grid sizes to test
        iterations: Number of cellular automaton iterations

    Returns:
        Dict[str, List[float]]: Dictionary of execution times
    """
    results = {"grid_sizes": grid_sizes, "times": []}

    for size in grid_sizes:
        # Create a random grid
        grid = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])

        # Time the cellular automaton operation
        execution_time = time_function(
            generator.apply_cellular_automaton,
            grid=grid,
            birth_set={3},
            survival_set={2, 3},
            iterations=iterations,
            wrap=True,
        )

        results["times"].append(execution_time)
        logging.info(f"Grid size {size}x{size}: {execution_time:.4f} seconds")

    return results


def benchmark_clustering(
    generator: BaseGenerator, grid_sizes: List[int], num_clusters: int = 5
) -> Dict[str, List[float]]:
    """
    Benchmark the create_clusters method.

    Args:
        generator: BaseGenerator instance
        grid_sizes: List of grid sizes to test
        num_clusters: Number of clusters to create

    Returns:
        Dict[str, List[float]]: Dictionary of execution times
    """
    results = {"grid_sizes": grid_sizes, "times": []}

    for size in grid_sizes:
        # Create a random grid
        grid = np.random.random((size, size))

        # Time the clustering operation
        execution_time = time_function(
            generator.create_clusters,
            grid=grid,
            num_clusters=num_clusters,
            cluster_value_multiplier=2.0,
        )

        results["times"].append(execution_time)
        logging.info(f"Grid size {size}x{size}: {execution_time:.4f} seconds")

    return results


def print_results(
    results_dict: Dict[str, Dict[str, List[float]]], title: str, ylabel: str
):
    """
    Print benchmark results in a table format.

    Args:
        results_dict: Dictionary of benchmark results
        title: Table title
        ylabel: Y-axis label description
    """
    print(f"\n{title}")
    print("-" * 60)

    # Print header
    header = "Grid Size"
    for label in results_dict.keys():
        header += f" | {label}"
    print(header)
    print("-" * 60)

    # Get the first results to determine grid sizes
    first_key = list(results_dict.keys())[0]
    grid_sizes = results_dict[first_key]["grid_sizes"]

    # Print each row
    for i, size in enumerate(grid_sizes):
        row = f"{size:9d}"
        for label, results in results_dict.items():
            row += f" | {results['times'][i]:.6f}s"
        print(row)

    print("-" * 60)


def main():
    """Run the benchmarks."""
    # Create a generator for benchmarking
    generator = BaseGenerator(entity_id="benchmark", width=100, height=100)

    # Define grid sizes to test
    grid_sizes = [10, 20, 50, 100, 200]

    # Benchmark cellular automaton
    logging.info("Benchmarking apply_cellular_automaton...")
    ca_results = benchmark_cellular_automaton(generator, grid_sizes)

    # Benchmark clustering
    logging.info("Benchmarking create_clusters...")
    cluster_results = benchmark_clustering(generator, grid_sizes)

    # Print results
    print_results(
        {"Cellular Automaton": ca_results},
        "Cellular Automaton Performance",
        "Execution Time (seconds)",
    )

    print_results(
        {"Clustering": cluster_results},
        "Clustering Performance",
        "Execution Time (seconds)",
    )


if __name__ == "__main__":
    main()
