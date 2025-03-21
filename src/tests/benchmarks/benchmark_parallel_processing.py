#!/usr/bin/env python3
"""
src/tests/benchmarks/benchmark_parallel_processing.py

Benchmark script for measuring the performance of parallel processing implementations
in the BaseGenerator class across different grid sizes.
"""

# Standard library imports
import logging
import os
import sys
import time
from typing import List, Optional

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np

# Local application imports
from entities.base_generator import BaseGenerator

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the class to benchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BenchmarkResults:
    """Class to store and visualize benchmark results."""

    def __init__(self):
        self.sequential_times = {}
        self.parallel_times = {}
        self.speedup_factors = {}
        self.grid_sizes = []

    def add_result(self, grid_size: int, sequential_time: float, parallel_time: float):
        """Add a benchmark result for a specific grid size."""
        self.grid_sizes.append(grid_size)
        self.sequential_times[grid_size] = sequential_time
        self.parallel_times[grid_size] = parallel_time
        self.speedup_factors[grid_size] = (
            sequential_time / parallel_time if parallel_time > 0 else 0
        )

    def print_results(self):
        """Print the benchmark results in a formatted table."""
        print("\n" + "=" * 80)
        print(
            f"{'Grid Size':^15} | {'Sequential (ms)':^20} | {'Parallel (ms)':^20} | {'Speedup Factor':^15}"
        )
        print("-" * 80)

        for size in sorted(self.grid_sizes):
            seq_time = self.sequential_times[size] * 1000  # Convert to milliseconds
            par_time = self.parallel_times[size] * 1000  # Convert to milliseconds
            speedup = self.speedup_factors[size]

            print(
                f"{size:^15} | {seq_time:^20.2f} | {par_time:^20.2f} | {speedup:^15.2f}"
            )

        print("=" * 80 + "\n")

    def plot_results(self, title: str, output_file: Optional[str] = None):
        """Plot the benchmark results."""
        plt.figure(figsize=(12, 8))

        # Sort grid sizes for proper plotting
        sorted_sizes = sorted(self.grid_sizes)

        # Extract data in sorted order
        seq_times = [
            self.sequential_times[size] * 1000 for size in sorted_sizes
        ]  # Convert to milliseconds
        par_times = [
            self.parallel_times[size] * 1000 for size in sorted_sizes
        ]  # Convert to milliseconds
        speedups = [self.speedup_factors[size] for size in sorted_sizes]

        # Create subplots
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot execution times
        ax1.plot(sorted_sizes, seq_times, "o-", label="Sequential", color="blue")
        ax1.plot(sorted_sizes, par_times, "s-", label="Parallel", color="green")
        self._set_plot_data(
            ax1, "Execution Time (ms)", title, " - Execution Time Comparison"
        )
        ax1.legend()
        ax1.grid(True)

        # Plot speedup factors
        ax2.plot(sorted_sizes, speedups, "D-", color="red")
        ax2.axhline(y=1, color="gray", linestyle="--")  # Add reference line at y=1
        self._set_plot_data(ax2, "Speedup Factor", title, " - Speedup Factor")
        ax2.grid(True)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            logging.info(f"Plot saved to {output_file}")

        plt.show()

    def _set_plot_data(self, ax, y_label, title, suffix):
        ax.set_xlabel("Grid Size")
        ax.set_ylabel(y_label)
        ax.set_title(f"{title}{suffix}")


def benchmark_cellular_automaton(
    generator: BaseGenerator, grid_sizes: List[int], iterations: int = 1, runs: int = 3
) -> BenchmarkResults:
    """
    Benchmark the cellular automaton implementation for different grid sizes.

    Args:
        generator: BaseGenerator instance to benchmark
        grid_sizes: List of grid sizes to benchmark
        iterations: Number of cellular automaton iterations to run
        runs: Number of benchmark runs for each configuration

    Returns:
        BenchmarkResults object with the benchmark results
    """
    results = BenchmarkResults()

    for size in grid_sizes:
        logging.info(f"Benchmarking cellular automaton with grid size {size}x{size}")

        # Create a grid with random values using the newer Generator API
        rng = np.random.default_rng(seed=42)  # Using a fixed seed for reproducibility
        grid = rng.random((size, size))
        grid = (grid > 0.7).astype(
            np.float64
        )  # Convert to binary grid with ~30% filled

        # Benchmark sequential implementation
        sequential_times = []
        for _ in range(runs):
            # Force sequential processing by temporarily setting a very high threshold
            original_threshold = generator._get_parallel_ca_threshold()
            generator._set_parallel_ca_threshold(
                size * size * 10
            )  # Set threshold higher than grid size

            start_time = time.time()
            generator.apply_cellular_automaton(
                grid=grid.copy(),
                birth_set={3},
                survival_set={2, 3},
                iterations=iterations,
                wrap=True,
            )
            end_time = time.time()
            sequential_times.append(end_time - start_time)

            # Restore original threshold
            generator._set_parallel_ca_threshold(original_threshold)

        # Use the median time for more stable results
        sequential_time = sorted(sequential_times)[runs // 2]

        # Benchmark parallel implementation
        parallel_times = []
        for _ in range(runs):
            # Force parallel processing by temporarily setting a very low threshold
            original_threshold = generator._get_parallel_ca_threshold()
            generator._set_parallel_ca_threshold(
                1
            )  # Set threshold to 1 to ensure parallel processing

            start_time = time.time()
            generator.apply_cellular_automaton(
                grid=grid.copy(),
                birth_set={3},
                survival_set={2, 3},
                iterations=iterations,
                wrap=True,
            )
            end_time = time.time()
            parallel_times.append(end_time - start_time)

            # Restore original threshold
            generator._set_parallel_ca_threshold(original_threshold)

        # Use the median time for more stable results
        parallel_time = sorted(parallel_times)[runs // 2]

        # Add result
        results.add_result(size, sequential_time, parallel_time)

        logging.info(
            f"Grid size {size}x{size}: Sequential: {sequential_time:.4f}s, Parallel: {parallel_time:.4f}s, Speedup: {sequential_time / parallel_time:.2f}x"
        )

    return results


def benchmark_clustering(
    generator: BaseGenerator,
    grid_sizes: List[int],
    num_clusters: int = 10,
    runs: int = 3,
) -> BenchmarkResults:
    """
    Benchmark the clustering implementation for different grid sizes.

    Args:
        generator: BaseGenerator instance to benchmark
        grid_sizes: List of grid sizes to benchmark
        num_clusters: Number of clusters to create
        runs: Number of benchmark runs for each configuration

    Returns:
        BenchmarkResults object with the benchmark results
    """
    results = BenchmarkResults()

    for size in grid_sizes:
        logging.info(f"Benchmarking clustering with grid size {size}x{size}")

        # Create a grid with random values using the newer Generator API
        rng = np.random.default_rng(seed=42)  # Using a fixed seed for reproducibility
        grid = rng.random((size, size))
        grid = (grid > 0.5).astype(
            np.float64
        )  # Convert to binary grid with ~50% filled

        # Benchmark sequential implementation
        sequential_times = []
        for _ in range(runs):
            # Force sequential processing by temporarily setting a very high threshold
            original_threshold = generator._get_parallel_clustering_threshold()
            generator._set_parallel_clustering_threshold(
                size * size * 10
            )  # Set threshold higher than grid size

            start_time = time.time()
            generator.create_clusters(
                grid=grid.copy(),
                num_clusters=num_clusters,
                cluster_value_multiplier=2.0,
            )
            end_time = time.time()
            sequential_times.append(end_time - start_time)

            # Restore original threshold
            generator._set_parallel_clustering_threshold(original_threshold)

        # Use the median time for more stable results
        sequential_time = sorted(sequential_times)[runs // 2]

        # Benchmark parallel implementation
        parallel_times = []
        for _ in range(runs):
            # Force parallel processing by temporarily setting a very low threshold
            original_threshold = generator._get_parallel_clustering_threshold()
            generator._set_parallel_clustering_threshold(
                1
            )  # Set threshold to 1 to ensure parallel processing

            start_time = time.time()
            generator.create_clusters(
                grid=grid.copy(),
                num_clusters=num_clusters,
                cluster_value_multiplier=2.0,
            )
            end_time = time.time()
            parallel_times.append(end_time - start_time)

            # Restore original threshold
            generator._set_parallel_clustering_threshold(original_threshold)

        # Use the median time for more stable results
        parallel_time = sorted(parallel_times)[runs // 2]

        # Add result
        results.add_result(size, sequential_time, parallel_time)

        logging.info(
            f"Grid size {size}x{size}: Sequential: {sequential_time:.4f}s, Parallel: {parallel_time:.4f}s, Speedup: {sequential_time / parallel_time:.2f}x"
        )

    return results


def main():
    """Main function to run the benchmarks."""
    # Create a BaseGenerator instance for benchmarking
    generator = BaseGenerator(
        entity_id="benchmark-generator",
        width=500,  # Set to the maximum size we'll benchmark
        height=500,
        seed=42,  # Use fixed seed for reproducibility
    )

    # Add helper methods for controlling thresholds during benchmarking
    def _get_parallel_ca_threshold():
        # Default threshold is 40000 (200x200 grid)
        return getattr(generator, "_parallel_ca_threshold", 40000)

    def _set_parallel_ca_threshold(threshold):
        setattr(generator, "_parallel_ca_threshold", threshold)

    def _get_parallel_clustering_threshold():
        # Default threshold is 40000 (200x200 grid)
        return getattr(generator, "_parallel_clustering_threshold", 40000)

    def _set_parallel_clustering_threshold(threshold):
        setattr(generator, "_parallel_clustering_threshold", threshold)

    # Add these methods to the generator instance
    generator._get_parallel_ca_threshold = _get_parallel_ca_threshold
    generator._set_parallel_ca_threshold = _set_parallel_ca_threshold
    generator._get_parallel_clustering_threshold = _get_parallel_clustering_threshold
    generator._set_parallel_clustering_threshold = _set_parallel_clustering_threshold

    # Define grid sizes to benchmark
    # Start with smaller grids and gradually increase to larger ones
    grid_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    # Benchmark cellular automaton
    ca_results = benchmark_cellular_automaton(generator, grid_sizes)
    ca_results.print_results()
    ca_results.plot_results(
        title="Cellular Automaton Parallel Processing Benchmark",
        output_file="ca_benchmark_results.png",
    )

    # Benchmark clustering
    clustering_results = benchmark_clustering(generator, grid_sizes)
    clustering_results.print_results()
    clustering_results.plot_results(
        title="Clustering Parallel Processing Benchmark",
        output_file="clustering_benchmark_results.png",
    )


if __name__ == "__main__":
    main()
