#!/usr/bin/env python3
"""
Benchmark script for GPU acceleration in Space Muck.

This script measures the performance of GPU-accelerated operations
compared to their CPU counterparts across different grid sizes and configurations.
"""

# Standard library imports
import argparse
import time

# Local application imports
from typing import Dict, List

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np

from utils.cellular_automaton_utils import apply_cellular_automaton
from utils.gpu_utils import (  # Import utilities
    apply_cellular_automaton_gpu,
    apply_noise_generation_gpu,
    get_available_backends,
    is_gpu_available,
)


def benchmark_cellular_automaton(
    grid_sizes: List[int], iterations: int = 5, repetitions: int = 3
) -> Dict[str, List[float]]:
    """
    Benchmark cellular automaton performance across different backends.

    Args:
        grid_sizes: List of grid sizes to test
        iterations: Number of cellular automaton iterations
        repetitions: Number of times to repeat each test for averaging

    Returns:
        Dict[str, List[float]]: Dictionary mapping backend names to execution times
    """
    # Get available backends
    backends = get_available_backends()
    if "cpu" not in backends:
        backends.append("cpu")

    # Initialize results dictionary
    results = {backend: [] for backend in backends}

    # Run benchmarks for each grid size
    for size in grid_sizes:
        print(f"Benchmarking grid size {size}x{size}...")

        # Create a random grid using the newer Generator API with a fixed seed for reproducibility
        rng = np.random.default_rng(
            seed=42
        )  # Using a fixed seed for consistent benchmarks
        grid = rng.integers(0, 2, (size, size), dtype=np.int8)

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
            results[backend].append(avg_time)
            print(f"  {backend}: {avg_time:.4f} seconds")

    return results


def benchmark_noise_generation(
    grid_sizes: List[int], repetitions: int = 3
) -> Dict[str, List[float]]:
    """
    Benchmark noise generation performance across different backends.

    Args:
        grid_sizes: List of grid sizes to test
        repetitions: Number of times to repeat each test for averaging

    Returns:
        Dict[str, List[float]]: Dictionary mapping backend names to execution times
    """
    # Get available backends
    backends = get_available_backends()
    if "cpu" not in backends:
        backends.append("cpu")

    # Initialize results dictionary
    results = {backend: [] for backend in backends}

    # Run benchmarks for each grid size
    for size in grid_sizes:
        print(f"Benchmarking noise generation for size {size}x{size}...")

        # Benchmark each backend
        for backend in backends:
            times = []

            for _ in range(repetitions):
                start_time = time.time()
                apply_noise_generation_gpu(size, size, backend=backend)
                end_time = time.time()

                times.append(end_time - start_time)

            # Record average time
            avg_time = sum(times) / len(times)
            results[backend].append(avg_time)
            print(f"  {backend}: {avg_time:.4f} seconds")

    return results


def plot_results(
    grid_sizes: List[int],
    ca_results: Dict[str, List[float]],
    noise_results: Dict[str, List[float]],
    output_file: str = "gpu_benchmark_results.png",
) -> None:
    """
    Plot benchmark results.

    Args:
        grid_sizes: List of grid sizes tested
        ca_results: Cellular automaton benchmark results
        noise_results: Noise generation benchmark results
        output_file: Output file path for the plot
    """
    # Create subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    _plot_results(ca_results, ax1, grid_sizes, "Cellular Automaton Performance")
    _plot_results(noise_results, ax2, grid_sizes, "Noise Generation Performance")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Results saved to {output_file}")


def _plot_results(results: Dict[str, List[float]], ax, grid_sizes, title: str):
    # Plot cellular automaton results
    for backend, times in results.items():
        ax.plot(grid_sizes, times, marker="o", label=backend)

    ax.set_title(title)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()


def main():
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description="Benchmark GPU acceleration")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024],
        help="Grid sizes to benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of cellular automaton iterations",
    )
    parser.add_argument(
        "--repetitions", type=int, default=3, help="Number of repetitions for each test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpu_benchmark_results.png",
        help="Output file for benchmark results plot",
    )
    args = parser.parse_args()

    print("GPU Acceleration Benchmark")
    print("=========================")
    print(f"Available backends: {get_available_backends()}")
    print(f"GPU available: {is_gpu_available()}")
    print()

    # Run benchmarks
    ca_results = benchmark_cellular_automaton(
        args.sizes, iterations=args.iterations, repetitions=args.repetitions
    )

    noise_results = benchmark_noise_generation(args.sizes, repetitions=args.repetitions)

    # Plot results
    plot_results(args.sizes, ca_results, noise_results, args.output)

    # Print summary
    print("\nPerformance Summary:")
    print("===================")

    # Calculate speedup for largest grid size
    largest_size = args.sizes[-1]
    size_index = args.sizes.index(largest_size)

    cpu_ca_time = ca_results.get("cpu", [0])[size_index]
    cpu_noise_time = noise_results.get("cpu", [0])[size_index]

    for backend in get_available_backends():
        if backend == "cpu":
            continue

        ca_time = ca_results.get(backend, [0])[size_index]
        noise_time = noise_results.get(backend, [0])[size_index]

        if cpu_ca_time > 0:
            ca_speedup = cpu_ca_time / ca_time
            print(f"{backend} cellular automaton speedup: {ca_speedup:.2f}x")

        if cpu_noise_time > 0:
            noise_speedup = cpu_noise_time / noise_time
            print(f"{backend} noise generation speedup: {noise_speedup:.2f}x")


if __name__ == "__main__":
    main()
