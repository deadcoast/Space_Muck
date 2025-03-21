#!/usr/bin/env python3
"""
src/tests/benchmarks/benchmark_gpu_noise_generation.py

Benchmark script for testing GPU-accelerated noise generation performance.

This script compares the performance of GPU-accelerated noise generation
versus CPU-based noise generation across different grid sizes.
"""

# Standard library imports
import logging
import os
import sys
import time
from typing import Dict, List

# Third-party library imports
import matplotlib.pyplot as plt

# Local application imports
from entities.base_generator import BaseGenerator
from utils.gpu_utils import get_available_backends, is_gpu_available

# numpy import removed (unused)

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules - moved before logging configuration to fix E402 errors

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _initialize_benchmark_results(sizes, noise_types):
    """Initialize the results dictionary for benchmarking.

    Args:
        sizes: List of grid sizes to test
        noise_types: List of noise types to test

    Returns:
        Dict: Initialized results dictionary
    """
    return {
        "cpu": {noise_type: [] for noise_type in noise_types},
        "gpu": {noise_type: [] for noise_type in noise_types},
        "sizes": sizes,
    }


def _create_generators(size, gpu_available):
    """Create CPU and GPU generators for benchmarking.

    Args:
        size: Grid size (width and height)
        gpu_available: Whether GPU is available

    Returns:
        tuple: (cpu_generator, gpu_generator or None)
    """
    # Create CPU generator
    cpu_generator = BaseGenerator(
        entity_id="cpu_benchmark", width=size, height=size, use_gpu=False
    )

    # Create GPU generator if available
    gpu_generator = None
    if gpu_available:
        gpu_generator = BaseGenerator(
            entity_id="gpu_benchmark",
            width=size,
            height=size,
            use_gpu=True,
            gpu_backend="auto",
        )

    return cpu_generator, gpu_generator


def _run_benchmark_iterations(generator, noise_type, iterations):
    """Run benchmark iterations for a specific generator and noise type.

    Args:
        generator: The generator to benchmark
        noise_type: Type of noise to generate
        iterations: Number of iterations to run

    Returns:
        float: Average time taken for the operations
    """
    times = []
    for _ in range(iterations):
        start_time = time.time()
        generator.generate_noise_layer(noise_type=noise_type)
        times.append(time.time() - start_time)

    return sum(times) / len(times)


def _log_gpu_results(noise_type, avg_gpu_time, avg_cpu_time):
    """Log GPU benchmark results with speedup calculation.

    Args:
        noise_type: Type of noise being benchmarked
        avg_gpu_time: Average GPU time
        avg_cpu_time: Average CPU time for speedup calculation
    """
    # Calculate speedup
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
    logging.info(
        f"  GPU {noise_type} noise: {avg_gpu_time:.4f} seconds (speedup: {speedup:.2f}x)"
    )


def run_noise_benchmark(
    sizes: List[int], noise_types: List[str] = None, iterations: int = 3
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run benchmark tests for noise generation with and without GPU acceleration.

    Args:
        sizes: List of grid sizes to test (width and height will be equal)
        noise_types: List of noise types to test
        iterations: Number of iterations to run for each test

    Returns:
        Dictionary with benchmark results
    """
    # Set default noise types if not provided
    if noise_types is None:
        noise_types = ["low", "medium", "high", "detail"]

    # Initialize results dictionary
    results = _initialize_benchmark_results(sizes, noise_types)

    # Check if GPU is available
    gpu_available = is_gpu_available()
    available_backends = get_available_backends()

    # Log GPU availability
    if not gpu_available:
        logging.warning("No GPU available. Only running CPU benchmarks.")
    else:
        logging.info(f"GPU is available with backends: {', '.join(available_backends)}")

    # Run benchmarks for each size
    for size in sizes:
        logging.info(f"Testing grid size: {size}x{size}")

        # Create generators
        cpu_generator, gpu_generator = _create_generators(size, gpu_available)

        # Test each noise type
        for noise_type in noise_types:
            # Run CPU benchmark
            avg_cpu_time = _run_benchmark_iterations(
                cpu_generator, noise_type, iterations
            )
            results["cpu"][noise_type].append(avg_cpu_time)
            logging.info(f"  CPU {noise_type} noise: {avg_cpu_time:.4f} seconds")

            # Run GPU benchmark if available
            if gpu_available:
                avg_gpu_time = _run_benchmark_iterations(
                    gpu_generator, noise_type, iterations
                )
                results["gpu"][noise_type].append(avg_gpu_time)
                _log_gpu_results(noise_type, avg_gpu_time, avg_cpu_time)
            else:
                results["gpu"][noise_type].append(0)  # No GPU available

    return results


def _initialize_multi_octave_results(sizes, octaves_list):
    """Initialize the results dictionary for multi-octave benchmark.

    Args:
        sizes: List of grid sizes to test
        octaves_list: List of octave configurations to test

    Returns:
        Dict: Initialized results dictionary
    """
    return {
        "cpu": {str(octaves): [] for octaves in octaves_list},
        "gpu": {str(octaves): [] for octaves in octaves_list},
        "sizes": sizes,
    }


def _run_multi_octave_iteration(generator, octaves):
    """Run a single multi-octave noise benchmark iteration.

    Args:
        generator: The generator to use for benchmarking
        octaves: List of octaves to use for noise generation

    Returns:
        float: Time taken for the operation
    """
    start_time = time.time()
    generator.generate_multi_octave_noise(octaves=octaves)
    return time.time() - start_time


def _log_multi_octave_gpu_results(octaves, avg_gpu_time, avg_cpu_time):
    """Log GPU benchmark results for multi-octave noise with speedup calculation.

    Args:
        octaves: The octave configuration being tested
        avg_gpu_time: Average GPU time
        avg_cpu_time: Average CPU time for speedup calculation
    """
    # Calculate speedup
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
    logging.info(
        f"  GPU octaves={octaves}: {avg_gpu_time:.4f} seconds (speedup: {speedup:.2f}x)"
    )


def run_multi_octave_benchmark(
    sizes: List[int], octaves_list: List[List[int]] = None, iterations: int = 3
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run benchmark tests for multi-octave noise generation with and without GPU acceleration.

    Args:
        sizes: List of grid sizes to test (width and height will be equal)
        octaves_list: List of octave configurations to test
        iterations: Number of iterations to run for each test

    Returns:
        Dictionary with benchmark results
    """
    # Set default octave configurations if not provided
    if octaves_list is None:
        octaves_list = [[3, 5, 8], [1, 2, 4, 8, 16]]

    # Initialize results dictionary
    results = _initialize_multi_octave_results(sizes, octaves_list)

    # Check if GPU is available
    gpu_available = is_gpu_available()
    available_backends = get_available_backends()

    # Log GPU availability
    if not gpu_available:
        logging.warning("No GPU available. Only running CPU benchmarks.")
    else:
        logging.info(f"GPU is available with backends: {', '.join(available_backends)}")

    # Run benchmarks for each grid size
    for size in sizes:
        logging.info(f"Testing multi-octave noise on grid size: {size}x{size}")

        # Create generators
        cpu_generator, gpu_generator = _create_generators(size, gpu_available)

        # Test each octave configuration
        for octaves in octaves_list:
            octaves_key = str(octaves)

            # Run CPU benchmark
            cpu_times = [
                _run_multi_octave_iteration(cpu_generator, octaves)
                for _ in range(iterations)
            ]
            avg_cpu_time = sum(cpu_times) / len(cpu_times)
            results["cpu"][octaves_key].append(avg_cpu_time)
            logging.info(f"  CPU octaves={octaves}: {avg_cpu_time:.4f} seconds")

            # Run GPU benchmark if available
            if gpu_available:
                gpu_times = [
                    _run_multi_octave_iteration(gpu_generator, octaves)
                    for _ in range(iterations)
                ]
                avg_gpu_time = sum(gpu_times) / len(gpu_times)
                results["gpu"][octaves_key].append(avg_gpu_time)
                _log_multi_octave_gpu_results(octaves, avg_gpu_time, avg_cpu_time)
            else:
                results["gpu"][octaves_key].append(0)  # No GPU available

    return results


def plot_benchmark_results(
    results: Dict[str, Dict[str, List[float]]], title: str, filename: str
):
    """
    Plot benchmark results.

    Args:
        results: Benchmark results dictionary
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(12, 8))

    sizes = results["sizes"]

    # Plot each noise type
    for noise_type in results["cpu"].keys():
        if noise_type == "sizes":
            continue

        cpu_times = results["cpu"][noise_type]
        gpu_times = results["gpu"][noise_type]

        plt.plot(sizes, cpu_times, "o-", label=f"CPU {noise_type}")

        if any(gpu_times):  # Only plot GPU if we have data
            plt.plot(sizes, gpu_times, "s-", label=f"GPU {noise_type}")

    plt.xlabel("Grid Size (width/height)")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename)
    logging.info(f"Plot saved to {filename}")

    # Also create a speedup plot
    plt.figure(figsize=(12, 8))

    for noise_type in results["cpu"].keys():
        if noise_type == "sizes":
            continue

        cpu_times = results["cpu"][noise_type]
        gpu_times = results["gpu"][noise_type]

        if any(gpu_times):  # Only calculate speedup if we have GPU data
            speedups = [
                cpu / gpu if gpu > 0 else 0 for cpu, gpu in zip(cpu_times, gpu_times)
            ]
            plt.plot(sizes, speedups, "o-", label=f"{noise_type}")

    plt.xlabel("Grid Size (width/height)")
    plt.ylabel("Speedup (CPU time / GPU time)")
    plt.title(f"GPU Speedup - {title}")
    plt.grid(True)
    plt.legend()
    plt.axhline(y=1.0, color="r", linestyle="-", alpha=0.3)  # Line at speedup = 1
    plt.tight_layout()

    # Save the speedup plot
    speedup_filename = filename.replace(".png", "_speedup.png")
    plt.savefig(speedup_filename)
    logging.info(f"Speedup plot saved to {speedup_filename}")


def main():
    """Run the benchmark tests."""
    # Grid sizes to test
    sizes = [50, 100, 200, 400, 800]

    # Run single noise layer benchmarks
    noise_results = run_noise_benchmark(sizes)
    plot_benchmark_results(
        noise_results, "Noise Generation Performance Comparison", "noise_benchmark.png"
    )

    # Run multi-octave noise benchmarks
    multi_octave_results = run_multi_octave_benchmark(sizes)
    plot_benchmark_results(
        multi_octave_results,
        "Multi-Octave Noise Generation Performance Comparison",
        "multi_octave_benchmark.png",
    )

    logging.info("Benchmarks completed!")


if __name__ == "__main__":
    main()
