"""
Verification script for BaseGenerator optimizations.

This script tests the optimized methods in BaseGenerator to ensure they work correctly
and provide the expected performance improvements.
"""

import sys
import os
import time
import numpy as np

# Only import what we need
# import matplotlib.pyplot as plt
# from typing import Dict, List, Tuple, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from entities.base_generator import BaseGenerator
from utils.visualization import GeneratorVisualizer


def test_noise_layer_generation():
    """Test the optimized noise layer generation."""
    print("\n=== Testing Noise Layer Generation ===")

    # Create generators with different sizes
    sizes = [(50, 50), (100, 100), (200, 200), (400, 400)]
    generators = []

    for width, height in sizes:
        generator = BaseGenerator(
            entity_id=f"test_noise_{width}x{height}",
            seed=42,
            width=width,
            height=height,
        )
        generators.append(generator)

    # Test performance for each size
    for i, generator in enumerate(generators):
        width, height = sizes[i]
        print(f"\nTesting noise generation for {width}x{height} grid:")

        # Time standard generation
        start_time = time.time()
        noise_grid = generator.generate_noise_layer("medium", scale=0.1)
        standard_time = time.time() - start_time
        print(f"  Standard generation time: {standard_time:.4f} seconds")

        # Force cache reset
        generator._noise_cache = {}

        # Time cached generation (second call)
        start_time = time.time()
        noise_grid_cached = generator.generate_noise_layer("medium", scale=0.1)
        cached_time = time.time() - start_time
        print(f"  Cached generation time: {cached_time:.4f} seconds")
        print(f"  Speedup factor: {standard_time / max(cached_time, 0.0001):.2f}x")

        # Verify results are the same
        if np.array_equal(noise_grid, noise_grid_cached):
            print("  ✓ Results match")
        else:
            print("  ✗ Results don't match")

    # Visualize a sample noise layer
    visualizer = GeneratorVisualizer()
    medium_generator = generators[1]  # 100x100
    noise_grid = medium_generator.generate_noise_layer("medium", scale=0.1)

    try:
        visualizer.visualize_grid(
            noise_grid, title="Noise Layer (100x100)", colormap="terrain", show=True
        )
        print("\n✓ Visualization successful")
    except Exception as e:
        print(f"\n✗ Visualization failed: {str(e)}")


def test_cellular_automaton():
    """Test the optimized cellular automaton implementation."""
    print("\n=== Testing Cellular Automaton ===")

    # Create generators with different sizes
    sizes = [(50, 50), (100, 100), (200, 200)]
    generators = []

    for width, height in sizes:
        generator = BaseGenerator(
            entity_id=f"test_ca_{width}x{height}", seed=42, width=width, height=height
        )
        generators.append(generator)

    # Test performance for each size
    for i, generator in enumerate(generators):
        width, height = sizes[i]
        print(f"\nTesting cellular automaton for {width}x{height} grid:")

        # Create a random grid
        grid = np.random.binomial(1, 0.4, (height, width))

        # Time standard CA
        start_time = time.time()
        ca_grid = generator.apply_cellular_automaton(
            grid, birth_set={3}, survival_set={2, 3}, iterations=3
        )
        standard_time = time.time() - start_time
        print(f"  Standard CA time: {standard_time:.4f} seconds")

        # Force cache reset
        generator._ca_cache = {}

        # Time cached CA (second call with same parameters)
        start_time = time.time()
        ca_grid_cached = generator.apply_cellular_automaton(
            grid, birth_set={3}, survival_set={2, 3}, iterations=3
        )
        cached_time = time.time() - start_time
        print(f"  Cached CA time: {cached_time:.4f} seconds")
        print(f"  Speedup factor: {standard_time / max(cached_time, 0.0001):.2f}x")

        # Verify results are the same
        if np.array_equal(ca_grid, ca_grid_cached):
            print("  ✓ Results match")
        else:
            print("  ✗ Results don't match")

    # Visualize a sample CA evolution
    visualizer = GeneratorVisualizer()
    medium_generator = generators[1]  # 100x100

    # Create a random grid
    grid = np.random.binomial(1, 0.4, (100, 100))

    # Apply CA for multiple iterations
    evolution_grids = []
    current_grid = grid.copy()
    for _ in range(5):
        evolution_grids.append(current_grid.copy())
        current_grid = medium_generator.apply_cellular_automaton(
            current_grid, birth_set={3}, survival_set={2, 3}, iterations=1
        )

    try:
        visualizer.visualize_evolution(
            evolution_grids,
            title="Cellular Automaton Evolution",
            colormap="binary",
            show=True,
        )
        print("\n✓ Evolution visualization successful")
    except Exception as e:
        print(f"\n✗ Evolution visualization failed: {str(e)}")


def test_clustering():
    """Test the optimized clustering implementation."""
    print("\n=== Testing Clustering ===")

    # Create generators with different sizes
    sizes = [(50, 50), (100, 100), (200, 200)]
    generators = []

    for width, height in sizes:
        generator = BaseGenerator(
            entity_id=f"test_cluster_{width}x{height}",
            seed=42,
            width=width,
            height=height,
        )
        generators.append(generator)

    # Test performance for each size
    for i, generator in enumerate(generators):
        width, height = sizes[i]
        print(f"\nTesting clustering for {width}x{height} grid:")

        # Create a random grid
        grid = np.random.binomial(1, 0.3, (height, width))

        # Time standard clustering
        start_time = time.time()
        clustered_grid = generator.create_clusters(grid, num_clusters=5)
        standard_time = time.time() - start_time
        print(f"  Standard clustering time: {standard_time:.4f} seconds")

        # Force cache reset
        generator._cluster_cache = {}

        # Time cached clustering (second call with same parameters)
        start_time = time.time()
        clustered_grid_cached = generator.create_clusters(grid, num_clusters=5)
        cached_time = time.time() - start_time
        print(f"  Cached clustering time: {cached_time:.4f} seconds")
        print(f"  Speedup factor: {standard_time / max(cached_time, 0.0001):.2f}x")

        # Verify results are the same
        if np.array_equal(clustered_grid, clustered_grid_cached):
            print("  ✓ Results match")
        else:
            print("  ✗ Results don't match")

    # Visualize clusters
    visualizer = GeneratorVisualizer()
    medium_generator = generators[1]  # 100x100

    # Create a random grid
    grid = np.random.binomial(1, 0.3, (100, 100))

    # Apply CA to make more coherent clusters
    grid = medium_generator.apply_cellular_automaton(
        grid, birth_set={3, 4, 5}, survival_set={2, 3, 4, 5}, iterations=2
    )

    # Get clustered grid
    clustered_grid = medium_generator.create_clusters(grid, num_clusters=5)

    # Create a visualization grid showing the difference
    original_grid = grid.copy()
    cluster_grid = clustered_grid - original_grid

    try:
        visualizer.visualize_grid(
            cluster_grid,
            title="Clusters (different colors)",
            colormap="tab20",
            show=True,
        )
        print("\n✓ Cluster visualization successful")
    except Exception as e:
        print(f"\n✗ Cluster visualization failed: {str(e)}")


def test_thresholding():
    """Test the optimized thresholding implementation."""
    print("\n=== Testing Thresholding ===")

    # Create generators with different sizes
    sizes = [(50, 50), (100, 100), (200, 200)]
    generators = []

    for width, height in sizes:
        generator = BaseGenerator(
            entity_id=f"test_threshold_{width}x{height}",
            seed=42,
            width=width,
            height=height,
        )
        generators.append(generator)

    # Test performance for each size
    for i, generator in enumerate(generators):
        width, height = sizes[i]
        print(f"\nTesting thresholding for {width}x{height} grid:")

        # Create a random grid with float values
        grid = np.random.random((height, width))

        # Time standard thresholding
        start_time = time.time()
        threshold_grid = generator.apply_threshold(grid, 0.5)
        standard_time = time.time() - start_time
        print(f"  Standard thresholding time: {standard_time:.4f} seconds")

        # Force cache reset
        generator._threshold_cache = {}

        # Time cached thresholding (second call with same parameters)
        start_time = time.time()
        threshold_grid_cached = generator.apply_threshold(grid, 0.5)
        cached_time = time.time() - start_time
        print(f"  Cached thresholding time: {cached_time:.4f} seconds")
        print(f"  Speedup factor: {standard_time / max(cached_time, 0.0001):.2f}x")

        # Verify results are the same
        if np.array_equal(threshold_grid, threshold_grid_cached):
            print("  ✓ Results match")
        else:
            print("  ✗ Results don't match")

    # Visualize thresholding
    visualizer = GeneratorVisualizer()
    medium_generator = generators[1]  # 100x100

    # Create a noise grid
    noise_grid = medium_generator.generate_noise_layer("medium", scale=0.1)

    # Apply different thresholds
    thresholds = [0.3, 0.5, 0.7]
    threshold_grids = []

    for threshold in thresholds:
        threshold_grid = medium_generator.apply_threshold(noise_grid, threshold)
        threshold_grids.append(threshold_grid)

    try:
        visualizer.visualize_grid_comparison(
            threshold_grids,
            titles=[f"Threshold: {t}" for t in thresholds],
            colormap="binary",
            show=True,
        )
        print("\n✓ Threshold comparison visualization successful")
    except Exception as e:
        print(f"\n✗ Threshold comparison visualization failed: {str(e)}")


def main():
    """Run all verification tests."""
    print("=== BaseGenerator Optimization Verification ===")

    # Test noise layer generation
    test_noise_layer_generation()

    # Test cellular automaton
    test_cellular_automaton()

    # Test clustering
    test_clustering()

    # Test thresholding
    test_thresholding()

    print("\n=== Verification Complete ===")


if __name__ == "__main__":
    main()
