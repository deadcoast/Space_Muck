"""
Test script for visualization utilities.

This script demonstrates the visualization capabilities for generator outputs.
"""

import os
import sys
import logging
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entities.base_generator import BaseGenerator
from utils.visualization import GeneratorVisualizer, visualize_generator_output
from utils.noise_generator import NoiseGenerator


def test_visualize_noise_layers():
    """Test visualization of noise layers."""
    print("Testing noise layer visualization...")
    
    # Create a generator
    width, height = 100, 100
    seed = 42
    noise_gen = NoiseGenerator(seed=seed)
    generator = BaseGenerator(width, height, seed=seed, noise_generator=noise_gen)
    
    # Create visualizer
    visualizer = GeneratorVisualizer(output_dir="test_visualizations")
    
    # Generate and visualize different noise types
    noise_types = ["low", "medium", "high", "detail"]
    noise_grids = []
    noise_titles = []
    
    for noise_type in noise_types:
        grid = generator.generate_noise_layer(noise_type=noise_type)
        noise_grids.append(grid)
        noise_titles.append(f"{noise_type.capitalize()} Noise")
    
    # Compare noise layers
    visualizer.compare_grids(
        noise_grids,
        noise_titles,
        colormap="terrain",
        show=True,
        save=True,
        filename="noise_comparison.png"
    )
    
    print("Noise layer visualization completed.")


def test_visualize_cellular_automaton():
    """Test visualization of cellular automaton evolution."""
    print("Testing cellular automaton visualization...")
    
    # Create a generator
    width, height = 100, 100
    seed = 42
    noise_gen = NoiseGenerator(seed=seed)
    generator = BaseGenerator(width, height, seed=seed, noise_generator=noise_gen)
    
    # Create visualizer
    visualizer = GeneratorVisualizer(output_dir="test_visualizations")
    
    # Generate base noise and apply threshold
    base_grid = generator.generate_noise_layer(noise_type="medium")
    binary_grid = generator.apply_threshold(base_grid, 0.5, 1.0)
    
    # Apply cellular automaton with different iterations
    ca_grids = [binary_grid]
    ca_titles = ["Initial Grid"]
    
    for iterations in [1, 3, 5]:
        ca_grid = generator.apply_cellular_automaton(
            binary_grid.copy(),
            iterations=iterations
        )
        ca_grids.append(ca_grid)
        ca_titles.append(f"After {iterations} Iterations")
    
    # Visualize evolution
    visualizer.compare_grids(
        ca_grids,
        ca_titles,
        colormap="binary",
        show=True,
        save=True,
        filename="ca_evolution.png"
    )
    
    # Create animation
    visualizer.visualize_evolution(
        ca_grids,
        title="Cellular Automaton Evolution",
        colormap="binary",
        show=True,
        save=True,
        filename="ca_evolution_grid.png",
        animation=True,
        animation_filename="ca_evolution.gif"
    )
    
    print("Cellular automaton visualization completed.")


def test_visualize_clusters():
    """Test visualization of clustering."""
    print("Testing cluster visualization...")
    
    # Create a generator
    width, height = 100, 100
    seed = 42
    noise_gen = NoiseGenerator(seed=seed)
    generator = BaseGenerator(width, height, seed=seed, noise_generator=noise_gen)
    
    # Create visualizer
    visualizer = GeneratorVisualizer(output_dir="test_visualizations")
    
    # Generate base noise and apply threshold
    base_grid = generator.generate_noise_layer(noise_type="medium")
    thresholded_grid = generator.apply_threshold(base_grid, 0.5, 1.0)
    
    # Create clusters
    clustered_grid = generator.create_clusters(
        thresholded_grid,
        num_clusters=5,
        cluster_value_multiplier=2.0
    )
    
    # Visualize before and after
    visualizer.compare_grids(
        [thresholded_grid, clustered_grid],
        ["Before Clustering", "After Clustering"],
        colormap="terrain",
        show=True,
        save=True,
        filename="clustering.png"
    )
    
    print("Cluster visualization completed.")


def test_convenience_function():
    """Test the convenience function for visualizing generator outputs."""
    print("Testing convenience function...")
    
    # Create a generator
    width, height = 100, 100
    seed = 42
    noise_gen = NoiseGenerator(seed=seed)
    generator = BaseGenerator(width, height, seed=seed, noise_generator=noise_gen)
    
    # Use convenience function
    visualize_generator_output(
        generator,
        output_dir="test_visualizations",
        show=True,
        save=True,
        colormap="terrain"
    )
    
    print("Convenience function test completed.")


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("test_visualizations", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run tests
    test_visualize_noise_layers()
    test_visualize_cellular_automaton()
    test_visualize_clusters()
    test_convenience_function()
    
    print("All visualization tests completed successfully!")
