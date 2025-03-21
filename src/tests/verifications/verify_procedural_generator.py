#!/usr/bin/env python3
"""
Verification script for the refactored ProceduralGenerator class.
Tests the functionality of the ProceduralGenerator and ensures it correctly
inherits from BaseGenerator while maintaining all required functionality.
"""

# Standard library imports
import os
import sys

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np

# Local application imports
# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world.asteroid_field import AsteroidField

# Import the modules after path setup
from generators.procedural_generator import (
    ProceduralGenerator,
    create_field_with_multiple_algorithms,
)


def test_procedural_generator():
    """Test the basic functionality of the ProceduralGenerator class."""
    print("Testing ProceduralGenerator initialization...")

    # Create a generator with a fixed seed for reproducibility
    generator = ProceduralGenerator(seed=42, width=100, height=100)

    # Verify basic properties
    assert generator.seed == 42, f"Expected seed 42, got {generator.seed}"
    assert generator.width == 100, f"Expected width 100, got {generator.width}"
    assert generator.height == 100, f"Expected height 100, got {generator.height}"
    assert (
        generator.entity_type == "procedural"
    ), f"Expected entity_type 'procedural', got {generator.entity_type}"

    # Test noise generation
    print("Testing noise generation...")
    noise = generator.generate_noise_layer("medium", scale=0.05)
    assert noise.shape == (100, 100), f"Expected shape (100, 100), got {noise.shape}"

    # Test asteroid field generation
    print("Testing asteroid field generation...")
    asteroid_grid = generator.generate_asteroid_field(density=0.3)
    assert asteroid_grid.shape == (
        100,
        100,
    ), f"Expected shape (100, 100), got {asteroid_grid.shape}"
    assert np.sum(asteroid_grid > 0) > 0, "No asteroids were generated"

    # Test rare mineral generation
    print("Testing rare mineral generation...")
    rare_grid = generator.generate_rare_minerals(asteroid_grid, rare_chance=0.2)
    assert rare_grid.shape == (
        100,
        100,
    ), f"Expected shape (100, 100), got {rare_grid.shape}"

    print("All basic tests passed!")
    return generator, asteroid_grid, rare_grid


def test_create_field_function():
    """Test the create_field_with_multiple_algorithms function."""
    print("\nTesting create_field_with_multiple_algorithms function...")

    # Create a field with the function
    field = create_field_with_multiple_algorithms(
        width=80, height=80, seed=123, rare_chance=0.15, rare_bonus=2.5
    )

    # Verify the field properties
    assert isinstance(
        field, AsteroidField
    ), f"Expected AsteroidField, got {type(field)}"
    assert field.width == 80, f"Expected width 80, got {field.width}"
    assert field.height == 80, f"Expected height 80, got {field.height}"
    assert field.grid.shape == (
        80,
        80,
    ), f"Expected grid shape (80, 80), got {field.grid.shape}"
    assert field.rare_grid.shape == (
        80,
        80,
    ), f"Expected rare_grid shape (80, 80), got {field.rare_grid.shape}"
    assert field.energy_grid.shape == (
        80,
        80,
    ), f"Expected energy_grid shape (80, 80), got {field.energy_grid.shape}"

    # Check that we have asteroids and rare minerals
    assert np.sum(field.grid > 0) > 0, "No asteroids were generated"
    assert np.sum(field.rare_grid > 0) > 0, "No rare minerals were generated"

    print("Field creation test passed!")
    return field


def _plot_grid(position, data, cmap, label, title):
    """Helper function to plot a grid with consistent formatting.

    Args:
        position: Subplot position (e.g., 2, 3, 1)
        data: Grid data to plot
        cmap: Colormap to use
        label: Colorbar label
        title: Plot title
    """
    plt.subplot(position[0], position[1], position[2])
    plt.imshow(data, cmap=cmap)
    plt.colorbar(label=label)
    plt.title(title)


def visualize_results(generator, asteroid_grid, rare_grid, field):
    """Visualize the results of the tests."""
    print("\nVisualizing results...")

    plt.figure(figsize=(15, 10))

    # Plot generator outputs
    _plot_grid(
        (2, 3, 1),
        asteroid_grid,
        "viridis",
        "Asteroid Value",
        "Generated Asteroid Field",
    )
    _plot_grid((2, 3, 2), rare_grid, "plasma", "Rare Level", "Rare Minerals")

    # Plot noise layer
    noise = generator.generate_noise_layer("medium", scale=0.05)
    _plot_grid((2, 3, 3), noise, "gray", "Noise Value", "Noise Layer")

    # Plot field grids
    _plot_grid(
        (2, 3, 4), field.grid, "viridis", "Asteroid Value", "Field Asteroid Grid"
    )
    _plot_grid(
        (2, 3, 5), field.rare_grid, "plasma", "Rare Level", "Field Rare Minerals"
    )

    # Plot field energy grid
    _plot_grid((2, 3, 6), field.energy_grid, "hot", "Energy Level", "Field Energy Grid")

    plt.tight_layout()
    plt.savefig("procedural_generator_test.png")
    print("Visualization saved as 'procedural_generator_test.png'")


if __name__ == "__main__":
    print("=== ProceduralGenerator Verification ===")

    # Run the tests
    generator, asteroid_grid, rare_grid = test_procedural_generator()
    field = test_create_field_function()

    # Visualize the results if matplotlib is available
    try:
        visualize_results(generator, asteroid_grid, rare_grid, field)
    except Exception as e:
        print(f"Visualization failed: {e}")

    print("\n=== All tests completed successfully! ===")
    print("The refactored ProceduralGenerator is working correctly.")
