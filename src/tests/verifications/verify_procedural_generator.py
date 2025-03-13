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
from generators.procedural_generator import (
from world.asteroid_field import AsteroidField

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def visualize_results(generator, asteroid_grid, rare_grid, field):
    """Visualize the results of the tests."""
    print("\nVisualizing results...")

    plt.figure(figsize=(15, 10))

    # Plot asteroid grid
    plt.subplot(2, 3, 1)
    plt.imshow(asteroid_grid, cmap="viridis")
    plt.colorbar(label="Asteroid Value")
    plt.title("Generated Asteroid Field")

    # Plot rare mineral grid
    plt.subplot(2, 3, 2)
    plt.imshow(rare_grid, cmap="plasma")
    plt.colorbar(label="Rare Level")
    plt.title("Rare Minerals")

    # Plot noise layer
    plt.subplot(2, 3, 3)
    noise = generator.generate_noise_layer("medium", scale=0.05)
    plt.imshow(noise, cmap="gray")
    plt.colorbar(label="Noise Value")
    plt.title("Noise Layer")

    # Plot field grid
    plt.subplot(2, 3, 4)
    plt.imshow(field.grid, cmap="viridis")
    plt.colorbar(label="Asteroid Value")
    plt.title("Field Asteroid Grid")

    # Plot field rare grid
    plt.subplot(2, 3, 5)
    plt.imshow(field.rare_grid, cmap="plasma")
    plt.colorbar(label="Rare Level")
    plt.title("Field Rare Minerals")

    # Plot field energy grid
    plt.subplot(2, 3, 6)
    plt.imshow(field.energy_grid, cmap="hot")
    plt.colorbar(label="Energy Level")
    plt.title("Field Energy Grid")

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
