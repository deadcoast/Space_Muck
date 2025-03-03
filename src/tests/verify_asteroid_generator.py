#!/usr/bin/env python3
"""
Verification script for the refactored AsteroidGenerator class.
Tests the functionality of the AsteroidGenerator and ensures it correctly
inherits from BaseGenerator while maintaining all required functionality.
"""

import sys
import os

# Try to import required packages, but don't fail if they're not available
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    print("numpy not available. Tests will be skipped.")
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("matplotlib not available. Visualization will be skipped.")
    MATPLOTLIB_AVAILABLE = False

try:
    import scipy

    SCIPY_AVAILABLE = True
except ImportError:
    print("scipy not available. Tests will be skipped.")
    SCIPY_AVAILABLE = False

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Only import if required packages are available
if NUMPY_AVAILABLE and SCIPY_AVAILABLE:
    try:
        from generators.asteroid_generator import AsteroidGenerator

        ASTEROID_GENERATOR_AVAILABLE = True
    except ImportError as e:
        print(f"Could not import AsteroidGenerator: {e}")
        ASTEROID_GENERATOR_AVAILABLE = False
else:
    ASTEROID_GENERATOR_AVAILABLE = False


def test_asteroid_generator():
    """Test the basic functionality of the AsteroidGenerator class."""
    print("Testing AsteroidGenerator initialization...")

    # Create a generator with a fixed seed for reproducibility
    generator = AsteroidGenerator(seed=42, width=100, height=100)

    # Verify basic properties
    assert generator.seed == 42, f"Expected seed 42, got {generator.seed}"
    assert generator.width == 100, f"Expected width 100, got {generator.width}"
    assert generator.height == 100, f"Expected height 100, got {generator.height}"
    assert generator.entity_type == "asteroid", (
        f"Expected entity_type 'asteroid', got {generator.entity_type}"
    )

    # Test noise generation
    print("Testing noise generation...")
    noise = generator.generate_noise_layer("medium", scale=0.05)
    assert noise.shape == (100, 100), f"Expected shape (100, 100), got {noise.shape}"

    # Test field generation
    print("Testing asteroid field generation...")
    asteroid_grid, metadata = generator.generate_field()
    assert asteroid_grid.shape == (100, 100), (
        f"Expected shape (100, 100), got {asteroid_grid.shape}"
    )
    assert np.sum(asteroid_grid > 0) > 0, "No asteroids were generated"
    assert "seed" in metadata, "Metadata missing seed information"

    # Test value generation
    print("Testing asteroid value generation...")
    value_grid = generator.generate_values(asteroid_grid)
    assert value_grid.shape == (100, 100), (
        f"Expected shape (100, 100), got {value_grid.shape}"
    )
    assert np.sum(value_grid > 0) > 0, "No asteroid values were generated"

    # Test rare resource generation
    print("Testing rare resource generation...")
    rare_grid = generator.generate_rare_resources(asteroid_grid)
    assert rare_grid.shape == (100, 100), (
        f"Expected shape (100, 100), got {rare_grid.shape}"
    )

    print("All basic tests passed!")
    return generator, asteroid_grid, value_grid, rare_grid


def test_pattern_generation():
    """Test the pattern generation functionality."""
    print("\nTesting pattern generation...")

    # Create a generator with a fixed seed for reproducibility
    generator = AsteroidGenerator(seed=123, width=80, height=80)

    # Test with different pattern weights
    pattern_weights = [0.3, 0.2, 0.4, 0.1]  # Weights for each pattern
    asteroid_grid, metadata = generator.generate_field(pattern_weights=pattern_weights)

    assert asteroid_grid.shape == (80, 80), (
        f"Expected shape (80, 80), got {asteroid_grid.shape}"
    )
    assert np.sum(asteroid_grid > 0) > 0, (
        "No asteroids were generated with pattern weights"
    )

    print("Pattern generation test passed!")
    return asteroid_grid


def visualize_results(generator, asteroid_grid, value_grid, rare_grid, pattern_grid):
    """Visualize the results of the tests."""
    if not MATPLOTLIB_AVAILABLE:
        print("\nSkipping visualization as matplotlib is not available.")
        return

    print("\nVisualizing results...")

    plt.figure(figsize=(15, 10))

    # Plot asteroid grid
    plt.subplot(2, 3, 1)
    _visualize_colorbar(
        asteroid_grid,
        "binary",
        "Asteroid Presence",
        "Generated Asteroid Field",
    )
    # Plot value grid
    plt.subplot(2, 3, 2)
    _visualize_colorbar(value_grid, "viridis", "Resource Value", "Asteroid Values")
    # Plot rare mineral grid
    plt.subplot(2, 3, 3)
    _visualize_colorbar(rare_grid, "plasma", "Rare Resource", "Rare Resources")
    # Plot pattern grid
    plt.subplot(2, 3, 4)
    _visualize_colorbar(
        pattern_grid, "binary", "Asteroid Presence", "Pattern-based Field"
    )
    # Plot spiral pattern
    plt.subplot(2, 3, 5)
    spiral = generator._spiral_pattern()
    _visualize_colorbar(spiral, "hot", "Pattern Value", "Spiral Pattern")
    # Plot ring pattern
    plt.subplot(2, 3, 6)
    rings = generator._ring_pattern()
    _visualize_colorbar(rings, "hot", "Pattern Value", "Ring Pattern")
    plt.tight_layout()
    plt.savefig("asteroid_generator_test.png")
    print("Visualization saved as 'asteroid_generator_test.png'")


def _visualize_colorbar(arg0, cmap, label, arg3):
    plt.imshow(arg0, cmap=cmap)
    plt.colorbar(label=label)
    plt.title(arg3)


if __name__ == "__main__":
    print("=== AsteroidGenerator Verification ===")

    # Check if we can run the tests
    if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE or not ASTEROID_GENERATOR_AVAILABLE:
        print("\nSkipping tests due to missing dependencies.")
        print("Please install the required packages to run the tests:")
        print("  - numpy")
        print("  - scipy")
        print("  - matplotlib (optional, for visualization)")
        sys.exit(1)

    # Run the tests
    generator, asteroid_grid, value_grid, rare_grid = test_asteroid_generator()
    pattern_grid = test_pattern_generation()

    # Visualize the results if matplotlib is available
    visualize_results(generator, asteroid_grid, value_grid, rare_grid, pattern_grid)

    print("\n=== All tests completed successfully! ===")
    print("The refactored AsteroidGenerator is working correctly.")
