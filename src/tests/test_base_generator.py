#!/usr/bin/env python3
"""
Unit tests for the BaseGenerator class.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the class to test
from entities.base_generator import BaseGenerator
from utils.noise_generator import NoiseGenerator
from utils.dependency_injection import DependencyContainer


class MockNoiseGenerator(NoiseGenerator):
    """Mock implementation of NoiseGenerator for testing."""

    def __init__(self, return_value=0.5):
        self.return_value = return_value
        self.calls = []

    def generate_noise(self, width, height, scale=0.1, octaves=1, seed=None):
        self.calls.append(("generate_noise", width, height, scale, octaves, seed))
        return np.full((height, width), self.return_value)

    def generate_multi_octave_noise(
        self, width, height, scale=0.1, octaves=None, weights=None, seed=None
    ):
        self.calls.append(
            (
                "generate_multi_octave_noise",
                width,
                height,
                scale,
                octaves,
                weights,
                seed,
            )
        )
        return np.full((height, width), self.return_value)


class TestBaseGenerator(unittest.TestCase):
    """Test cases for the BaseGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock noise generator
        self.mock_noise_generator = MockNoiseGenerator(return_value=0.5)

        # Set up dependency container
        self.container = DependencyContainer()
        self.container.register(
            NoiseGenerator, lambda: self.mock_noise_generator, singleton=True
        )

        # Create a generator for testing
        self.generator = BaseGenerator(
            entity_id="gen-123",
            entity_type="test_generator",
            seed=42,
            width=50,
            height=60,
            color=(100, 200, 100),
            position=(5, 10),
            noise_generator=self.mock_noise_generator,
        )

    def tearDown(self):
        """Tear down test fixtures."""
        pass

    def test_initialization(self):
        """Test that generator initializes with correct values."""
        # Test with explicit values
        self.assertEqual(self.generator.entity_id, "gen-123")
        self.assertEqual(self.generator.entity_type, "test_generator")
        self.assertEqual(self.generator.seed, 42)
        self.assertEqual(self.generator.width, 50)
        self.assertEqual(self.generator.height, 60)
        self.assertEqual(self.generator.color, (100, 200, 100))
        self.assertEqual(self.generator.position, (5, 10))

        # Test that noise generator is set
        self.assertEqual(self.generator.noise_generator, self.mock_noise_generator)

        # Test that noise cache is initialized
        self.assertEqual(self.generator._noise_cache, {})

        # Test default parameters
        self.assertIn("density", self.generator.parameters)
        self.assertIn("complexity", self.generator.parameters)
        self.assertIn("turbulence", self.generator.parameters)
        self.assertIn("iterations", self.generator.parameters)
        self.assertIn("rare_chance", self.generator.parameters)

    def test_default_initialization(self):
        """Test initialization with default values."""
        # Create a new mock for default initialization
        default_mock = MockNoiseGenerator(return_value=0.3)

        # Create a generator with default values but explicit noise generator
        generator = BaseGenerator(noise_generator=default_mock)

        # Test default values
        self.assertEqual(generator.entity_type, "generator")
        self.assertEqual(generator.width, 100)
        self.assertEqual(generator.height, 100)
        self.assertEqual(generator.color, (100, 200, 100))
        self.assertIsNone(generator.position)

        # Seed should be set to a random value
        self.assertIsNotNone(generator.seed)

        # Noise generator should be set to our mock
        self.assertEqual(generator.noise_generator, default_mock)

    def test_generate_noise_layer(self):
        """Test the generate_noise_layer method."""
        # Generate a noise layer
        noise_layer = self.generator.generate_noise_layer(
            noise_type="medium", scale=0.1
        )

        # Verify the shape of the noise layer
        self.assertEqual(noise_layer.shape, (60, 50))  # (height, width)

        # Verify all values are set to the mock return value
        self.assertTrue(np.all(noise_layer == 0.5))

        # Verify the noise generator was called with correct parameters
        last_call = self.mock_noise_generator.calls[-1]
        self.assertEqual(last_call[0], "generate_noise")  # Method name
        self.assertEqual(last_call[1], 50)  # width
        self.assertEqual(last_call[2], 60)  # height
        self.assertEqual(last_call[3], 0.1)  # scale
        self.assertEqual(last_call[4], 5)  # octaves for medium

        # Test with an unknown noise type (should default to "medium")
        noise_layer = self.generator.generate_noise_layer(
            noise_type="unknown", scale=0.1
        )
        self.assertEqual(noise_layer.shape, (60, 50))

        # Test caching
        # Call again with same parameters
        self.mock_noise_generator.calls = []  # Clear call history
        noise_layer2 = self.generator.generate_noise_layer(
            noise_type="medium", scale=0.1
        )

        # Should use cached value, so no new calls to noise generator
        self.assertEqual(len(self.mock_noise_generator.calls), 0)

    def test_apply_cellular_automaton(self):
        """Test the apply_cellular_automaton method."""
        # Create a test generator with smaller dimensions
        test_generator = BaseGenerator(
            entity_id="test-ca",
            width=10,
            height=10,
            noise_generator=self.mock_noise_generator,
        )

        # Create a simple grid with some cells
        grid = np.zeros((10, 10))
        grid[4:7, 4:7] = 1  # 3x3 block of cells in the middle

        # Apply cellular automaton with default rules (Conway's Game of Life)
        result = test_generator.apply_cellular_automaton(
            grid=grid, birth_set={3}, survival_set={2, 3}, iterations=1, wrap=True
        )

        # Verify the result is the same shape
        self.assertEqual(result.shape, (10, 10))

        # Create a test case where we know the expected outcome
        # For a 3x3 block in Conway's Game of Life, the corners die and the edges survive
        # But our implementation preserves original values where cells are alive
        expected = np.zeros((10, 10))
        expected[4:7, 4:7] = 1  # Original 3x3 block
        expected[4, 4] = 0  # Top-left corner dies
        expected[4, 6] = 0  # Top-right corner dies
        expected[6, 4] = 0  # Bottom-left corner dies
        expected[6, 6] = 0  # Bottom-right corner dies

        # Test with a controlled grid and known rules
        controlled_grid = np.zeros((10, 10))
        controlled_grid[4:7, 4:7] = 1  # 3x3 block

        # Create a binary version for testing
        binary_grid = (controlled_grid > 0).astype(np.int8)

        # Apply rules manually for one iteration
        new_grid = binary_grid.copy()

        # Apply cellular automaton rules manually
        for y in range(10):
            for x in range(10):
                # Count live neighbors
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = (x + dx) % 10, (y + dy) % 10  # Wrap around
                        neighbors += binary_grid[ny, nx]

                # Apply rules
                if binary_grid[y, x] == 1:
                    # Cell is alive
                    if neighbors not in {2, 3}:
                        new_grid[y, x] = 0  # Cell dies
                elif neighbors == 3:
                    new_grid[y, x] = 1  # Cell is born

        # Apply the method
        result = test_generator.apply_cellular_automaton(
            grid=controlled_grid,
            birth_set={3},
            survival_set={2, 3},
            iterations=1,
            wrap=True,
        )

        # Verify the result matches our manual calculation
        # Note: The method preserves original values where cells are alive
        expected_result = controlled_grid * new_grid
        np.testing.assert_array_equal(result, expected_result)

    def test_create_clusters(self):
        """Test the create_clusters method."""
        # Create a test generator with smaller dimensions
        test_generator = BaseGenerator(
            entity_id="test-clusters",
            width=10,
            height=10,
            noise_generator=self.mock_noise_generator,
        )

        # Create a simple grid
        grid = np.ones((10, 10))

        # Apply clustering
        result = test_generator.create_clusters(
            grid=grid, num_clusters=3, cluster_value_multiplier=2.0
        )

        # Verify the result is the same shape
        self.assertEqual(result.shape, (10, 10))

        # Verify that some values are higher than the original
        self.assertTrue(np.any(result > 1.0))

        # Test with an empty grid (should return the original grid)
        empty_grid = np.zeros((10, 10))
        result = test_generator.create_clusters(
            grid=empty_grid, num_clusters=3, cluster_value_multiplier=2.0
        )
        np.testing.assert_array_equal(result, empty_grid)

        # Test with fewer non-zero cells than clusters
        sparse_grid = np.zeros((10, 10))
        sparse_grid[0, 0] = 1
        sparse_grid[9, 9] = 1
        result = test_generator.create_clusters(
            grid=sparse_grid, num_clusters=5, cluster_value_multiplier=2.0
        )
        # Should return the original grid without changes
        np.testing.assert_array_equal(result, sparse_grid)

    def test_to_dict(self):
        """Test the to_dict method."""
        data = self.generator.to_dict()

        # Verify the dictionary contains all required fields
        self.assertEqual(data["entity_id"], "gen-123")
        self.assertEqual(data["entity_type"], "test_generator")
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["width"], 50)
        self.assertEqual(data["height"], 60)
        self.assertEqual(data["color"], (100, 200, 100))
        self.assertEqual(data["position"], (5, 10))
        self.assertEqual(data["parameters"], self.generator.parameters)

    def test_generate_multi_octave_noise(self):
        """Test the generate_multi_octave_noise method."""
        # Generate multi-octave noise
        noise_layer = self.generator.generate_multi_octave_noise(
            scale=0.2, octaves=[2, 4, 6], weights=[0.7, 0.2, 0.1]
        )

        # Verify the shape of the noise layer
        self.assertEqual(noise_layer.shape, (60, 50))  # (height, width)

        # Verify all values are set to the mock return value
        self.assertTrue(np.all(noise_layer == 0.5))

        # Verify the noise generator was called with correct parameters
        last_call = self.mock_noise_generator.calls[-1]
        self.assertEqual(last_call[0], "generate_multi_octave_noise")  # Method name
        self.assertEqual(last_call[1], 50)  # width
        self.assertEqual(last_call[2], 60)  # height
        self.assertEqual(last_call[3], 0.2)  # scale
        self.assertEqual(last_call[4], [2, 4, 6])  # octaves
        self.assertEqual(last_call[5], [0.7, 0.2, 0.1])  # weights

        # Test with default parameters
        self.mock_noise_generator.calls = []  # Clear call history
        noise_layer = self.generator.generate_multi_octave_noise(scale=0.1)

        # Verify default parameters were used
        last_call = self.mock_noise_generator.calls[-1]
        self.assertEqual(last_call[4], [3, 5, 8])  # default octaves
        self.assertEqual(last_call[5], [1.0, 0.5, 0.25])  # default weights

        # Test caching
        self.mock_noise_generator.calls = []  # Clear call history
        noise_layer2 = self.generator.generate_multi_octave_noise(scale=0.1)

        # Should use cached value, so no new calls to noise generator
        self.assertEqual(len(self.mock_noise_generator.calls), 0)


if __name__ == "__main__":
    unittest.main()
