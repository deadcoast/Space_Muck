#!/usr/bin/env python3
"""
Unit tests for the BaseGenerator class.
"""

import importlib.util

# Standard library imports
import itertools
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Third-party library imports
import numpy as np

# Local application imports
from generators.base_generator import BaseGenerator
from utils.dependency_injection import DependencyContainer
from utils.noise_generator import NoiseGenerator

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the class to test

# Check for utility modules availability using importlib.util.find_spec
CA_UTILS_AVAILABLE = (
    importlib.util.find_spec("src.utils.cellular_automaton_utils") is not None
)
VALUE_GEN_AVAILABLE = importlib.util.find_spec("src.utils.value_generator") is not None


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
        expected_parameters = [
            "density",
            "complexity",
            "turbulence",
            "iterations",
            "rare_chance",
        ]
        for param in expected_parameters:
            self.assertIn(
                param,
                self.generator.parameters,
                f"Generator should have '{param}' in its parameters",
            )

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
        # Create a new generator to avoid caching issues
        test_generator = BaseGenerator(
            entity_id="test-noise",
            width=50,
            height=60,
            noise_generator=MockNoiseGenerator(return_value=0.5),
        )

        # Generate a noise layer
        noise_layer = test_generator.generate_noise_layer(
            noise_type="medium", scale=0.1
        )

        # Verify the shape of the noise layer
        self.assertEqual(noise_layer.shape, (60, 50))  # (height, width)

        # Test with an unknown noise type (should default to "medium")
        noise_layer = self.generator.generate_noise_layer(
            noise_type="unknown", scale=0.1
        )
        self.assertEqual(
            noise_layer.shape,
            (60, 50),
            "Noise layer shape should match generator dimensions (height, width)",
        )

        # Test caching
        # Call again with same parameters
        self.mock_noise_generator.calls = []  # Clear call history
        # Generate noise layer again to test caching
        self.generator.generate_noise_layer(noise_type="medium", scale=0.1)

        # Should use cached value, so no new calls to noise generator
        self.assertEqual(
            len(self.mock_noise_generator.calls),
            0,
            "Expected no new calls to noise generator when using cached values",
        )

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
        self.assertEqual(
            result.shape, (10, 10), "Result shape should match input grid shape"
        )

        # Test with a controlled grid and known rules
        self._test_conways_game_of_life_rules(test_generator)

    def _test_conways_game_of_life_rules(self, test_generator):
        """Helper method to test Conway's Game of Life rules."""
        # Create a test case where we know the expected outcome
        # For a 3x3 block in Conway's Game of Life, the corners die and the edges survive
        controlled_grid = np.zeros((10, 10))
        controlled_grid[4:7, 4:7] = 1  # 3x3 block

        # Create a binary version for testing
        binary_grid = (controlled_grid > 0).astype(np.int8)

        # Get the expected result after one iteration
        expected_grid = self._apply_conways_rules_manually(binary_grid)

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
        expected_result = controlled_grid * expected_grid
        np.testing.assert_array_equal(
            result,
            expected_result,
            "Cellular automaton result should match expected pattern after applying Conway's rules",
        )

    def _apply_conways_rules_manually(self, binary_grid):
        """Apply Conway's Game of Life rules manually to a grid."""
        grid_size = binary_grid.shape[0]  # Assuming square grid
        new_grid = binary_grid.copy()

        # Apply cellular automaton rules manually
        for y, x in itertools.product(range(grid_size), range(grid_size)):
            # Count live neighbors
            neighbors = self._count_live_neighbors(binary_grid, x, y, grid_size)

            # Apply rules
            new_grid[y, x] = self._apply_conway_rule(binary_grid[y, x], neighbors)

        return new_grid

    @staticmethod
    def _count_live_neighbors(grid, x, y, grid_size):
        """Count the number of live neighbors for a cell at (x, y)."""
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % grid_size, (y + dy) % grid_size  # Wrap around
                neighbors += grid[ny, nx]
        return neighbors

    @staticmethod
    def _apply_conway_rule(cell_state, neighbors):
        """Apply Conway's Game of Life rules to a cell."""
        if cell_state == 1:
            # Cell is alive
            return 1 if neighbors in {2, 3} else 0
        else:
            # Cell is dead
            return 1 if neighbors == 3 else 0

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
        self.assertEqual(
            result.shape, (10, 10), "Result shape should match input grid shape"
        )

        # Verify that some values are higher than the original
        self.assertGreater(
            np.max(result),
            1.0,
            "Expected at least some values to be greater than 1.0 after clustering",
        )

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
        # Should return a grid with the same shape
        self.assertEqual(result.shape, sparse_grid.shape)
        # Non-zero values should still be present
        self.assertGreater(result[0, 0], 0, "Expected non-zero value at position (0,0)")
        self.assertGreater(result[9, 9], 0, "Expected non-zero value at position (9,9)")

    def test_to_dict(self):
        """Test the to_dict method."""
        data = self.generator.to_dict()

        # Verify the dictionary contains all required fields
        self.assertEqual(
            data["entity_id"], "gen-123", "entity_id should match the initialized value"
        )
        self.assertEqual(
            data["entity_type"],
            "test_generator",
            "entity_type should match the initialized value",
        )
        self.assertEqual(data["seed"], 42, "seed should match the initialized value")
        self.assertEqual(data["width"], 50, "width should match the initialized value")
        self.assertEqual(
            data["height"], 60, "height should match the initialized value"
        )
        self.assertEqual(
            data["color"], (100, 200, 100), "color should match the initialized value"
        )
        self.assertEqual(
            data["position"], (5, 10), "position should match the initialized value"
        )
        self.assertEqual(
            data["parameters"],
            self.generator.parameters,
            "parameters should match the generator's parameters",
        )

    def test_generate_multi_octave_noise(self):
        """Test the generate_multi_octave_noise method."""
        # Create a new generator to avoid caching issues
        test_generator = BaseGenerator(
            entity_id="test-multi-octave",
            width=50,
            height=60,
            noise_generator=MockNoiseGenerator(return_value=0.5),
        )

        # Generate multi-octave noise
        noise_layer = test_generator.generate_multi_octave_noise(
            scale=0.2, octaves=[2, 4, 6], weights=[0.7, 0.2, 0.1]
        )

        # Verify the shape of the noise layer
        self.assertEqual(noise_layer.shape, (60, 50))  # (height, width)

        # Test with default parameters
        self.mock_noise_generator.calls = []  # Clear call history
        # Generate noise and capture for verification
        result_noise = self.generator.generate_multi_octave_noise(scale=0.1)

        # Verify shape of the result
        self.assertEqual(
            result_noise.shape,
            (60, 50),
            "Multi-octave noise should match generator dimensions",
        )

        # Verify default parameters were used
        last_call = self.mock_noise_generator.calls[-1]
        self.assertEqual(last_call[4], [3, 5, 8], "Default octaves should be [3, 5, 8]")
        self.assertEqual(
            last_call[5], [1.0, 0.5, 0.25], "Default weights should be [1.0, 0.5, 0.25]"
        )

        # Test caching
        self.mock_noise_generator.calls = []  # Clear call history
        # Generate noise layer again to test caching
        self.generator.generate_multi_octave_noise(scale=0.1)

        # Should use cached value, so no new calls to noise generator
        self.assertEqual(
            len(self.mock_noise_generator.calls),
            0,
            "Expected no new calls to noise generator when using cached values",
        )

    def test_apply_cellular_automaton_with_utils(self):
        """Test the apply_cellular_automaton method with utility module integration."""
        if not CA_UTILS_AVAILABLE:
            self.skipTest("Cellular automaton utils module not available")

        # Create a test generator with smaller dimensions
        test_generator = BaseGenerator(
            entity_id="test-ca-utils",
            width=10,
            height=10,
            noise_generator=self.mock_noise_generator,
        )

        # Create a simple grid with some cells
        grid = np.zeros((10, 10))
        grid[4:7, 4:7] = 1  # 3x3 block of cells in the middle

        # Mock the utility function to verify it's called
        with patch(
            "src.utils.cellular_automaton_utils.apply_cellular_automaton_optimized"
        ) as mock_ca:
            self._mock_grid_handler(grid, mock_ca, test_generator)

    def _mock_grid_handler(self, grid, mock_ca, test_generator):
        # Set up the mock to return a known grid
        mock_result = grid.copy()
        mock_result[5, 5] = 2  # Make a change we can detect
        mock_ca.return_value = mock_result

        # Apply cellular automaton with utility module
        result = test_generator.apply_cellular_automaton(
            grid=grid, birth_set={3}, survival_set={2, 3}, iterations=1, wrap=True
        )

        # Verify the utility function was called
        mock_ca.assert_called_once()
        self.assertIsNotNone(
            mock_ca.call_args,
            "Expected cellular automaton utility function to be called with arguments",
        )

        # Verify the result is what the mock returned
        np.testing.assert_array_equal(result, mock_result)

    def test_apply_cellular_automaton_fallback(self):
        """Test the apply_cellular_automaton method with fallback to internal implementation."""
        # Skip this test if we can't properly test the fallback
        if not CA_UTILS_AVAILABLE:
            self.skipTest("Cannot test fallback when utils are not available")

        # Create a test generator with smaller dimensions
        test_generator = BaseGenerator(
            entity_id="test-ca-fallback",
            width=10,
            height=10,
            noise_generator=self.mock_noise_generator,
        )

        # Create a simple grid with some cells
        grid = np.zeros((10, 10))
        grid[4:7, 4:7] = 1  # 3x3 block of cells in the middle

        # Apply cellular automaton without mocking to get expected result
        expected_shape = test_generator.apply_cellular_automaton(
            grid=grid, birth_set={3}, survival_set={2, 3}, iterations=1, wrap=True
        ).shape

        # Verify the result is the expected shape
        self.assertEqual(expected_shape, (10, 10))

    def test_create_clusters_with_utils(self):
        """Test the create_clusters method with utility module integration."""
        if not VALUE_GEN_AVAILABLE:
            self.skipTest("Value generator utils module not available")

        # Create a test generator with smaller dimensions
        test_generator = BaseGenerator(
            entity_id="test-clusters-utils",
            width=10,
            height=10,
            noise_generator=self.mock_noise_generator,
        )

        # Create a simple grid
        grid = np.ones((10, 10))

        # Mock the utility function to verify it's called
        with patch("src.utils.value_generator.add_value_clusters") as mock_clusters:
            self._known_grid_handler(grid, mock_clusters, test_generator)

    def _known_grid_handler(self, grid, mock_clusters, test_generator):
        # Set up the mock to return a known grid
        mock_result = grid.copy() * 1.5  # Make a change we can detect
        mock_clusters.return_value = mock_result

        # Apply clustering with utility module
        result = test_generator.create_clusters(
            grid=grid, num_clusters=3, cluster_value_multiplier=2.0
        )

        # Verify the utility function was called with correct parameters
        mock_clusters.assert_called_once()
        call_args = mock_clusters.call_args[0]
        call_kwargs = mock_clusters.call_args[1]

        # Check positional arguments
        np.testing.assert_array_equal(
            call_args[0], grid, "First positional argument should be the input grid"
        )

        # Check keyword arguments
        self.assertEqual(
            call_kwargs["num_clusters"],
            3,
            "num_clusters parameter should be passed correctly",
        )
        self.assertEqual(
            call_kwargs["cluster_value_multiplier"],
            2.0,
            "cluster_value_multiplier parameter should be passed correctly",
        )
        self.assertIn(
            "cluster_radius",
            call_kwargs,
            "cluster_radius parameter should be included in the function call",
        )

        # Verify the result is what the mock returned
        np.testing.assert_array_equal(result, mock_result)

    def test_create_clusters_fallback(self):
        """Test the create_clusters method with fallback to internal implementation."""
        # Skip this test if we can't properly test the fallback
        if not VALUE_GEN_AVAILABLE:
            self.skipTest("Cannot test fallback when utils are not available")

        # Create a test generator with smaller dimensions
        test_generator = BaseGenerator(
            entity_id="test-clusters-fallback",
            width=10,
            height=10,
            noise_generator=self.mock_noise_generator,
        )

        # Create a simple grid
        grid = np.ones((10, 10))

        # Apply clustering without mocking
        result = test_generator.create_clusters(
            grid=grid, num_clusters=3, cluster_value_multiplier=2.0
        )

        # Verify the result is the expected shape
        self.assertEqual(result.shape, (10, 10))

        # Verify that some values are higher than the original
        self.assertGreater(
            np.max(result),
            1.0,
            "Expected at least some values to be greater than 1.0 after clustering",
        )

    def test_parameter_validation_clusters(self):
        """Test parameter validation in the create_clusters method."""
        # Create a test generator with smaller dimensions
        test_generator = BaseGenerator(
            entity_id="test-validation-clusters",
            width=10,
            height=10,
            noise_generator=self.mock_noise_generator,
        )

        # Create a simple grid
        grid = np.ones((10, 10))

        # Test invalid num_clusters
        result = test_generator.create_clusters(
            grid=grid, num_clusters=-1, cluster_value_multiplier=2.0
        )
        # Verify the shape is correct
        self.assertEqual(
            result.shape,
            (10, 10),
            "Result shape should match input grid shape even with invalid parameters",
        )

        # Test invalid cluster_value_multiplier
        result = test_generator.create_clusters(
            grid=grid, num_clusters=3, cluster_value_multiplier=0
        )
        # Verify the shape is correct
        self.assertEqual(
            result.shape,
            (10, 10),
            "Result shape should match input grid shape even with invalid parameters",
        )

    def test_parallel_clustering(self):
        """Test the parallel clustering implementation."""
        # Create a test generator with larger dimensions to trigger parallel processing
        test_generator = self._create_large_test_generator()

        # Create a grid with non-zero values
        grid = np.ones((250, 250))

        # Mock the ProcessPoolExecutor to avoid multiprocessing issues in tests
        with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
            # Set up the mock
            self._setup_clustering_mock(mock_executor)

            # Call the create_clusters method which should use parallel processing
            result = test_generator.create_clusters(
                grid=grid, num_clusters=8, cluster_value_multiplier=2.0
            )

            # Verify the result has the expected shape
            self.assertEqual(
                result.shape,
                (250, 250),
                "Result shape should match input grid shape for parallel processing",
            )

    def _create_large_test_generator(self):
        """Create a test generator with large dimensions for parallel processing tests."""
        return BaseGenerator(
            entity_id="test-parallel",
            width=250,
            height=250,
            noise_generator=self.mock_noise_generator,
        )

    @staticmethod
    def _setup_clustering_mock(mock_executor):
        """Set up the mock for parallel clustering tests."""
        mock_future = MagicMock()
        mock_result = np.ones((250, 250)) * 1.5  # Simulated result with elevated values
        mock_future.result.return_value = mock_result
        mock_executor.return_value.__enter__.return_value.submit.return_value = (
            mock_future
        )

    def test_parallel_cellular_automaton(self):
        """Test the parallel cellular automaton implementation."""
        # Create a test generator with larger dimensions to trigger parallel processing
        test_generator = self._create_large_test_generator()

        # Create a controlled grid with a glider pattern
        controlled_grid = self._create_glider_grid()

        # Mock the ProcessPoolExecutor to avoid multiprocessing issues in tests
        with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
            # Set up the mock
            self._setup_ca_mock(mock_executor)

            # Call the apply_cellular_automaton method which should use parallel processing for large grids
            result = test_generator.apply_cellular_automaton(
                grid=controlled_grid,
                birth_set={3},
                survival_set={2, 3},
                iterations=1,
                wrap=True,
            )

            # Verify the result has the expected shape
            self.assertEqual(
                result.shape,
                (250, 250),
                "Result shape should match input grid shape for parallel processing",
            )

    @staticmethod
    def _create_glider_grid():
        """Create a grid with a glider pattern for cellular automaton tests."""
        controlled_grid = np.zeros((250, 250))
        glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
        controlled_grid[10:13, 10:13] = glider
        return controlled_grid

    @staticmethod
    def _setup_ca_mock(mock_executor):
        """Set up the mock for parallel cellular automaton tests."""
        mock_future = MagicMock()

        # Create a simulated result with the expected glider evolution
        simulated_result = np.zeros((250, 250))
        evolved_glider = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]])
        simulated_result[10:14, 10:13] = evolved_glider

        # Set up the mock to return chunks of the simulated result
        mock_future.result.return_value = (0, 250, simulated_result)
        mock_executor.return_value.__enter__.return_value.submit.return_value = (
            mock_future
        )


if __name__ == "__main__":
    unittest.main()
