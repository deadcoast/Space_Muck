#!/usr/bin/env python3
"""
Unit tests for the AsteroidGenerator class.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock modules before importing entities
sys.modules["perlin_noise"] = MagicMock()
sys.modules["perlin_noise"].PerlinNoise = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.ndimage"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["skimage"] = MagicMock()
sys.modules["skimage.measure"] = MagicMock()

# Mock src modules
sys.modules["src.utils.logging_setup"] = MagicMock()
sys.modules["src.utils.logging_setup"].log_performance_start = MagicMock()
sys.modules["src.utils.logging_setup"].log_performance_end = MagicMock()
sys.modules["src.utils.logging_setup"].log_exception = MagicMock()
sys.modules["src.utils.logging_setup"].LogContext = MagicMock()

# Import the classes to test
from generators.asteroid_generator import AsteroidGenerator
from entities.base_generator import BaseGenerator
from world.asteroid_field import AsteroidField


class TestAsteroidGenerator(unittest.TestCase):
    """Test cases for the AsteroidGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock for PerlinNoise
        self.perlin_mock = MagicMock()
        self.perlin_mock.return_value = 0.5

        # Patch the PerlinNoise class
        self.perlin_patcher = patch(
            "perlin_noise.PerlinNoise", return_value=self.perlin_mock
        )
        self.perlin_patcher.start()

        # Create a generator for testing
        self.generator = AsteroidGenerator(
            entity_id="ast-123",
            seed=42,
            width=50,
            height=60,
            color=(150, 150, 100),
            position=(5, 10),
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.perlin_patcher.stop()

    def test_initialization(self):
        """Test that generator initializes with correct values."""
        # Test with explicit values
        self.assertEqual(self.generator.entity_id, "ast-123")
        self.assertEqual(self.generator.entity_type, "asteroid")
        self.assertEqual(self.generator.seed, 42)
        self.assertEqual(self.generator.width, 50)
        self.assertEqual(self.generator.height, 60)
        self.assertEqual(self.generator.color, (150, 150, 100))
        self.assertEqual(self.generator.position, (5, 10))

        # Test inheritance
        self.assertIsInstance(self.generator, BaseGenerator)

    def test_default_initialization(self):
        """Test initialization with default values."""
        generator = AsteroidGenerator()

        # Test default values
        self.assertEqual(generator.entity_type, "asteroid")
        self.assertEqual(generator.width, 100)
        self.assertEqual(generator.height, 100)
        self.assertEqual(generator.color, (150, 150, 100))
        self.assertIsNone(generator.position)

        # Seed should be set to a random value
        self.assertIsNotNone(generator.seed)

    def test_generate_asteroid_belt(self):
        """Test the generate_asteroid_belt method."""
        # Generate an asteroid belt
        asteroid_grid = self.generator.generate_asteroid_belt(
            center_distance=0.5, belt_width=0.2, density=0.7
        )

        # Verify the shape of the grid
        self.assertEqual(asteroid_grid.shape, (60, 50))  # (height, width)

        # Test with different parameters
        asteroid_grid = self.generator.generate_asteroid_belt(
            center_distance=0.3, belt_width=0.1, density=0.5, noise_scale=0.05
        )
        self.assertEqual(asteroid_grid.shape, (60, 50))

    def test_generate_asteroid_cluster(self):
        """Test the generate_asteroid_cluster method."""
        # Generate an asteroid cluster
        asteroid_grid = self.generator.generate_asteroid_cluster(
            num_clusters=3, cluster_size=10, density=0.7
        )

        # Verify the shape of the grid
        self.assertEqual(asteroid_grid.shape, (60, 50))

        # Test with different parameters
        asteroid_grid = self.generator.generate_asteroid_cluster(
            num_clusters=5, cluster_size=15, density=0.8, noise_scale=0.05
        )
        self.assertEqual(asteroid_grid.shape, (60, 50))

    def test_generate_asteroid_field(self):
        """Test the generate_asteroid_field method."""
        # Generate an asteroid field
        asteroid_grid = self.generator.generate_asteroid_field(
            field_type="belt", density=0.7
        )

        # Verify the shape of the grid
        self.assertEqual(asteroid_grid.shape, (60, 50))

        # Test with "cluster" field type
        asteroid_grid = self.generator.generate_asteroid_field(
            field_type="cluster", density=0.6
        )
        self.assertEqual(asteroid_grid.shape, (60, 50))

        # Test with "mixed" field type
        asteroid_grid = self.generator.generate_asteroid_field(
            field_type="mixed", density=0.5
        )
        self.assertEqual(asteroid_grid.shape, (60, 50))

        # Test with invalid field type (should default to "mixed")
        asteroid_grid = self.generator.generate_asteroid_field(
            field_type="invalid", density=0.4
        )
        self.assertEqual(asteroid_grid.shape, (60, 50))

    def test_generate_mineral_distribution(self):
        """Test the generate_mineral_distribution method."""
        # Create a simple asteroid grid
        asteroid_grid = np.ones((60, 50))

        # Generate mineral distribution
        mineral_grid = self.generator.generate_mineral_distribution(
            asteroid_grid=asteroid_grid, rare_chance=0.2, rare_bonus=2.5
        )

        # Verify the shape of the grid
        self.assertEqual(mineral_grid.shape, (60, 50))

        # Test with different parameters
        mineral_grid = self.generator.generate_mineral_distribution(
            asteroid_grid=asteroid_grid,
            rare_chance=0.3,
            rare_bonus=3.0,
            distribution_type="clustered",
        )
        self.assertEqual(mineral_grid.shape, (60, 50))

        # Test with "gradient" distribution type
        mineral_grid = self.generator.generate_mineral_distribution(
            asteroid_grid=asteroid_grid,
            rare_chance=0.4,
            rare_bonus=2.0,
            distribution_type="gradient",
        )
        self.assertEqual(mineral_grid.shape, (60, 50))

        # Test with invalid distribution type (should default to "random")
        mineral_grid = self.generator.generate_mineral_distribution(
            asteroid_grid=asteroid_grid,
            rare_chance=0.1,
            rare_bonus=1.5,
            distribution_type="invalid",
        )
        self.assertEqual(mineral_grid.shape, (60, 50))

    def test_generate_energy_field(self):
        """Test the generate_energy_field method."""
        # Create simple grids
        asteroid_grid = np.ones((60, 50))
        mineral_grid = np.zeros((60, 50))

        # Generate energy field
        energy_grid = self.generator.generate_energy_field(
            asteroid_grid=asteroid_grid, mineral_grid=mineral_grid, energy_chance=0.1
        )

        # Verify the shape of the grid
        self.assertEqual(energy_grid.shape, (60, 50))

        # Test with different parameters
        energy_grid = self.generator.generate_energy_field(
            asteroid_grid=asteroid_grid,
            mineral_grid=mineral_grid,
            energy_chance=0.2,
            energy_value=3.0,
            energy_type="radiation",
        )
        self.assertEqual(energy_grid.shape, (60, 50))

        # Test with "plasma" energy type
        energy_grid = self.generator.generate_energy_field(
            asteroid_grid=asteroid_grid,
            mineral_grid=mineral_grid,
            energy_chance=0.3,
            energy_value=4.0,
            energy_type="plasma",
        )
        self.assertEqual(energy_grid.shape, (60, 50))

        # Test with invalid energy type (should default to "standard")
        energy_grid = self.generator.generate_energy_field(
            asteroid_grid=asteroid_grid,
            mineral_grid=mineral_grid,
            energy_chance=0.4,
            energy_value=5.0,
            energy_type="invalid",
        )
        self.assertEqual(energy_grid.shape, (60, 50))

    def test_create_asteroid_field(self):
        """Test the create_asteroid_field method."""
        # Mock AsteroidField
        asteroid_field_mock = MagicMock(spec=AsteroidField)

        # Patch the AsteroidField class
        with patch(
            "generators.asteroid_generator.AsteroidField",
            return_value=asteroid_field_mock,
        ):
            # Create a field
            field = self.generator.create_asteroid_field(
                field_type="belt", density=0.7, rare_chance=0.2, rare_bonus=2.5
            )

            # Verify the field is created
            self.assertEqual(field, asteroid_field_mock)

            # Test with different parameters
            field = self.generator.create_asteroid_field(
                field_type="cluster",
                density=0.6,
                rare_chance=0.3,
                rare_bonus=3.0,
                energy_chance=0.1,
            )
            self.assertEqual(field, asteroid_field_mock)


if __name__ == "__main__":
    unittest.main()
