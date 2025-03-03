#!/usr/bin/env python3
"""
Unit tests for the ProceduralGenerator class.
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
from generators.procedural_generator import (
    ProceduralGenerator,
    create_field_with_multiple_algorithms,
)
from entities.base_generator import BaseGenerator
from world.asteroid_field import AsteroidField


class TestProceduralGenerator(unittest.TestCase):
    """Test cases for the ProceduralGenerator class."""

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
        self.generator = ProceduralGenerator(
            entity_id="proc-123",
            seed=42,
            width=50,
            height=60,
            color=(100, 200, 100),
            position=(5, 10),
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.perlin_patcher.stop()

    def test_initialization(self):
        """Test that generator initializes with correct values."""
        # Test with explicit values
        self.assertEqual(self.generator.entity_id, "proc-123")
        self.assertEqual(self.generator.entity_type, "procedural")
        self.assertEqual(self.generator.seed, 42)
        self.assertEqual(self.generator.width, 50)
        self.assertEqual(self.generator.height, 60)
        self.assertEqual(self.generator.color, (100, 200, 100))
        self.assertEqual(self.generator.position, (5, 10))

        # Test inheritance
        self.assertIsInstance(self.generator, BaseGenerator)

    def test_default_initialization(self):
        """Test initialization with default values."""
        generator = ProceduralGenerator()

        # Test default values
        self.assertEqual(generator.entity_type, "procedural")
        self.assertEqual(generator.width, 100)
        self.assertEqual(generator.height, 100)
        self.assertEqual(generator.color, (100, 200, 100))
        self.assertIsNone(generator.position)

        # Seed should be set to a random value
        self.assertIsNotNone(generator.seed)

    def test_generate_asteroid_field(self):
        """Test the generate_asteroid_field method."""
        # Set up the mock to return a specific value
        self.perlin_mock.return_value = 0.5

        # Generate an asteroid field
        asteroid_grid = self.generator.generate_asteroid_field(density=0.3)

        # Verify the shape of the grid
        self.assertEqual(asteroid_grid.shape, (60, 50))  # (height, width)

        # Test with different parameters
        asteroid_grid = self.generator.generate_asteroid_field(
            density=0.5, noise_scale=0.1, threshold=0.4
        )
        self.assertEqual(asteroid_grid.shape, (60, 50))

    def test_generate_rare_minerals(self):
        """Test the generate_rare_minerals method."""
        # Create a simple asteroid grid
        asteroid_grid = np.ones((60, 50))

        # Generate rare minerals
        rare_grid = self.generator.generate_rare_minerals(
            asteroid_grid=asteroid_grid, rare_chance=0.2, rare_bonus=2.5
        )

        # Verify the shape of the grid
        self.assertEqual(rare_grid.shape, (60, 50))

        # Test with different parameters
        rare_grid = self.generator.generate_rare_minerals(
            asteroid_grid=asteroid_grid,
            rare_chance=0.5,
            rare_bonus=3.0,
            anomaly_chance=0.1,
        )
        self.assertEqual(rare_grid.shape, (60, 50))

    def test_generate_energy_sources(self):
        """Test the generate_energy_sources method."""
        # Create simple grids
        asteroid_grid = np.ones((60, 50))
        rare_grid = np.zeros((60, 50))

        # Generate energy sources
        energy_grid = self.generator.generate_energy_sources(
            asteroid_grid=asteroid_grid, rare_grid=rare_grid, energy_chance=0.1
        )

        # Verify the shape of the grid
        self.assertEqual(energy_grid.shape, (60, 50))

        # Test with different parameters
        energy_grid = self.generator.generate_energy_sources(
            asteroid_grid=asteroid_grid,
            rare_grid=rare_grid,
            energy_chance=0.3,
            energy_value=5.0,
        )
        self.assertEqual(energy_grid.shape, (60, 50))

    def test_generate_multi_layer_asteroid_field(self):
        """Test the generate_multi_layer_asteroid_field method."""
        # Generate a multi-layer asteroid field
        asteroid_grid = self.generator.generate_multi_layer_asteroid_field(
            density=0.3, noise_scale=0.1, threshold=0.4
        )

        # Verify the shape of the grid
        self.assertEqual(asteroid_grid.shape, (60, 50))

        # Test with different parameters
        asteroid_grid = self.generator.generate_multi_layer_asteroid_field(
            density=0.5, noise_scale=0.2, threshold=0.3, octaves=[3, 5, 8]
        )
        self.assertEqual(asteroid_grid.shape, (60, 50))

    def test_generate_tiered_mineral_distribution(self):
        """Test the generate_tiered_mineral_distribution method."""
        # Create a simple asteroid grid
        asteroid_grid = np.ones((60, 50))

        # Generate tiered mineral distribution
        rare_grid = self.generator.generate_tiered_mineral_distribution(
            asteroid_grid=asteroid_grid, rare_chance=0.2, rare_bonus=2.5
        )

        # Verify the shape of the grid
        self.assertEqual(rare_grid.shape, (60, 50))

        # Test with different parameters
        rare_grid = self.generator.generate_tiered_mineral_distribution(
            asteroid_grid=asteroid_grid, rare_chance=0.5, rare_bonus=3.0, tiers=4
        )
        self.assertEqual(rare_grid.shape, (60, 50))

    def test_create_field_with_multiple_algorithms(self):
        """Test the create_field_with_multiple_algorithms function."""
        # Mock AsteroidField
        asteroid_field_mock = MagicMock(spec=AsteroidField)

        # Patch the AsteroidField class
        with patch(
            "generators.procedural_generator.AsteroidField",
            return_value=asteroid_field_mock,
        ):
            # Create a field
            field = create_field_with_multiple_algorithms(
                width=80, height=90, seed=123, rare_chance=0.15, rare_bonus=2.5
            )

            # Verify the field is created
            self.assertEqual(field, asteroid_field_mock)

            # Test with different parameters
            field = create_field_with_multiple_algorithms(
                width=100,
                height=100,
                seed=456,
                rare_chance=0.2,
                rare_bonus=3.0,
                energy_chance=0.1,
            )
            self.assertEqual(field, asteroid_field_mock)


if __name__ == "__main__":
    unittest.main()
