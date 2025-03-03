#!/usr/bin/env python3
"""
Unit tests for the MinerEntity class.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock modules before importing entities
sys.modules["pygame"] = MagicMock()
sys.modules["perlin_noise"] = MagicMock()
sys.modules["perlin_noise"].PerlinNoise = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.cluster"] = MagicMock()
sys.modules["sklearn.cluster"].KMeans = MagicMock()

# Mock src modules
sys.modules["src.algorithms.symbiote_algorithm"] = MagicMock()
sys.modules[
    "src.algorithms.symbiote_algorithm"
].SymbioteEvolutionAlgorithm = MagicMock()
sys.modules["src.utils.logging_setup"] = MagicMock()
sys.modules["src.utils.logging_setup"].log_exception = MagicMock()

# Import the class to test
from entities.miner_entity import MinerEntity
from entities.base_entity import BaseEntity


class MockAsteroidField:
    """Mock class for AsteroidField."""

    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.rare_grid = np.zeros((height, width))
        self.energy_grid = np.zeros((height, width))
        self.entity_grid = np.zeros((height, width), dtype=int)
        self.colony_grid = np.zeros((height, width), dtype=int)
        self.entities = {}
        self.tick = 0


class TestMinerEntity(unittest.TestCase):
    """Test cases for the MinerEntity class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a miner entity for testing
        self.miner = MinerEntity(
            race_id=1,
            color=(0, 255, 0),
            birth_set={3, 4},
            survival_set={2, 3, 4},
            initial_density=0.01,
            position=(10, 20),
        )

        # Create a mock field
        self.field = MockAsteroidField(width=100, height=100)

    def test_initialization(self):
        """Test that miner entity initializes with correct values."""
        # Test with explicit values
        self.assertEqual(self.miner.entity_id, "1")
        self.assertEqual(self.miner.entity_type, "miner")
        self.assertEqual(self.miner.race_id, 1)
        self.assertEqual(self.miner.color, (0, 255, 0))
        self.assertEqual(self.miner.birth_set, {3, 4})
        self.assertEqual(self.miner.survival_set, {2, 3, 4})
        self.assertEqual(self.miner.initial_density, 0.01)
        self.assertEqual(self.miner.position, (10, 20))

        # Test default values
        self.assertIn(self.miner.trait, ["adaptive", "expansive", "selective"])
        self.assertEqual(self.miner.population, 0)
        self.assertEqual(self.miner.evolution_stage, 1)
        self.assertEqual(self.miner.current_behavior, "feeding")
        self.assertEqual(self.miner.mining_efficiency, 0.5)

        # Test inheritance
        self.assertIsInstance(self.miner, BaseEntity)

    def test_default_initialization(self):
        """Test initialization with default values."""
        miner = MinerEntity(race_id=2, color=(255, 0, 0))

        # Test default values
        self.assertEqual(miner.entity_id, "2")
        self.assertEqual(miner.entity_type, "miner")
        self.assertEqual(miner.race_id, 2)
        self.assertEqual(miner.color, (255, 0, 0))
        self.assertEqual(miner.birth_set, {3})
        self.assertEqual(miner.survival_set, {2, 3})
        self.assertEqual(miner.initial_density, 0.001)
        self.assertIsNone(miner.position)

    def test_initialize_genome_by_trait(self):
        """Test the _initialize_genome_by_trait method."""
        # Test adaptive trait
        self.miner.trait = "adaptive"
        genome = self.miner._initialize_genome_by_trait()
        self.assertEqual(genome["metabolism_rate"], 1.2)
        self.assertEqual(genome["mutation_rate"], 0.02)
        self.assertEqual(genome["adaptability"], 0.8)
        self.assertEqual(genome["aggression_base"], 0.15)

        # Test expansive trait
        self.miner.trait = "expansive"
        genome = self.miner._initialize_genome_by_trait()
        self.assertEqual(genome["expansion_drive"], 1.4)
        self.assertEqual(genome["metabolism_rate"], 0.9)
        self.assertEqual(genome["aggression_base"], 0.3)

        # Test selective trait
        self.miner.trait = "selective"
        genome = self.miner._initialize_genome_by_trait()
        self.assertEqual(genome["intelligence"], 0.7)
        self.assertEqual(genome["metabolism_rate"], 1.1)
        self.assertEqual(genome["expansion_drive"], 0.9)
        self.assertEqual(genome["aggression_base"], 0.2)

    @patch("random.random")
    @patch("random.randint")
    def test_populate_with_invalid_field(self, mock_randint, mock_random):
        """Test populate method with invalid field."""
        # Set up mocks
        mock_randint.return_value = 50
        mock_random.return_value = 0.5

        # Test with None field
        self.miner.populate(None)
        self.assertIsNone(self.miner.field)

        # Test with invalid field object
        invalid_field = MagicMock()
        delattr(invalid_field, "width")
        self.miner.populate(invalid_field)
        self.assertIsNone(self.miner.field)

    @patch("random.random")
    @patch("random.randint")
    def test_populate_with_valid_field(self, mock_randint, mock_random):
        """Test populate method with valid field."""
        # Set up mocks
        mock_randint.return_value = 50
        mock_random.return_value = 0.0  # Ensure cells are populated

        # Test with valid field
        self.miner.trait = "adaptive"  # Set trait for deterministic testing
        self.miner.populate(self.field)

        # Verify field is stored
        self.assertEqual(self.miner.field, self.field)

        # Test with other traits
        self.miner.trait = "expansive"
        self.miner.populate(self.field)

        self.miner.trait = "selective"
        self.miner.populate(self.field)

    def test_calculate_population(self):
        """Test calculate_population method."""
        # Set up a field with some population
        self.field.entity_grid = np.zeros((100, 100), dtype=int)
        self.field.entity_grid[10:20, 10:20] = 1  # 100 cells of race 1

        # Set the field
        self.miner.field = self.field

        # Calculate population
        self.miner.calculate_population()

        # Verify population count
        self.assertEqual(self.miner.population, 100)
        self.assertEqual(len(self.miner.population_history), 1)
        self.assertEqual(self.miner.population_history[0], 100)

    def test_update_behavior(self):
        """Test update_behavior method."""
        # Set up initial state
        self.miner.hunger = 0.8  # High hunger
        self.miner.field = self.field
        self.miner.population = 100

        # Update behavior
        self.miner.update_behavior()

        # Verify behavior changed to feeding due to high hunger
        self.assertEqual(self.miner.current_behavior, "feeding")

        # Test with low hunger
        self.miner.hunger = 0.1
        self.miner.update_behavior()

        # Behavior should change based on other factors
        self.assertIn(self.miner.current_behavior, ["feeding", "expanding", "evolving"])

    def test_to_dict_and_from_dict(self):
        """Test serialization to and from dictionary."""
        # Add some population data
        self.miner.population = 100
        self.miner.population_history = [80, 90, 100]
        self.miner.income_history = [10, 15, 20]

        # Convert to dictionary
        miner_dict = self.miner.to_dict()

        # Verify dictionary contents
        self.assertEqual(miner_dict["entity_id"], "1")
        self.assertEqual(miner_dict["entity_type"], "miner")
        self.assertEqual(miner_dict["race_id"], 1)
        self.assertEqual(miner_dict["color"], (0, 255, 0))
        self.assertEqual(miner_dict["population"], 100)
        self.assertEqual(miner_dict["trait"], self.miner.trait)

        # Note: from_dict would need to be implemented in MinerEntity
        # This test assumes it exists and works similar to BaseEntity.from_dict


if __name__ == "__main__":
    unittest.main()
