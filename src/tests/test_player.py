"""
Test module for the Player class.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the dependencies before importing
sys.modules["perlin_noise"] = MagicMock()
sys.modules["sklearn.cluster"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["pygame"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()

from src.entities.player import Player


import numpy as np


class MockField:
    """Mock asteroid field for testing."""

    def __init__(self):
        """Initialize a mock field."""
        self.width = 100
        self.height = 100
        self.grid = np.zeros((self.height, self.width))
        self.rare_grid = np.zeros((self.height, self.width))
        self.rare_bonus_multiplier = 2.0

        # Add some test asteroids
        self.grid[50, 50] = 100  # Regular asteroid
        self.grid[60, 60] = 200  # Rare asteroid
        self.rare_grid[60, 60] = 1
        self.grid[70, 70] = 300  # Anomaly
        self.rare_grid[70, 70] = 2


class TestPlayer(unittest.TestCase):
    """Test cases for the Player class."""

    def setUp(self):
        """Set up test fixtures."""
        self.player = Player()
        self.field = MockField()
        self.player.territory_center = (10, 10)

    def test_player_attributes(self):
        """Test that Player has the correct attributes."""
        self.assertTrue(self.player.is_player)
        self.assertEqual(self.player.credits, 1000)
        self.assertEqual(self.player.ship_level, 1)
        self.assertEqual(self.player.mining_speed, 1.0)
        self.assertEqual(self.player.inventory, {})
        self.assertEqual(self.player.discovered_anomalies, set())
        self.assertIsNone(self.player.current_quest)
        self.assertEqual(self.player.completed_quests, [])
        self.assertEqual(self.player.trait, "adaptive")
        self.assertEqual(self.player.mining_efficiency, 0.8)

    def test_move(self):
        """Test player movement."""
        # Test valid move
        result = self.player.move(5, 5, self.field)
        self.assertTrue(result)
        self.assertEqual(self.player.territory_center, (15, 15))

        # Test invalid move (out of bounds)
        self.player.territory_center = (95, 95)
        result = self.player.move(10, 10, self.field)
        self.assertFalse(result)
        self.assertEqual(self.player.territory_center, (95, 95))

    def test_mine(self):
        """Test mining functionality."""
        # Test mining empty space
        result = self.player.mine(10, 10, self.field)
        self.assertFalse(result["success"])

        # Test mining regular asteroid
        result = self.player.mine(50, 50, self.field)
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "common")
        self.assertEqual(result["value"], 80)  # 100 * 0.8 mining efficiency

        # Test mining rare asteroid
        result = self.player.mine(60, 60, self.field)
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "rare")
        self.assertEqual(result["value"], 320)  # 200 * 0.8 * 2.0 rare bonus

        # Test mining anomaly
        result = self.player.mine(70, 70, self.field)
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "anomaly")
        self.assertEqual(result["value"], 960)  # 300 * 0.8 * 2.0 * 2 anomaly bonus
        self.assertIn("anomaly_70_70", self.player.discovered_anomalies)


if __name__ == "__main__":
    unittest.main()
