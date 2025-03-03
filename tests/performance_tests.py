"""
Performance tests for Space Muck.

These tests verify that the game maintains acceptable performance
under various conditions.
"""

import os
import sys
import time
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import *
from src.world.asteroid_field import AsteroidField
from src.entities.player import Player
from src.entities.miner_entity import MinerEntity
from src.world.procedural_generation import generate_field


class TestAsteroidFieldPerformance(unittest.TestCase):
    """Test performance of the AsteroidField class."""

    def setUp(self):
        """Set up test environment."""
        # Create a large field for performance testing
        self.field = AsteroidField(width=400, height=300)

    def test_update_performance(self):
        """Test performance of the update method."""
        # Time the update method
        start_time = time.time()
        self.field.update()
        end_time = time.time()

        # Update should complete in reasonable time
        self.assertLess(end_time - start_time, 0.5, "Update took too long")

    def test_update_entities_performance(self):
        """Test performance of the update_entities method."""
        # Add several races to test with
        for i in range(3):
            race = MinerEntity(race_id=i + 1, color=(255, 0, 0))
            self.field.races.append(race)

        # Add entities to the field
        entities_per_race = 1000
        for race in self.field.races:
            coords = np.random.randint(
                0, min(self.field.width, self.field.height), size=(entities_per_race, 2)
            )
            for x, y in coords:
                if x < self.field.width and y < self.field.height:
                    self.field.entity_grid[y, x] = race.race_id

        # Time the update_entities method
        start_time = time.time()
        self.field.update_entities()
        end_time = time.time()

        # Should complete in reasonable time even with many entities
        self.assertLess(end_time - start_time, 2.0, "Entity update took too long")

    def test_procedural_generation_performance(self):
        """Test performance of procedural field generation."""
        width, height = 400, 300

        # Time the generation process
        start_time = time.time()
        field_data = generate_field(width, height)
        end_time = time.time()

        # Generation should complete in reasonable time
        self.assertLess(end_time - start_time, 5.0, "Field generation took too long")

        # Verify we got valid data
        self.assertIsNotNone(field_data)
        self.assertIn("grid", field_data)
        self.assertEqual(field_data["grid"].shape, (height, width))


class TestPlayerPerformance(unittest.TestCase):
    """Test performance of player operations."""

    def setUp(self):
        """Set up test environment."""
        self.field = AsteroidField(width=400, height=300)
        self.player = Player()

        # Initialize a densely populated field for testing
        self.field.grid.fill(10)  # Fill with asteroids
        self.field.rare_grid[::10, ::10] = 1  # Some rare asteroids

    def test_mining_performance(self):
        """Test performance of mining operations."""
        # Position player
        self.player.x = 200
        self.player.y = 150

        # Time mining operations
        start_time = time.time()
        for _ in range(100):  # Mine 100 times
            self.player.mine(self.field)
        end_time = time.time()

        # Mining should be fast enough
        avg_time = (end_time - start_time) / 100
        self.assertLess(avg_time, 0.01, "Mining operation too slow")

    def test_fleet_update_performance(self):
        """Test performance of fleet updates with many ships."""
        # Add maximum ships
        self.player.mining_ships = self.player.max_mining_ships
        self.player.ship_positions = list(
            zip(
                np.random.randint(0, self.field.width, self.player.max_mining_ships),
                np.random.randint(0, self.field.height, self.player.max_mining_ships),
            )
        )
        self.player.ship_health = [100] * self.player.max_mining_ships

        # Time fleet update
        start_time = time.time()
        self.player.update_fleet(self.field)
        end_time = time.time()

        # Fleet update should be reasonably fast
        self.assertLess(end_time - start_time, 0.1, "Fleet update too slow")

    def test_pathfinding_performance(self):
        """Test performance of pathfinding algorithm."""
        # Create start and end points
        start = (10, 10)
        end = (390, 290)  # Far corner

        # Time pathfinding
        start_time = time.time()
        path = self.player.generate_path(start, end, self.field)
        end_time = time.time()

        # Pathfinding should complete in reasonable time
        self.assertLess(end_time - start_time, 1.0, "Pathfinding took too long")
        self.assertTrue(len(path) > 0, "No path found")


class TestUIPerformance(unittest.TestCase):
    """Test UI component performance."""

    @patch("pygame.Surface")
    def setUp(self, mock_surface):
        """Set up test environment with mocked pygame."""
        self.mock_surface = mock_surface.return_value
        self.notifier = NotificationManager()
        self.shop = Shop()

    def test_notification_performance_many_notifications(self):
        """Test notification system with many notifications."""
        # Add many notifications
        for i in range(500):
            self.notifier.add(f"Test notification {i}", category="system")

        # Time the update and draw operations
        start_time = time.time()
        self.notifier.update()
        self.notifier.draw(self.mock_surface)
        end_time = time.time()

        # UI operations should remain responsive
        self.assertLess(end_time - start_time, 0.1, "Notification operations too slow")
