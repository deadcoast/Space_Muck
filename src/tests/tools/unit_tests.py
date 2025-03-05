"""
Unit tests for the Space Muck game.

This module contains tests for the core game components:
- AsteroidField
- Player
- MinerEntity
- Shop
- NotificationManager
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Add the src directory to the path so we can import game modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import *
from src.generators import AsteroidField
from src.entities.player import Player
from src.entities.miner_entity import MinerEntity
from src.ui.shop import Shop
from src.ui.notification import NotificationManager


class TestAsteroidField(unittest.TestCase):
    """Test suite for the AsteroidField class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a small field for faster testing
        self.field = AsteroidField(width=50, height=40)

        # Initialize with known values for testing
        self.field.grid = np.zeros((40, 50), dtype=np.int16)
        self.field.rare_grid = np.zeros((40, 50), dtype=np.int8)
        self.field.energy_grid = np.zeros((40, 50), dtype=np.float32)
        self.field.entity_grid = np.zeros((40, 50), dtype=np.int8)

        # Set some test values
        self.field.grid[10:20, 10:20] = 100  # Add a block of asteroids
        self.field.rare_grid[15:18, 15:18] = 1  # Some rare asteroids
        self.field.energy_grid[5:15, 5:15] = 0.5  # Add some energy

    def test_initialization(self):
        """Test if the field initializes properly."""
        field = AsteroidField(width=100, height=80)
        self.assertEqual(field.width, 100)
        self.assertEqual(field.height, 80)
        self.assertTrue(hasattr(field, "grid"))
        self.assertTrue(hasattr(field, "rare_grid"))
        self.assertTrue(hasattr(field, "energy_grid"))
        self.assertTrue(hasattr(field, "entity_grid"))

    def test_get_view_bounds(self):
        """Test if view bounds are calculated correctly."""
        self.field.camera_x = 25
        self.field.camera_y = 20
        self.field.zoom = 1.0

        view_x1, view_y1, view_x2, view_y2 = self.field.get_view_bounds()

        # Expected bounds based on center and VIEW_WIDTH/HEIGHT
        expected_x1 = max(0, 25 - VIEW_WIDTH // 2)
        expected_y1 = max(0, 20 - VIEW_HEIGHT // 2)
        expected_x2 = min(50, expected_x1 + VIEW_WIDTH)
        expected_y2 = min(40, expected_y1 + VIEW_HEIGHT)

        self.assertEqual(view_x1, expected_x1)
        self.assertEqual(view_y1, expected_y1)
        self.assertEqual(view_x2, expected_x2)
        self.assertEqual(view_y2, expected_y2)

    def test_manual_seed(self):
        """Test if manual seeding creates asteroids correctly."""
        # Clear the field first
        self.field.grid.fill(0)
        self.field.rare_grid.fill(0)

        # Seed at center with radius 3
        self.field.manual_seed(25, 20, radius=3)

        # Check that asteroids were created
        asteroid_count = np.sum(self.field.grid > 0)
        self.assertGreater(asteroid_count, 0)

        # Check seeding location (should have asteroids in a radius around x=25, y=20)
        self.assertTrue(np.any(self.field.grid[17:24, 22:29] > 0))

    def test_update(self):
        """Test if field updates according to cellular automaton rules."""
        # Save initial grid state
        initial_grid = self.field.grid.copy()
        initial_rare_grid = self.field.rare_grid.copy()

        # Update the field
        self.field.update()

        # Grid should change after update
        self.assertFalse(np.array_equal(initial_grid, self.field.grid))

    def test_update_entities(self):
        """Test entity update logic."""
        # Add a test race to the field
        test_race = MinerEntity(
            race_id=1,
            color=(255, 0, 0),
            birth_set={3},
            survival_set={2, 3},
            initial_density=0.1,
        )
        self.field.races = [test_race]

        # Add some entities
        self.field.entity_grid[5:10, 5:10] = 1

        # Run entity update
        race_income = self.field.update_entities()

        # Should return income info for the race
        self.assertIn(1, race_income)


class TestPlayer(unittest.TestCase):
    """Test suite for the Player class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.player = Player()
        self.field = AsteroidField(width=50, height=40)

        # Add some asteroids for testing mining
        self.field.grid[20:25, 20:25] = 100
        self.field.rare_grid[22:24, 22:24] = 1

    def test_initialization(self):
        """Test if player initializes correctly."""
        self.assertEqual(self.player.x, GRID_WIDTH // 2)
        self.assertEqual(self.player.y, GRID_HEIGHT // 2)
        self.assertEqual(self.player.mining_efficiency, 1.0)
        self.assertEqual(self.player.mining_range, 1)
        self.assertEqual(self.player.currency, 100)
        self.assertEqual(self.player.mining_ships, 1)

    def test_move(self):
        """Test player movement."""
        initial_x = self.player.x
        initial_y = self.player.y

        # Move right and down
        self.player.move(5, 3, self.field)

        # Check new position
        self.assertEqual(self.player.x, initial_x + 5)
        self.assertEqual(self.player.y, initial_y + 3)

    def test_move_boundary(self):
        """Test player movement respects field boundaries."""
        # Try to move outside the field
        self.player.x = 0
        self.player.y = 0
        self.player.move(-10, -10, self.field)

        # Should stay at boundary
        self.assertEqual(self.player.x, 0)
        self.assertEqual(self.player.y, 0)

    def test_mine(self):
        """Test mining functionality."""
        # Position player near asteroids
        self.player.x = 22
        self.player.y = 22

        # Mining should return value and update asteroid field
        initial_currency = self.player.currency
        initial_asteroids = np.sum(self.field.grid > 0)

        mined_value = self.player.mine(self.field)

        # Check results
        self.assertGreater(mined_value, 0)
        self.assertEqual(self.player.currency, initial_currency + mined_value)
        self.assertLess(np.sum(self.field.grid > 0), initial_asteroids)

    def test_auto_mine(self):
        """Test auto-mining functionality."""
        # Position player near asteroids
        self.player.x = 22
        self.player.y = 22
        self.player.auto_miners = 2

        # Auto-mining should return value and update asteroid field
        initial_currency = self.player.currency

        mined_value = self.player.auto_mine(self.field)

        # Check results
        self.assertGreater(mined_value, 0)
        self.assertEqual(self.player.currency, initial_currency + mined_value)

    def test_update_fleet(self):
        """Test fleet update logic."""
        # Add a test race that will attack ships
        test_race = MinerEntity(race_id=1, color=(255, 0, 0))
        test_race.hunger = 1.0  # Very hungry, will attack
        self.field.races = [test_race]

        # Place race entities near the player's ships
        self.player.x = 25
        self.player.y = 20
        self.player.ship_positions = [(25, 20), (26, 21)]
        self.player.ship_health = [100, 100]
        self.player.mining_ships = 2

        # Place entities near ships
        self.field.entity_grid[19:22, 24:27] = 1

        # Update fleet
        results = self.player.update_fleet(self.field)

        # Should have mined some minerals and possibly taken damage
        self.assertIn("minerals_mined", results)
        self.assertIn("damage_taken", results)
        self.assertIn("ships_lost", results)

    def test_add_ship(self):
        """Test adding ships to the fleet."""
        initial_ships = self.player.mining_ships

        # Add a ship
        self.player.currency = 1000  # Ensure we have enough currency
        result = self.player.add_ship()

        # Check results
        self.assertTrue(result)
        self.assertEqual(self.player.mining_ships, initial_ships + 1)
        self.assertEqual(len(self.player.ship_positions), initial_ships + 1)
        self.assertEqual(len(self.player.ship_health), initial_ships + 1)
        self.assertLess(self.player.currency, 1000)  # Should have spent some currency


class TestMinerEntity(unittest.TestCase):
    """Test suite for the MinerEntity class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.entity = MinerEntity(
            race_id=1,
            color=(255, 0, 0),
            birth_set={3},
            survival_set={2, 3},
            initial_density=0.1,
        )

    def test_initialization(self):
        """Test if entity initializes correctly."""
        self.assertEqual(self.entity.race_id, 1)
        self.assertEqual(self.entity.color, (255, 0, 0))
        self.assertEqual(self.entity.birth_set, {3})
        self.assertEqual(self.entity.survival_set, {2, 3})
        self.assertEqual(self.entity.initial_density, 0.1)
        self.assertTrue(hasattr(self.entity, "genome"))
        self.assertTrue(hasattr(self.entity, "trait"))

    def test_process_minerals(self):
        """Test mineral processing and effects on hunger and population."""
        initial_hunger = self.entity.hunger
        initial_population = self.entity.population

        # Process a batch of minerals
        minerals = {"common": 100, "rare": 10}
        pop_change, mutations = self.entity.process_minerals(minerals)

        # Check effects
        self.assertLess(self.entity.hunger, initial_hunger)  # Hunger should decrease
        self.assertNotEqual(pop_change, 0)  # Population should change

    def test_evolve(self):
        """Test evolution mechanics."""
        # Set up for evolution
        self.entity.evolution_points = self.entity.evolution_threshold + 10
        initial_stage = self.entity.evolution_stage

        # Trigger evolution
        self.entity.evolve()

        # Check results
        self.assertEqual(self.entity.evolution_stage, initial_stage + 1)
        self.assertLess(self.entity.evolution_points, self.entity.evolution_threshold)

    def test_evolve_genome(self):
        """Test genome evolution with traits."""
        # Save initial genome
        initial_genome = self.entity.genome.copy()

        # Evolve genome
        self.entity._evolve_genome()

        # Genome should change
        for key in self.entity.genome:
            if key in initial_genome:  # Some keys might be new
                self.assertNotEqual(initial_genome[key], self.entity.genome[key])


class TestShop(unittest.TestCase):
    """Test suite for the Shop class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.shop = Shop()
        self.player = Player()
        self.field = AsteroidField()
        self.notifier = NotificationManager()

    def test_initialization(self):
        """Test if shop initializes correctly."""
        self.assertTrue(len(self.shop.options) > 0)
        self.assertEqual(self.shop.current_category, "ship")
        self.assertEqual(self.shop.scroll_offset, 0)

    def test_get_filtered_options(self):
        """Test filtering options by category."""
        all_options = len(self.shop.options)

        # Filter by ship category
        self.shop.current_category = "ship"
        ship_options = self.shop.get_filtered_options()
        self.assertLess(len(ship_options), all_options)

        # Filter by field category
        self.shop.current_category = "field"
        field_options = self.shop.get_filtered_options()
        self.assertLess(len(field_options), all_options)

    def test_purchase_upgrade(self):
        """Test purchasing upgrades."""
        # Find a ship upgrade
        self.shop.current_category = "ship"
        options = self.shop.get_filtered_options()
        ship_upgrade = options[0]

        # Record initial stats
        initial_currency = self.player.currency

        # Try to purchase
        self.player.currency = ship_upgrade["cost"] + 100  # Ensure enough currency
        result = self.shop.purchase_upgrade(
            ship_upgrade, self.player, self.field, self.notifier
        )

        # Check results
        self.assertTrue(result)
        self.assertEqual(self.player.currency, initial_currency + 100)

    def test_toggle_expanded(self):
        """Test toggling between expanded and collapsed views."""
        initial_state = self.shop.expanded

        # Toggle state
        self.shop.toggle_expanded()

        # Check new state
        self.assertNotEqual(initial_state, self.shop.expanded)


class TestNotificationManager(unittest.TestCase):
    """Test suite for the NotificationManager class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.notifier = NotificationManager()

    def test_initialization(self):
        """Test if notifier initializes correctly."""
        self.assertEqual(len(self.notifier.notifications), 0)
        self.assertEqual(self.notifier.max_stored_notifications, 100)
        self.assertEqual(self.notifier.max_visible_notifications, 10)

    def test_add_notification(self):
        """Test adding notifications."""
        initial_count = len(self.notifier.notifications)

        # Add a notification
        self.notifier.add("Test notification", category="system")

        # Check results
        self.assertEqual(len(self.notifier.notifications), initial_count + 1)
        self.assertEqual(self.notifier.notifications[0]["text"], "Test notification")
        self.assertEqual(self.notifier.notifications[0]["category"], "system")

    def test_update(self):
        """Test updating notifications (expiry)."""
        # Add a notification with short duration
        self.notifier.add("Short notification", duration=1)
        initial_count = len(self.notifier.notifications)

        # Update should remove expired notifications
        self.notifier.update()
        self.assertLess(len(self.notifier.notifications), initial_count)

    def test_category_filtering(self):
        """Test notification category filtering."""
        # Add notifications in different categories
        self.notifier.add("System notification", category="system")
        self.notifier.add("Mining notification", category="mining")
        self.notifier.add("Race notification", category="race")

        # Filter by category
        self.notifier.active_filter = "system"
        filtered = self.notifier.get_filtered_notifications()
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["category"], "system")

        # Filter all
        self.notifier.active_filter = "all"
        all_notifications = self.notifier.get_filtered_notifications()
        self.assertEqual(len(all_notifications), 3)


if __name__ == "__main__":
    unittest.main()
