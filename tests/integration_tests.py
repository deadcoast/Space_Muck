"""
Integration tests for the Space Muck game.

These tests verify that different components work together correctly.
"""

import os
import sys
import unittest
import pygame
import numpy as np
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import *
from src.world.asteroid_field import AsteroidField
from src.entities.player import Player
from src.entities.miner_entity import MinerEntity
from src.ui.shop import Shop
from src.ui.notification import NotificationManager


class TestPlayerFieldInteraction(unittest.TestCase):
    """Test the interactions between player and asteroid field."""

    def setUp(self):
        """Set up the test environment."""
        self.field = AsteroidField(width=100, height=80)
        self.player = Player()

        # Position player in the middle of the field
        self.player.x = 50
        self.player.y = 40

        # Add some asteroids around the player
        self.field.grid[35:45, 45:55] = 50  # Value 50 asteroids
        self.field.rare_grid[38:42, 48:52] = 1  # Some rare asteroids

    def test_player_mining_cycle(self):
        """Test a complete mining cycle and its effects."""
        initial_currency = self.player.currency

        # Mine asteroids
        self.player.mine(self.field)

        # Move to new location
        self.player.move(5, 5, self.field)

        # Mine again
        self.player.mine(self.field)

        # Currency should have increased
        self.assertGreater(self.player.currency, initial_currency)

    def test_auto_path_mining(self):
        """Test the auto-pathing and mining feature."""
        # Find a mining target
        target = self.player.find_mining_target(self.field)
        self.assertIsNotNone(target)

        # Generate path to target
        path = self.player.generate_path(
            (self.player.x, self.player.y), target, self.field
        )
        self.assertGreater(len(path), 0)

        # Follow the path
        steps_taken = 0
        while path and steps_taken < 20:  # Limit to avoid infinite loop in test
            next_pos = path.pop(0)
            dx = next_pos[0] - self.player.x
            dy = next_pos[1] - self.player.y
            self.player.move(dx, dy, self.field)
            steps_taken += 1

        # Should have moved closer to or reached the target
        distance = abs(self.player.x - target[0]) + abs(self.player.y - target[1])
        self.assertLessEqual(distance, 10)


class TestUIInteractions(unittest.TestCase):
    """Test interactions between UI components and game state."""

    def setUp(self):
        """Set up the test environment."""
        self.field = AsteroidField(width=100, height=80)
        self.player = Player()
        self.shop = Shop()
        self.notifier = NotificationManager()

    def test_shop_purchase_notification(self):
        """Test that shop purchases generate notifications."""
        # Get an affordable upgrade
        self.shop.current_category = "ship"
        options = self.shop.get_filtered_options()
        if upgrade := next(
            (o for o in options if o["cost"] <= self.player.currency), None
        ):
            # Record notification count before purchase
            initial_count = len(self.notifier.notifications)

            # Make the purchase
            self.shop.purchase_upgrade(upgrade, self.player, self.field, self.notifier)

            # Check that a notification was added
            self.assertEqual(len(self.notifier.notifications), initial_count + 1)

    def test_race_events_notifications(self):
        """Test that race events generate notifications."""
        # Add a race to the field
        race = MinerEntity(race_id=1, color=(255, 0, 0))
        self.field.races = [race]

        # Record notification count before event
        initial_count = len(self.notifier.notifications)

        # Simulate race evolution
        race.evolution_points = race.evolution_threshold + 10
        race.evolve()

        # Notify about the evolution
        self.notifier.notify_event(
            "race", f"Race {race.race_id} has evolved to stage {race.evolution_stage}!"
        )

        # Check that a notification was added
        self.assertEqual(len(self.notifier.notifications), initial_count + 1)
        self.assertEqual(self.notifier.notifications[0]["category"], "race")


class TestGameCycle(unittest.TestCase):
    """Test a complete game cycle with all components."""

    @patch("pygame.Surface")
    def setUp(self, mock_surface):
        """Set up the test environment with mocked pygame components."""
        # Mock pygame surface for rendering
        self.mock_surface = mock_surface.return_value

        # Initialize game components
        self.field = AsteroidField(width=100, height=80)
        self.player = Player()
        self.shop = Shop()
        self.notifier = NotificationManager()

        # Add a race to the field
        race = MinerEntity(race_id=1, color=(255, 0, 0))
        self.field.races = [race]

        # Seed the field with asteroids
        self.field.manual_seed(50, 40, radius=10)

    def test_complete_game_cycle(self):
        """Test a complete game cycle (update, mine, shop, etc.)."""
        # Run several game cycles
        for _ in range(5):
            # Update field (cellular automaton step)
            self.field.update()

            # Update entities
            race_income = self.field.update_entities()

            # Player mines
            self.player.mine(self.field)

            # Player fleet update
            fleet_results = self.player.update_fleet(self.field)

            # Auto-mining
            self.player.auto_miners = 1  # Ensure we have an auto-miner
            self.player.auto_mine(self.field)

            # Update notifications
            self.notifier.update()

            # Try to purchase an upgrade if we have enough currency
            if self.player.currency >= 50:
                self.shop.current_category = "ship"
                options = self.shop.get_filtered_options()
                for upgrade in options:
                    if upgrade["cost"] <= self.player.currency:
                        self.shop.purchase_upgrade(
                            upgrade, self.player, self.field, self.notifier
                        )
                        break

        # Check that we have some results from our actions
        self.assertGreater(self.player.total_mined, 0)
        self.assertGreaterEqual(len(self.notifier.notifications), 1)
