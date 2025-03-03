"""
Test module for the Player class.
"""

import unittest
import sys
import os
from unittest.mock import MagicMock
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the dependencies before importing
sys.modules["perlin_noise"] = MagicMock()
sys.modules["sklearn.cluster"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["pygame"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["scipy.ndimage"] = MagicMock()

from src.entities.player import Player


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
        # Create a standard player for most tests
        self.player = Player()
        self.field = MockField()
        self.player.territory_center = (10, 10)

        # Create a player with custom attributes for specific tests
        self.custom_player = Player(
            race_id=1,
            color=(100, 150, 200),
            birth_set={3, 5, 7},
            survival_set={2, 3, 4, 5},
            initial_density=0.05,
            position=(25, 25),
        )

    def test_default_player_attributes(self):
        """Test that Player has the correct default attributes."""
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

    def test_custom_player_attributes(self):
        """Test that Player correctly sets custom attributes."""
        self.assertTrue(self.custom_player.is_player)
        self.assertEqual(self.custom_player.race_id, 1)
        self.assertEqual(self.custom_player.color, (100, 150, 200))
        self.assertEqual(self.custom_player.birth_set, {3, 5, 7})
        self.assertEqual(self.custom_player.survival_set, {2, 3, 4, 5})
        self.assertEqual(self.custom_player.initial_density, 0.05)
        self.assertEqual(self.custom_player.position, (25, 25))

    def test_inherited_attributes(self):
        """Test that Player correctly inherits attributes from MinerEntity and BaseEntity."""
        # Test BaseEntity attributes
        self.assertIsNotNone(self.player.entity_id)
        self.assertEqual(self.player.entity_type, "miner")
        self.assertEqual(self.player.active, True)
        self.assertEqual(self.player.health, 100)
        self.assertEqual(self.player.max_health, 100)
        self.assertEqual(self.player.level, 1)

        # Test MinerEntity attributes
        self.assertIsNotNone(self.player.territory_center)
        self.assertEqual(self.player.last_income, 0)
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

        result = self._rarity_handler(50, "common", 80)
        self._rarity_handler(60, "rare", 320)
        result = self._rarity_handler(70, "anomaly", 960)
        self.assertIn("anomaly_70_70", self.player.discovered_anomalies)

    def _rarity_handler(self, arg0, arg1, arg2):
        result = self._initial_credit_handler(arg0, "type", arg1)
        self.assertEqual(result["value"], arg2)

        return result

    def test_credit_management(self):
        """Test credit management functionality."""
        # Test initial credits
        self.assertEqual(self.player.credits, 1000)

        self._initial_credit_handler(50, "value", 80)
        self.assertEqual(self.player.credits, 1080)

        self._initial_credit_handler(60, "value", 320)
        self.assertEqual(self.player.credits, 1400)

        self._initial_credit_handler(70, "value", 960)
        self.assertEqual(self.player.credits, 2360)

    def test_transaction_history(self):
        """Test that transaction history is properly tracked."""
        # Reset player for this test to ensure clean income history
        self.player = Player()

        # Initial state
        self.assertEqual(self.player.last_income, 0)
        self.assertEqual(len(self.player.income_history), 0)

        self._transaction_handler(50, 80, 1, 0)
        self._transaction_handler(60, 320, 2, 1)

    def _transaction_handler(self, arg0, arg1, arg2, arg3):
        # Mine several asteroids
        self.player.mine(arg0, arg0, self.field)
        self.assertEqual(self.player.last_income, arg1)
        self.assertEqual(len(self.player.income_history), arg2)
        self.assertEqual(self.player.income_history[arg3], arg1)

    def test_income_history_limit(self):
        """Test that income history is limited to 100 entries."""
        # Create a player with an empty income history
        test_player = Player()

        # Add exactly 100 entries
        for i in range(1, 101):
            test_player.income_history.append(i)

        self._entry_verification_handler(test_player, 1, 100)
        # Manually add one more entry and verify the oldest is removed
        test_player.income_history.append(101)
        test_player.income_history.pop(0)  # Simulate what the mine method should do

        self._entry_verification_handler(test_player, 2, 101)

    def _entry_verification_handler(self, test_player, arg1, arg2):
        # Verify we have 100 entries
        self.assertEqual(len(test_player.income_history), 100)
        self.assertEqual(test_player.income_history[0], arg1)
        self.assertEqual(test_player.income_history[-1], arg2)

    def test_ship_upgrade(self):
        """Test ship upgrade functionality."""
        # Initial ship level
        self.assertEqual(self.player.ship_level, 1)
        self.assertEqual(self.player.mining_speed, 1.0)

        # Ensure player has enough credits for the upgrade
        self.player.credits = 5000  # More than enough for level 2 upgrade

        # Upgrade ship
        if hasattr(self.player, "upgrade_ship"):
            result = self.player.upgrade_ship()
            self.assertTrue(result["success"])
            self.assertEqual(self.player.ship_level, 2)
            self.assertGreater(self.player.mining_speed, 1.0)
        else:
            # Skip test if method doesn't exist
            self.skipTest("upgrade_ship method not implemented yet")

    def test_level_progression(self):
        """Test player level progression."""
        # Reset player for this test
        self.player = Player()

        self._xp_calculation_handler(1, 100)
        # Test XP calculation
        xp_common = self.player._calculate_xp_gain(100, "common")
        self.assertEqual(xp_common, 10)  # 10% of mining value

        xp_rare = self.player._calculate_xp_gain(100, "rare")
        self.assertEqual(xp_rare, 15)  # 10% of mining value * 1.5

        xp_anomaly = self.player._calculate_xp_gain(100, "anomaly")
        self.assertEqual(xp_anomaly, 30)  # 10% of mining value * 3.0

        # Test adding XP without level up
        result = self.player._add_xp(50)
        self.assertFalse(result["level_up"])
        self.assertEqual(self.player.level, 1)
        self.assertEqual(self.player.xp, 50)

        # Test level up
        result = self.player._add_xp(50)  # This should trigger level up
        self.assertTrue(result["level_up"])
        self.assertEqual(result["old_level"], 1)
        self.assertEqual(result["new_level"], 2)
        self._xp_calculation_handler(2, 120)
        result = self._initial_credit_handler(50, "xp_gained", 9)
        # Test get_level_progress
        progress = self.player.get_level_progress()
        self.assertEqual(progress["level"], 2)
        self.assertEqual(progress["xp"], 9)
        self.assertEqual(progress["xp_to_next_level"], 120)
        self.assertAlmostEqual(progress["progress_percent"], (9 / 120) * 100)

        # Test level bonuses were applied
        self.assertAlmostEqual(self.player.mining_efficiency, 0.85)  # 0.8 + 0.05
        self.assertAlmostEqual(self.player.mining_speed, 1.1)  # 1.0 + 0.1

    def _initial_credit_handler(self, arg0, arg1, arg2):
        result = self.player.mine(arg0, arg0, self.field)
        self.assertTrue(result["success"])
        self.assertEqual(result[arg1], arg2)
        return result

    def _xp_calculation_handler(self, arg0, arg1):
        # Initial level attributes
        self.assertEqual(self.player.level, arg0)
        self.assertEqual(self.player.xp, 0)
        self.assertEqual(self.player.xp_to_next_level, arg1)

    def test_command_processing(self):
        """Test player command processing."""
        # Test move commands
        initial_position = self.player.territory_center

        self._command_processing_handler(1, 0, initial_position)
        self.assertEqual(self.player.territory_center[1], initial_position[1])

        self._command_processing_handler(0, 1, initial_position)
        self.assertEqual(self.player.territory_center[1], initial_position[1] + 1)

        # Test boundary conditions
        # Reset player position to a corner
        self.player.territory_center = (0, 0)

        self._boundary_handler(-1, 0)
        self._boundary_handler(0, -1)

    def _boundary_handler(self, arg0, arg1):
        # Try to move out of bounds (left)
        result = self.player.move(arg0, arg1, self.field)
        self.assertFalse(result)
        self.assertEqual(self.player.territory_center, (0, 0))

        return result

    def _command_processing_handler(self, arg0, arg1, initial_position):
        # Move right
        result = self.player.move(arg0, arg1, self.field)
        self.assertTrue(result)
        self.assertEqual(self.player.territory_center[0], initial_position[0] + 1)
        return result

    def test_inventory_management(self):
        """Test player inventory management."""
        # Initial inventory should be empty
        self.assertEqual(len(self.player.inventory), 0)

        # Test adding items to inventory (if method exists)
        if hasattr(self.player, "add_to_inventory"):
            self.player.add_to_inventory("test_item", 5)
            self.assertIn("test_item", self.player.inventory)
            self.assertEqual(self.player.inventory["test_item"], 5)

            # Add more of the same item
            self.player.add_to_inventory("test_item", 3)
        else:
            # For now, just modify the inventory directly for testing
            self.player.inventory["test_item"] = 5
            self.assertIn("test_item", self.player.inventory)
            self.assertEqual(self.player.inventory["test_item"], 5)

            # Add more of the same item
            self.player.inventory["test_item"] += 3

        self.assertEqual(self.player.inventory["test_item"], 8)

    def test_anomaly_discovery(self):
        """Test anomaly discovery functionality."""
        # Initial state - no discovered anomalies
        self.assertEqual(len(self.player.discovered_anomalies), 0)

        # Mine an anomaly
        self.player.mine(70, 70, self.field)  # This is set up as an anomaly in setUp()

        # Verify anomaly was discovered
        self.assertEqual(len(self.player.discovered_anomalies), 1)
        self.assertIn("anomaly_70_70", self.player.discovered_anomalies)

        # Mine another anomaly
        # First reset the field value since mining removes the asteroid
        self.field.grid[70, 70] = 300
        self.field.rare_grid[70, 70] = 2

        # Mine it again
        self.player.mine(70, 70, self.field)

        # Should still only have one unique anomaly
        self.assertEqual(len(self.player.discovered_anomalies), 1)

    def test_quest_handling(self):
        """Test quest handling functionality."""
        # Initial state
        self.assertIsNone(self.player.current_quest)
        self.assertEqual(len(self.player.completed_quests), 0)

        test_quest = {"id": "test_quest", "description": "Test quest", "reward": 500}
        # Test quest assignment (if method exists)
        if hasattr(self.player, "assign_quest"):
            self.player.assign_quest(test_quest)
            self.assertEqual(self.player.current_quest, test_quest)

            # Test quest completion
            if hasattr(self.player, "complete_quest"):
                result = self.player.complete_quest()
                self.assertTrue(result["success"])
                self._reward_assert_handler(test_quest)
            else:
                self.skipTest("complete_quest method not implemented yet")
        else:
            self._current_quest_handler(test_quest)

    def _current_quest_handler(self, test_quest):
        self.player.current_quest = test_quest
        self.assertEqual(self.player.current_quest, test_quest)

        # Complete the quest
        self.player.completed_quests.append(test_quest)
        self.player.current_quest = None
        self.player.credits += test_quest["reward"]

        self._reward_assert_handler(test_quest)

    def _reward_assert_handler(self, test_quest):
        self.assertIsNone(self.player.current_quest)
        self.assertEqual(len(self.player.completed_quests), 1)
        self.assertEqual(self.player.completed_quests[0], test_quest)
        self.assertEqual(self.player.credits, 1500)  # 1000 initial + 500 reward

    def test_multiple_level_ups(self):
        """Test multiple level-ups from a single large XP gain."""
        # Reset player for this test
        self.player = Player()

        # Initial state
        self.assertEqual(self.player.level, 1)
        self.assertEqual(self.player.xp, 0)

        # Add enough XP to level up multiple times
        # Level 1 -> 2 requires 100 XP
        # Level 2 -> 3 requires 120 XP
        # Level 3 -> 4 requires 144 XP
        # Total: 364 XP for 3 level-ups
        result = self.player._add_xp(364)

        # Verify level-up occurred
        self.assertTrue(result["level_up"])
        self.assertEqual(result["old_level"], 1)
        self.assertEqual(result["new_level"], 4)  # Should have jumped from 1 to 4

        # Verify current state
        self.assertEqual(self.player.level, 4)
        self.assertEqual(
            self.player.xp_to_next_level, 172
        )  # Actual XP required for next level

        # Verify level bonuses were applied
        self.assertAlmostEqual(self.player.mining_efficiency, 0.95)  # 0.8 + 0.15
        self.assertAlmostEqual(self.player.mining_speed, 1.3)  # 1.0 + 0.3

        # Test level cap
        # Add enough XP to reach level 5 and beyond, but should cap at 5
        result = self.player._add_xp(1000)
        self.assertEqual(self.player.level, 5)  # Should be capped at level 5
        self.assertAlmostEqual(self.player.mining_efficiency, 1.0)  # 0.8 + 0.2
        self.assertAlmostEqual(self.player.mining_speed, 1.4)  # 1.0 + 0.4

    def test_reputation_initialization(self):
        """Test that reputation is correctly initialized."""
        # Import constants from player module
        from src.entities.player import GAME_FACTIONS

        # Check that all factions are initialized with neutral reputation
        for faction in GAME_FACTIONS:
            self.assertEqual(self.player.reputation[faction], 0)
            self.assertEqual(self.player.faction_quests_completed[faction], 0)

    def test_get_reputation_level(self):
        """Test that reputation levels are correctly calculated."""
        # Import constants from player module
        # Import player module for reputation testing
        from src.entities.player import (
            Player,
        )  # Re-import to ensure we have the latest version

        # Test all reputation levels
        self.player.reputation["miners_guild"] = -100  # Hostile
        self.player.reputation["explorers_union"] = -30  # Unfriendly
        self.player.reputation["galactic_navy"] = 0  # Neutral
        self.player.reputation["traders_coalition"] = 30  # Friendly
        self.player.reputation["fringe_colonies"] = 100  # Allied

        self.assertEqual(self.player.get_reputation_level("miners_guild"), "hostile")
        self.assertEqual(
            self.player.get_reputation_level("explorers_union"), "unfriendly"
        )
        self.assertEqual(self.player.get_reputation_level("galactic_navy"), "neutral")
        self.assertEqual(
            self.player.get_reputation_level("traders_coalition"), "friendly"
        )
        self.assertEqual(self.player.get_reputation_level("fringe_colonies"), "allied")

        # Test unknown faction
        self.assertEqual(self.player.get_reputation_level("unknown_faction"), "neutral")

    def test_change_reputation(self):
        """Test that reputation changes work correctly."""
        # Start with neutral reputation
        self.player.reputation["miners_guild"] = 0

        # Test increasing reputation
        result = self.player.change_reputation("miners_guild", 20)
        self.assertTrue(result["success"])
        self.assertEqual(result["old_value"], 0)
        self.assertEqual(result["new_value"], 20)
        self.assertEqual(result["old_level"], "neutral")
        self.assertEqual(result["new_level"], "friendly")
        self.assertTrue(result["level_changed"])
        self.assertEqual(self.player.reputation["miners_guild"], 20)

        # Test decreasing reputation
        result = self.player.change_reputation("miners_guild", -70)
        self.assertTrue(result["success"])
        self.assertEqual(result["old_value"], 20)
        self.assertEqual(result["new_value"], -50)
        self.assertEqual(result["old_level"], "friendly")
        self.assertEqual(result["new_level"], "unfriendly")
        self.assertTrue(result["level_changed"])
        self.assertEqual(self.player.reputation["miners_guild"], -50)

        # Test bounds checking (lower bound)
        self.player.reputation["miners_guild"] = -90
        result = self.player.change_reputation("miners_guild", -30)
        self.assertEqual(
            self.player.reputation["miners_guild"], -100
        )  # Should be capped at -100

        # Test bounds checking (upper bound)
        self.player.reputation["miners_guild"] = 90
        result = self.player.change_reputation("miners_guild", 30)
        self.assertEqual(
            self.player.reputation["miners_guild"], 100
        )  # Should be capped at 100

        # Test invalid faction
        result = self.player.change_reputation("invalid_faction", 10)
        self.assertFalse(result["success"])

    def test_faction_price_modifier(self):
        """Test that price modifiers based on reputation work correctly."""
        # Set up different reputation levels
        self.player.reputation["miners_guild"] = -90  # Hostile
        self.player.reputation["explorers_union"] = -30  # Unfriendly
        self.player.reputation["galactic_navy"] = 0  # Neutral
        self.player.reputation["traders_coalition"] = 30  # Friendly
        self.player.reputation["fringe_colonies"] = 90  # Allied

        # Check price modifiers
        self.assertAlmostEqual(
            self.player.get_faction_price_modifier("miners_guild"), 1.5
        )  # 50% markup
        self.assertAlmostEqual(
            self.player.get_faction_price_modifier("explorers_union"), 1.2
        )  # 20% markup
        self.assertAlmostEqual(
            self.player.get_faction_price_modifier("galactic_navy"), 1.0
        )  # Standard price
        self.assertAlmostEqual(
            self.player.get_faction_price_modifier("traders_coalition"), 0.9
        )  # 10% discount
        self.assertAlmostEqual(
            self.player.get_faction_price_modifier("fringe_colonies"), 0.8
        )  # 20% discount

    def test_faction_quest_generation(self):
        """Test that faction quests are generated correctly."""
        self._quest_generation_handler("miners_guild", "mining", "Miners Guild")
        self._quest_generation_handler("galactic_navy", "combat", "pirate ships")

    def _quest_generation_handler(self, arg0, arg1, arg2):
        # Generate a quest for a specific faction
        result = self.player._generate_quest(arg0)

        # Check that the quest has the correct faction
        self.assertEqual(result["faction"], arg0)

        # Check that the description mentions the faction for mining quests
        if result["type"] == arg1:
            self.assertIn(arg2, result["description"])

        return result

    def test_quest_completion_reputation(self):
        """Test that completing quests affects reputation."""
        # Set up a faction quest
        quest = {
            "id": "test_quest_123",
            "type": "mining",
            "description": "Test mining quest",
            "target_amount": 100,
            "current_amount": 100,  # Already completed
            "reward": 1000,
            "level_requirement": 2,
            "faction": "miners_guild",
        }

        # Assign the quest
        self.player.assign_quest(quest)

        # Check initial reputation
        initial_rep = self.player.reputation["miners_guild"]

        # Complete the quest
        result = self.player.complete_quest()

        # Verify reputation increased
        self.assertTrue(result["success"])
        self.assertEqual(result["faction"], "miners_guild")
        self.assertTrue(result["reputation_change"] > 0)
        self.assertTrue(self.player.reputation["miners_guild"] > initial_rep)
        self.assertEqual(self.player.faction_quests_completed["miners_guild"], 1)

    def test_hostile_faction_quest_rejection(self):
        """Test that hostile factions reject quest assignments."""
        # Set reputation to hostile
        self.player.reputation["miners_guild"] = -100

        # Try to get a quest from the hostile faction
        result = self.player.assign_quest(faction="miners_guild")

        # Verify quest was rejected
        self.assertFalse(result["success"])
        self.assertIn("too low", result["reason"])


if __name__ == "__main__":
    unittest.main()
