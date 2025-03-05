"""
Tests for the encounter generator module.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.entities.player import Player
from src.entities.enemy_ship import EnemyShip
from src.systems.combat_system import CombatSystem
from src.systems.encounter_generator import EncounterGenerator


class TestEncounterGenerator(unittest.TestCase):
    """Test cases for the encounter generator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock player
        self.player = MagicMock(spec=Player)
        self.player.level = 3
        self.player.in_combat = False
        self.player.position = (50, 50)
        self.player.reputation = {"galactic_navy": 0, "fringe_colonies": 0}
        self.player.current_quest = None

        # Create a mock combat system
        self.combat_system = MagicMock(spec=CombatSystem)

        # Mock generate_enemy method
        mock_enemy = MagicMock(spec=EnemyShip)
        mock_enemy.difficulty = "medium"
        mock_enemy.ship_type = "pirate"
        mock_enemy.level = 3
        mock_enemy.faction = None
        mock_enemy.aggression = 0.5
        mock_enemy.position = (50, 50)
        mock_enemy.get_stats.return_value = {"ship_type": "pirate", "level": 3}

        self.combat_system.generate_enemy.return_value = mock_enemy
        self.combat_system.start_combat.return_value = {"success": True}

        # Create the encounter generator
        self.encounter_generator = EncounterGenerator(self.player, self.combat_system)

    def test_zone_danger_levels(self):
        """Test zone danger level initialization."""
        # Verify zones were created
        self.assertGreater(len(self.encounter_generator.zone_danger_levels), 0)

        # Test getting danger level for a position
        danger = self.encounter_generator._get_zone_danger((200, 150))
        self.assertGreater(danger, 0)

        # Test danger level for corner (should be higher)
        corner_danger = self.encounter_generator._get_zone_danger((10, 10))
        center_danger = self.encounter_generator._get_zone_danger((200, 150))
        print(f"Corner danger: {corner_danger}, Center danger: {center_danger}")
        self.assertGreater(corner_danger, center_danger)

    def test_check_for_encounter_cooldown(self):
        """Test encounter cooldown."""
        # Set cooldown
        self.encounter_generator.encounter_cooldown = 3

        # Check for encounter
        result = self.encounter_generator.check_for_encounter()

        # Verify no encounter due to cooldown
        self.assertFalse(result["encounter"])
        self.assertEqual(result["reason"], "On cooldown")
        self.assertEqual(self.encounter_generator.encounter_cooldown, 2)

    def test_check_for_encounter_in_combat(self):
        """Test no encounters while in combat."""
        # Set player in combat
        self.player.in_combat = True

        # Check for encounter
        result = self.encounter_generator.check_for_encounter()

        # Verify no encounter due to being in combat
        self.assertFalse(result["encounter"])
        self.assertEqual(result["reason"], "Already in combat")

    @patch("random.random")
    def test_check_for_encounter_success(self, mock_random):
        """Test successful encounter generation."""
        # Force encounter to happen
        mock_random.return_value = 0.01  # Very low value to ensure encounter happens

        # Mock the _generate_encounter method
        self.encounter_generator._generate_encounter = MagicMock(
            return_value={"encounter": True, "type": "combat"}
        )

        # Check for encounter
        result = self.encounter_generator.check_for_encounter()

        # Verify encounter was generated
        self.assertTrue(result["encounter"])
        self.encounter_generator._generate_encounter.assert_called_once()

    @patch("random.random")
    def test_check_for_encounter_failure(self, mock_random):
        """Test failed encounter check."""
        # Force no encounter
        mock_random.return_value = 0.99  # Very high value to ensure no encounter

        # Check for encounter
        result = self.encounter_generator.check_for_encounter()

        # Verify no encounter due to random chance
        self.assertFalse(result["encounter"])
        self.assertEqual(result["reason"], "Random chance")

    @patch("random.random")
    @patch("random.choice")
    def test_generate_combat_encounter(self, mock_choice, mock_random):
        """Test combat encounter generation."""
        # Mock random choices
        mock_random.return_value = 0.2  # Below 0.3 for faction encounter
        mock_choice.return_value = "galactic_navy"

        # Generate combat encounter
        result = self.encounter_generator._generate_combat_encounter()

        # Verify combat encounter
        self.assertTrue(result["encounter"])
        self.assertEqual(result["type"], "combat")
        self.assertIn("message", result)
        self.assertIn("enemy", result)
        self.assertTrue(result["combat_started"])

        # Verify combat system methods were called
        self.combat_system.generate_enemy.assert_called_once()
        self.combat_system.start_combat.assert_called_once()

    def test_generate_quest_encounter_no_quest(self):
        """Test quest encounter generation with no active quest."""
        # Generate quest encounter with no active quest
        result = self.encounter_generator.generate_quest_encounter("combat")

        # Verify no encounter
        self.assertFalse(result["encounter"])
        self.assertEqual(result["reason"], "No active combat quest")

    def test_generate_quest_encounter_combat(self):
        """Test combat quest encounter generation."""
        # Set up active combat quest
        self.player.current_quest = {
            "type": "combat",
            "difficulty": "hard",
            "target_faction": "fringe_colonies",
            "target_type": "pirate",
            "target_enemies": 5,
            "current_enemies": 2,
        }

        # Generate quest encounter
        result = self.encounter_generator.generate_quest_encounter("combat")

        # Verify quest encounter
        self.assertTrue(result["encounter"])
        self.assertEqual(result["type"], "quest_combat")
        self.assertIn("message", result)
        self.assertIn("enemy", result)
        self.assertTrue(result["combat_started"])
        self.assertTrue(result["quest_related"])

        # Verify combat system methods were called with correct parameters
        self.combat_system.generate_enemy.assert_called_once_with(
            difficulty="hard", faction="fringe_colonies", position=self.player.position
        )
        self.combat_system.start_combat.assert_called_once()

    def test_generate_quest_encounter_unknown_type(self):
        """Test encounter generation with unknown quest type."""
        # Generate encounter with unknown quest type
        result = self.encounter_generator.generate_quest_encounter("unknown_type")

        # Verify no encounter
        self.assertFalse(result["encounter"])
        self.assertEqual(result["reason"], "Unknown quest type: unknown_type")


if __name__ == "__main__":
    unittest.main()
