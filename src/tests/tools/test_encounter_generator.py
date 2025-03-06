"""
Tests for the encounter generator module.
"""

import unittest
from unittest.mock import patch
import random

from src.entities.player import Player
from src.entities.enemy_ship import EnemyShip
from src.systems.combat_system import CombatSystem
from src.systems.encounter_generator import EncounterGenerator


class TestEncounterGenerator(unittest.TestCase):
    """Test cases for the encounter generator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a real player instance
        self.player = Player(position=(50, 50))
        self.player.level = 3
        self.player.in_combat = False

        # Ensure player has appropriate reputation values
        for faction in self.player.reputation:
            self.player.reputation[faction] = 0

        self.player.current_quest = None

        # Create a real combat system
        self.combat_system = CombatSystem(self.player)

        # Create the encounter generator
        self.encounter_generator = EncounterGenerator(self.player, self.combat_system)

        # Set a seed for reproducible random tests
        random.seed(42)

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

    def test_check_for_encounter_success(self):
        """Test successful encounter generation."""
        # Temporarily override the encounter_chance_base to ensure encounters occur
        original_chance = self.encounter_generator.encounter_chance_base
        self.encounter_generator.encounter_chance_base = 1.0  # 100% chance of encounter

        # Save the original _generate_encounter method
        original_generate_encounter = self.encounter_generator._generate_encounter

        # Define a test encounter result
        test_encounter = {"encounter": True, "type": "test_combat"}

        # Create a wrapper to track if the method was called
        def tracking_generate_encounter(*args, **kwargs):
            tracking_generate_encounter.called = True
            return test_encounter

        tracking_generate_encounter.called = False
        self.encounter_generator._generate_encounter = tracking_generate_encounter

        try:
            # Check for encounter
            result = self.encounter_generator.check_for_encounter()

            # Verify encounter was generated
            self.assertTrue(result["encounter"])
            self.assertEqual(result["type"], "test_combat")
            self.assertTrue(tracking_generate_encounter.called)
        finally:
            # Restore original methods and values
            self.encounter_generator._generate_encounter = original_generate_encounter
            self.encounter_generator.encounter_chance_base = original_chance

    def test_check_for_encounter_failure(self):
        """Test failed encounter check."""
        # Temporarily override the encounter_chance_base to ensure no encounters occur
        original_chance = self.encounter_generator.encounter_chance_base
        self.encounter_generator.encounter_chance_base = 0.0  # 0% chance of encounter

        try:
            # Check for encounter
            result = self.encounter_generator.check_for_encounter()

            # Verify no encounter due to random chance
            self.assertFalse(result["encounter"])
            self.assertEqual(result["reason"], "Random chance")
        finally:
            # Restore original value
            self.encounter_generator.encounter_chance_base = original_chance

    def test_generate_combat_encounter(self):
        """Test combat encounter generation."""
        # Save original player state to restore later
        original_in_combat = self.player.in_combat
        self.player.in_combat = False

        try:
            # Generate combat encounter
            result = self.encounter_generator._generate_combat_encounter()

            # Verify combat encounter
            self.assertTrue(result["encounter"])
            self.assertEqual(result["type"], "combat")
            self.assertIn("message", result)
            self.assertIn("enemy", result)
            self.assertTrue(result["combat_started"])

            # Verify combat was started
            self.assertTrue(self.player.in_combat)
            self.assertIsNotNone(self.player.current_enemy)
        finally:
            # Restore original player state
            self.player.in_combat = original_in_combat
            self.combat_system.end_combat()

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

        # Save original player state to restore later
        original_in_combat = self.player.in_combat
        self.player.in_combat = False

        try:
            # Generate quest encounter
            result = self.encounter_generator.generate_quest_encounter("combat")

            # Verify quest encounter
            self.assertTrue(result["encounter"])
            self.assertEqual(result["type"], "quest_combat")
            self.assertIn("message", result)
            self.assertIn("enemy", result)
            self.assertTrue(result["combat_started"])
            self.assertTrue(result["quest_related"])

            # Verify combat was started with the right type of enemy
            self.assertTrue(self.player.in_combat)
            self.assertIsNotNone(self.player.current_enemy)
            self.assertEqual(self.player.current_enemy.faction, "fringe_colonies")
            self.assertEqual(self.player.current_enemy.difficulty, "hard")
        finally:
            # Restore original player state
            self.player.in_combat = original_in_combat
            self.combat_system.end_combat()

    def test_generate_quest_encounter_unknown_type(self):
        """Test encounter generation with unknown quest type."""
        # Generate encounter with unknown quest type
        result = self.encounter_generator.generate_quest_encounter("unknown_type")

        # Verify no encounter
        self.assertFalse(result["encounter"])
        self.assertEqual(result["reason"], "Unknown quest type: unknown_type")


if __name__ == "__main__":
    unittest.main()
