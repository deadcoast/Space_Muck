"""
Tests for the combat system module.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.entities.player import Player
from src.entities.enemy_ship import EnemyShip
from src.systems.combat_system import CombatSystem


class TestCombatSystem(unittest.TestCase):
    """Test cases for the combat system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock player
        self.player = MagicMock(spec=Player)
        self.player.level = 3
        self.player.weapon_level = 2
        self.player.shield_level = 2
        self.player.hull_level = 2
        self.player.attack_power = 10
        self.player.shield_strength = 50
        self.player.hull_strength = 100
        self.player.max_shield = 50
        self.player.max_hull = 100
        self.player.evasion = 0.2
        self.player.armor = 0.1
        self.player.critical_chance = 0.1
        self.player.attack_speed = 1.0
        self.player.shield_recharge_rate = 5
        self.player.ships_defeated = 0
        self.player.in_combat = False
        self.player.current_enemy = None
        self.player.position = (50, 50)
        self.player.credits = 1000
        self.player.inventory = {}
        self.player.reputation = {"galactic_navy": 0, "fringe_colonies": 0}
        self.player.current_quest = None

        # Set up player methods
        self.player.attack.return_value = {
            "damage_dealt": 10,
            "critical_hit": False,
            "evaded": False,
            "destroyed": False,
        }
        self.player.take_damage.return_value = {
            "damage_taken": 5,
            "shield_damage": 5,
            "hull_damage": 0,
            "destroyed": False,
        }
        self.player.recharge_shield.return_value = 5
        self.player.get_combat_stats.return_value = {
            "attack_power": 10,
            "shield_strength": 50,
            "hull_strength": 100,
        }
        self.player._add_xp.return_value = {"xp_added": 50, "level_up": False}
        self.player.change_reputation.return_value = {
            "faction": "galactic_navy",
            "old_value": 0,
            "new_value": -2,
        }

        # Create the combat system
        self.combat_system = CombatSystem(self.player)

    def test_generate_enemy(self):
        """Test enemy generation."""
        # Test generating enemy with default parameters
        enemy = self.combat_system.generate_enemy()

        # Verify enemy was created
        self.assertIsInstance(enemy, EnemyShip)
        self.assertIn(enemy.difficulty, ["easy", "medium", "hard", "elite"])
        self.assertIn(enemy.ship_type, ["pirate", "patrol", "mercenary", "elite"])
        self.assertGreaterEqual(enemy.level, 1)
        self.assertLessEqual(enemy.level, 5)

        # Test generating enemy with specific parameters
        enemy = self.combat_system.generate_enemy(
            difficulty="hard", level=4, faction="galactic_navy", position=(30, 30)
        )

        self.assertEqual(enemy.difficulty, "hard")
        self.assertEqual(enemy.level, 4)
        self.assertEqual(enemy.faction, "galactic_navy")
        self.assertEqual(enemy.position, (30, 30))

    def test_start_combat(self):
        """Test starting combat."""
        # Start combat with auto-generated enemy
        result = self.combat_system.start_combat()

        # Verify combat started
        self.assertTrue(result["success"])
        self.assertTrue(self.combat_system.combat_active)
        self.assertIsNotNone(self.combat_system.current_enemy)
        self.assertTrue(self.player.in_combat)
        self.assertIsNotNone(self.player.current_enemy)

        # Verify enemy is in combat
        self.assertTrue(self.combat_system.current_enemy.in_combat)
        self.assertEqual(self.combat_system.current_enemy.target, self.player)

        # Test starting combat when already in combat
        result = self.combat_system.start_combat()
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "Already in combat")

    def test_end_combat(self):
        """Test ending combat."""
        # Start combat first
        self.combat_system.start_combat()

        # End combat
        result = self.combat_system.end_combat("Test end")

        # Verify combat ended
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "Test end")
        self.assertFalse(self.combat_system.combat_active)
        self.assertIsNone(self.combat_system.current_enemy)
        self.assertFalse(self.player.in_combat)

        # Test ending combat when not in combat
        result = self.combat_system.end_combat()
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "No active combat to end")

    def test_player_attack(self):
        """Test player attack."""
        # Start combat
        self.combat_system.start_combat()

        # Mock the enemy
        enemy = self.combat_system.current_enemy
        enemy.take_damage = MagicMock(
            return_value={
                "damage_taken": 10,
                "shield_damage": 10,
                "hull_damage": 0,
                "destroyed": False,
            }
        )
        enemy.get_stats = MagicMock(
            return_value={"shield_strength": 40, "hull_strength": 100}
        )

        # Execute player attack
        result = self.combat_system.player_attack()

        # Verify attack
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "player_attack")
        self.player.attack.assert_called_once_with(enemy)

        # Test player attack that destroys enemy
        self.player.attack.reset_mock()
        self.player.attack.return_value = {
            "damage_dealt": 50,
            "critical_hit": True,
            "evaded": False,
            "destroyed": True,
        }

        enemy.get_loot = MagicMock(
            return_value={
                "credits": 100,
                "xp": 50,
                "items": [{"type": "weapon_part", "name": "Laser Capacitor"}],
            }
        )

        # Execute player attack
        result = self.combat_system.player_attack()

        # Verify enemy destroyed
        self.assertTrue(result["enemy_destroyed"])
        self.assertTrue(result["combat_ended"])
        self.assertEqual(result["credits_gained"], 100)
        self.assertEqual(result["xp_gained"], 50)
        self.assertEqual(self.player.ships_defeated, 1)

    def test_enemy_attack(self):
        """Test enemy attack."""
        # Start combat
        self.combat_system.start_combat()

        # Mock the enemy
        enemy = self.combat_system.current_enemy
        enemy.attack = MagicMock(
            return_value={
                "damage_dealt": 8,
                "critical_hit": False,
                "evaded": False,
                "destroyed": False,
            }
        )
        enemy.get_stats = MagicMock(
            return_value={"attack_power": 8, "shield_strength": 30, "hull_strength": 80}
        )

        # Execute enemy attack
        result = self.combat_system.enemy_attack()

        # Verify attack
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "enemy_attack")
        enemy.attack.assert_called_once_with(self.player)

        # Test enemy attack that defeats player
        enemy.attack.reset_mock()
        enemy.attack.return_value = {
            "damage_dealt": 150,
            "critical_hit": True,
            "evaded": False,
            "destroyed": True,
        }

        # Execute enemy attack
        result = self.combat_system.enemy_attack()

        # Verify player defeated
        self.assertTrue(result["player_defeated"])
        self.assertTrue(result["combat_ended"])
        self.assertIn("credit_loss", result)

    def test_execute_combat_turn(self):
        """Test executing a full combat turn."""
        # Start combat
        self.combat_system.start_combat()

        # Mock the enemy
        enemy = self.combat_system.current_enemy
        enemy.attack_speed = 0.8  # Slower than player
        enemy.attack = MagicMock(
            return_value={
                "damage_dealt": 8,
                "critical_hit": False,
                "evaded": False,
                "destroyed": False,
            }
        )
        enemy.take_damage = MagicMock(
            return_value={
                "damage_taken": 10,
                "shield_damage": 10,
                "hull_damage": 0,
                "destroyed": False,
            }
        )
        enemy.recharge_shield = MagicMock(return_value=3)
        enemy.get_stats = MagicMock(
            return_value={"attack_power": 8, "shield_strength": 30, "hull_strength": 80}
        )

        # Execute combat turn
        result = self.combat_system.execute_combat_turn()

        # Verify turn execution
        self.assertTrue(result["success"])
        self.assertEqual(result["turn"], 1)
        self.assertTrue(result["player_goes_first"])
        self.assertEqual(result["player_shield_recharged"], 5)
        self.assertEqual(result["enemy_shield_recharged"], 3)

        # Verify both attacks happened
        self.assertIn("first_attack", result)
        self.assertIn("second_attack", result)

        # Test turn with player going second
        enemy.attack_speed = 1.2  # Faster than player

        # Execute combat turn
        result = self.combat_system.execute_combat_turn()

        # Verify turn execution
        self.assertTrue(result["success"])
        self.assertEqual(result["turn"], 2)
        self.assertFalse(result["player_goes_first"])

    def test_player_flee(self):
        """Test player fleeing from combat."""
        # Create a mock enemy with all required attributes
        mock_enemy = MagicMock(spec=EnemyShip)
        mock_enemy.aggression = 0.5
        mock_enemy.faction = "galactic_navy"
        mock_enemy.ship_type = "patrol"
        mock_enemy.level = 3
        mock_enemy.difficulty = "medium"
        mock_enemy.get_stats.return_value = {"ship_type": "patrol", "level": 3}

        # Start combat with our mock enemy
        self.combat_system.start_combat(mock_enemy)

        # Mock random to ensure flee success
        with patch("random.random", return_value=0.1):
            # Execute flee attempt
            result = self.combat_system.player_flee()

            # Verify successful flee
            self.assertTrue(result["success"])
            self.assertTrue(result["flee_success"])
            self.assertFalse(self.combat_system.combat_active)

            # Verify reputation change
            self.assertIn("reputation_change", result)

        # Create another mock enemy for failed flee test
        mock_enemy = MagicMock(spec=EnemyShip)
        mock_enemy.aggression = 0.5
        mock_enemy.faction = "fringe_colonies"
        mock_enemy.ship_type = "pirate"
        mock_enemy.level = 2
        mock_enemy.difficulty = "easy"
        mock_enemy.get_stats.return_value = {"ship_type": "pirate", "level": 2}
        mock_enemy.attack = MagicMock(
            return_value={
                "damage_dealt": 8,
                "critical_hit": False,
                "evaded": False,
                "destroyed": False,
            }
        )

        # Start combat with our second mock enemy
        self.combat_system.start_combat(mock_enemy)

        # Mock random to ensure flee failure
        with patch("random.random", return_value=0.9):
            # Execute flee attempt
            result = self.combat_system.player_flee()

            # Verify failed flee
            self.assertTrue(result["success"])
            self.assertFalse(result["flee_success"])
            self.assertTrue(self.combat_system.combat_active)

            # Verify enemy got free attack
            self.assertIn("enemy_attack", result)
            mock_enemy.attack.assert_called_once_with(self.player)


if __name__ == "__main__":
    unittest.main()
