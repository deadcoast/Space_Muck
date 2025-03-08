"""
Tests for the combat system module.
"""

import unittest
import random

from entities.player import Player
from entities.enemy_ship import EnemyShip
from systems.combat_system import CombatSystem


class DeterministicCombatSystem(CombatSystem):
    """A special version of CombatSystem that allows controlling flee outcomes for testing."""

    def __init__(self, player):
        """Initialize with a player and default flee outcome."""
        super().__init__(player)
        self._flee_outcome = None

    def set_flee_outcome(self, outcome):
        """Set whether flee attempts should succeed or fail.

        Args:
            outcome (bool): True for successful flee, False for failed flee
        """
        self._flee_outcome = outcome

    def player_flee(self):
        """Override flee logic to produce a deterministic outcome.

        Returns:
            Dict with flee results, determined by _flee_outcome
        """
        if not self.combat_active or not self.current_enemy:
            return {"success": False, "reason": "No active combat"}

        # Store current enemy reference
        enemy = self.current_enemy

        if self._flee_outcome is True:
            # Force successful flee
            return self._handle_successful_flee(enemy)
        # Force failed flee
        log_message = "Player failed to flee! Enemy gets a free attack."
        self.combat_log.append(log_message)

        # Enemy attack
        attack_result = self.enemy_attack()

        return {
            "success": True,
            "flee_success": False,
            "message": log_message,
            "enemy_attack": attack_result,
            "enemy_stats": enemy.get_stats(),
            "player_stats": self.player.get_combat_stats(),
        }


class TestCombatSystem(unittest.TestCase):
    """Test cases for the combat system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a real player instance
        self.player = Player(position=(50, 50))

        # Set up specific player attributes for testing
        self.player.level = 3
        self.player.weapon_level = 2
        self.player.shield_level = 2
        self.player.hull_level = 2
        self.player.attack_power = 50  # Increased attack power to ensure damage
        self.player.max_shield = 50
        self.player.current_shield = 50
        self.player.max_hull = 100
        self.player.current_hull = 100
        self.player.evasion = 0.2
        self.player.armor = 0.1
        self.player.crit_chance = 0.5  # Increased crit chance for more damage
        self.player.attack_speed = 1.0
        self.player.shield_recharge = 5
        self.player.ships_defeated = 0
        self.player.in_combat = False
        self.player.current_enemy = None
        self.player.credits = 1000
        self.player.inventory = {}

        # Ensure player has appropriate reputation values
        for faction in self.player.reputation:
            self.player.reputation[faction] = 0

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

        # Get reference to the enemy
        enemy = self.combat_system.current_enemy

        # Ensure enemy has low evasion and armor to guarantee hit
        enemy.evasion = 0.0
        enemy.armor = 0.0

        # Store initial shield and hull values to verify damage later
        initial_shield = enemy.current_shield
        initial_hull = enemy.current_hull

        # Execute player attack
        result = self.combat_system.player_attack()

        # Verify attack
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "player_attack")

        # Verify damage was dealt (shield or hull should have changed)
        current_stats = enemy.get_stats()
        self.assertTrue(
            current_stats["current_shield"] < initial_shield
            or current_stats["current_hull"] < initial_hull
        )

        # End current combat and start a new one
        self.combat_system.end_combat()
        self.combat_system.start_combat()

        # Get new enemy reference
        enemy = self.combat_system.current_enemy

        # Ensure enemy has zero evasion and armor
        enemy.evasion = 0.0
        enemy.armor = 0.0

        # Boost player's attack power to ensure enemy destruction
        original_attack_power = self.player.attack_power
        self.player.attack_power = 1000  # Very high damage

        # Set enemy's hull to 1 to ensure next attack destroys it
        enemy.current_shield = 0
        enemy.current_hull = 1

        # Track initial values for verification
        initial_credits = self.player.credits
        initial_xp = self.player.xp
        initial_ships_defeated = self.player.ships_defeated

        # Execute player attack that will destroy the enemy
        result = self.combat_system.player_attack()

        # Restore player's attack power
        self.player.attack_power = original_attack_power

        # Verify enemy destroyed (check combat ended and enemy destroyed flags)
        self.assertTrue(result["combat_ended"])
        self.assertTrue(result["enemy_destroyed"])

        # Verify player gained rewards
        self.assertGreater(self.player.credits, initial_credits)
        self.assertGreater(self.player.xp, initial_xp)
        self.assertEqual(self.player.ships_defeated, initial_ships_defeated + 1)

    def test_enemy_attack(self):
        """Test enemy attack."""
        # Start combat
        self.combat_system.start_combat()

        # Get reference to the enemy
        enemy = self.combat_system.current_enemy

        # Store initial player shield and hull values
        initial_shield = self.player.current_shield
        initial_hull = self.player.current_hull

        # Execute enemy attack
        result = self.combat_system.enemy_attack()

        # Verify attack
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "enemy_attack")

        # Verify player took damage (shield or hull should have changed)
        self.assertTrue(
            self.player.current_shield < initial_shield
            or self.player.current_hull < initial_hull
        )

        # End current combat and start a new one
        self.combat_system.end_combat()
        self.combat_system.start_combat()

        # Get reference to the enemy and boost its attack power
        enemy = self.combat_system.current_enemy
        enemy.attack_power = 500  # Set very high attack power

        # Set player's hull and shield to ensure next attack defeats them
        self.player.current_shield = 0
        self.player.current_hull = 1

        # Ensure player's evasion is zero so enemy attack will hit
        self.player.evasion = 0.0

        # Track initial credits for verification
        initial_credits = self.player.credits

        # Execute enemy attack that will defeat the player
        result = self.combat_system.enemy_attack()

        # Verify player defeated
        self.assertTrue(result["player_defeated"])
        self.assertTrue(result["combat_ended"])
        self.assertIn("credit_loss", result)
        self.assertTrue(self.player.credits < initial_credits)

    def test_execute_combat_turn(self):
        """Test executing a full combat turn."""
        # Start combat
        self.combat_system.start_combat()

        # Get reference to the enemy
        enemy = self.combat_system.current_enemy

        # Make player faster than enemy to test player going first
        self.player.attack_speed = 1.5
        enemy.attack_speed = 0.8  # Slower than player

        # Store initial values for verification
        initial_player_shield = self.player.current_shield
        initial_player_hull = self.player.current_hull
        initial_enemy_shield = enemy.current_shield
        initial_enemy_hull = enemy.current_hull

        # Execute combat turn
        result = self.combat_system.execute_combat_turn()

        # Verify turn execution
        self.assertTrue(result["success"])
        self.assertEqual(result["turn"], 1)
        self.assertTrue(result["player_goes_first"])

        # Verify shield recharge occurred if shields were not at maximum
        if initial_player_shield < self.player.max_shield:
            self.assertGreater(
                self.player.current_shield,
                initial_player_shield - result["player_damage_taken"],
            )

        if initial_enemy_shield < enemy.max_shield:
            self.assertGreater(
                enemy.current_shield,
                initial_enemy_shield - result["enemy_damage_taken"],
            )

        # Verify both attacks happened
        self.assertIn("first_attack", result)
        self.assertIn("second_attack", result)

        # Test turn with player going second
        self.player.attack_speed = 0.7  # Slower than enemy
        enemy.attack_speed = 1.2  # Faster than player

        # Execute combat turn
        result = self.combat_system.execute_combat_turn()

        # Verify turn execution
        self.assertTrue(result["success"])
        self.assertEqual(result["turn"], 2)
        self.assertFalse(result["player_goes_first"])

    def test_player_flee(self):
        """Test player fleeing from combat."""
        # Create a deterministic combat system for testing
        self.combat_system = DeterministicCombatSystem(self.player)

        # Test successful flee scenario
        self._test_successful_flee()

        # Test failed flee scenario
        self._test_failed_flee()

    def _test_successful_flee(self):
        """Test player successfully fleeing from combat."""
        # Create a real enemy with specific attributes
        enemy = EnemyShip(
            ship_type="patrol", difficulty="medium", level=3, faction="galactic_navy"
        )

        # Configure enemy for very likely flee success
        enemy.aggression = 0.1  # Low aggression makes flee easier

        # Set the DeterministicCombatSystem to allow successful flee
        self.combat_system.set_flee_outcome(True)

        # Start combat with our real enemy
        self.combat_system.start_combat(enemy)

        # Execute flee attempt
        result = self.combat_system.player_flee()

        # Verify successful flee
        self.assertTrue(result["success"])
        self.assertTrue(result["flee_success"])
        self.assertFalse(self.combat_system.combat_active)

        # Verify reputation change (only if enemy.faction is galactic_navy)
        if enemy.faction == "galactic_navy":
            self.assertIn("reputation_change", result)

    def _test_failed_flee(self):
        """Test player failing to flee from combat."""
        # Create another real enemy for failed flee test
        enemy = EnemyShip(
            ship_type="pirate",
            difficulty="hard",  # Harder difficulty
            level=5,  # Higher level
            faction="fringe_colonies",
        )

        # Configure enemy with high aggression to make flee difficult
        enemy.aggression = 0.9  # High aggression makes flee harder

        # Set the DeterministicCombatSystem to prevent flee
        self.combat_system.set_flee_outcome(False)

        # Track initial player health for verification
        self.player.current_shield = self.player.max_shield
        self.player.current_hull = self.player.max_hull
        initial_shield = self.player.current_shield
        initial_hull = self.player.current_hull

        # Start combat with our second real enemy
        self.combat_system.start_combat(enemy)

        # Execute flee attempt with deterministic failure
        result = self.combat_system.player_flee()

        # Verify failed flee
        self.assertTrue(result["success"])
        self.assertFalse(result["flee_success"])
        self.assertTrue(self.combat_system.combat_active)

        # Verify enemy got free attack
        self.assertIn("enemy_attack", result)

        # Verify player took damage from the free attack
        self.assertTrue(
            self.player.current_shield < initial_shield
            or self.player.current_hull < initial_hull
        )


if __name__ == "__main__":
    unittest.main()
