"""
Combat system module: Handles combat encounters between the player and enemy ships.
"""

import logging
import random
from typing import Dict, Optional, Tuple, Any

from ..entities.player import Player
from ..entities.enemy_ship import EnemyShip


class CombatSystem:
    """
    Manages combat encounters between the player and enemy ships.
    """

    def __init__(self, player: Player) -> None:
        """
        Initialize the combat system.

        Args:
            player: The player instance
        """
        self.player = player
        self.current_enemy = None
        self.combat_active = False
        self.turn_count = 0
        self.max_turns = 20  # Maximum turns before combat auto-resolves
        self.combat_log = []

        logging.info("Combat system initialized")

    def generate_enemy(
        self,
        difficulty: str = None,
        level: Optional[int] = None,
        faction: Optional[str] = None,
        position: Optional[Tuple[int, int]] = None,
    ) -> EnemyShip:
        """
        Generate an enemy ship for combat.

        Args:
            difficulty: Difficulty level (easy, medium, hard, elite)
            level: Enemy level (1-5)
            faction: Optional faction alignment
            position: Initial position as (x, y) tuple

        Returns:
            EnemyShip: The generated enemy ship
        """
        # If no difficulty specified, determine based on player level
        if difficulty is None:
            if self.player.level <= 2:
                difficulty_options = ["easy", "medium"]
                weights = [0.7, 0.3]
            elif self.player.level <= 4:
                difficulty_options = ["easy", "medium", "hard"]
                weights = [0.2, 0.6, 0.2]
            else:
                difficulty_options = ["medium", "hard", "elite"]
                weights = [0.4, 0.4, 0.2]

            difficulty = random.choices(difficulty_options, weights=weights, k=1)[0]

        # If no level specified, determine based on player level
        if level is None:
            # Enemy level is player level +/- 1, clamped to 1-5
            level_variation = random.choice([-1, 0, 0, 1])
            level = max(1, min(5, self.player.level + level_variation))

        # If no faction specified, determine based on location or random
        if faction is None and random.random() < 0.3:
            from ..entities.player import GAME_FACTIONS

            faction = random.choice(GAME_FACTIONS)

        # Determine ship type based on difficulty and faction
        if difficulty == "elite":
            ship_type = "elite"
        elif faction == "galactic_navy":
            ship_type = "patrol"
        elif faction == "fringe_colonies":
            ship_type = "pirate"
        else:
            # Weight toward pirates for harder difficulties
            weights = [0.6, 0.2, 0.2] if difficulty == "hard" else [0.4, 0.3, 0.3]
            ship_type = random.choices(
                ["pirate", "patrol", "mercenary"], weights=weights, k=1
            )[0]

        # Create the enemy ship
        enemy = EnemyShip(
            ship_type=ship_type,
            difficulty=difficulty,
            level=level,
            faction=faction,
            position=position,
        )

        logging.info(f"Generated {difficulty} {ship_type} enemy (Level {level})")
        return enemy

    def start_combat(self, enemy: Optional[EnemyShip] = None) -> Dict[str, Any]:
        """
        Start a combat encounter with an enemy ship.

        Args:
            enemy: Optional enemy ship. If None, one will be generated.

        Returns:
            Dict with combat initialization results
        """
        # Check if player is already in combat
        if self.combat_active:
            return {
                "success": False,
                "reason": "Already in combat",
                "current_enemy": (
                    self.current_enemy.get_stats() if self.current_enemy else None
                ),
            }

        # Generate enemy if none provided
        if enemy is None:
            enemy = self.generate_enemy()

        # Initialize combat state
        self.current_enemy = enemy
        self.combat_active = True
        self.turn_count = 0
        self.combat_log = []

        # Set player and enemy combat flags
        self.player.in_combat = True
        self.player.current_enemy = enemy
        enemy.in_combat = True
        enemy.target = self.player

        # Log the start of combat
        combat_start_log = f"Combat started with {enemy.ship_type} ship (Level {enemy.level}, {enemy.difficulty})"
        self.combat_log.append(combat_start_log)
        logging.info(combat_start_log)

        return {
            "success": True,
            "enemy": enemy.get_stats(),
            "player": self.player.get_combat_stats(),
            "message": combat_start_log,
        }

    def end_combat(self, reason: str = "Combat ended") -> Dict[str, Any]:
        """
        End the current combat encounter.

        Args:
            reason: Reason for ending combat

        Returns:
            Dict with combat end results
        """
        if not self.combat_active or not self.current_enemy:
            return {"success": False, "reason": "No active combat to end"}

        # Log the end of combat
        combat_end_log = f"Combat ended: {reason}"
        self.combat_log.append(combat_end_log)
        logging.info(combat_end_log)

        # Reset combat state
        enemy_stats = self.current_enemy.get_stats() if self.current_enemy else None
        self.player.in_combat = False
        if self.current_enemy:
            self.current_enemy.in_combat = False
            self.current_enemy.target = None

        self.combat_active = False
        self.current_enemy = None

        return {
            "success": True,
            "reason": reason,
            "enemy": enemy_stats,
            "player": self.player.get_combat_stats(),
            "combat_log": self.combat_log,
            "turns": self.turn_count,
        }

    def player_attack(self) -> Dict[str, Any]:
        """
        Execute a player attack against the current enemy.

        Returns:
            Dict with attack results
        """
        if not self.combat_active or not self.current_enemy:
            return {"success": False, "reason": "No active combat"}

        # Player attacks enemy
        attack_result = self.player.attack(self.current_enemy)

        # Log the attack
        if attack_result.get("evaded", False):
            log_message = (
                f"Player attack missed! {self.current_enemy.ship_type} ship evaded."
            )
        elif attack_result.get("critical_hit", False):
            log_message = f"Player scored a CRITICAL HIT! Dealt {attack_result.get('damage_dealt', 0)} damage to {self.current_enemy.ship_type} ship."
        else:
            log_message = f"Player attacked for {attack_result.get('damage_dealt', 0)} damage to {self.current_enemy.ship_type} ship."

        self.combat_log.append(log_message)

        # Check if enemy is destroyed
        if attack_result.get("destroyed", False):
            self.player.ships_defeated += 1
            return self._handle_enemy_destroyed()

        return {
            "success": True,
            "action": "player_attack",
            "result": attack_result,
            "message": log_message,
            "enemy_stats": self.current_enemy.get_stats(),
            "player_stats": self.player.get_combat_stats(),
        }

    def enemy_attack(self) -> Dict[str, Any]:
        """
        Execute an enemy attack against the player.

        Returns:
            Dict with attack results
        """
        if not self.combat_active or not self.current_enemy:
            return {"success": False, "reason": "No active combat"}

        # Enemy attacks player
        attack_result = self.current_enemy.attack(self.player)

        # Log the attack
        if attack_result.get("evaded", False):
            log_message = "Enemy attack missed! Player evaded."
        elif attack_result.get("critical_hit", False):
            log_message = f"Enemy scored a CRITICAL HIT! Dealt {attack_result.get('damage_dealt', 0)} damage to player."
        else:
            log_message = f"Enemy attacked for {attack_result.get('damage_dealt', 0)} damage to player."

        self.combat_log.append(log_message)

        # Check if player is "destroyed" (in this implementation, player hull is set to 1 instead)
        if attack_result.get("destroyed", False):
            return self._handle_player_defeated()

        return {
            "success": True,
            "action": "enemy_attack",
            "result": attack_result,
            "message": log_message,
            "enemy_stats": self.current_enemy.get_stats(),
            "player_stats": self.player.get_combat_stats(),
        }

    def execute_combat_turn(self) -> Dict[str, Any]:
        """
        Execute a full combat turn (player and enemy actions).

        Returns:
            Dict with turn results
        """
        if not self.combat_active or not self.current_enemy:
            return {"success": False, "reason": "No active combat"}

        self.turn_count += 1
        turn_log = f"--- Turn {self.turn_count} ---"
        self.combat_log.append(turn_log)

        # Recharge shields at the start of the turn
        player_recharge = self.player.recharge_shield()
        enemy_recharge = self.current_enemy.recharge_shield()

        if player_recharge > 0:
            self.combat_log.append(
                f"Player shields recharged by {player_recharge} points."
            )
        if enemy_recharge > 0:
            self.combat_log.append(
                f"Enemy shields recharged by {enemy_recharge} points."
            )

        # Determine initiative (who attacks first) based on attack speed
        player_goes_first = self.player.attack_speed >= self.current_enemy.attack_speed

        # Execute attacks in order of initiative
        first_attack = None
        second_attack = None

        if player_goes_first:
            first_attack = self.player_attack()
            # Only do enemy attack if they weren't destroyed
            if not first_attack.get("combat_ended", False):
                second_attack = self.enemy_attack()
        else:
            first_attack = self.enemy_attack()
            # Only do player attack if they weren't defeated
            if not first_attack.get("combat_ended", False):
                second_attack = self.player_attack()

        # Check for combat timeout
        if self.turn_count >= self.max_turns:
            return self.end_combat("Combat timeout - maximum turns reached")

        # Compile results
        return {
            "success": True,
            "turn": self.turn_count,
            "player_goes_first": player_goes_first,
            "first_attack": first_attack,
            "second_attack": second_attack,
            "player_shield_recharged": player_recharge,
            "enemy_shield_recharged": enemy_recharge,
            "combat_log": self.combat_log[-5:],  # Last 5 log entries
            "enemy_stats": (
                self.current_enemy.get_stats() if self.current_enemy else None
            ),
            "player_stats": self.player.get_combat_stats(),
        }

    def player_flee(self) -> Dict[str, Any]:
        """
        Attempt to flee from combat.

        Returns:
            Dict with flee results
        """
        if not self.combat_active or not self.current_enemy:
            return {"success": False, "reason": "No active combat"}

        # Store current enemy reference to avoid it becoming None during the method execution
        enemy = self.current_enemy

        # Chance to flee depends on player level and enemy aggression
        flee_chance = 0.5 + (self.player.level * 0.05) - (enemy.aggression * 0.2)
        flee_chance = max(0.1, min(0.9, flee_chance))  # Clamp between 10% and 90%

        if random.random() < flee_chance:
            return self._handle_successful_flee(enemy)
        # Failed flee attempt - enemy gets a free attack
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

    def _handle_successful_flee(self, enemy):
        # Successful flee
        log_message = "Player successfully fled from combat!"
        self.combat_log.append(log_message)

        reputation_change = (
            self.player.change_reputation("galactic_navy", -2)
            if enemy.faction == "galactic_navy"
            else None
        )
        # End combat
        end_result = self.end_combat("Player fled")
        end_result["message"] = log_message
        end_result["flee_success"] = True

        if reputation_change:
            end_result["reputation_change"] = reputation_change

        return end_result

    def _handle_enemy_destroyed(self) -> Dict[str, Any]:
        """
        Handle enemy ship destruction.

        Returns:
            Dict with results
        """
        # Get loot from enemy
        loot = self.current_enemy.get_loot()

        # Award credits and XP to player
        self.player.credits += loot["credits"]
        xp_result = self.player._add_xp(loot["xp"])

        # Add items to player inventory
        for item in loot["items"]:
            if item["type"] not in self.player.inventory:
                self.player.inventory[item["type"]] = 0
            self.player.inventory[item["type"]] += 1

        # Log the destruction
        log_message = f"{self.current_enemy.ship_type} ship destroyed! Gained {loot['credits']} credits and {loot['xp']} XP."
        if loot["items"]:
            items_str = ", ".join([f"{item['name']}" for item in loot["items"]])
            log_message += f" Found items: {items_str}."

        self.combat_log.append(log_message)

        # Update combat quests if applicable
        if (
            self.player.current_quest
            and self.player.current_quest.get("type") == "combat"
        ):
            self.player.current_quest["current_enemies"] = min(
                self.player.current_quest.get("current_enemies", 0) + 1,
                self.player.current_quest.get("target_enemies", 0),
            )

            # Check if quest is now complete
            quest_complete = (
                self.player.current_quest["current_enemies"]
                >= self.player.current_quest["target_enemies"]
            )

            if quest_complete:
                log_message += " Combat quest objective completed!"

        # Update reputation if enemy had faction alignment
        reputation_change = None
        if self.current_enemy.faction:
            # Defeating a faction ship reduces reputation with that faction
            if self.current_enemy.faction in [
                "galactic_navy",
                "traders_coalition",
                "miners_guild",
                "explorers_union",
            ]:
                reputation_change = self.player.change_reputation(
                    self.current_enemy.faction, -3
                )

            # But increases reputation with opposing factions
            if (
                self.current_enemy.faction == "galactic_navy"
                and self.current_enemy.ship_type == "patrol"
            ):
                reputation_change = self.player.change_reputation("fringe_colonies", 2)

        # End the combat
        end_result = self.end_combat("Enemy destroyed")

        # Add results
        end_result.update(
            {
                "combat_ended": True,
                "enemy_destroyed": True,
                "message": log_message,
                "loot": loot,
                "xp_gained": loot["xp"],
                "credits_gained": loot["credits"],
                "level_up": xp_result.get("level_up", False),
            }
        )

        if reputation_change:
            end_result["reputation_change"] = reputation_change

        return end_result

    def _handle_player_defeated(self) -> Dict[str, Any]:
        """
        Handle player defeat in combat.

        Returns:
            Dict with results
        """
        # In this implementation, player isn't actually destroyed but suffers penalties

        # Calculate credit loss (10-20% of current credits)
        credit_loss_percent = random.uniform(0.1, 0.2)
        credit_loss = int(self.player.credits * credit_loss_percent)
        self.player.credits = max(0, self.player.credits - credit_loss)

        # Log the defeat
        log_message = f"Player ship critically damaged! Lost {credit_loss} credits."
        self.combat_log.append(log_message)

        # Update reputation if enemy had faction alignment
        reputation_change = None
        if self.current_enemy.faction and self.current_enemy.faction in [
            "galactic_navy",
            "traders_coalition",
            "miners_guild",
            "explorers_union",
        ]:
            reputation_change = self.player.change_reputation(
                self.current_enemy.faction, 1
            )

        # End the combat
        end_result = self.end_combat("Player defeated")

        # Add results
        end_result.update(
            {
                "combat_ended": True,
                "player_defeated": True,
                "message": log_message,
                "credit_loss": credit_loss,
                "credit_loss_percent": credit_loss_percent,
            }
        )

        if reputation_change:
            end_result["reputation_change"] = reputation_change

        return end_result
