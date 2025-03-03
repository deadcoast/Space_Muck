"""
EnemyShip class: Represents enemy ships that the player can encounter and engage in combat.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any

from src.config import (
    COMBAT_BASE_ATTACK_POWER,
    COMBAT_BASE_ATTACK_SPEED,
    COMBAT_BASE_WEAPON_RANGE,
    COMBAT_BASE_CRIT_CHANCE,
    COMBAT_BASE_SHIELD_STRENGTH,
    COMBAT_BASE_SHIELD_RECHARGE,
    COMBAT_BASE_HULL_STRENGTH,
    COMBAT_BASE_EVASION,
    COMBAT_BASE_ARMOR,
    COMBAT_DIFFICULTY_MULTIPLIER,
    COMBAT_ENEMY_TYPES,
    COLOR_ERROR,
)
from src.entities.base_entity import BaseEntity


class EnemyShip(BaseEntity):
    """
    Represents an enemy ship that can engage in combat with the player.
    Extends BaseEntity to leverage common entity functionality.
    """

    def __init__(
        self,
        ship_type: str = "pirate",
        difficulty: str = "medium",
        level: int = 1,
        faction: Optional[str] = None,
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize a new enemy ship.

        Args:
            ship_type: Type of enemy ship (pirate, patrol, mercenary, elite)
            difficulty: Difficulty level (easy, medium, hard, elite)
            level: Ship level (1-5)
            faction: Optional faction alignment
            position: Initial position as (x, y) tuple
        """
        # Validate ship type
        if ship_type not in COMBAT_ENEMY_TYPES:
            ship_type = COMBAT_ENEMY_TYPES[0]  # Default to first type if invalid

        # Validate difficulty
        if difficulty not in COMBAT_DIFFICULTY_MULTIPLIER:
            difficulty = "medium"  # Default to medium if invalid

        # Validate level
        level = max(1, min(5, level))  # Ensure level is between 1 and 5

        # Generate a unique ID for this enemy ship
        ship_id = f"{ship_type}_{random.randint(1000, 9999)}"

        # Determine color based on ship type
        if ship_type == "pirate":
            color = (200, 50, 50)  # Red for pirates
        elif ship_type == "patrol":
            color = (50, 50, 200)  # Blue for patrol ships
        elif ship_type == "mercenary":
            color = (200, 200, 50)  # Yellow for mercenaries
        elif ship_type == "elite":
            color = (200, 50, 200)  # Purple for elite ships
        else:
            color = COLOR_ERROR  # Default error color

        # Call the parent class constructor
        super().__init__(
            entity_id=ship_id, entity_type="enemy_ship", color=color, position=position
        )

        # Store basic ship information
        self.ship_type = ship_type
        self.difficulty = difficulty
        self.level = level
        self.faction = faction

        # Apply difficulty multiplier to all stats
        difficulty_mult = COMBAT_DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)

        # Calculate level-based stat improvements (20% increase per level)
        level_mult = 1.0 + (level - 1) * 0.2

        # Initialize combat attributes with difficulty and level scaling
        self.attack_power = int(COMBAT_BASE_ATTACK_POWER * difficulty_mult * level_mult)
        self.attack_speed = COMBAT_BASE_ATTACK_SPEED * (1 + (level - 1) * 0.1)
        self.weapon_range = COMBAT_BASE_WEAPON_RANGE + (level - 1)
        self.crit_chance = COMBAT_BASE_CRIT_CHANCE + (level - 1) * 0.01

        self.max_shield = int(
            COMBAT_BASE_SHIELD_STRENGTH * difficulty_mult * level_mult
        )
        self.current_shield = self.max_shield
        self.shield_recharge = COMBAT_BASE_SHIELD_RECHARGE * (1 + (level - 1) * 0.1)

        self.max_hull = int(COMBAT_BASE_HULL_STRENGTH * difficulty_mult * level_mult)
        self.current_hull = self.max_hull

        self.evasion = COMBAT_BASE_EVASION + (level - 1) * 0.02
        self.armor = COMBAT_BASE_ARMOR + (level - 1) * 0.01

        # Ship behavior attributes
        self.aggression = 0.5  # Default aggression level (0-1)
        if ship_type == "pirate":
            self.aggression = 0.8  # Pirates are more aggressive
        elif ship_type == "patrol":
            self.aggression = 0.3  # Patrols are less aggressive

        # Combat state
        self.in_combat = False
        self.target = None
        self.combat_turns = 0
        self.last_action = None

        # Loot and rewards
        self.credits_reward = int(100 * difficulty_mult * level_mult)
        self.xp_reward = int(50 * difficulty_mult * level_mult)

        logging.info(f"Created {difficulty} {ship_type} enemy ship (Level {level})")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get the current stats of the enemy ship.

        Returns:
            Dict with ship statistics
        """
        return {
            "id": self.entity_id,
            "type": self.ship_type,
            "level": self.level,
            "difficulty": self.difficulty,
            "faction": self.faction,
            "attack_power": self.attack_power,
            "attack_speed": self.attack_speed,
            "weapon_range": self.weapon_range,
            "crit_chance": self.crit_chance,
            "current_shield": self.current_shield,
            "max_shield": self.max_shield,
            "shield_recharge": self.shield_recharge,
            "current_hull": self.current_hull,
            "max_hull": self.max_hull,
            "evasion": self.evasion,
            "armor": self.armor,
            "aggression": self.aggression,
            "credits_reward": self.credits_reward,
            "xp_reward": self.xp_reward,
        }

    def is_destroyed(self) -> bool:
        """
        Check if the ship is destroyed.

        Returns:
            bool: True if the ship is destroyed, False otherwise
        """
        return self.current_hull <= 0

    def take_damage(self, damage: int) -> Dict[str, Any]:
        """
        Apply damage to the ship, affecting shields first then hull.

        Args:
            damage: Amount of damage to apply

        Returns:
            Dict with damage results
        """
        original_damage = damage
        shield_damage = 0
        hull_damage = 0

        # Apply armor damage reduction
        damage = int(damage * (1 - self.armor))

        # Check for evasion
        if random.random() < self.evasion:
            return {
                "success": True,
                "evaded": True,
                "damage_dealt": 0,
                "shield_damage": 0,
                "hull_damage": 0,
                "original_damage": original_damage,
                "reduced_damage": damage,
                "current_shield": self.current_shield,
                "current_hull": self.current_hull,
                "destroyed": False,
            }

        # Apply damage to shields first
        if self.current_shield > 0:
            if damage <= self.current_shield:
                self.current_shield -= damage
                shield_damage = damage
                damage = 0
            else:
                shield_damage = self.current_shield
                damage -= self.current_shield
                self.current_shield = 0

        # Apply remaining damage to hull
        if damage > 0:
            self.current_hull -= damage
            hull_damage = damage

        # Check if ship is destroyed
        destroyed = self.is_destroyed()
        if destroyed:
            logging.info(f"Enemy ship {self.entity_id} destroyed")

        return {
            "success": True,
            "evaded": False,
            "damage_dealt": shield_damage + hull_damage,
            "shield_damage": shield_damage,
            "hull_damage": hull_damage,
            "original_damage": original_damage,
            "reduced_damage": shield_damage + hull_damage,
            "current_shield": self.current_shield,
            "current_hull": self.current_hull,
            "destroyed": destroyed,
        }

    def recharge_shield(self) -> int:
        """
        Recharge the ship's shields based on shield recharge rate.

        Returns:
            int: Amount of shield recharged
        """
        if self.current_shield < self.max_shield:
            recharge_amount = int(self.shield_recharge)
            old_shield = self.current_shield
            self.current_shield = min(
                self.max_shield, self.current_shield + recharge_amount
            )
            return self.current_shield - old_shield
        return 0

    def attack(self, target: Any) -> Dict[str, Any]:
        """
        Attack a target.

        Args:
            target: The target to attack (usually Player)

        Returns:
            Dict with attack results
        """
        # Calculate base damage
        damage = self.attack_power

        # Check for critical hit
        is_critical = random.random() < self.crit_chance
        if is_critical:
            damage = int(damage * 2)

        # Apply damage to target if it has a take_damage method
        if hasattr(target, "take_damage"):
            result = target.take_damage(damage)
            result["attacker"] = self.entity_id
            result["critical_hit"] = is_critical
            return result
        else:
            return {"success": False, "reason": "Invalid target", "damage_dealt": 0}

    def get_loot(self) -> Dict[str, Any]:
        """
        Get the loot from this ship when destroyed.

        Returns:
            Dict with loot information
        """
        # Base loot
        loot = {"credits": self.credits_reward, "xp": self.xp_reward, "items": []}

        # Chance to drop special items based on ship type and level
        item_chance = 0.1 + (self.level - 1) * 0.05  # 10% at level 1, +5% per level

        if random.random() < item_chance:
            # Generate a random item based on ship type
            if self.ship_type == "pirate":
                item = {
                    "type": "weapon_part",
                    "value": 50 * self.level,
                    "name": "Pirate Weapon Fragment",
                }
            elif self.ship_type == "patrol":
                item = {
                    "type": "shield_part",
                    "value": 60 * self.level,
                    "name": "Patrol Shield Component",
                }
            elif self.ship_type == "mercenary":
                item = {
                    "type": "engine_part",
                    "value": 70 * self.level,
                    "name": "Mercenary Engine Part",
                }
            elif self.ship_type == "elite":
                item = {
                    "type": "rare_tech",
                    "value": 100 * self.level,
                    "name": "Elite Technology Fragment",
                }
            else:
                item = {
                    "type": "scrap",
                    "value": 30 * self.level,
                    "name": "Scrap Metal",
                }

            loot["items"].append(item)

        return loot
