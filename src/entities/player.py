"""
Player class: Represents the player character in the game, extends MinerEntity.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import logging

from src.entities.miner_entity import MinerEntity
from src.config import (
    COLOR_PLAYER,
    COMBAT_BASE_ATTACK_POWER,
    COMBAT_BASE_ATTACK_SPEED,
    COMBAT_BASE_WEAPON_RANGE,
    COMBAT_BASE_CRIT_CHANCE,
    COMBAT_BASE_SHIELD_STRENGTH,
    COMBAT_BASE_SHIELD_RECHARGE,
    COMBAT_BASE_HULL_STRENGTH,
    COMBAT_BASE_EVASION,
    COMBAT_BASE_ARMOR,
    COMBAT_WEAPON_UPGRADE_COST,
    COMBAT_SHIELD_UPGRADE_COST,
    COMBAT_HULL_UPGRADE_COST,
    COMBAT_CRIT_MULTIPLIER,
)

# Reputation constants
REPUTATION_LEVELS = {
    "hostile": (-100, -51),  # Hostile: -100 to -51
    "unfriendly": (-50, -11),  # Unfriendly: -50 to -11
    "neutral": (-10, 10),  # Neutral: -10 to 10
    "friendly": (11, 50),  # Friendly: 11 to 50
    "allied": (51, 100),  # Allied: 51 to 100
}

# Available factions in the game
GAME_FACTIONS = [
    "miners_guild",  # Mining-focused faction
    "explorers_union",  # Exploration-focused faction
    "galactic_navy",  # Combat-focused faction
    "traders_coalition",  # Trading-focused faction
    "fringe_colonies",  # Outlaw/independent faction
]


class Player(MinerEntity):
    """
    Represents the player character in the game.
    Extends MinerEntity to leverage the same base functionality while adding player-specific features.
    """

    def __init__(
        self,
        race_id: int = 0,  # Player is typically race_id 0
        color: Tuple[int, int, int] = COLOR_PLAYER,
        birth_set: Optional[Set[int]] = None,
        survival_set: Optional[Set[int]] = None,
        initial_density: float = 0.001,
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize the player character.

        Args:
            race_id: Unique identifier for the player (default: 0)
            color: RGB color tuple for visualization (default: COLOR_PLAYER)
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            initial_density: Initial population density (0-1)
            position: Initial position as (x, y) tuple
        """
        # Initialize the base MinerEntity
        super().__init__(
            race_id, color, birth_set, survival_set, initial_density, position
        )

        # Player-specific attributes
        self.is_player = True
        self.credits = 1000  # Starting credits
        self.ship_level = 1  # Starting ship level
        self.mining_speed = 1.0  # Base mining speed multiplier
        self.inventory = {}  # Player's inventory
        self.discovered_anomalies = set()  # Set of discovered anomalies
        self.current_quest = None  # Current active quest
        self.completed_quests = []  # List of completed quests

        # Combat attributes
        self.weapon_level = 1  # Starting weapon level
        self.shield_level = 1  # Starting shield level
        self.hull_level = 1  # Starting hull level

        # Combat stats
        self.attack_power = COMBAT_BASE_ATTACK_POWER  # Base attack power
        self.attack_speed = COMBAT_BASE_ATTACK_SPEED  # Attacks per time unit
        self.weapon_range = COMBAT_BASE_WEAPON_RANGE  # Maximum attack distance
        self.crit_chance = COMBAT_BASE_CRIT_CHANCE  # Critical hit chance

        self.max_shield = COMBAT_BASE_SHIELD_STRENGTH  # Maximum shield points
        self.current_shield = self.max_shield  # Current shield points
        self.shield_recharge = COMBAT_BASE_SHIELD_RECHARGE  # Shield recharge rate

        self.max_hull = COMBAT_BASE_HULL_STRENGTH  # Maximum hull points
        self.current_hull = self.max_hull  # Current hull points

        self.evasion = COMBAT_BASE_EVASION  # Chance to evade attacks
        self.armor = COMBAT_BASE_ARMOR  # Damage reduction percentage

        # Combat tracking
        self.in_combat = False  # Whether player is in combat
        self.current_enemy = None  # Current enemy in combat
        self.combat_history = []  # History of combat encounters
        self.ships_defeated = 0  # Number of ships defeated

        # Level progression system
        self.level = 1  # Starting level
        self.xp = 0  # Experience points
        self.xp_to_next_level = 100  # XP needed for level 2
        self.level_bonuses = {  # Bonuses applied at each level
            # level: (mining_efficiency_bonus, mining_speed_bonus)
            1: (0.0, 0.0),  # Level 1: No bonus (base values)
            2: (0.05, 0.1),  # Level 2: +5% efficiency, +10% speed
            3: (0.1, 0.2),  # Level 3: +10% efficiency, +20% speed
            4: (0.15, 0.3),  # Level 4: +15% efficiency, +30% speed
            5: (0.2, 0.4),  # Level 5: +20% efficiency, +40% speed
        }

        # Reputation system
        self.reputation = {}
        # Initialize neutral reputation with all factions
        for faction in GAME_FACTIONS:
            self.reputation[faction] = 0
        self.faction_quests_completed = {faction: 0 for faction in GAME_FACTIONS}

        # Override some MinerEntity defaults for the player
        self.trait = "adaptive"  # Player always starts as adaptive
        self.mining_efficiency = 0.8  # Higher than base MinerEntity

        logging.info(f"Player initialized with {self.credits} credits")

    def move(self, dx: int, dy: int, field) -> bool:
        """
        Move the player in the specified direction.

        Args:
            dx: Change in x position
            dy: Change in y position
            field: The asteroid field the player is in

        Returns:
            bool: True if the move was successful, False otherwise
        """
        # Calculate new position
        new_x = self.territory_center[0] + dx if self.territory_center else dx
        new_y = self.territory_center[1] + dy if self.territory_center else dy

        # Check bounds
        if 0 <= new_x < field.width and 0 <= new_y < field.height:
            # Update territory center
            self.territory_center = (new_x, new_y)
            return True
        return False

    def _calculate_xp_gain(self, mining_value: int, mineral_type: str) -> int:
        """
        Calculate XP gain based on mining value and mineral type.

        Args:
            mining_value: The value obtained from mining
            mineral_type: The type of mineral mined (common, rare, anomaly)

        Returns:
            int: The amount of XP gained
        """
        # Base XP is 10% of mining value
        xp_gain = int(mining_value * 0.1)

        # Bonus XP for rare and anomalous minerals
        if mineral_type == "rare":
            xp_gain = int(xp_gain * 1.5)  # 50% bonus for rare minerals
        elif mineral_type == "anomaly":
            xp_gain = int(xp_gain * 3.0)  # 200% bonus for anomalies

        return max(1, xp_gain)  # Ensure at least 1 XP is gained

    def _add_xp(self, xp_amount: int) -> Dict[str, Any]:
        """
        Add XP to the player and handle level-ups.

        Args:
            xp_amount: Amount of XP to add

        Returns:
            Dict with level-up information
        """
        self.xp += xp_amount
        level_up_occurred = False
        old_level = self.level

        # Check for level up
        while (
            self.xp >= self.xp_to_next_level and self.level < 5
        ):  # Cap at level 5 for now
            self.level += 1
            self.xp -= self.xp_to_next_level
            # Increase XP required for next level (20% more each level)
            self.xp_to_next_level = int(self.xp_to_next_level * 1.2)
            level_up_occurred = True

            # Apply level bonuses
            if self.level in self.level_bonuses:
                efficiency_bonus, speed_bonus = self.level_bonuses[self.level]
                # Apply bonuses to base values (0.8 mining_efficiency, 1.0 mining_speed)
                self.mining_efficiency = 0.8 + efficiency_bonus
                self.mining_speed = 1.0 + speed_bonus

            logging.info(
                f"Player leveled up to {self.level}! New mining efficiency: {self.mining_efficiency}, speed: {self.mining_speed}"
            )

        return {
            "level_up": level_up_occurred,
            "old_level": old_level,
            "new_level": self.level,
            "xp_gained": xp_amount,
            "xp_current": self.xp,
            "xp_to_next": self.xp_to_next_level,
        }

    def get_level_progress(self) -> Dict[str, Any]:
        """
        Get the player's current level progress information.

        Returns:
            Dict with level progress information
        """
        return {
            "level": self.level,
            "xp": self.xp,
            "xp_to_next_level": self.xp_to_next_level,
            "progress_percent": (self.xp / self.xp_to_next_level) * 100
            if self.xp_to_next_level > 0
            else 100,
            "mining_efficiency": self.mining_efficiency,
            "mining_speed": self.mining_speed,
        }

    def upgrade_ship(self) -> Dict[str, Any]:
        """
        Upgrade the player's ship to the next level.

        Returns:
            Dict with upgrade results
        """
        # Ship upgrade costs and benefits
        ship_upgrades = {
            # level: (cost, cargo_bonus, mining_speed_bonus)
            1: (0, 0, 0),  # Base level
            2: (2000, 10, 0.2),  # Level 2
            3: (5000, 20, 0.3),  # Level 3
            4: (12000, 35, 0.4),  # Level 4
            5: (25000, 50, 0.5),  # Level 5
        }

        # Check if already at max level
        max_ship_level = max(ship_upgrades.keys())
        if self.ship_level >= max_ship_level:
            return {
                "success": False,
                "reason": f"Ship already at maximum level {max_ship_level}",
                "current_level": self.ship_level,
            }

        # Get upgrade cost and benefits
        next_level = self.ship_level + 1
        upgrade_cost, cargo_bonus, speed_bonus = ship_upgrades[next_level]

        # Check if player has enough credits
        if self.credits < upgrade_cost:
            return {
                "success": False,
                "reason": f"Not enough credits. Need {upgrade_cost}, have {self.credits}",
                "current_level": self.ship_level,
                "cost": upgrade_cost,
                "credits": self.credits,
            }

        # Apply the upgrade
        self.credits -= upgrade_cost
        self.ship_level = next_level

        # Apply bonuses (mining speed is additive to the base value)
        base_mining_speed = (
            1.0 + self.level_bonuses.get(self.level, (0, 0))[1]
        )  # Get current level bonus
        self.mining_speed = base_mining_speed + speed_bonus

        # Add cargo capacity to inventory (if not already present)
        if "cargo_capacity" not in self.inventory:
            self.inventory["cargo_capacity"] = 100  # Base capacity
        self.inventory["cargo_capacity"] += cargo_bonus

        logging.info(
            f"Ship upgraded to level {self.ship_level}. New mining speed: {self.mining_speed}, cargo capacity: {self.inventory['cargo_capacity']}"
        )

        return {
            "success": True,
            "new_level": self.ship_level,
            "cost": upgrade_cost,
            "remaining_credits": self.credits,
            "mining_speed": self.mining_speed,
            "cargo_capacity": self.inventory["cargo_capacity"],
        }

    def assign_quest(
        self, quest: Optional[Dict[str, Any]] = None, faction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assign a new quest to the player or generate one if none provided.

        Args:
            quest: Optional quest dictionary to assign. If None, a quest will be generated.
            faction: Optional faction to get a quest from. If None, a random faction is chosen.

        Returns:
            Dict with quest assignment results
        """
        # Check if player already has a quest
        if self.current_quest is not None:
            return {
                "success": False,
                "reason": "Player already has an active quest",
                "current_quest": self.current_quest,
            }

        # If faction is specified, check if it's valid
        if faction is not None and faction not in GAME_FACTIONS:
            return {"success": False, "reason": f"Unknown faction: {faction}"}

        # Check if reputation is too low for the faction
        if faction is not None and self.get_reputation_level(faction) == "hostile":
            return {
                "success": False,
                "reason": f"Reputation with {faction} is too low to accept quests",
            }

        # If no quest provided, generate one based on player level and faction
        if quest is None:
            quest = self._generate_quest(faction)

        # Assign the quest
        self.current_quest = quest
        logging.info(f"Quest assigned: {quest['id']} - {quest['description']}")

        return {"success": True, "quest": quest}

    def _generate_quest(self, faction: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a quest appropriate for the player's level and optionally for a specific faction.

        Args:
            faction: Optional faction to generate a quest for. If None, a random faction is chosen.

        Returns:
            Dict containing quest details
        """
        import random

        # Select faction if not provided
        if faction is None:
            # Filter out hostile factions (player can't get quests from them)
            available_factions = [
                f for f in GAME_FACTIONS if self.get_reputation_level(f) != "hostile"
            ]

            # If no factions available (all hostile), return a generic non-faction quest
            if not available_factions:
                faction = None
            else:
                faction = random.choice(available_factions)

        # Quest types - each faction has preferred quest types
        faction_quest_preferences = {
            "miners_guild": ["mining", "mining", "exploration"],  # Mining focused
            "explorers_union": [
                "exploration",
                "exploration",
                "mining",
            ],  # Exploration focused
            "galactic_navy": ["combat", "combat", "exploration"],  # Combat focused
            "traders_coalition": ["mining", "exploration", "combat"],  # Balanced
            "fringe_colonies": [
                "combat",
                "mining",
                "exploration",
            ],  # Slightly combat focused
        }

        # Default quest types if no faction
        quest_types = ["mining", "exploration", "combat"]

        # Select quest type based on faction preference or random if no faction
        if faction is not None and faction in faction_quest_preferences:
            quest_type = random.choice(faction_quest_preferences[faction])
        else:
            quest_type = random.choice(quest_types)

        # Scale difficulty and rewards based on player level
        difficulty_multiplier = 1.0 + (self.level - 1) * 0.5  # 1.0, 1.5, 2.0, 2.5, 3.0
        reward_multiplier = 1.0 + (self.level - 1) * 0.7  # 1.0, 1.7, 2.4, 3.1, 3.8

        # Apply faction reputation modifier to rewards
        if faction is not None:
            # Allied and friendly factions give better rewards
            rep_level = self.get_reputation_level(faction)
            if rep_level == "allied":
                reward_multiplier *= 1.2  # 20% bonus
            elif rep_level == "friendly":
                reward_multiplier *= 1.1  # 10% bonus

        # Generate quest based on type
        if quest_type == "mining":
            target_amount = int(500 * difficulty_multiplier)
            reward = int(800 * reward_multiplier)

            # Customize description based on faction
            if faction == "miners_guild":
                description = (
                    f"Mine {target_amount} worth of minerals for the Miners Guild"
                )
            elif faction == "traders_coalition":
                description = f"Collect {target_amount} worth of minerals for the Traders Coalition"
            else:
                description = f"Mine {target_amount} worth of minerals"

            quest = {
                "id": f"mining_{self.level}_{random.randint(1000, 9999)}",
                "type": "mining",
                "description": description,
                "target_amount": target_amount,
                "current_amount": 0,
                "reward": reward,
                "level_requirement": self.level,
            }

        elif quest_type == "exploration":
            target_anomalies = max(1, int(2 * difficulty_multiplier))
            reward = int(1200 * reward_multiplier)

            # Customize description based on faction
            if faction == "explorers_union":
                description = (
                    f"Chart {target_anomalies} new anomalies for the Explorers Union"
                )
            elif faction == "galactic_navy":
                description = (
                    f"Scout {target_anomalies} anomalies for the Galactic Navy"
                )
            else:
                description = f"Discover {target_anomalies} new anomalies"

            quest = {
                "id": f"exploration_{self.level}_{random.randint(1000, 9999)}",
                "type": "exploration",
                "description": description,
                "target_anomalies": target_anomalies,
                "current_anomalies": 0,
                "reward": reward,
                "level_requirement": self.level,
            }

        else:  # combat
            target_enemies = max(2, int(3 * difficulty_multiplier))
            reward = int(1500 * reward_multiplier)

            # Customize description based on faction
            if faction == "galactic_navy":
                description = (
                    f"Eliminate {target_enemies} pirate ships for the Galactic Navy"
                )
            elif faction == "fringe_colonies":
                description = (
                    f"Take down {target_enemies} rival ships for the Fringe Colonies"
                )
            else:
                description = f"Defeat {target_enemies} enemy ships"

            quest = {
                "id": f"combat_{self.level}_{random.randint(1000, 9999)}",
                "type": "combat",
                "description": description,
                "target_enemies": target_enemies,
                "current_enemies": 0,
                "reward": reward,
                "level_requirement": self.level,
            }

        # Add faction information if applicable
        if faction is not None:
            quest["faction"] = faction

        return quest

    def get_reputation_level(self, faction: str) -> str:
        """
        Get the reputation level with a specific faction.

        Args:
            faction: The faction to check reputation with

        Returns:
            String representing the reputation level (hostile, unfriendly, neutral, friendly, allied)
        """
        if faction not in self.reputation:
            return "neutral"  # Default to neutral for unknown factions

        rep_value = self.reputation[faction]

        # Find the appropriate reputation level based on the value
        for level, (min_val, max_val) in REPUTATION_LEVELS.items():
            if min_val <= rep_value <= max_val:
                return level

        # Fallback (should never happen with proper bounds)
        return "neutral"

    def change_reputation(self, faction: str, amount: int) -> Dict[str, Any]:
        """
        Change the player's reputation with a specific faction.

        Args:
            faction: The faction to change reputation with
            amount: Amount to change (positive or negative)

        Returns:
            Dict with reputation change results
        """
        if faction not in GAME_FACTIONS:
            return {"success": False, "reason": f"Unknown faction: {faction}"}

        old_level = self.get_reputation_level(faction)
        old_value = self.reputation[faction]

        # Apply the change, ensuring we stay within bounds (-100 to 100)
        self.reputation[faction] = max(
            -100, min(100, self.reputation[faction] + amount)
        )

        new_level = self.get_reputation_level(faction)
        new_value = self.reputation[faction]

        # Log significant reputation changes
        if old_level != new_level:
            logging.info(
                f"Reputation with {faction} changed from {old_level} to {new_level}"
            )

        return {
            "success": True,
            "faction": faction,
            "old_value": old_value,
            "new_value": new_value,
            "old_level": old_level,
            "new_level": new_level,
            "level_changed": old_level != new_level,
        }

    def get_faction_price_modifier(self, faction: str) -> float:
        """
        Get price modifier when trading with a specific faction.

        Args:
            faction: The faction to check

        Returns:
            Float representing price modifier (< 1.0 means discount, > 1.0 means markup)
        """
        rep_level = self.get_reputation_level(faction)

        # Price modifiers based on reputation level
        modifiers = {
            "hostile": 1.5,  # 50% markup
            "unfriendly": 1.2,  # 20% markup
            "neutral": 1.0,  # Standard prices
            "friendly": 0.9,  # 10% discount
            "allied": 0.8,  # 20% discount
        }

        return modifiers.get(rep_level, 1.0)

    def complete_quest(self) -> Dict[str, Any]:
        """
        Complete the current quest and receive rewards.

        Returns:
            Dict with quest completion results
        """
        # Check if player has a quest
        if self.current_quest is None:
            return {"success": False, "reason": "No active quest to complete"}

        # Get quest details
        quest = self.current_quest
        quest_type = quest.get("type", "unknown")

        # Check if quest is completable
        if quest_type == "mining" and quest.get("current_amount", 0) < quest.get(
            "target_amount", 0
        ):
            return {
                "success": False,
                "reason": f"Mining quest not complete. Current: {quest.get('current_amount')}, Target: {quest.get('target_amount')}",
            }
        elif quest_type == "exploration" and quest.get(
            "current_anomalies", 0
        ) < quest.get("target_anomalies", 0):
            return {
                "success": False,
                "reason": f"Exploration quest not complete. Current: {quest.get('current_anomalies')}, Target: {quest.get('target_anomalies')}",
            }
        elif quest_type == "combat" and quest.get("current_enemies", 0) < quest.get(
            "target_enemies", 0
        ):
            return {
                "success": False,
                "reason": f"Combat quest not complete. Current: {quest.get('current_enemies')}, Target: {quest.get('target_enemies')}",
            }

        # Complete the quest
        reward = quest.get("reward", 0)
        self.credits += reward

        # Add to completed quests and clear current quest
        self.completed_quests.append(quest)
        self.current_quest = None

        # Award bonus XP for completing the quest
        bonus_xp = int(reward * 0.1)  # 10% of reward as XP
        xp_result = self._add_xp(bonus_xp)

        # Update reputation if this is a faction quest
        reputation_change = 0
        faction = quest.get("faction", None)
        if faction in GAME_FACTIONS:
            # Reputation gain depends on quest level
            level_req = quest.get("level_requirement", 1)
            reputation_change = 5 + (level_req - 1) * 2  # 5 for level 1, +2 per level
            self.change_reputation(faction, reputation_change)
            self.faction_quests_completed[faction] += 1

        logging.info(
            f"Quest completed: {quest['id']}. Reward: {reward} credits, {bonus_xp} XP"
        )

        result = {
            "success": True,
            "quest": quest,
            "reward": reward,
            "bonus_xp": bonus_xp,
            "level_up": xp_result.get("level_up", False),
            "new_level": xp_result.get("new_level", self.level),
            "total_credits": self.credits,
        }

        # Add reputation info if applicable
        if faction in GAME_FACTIONS:
            result["faction"] = faction
            result["reputation_change"] = reputation_change
            result["new_reputation"] = self.reputation[faction]
            result["reputation_level"] = self.get_reputation_level(faction)

        return result

    def upgrade_weapon(self) -> Dict[str, Any]:
        """
        Upgrade the player's weapon system to the next level.

        Returns:
            Dict with upgrade results
        """
        # Check if already at max level
        max_weapon_level = len(COMBAT_WEAPON_UPGRADE_COST) - 1
        if self.weapon_level >= max_weapon_level:
            return {
                "success": False,
                "reason": f"Weapon already at maximum level {max_weapon_level}",
                "current_level": self.weapon_level,
            }

        # Get upgrade cost
        next_level = self.weapon_level + 1
        upgrade_cost = COMBAT_WEAPON_UPGRADE_COST[next_level]

        # Check if player has enough credits
        if self.credits < upgrade_cost:
            return {
                "success": False,
                "reason": f"Not enough credits. Need {upgrade_cost}, have {self.credits}",
                "current_level": self.weapon_level,
                "cost": upgrade_cost,
                "credits": self.credits,
            }

        # Apply the upgrade
        self.credits -= upgrade_cost
        self.weapon_level = next_level

        # Update combat stats (20% increase per level)
        self.attack_power = int(
            COMBAT_BASE_ATTACK_POWER * (1 + (self.weapon_level - 1) * 0.2)
        )
        self.attack_speed = COMBAT_BASE_ATTACK_SPEED * (
            1 + (self.weapon_level - 1) * 0.1
        )
        self.weapon_range = COMBAT_BASE_WEAPON_RANGE + (self.weapon_level - 1)
        self.crit_chance = COMBAT_BASE_CRIT_CHANCE + (self.weapon_level - 1) * 0.01

        logging.info(
            f"Weapon upgraded to level {self.weapon_level}. New attack power: {self.attack_power}"
        )

        return {
            "success": True,
            "new_level": self.weapon_level,
            "cost": upgrade_cost,
            "remaining_credits": self.credits,
            "attack_power": self.attack_power,
            "attack_speed": self.attack_speed,
            "weapon_range": self.weapon_range,
            "crit_chance": self.crit_chance,
        }

    def upgrade_shield(self) -> Dict[str, Any]:
        """
        Upgrade the player's shield system to the next level.

        Returns:
            Dict with upgrade results
        """
        # Check if already at max level
        max_shield_level = len(COMBAT_SHIELD_UPGRADE_COST) - 1
        if self.shield_level >= max_shield_level:
            return {
                "success": False,
                "reason": f"Shield already at maximum level {max_shield_level}",
                "current_level": self.shield_level,
            }

        # Get upgrade cost
        next_level = self.shield_level + 1
        upgrade_cost = COMBAT_SHIELD_UPGRADE_COST[next_level]

        # Check if player has enough credits
        if self.credits < upgrade_cost:
            return {
                "success": False,
                "reason": f"Not enough credits. Need {upgrade_cost}, have {self.credits}",
                "current_level": self.shield_level,
                "cost": upgrade_cost,
                "credits": self.credits,
            }

        # Apply the upgrade
        self.credits -= upgrade_cost
        self.shield_level = next_level

        # Update shield stats (30% increase per level)
        old_max_shield = self.max_shield
        self.max_shield = int(
            COMBAT_BASE_SHIELD_STRENGTH * (1 + (self.shield_level - 1) * 0.3)
        )
        self.current_shield += (
            self.max_shield - old_max_shield
        )  # Add the difference to current shields
        self.shield_recharge = COMBAT_BASE_SHIELD_RECHARGE * (
            1 + (self.shield_level - 1) * 0.2
        )

        logging.info(
            f"Shield upgraded to level {self.shield_level}. New max shield: {self.max_shield}"
        )

        return {
            "success": True,
            "new_level": self.shield_level,
            "cost": upgrade_cost,
            "remaining_credits": self.credits,
            "max_shield": self.max_shield,
            "current_shield": self.current_shield,
            "shield_recharge": self.shield_recharge,
        }

    def upgrade_hull(self) -> Dict[str, Any]:
        """
        Upgrade the player's hull to the next level.

        Returns:
            Dict with upgrade results
        """
        # Check if already at max level
        max_hull_level = len(COMBAT_HULL_UPGRADE_COST) - 1
        if self.hull_level >= max_hull_level:
            return {
                "success": False,
                "reason": f"Hull already at maximum level {max_hull_level}",
                "current_level": self.hull_level,
            }

        # Get upgrade cost
        next_level = self.hull_level + 1
        upgrade_cost = COMBAT_HULL_UPGRADE_COST[next_level]

        # Check if player has enough credits
        if self.credits < upgrade_cost:
            return {
                "success": False,
                "reason": f"Not enough credits. Need {upgrade_cost}, have {self.credits}",
                "current_level": self.hull_level,
                "cost": upgrade_cost,
                "credits": self.credits,
            }

        # Apply the upgrade
        self.credits -= upgrade_cost
        self.hull_level = next_level

        # Update hull stats (25% increase per level) and defensive stats
        old_max_hull = self.max_hull
        self.max_hull = int(
            COMBAT_BASE_HULL_STRENGTH * (1 + (self.hull_level - 1) * 0.25)
        )
        self.current_hull += (
            self.max_hull - old_max_hull
        )  # Add the difference to current hull
        self.evasion = COMBAT_BASE_EVASION + (self.hull_level - 1) * 0.02
        self.armor = COMBAT_BASE_ARMOR + (self.hull_level - 1) * 0.01

        logging.info(
            f"Hull upgraded to level {self.hull_level}. New max hull: {self.max_hull}"
        )

        return {
            "success": True,
            "new_level": self.hull_level,
            "cost": upgrade_cost,
            "remaining_credits": self.credits,
            "max_hull": self.max_hull,
            "current_hull": self.current_hull,
            "evasion": self.evasion,
            "armor": self.armor,
        }

    def take_damage(self, damage: int) -> Dict[str, Any]:
        """
        Apply damage to the player's ship, affecting shields first then hull.

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
        destroyed = self.current_hull <= 0
        if destroyed:
            logging.warning(f"Player ship destroyed!")
            # Reset hull to 1 to prevent actual game over in this implementation
            self.current_hull = 1

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
        Recharge the player's shields based on shield recharge rate.

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

    def repair_hull(self, amount: int) -> Dict[str, Any]:
        """
        Repair the player's hull by the specified amount.

        Args:
            amount: Amount of hull points to repair

        Returns:
            Dict with repair results
        """
        if self.current_hull >= self.max_hull:
            return {
                "success": False,
                "reason": "Hull already at maximum",
                "current_hull": self.current_hull,
                "max_hull": self.max_hull,
            }

        old_hull = self.current_hull
        self.current_hull = min(self.max_hull, self.current_hull + amount)
        repair_amount = self.current_hull - old_hull

        return {
            "success": True,
            "repair_amount": repair_amount,
            "current_hull": self.current_hull,
            "max_hull": self.max_hull,
            "fully_repaired": self.current_hull >= self.max_hull,
        }

    def attack(self, target: Any) -> Dict[str, Any]:
        """
        Attack a target.

        Args:
            target: The target to attack (usually an EnemyShip)

        Returns:
            Dict with attack results
        """
        # Import here to avoid circular imports
        import random

        # Calculate base damage
        damage = self.attack_power

        # Check for critical hit
        is_critical = random.random() < self.crit_chance
        if is_critical:
            damage = int(damage * COMBAT_CRIT_MULTIPLIER)

        # Apply damage to target if it has a take_damage method
        if hasattr(target, "take_damage"):
            result = target.take_damage(damage)
            result["attacker"] = "player"
            result["critical_hit"] = is_critical
            return result
        else:
            return {"success": False, "reason": "Invalid target", "damage_dealt": 0}

    def get_combat_stats(self) -> Dict[str, Any]:
        """
        Get the player's current combat statistics.

        Returns:
            Dict with combat statistics
        """
        return {
            "weapon_level": self.weapon_level,
            "shield_level": self.shield_level,
            "hull_level": self.hull_level,
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
            "ships_defeated": self.ships_defeated,
        }

    def mine(self, x: int, y: int, field) -> Dict[str, Any]:
        """
        Mine an asteroid at the specified position.

        Args:
            x: X position to mine
            y: Y position to mine
            field: The asteroid field to mine from

        Returns:
            Dict with mining results (value, type, etc.)
        """
        if not (0 <= x < field.width and 0 <= y < field.height):
            return {"success": False, "reason": "Position out of bounds"}

        if field.grid[y, x] <= 0:
            return {"success": False, "reason": "No asteroid to mine"}

        # Get asteroid value and type
        value = field.grid[y, x]
        rare_type = field.rare_grid[y, x]

        # Calculate mining value with player's efficiency
        mining_value = int(value * self.mining_efficiency * self.mining_speed)

        # Apply bonuses for rare types
        if rare_type == 1:  # Rare
            mining_value = int(mining_value * field.rare_bonus_multiplier)
            mineral_type = "rare"
        elif rare_type == 2:  # Anomalous
            mining_value = int(mining_value * field.rare_bonus_multiplier * 2)
            mineral_type = "anomaly"
            # Add to discovered anomalies
            self.discovered_anomalies.add(f"anomaly_{x}_{y}")

            # Update exploration quest if active
            if self.current_quest and self.current_quest.get("type") == "exploration":
                self.current_quest["current_anomalies"] = min(
                    self.current_quest.get("current_anomalies", 0) + 1,
                    self.current_quest.get("target_anomalies", 0),
                )
        else:
            mineral_type = "common"

        # Add to player's credits
        self.credits += mining_value

        # Remove the asteroid
        field.grid[y, x] = 0
        field.rare_grid[y, x] = 0

        # Update mining stats
        self.last_income = mining_value
        self.income_history.append(mining_value)
        if len(self.income_history) > 100:
            self.income_history.pop(0)

        # Update mining quest if active
        if self.current_quest and self.current_quest.get("type") == "mining":
            self.current_quest["current_amount"] = min(
                self.current_quest.get("current_amount", 0) + mining_value,
                self.current_quest.get("target_amount", 0),
            )

        # Award XP based on mining value and mineral type
        xp_gain = self._calculate_xp_gain(mining_value, mineral_type)
        self._add_xp(xp_gain)

        return {
            "success": True,
            "value": mining_value,
            "type": mineral_type,
            "total_credits": self.credits,
            "xp_gained": xp_gain,
            "level": self.level,
            "xp": self.xp,
        }
