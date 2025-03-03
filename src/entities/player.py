"""
Player class: Represents the player character in the game, extends MinerEntity.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import logging

from src.entities.miner_entity import MinerEntity
from src.config import COLOR_PLAYER


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

        return {
            "success": True,
            "value": mining_value,
            "type": mineral_type,
            "total_credits": self.credits,
        }
