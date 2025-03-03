"""
Fleet class: Represents a fleet of ships in the game, extends BaseEntity.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import logging
import uuid

from entities.base_entity import BaseEntity
# Import config constants as needed


class Fleet(BaseEntity):
    """
    Represents a fleet of ships in the game.
    Extends BaseEntity to leverage common entity functionality.
    """

    def __init__(
        self,
        fleet_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        fleet_name: str = "Unnamed Fleet",
        color: Tuple[int, int, int] = (100, 100, 255),
        position: Optional[Tuple[int, int]] = None,
        ships: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a fleet of ships.

        Args:
            fleet_id: Unique identifier for the fleet (defaults to a UUID)
            owner_id: ID of the entity that owns this fleet
            fleet_name: Name of the fleet
            color: RGB color tuple for visualization
            position: Initial position as (x, y) tuple
            ships: Dictionary of ships in the fleet
        """
        # Call the parent class constructor
        super().__init__(
            entity_id=fleet_id, entity_type="fleet", color=color, position=position
        )

        # Fleet-specific attributes
        self.owner_id = owner_id
        self.fleet_name = fleet_name
        self.ships = ships or {}
        self.formation = "standard"  # Default formation
        self.speed = 1.0  # Base speed multiplier
        self.is_moving = False
        self.destination = None
        self.path = []
        self.fuel = 100
        self.max_fuel = 100

        logging.info(f"Fleet created: {self.fleet_name} (ID: {self.entity_id})")

    def add_ship(self, ship_id: str, ship_data: Dict[str, Any]) -> None:
        """
        Add a ship to the fleet.

        Args:
            ship_id: Unique identifier for the ship
            ship_data: Dictionary containing ship data
        """
        self.ships[ship_id] = ship_data
        logging.info(f"Ship added to fleet {self.fleet_name}: {ship_id}")

    def remove_ship(self, ship_id: str) -> bool:
        """
        Remove a ship from the fleet.

        Args:
            ship_id: ID of the ship to remove

        Returns:
            bool: True if the ship was removed, False if it wasn't in the fleet
        """
        if ship_id in self.ships:
            del self.ships[ship_id]
            logging.info(f"Ship removed from fleet {self.fleet_name}: {ship_id}")
            return True
        return False

    def set_formation(self, formation: str) -> None:
        """
        Set the fleet formation.

        Args:
            formation: Formation type (e.g., "standard", "defensive", "offensive")
        """
        self.formation = formation
        logging.info(f"Fleet {self.fleet_name} formation set to {formation}")

    def set_destination(self, x: int, y: int) -> None:
        """
        Set the fleet's destination.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.destination = (x, y)
        self.is_moving = True
        self.path = self._calculate_path()
        logging.info(f"Fleet {self.fleet_name} destination set to ({x}, {y})")

    def _calculate_path(self) -> List[Tuple[int, int]]:
        """
        Calculate a path to the destination.
        This is a placeholder for a more complex pathfinding algorithm.

        Returns:
            List[Tuple[int, int]]: List of coordinates representing the path
        """
        # Simple direct path for now
        if not self.position or not self.destination:
            return []

        start_x, start_y = self.position
        end_x, end_y = self.destination

        # Very basic straight line path
        # In a real implementation, this would use A* or another pathfinding algorithm
        path = []
        current_x, current_y = start_x, start_y

        while (current_x, current_y) != (end_x, end_y):
            if current_x < end_x:
                current_x += 1
            elif current_x > end_x:
                current_x -= 1

            if current_y < end_y:
                current_y += 1
            elif current_y > end_y:
                current_y -= 1

            path.append((current_x, current_y))

        return path

    def update(self, game_state: Dict[str, Any]) -> None:
        """
        Update the fleet state based on the current game state.

        Args:
            game_state: Dictionary containing the current game state
        """
        # Call the parent class update method
        super().update(game_state)

        # Move the fleet if it's moving
        if self.is_moving and self.path:
            next_position = self.path.pop(0)
            self.set_position(*next_position)

            # Check if we've reached the destination
            if not self.path:
                self.is_moving = False
                self.destination = None
                logging.info(f"Fleet {self.fleet_name} reached destination")

        # Update fuel consumption
        if self.is_moving and self.fuel > 0:
            self.fuel -= 0.1 * len(self.ships)  # Fuel consumption based on fleet size
            if self.fuel <= 0:
                self.fuel = 0
                self.is_moving = False
                logging.warning(f"Fleet {self.fleet_name} ran out of fuel")

    def get_fleet_strength(self) -> float:
        """
        Calculate the total strength of the fleet based on its ships.

        Returns:
            float: The fleet's total strength
        """
        strength = 0.0
        for ship_id, ship_data in self.ships.items():
            strength += ship_data.get("strength", 0)
        return strength

    def get_ship_count(self) -> int:
        """
        Get the number of ships in the fleet.

        Returns:
            int: Number of ships
        """
        return len(self.ships)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the fleet to a dictionary for serialization.
        Extends the BaseEntity to_dict method.

        Returns:
            Dict[str, Any]: Dictionary representation of the fleet
        """
        data = super().to_dict()
        data.update(
            {
                "owner_id": self.owner_id,
                "fleet_name": self.fleet_name,
                "ships": self.ships,
                "formation": self.formation,
                "speed": self.speed,
                "is_moving": self.is_moving,
                "destination": self.destination,
                "fuel": self.fuel,
                "max_fuel": self.max_fuel,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fleet":
        """
        Create a fleet from a dictionary.

        Args:
            data: Dictionary containing fleet data

        Returns:
            Fleet: New fleet instance
        """
        fleet = cls(
            fleet_id=data.get("entity_id"),
            owner_id=data.get("owner_id"),
            fleet_name=data.get("fleet_name", "Unnamed Fleet"),
            color=data.get("color", (100, 100, 255)),
            position=data.get("position"),
            ships=data.get("ships", {}),
        )
        fleet.formation = data.get("formation", "standard")
        fleet.speed = data.get("speed", 1.0)
        fleet.is_moving = data.get("is_moving", False)
        fleet.destination = data.get("destination")
        fleet.fuel = data.get("fuel", 100)
        fleet.max_fuel = data.get("max_fuel", 100)
        return fleet
