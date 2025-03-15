"""
Fleet Management System: Handles fleet creation, management, and operations.

This module provides functionality for creating and managing fleets of ships,
including formation management, movement coordination, and fleet-wide operations.
"""

import heapq

# Standard library imports
import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Local application imports
from config import GAME_MAP_SIZE
from entities.enemy_ship import EnemyShip

# Third-party library imports


# Fleet formation types
FLEET_FORMATIONS = {
    "line": "Ships arranged in a horizontal line",
    "column": "Ships arranged in a vertical line",
    "wedge": "Ships arranged in a V formation",
    "echelon": "Ships arranged in a diagonal line",
    "circle": "Ships arranged in a circular formation",
    "scatter": "Ships arranged randomly within a defined area",
}

# Fleet command hierarchy levels
COMMAND_HIERARCHY = {
    "flagship": "Fleet command ship",
    "squadron_leader": "Leader of a squadron within the fleet",
    "wing_commander": "Commander of a wing of ships",
    "ship_captain": "Individual ship captain",
}

# Fleet movement patterns
MOVEMENT_PATTERNS = {
    "patrol": "Regular patrol between designated points",
    "escort": "Protective formation around a designated asset",
    "intercept": "Move to intercept a designated target",
    "blockade": "Maintain position to block a designated area",
    "scout": "Explore and gather intelligence in a designated area",
}


class Fleet:
    """
    Represents a fleet of ships that can be managed as a single unit.
    """

    def __init__(
        self,
        fleet_id: str,
        name: str,
        faction: Optional[str] = None,
        position: Optional[Tuple[int, int]] = None,
        formation: str = "line",
        commander_level: int = 1,
    ) -> None:
        """
        Initialize a new fleet.

        Args:
            fleet_id: Unique identifier for the fleet
            name: Display name for the fleet
            faction: Optional faction alignment
            position: Initial position as (x, y) tuple
            formation: Initial formation type
            commander_level: Level of the fleet commander (affects capabilities)
        """
        self.fleet_id = fleet_id
        self.name = name
        self.faction = faction
        self.position = position or (
            random.randint(0, GAME_MAP_SIZE[0]),
            random.randint(0, GAME_MAP_SIZE[1]),
        )
        self.formation = formation if formation in FLEET_FORMATIONS else "line"
        self.commander_level = max(1, min(5, commander_level))

        # Fleet composition
        self.ships: List[EnemyShip] = []
        self.flagship: Optional[EnemyShip] = None

        # Fleet attributes
        self.speed = 1.0  # Base movement speed
        self.combat_effectiveness = 1.0  # Base combat effectiveness multiplier
        self.morale = 1.0  # Base morale (affects performance)
        self.detection_range = 10  # Base detection range

        # Fleet state
        self.is_active = True
        self.current_orders = None
        self.destination = None
        self.waypoints = []
        self.path: List[Tuple[int, int]] = []  # For pathfinding

        # Combat state
        self.in_combat = False
        self.target_fleet: Optional["Fleet"] = None
        self.combat_stance = "balanced"  # balanced, aggressive, defensive, evasive

        # Resource management
        self.resources = {
            "common_minerals": 100,
            "rare_minerals": 50,
            "anomalous_materials": 25,
            "fuel_cells": 200,
            "ship_parts": 100,
        }
        self.resource_capacity = {
            "common_minerals": 1000,
            "rare_minerals": 500,
            "anomalous_materials": 250,
            "fuel_cells": 2000,
            "ship_parts": 1000,
        }
        self.resource_distribution_method = "priority"  # priority, equal, proportional
        self.auto_engage = (
            False  # Whether to automatically engage enemy fleets when detected
        )
        self.resource_consumption_rate = {  # Units per game day
            "common_minerals": 0.5,
            "rare_minerals": 0.2,
            "anomalous_materials": 0.1,
            "fuel_cells": 2.0,
            "ship_parts": 0.5,
        }
        self.resource_priority = {  # Priority for resource distribution (1-10)
            "common_minerals": 3,
            "rare_minerals": 5,
            "anomalous_materials": 7,
            "fuel_cells": 10,  # Highest priority
            "ship_parts": 8,
        }
        self.last_resource_update = 0  # Last time resources were updated
        self.resource_distribution_method = "priority"  # priority, equal, proportional
        self.attack_same_faction = False  # Whether to attack fleets of the same faction

        logging.info(f"Created fleet '{name}' with ID {fleet_id}")

    def add_ship(self, ship: EnemyShip, is_flagship: bool = False) -> bool:
        """
        Add a ship to the fleet.

        Args:
            ship: The ship to add
            is_flagship: Whether this ship should be the fleet flagship

        Returns:
            bool: True if the ship was added successfully, False otherwise
        """
        if ship in self.ships:
            logging.warning(
                f"Ship {ship.entity_id} is already in fleet {self.fleet_id}"
            )
            return False

        self.ships.append(ship)

        # Update ship position based on fleet formation
        self._update_ship_positions()

        # Set as flagship if specified or if this is the first ship
        if is_flagship or len(self.ships) == 1:
            self.set_flagship(ship)

        logging.info(f"Added ship {ship.entity_id} to fleet {self.fleet_id}")
        return True

    def remove_ship(self, ship: EnemyShip) -> bool:
        """
        Remove a ship from the fleet.

        Args:
            ship: The ship to remove

        Returns:
            bool: True if the ship was removed successfully, False otherwise
        """
        if ship not in self.ships:
            logging.warning(f"Ship {ship.entity_id} is not in fleet {self.fleet_id}")
            return False

        self.ships.remove(ship)

        # If the removed ship was the flagship, select a new flagship
        if self.flagship == ship:
            self.flagship = self.ships[0] if self.ships else None

        # Update positions of remaining ships
        self._update_ship_positions()

        logging.info(f"Removed ship {ship.entity_id} from fleet {self.fleet_id}")
        return True

    def set_flagship(self, ship: EnemyShip) -> bool:
        """
        Set a ship as the fleet flagship.

        Args:
            ship: The ship to set as flagship

        Returns:
            bool: True if the flagship was set successfully, False otherwise
        """
        if ship not in self.ships:
            logging.warning(
                f"Cannot set ship {ship.entity_id} as flagship: not in fleet {self.fleet_id}"
            )
            return False

        self.flagship = ship

        # Update positions to ensure flagship is in the correct position
        self._update_ship_positions()

        logging.info(f"Set ship {ship.entity_id} as flagship for fleet {self.fleet_id}")
        return True

    def _update_ship_positions(self) -> None:
        """
        Update the positions of all ships in the fleet based on the current formation.
        """
        if not self.ships:
            return

        # Fleet position is centered on the flagship or the first ship
        center_ship = self.flagship or self.ships[0]
        self.position = center_ship.position

        # Apply formation-specific positioning
        if self.formation == "line":
            self._apply_line_formation()
        elif self.formation == "column":
            self._apply_column_formation()
        elif self.formation == "wedge":
            self._apply_wedge_formation()
        elif self.formation == "echelon":
            self._apply_echelon_formation()
        elif self.formation == "circle":
            self._apply_circle_formation()
        else:  # Default to scatter
            self._apply_scatter_formation()

    def _apply_line_formation(self) -> None:
        """
        Arrange ships in a horizontal line formation.
        Flagship (or first ship) is in the center, other ships extend to the left and right.
        """
        if not self.ships:
            return

        # Get the center position (flagship or first ship)
        center_ship = self.flagship or self.ships[0]
        center_x, center_y = center_ship.position

        # Spacing between ships
        spacing = 2

        # Place ships in a line centered on the flagship
        ship_count = len(self.ships)
        for i, ship in enumerate(self.ships):
            # Skip the center ship (flagship) as it stays at the center
            if ship == center_ship:
                continue

            # Calculate position in line
            # If odd number of ships, flagship is in the middle
            # If even number, flagship is slightly to the left of middle
            if ship_count % 2 == 1 or i < ship_count // 2:  # Odd number of ships
                offset = i - ship_count // 2
            else:
                offset = i - ship_count // 2 + 1

            # Set the ship's position
            ship.position = (center_x + offset * spacing, center_y)

    def _apply_column_formation(self) -> None:
        """
        Arrange ships in a vertical column formation.
        Flagship (or first ship) is at the front, other ships follow behind.
        """
        if not self.ships:
            return

        # Get the center position (flagship or first ship)
        center_ship = self.flagship or self.ships[0]
        center_x, center_y = center_ship.position

        # Spacing between ships
        spacing = 2

        # Place flagship at the front, other ships behind in a column
        ships_to_position = [ship for ship in self.ships if ship != center_ship]

        # Position the center ship (flagship)
        # No need to change its position as it's already at the center

        # Position other ships in a column behind the flagship
        for i, ship in enumerate(ships_to_position):
            ship.position = (center_x, center_y + (i + 1) * spacing)

    def _apply_wedge_formation(self) -> None:
        """
        Arrange ships in a V (wedge) formation.
        Flagship (or first ship) is at the point of the V, other ships form the arms.
        """
        if not self.ships:
            return

        # Get the center position (flagship or first ship)
        center_ship = self.flagship or self.ships[0]
        center_x, center_y = center_ship.position

        # Spacing between ships
        spacing = 2

        # Place flagship at the point, other ships form the V
        ships_to_position = [ship for ship in self.ships if ship != center_ship]
        left_wing = ships_to_position[: len(ships_to_position) // 2]
        right_wing = ships_to_position[len(ships_to_position) // 2 :]

        # Position left wing
        for i, ship in enumerate(left_wing):
            ship.position = (center_x - (i + 1) * spacing, center_y + (i + 1) * spacing)

        # Position right wing
        for i, ship in enumerate(right_wing):
            ship.position = (center_x + (i + 1) * spacing, center_y + (i + 1) * spacing)

    def _apply_echelon_formation(self) -> None:
        """
        Arrange ships in a diagonal line (echelon) formation.
        Flagship (or first ship) is at the front, other ships follow diagonally.
        """
        if not self.ships:
            return

        # Get the center position (flagship or first ship)
        center_ship = self.flagship or self.ships[0]
        center_x, center_y = center_ship.position

        # Spacing between ships
        spacing = 2

        # Place flagship at the front, other ships follow in a diagonal line
        ships_to_position = [ship for ship in self.ships if ship != center_ship]

        # Position other ships in a diagonal line behind the flagship
        for i, ship in enumerate(ships_to_position):
            ship.position = (center_x + (i + 1) * spacing, center_y + (i + 1) * spacing)

    def _apply_circle_formation(self) -> None:
        """
        Arrange ships in a circular formation.
        Flagship (or first ship) is in the center, other ships form a circle around it.
        """
        if not self.ships:
            return

        # Get the center position (flagship or first ship)
        center_ship = self.flagship or self.ships[0]
        center_x, center_y = center_ship.position

        # Radius of the circle
        radius = 3

        # Place flagship in the center, other ships in a circle
        ships_to_position = [ship for ship in self.ships if ship != center_ship]

        # If there are no other ships, we're done
        if not ships_to_position:
            return

        # Position other ships in a circle around the flagship
        angle_step = 2 * math.pi / len(ships_to_position)
        for i, ship in enumerate(ships_to_position):
            angle = i * angle_step
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            ship.position = (x, y)

    def _apply_scatter_formation(self) -> None:
        """
        Arrange ships in a random scatter formation within a defined area.
        Flagship (or first ship) is in the center, other ships are randomly positioned.
        """
        if not self.ships:
            return

        # Get the center position (flagship or first ship)
        center_ship = self.flagship or self.ships[0]
        center_x, center_y = center_ship.position

        # Maximum distance from center
        max_distance = 5

        # Place flagship in the center, other ships scattered randomly
        ships_to_position = [ship for ship in self.ships if ship != center_ship]

        # Position other ships randomly around the flagship
        for ship in ships_to_position:
            # Random offset from center
            dx = random.randint(-max_distance, max_distance)
            dy = random.randint(-max_distance, max_distance)

            # Ensure minimum distance from center (avoid stacking)
            if abs(dx) < 1 and abs(dy) < 1:
                dx = 1 if random.random() < 0.5 else -1
                dy = 1 if random.random() < 0.5 else -1

            ship.position = (center_x + dx, center_y + dy)

    def set_formation(self, formation: str) -> bool:
        """
        Change the fleet formation.

        Args:
            formation: The new formation type

        Returns:
            bool: True if the formation was changed successfully, False otherwise
        """
        if formation not in FLEET_FORMATIONS:
            logging.warning(f"Invalid formation type: {formation}")
            return False

        self.formation = formation
        self._update_ship_positions()

        logging.info(f"Fleet {self.fleet_id} changed formation to {formation}")
        return True

    def _calculate_resource_status_multiplier(self) -> float:
        """Calculate a multiplier based on the fleet's resource status.

        Returns:
            Float multiplier (0.5-1.0) based on critical resource levels
        """
        if not self.ships:
            return 1.0

        # Check fuel cells and ship parts levels
        fuel_status = min(1.0, self.resources["fuel_cells"] / (len(self.ships) * 10))
        parts_status = min(1.0, self.resources["ship_parts"] / (len(self.ships) * 5))

        # Weight fuel more heavily than parts
        weighted_status = (fuel_status * 0.7) + (parts_status * 0.3)

        # Scale from 0.5 to 1.0 (even with no resources, fleet still has 50% effectiveness)
        return max(0.5, weighted_status)

    def _consume_resources(self, delta_time: float) -> None:
        """Consume fleet resources based on activity level and time passed.

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        if not self.ships:
            return

        # Convert delta_time to days (assuming 1 day = 86400 seconds)
        days_passed = delta_time / 86400.0

        # Base consumption multiplier
        consumption_mult = 1.0

        # Increase consumption during combat
        if self.in_combat:
            consumption_mult *= 2.0

        # Increase consumption during movement
        if self.path or self.destination:
            consumption_mult *= 1.5

        # Calculate consumption for each resource type
        for resource_type, base_rate in self.resource_consumption_rate.items():
            # Consumption depends on fleet size
            total_consumption = (
                base_rate * len(self.ships) * days_passed * consumption_mult
            )

            # Remove resources (with a minimum of 0)
            self.resources[resource_type] = max(
                0, self.resources[resource_type] - total_consumption
            )

        # Update morale based on resource levels
        self._update_morale_from_resources()

    def _update_morale_from_resources(self) -> None:
        """Update fleet morale based on resource levels."""
        if not self.ships:
            return

        # Check critical resources (fuel and parts)
        fuel_ratio = self.resources["fuel_cells"] / (len(self.ships) * 10)
        parts_ratio = self.resources["ship_parts"] / (len(self.ships) * 5)

        # If resources are critically low, reduce morale
        if fuel_ratio < 0.2 or parts_ratio < 0.2:
            # Reduce morale by up to 30% based on resource scarcity
            morale_penalty = 0.3 * (1 - min(fuel_ratio, parts_ratio) / 0.2)
            self.morale = max(0.5, self.morale - morale_penalty)
        else:
            # Gradually restore morale if resources are sufficient
            self.morale = min(1.0, self.morale + 0.01)

    def _distribute_resources(self) -> None:
        """Distribute resources among ships based on the selected distribution method."""
        if not self.ships:
            return

        # Choose distribution method based on setting
        if self.resource_distribution_method == "equal":
            self._distribute_resources_equally()
        elif self.resource_distribution_method == "proportional":
            self._distribute_resources_proportionally()
        else:  # Default to priority-based
            self._distribute_resources_by_priority()

    def _distribute_resources_equally(self) -> None:
        """Distribute resources equally among all ships."""
        # This is a placeholder for the actual ship resource distribution
        # In a real implementation, each ship would have its own resource storage
        logging.debug(
            f"Fleet {self.fleet_id} distributing resources equally among {len(self.ships)} ships"
        )

    def _distribute_resources_proportionally(self) -> None:
        """Distribute resources proportionally based on ship size/capacity."""
        # This is a placeholder for the actual ship resource distribution
        # In a real implementation, each ship would have its own resource storage
        logging.debug(
            f"Fleet {self.fleet_id} distributing resources proportionally among {len(self.ships)} ships"
        )

    def _distribute_resources_by_priority(self) -> None:
        """Distribute resources based on ship priority (flagship first, then by role)."""
        # This is a placeholder for the actual ship resource distribution
        # In a real implementation, each ship would have its own resource storage

        # Ensure flagship gets resources first
        if self.flagship:
            logging.debug(
                f"Fleet {self.fleet_id} prioritizing resources for flagship {self.flagship.ship_id}"
            )

        # Then distribute by ship role priority
        role_priority = {
            "capital": 0,
            "carrier": 1,
            "battleship": 2,
            "cruiser": 3,
            "destroyer": 4,
            "frigate": 5,
            "corvette": 6,
            "fighter": 7,
            "support": 8,
        }

        # Sort ships by role priority
        self.ships = sorted(
            self.ships,
            key=lambda s: role_priority.get(getattr(s, "role", "support"), 9),
        )

        logging.debug(
            f"Fleet {self.fleet_id} distributing resources by priority among {len(self.ships)} ships"
        )

    def add_resources(self, resource_type: str, amount: float) -> bool:
        """Add resources to the fleet.

        Args:
            resource_type: Type of resource to add
            amount: Amount to add

        Returns:
            bool: True if resources were added successfully
        """
        if resource_type not in self.resources:
            logging.warning(
                f"Cannot add unknown resource type {resource_type} to fleet {self.fleet_id}"
            )
            return False

        if amount <= 0:
            logging.warning(
                f"Cannot add non-positive amount {amount} of {resource_type} to fleet {self.fleet_id}"
            )
            return False

        # Check capacity
        available_capacity = (
            self.resource_capacity[resource_type] - self.resources[resource_type]
        )
        if amount > available_capacity:
            # Only add what we can fit
            amount = available_capacity
            logging.info(
                f"Fleet {self.fleet_id} can only add {amount} of {resource_type} due to capacity limits"
            )

        # Add resources
        self.resources[resource_type] += amount
        logging.info(f"Added {amount} of {resource_type} to fleet {self.fleet_id}")

        return True

    def remove_resources(self, resource_type: str, amount: float) -> bool:
        """Remove resources from the fleet.

        Args:
            resource_type: Type of resource to remove
            amount: Amount to remove

        Returns:
            bool: True if resources were removed successfully
        """
        if resource_type not in self.resources:
            logging.warning(
                f"Cannot remove unknown resource type {resource_type} from fleet {self.fleet_id}"
            )
            return False

        if amount <= 0:
            logging.warning(
                f"Cannot remove non-positive amount {amount} of {resource_type} from fleet {self.fleet_id}"
            )
            return False

        # Check if we have enough
        if amount > self.resources[resource_type]:
            logging.warning(
                f"Cannot remove {amount} of {resource_type} from fleet {self.fleet_id}, only {self.resources[resource_type]} available"
            )
            return False

        # Remove resources
        self.resources[resource_type] -= amount
        logging.info(f"Removed {amount} of {resource_type} from fleet {self.fleet_id}")

        return True

    def transfer_resources(
        self, target_fleet: "Fleet", resource_type: str, amount: float
    ) -> bool:
        """Transfer resources to another fleet.

        Args:
            target_fleet: Fleet to transfer resources to
            resource_type: Type of resource to transfer
            amount: Amount to transfer

        Returns:
            bool: True if resources were transferred successfully
        """
        # Check if fleets are close enough
        if not self._is_fleet_in_range(target_fleet, transfer_range=2.0):
            logging.warning(
                f"Cannot transfer resources to fleet {target_fleet.fleet_id}, not in range"
            )
            return False

        # Try to remove resources from this fleet
        if not self.remove_resources(resource_type, amount):
            return False

        # Try to add resources to target fleet
        added_amount = min(
            amount,
            target_fleet.resource_capacity[resource_type]
            - target_fleet.resources[resource_type],
        )
        target_fleet.add_resources(resource_type, added_amount)

        # If target fleet couldn't accept all resources, return the excess
        if added_amount < amount:
            self.add_resources(resource_type, amount - added_amount)
            logging.info(
                f"Returned {amount - added_amount} of {resource_type} to fleet {self.fleet_id} due to capacity limits"
            )

        logging.info(
            f"Transferred {added_amount} of {resource_type} from fleet {self.fleet_id} to fleet {target_fleet.fleet_id}"
        )
        return True

    def set_resource_distribution_method(self, method: str) -> bool:
        """Set the resource distribution method.

        Args:
            method: One of 'priority', 'equal', or 'proportional'

        Returns:
            bool: True if the method was set successfully
        """
        valid_methods = ["priority", "equal", "proportional"]
        if method not in valid_methods:
            logging.warning(
                f"Invalid resource distribution method {method} for fleet {self.fleet_id}"
            )
            return False

        self.resource_distribution_method = method
        logging.info(
            f"Fleet {self.fleet_id} changed resource distribution method to {method}"
        )
        return True

    def get_resource_status(self) -> Dict[str, Dict[str, float]]:
        """Get the current resource status of the fleet.

        Returns:
            Dict containing current resource levels, capacities, and percentages
        """
        status = {
            "levels": self.resources.copy(),
            "capacities": self.resource_capacity.copy(),
            "percentages": {},
        }

        # Calculate percentages
        for resource_type in self.resources:
            if self.resource_capacity[resource_type] > 0:
                status["percentages"][resource_type] = (
                    self.resources[resource_type]
                    / self.resource_capacity[resource_type]
                )
            else:
                status["percentages"][resource_type] = 0.0

        return status

    def _is_fleet_in_range(
        self, other_fleet: "Fleet", transfer_range: float = 1.0
    ) -> bool:
        """Check if another fleet is within transfer range.

        Args:
            other_fleet: The fleet to check
            transfer_range: Maximum distance for transfer

        Returns:
            bool: True if the fleet is in range
        """
        if not self.position or not other_fleet.position:
            return False

        # Calculate distance between fleets
        dx = self.position[0] - other_fleet.position[0]
        dy = self.position[1] - other_fleet.position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        return distance <= transfer_range

    def get_fleet_strength(self) -> float:
        """
        Calculate the overall combat strength of the fleet.

        Returns:
            float: The fleet's combat strength value
        """
        if not self.ships:
            return 0.0

        # Base strength is the sum of all ships' attack power
        base_strength = sum(ship.attack_power for ship in self.ships)

        # Apply formation multipliers
        formation_multipliers = {
            "line": 1.0,  # Balanced
            "column": 0.9,  # Weaker but faster
            "wedge": 1.2,  # Strong offensive formation
            "echelon": 1.1,  # Good for flanking
            "circle": 1.3,  # Strong defensive formation
            "scatter": 0.8,  # Weak but unpredictable
        }

        formation_mult = formation_multipliers.get(self.formation, 1.0)

        # Apply commander level bonus (5% per level)
        commander_mult = 1.0 + (self.commander_level - 1) * 0.05

        # Apply morale modifier
        morale_mult = self.morale

        # Apply resource status modifier
        resource_mult = self._calculate_resource_status_multiplier()

        return (
            base_strength
            * formation_mult
            * commander_mult
            * morale_mult
            * resource_mult
        )

    def move_to(
        self,
        destination: Tuple[int, int],
        speed_multiplier: float = 1.0,
        use_pathfinding: bool = False,
        obstacle_check: Optional[Callable[[Tuple[int, int]], bool]] = None,
    ) -> bool:
        """
        Set the fleet to move toward a destination.

        Args:
            destination: The target position as (x, y) tuple
            speed_multiplier: Optional speed multiplier for this movement
            use_pathfinding: Whether to use pathfinding to navigate around obstacles
            obstacle_check: Function that returns True if a position contains an obstacle

        Returns:
            bool: True if the movement was initiated successfully, False otherwise
        """
        if not self.ships:
            logging.warning(f"Cannot move fleet {self.fleet_id}: no ships")
            return False

        # Clear existing path
        self.path = []

        # If using pathfinding and obstacle check is provided
        if use_pathfinding and obstacle_check:
            # Find path to destination
            path_found = self.find_path_to(destination, obstacle_check)

            if not path_found:
                logging.warning(
                    f"Fleet {self.fleet_id} could not find path to {destination}"
                )
                return False

            # Path is now set, no need to set destination directly
            logging.info(
                f"Fleet {self.fleet_id} found path to {destination} with {len(self.path)} waypoints"
            )
        else:
            # Direct movement without pathfinding
            # Set the destination
            self.destination = destination

        # Clear any existing waypoints
        self.waypoints = []

        # Set the current orders
        self.current_orders = {
            "type": "move_to",
            "destination": destination,
            "speed_multiplier": speed_multiplier,
        }

        logging.info(f"Fleet {self.fleet_id} moving to {destination}")
        return True

    def patrol_between(self, points: List[Tuple[int, int]], loop: bool = True) -> bool:
        """
        Set the fleet to patrol between a series of points.

        Args:
            points: List of positions to patrol between
            loop: Whether to loop back to the first point after reaching the last

        Returns:
            bool: True if the patrol was initiated successfully, False otherwise
        """
        if not self.ships:
            logging.warning(f"Cannot set patrol for fleet {self.fleet_id}: no ships")
            return False

        if len(points) < 2:
            logging.warning(
                f"Cannot set patrol for fleet {self.fleet_id}: need at least 2 points"
            )
            return False

        # Set the waypoints
        self.waypoints = points.copy()

        # Set the destination to the first waypoint
        self.destination = self.waypoints[0]

        # Set the current orders
        self.current_orders = {
            "type": "patrol",
            "waypoints": points,
            "current_waypoint_index": 0,
            "loop": loop,
        }

        logging.info(f"Fleet {self.fleet_id} patrolling between {len(points)} points")
        return True

    def escort(self, target_entity: Any, distance: int = 3) -> bool:
        """
        Set the fleet to escort a target entity.

        Args:
            target_entity: The entity to escort (must have a position attribute)
            distance: The distance to maintain from the target

        Returns:
            bool: True if the escort was initiated successfully, False otherwise
        """
        if not self.ships:
            logging.warning(f"Cannot set escort for fleet {self.fleet_id}: no ships")
            return False

        if not hasattr(target_entity, "position"):
            logging.warning(
                f"Cannot set escort for fleet {self.fleet_id}: target has no position"
            )
            return False

        # Set the current orders
        self.current_orders = {
            "type": "escort",
            "target": target_entity,
            "distance": distance,
        }

        logging.info(f"Fleet {self.fleet_id} escorting target at distance {distance}")
        return True

    def update(self, delta_time: float) -> None:
        """
        Update the fleet state based on current orders.

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        if not self.ships or not self.is_active:
            return

        # If we have a path, follow it
        if self.path and len(self.path) > 1:
            self.move_along_path(delta_time)
        # Otherwise, follow current orders
        elif self.current_orders:
            order_type = self.current_orders.get("type")

            if order_type == "move_to":
                self._update_move_to(delta_time)
            elif order_type == "patrol":
                self._update_patrol(delta_time)
            elif order_type == "escort":
                self._update_escort(delta_time)

        # Update ship positions based on formation after fleet movement
        self._update_ship_positions()

        # Handle combat if in combat
        if self.in_combat and self.target_fleet:
            self._handle_combat(delta_time)
        # Check for enemies if auto-engage is enabled and not in combat
        elif hasattr(self, "auto_engage") and self.auto_engage:
            # In a real implementation, we would pass the actual list of nearby fleets
            # For now, we'll pass None and let the method handle it
            self._check_for_enemies(None)

        # Consume resources based on fleet activity
        self._consume_resources(delta_time)

        # Distribute resources among ships
        self._distribute_resources()

    def _update_move_to(self, delta_time: float) -> None:
        """
        Update fleet movement toward a destination.

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        if not self.destination:
            return

        # Get current position and destination
        current_x, current_y = self.position
        dest_x, dest_y = self.destination

        # Calculate direction vector
        dx = dest_x - current_x
        dy = dest_y - current_y

        # Calculate distance to destination
        distance = math.sqrt(dx * dx + dy * dy)

        # If we've reached the destination (within 0.5 units), stop
        if distance < 0.5:
            self.current_orders = None
            self.destination = None
            logging.info(f"Fleet {self.fleet_id} reached destination")
            return

        # Normalize direction vector
        if distance > 0:
            dx /= distance
            dy /= distance

        # Calculate movement speed
        speed = self.speed * self.current_orders.get("speed_multiplier", 1.0)

        # Apply formation-specific speed modifiers
        formation_speed_modifiers = {
            "line": 1.0,  # Balanced
            "column": 1.2,  # Faster
            "wedge": 0.9,  # Slower but stronger
            "echelon": 1.1,  # Good for flanking
            "circle": 0.8,  # Slow but defensive
            "scatter": 1.1,  # Slightly faster
        }

        speed *= formation_speed_modifiers.get(self.formation, 1.0)

        # Calculate movement distance this update
        move_distance = speed * delta_time

        # Limit movement to not overshoot the destination
        move_distance = min(move_distance, distance)

        # Calculate new position
        new_x = current_x + dx * move_distance
        new_y = current_y + dy * move_distance

        # Update fleet position
        self.position = (new_x, new_y)

    def find_path_to(
        self,
        destination: Tuple[int, int],
        obstacle_check: Callable[[Tuple[int, int]], bool],
    ) -> bool:
        """
        Find a path to the destination using A* pathfinding.

        Args:
            destination: The target position as (x, y) tuple
            obstacle_check: Function that returns True if a position contains an obstacle

        Returns:
            bool: True if a path was found, False otherwise
        """
        # Reset current path
        self.path = []

        # If destination is the same as current position, we're already there
        if self.position == destination:
            return True

        # Initialize A* algorithm data structures
        start = self.position
        goal = destination
        open_set, came_from, g_score, f_score, closed_set = (
            self._initialize_pathfinding(start, goal)
        )

        # Main pathfinding loop
        return self._execute_pathfinding_loop(
            open_set, closed_set, came_from, g_score, f_score, goal, obstacle_check
        )

    def _initialize_pathfinding(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize data structures for A* pathfinding.

        Args:
            start: Starting position
            goal: Goal position

        Returns:
            Tuple containing open_set, came_from, g_score, f_score, and closed_set
        """
        # Priority queue for open set
        open_set = []
        heapq.heappush(open_set, (0, start))  # (f_score, position)

        # For node tracking
        came_from = {}

        # g_score: cost from start to current node
        g_score = {start: 0}

        # f_score: estimated cost from start to goal through current node
        f_score = {start: self._heuristic(start, goal)}

        # Set of visited nodes
        closed_set: Set[Tuple[int, int]] = set()

        return open_set, came_from, g_score, f_score, closed_set

    def _execute_pathfinding_loop(
        self,
        open_set,
        closed_set,
        came_from,
        g_score,
        f_score,
        goal: Tuple[int, int],
        obstacle_check: Callable[[Tuple[int, int]], bool],
    ) -> bool:
        """
        Execute the main A* pathfinding loop.

        Args:
            open_set: Priority queue of nodes to evaluate
            closed_set: Set of already evaluated nodes
            came_from: Dictionary mapping each node to its predecessor
            g_score: Dictionary of costs from start to each node
            f_score: Dictionary of estimated total costs
            goal: Target position
            obstacle_check: Function to check if a position has an obstacle

        Returns:
            bool: True if path found, False otherwise
        """
        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)

            # If we reached the goal, reconstruct and return the path
            if current == goal:
                self.path = self._reconstruct_path(came_from, current)
                return True

            # Add current to closed set
            closed_set.add(current)

            # Process all neighbors
            if not self._process_neighbors(
                current,
                goal,
                open_set,
                closed_set,
                came_from,
                g_score,
                f_score,
                obstacle_check,
            ):
                continue

        # No path found
        logging.warning(f"Fleet {self.fleet_id} could not find path to {goal}")
        return False

    def _process_neighbors(
        self,
        current: Tuple[int, int],
        goal: Tuple[int, int],
        open_set,
        closed_set,
        came_from,
        g_score,
        f_score,
        obstacle_check: Callable[[Tuple[int, int]], bool],
    ) -> bool:
        """
        Process all neighbors of the current node in pathfinding.

        Args:
            current: Current position being evaluated
            goal: Target position
            open_set: Priority queue of nodes to evaluate
            closed_set: Set of already evaluated nodes
            came_from: Dictionary mapping each node to its predecessor
            g_score: Dictionary of costs from start to each node
            f_score: Dictionary of estimated total costs
            obstacle_check: Function to check if a position has an obstacle

        Returns:
            bool: True if processing should continue, False otherwise
        """
        for neighbor in self._get_neighbors(current):
            # Skip if neighbor is an obstacle or already evaluated
            if neighbor in closed_set or obstacle_check(neighbor):
                continue

            # Calculate tentative g_score
            tentative_g_score = g_score.get(current, float("inf")) + self._distance(
                current, neighbor
            )

            # If this path is better than any previous one
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                self._update_path_data(
                    neighbor,
                    current,
                    tentative_g_score,
                    goal,
                    came_from,
                    g_score,
                    f_score,
                    open_set,
                )

        return True

    def _update_path_data(
        self,
        neighbor: Tuple[int, int],
        current: Tuple[int, int],
        tentative_g_score: float,
        goal: Tuple[int, int],
        came_from,
        g_score,
        f_score,
        open_set,
    ) -> None:
        """
        Update pathfinding data for a better path.

        Args:
            neighbor: Neighbor node being evaluated
            current: Current node
            tentative_g_score: New g_score for neighbor
            goal: Target position
            came_from: Dictionary mapping each node to its predecessor
            g_score: Dictionary of costs from start to each node
            f_score: Dictionary of estimated total costs
            open_set: Priority queue of nodes to evaluate
        """
        # Record this path
        came_from[neighbor] = current
        g_score[neighbor] = tentative_g_score
        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)

        # Add to open set if not already there
        if neighbor not in [pos for _, pos in open_set]:
            heapq.heappush(open_set, (f_score[neighbor], neighbor))

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate heuristic (estimated distance) between two points.

        Args:
            a: First position
            b: Second position

        Returns:
            float: Estimated distance
        """
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate actual distance between two points.

        Args:
            a: First position
            b: Second position

        Returns:
            float: Actual distance
        """
        # Euclidean distance
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get all valid neighboring positions.

        Args:
            position: Current position

        Returns:
            List[Tuple[int, int]]: List of valid neighboring positions
        """
        x, y = position
        neighbors = []

        # 8-directional movement
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Skip the current position
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # Check if within map bounds
                if 0 <= nx < GAME_MAP_SIZE[0] and 0 <= ny < GAME_MAP_SIZE[1]:
                    neighbors.append((nx, ny))

        return neighbors

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct path from came_from dictionary.

        Args:
            came_from: Dictionary mapping each node to its predecessor
            current: Current node (goal)

        Returns:
            List[Tuple[int, int]]: Path from start to goal
        """
        total_path = [current]

        while current in came_from:
            current = came_from[current]
            total_path.append(current)

        # Reverse to get path from start to goal
        total_path.reverse()

        return total_path

    def move_along_path(self, delta_time: float) -> None:
        """
        Move the fleet along the calculated path.

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        if not self.path or len(self.path) <= 1:
            return

        # Get next waypoint from path
        next_waypoint = self.path[1]  # Index 0 is current position

        # Set as destination and move toward it
        self.destination = next_waypoint
        self._update_move_to(delta_time)

        # If we've reached the waypoint, remove it from the path
        if not self.destination:
            self.path.pop(0)

            # If we've reached the end of the path
            if len(self.path) <= 1:
                logging.info(f"Fleet {self.fleet_id} reached end of path")
                self.path = []
            else:
                # Set next waypoint as destination
                self.destination = self.path[1]

    def _update_patrol(self, delta_time: float) -> None:
        """
        Update fleet movement for patrol orders.

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        if not self.waypoints:
            return

        # Get current waypoint index
        current_index = self.current_orders.get("current_waypoint_index", 0)

        # Set destination to current waypoint if not already set
        if not self.destination:
            self.destination = self.waypoints[current_index]

        # Move toward the current waypoint
        self._update_move_to(delta_time)

        # If we've reached the waypoint (destination is None after _update_move_to)
        if not self.destination:
            # Move to the next waypoint
            current_index += 1

            # Check if we've reached the end of the waypoints
            if current_index >= len(self.waypoints):
                # If looping, go back to the first waypoint
                if self.current_orders.get("loop", True):
                    current_index = 0
                else:
                    # If not looping, stop patrol
                    self.current_orders = None
                    logging.info(f"Fleet {self.fleet_id} completed patrol")
                    return

            # Update current waypoint index and set new destination
            self.current_orders["current_waypoint_index"] = current_index
            self.destination = self.waypoints[current_index]

    def _update_escort(self, delta_time: float) -> None:
        """
        Update fleet movement for escort orders.

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        target = self.current_orders.get("target")

        # Check if target still exists and has a position
        if not target or not hasattr(target, "position"):
            self.current_orders = None
            logging.info(f"Fleet {self.fleet_id} escort target lost")
            return

        # Get target position
        target_x, target_y = target.position

        # Get current position
        current_x, current_y = self.position

        # Calculate direction vector to target
        dx = target_x - current_x
        dy = target_y - current_y

        # Calculate distance to target
        distance = math.sqrt(dx * dx + dy * dy)

        # Get desired escort distance
        escort_distance = self.current_orders.get("distance", 3)

        # If we're at the right distance (within 0.5 units), just update position to maintain formation
        if abs(distance - escort_distance) < 0.5:
            # No need to move, just maintain formation
            return

        # Normalize direction vector
        if distance > 0:
            dx /= distance
            dy /= distance

        # Calculate ideal position based on escort distance
        if distance > escort_distance:
            # Move closer to target
            ideal_distance = distance - escort_distance
        else:
            # Move away from target
            ideal_distance = escort_distance - distance
            dx = -dx
            dy = -dy

        # Calculate movement speed
        speed = self.speed * 1.2  # Escorts move slightly faster to keep up

        # Calculate movement distance this update
        move_distance = speed * delta_time

        # Limit movement to not overshoot the ideal position
        move_distance = min(move_distance, ideal_distance)

        # Calculate new position
        new_x = current_x + dx * move_distance
        new_y = current_y + dy * move_distance

        # Update fleet position
        self.position = (new_x, new_y)

    def engage_fleet(self, target_fleet: "Fleet", stance: str = "balanced") -> bool:
        """
        Engage another fleet in combat.

        Args:
            target_fleet: The fleet to engage
            stance: Combat stance (balanced, aggressive, defensive, evasive)

        Returns:
            bool: True if engagement was initiated successfully, False otherwise
        """
        if not self.ships:
            logging.warning(f"Cannot engage with fleet {self.fleet_id}: no ships")
            return False

        if not target_fleet or not target_fleet.ships:
            logging.warning(
                f"Cannot engage with fleet {self.fleet_id}: invalid target fleet"
            )
            return False

        # Check if target is within range
        distance = self._distance(self.position, target_fleet.position)
        if distance > self.detection_range:
            logging.warning(
                f"Cannot engage with fleet {self.fleet_id}: target out of range"
            )
            return False

        # Set combat state
        self.in_combat = True
        self.target_fleet = target_fleet
        self.combat_stance = (
            stance
            if stance in {"balanced", "aggressive", "defensive", "evasive"}
            else "balanced"
        )

        # Clear any movement orders
        self.current_orders = {
            "type": "combat",
            "target": target_fleet,
            "stance": self.combat_stance,
        }

        logging.info(
            f"Fleet {self.fleet_id} engaging fleet {target_fleet.fleet_id} with stance {stance}"
        )
        return True

    def disengage(self) -> bool:
        """
        Disengage from current combat.

        Returns:
            bool: True if disengagement was successful, False otherwise
        """
        if not self.in_combat:
            return False

        self.in_combat = False
        self.target_fleet = None
        self.current_orders = None

        logging.info(f"Fleet {self.fleet_id} disengaged from combat")
        return True

    def _handle_combat(self, delta_time: float) -> None:
        """
        Handle combat between fleets.

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        if not self.in_combat or not self.target_fleet or not self.target_fleet.ships:
            self.in_combat = False
            self.target_fleet = None
            return

        # Get our combat stance
        our_stance = self.combat_stance
        # Note: target stance is retrieved in _get_target_stance when needed

        # Calculate distance to target fleet
        distance = self._distance(self.position, self.target_fleet.position)

        # Determine ideal combat distance based on stance
        ideal_distance = self._get_ideal_combat_distance(our_stance)

        # Move toward or away from target to maintain ideal distance
        if abs(distance - ideal_distance) > 0.5:
            self._adjust_position_for_combat(distance, ideal_distance, delta_time)
        # Determine if we can attack based on distance
        attack_range = 7.0  # Base attack range

        if distance <= attack_range:
            self._execute_attack(delta_time)

    def _get_ideal_combat_distance(self, stance: str) -> float:
        """
        Determine the ideal combat distance based on the fleet's stance.

        Args:
            stance: The combat stance of the fleet

        Returns:
            float: The ideal distance to maintain during combat
        """
        ideal_distances = {
            "balanced": 5.0,  # Medium range
            "aggressive": 2.0,  # Close range
            "defensive": 8.0,  # Long range
            "evasive": 10.0,  # Very long range
        }

        return ideal_distances.get(stance, 5.0)

    def _adjust_position_for_combat(self, distance, ideal_distance, delta_time):
        # Calculate direction vector
        dx = self.target_fleet.position[0] - self.position[0]
        dy = self.target_fleet.position[1] - self.position[1]

        # Normalize
        if distance > 0:
            dx /= distance
            dy /= distance

        # Determine if we need to move closer or further
        if distance > ideal_distance:
            # Move closer
            move_distance = min(self.speed * delta_time, distance - ideal_distance)
        else:
            # Move away
            move_distance = min(self.speed * delta_time, ideal_distance - distance)
            dx = -dx
            dy = -dy

        # Update position
        new_x = self.position[0] + dx * move_distance
        new_y = self.position[1] + dy * move_distance

        # Ensure we stay within map bounds
        new_x = max(0, min(GAME_MAP_SIZE[0] - 1, new_x))
        new_y = max(0, min(GAME_MAP_SIZE[1] - 1, new_y))

        self.position = (new_x, new_y)

    def _execute_attack(self, delta_time: float) -> None:
        """
        Execute an attack against the target fleet.

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        if not self.target_fleet or not self.target_fleet.ships:
            return

        # Get target's stance
        target_stance = self._get_target_stance()

        # Calculate damage based on stances and fleet strength
        final_damage = self._calculate_combat_damage(delta_time, target_stance)

        # Apply damage to target fleet's ships
        self._apply_damage_to_fleet(self.target_fleet, final_damage)

        logging.info(
            f"Fleet {self.fleet_id} dealt {final_damage:.2f} damage to fleet {self.target_fleet.fleet_id}"
        )

    def _get_target_stance(self) -> str:
        """
        Get the combat stance of the target fleet.

        Returns:
            str: The target fleet's combat stance
        """
        return (
            self.target_fleet.combat_stance
            if hasattr(self.target_fleet, "combat_stance")
            else "balanced"
        )

    def _calculate_combat_damage(self, delta_time: float, target_stance: str) -> float:
        """
        Calculate the damage to be dealt to the target fleet.

        Args:
            delta_time: Time elapsed since last update in seconds
            target_stance: The combat stance of the target fleet

        Returns:
            float: The final damage amount
        """
        # Calculate offensive multiplier based on our stance
        stance_damage_multipliers = {
            "balanced": 1.0,  # Balanced damage
            "aggressive": 1.5,  # High damage
            "defensive": 0.7,  # Low damage
            "evasive": 0.4,  # Very low damage
        }
        damage_mult = stance_damage_multipliers.get(self.combat_stance, 1.0)

        # Calculate base damage from our fleet strength
        base_damage = self.get_fleet_strength() * damage_mult * delta_time

        # Apply target's defensive modifiers based on their stance
        stance_defense_multipliers = {
            "balanced": 1.0,  # Balanced defense
            "aggressive": 0.7,  # Low defense
            "defensive": 1.5,  # High defense
            "evasive": 1.3,  # Good defense
        }
        defense_mult = stance_defense_multipliers.get(target_stance, 1.0)

        # Calculate final damage
        return base_damage / defense_mult

    def _apply_damage_to_fleet(self, target_fleet: "Fleet", damage: float) -> None:
        """
        Apply damage to ships in the target fleet.

        Args:
            target_fleet: The fleet to apply damage to
            damage: Amount of damage to apply
        """
        if not target_fleet or not target_fleet.ships:
            return

        # Get the appropriate damage distribution function based on formation
        distribute_func = self._get_damage_distribution_function(target_fleet.formation)

        # Apply the damage distribution
        distribute_func(target_fleet, damage)

        # Handle destroyed ships
        self._handle_destroyed_ships(target_fleet)

    def _get_damage_distribution_function(
        self, formation: str
    ) -> Callable[["Fleet", float], None]:
        """
        Get the appropriate damage distribution function based on formation.

        Args:
            formation: The formation type

        Returns:
            Callable: The damage distribution function
        """
        formation_damage_distribution = {
            "line": self._distribute_damage_evenly,
            "column": self._distribute_damage_front_heavy,
            "wedge": self._distribute_damage_point_heavy,
            "echelon": self._distribute_damage_flank_heavy,
            "circle": self._distribute_damage_flagship_protected,
            "scatter": self._distribute_damage_randomly,
        }

        return formation_damage_distribution.get(
            formation, self._distribute_damage_evenly
        )

    def _handle_destroyed_ships(self, target_fleet: "Fleet") -> None:
        """
        Remove destroyed ships and check if the fleet is empty.

        Args:
            target_fleet: The fleet to check for destroyed ships
        """
        # Check for destroyed ships and remove them
        destroyed_ships = [ship for ship in target_fleet.ships if ship.health <= 0]
        for ship in destroyed_ships:
            target_fleet.remove_ship(ship)
            logging.info(f"Ship destroyed in fleet {target_fleet.fleet_id}")

        # Check if the fleet is now empty
        if not target_fleet.ships:
            logging.info(f"Fleet {target_fleet.fleet_id} has been destroyed")
            target_fleet.is_active = False

    def _distribute_damage_evenly(self, target_fleet: "Fleet", damage: float) -> None:
        """
        Distribute damage evenly among all ships.

        Args:
            target_fleet: The fleet to apply damage to
            damage: Amount of damage to apply
        """
        if not target_fleet.ships:
            return

        damage_per_ship = damage / len(target_fleet.ships)

        for ship in target_fleet.ships:
            ship.health -= damage_per_ship

    def _distribute_damage_front_heavy(
        self, target_fleet: "Fleet", damage: float
    ) -> None:
        """
        Distribute damage with more damage to front ships (column formation).

        Args:
            target_fleet: The fleet to apply damage to
            damage: Amount of damage to apply
        """
        if not target_fleet.ships:
            return

        # In column formation, first ship takes 40% of damage, rest distributed among others
        front_ship_damage = damage * 0.4
        target_fleet.ships[0].health -= front_ship_damage

        # Distribute remaining damage
        if len(target_fleet.ships) > 1:
            remaining_damage = damage * 0.6

            damage_per_remaining_ship = remaining_damage / (len(target_fleet.ships) - 1)
            for ship in target_fleet.ships[1:]:
                ship.health -= damage_per_remaining_ship

    def _distribute_damage_point_heavy(
        self, target_fleet: "Fleet", damage: float
    ) -> None:
        """
        Distribute damage with more damage to the point ship (wedge formation).

        Args:
            target_fleet: The fleet to apply damage to
            damage: Amount of damage to apply
        """
        if not target_fleet.ships:
            return

        # In wedge formation, point ship (flagship or first) takes 50% of damage
        point_ship = target_fleet.flagship or target_fleet.ships[0]
        point_ship_damage = damage * 0.5
        # Apply damage to point ship
        point_ship.health -= point_ship_damage

        if other_ships := [ship for ship in target_fleet.ships if ship != point_ship]:
            remaining_damage = damage * 0.5

            damage_per_remaining_ship = remaining_damage / len(other_ships)
            for ship in other_ships:
                ship.health -= damage_per_remaining_ship

    def _distribute_damage_flank_heavy(
        self, target_fleet: "Fleet", damage: float
    ) -> None:
        """
        Distribute damage with more damage to flank ships (echelon formation).

        Args:
            target_fleet: The fleet to apply damage to
            damage: Amount of damage to apply
        """
        if not target_fleet.ships:
            return

        # In echelon formation, last ship takes 40% of damage (exposed flank)
        if len(target_fleet.ships) >= 2:
            self._apply_damage_to_flank_and_others(target_fleet, damage)
        else:
            # If only one ship, it takes all damage
            target_fleet.ships[0].health -= damage

    def _apply_damage_to_flank_and_others(self, target_fleet, damage):
        flank_ship = target_fleet.ships[-1]
        flank_ship_damage = damage * 0.4
        remaining_damage = damage * 0.6

        # Apply damage to flank ship
        flank_ship.health -= flank_ship_damage

        # Distribute remaining damage
        other_ships = target_fleet.ships[:-1]
        damage_per_remaining_ship = remaining_damage / len(other_ships)
        for ship in other_ships:
            ship.health -= damage_per_remaining_ship

    def _distribute_damage_flagship_protected(
        self, target_fleet: "Fleet", damage: float
    ) -> None:
        """
        Distribute damage with flagship protected (circle formation).

        Args:
            target_fleet: The fleet to apply damage to
            damage: Amount of damage to apply
        """
        if not target_fleet.ships:
            return

        # In circle formation, flagship takes only 10% of damage, others take more
        if target_fleet.flagship and len(target_fleet.ships) > 1:
            self._apply_damage_with_flagship_protection(damage, target_fleet)
        else:
            # If no flagship or only one ship, distribute evenly
            self._distribute_damage_evenly(target_fleet, damage)

    def _apply_damage_with_flagship_protection(self, damage, target_fleet):
        flagship_damage = damage * 0.1
        remaining_damage = damage * 0.9

        # Apply damage to flagship
        target_fleet.flagship.health -= flagship_damage

        # Distribute remaining damage to other ships
        other_ships = [
            ship for ship in target_fleet.ships if ship != target_fleet.flagship
        ]
        damage_per_remaining_ship = remaining_damage / len(other_ships)
        for ship in other_ships:
            ship.health -= damage_per_remaining_ship

    def _distribute_damage_randomly(self, target_fleet: "Fleet", damage: float) -> None:
        """
        Distribute damage randomly among ships (scatter formation).

        Args:
            target_fleet: The fleet to apply damage to
            damage: Amount of damage to apply
        """
        if not target_fleet.ships:
            return

        # In scatter formation, damage is distributed randomly
        # Some ships might take more damage, others less or none

        # Determine how many ships will be hit (between 50-90% of ships)
        num_ships = len(target_fleet.ships)
        num_ships_hit = max(1, int(num_ships * random.uniform(0.5, 0.9)))

        # Select random ships to hit
        ships_to_hit = random.sample(target_fleet.ships, num_ships_hit)

        # Distribute damage among selected ships
        damage_per_ship = damage / num_ships_hit
        for ship in ships_to_hit:
            ship.health -= damage_per_ship

    def set_auto_engagement(
        self, auto_engage: bool, attack_same_faction: bool = False
    ) -> None:
        """
        Configure auto-engagement settings for the fleet.

        Args:
            auto_engage: Whether to automatically engage enemy fleets when detected
            attack_same_faction: Whether to attack fleets of the same faction
        """
        self.auto_engage = auto_engage
        self.attack_same_faction = attack_same_faction

        if auto_engage:
            logging.info(
                f"Fleet {self.fleet_id} auto-engagement enabled. Attack same faction: {attack_same_faction}"
            )
        else:
            logging.info(f"Fleet {self.fleet_id} auto-engagement disabled")

    def _check_for_enemies(self, nearby_fleets: List["Fleet"] = None) -> None:
        """
        Check for nearby enemy fleets and engage if auto-engage is enabled.

        Args:
            nearby_fleets: Optional list of known nearby fleets to check
        """
        if (
            not self.is_active
            or self.in_combat
            or not hasattr(self, "auto_engage")
            or not self.auto_engage
        ):
            return

        # If no nearby fleets provided, we can't check
        # In a real implementation, this would query a spatial index or fleet manager
        if not nearby_fleets:
            return

        for fleet in nearby_fleets:
            # Skip if it's our own fleet or not active
            if fleet == self or not fleet.is_active:
                continue

            # Skip if it's the same faction (unless we're set to attack same faction)
            if self.faction == fleet.faction and not getattr(
                self, "attack_same_faction", False
            ):
                continue

            # Check if in detection range
            distance = self._distance(self.position, fleet.position)
            if distance <= self.detection_range:
                # Engage the fleet with our default stance
                self.engage_fleet(fleet, self.combat_stance)
                # Only engage one fleet at a time
                break
