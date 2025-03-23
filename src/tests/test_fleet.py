#!/usr/bin/env python3
"""
Unit tests for the Fleet class.
"""

# Standard library imports
import os
import sys
import unittest

# Local application imports
from entities.base_entity import BaseEntity
from entities.fleet import Fleet

# Third-party library imports


# Removed unused mock imports

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the class to test


class TestFleet(unittest.TestCase):
    """Test cases for the Fleet class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a fleet for testing
        self.fleet = Fleet(
            fleet_id="fleet-123",
            owner_id="player-1",
            fleet_name="Test Fleet",
            color=(100, 150, 200),
            position=(10, 20),
            ships={
                "ship-1": {"type": "scout", "strength": 10},
                "ship-2": {"type": "destroyer", "strength": 30},
            },
        )

    def test_initialization(self):
        """Test that fleet initializes with correct values."""
        # Test with explicit values
        self.assertEqual(self.fleet.entity_id, "fleet-123")
        self.assertEqual(self.fleet.entity_type, "fleet")
        self.assertEqual(self.fleet.owner_id, "player-1")
        self.assertEqual(self.fleet.fleet_name, "Test Fleet")
        self.assertEqual(self.fleet.color, (100, 150, 200))
        self.assertEqual(self.fleet.position, (10, 20))
        self.assertEqual(len(self.fleet.ships), 2)
        self.assertEqual(self.fleet.formation, "standard")
        self.assertEqual(self.fleet.speed, 1.0)
        self.assertFalse(self.fleet.is_moving)
        self.assertIsNone(self.fleet.destination)
        self.assertEqual(self.fleet.path, [])
        self.assertEqual(self.fleet.fuel, 100)
        self.assertEqual(self.fleet.max_fuel, 100)

        # Test inheritance
        self.assertIsInstance(self.fleet, BaseEntity)

    def test_default_initialization(self):
        """Test initialization with default values."""
        fleet = Fleet()

        # Test default values
        self.assertIsNotNone(fleet.entity_id)
        self._verify_default_values(fleet)
        self.assertEqual(fleet.path, [])
        self.assertEqual(fleet.fuel, 100)
        self.assertEqual(fleet.max_fuel, 100)

    def test_add_ship(self):
        """Test adding a ship to the fleet."""
        # Add a new ship
        self.fleet.add_ship("ship-3", {"type": "cruiser", "strength": 50})

        # Verify the ship was added
        self.assertIn("ship-3", self.fleet.ships)
        self.assertEqual(self.fleet.ships["ship-3"]["type"], "cruiser")
        self.assertEqual(self.fleet.ships["ship-3"]["strength"], 50)
        self.assertEqual(len(self.fleet.ships), 3)

    def test_remove_ship(self):
        """Test removing a ship from the fleet."""
        # Remove an existing ship
        result = self.fleet.remove_ship("ship-1")

        # Verify the ship was removed
        self.assertTrue(result)
        self.assertNotIn("ship-1", self.fleet.ships)
        self.assertEqual(len(self.fleet.ships), 1)

        # Try to remove a non-existent ship
        result = self.fleet.remove_ship("non-existent-ship")

        # Verify the operation failed
        self.assertFalse(result)
        self.assertEqual(len(self.fleet.ships), 1)

    def test_set_formation(self):
        """Test setting the fleet formation."""
        # Set a new formation
        self.fleet.set_formation("defensive")

        # Verify the formation was set
        self.assertEqual(self.fleet.formation, "defensive")

        # Set another formation
        self.fleet.set_formation("offensive")

        # Verify the formation was updated
        self.assertEqual(self.fleet.formation, "offensive")

    def test_set_destination(self):
        """Test setting the fleet destination."""
        # Set a destination
        self.fleet.set_destination(50, 60)

        # Verify the destination was set
        self.assertEqual(self.fleet.destination, (50, 60))
        self.assertTrue(self.fleet.is_moving)
        self.assertGreater(len(self.fleet.path), 0)

    def test_calculate_path(self):
        """Test the path calculation method."""
        # Set position and destination
        self.fleet.position = (0, 0)
        self.fleet.destination = (2, 2)

        # Calculate path
        path = self.fleet._calculate_path()

        # Verify the path
        expected_path = [(1, 1), (2, 2)]
        self.assertEqual(path, expected_path)

        # Test with no position
        self.fleet.position = None
        path = self.fleet._calculate_path()
        self.assertEqual(path, [])

        # Test with no destination
        self.fleet.position = (0, 0)
        self.fleet.destination = None
        path = self.fleet._calculate_path()
        self.assertEqual(path, [])

    def test_update(self):
        """Test the update method."""
        # Set up for movement
        self.fleet.position = (0, 0)
        self.fleet.set_destination(2, 2)
        self.fleet.path = [(1, 1), (2, 2)]
        self.fleet.is_moving = True

        # Update the fleet
        game_state = {"tick": 100}
        self.fleet.update(game_state)

        # Verify the fleet moved
        self.assertEqual(self.fleet.position, (1, 1))
        self.assertEqual(self.fleet.path, [(2, 2)])
        self.assertTrue(self.fleet.is_moving)

        # Update again to reach destination
        self.fleet.update(game_state)

        # Verify the fleet reached its destination
        self.assertEqual(self.fleet.position, (2, 2))
        self.assertEqual(self.fleet.path, [])
        self.assertFalse(self.fleet.is_moving)
        self.assertIsNone(self.fleet.destination)

        # Test fuel consumption
        self.fleet.position = (0, 0)
        self.fleet.set_destination(100, 100)
        self.fleet.fuel = 0.1  # Just enough for one move
        self.fleet.update(game_state)

        # Verify fuel was consumed and movement stopped
        self.assertEqual(self.fleet.fuel, 0)
        self.assertFalse(self.fleet.is_moving)

    def test_get_fleet_strength(self):
        """Test calculating fleet strength."""
        # Calculate strength
        strength = self.fleet.get_fleet_strength()

        # Verify the strength
        expected_strength = 10 + 30  # scout (10) + destroyer (30)
        self.assertEqual(strength, expected_strength)

        # Add a ship and recalculate
        self.fleet.add_ship("ship-3", {"type": "cruiser", "strength": 50})
        strength = self.fleet.get_fleet_strength()

        # Verify the updated strength
        expected_strength = 10 + 30 + 50
        self.assertEqual(strength, expected_strength)

        # Test with a ship that has no strength value
        self.fleet.add_ship("ship-4", {"type": "transport"})
        strength = self.fleet.get_fleet_strength()

        # Strength should remain the same
        self.assertEqual(strength, expected_strength)

    def test_get_ship_count(self):
        """Test getting the ship count."""
        # Get initial count
        count = self.fleet.get_ship_count()

        # Verify the count
        self.assertEqual(count, 2)

        # Add a ship and recount
        self.fleet.add_ship("ship-3", {"type": "cruiser", "strength": 50})
        count = self.fleet.get_ship_count()

        # Verify the updated count
        self.assertEqual(count, 3)

        # Remove a ship and recount
        self.fleet.remove_ship("ship-1")
        count = self.fleet.get_ship_count()

        # Verify the updated count
        self.assertEqual(count, 2)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        # Convert to dictionary
        fleet_dict = self.fleet.to_dict()

        # Verify dictionary contents
        self.assertEqual(fleet_dict["entity_id"], "fleet-123")
        self.assertEqual(fleet_dict["entity_type"], "fleet")
        self.assertEqual(fleet_dict["owner_id"], "player-1")
        self.assertEqual(fleet_dict["fleet_name"], "Test Fleet")
        self.assertEqual(fleet_dict["color"], (100, 150, 200))
        self.assertEqual(fleet_dict["position"], (10, 20))
        self.assertEqual(fleet_dict["ships"]["ship-1"]["type"], "scout")
        self.assertEqual(fleet_dict["ships"]["ship-2"]["type"], "destroyer")
        self.assertEqual(fleet_dict["formation"], "standard")
        self.assertEqual(fleet_dict["speed"], 1.0)
        self.assertFalse(fleet_dict["is_moving"])
        self.assertIsNone(fleet_dict["destination"])
        self.assertEqual(fleet_dict["fuel"], 100)
        self.assertEqual(fleet_dict["max_fuel"], 100)

    def test_from_dict(self):
        """Test creating a fleet from a dictionary."""
        # Create a dictionary
        fleet_dict = {
            "entity_id": "fleet-456",
            "entity_type": "fleet",
            "owner_id": "player-2",
            "fleet_name": "Test Fleet 2",
            "color": (200, 100, 150),
            "position": (30, 40),
            "ships": {
                "ship-5": {"type": "scout", "strength": 15},
                "ship-6": {"type": "carrier", "strength": 60},
            },
            "formation": "offensive",
            "speed": 1.5,
            "is_moving": True,
            "destination": (50, 60),
            "fuel": 80,
            "max_fuel": 120,
        }

        # Create a fleet from the dictionary
        fleet = Fleet.from_dict(fleet_dict)

        # Verify the fleet attributes
        self.assertEqual(fleet.entity_id, "fleet-456")
        self.assertEqual(fleet.entity_type, "fleet")
        self.assertEqual(fleet.owner_id, "player-2")
        self.assertEqual(fleet.fleet_name, "Test Fleet 2")
        self.assertEqual(fleet.color, (200, 100, 150))
        self.assertEqual(fleet.position, (30, 40))
        self.assertEqual(len(fleet.ships), 2)
        self.assertEqual(fleet.ships["ship-5"]["type"], "scout")
        self.assertEqual(fleet.ships["ship-6"]["type"], "carrier")
        self.assertEqual(fleet.formation, "offensive")
        self.assertEqual(fleet.speed, 1.5)
        self.assertTrue(fleet.is_moving)
        self.assertEqual(fleet.destination, (50, 60))
        self.assertEqual(fleet.fuel, 80)
        self.assertEqual(fleet.max_fuel, 120)

        # Test with minimal data
        minimal_dict = {"entity_id": "fleet-789"}

        # Create a fleet from minimal dictionary
        fleet = Fleet.from_dict(minimal_dict)

        # Verify default values are used for missing fields
        self.assertEqual(fleet.entity_id, "fleet-789")
        self._verify_default_values(fleet)
        self.assertEqual(fleet.fuel, 100)
        self.assertEqual(fleet.max_fuel, 100)

    def _verify_default_values(self, fleet):
        # Verify default values
        self.assertEqual(fleet.entity_type, "fleet")
        self.assertIsNone(fleet.owner_id)
        self.assertEqual(fleet.fleet_name, "Unnamed Fleet")
        self.assertEqual(fleet.color, (100, 100, 255))
        self.assertIsNone(fleet.position)
        self.assertEqual(fleet.ships, {})
        self.assertEqual(fleet.formation, "standard")
        self.assertEqual(fleet.speed, 1.0)
        self.assertFalse(fleet.is_moving)
        self.assertIsNone(fleet.destination)


if __name__ == "__main__":
    unittest.main()
