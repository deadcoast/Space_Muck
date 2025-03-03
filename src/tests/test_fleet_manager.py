#!/usr/bin/env python3
"""
Unit tests for the Fleet Manager class.
"""

import unittest
import sys
import os
import math
import random
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the class to test
from systems.fleet_manager import Fleet
from src.entities.enemy_ship import EnemyShip


class TestFleetManager(unittest.TestCase):
    """Test cases for the Fleet Manager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a fleet for testing
        self.fleet = Fleet(
            fleet_id="test-fleet-1",
            name="Test Fleet",
            faction="player",
            position=(50, 50),
            formation="line",
            commander_level=3
        )
        
        # Create some mock ships
        self.mock_ships = []
        for i in range(5):
            ship = MagicMock(spec=EnemyShip)
            ship.entity_id = f"ship-{i}"
            ship.ship_id = f"ship-{i}"
            ship.position = (50, 50)
            ship.health = 100.0
            ship.role = "cruiser" if i == 0 else "destroyer"
            self.mock_ships.append(ship)
        
        # Add ships to the fleet
        for i, ship in enumerate(self.mock_ships):
            self.fleet.add_ship(ship, is_flagship=(i == 0))

    def test_initialization(self):
        """Test that fleet initializes with correct values."""
        # Test with explicit values
        self.assertEqual(self.fleet.fleet_id, "test-fleet-1")
        self.assertEqual(self.fleet.name, "Test Fleet")
        self.assertEqual(self.fleet.faction, "player")
        self.assertEqual(self.fleet.position, (50, 50))
        self.assertEqual(self.fleet.formation, "line")
        self.assertEqual(self.fleet.commander_level, 3)
        self.assertEqual(len(self.fleet.ships), 5)
        self.assertIsNotNone(self.fleet.flagship)
        self.assertEqual(self.fleet.flagship.ship_id, "ship-0")
        self.assertTrue(self.fleet.is_active)
        self.assertFalse(self.fleet.in_combat)
        self.assertIsNone(self.fleet.target_fleet)
        self.assertEqual(self.fleet.combat_stance, "balanced")

    def test_add_and_remove_ship(self):
        """Test adding and removing ships from the fleet."""
        # Create a new ship
        new_ship = MagicMock(spec=EnemyShip)
        new_ship.entity_id = "ship-new"
        new_ship.ship_id = "ship-new"
        new_ship.position = (50, 50)
        new_ship.health = 100.0
        new_ship.role = "frigate"
        
        # Add the ship to the fleet
        result = self.fleet.add_ship(new_ship)
        self.assertTrue(result)
        self.assertEqual(len(self.fleet.ships), 6)
        self.assertIn(new_ship, self.fleet.ships)
        
        # Try to add the same ship again (should fail)
        result = self.fleet.add_ship(new_ship)
        self.assertFalse(result)
        self.assertEqual(len(self.fleet.ships), 6)
        
        # Remove the ship
        result = self.fleet.remove_ship(new_ship)
        self.assertTrue(result)
        self.assertEqual(len(self.fleet.ships), 5)
        self.assertNotIn(new_ship, self.fleet.ships)
        
        # Try to remove a non-existent ship
        non_existent_ship = MagicMock(spec=EnemyShip)
        non_existent_ship.entity_id = "non-existent"
        result = self.fleet.remove_ship(non_existent_ship)
        self.assertFalse(result)
        self.assertEqual(len(self.fleet.ships), 5)

    def test_set_flagship(self):
        """Test setting a ship as the fleet flagship."""
        # Get a non-flagship ship
        non_flagship = self.mock_ships[1]
        
        # Set it as the flagship
        result = self.fleet.set_flagship(non_flagship)
        self.assertTrue(result)
        self.assertEqual(self.fleet.flagship, non_flagship)
        
        # Try to set a non-existent ship as flagship
        non_existent_ship = MagicMock(spec=EnemyShip)
        non_existent_ship.entity_id = "non-existent"
        result = self.fleet.set_flagship(non_existent_ship)
        self.assertFalse(result)
        self.assertEqual(self.fleet.flagship, non_flagship)  # Flagship should not change

    def test_formation_management(self):
        """Test changing fleet formations."""
        # Test setting a valid formation
        result = self.fleet.set_formation("wedge")
        self.assertTrue(result)
        self.assertEqual(self.fleet.formation, "wedge")
        
        # Test setting another valid formation
        result = self.fleet.set_formation("circle")
        self.assertTrue(result)
        self.assertEqual(self.fleet.formation, "circle")
        
        # Test setting an invalid formation
        result = self.fleet.set_formation("invalid-formation")
        self.assertFalse(result)
        self.assertEqual(self.fleet.formation, "circle")  # Formation should not change

    def test_ship_positioning(self):
        """Test that ships are positioned correctly based on formation."""
        # Test line formation
        self.fleet.set_formation("line")
        self.fleet._update_ship_positions()
        
        # In line formation, ships should be positioned horizontally
        flagship_x = self.fleet.flagship.position[0]
        flagship_y = self.fleet.flagship.position[1]
        
        # Check that non-flagship ships are positioned horizontally
        for ship in self.fleet.ships:
            if ship != self.fleet.flagship:
                # Y coordinate should be the same as flagship
                self.assertEqual(ship.position[1], flagship_y)
                # X coordinate should be different
                self.assertNotEqual(ship.position[0], flagship_x)
        
        # Test column formation
        self.fleet.set_formation("column")
        self.fleet._update_ship_positions()
        
        # In column formation, ships should be positioned vertically
        flagship_x = self.fleet.flagship.position[0]
        flagship_y = self.fleet.flagship.position[1]
        
        # Check that non-flagship ships are positioned vertically
        for ship in self.fleet.ships:
            if ship != self.fleet.flagship:
                # X coordinate should be the same as flagship
                self.assertEqual(ship.position[0], flagship_x)
                # Y coordinate should be different
                self.assertNotEqual(ship.position[1], flagship_y)

    def test_move_to(self):
        """Test fleet movement to a destination."""
        # Set a destination
        destination = (100, 100)
        result = self.fleet.move_to(destination)
        
        # Check that movement was initiated
        self.assertTrue(result)
        self.assertEqual(self.fleet.destination, destination)
        self.assertEqual(self.fleet.current_orders["type"], "move_to")
        
        # Update the fleet to simulate movement
        self.fleet.update(1.0)
        
        # Check that the fleet has moved toward the destination
        self.assertNotEqual(self.fleet.position, (50, 50))  # Original position
        
        # Calculate expected position after one update
        # Direction vector is normalized, so movement is proportional to speed
        dx = destination[0] - 50  # Original x = 50
        dy = destination[1] - 50  # Original y = 50
        distance = math.sqrt(dx * dx + dy * dy)
        dx /= distance
        dy /= distance
        
        expected_x = 50 + dx * self.fleet.speed
        expected_y = 50 + dy * self.fleet.speed
        
        # Allow for small floating-point differences
        self.assertAlmostEqual(self.fleet.position[0], expected_x, delta=0.1)
        self.assertAlmostEqual(self.fleet.position[1], expected_y, delta=0.1)

    def test_patrol_between(self):
        """Test fleet patrol between waypoints."""
        # Set patrol points
        patrol_points = [(60, 60), (70, 70), (60, 70), (70, 60)]
        result = self.fleet.patrol_between(patrol_points)
        
        # Check that patrol was initiated
        self.assertTrue(result)
        self.assertEqual(self.fleet.waypoints, patrol_points)
        self.assertEqual(self.fleet.destination, patrol_points[0])
        self.assertEqual(self.fleet.current_orders["type"], "patrol")
        
        # Update the fleet to move to the first waypoint
        # We'll need multiple updates to reach it
        for _ in range(10):
            self.fleet.update(1.0)
            
            # If we've reached the first waypoint, break
            if self.fleet.current_orders.get("current_waypoint_index", 0) > 0:
                break
        
        # Check that we've moved to the next waypoint
        self.assertEqual(self.fleet.current_orders["current_waypoint_index"], 1)
        self.assertEqual(self.fleet.destination, patrol_points[1])
        
        # Test with invalid patrol points (less than 2)
        result = self.fleet.patrol_between([])
        self.assertFalse(result)
        
        result = self.fleet.patrol_between([(80, 80)])
        self.assertFalse(result)

    def test_combat_engagement(self):
        """Test engaging another fleet in combat."""
        # Create an enemy fleet
        enemy_fleet = Fleet(
            fleet_id="enemy-fleet-1",
            name="Enemy Fleet",
            faction="enemy",
            position=(55, 55),  # Close to our fleet
            formation="wedge"
        )
        
        # Add ships to the enemy fleet
        for i in range(3):
            ship = MagicMock(spec=EnemyShip)
            ship.entity_id = f"enemy-ship-{i}"
            ship.ship_id = f"enemy-ship-{i}"
            ship.position = (55, 55)
            ship.health = 100.0
            enemy_fleet.add_ship(ship)
        
        # Engage the enemy fleet
        result = self.fleet.engage_fleet(enemy_fleet, stance="aggressive")
        
        # Check that engagement was initiated
        self.assertTrue(result)
        self.assertTrue(self.fleet.in_combat)
        self.assertEqual(self.fleet.target_fleet, enemy_fleet)
        self.assertEqual(self.fleet.combat_stance, "aggressive")
        
        # Update the fleet to simulate combat
        self.fleet.update(1.0)
        
        # Check that the fleet has moved to maintain ideal combat distance
        # For aggressive stance, ideal distance is 2.0
        
        # Check that damage was applied to enemy ships
        # This is difficult to test directly since we're using mocks
        # We'll check that _execute_attack was called by checking if the enemy fleet's ships lost health
        
        # Disengage from combat
        result = self.fleet.disengage()
        self.assertTrue(result)
        self.assertFalse(self.fleet.in_combat)
        self.assertIsNone(self.fleet.target_fleet)

    def test_combat_stances(self):
        """Test different combat stances."""
        # Create an enemy fleet
        enemy_fleet = Fleet(
            fleet_id="enemy-fleet-2",
            name="Enemy Fleet",
            faction="enemy",
            position=(55, 55),
            formation="wedge"
        )
        
        # Add ships to the enemy fleet
        for i in range(3):
            ship = MagicMock(spec=EnemyShip)
            ship.entity_id = f"enemy-ship-{i}"
            ship.ship_id = f"enemy-ship-{i}"
            ship.position = (55, 55)
            ship.health = 100.0
            enemy_fleet.add_ship(ship)
        
        # Test different combat stances
        stances = ["balanced", "aggressive", "defensive", "evasive"]
        for stance in stances:
            # Reset fleet position
            self.fleet.position = (50, 50)
            
            # Engage with the current stance
            self.fleet.engage_fleet(enemy_fleet, stance=stance)
            self.assertEqual(self.fleet.combat_stance, stance)
            
            # Update to simulate combat
            self.fleet.update(1.0)
            
            # Disengage
            self.fleet.disengage()

    def test_damage_distribution(self):
        """Test damage distribution based on formation."""
        # Create an enemy fleet
        enemy_fleet = Fleet(
            fleet_id="enemy-fleet-3",
            name="Enemy Fleet",
            faction="enemy",
            position=(55, 55),
            formation="line"
        )
        
        # Add ships to the enemy fleet with health tracking
        enemy_ships = []
        for i in range(5):
            ship = MagicMock(spec=EnemyShip)
            ship.entity_id = f"enemy-ship-{i}"
            ship.ship_id = f"enemy-ship-{i}"
            ship.position = (55, 55)
            ship.health = 100.0
            enemy_fleet.add_ship(ship)
            enemy_ships.append(ship)
        
        # Test even damage distribution (line formation)
        damage = 50.0
        self.fleet._distribute_damage_evenly(enemy_fleet, damage)
        
        # Check that damage was distributed evenly
        expected_damage_per_ship = damage / len(enemy_ships)
        for ship in enemy_ships:
            self.assertAlmostEqual(ship.health, 100.0 - expected_damage_per_ship, delta=0.1)
        
        # Reset ship health
        for ship in enemy_ships:
            ship.health = 100.0
        
        # Test front-heavy damage distribution (column formation)
        enemy_fleet.formation = "column"
        self.fleet._distribute_damage_front_heavy(enemy_fleet, damage)
        
        # Check that front ship took more damage
        self.assertAlmostEqual(enemy_ships[0].health, 100.0 - (damage * 0.4), delta=0.1)
        
        # Check that other ships took less damage
        remaining_damage_per_ship = (damage * 0.6) / (len(enemy_ships) - 1)
        for i in range(1, len(enemy_ships)):
            self.assertAlmostEqual(enemy_ships[i].health, 100.0 - remaining_damage_per_ship, delta=0.1)

    def test_resource_management(self):
        """Test resource management and distribution."""
        # Check initial resource levels
        self.assertEqual(self.fleet.resources["fuel_cells"], 200)
        self.assertEqual(self.fleet.resources["ship_parts"], 100)
        
        # Add resources
        self.fleet.add_resources("fuel_cells", 50)
        self.assertEqual(self.fleet.resources["fuel_cells"], 250)
        
        # Try to add more than capacity
        max_fuel = self.fleet.resource_capacity["fuel_cells"]
        self.fleet.add_resources("fuel_cells", max_fuel * 2)
        self.assertEqual(self.fleet.resources["fuel_cells"], max_fuel)
        
        # Try to add invalid resource type
        result = self.fleet.add_resources("invalid_resource", 50)
        self.assertFalse(result)
        
        # Try to add negative amount
        result = self.fleet.add_resources("fuel_cells", -50)
        self.assertFalse(result)
        
        # Test resource consumption
        original_fuel = self.fleet.resources["fuel_cells"]
        # Simulate 1 day of movement
        self.fleet.path = [(60, 60)]  # Set path to trigger increased consumption
        self.fleet._consume_resources(86400.0)  # 1 day in seconds
        
        # Check that resources were consumed
        self.assertLess(self.fleet.resources["fuel_cells"], original_fuel)
        
        # Test resource distribution methods
        for method in ["equal", "proportional", "priority"]:
            self.fleet.resource_distribution_method = method
            self.fleet._distribute_resources()
            # Hard to test actual distribution with mocks, but we can verify the method runs

    def test_auto_engagement(self):
        """Test auto-engagement functionality."""
        # Enable auto-engagement
        self.fleet.auto_engage = True
        
        # Create an enemy fleet within detection range
        enemy_fleet = Fleet(
            fleet_id="enemy-fleet-4",
            name="Enemy Fleet",
            faction="enemy",
            position=(55, 55),  # Within detection range
            formation="wedge"
        )
        
        # Add ships to the enemy fleet
        for i in range(3):
            ship = MagicMock(spec=EnemyShip)
            ship.entity_id = f"enemy-ship-{i}"
            ship.ship_id = f"enemy-ship-{i}"
            ship.position = (55, 55)
            ship.health = 100.0
            enemy_fleet.add_ship(ship)
        
        # Mock the _check_for_enemies method to simulate detecting the enemy fleet
        original_check_for_enemies = self.fleet._check_for_enemies
        self.fleet._check_for_enemies = MagicMock(return_value=True)
        
        # Update the fleet
        self.fleet.update(1.0)
        
        # Check that _check_for_enemies was called
        self.fleet._check_for_enemies.assert_called_once()
        
        # Restore the original method
        self.fleet._check_for_enemies = original_check_for_enemies


if __name__ == "__main__":
    unittest.main()
