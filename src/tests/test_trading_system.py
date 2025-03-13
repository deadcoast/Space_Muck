"""
Test suite for the trading system functionality.
"""

# Standard library imports
import random

# Third-party library imports

# Local application imports
from systems.trading_system import TradingSystem
import unittest

class MockPlayer:
    """Mock player class for testing trading system."""

    def __init__(self, level=1, credits=1000):
        self.level = level
        self.credits = credits
        self.inventory = {}
        self.rare_inventory = {}
        self.current_station = None
        self.trading_history = []
        self.faction_reputation = {
            "miners_guild": 0,
            "explorers_union": 0,
            "galactic_navy": 0,
            "traders_coalition": 0,
            "fringe_colonies": 0,
        }

    def get_faction_price_modifier(self, faction_id):
        """Mock faction price modifier."""
        reputation = self.faction_reputation.get(faction_id, 0)
        # -100 to 100 reputation maps to 1.5 to 0.5 price modifier
        return max(0.5, min(1.5, 1.0 - (reputation / 200)))

    def update_faction_reputation(self, faction_id, amount):
        """Mock faction reputation update."""
        if faction_id in self.faction_reputation:
            self.faction_reputation[faction_id] += amount
            # Clamp reputation between -100 and 100
            self.faction_reputation[faction_id] = max(
                -100, min(100, self.faction_reputation[faction_id])
            )

class TestTradingSystem(unittest.TestCase):
    """Test cases for the trading system."""

    def setUp(self):
        """Set up test fixtures."""
        # Seed random for reproducible tests
        random.seed(42)

        # Create trading system
        self.trading_system = TradingSystem()

        # Create a mock player
        self.player = MockPlayer()

        # Create test trading stations
        self.trading_system.create_trading_station(
            "station_alpha", (25, 25), "miners_guild"
        )
        self.trading_system.create_trading_station(
            "station_beta", (75, 75), "traders_coalition"
        )

    def test_commodity_initialization(self):
        """Test that commodities are properly initialized."""
        self.assertEqual(len(self.trading_system.commodities), 5)
        self.assertIn("common_minerals", self.trading_system.commodities)
        self.assertIn("rare_minerals", self.trading_system.commodities)
        self.assertIn("anomalous_materials", self.trading_system.commodities)
        self.assertIn("fuel_cells", self.trading_system.commodities)
        self.assertIn("ship_parts", self.trading_system.commodities)

    def test_trading_station_creation(self):
        """Test that trading stations can be created."""
        self.assertEqual(len(self.trading_system.trading_stations), 2)

        # Check station properties
        station = self.trading_system.trading_stations["station_alpha"]
        self.assertEqual(station["position"], (25, 25))
        self.assertEqual(station["faction_id"], "miners_guild")
        self.assertIn("prices", station)
        self.assertIn("inventory", station)

        # Check that prices are initialized
        self.assertEqual(len(station["prices"]), 5)
        self.assertIn("common_minerals", station["prices"])

    def test_price_fluctuation(self):
        """Test that prices fluctuate when market updates."""
        # Get initial prices
        initial_price = self.trading_system.commodities["common_minerals"][
            "current_price"
        ]

        # Update market multiple times
        for i in range(5):
            self.trading_system.update(i * 1000)

        # Check that price has changed
        new_price = self.trading_system.commodities["common_minerals"]["current_price"]
        self.assertNotEqual(initial_price, new_price)

    def test_market_events(self):
        """Test that market events affect prices."""
        # Force a market event
        event = {
            "name": "Test Event",
            "description": "Test event for price changes",
            "start_time": 0,
            "end_time": 2000,
            "affected_commodities": ["common_minerals"],
            "price_modifier": 2.0,  # Double the price
        }
        self.trading_system.active_events.append(event)

        # Get initial price
        initial_price = self.trading_system.commodities["common_minerals"][
            "current_price"
        ]

        # Update market
        self.trading_system.update(1000)

        # Check that price has been affected by event
        new_price = self.trading_system.commodities["common_minerals"]["current_price"]
        # Price should be higher due to event
        self.assertGreater(new_price, initial_price)

        # Test event expiration
        self.trading_system.update(3000)  # After event end time
        self.assertEqual(len(self.trading_system.active_events), 0)

    def test_buy_commodity(self):
        """Test buying commodities from a station."""
        # Set up test
        station_id = "station_alpha"
        commodity_id = "common_minerals"
        quantity = 5

        # Ensure station has enough inventory
        self.trading_system.trading_stations[station_id]["inventory"][commodity_id] = 10

        # Get initial values
        initial_credits = self.player.credits
        initial_inventory = self.player.inventory.get(commodity_id, 0)
        initial_station_inventory = self.trading_system.trading_stations[station_id][
            "inventory"
        ][commodity_id]

        # Buy commodity
        result = self.trading_system.buy_commodity(
            station_id, commodity_id, quantity, self.player
        )

        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["quantity"], quantity)

        # Check player state
        self.assertLess(self.player.credits, initial_credits)  # Credits decreased
        self.assertEqual(
            self.player.inventory.get(commodity_id, 0), initial_inventory + quantity
        )

        # Check station state
        self.assertEqual(
            self.trading_system.trading_stations[station_id]["inventory"][commodity_id],
            initial_station_inventory - quantity,
        )

    def test_sell_commodity(self):
        """Test selling commodities to a station."""
        # Set up test
        station_id = "station_beta"
        commodity_id = "rare_minerals"
        quantity = 3

        # Ensure player has enough inventory
        self.player.inventory[commodity_id] = 5

        # Get initial values
        initial_credits = self.player.credits
        initial_inventory = self.player.inventory[commodity_id]
        initial_station_inventory = self.trading_system.trading_stations[station_id][
            "inventory"
        ].get(commodity_id, 0)

        # Sell commodity
        result = self.trading_system.sell_commodity(
            station_id, commodity_id, quantity, self.player
        )

        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["quantity"], quantity)

        # Check player state
        self.assertGreater(self.player.credits, initial_credits)  # Credits increased
        self.assertEqual(
            self.player.inventory.get(commodity_id, 0), initial_inventory - quantity
        )

        # Check station state
        self.assertEqual(
            self.trading_system.trading_stations[station_id]["inventory"][commodity_id],
            initial_station_inventory + quantity,
        )

    def test_faction_price_modifiers(self):
        """Test that faction reputation affects prices."""
        station_id = "station_alpha"
        commodity_id = "common_minerals"

        # Get base price
        base_price = self.trading_system.get_station_buy_price(station_id, commodity_id)

        # Improve reputation with miners_guild
        self.player.faction_reputation["miners_guild"] = 50  # Good reputation

        # Get new price
        good_rep_price = self.trading_system.get_station_buy_price(
            station_id, commodity_id, self.player
        )

        # Price should be lower with good reputation
        self.assertLess(good_rep_price, base_price)

        # Test bad reputation
        self.player.faction_reputation["miners_guild"] = -50  # Bad reputation
        bad_rep_price = self.trading_system.get_station_buy_price(
            station_id, commodity_id, self.player
        )

        # Price should be higher with bad reputation
        self.assertGreater(bad_rep_price, base_price)

    def test_quest_generation(self):
        """Test that quests are generated appropriately."""
        # Generate a quest for level 1 player
        quest = self.trading_system.generate_trading_quest(1)

        # Check quest properties
        self.assertIn("type", quest)
        self.assertIn("title", quest)
        self.assertIn("description", quest)
        self.assertIn("reward", quest)

        # Level 1 should only get delivery or procurement quests
        self.assertIn(quest["type"], ["trading_delivery", "trading_procurement"])

        # Generate a quest for level 3 player (should include market_manipulation)
        # Removed unused assignment to high_level_quest
        self.trading_system.generate_trading_quest(
            3
        )  # Just call the method without storing the result

        # Generate multiple quests to ensure we get different types
        quest_types = set()
        for _ in range(10):
            quest = self.trading_system.generate_trading_quest(3)
            quest_types.add(quest["type"])

        # Should include market_manipulation quests for level 3
        self.assertIn("trading_market_manipulation", quest_types)

    def _setup_quest_test_scenario(self, commodity_quantity, station_name):
        """Helper method to set up quest test scenarios.

        Args:
            commodity_quantity: Quantity of common_minerals to set in player inventory
            station_name: Station to set as player's current station
        """
        self.player.inventory["common_minerals"] = commodity_quantity
        self.player.current_station = station_name

    def test_quest_completion(self):
        """Test quest completion logic."""
        # Create a delivery quest
        quest = {
            "type": "trading_delivery",
            "commodity_id": "common_minerals",
            "quantity": 5,
            "destination_station": "station_alpha",
            "reward": 100,
            "faction_id": "miners_guild",
            "difficulty": 1,
            "completed": False,
        }

        # Player doesn't have enough inventory
        self._setup_quest_test_scenario(3, "station_alpha")
        self.assertFalse(self.trading_system.check_quest_completion(quest, self.player))

        # Player has enough inventory but at wrong station
        self._setup_quest_test_scenario(10, "station_beta")
        self.assertFalse(self.trading_system.check_quest_completion(quest, self.player))

        # Player has enough inventory and at correct station
        self._setup_quest_test_scenario(10, "station_alpha")
        initial_credits = self.player.credits
        initial_reputation = self.player.faction_reputation["miners_guild"]

        self.assertTrue(self.trading_system.check_quest_completion(quest, self.player))

        # Check rewards
        self.assertEqual(self.player.credits, initial_credits + quest["reward"])
        self.assertGreater(
            self.player.faction_reputation["miners_guild"], initial_reputation
        )

        # Check inventory was reduced
        self.assertEqual(self.player.inventory["common_minerals"], 5)  # 10 - 5

if __name__ == "__main__":
    unittest.main()
