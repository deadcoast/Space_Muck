"""
src/systems/trading_system.py

Trading System: Handles resource trading, market mechanics, and price fluctuations.

This module provides a comprehensive trading system with dynamic pricing,
market events, and integration with the player's inventory and economy.
"""

# Standard library imports
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

# Local application imports
from config import GAME_MAP_SIZE

# Third-party library imports


# Standard library imports

# Local application imports


class TradingSystem:
    """
    Core class that manages the trading system, including market mechanics,
    price fluctuations, and trading stations.
    """

    def __init__(self) -> None:
        """Initialize the trading system with default commodities and stations."""
        # Initialize base commodities with their default prices
        self.commodities = {
            "common_minerals": {
                "base_price": 10,
                "volatility": 0.1,  # How much prices fluctuate (0-1)
                "current_price": 10,
                "supply": 1.0,  # Supply factor (< 1 = shortage, > 1 = surplus)
                "description": "Common minerals used in basic manufacturing.",
            },
            "rare_minerals": {
                "base_price": 30,
                "volatility": 0.2,
                "current_price": 30,
                "supply": 1.0,
                "description": "Rare minerals with unique properties, valuable for advanced technology.",
            },
            "anomalous_materials": {
                "base_price": 75,
                "volatility": 0.3,
                "current_price": 75,
                "supply": 0.8,  # Slightly scarce by default
                "description": "Exotic materials with unusual properties, highly sought after.",
            },
            "fuel_cells": {
                "base_price": 25,
                "volatility": 0.15,
                "current_price": 25,
                "supply": 1.0,
                "description": "Standard fuel cells for spacecraft propulsion.",
            },
            "ship_parts": {
                "base_price": 50,
                "volatility": 0.1,
                "current_price": 50,
                "supply": 1.0,
                "description": "Replacement parts for ship maintenance and upgrades.",
            },
        }

        # Trading stations will be initialized later
        self.trading_stations = {}

        # Market events that can affect prices
        self.active_events = []

        # Initialize market update timer
        self.last_update = 0
        self.update_interval = 300  # Update market every 300 game ticks

        logging.info("Trading system initialized with 5 commodity types")

        # Initialize market update timer
        self.last_update = 0
        self.update_interval = 300  # Update market every 300 game ticks

    def update(self, game_time: int) -> List[Dict[str, Any]]:
        """
        Update market prices and handle market events.

        Args:
            game_time: Current game time in ticks

        Returns:
            List of market events that occurred during this update
        """
        # Only update at certain intervals
        if game_time - self.last_update < self.update_interval:
            return []

        self.last_update = game_time

        # Process expired events
        self._process_expired_events(game_time)

        # Random chance to generate a new market event
        if random.random() < 0.2:  # 20% chance per update
            if new_event := self._generate_market_event(game_time):
                self.active_events.append(new_event)
                logging.info(f"New market event: {new_event['name']}")

        # Update prices for all commodities
        for commodity_id, data in self.commodities.items():
            self._update_commodity_price(commodity_id, data)

        # Update trading station prices
        for station_id, station in self.trading_stations.items():
            self._update_station_prices(station)

        # Return active events for notification
        return self.active_events

    def _update_commodity_price(self, commodity_id: str, data: Dict[str, Any]) -> None:
        """
        Update the price of a single commodity based on volatility and supply.

        Args:
            commodity_id: Identifier for the commodity
            data: Commodity data dictionary
        """
        base_price = data["base_price"]
        volatility = data["volatility"]
        supply = data["supply"]

        # Calculate random price fluctuation
        # More volatile commodities have larger potential price swings
        fluctuation = random.uniform(-volatility, volatility)

        # Supply affects price - shortages increase prices, surpluses decrease them
        supply_factor = 2.0 - supply  # Inverted so lower supply = higher prices

        # Apply event modifiers
        event_modifier = self._get_event_modifier(commodity_id)

        # Calculate new price
        new_price = base_price * (1 + fluctuation) * supply_factor * event_modifier

        # Ensure price doesn't go too low or too high
        min_price = base_price * 0.5
        max_price = base_price * 3.0
        new_price = max(min_price, min(new_price, max_price))

        # Update the commodity price
        self.commodities[commodity_id]["current_price"] = round(new_price, 1)

        # Small random change to supply
        supply_change = random.uniform(-0.05, 0.05)
        new_supply = supply + supply_change
        # Keep supply within reasonable bounds
        new_supply = max(0.5, min(new_supply, 1.5))
        self.commodities[commodity_id]["supply"] = new_supply

    def _get_event_modifier(self, commodity_id: str) -> float:
        """
        Calculate price modifier based on active market events.

        Args:
            commodity_id: Identifier for the commodity

        Returns:
            float: Price modifier (1.0 = no change)
        """
        modifier = 1.0

        for event in self.active_events:
            if commodity_id in event["affected_commodities"]:
                modifier *= event["price_modifier"]

        return modifier

    def _process_expired_events(self, game_time: int) -> None:
        """
        Remove expired market events.

        Args:
            game_time: Current game time in ticks
        """
        self.active_events = [
            event for event in self.active_events if event["end_time"] > game_time
        ]

    def _generate_market_event(self, game_time: int) -> Optional[Dict[str, Any]]:
        """
        Generate a random market event.

        Args:
            game_time: Current game time in ticks

        Returns:
            Dict containing event details or None if no event generated
        """
        # Possible event types
        event_types = [
            {
                "name": "Mineral Shortage",
                "description": "A shortage of minerals has caused prices to rise.",
                "duration": 1200,  # Event lasts for 1200 ticks
                "affected_commodities": ["common_minerals", "rare_minerals"],
                "price_modifier": 1.5,  # 50% price increase
            },
            {
                "name": "Fuel Crisis",
                "description": "A fuel crisis has led to increased fuel cell prices.",
                "duration": 900,
                "affected_commodities": ["fuel_cells"],
                "price_modifier": 1.8,  # 80% price increase
            },
            {
                "name": "Technology Breakthrough",
                "description": "A new technology breakthrough has increased demand for anomalous materials.",
                "duration": 1500,
                "affected_commodities": ["anomalous_materials"],
                "price_modifier": 1.7,  # 70% price increase
            },
            {
                "name": "Manufacturing Surplus",
                "description": "Overproduction has led to a surplus of ship parts.",
                "duration": 1000,
                "affected_commodities": ["ship_parts"],
                "price_modifier": 0.6,  # 40% price decrease
            },
            {
                "name": "Mining Boom",
                "description": "A mining boom has flooded the market with common minerals.",
                "duration": 1100,
                "affected_commodities": ["common_minerals"],
                "price_modifier": 0.7,  # 30% price decrease
            },
        ]

        # Select a random event
        event_template = random.choice(event_types)

        # Create the event instance
        event = event_template.copy()
        event["start_time"] = game_time
        event["end_time"] = game_time + event["duration"]

        return event

    def _update_station_prices(self, station: Dict[str, Any]) -> None:
        """
        Update prices at a specific trading station based on location and faction.

        Args:
            station: Trading station data dictionary
        """
        location_factor = station.get("location_factor", 1.0)

        # Update each commodity price for this station
        for commodity_id in self.commodities:
            base_price = self.commodities[commodity_id]["current_price"]

            # Apply location modifier
            modified_price = base_price * location_factor

            # Store the modified price in the station's price list
            station["prices"][commodity_id] = round(modified_price, 1)

    def create_trading_station(
        self,
        station_id: str,
        position: Tuple[int, int],
        faction_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new trading station at the specified position.

        Args:
            station_id: Unique identifier for the station
            position: (x, y) coordinates of the station
            faction_id: Optional faction that controls this station

        Returns:
            The newly created trading station data
        """
        # Calculate location factor based on distance from center
        # Stations further from center have higher prices
        grid_width, grid_height = GAME_MAP_SIZE
        center_x, center_y = grid_width // 2, grid_height // 2
        x, y = position
        distance_from_center = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_distance = math.sqrt((grid_width // 2) ** 2 + (grid_height // 2) ** 2)
        location_factor = (
            1.0 + (distance_from_center / max_distance) * 0.3
        )  # Up to 30% higher prices

        # Initialize station with default values
        station = {
            "id": station_id,
            "position": position,
            "faction_id": faction_id,
            "location_factor": location_factor,
            "prices": {},
            "inventory": {
                commodity_id: random.randint(10, 50)
                for commodity_id in self.commodities
            },
            "last_restocked": 0,
            "restock_interval": 600,  # Restock every 600 game ticks
        }

        # Initialize prices for this station
        for commodity_id in self.commodities:
            station["prices"][commodity_id] = self.commodities[commodity_id][
                "current_price"
            ]

        # Add to trading stations dictionary
        self.trading_stations[station_id] = station

        logging.info(f"Created trading station {station_id} at position {position}")
        return station

    def get_station_buy_price(
        self, station_id: str, commodity_id: str, player=None
    ) -> float:
        """
        Get the price to buy a commodity from a station.

        Args:
            station_id: ID of the trading station
            commodity_id: ID of the commodity to buy
            player: Optional player object to apply reputation modifiers

        Returns:
            The buy price for the commodity
        """
        if station_id not in self.trading_stations:
            logging.warning(f"Trading station {station_id} not found")
            return 0.0

        station = self.trading_stations[station_id]
        base_price = station["prices"].get(commodity_id, 0.0)

        # Apply player's faction reputation modifier if available
        if (
            player
            and station["faction_id"]
            and hasattr(player, "get_faction_price_modifier")
        ):
            faction_modifier = player.get_faction_price_modifier(station["faction_id"])
            base_price *= faction_modifier

        return round(base_price, 1)

    def get_station_sell_price(
        self, station_id: str, commodity_id: str, player=None
    ) -> float:
        """
        Get the price to sell a commodity to a station.

        Args:
            station_id: ID of the trading station
            commodity_id: ID of the commodity to sell
            player: Optional player object to apply reputation modifiers

        Returns:
            The sell price for the commodity
        """
        if station_id not in self.trading_stations:
            logging.warning(f"Trading station {station_id} not found")
            return 0.0

        station = self.trading_stations[station_id]
        # Sell price is typically lower than buy price (station markup)
        base_price = station["prices"].get(commodity_id, 0.0) * 0.7  # 30% markdown

        # Apply player's faction reputation modifier if available
        if (
            player
            and station["faction_id"]
            and hasattr(player, "get_faction_price_modifier")
        ):
            faction_modifier = player.get_faction_price_modifier(station["faction_id"])
            # Better reputation means better sell prices
            base_price *= faction_modifier

        return round(base_price, 1)

    def buy_commodity(
        self, station_id: str, commodity_id: str, quantity: int, player
    ) -> Dict[str, Any]:
        """
        Player buys a commodity from a trading station.

        Args:
            station_id: ID of the trading station
            commodity_id: ID of the commodity to buy
            quantity: Amount to buy
            player: Player object making the purchase

        Returns:
            Dict with transaction results
        """
        if station_id not in self.trading_stations:
            return {
                "success": False,
                "message": f"Trading station {station_id} not found",
            }

        station = self.trading_stations[station_id]

        # Check if station has enough inventory
        available = station["inventory"].get(commodity_id, 0)
        if available < quantity:
            return {
                "success": False,
                "message": f"Not enough {commodity_id} available. Only {available} in stock.",
            }

        # Calculate total cost
        unit_price = self.get_station_buy_price(station_id, commodity_id, player)
        total_cost = unit_price * quantity

        # Check if player has enough credits
        if hasattr(player, "credits") and player.credits < total_cost:
            return {
                "success": False,
                "message": f"Not enough credits. Need {total_cost}, have {player.credits}.",
            }

        # Process transaction
        station["inventory"][commodity_id] -= quantity
        player.credits -= total_cost

        # Add to player inventory
        if hasattr(player, "inventory"):
            if commodity_id not in player.inventory:
                player.inventory[commodity_id] = 0
            player.inventory[commodity_id] += quantity

        return {
            "success": True,
            "message": f"Purchased {quantity} {commodity_id} for {total_cost} credits.",
            "quantity": quantity,
            "unit_price": unit_price,
            "total_cost": total_cost,
        }

    def sell_commodity(
        self, station_id: str, commodity_id: str, quantity: int, player
    ) -> Dict[str, Any]:
        """
        Player sells a commodity to a trading station.

        Args:
            station_id: ID of the trading station
            commodity_id: ID of the commodity to sell
            quantity: Amount to sell
            player: Player object making the sale

        Returns:
            Dict with transaction results
        """
        if station_id not in self.trading_stations:
            return {
                "success": False,
                "message": f"Trading station {station_id} not found",
            }

        station = self.trading_stations[station_id]

        # Check if player has enough inventory
        if not hasattr(player, "inventory") or commodity_id not in player.inventory:
            return {
                "success": False,
                "message": f"You don't have any {commodity_id} to sell.",
            }

        available = player.inventory.get(commodity_id, 0)
        if available < quantity:
            return {
                "success": False,
                "message": f"Not enough {commodity_id} in inventory. Only have {available}.",
            }

        # Calculate total payment
        unit_price = self.get_station_sell_price(station_id, commodity_id, player)
        total_payment = unit_price * quantity

        # Process transaction
        player.inventory[commodity_id] -= quantity
        if player.inventory[commodity_id] <= 0:
            del player.inventory[commodity_id]  # Remove empty entries

        player.credits += total_payment

        # Add to station inventory
        if commodity_id not in station["inventory"]:
            station["inventory"][commodity_id] = 0
        station["inventory"][commodity_id] += quantity

        # Update supply based on sales
        if commodity_id in self.commodities:
            # Selling commodities slightly increases supply
            current_supply = self.commodities[commodity_id]["supply"]
            # Small increase in supply, proportional to quantity sold
            supply_increase = min(0.05, quantity * 0.001)
            self.commodities[commodity_id]["supply"] = min(
                1.5, current_supply + supply_increase
            )

        return {
            "success": True,
            "message": f"Sold {quantity} {commodity_id} for {total_payment} credits.",
            "quantity": quantity,
            "unit_price": unit_price,
            "total_payment": total_payment,
        }

    def restock_stations(self, game_time: int) -> None:
        """
        Periodically restock trading stations with new inventory.

        Args:
            game_time: Current game time in ticks
        """
        for station_id, station in self.trading_stations.items():
            # Check if it's time to restock
            if game_time - station["last_restocked"] >= station["restock_interval"]:
                station["last_restocked"] = game_time

                # Restock each commodity
                for commodity_id in self.commodities:
                    current_stock = station["inventory"].get(commodity_id, 0)
                    # Add a random amount, more for common items, less for rare
                    if commodity_id == "common_minerals":
                        restock_amount = random.randint(15, 30)
                    elif commodity_id == "rare_minerals":
                        restock_amount = random.randint(5, 15)
                    elif commodity_id == "anomalous_materials":
                        restock_amount = random.randint(2, 8)
                    else:  # fuel_cells and ship_parts
                        restock_amount = random.randint(8, 20)

                    station["inventory"][commodity_id] = current_stock + restock_amount

                logging.info(f"Restocked trading station {station_id}")

    def generate_trading_quest(
        self, player_level: int, faction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a trading-related quest appropriate for the player's level.

        Args:
            player_level: Current level of the player
            faction_id: Optional faction to associate with the quest

        Returns:
            Dictionary containing quest details
        """
        # Scale quest difficulty with player level
        difficulty_multiplier = 0.8 + (
            player_level * 0.2
        )  # 1.0 at level 1, 1.8 at level 5

        # Select quest type based on player level
        quest_type = self._select_quest_type_for_level(player_level)

        # Select a commodity based on quest type and player level
        commodity_id = self._select_commodity_for_quest(quest_type, player_level)

        # Base reward scales with player level and commodity value
        base_value = self.commodities[commodity_id]["base_price"]

        # Generate quest based on type
        if quest_type == "delivery" or quest_type not in [
            "procurement",
            "market_manipulation",
            "rare_commodity",
        ]:
            return self._generate_delivery_quest(
                player_level,
                faction_id,
                commodity_id,
                base_value,
                difficulty_multiplier,
            )
        elif quest_type == "procurement":
            return self._generate_procurement_quest(
                player_level,
                faction_id,
                commodity_id,
                base_value,
                difficulty_multiplier,
            )
        elif quest_type == "market_manipulation":
            return self._generate_market_manipulation_quest(
                player_level,
                faction_id,
                commodity_id,
                base_value,
                difficulty_multiplier,
            )
        else:
            return self._generate_rare_commodity_quest(
                player_level,
                faction_id,
                commodity_id,
                base_value,
                difficulty_multiplier,
            )

    def _select_quest_type_for_level(self, player_level: int) -> str:
        """Select an appropriate quest type based on player level."""
        quest_types = ["delivery", "procurement"]
        if player_level >= 3:
            quest_types.append(
                "market_manipulation"
            )  # More complex quest for higher levels
        if player_level >= 4:
            quest_types.append(
                "rare_commodity"
            )  # Very challenging quest for high levels

        return random.choice(quest_types)

    def _select_commodity_for_quest(self, quest_type: str, player_level: int) -> str:
        """Select an appropriate commodity based on quest type and player level."""
        if quest_type == "rare_commodity" or (
            quest_type == "procurement" and player_level >= 4
        ):
            commodity_pool = ["anomalous_materials", "rare_minerals"]
        elif player_level >= 3:
            commodity_pool = ["rare_minerals", "fuel_cells", "ship_parts"]
        else:
            commodity_pool = ["common_minerals", "fuel_cells"]

        return random.choice(commodity_pool)

    def _select_destination_station(self, faction_id: Optional[str] = None) -> str:
        """Select an appropriate destination station, preferably of the given faction."""
        if faction_id:
            if faction_stations := [
                s_id
                for s_id, s in self.trading_stations.items()
                if s.get("faction_id") == faction_id
            ]:
                return random.choice(faction_stations)

        # If no faction stations or no faction specified, choose any station
        return random.choice(list(self.trading_stations.keys()))

    def _calculate_quantity_and_reward(
        self,
        base_value: int,
        difficulty_multiplier: float,
        quantity_base: int,
        quantity_multiplier: int,
        reward_percentage: float,
    ) -> Tuple[int, int]:
        """Calculate quantity and reward for a quest based on standard formula."""
        quantity = max(
            quantity_base,
            int(quantity_multiplier * difficulty_multiplier / (base_value / 10)),
        )
        reward = int(base_value * quantity * reward_percentage * difficulty_multiplier)
        return quantity, reward

    def _generate_delivery_quest(
        self,
        player_level: int,
        faction_id: Optional[str],
        commodity_id: str,
        base_value: int,
        difficulty_multiplier: float,
    ) -> Dict[str, Any]:
        """Generate a delivery quest: Transport goods from one station to another."""
        # Calculate quantity and reward
        quantity, reward = self._calculate_quantity_and_reward(
            base_value,
            difficulty_multiplier,
            5,
            20,
            0.3,  # 30% profit margin
        )

        # Select random source and destination stations
        source_id, dest_id = self._select_source_and_destination_stations()

        return {
            "type": "trading_delivery",
            "title": f"Deliver {commodity_id.replace('_', ' ')} to {dest_id}",
            "description": f"Transport {quantity} {commodity_id.replace('_', ' ')} from {source_id} to {dest_id}.",
            "commodity_id": commodity_id,
            "quantity": quantity,
            "source_station": source_id,
            "destination_station": dest_id,
            "reward": reward,
            "faction_id": faction_id,
            "difficulty": player_level,
            "completed": False,
        }

    def _select_source_and_destination_stations(self) -> Tuple[str, str]:
        """Select source and destination stations for delivery quests."""
        if len(self.trading_stations) >= 2:
            station_ids = list(self.trading_stations.keys())
            source_id = random.choice(station_ids)
            dest_id = random.choice([s for s in station_ids if s != source_id])
        else:
            # Fallback if not enough stations
            source_id = (
                "station_1"
                if "station_1" in self.trading_stations
                else list(self.trading_stations.keys())[0]
            )
            dest_id = "station_2"

        return source_id, dest_id

    def _generate_procurement_quest(
        self,
        player_level: int,
        faction_id: Optional[str],
        commodity_id: str,
        base_value: int,
        difficulty_multiplier: float,
    ) -> Dict[str, Any]:
        """Generate a procurement quest: Acquire specific goods for a faction."""
        # Calculate quantity and reward
        quantity, reward = self._calculate_quantity_and_reward(
            base_value,
            difficulty_multiplier,
            3,
            15,
            0.5,  # 50% profit margin
        )

        # Select destination station
        dest_id = self._select_destination_station(faction_id)

        return {
            "type": "trading_procurement",
            "title": f"Procure {commodity_id.replace('_', ' ')} for {dest_id}",
            "description": f"Acquire and deliver {quantity} {commodity_id.replace('_', ' ')} to {dest_id}.",
            "commodity_id": commodity_id,
            "quantity": quantity,
            "destination_station": dest_id,
            "reward": reward,
            "faction_id": faction_id,
            "difficulty": player_level,
            "completed": False,
        }

    def _generate_market_manipulation_quest(
        self,
        player_level: int,
        faction_id: Optional[str],
        commodity_id: str,
        base_value: int,
        difficulty_multiplier: float,
    ) -> Dict[str, Any]:
        """Generate a market manipulation quest: Buy low, sell high by exploiting market events."""
        # Calculate quantity and reward
        quantity, reward = self._calculate_quantity_and_reward(
            base_value,
            difficulty_multiplier,
            2,
            10,
            0.8,  # 80% profit margin
        )

        return {
            "type": "trading_market_manipulation",
            "title": f"Market opportunity: {commodity_id.replace('_', ' ')}",
            "description": f"Buy {quantity} {commodity_id.replace('_', ' ')} during a price drop and sell when prices rise.",
            "commodity_id": commodity_id,
            "quantity": quantity,
            "target_profit_margin": 0.3,  # Need to make at least 30% profit
            "reward": reward,
            "faction_id": faction_id,
            "difficulty": player_level,
            "completed": False,
            "buy_price_threshold": base_value
            * 0.7,  # Buy when price drops below 70% of base
        }

    def _generate_rare_commodity_quest(
        self,
        player_level: int,
        faction_id: Optional[str],
        commodity_id: str,
        base_value: int,
        difficulty_multiplier: float,
    ) -> Dict[str, Any]:
        """Generate a rare commodity quest: Find and deliver a very rare and valuable item."""
        # Calculate quantity and reward
        quantity, reward = self._calculate_quantity_and_reward(
            base_value,
            difficulty_multiplier,
            1,
            5,
            1.2,  # 120% profit margin
        )

        # Select destination station
        dest_id = self._select_destination_station(faction_id)

        return {
            "type": "trading_rare_commodity",
            "title": f"Acquire rare {commodity_id.replace('_', ' ')}",
            "description": f"Find and deliver {quantity} rare quality {commodity_id.replace('_', ' ')} to {dest_id}.",
            "commodity_id": commodity_id,
            "quantity": quantity,
            "destination_station": dest_id,
            "reward": reward,
            "faction_id": faction_id,
            "difficulty": player_level,
            "completed": False,
        }

    def check_quest_completion(self, quest: Dict[str, Any], player) -> bool:
        """
        Check if a trading quest has been completed.

        Args:
            quest: The quest to check
            player: Player object

        Returns:
            True if quest is completed, False otherwise
        """
        if quest["completed"]:
            return True

        quest_type = quest["type"]

        # Dispatch to appropriate handler based on quest type
        if quest_type in ["trading_delivery", "trading_procurement"]:
            return self._process_standard_delivery_quest_reward(player, quest)
        elif quest_type == "trading_market_manipulation":
            return self._process_market_manipulation_quest(player, quest)
        elif quest_type == "trading_rare_commodity":
            return self._process_rare_commodity_quest_reward(player, quest)

        return False

    def _process_market_manipulation_quest(self, player, quest: Dict[str, Any]) -> bool:
        """
        Process a market manipulation quest where player must buy low and sell high.

        Args:
            player: Player object
            quest: Quest data dictionary

        Returns:
            bool: True if quest completed successfully, False otherwise
        """
        # Check if player has trading history
        if not hasattr(player, "trading_history"):
            return False

        # Get relevant transactions
        buy_transactions, sell_transactions = self._get_commodity_transactions(
            player.trading_history, quest["commodity_id"]
        )

        # Check for successful trades meeting quest criteria
        return self._check_for_profitable_trades(
            buy_transactions, sell_transactions, quest, player
        )

    def _get_commodity_transactions(
        self, trading_history, commodity_id: str
    ) -> Tuple[list, list]:
        """
        Extract buy and sell transactions for a specific commodity.

        Args:
            trading_history: List of trading transactions
            commodity_id: ID of the commodity to filter for

        Returns:
            Tuple containing lists of buy and sell transactions
        """
        buy_transactions = [
            t
            for t in trading_history
            if t["type"] == "buy" and t["commodity_id"] == commodity_id
        ]

        sell_transactions = [
            t
            for t in trading_history
            if t["type"] == "sell" and t["commodity_id"] == commodity_id
        ]

        return buy_transactions, sell_transactions

    def _check_for_profitable_trades(
        self, buy_transactions, sell_transactions, quest, player
    ) -> bool:
        """
        Check if any buy-sell pair meets the profit margin requirements.

        Args:
            buy_transactions: List of buy transactions
            sell_transactions: List of sell transactions
            quest: Quest data dictionary
            player: Player object

        Returns:
            bool: True if a qualifying trade was found, False otherwise
        """
        for buy in buy_transactions:
            # Skip if buy price is above threshold
            if buy["unit_price"] > quest["buy_price_threshold"]:
                continue

            # Check each sell transaction that happened after this buy
            for sell in sell_transactions:
                if sell["transaction_time"] <= buy["transaction_time"]:
                    continue

                # Calculate profit margin
                profit_margin = (sell["unit_price"] - buy["unit_price"]) / buy[
                    "unit_price"
                ]

                # If profit margin meets or exceeds target, quest is complete
                if profit_margin >= quest["target_profit_margin"]:
                    return self._apply_quest_rewards_and_reputation(quest, player, 8, 3)

        return False

    def _process_rare_commodity_quest_reward(self, player, quest):
        # Check if player is at destination station with required rare goods
        if (
            not hasattr(player, "current_station")
            or player.current_station != quest["destination_station"]
        ):
            return False

        # For rare commodities, we check a special inventory section
        if (
            not hasattr(player, "rare_inventory")
            or quest["commodity_id"] not in player.rare_inventory
        ):
            return False

        if player.rare_inventory[quest["commodity_id"]] < quest["quantity"]:
            return False

        # Remove the delivered goods from player rare inventory
        player.rare_inventory[quest["commodity_id"]] -= quest["quantity"]
        if player.rare_inventory[quest["commodity_id"]] <= 0:
            del player.rare_inventory[quest["commodity_id"]]

        return self._apply_quest_rewards_and_reputation(quest, player, 10, 4)

    def _process_standard_delivery_quest_reward(self, player, quest):
        # Check if player is at destination station with required goods
        if (
            not hasattr(player, "current_station")
            or player.current_station != quest["destination_station"]
        ):
            return False

        if (
            not hasattr(player, "inventory")
            or quest["commodity_id"] not in player.inventory
        ):
            return False

        if player.inventory[quest["commodity_id"]] < quest["quantity"]:
            return False

        # Remove the delivered goods from player inventory
        player.inventory[quest["commodity_id"]] -= quest["quantity"]
        if player.inventory[quest["commodity_id"]] <= 0:
            del player.inventory[quest["commodity_id"]]

        if station := self.trading_stations.get(quest["destination_station"]):
            if quest["commodity_id"] not in station["inventory"]:
                station["inventory"][quest["commodity_id"]] = 0
            station["inventory"][quest["commodity_id"]] += quest["quantity"]

        return self._apply_quest_rewards_and_reputation(quest, player, 5, 2)

    def _apply_quest_rewards_and_reputation(
        self, quest, player, reputation_base, reputation_multiplier
    ):
        # Award credits and update reputation
        player.credits += quest["reward"]
        if hasattr(player, "update_faction_reputation") and quest["faction_id"]:
            reputation_gain = (
                reputation_base + quest["difficulty"] * reputation_multiplier
            )
            player.update_faction_reputation(quest["faction_id"], reputation_gain)

        return True
