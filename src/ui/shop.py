"""
Shop class: Handles the in-game shop for upgrades.

This module provides a flexible shop system with categories, collapsible UI,
and customizable upgrade options that affect different aspects of gameplay.
"""

# Standard library imports
import random
import time
from typing import Any, Dict, List, Tuple

# Third-party library imports
import pygame

# Local application imports
from config import (
    COLOR_RACE_1,
    COLOR_RACE_2,
    COLOR_RACE_3,
    COLOR_TEXT,
)
from config import COLOR_UI_BUTTON as COLOR_BUTTON
from config import COLOR_UI_BUTTON_HOVER as COLOR_BUTTON_HOVER
from config import (
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

from .draw_utils import draw_button, draw_panel, draw_text


# Forward references for type hints
class AsteroidField:
    """Type hint for AsteroidField class."""

    pass


class NotificationManager:
    """Type hint for NotificationManager class."""

    pass


class Player:
    """Type hint for Player class."""

    pass


class Shop:
    """
    The shop offers upgrade options that affect both your mining ship and the asteroid field.
    Features a category system, collapsible UI, and dynamic upgrade availability.
    """

    def __init__(self) -> None:
        """Initialize shop with default options and UI state."""
        # UI state
        self.expanded: bool = True  # Whether shop is expanded or collapsed
        self.current_category: str = "ship"  # Current category filter
        self.categories: List[str] = ["ship", "field", "race", "special"]
        self.scroll_offset: int = 0
        self.max_visible_items: int = 7
        self.selected_item: int = -1  # Currently selected item index
        self.hover_item: int = -1  # Item being hovered
        self.show_tooltip: bool = True
        self.animation_progress: float = 1.0  # Animation for expanding/collapsing (0-1)
        self.panel_alpha: int = 220  # Panel transparency

        # Shop positioning and sizing
        self.position_expanded: Tuple[int, int, int, int] = (
            WINDOW_WIDTH - 350,
            50,
            300,
            WINDOW_HEIGHT - 100,
        )
        self.position_collapsed: Tuple[int, int, int, int] = (
            WINDOW_WIDTH - 120,
            50,
            100,
            200,
        )

        # Shop appearance
        self.title_height: int = 40
        self.category_height: int = 30
        self.item_height: int = 80

        # Define upgrade options
        self.options: List[Dict[str, Any]] = []
        self._initialize_options()

        # Unlockable options (hidden until conditions met)
        self.hidden_options: List[Dict[str, Any]] = []
        self._initialize_hidden_options()

    def _initialize_options(self) -> None:
        """Initialize the standard shop upgrade options."""
        # Ship upgrades
        self.options.extend(
            [
                {
                    "id": "mining_efficiency",
                    "name": "Mining Efficiency",
                    "cost": 50,
                    "description": "Increase mining efficiency by 10%.\nEarn more resources from each asteroid.",
                    "action": lambda player, field: setattr(
                        player, "mining_efficiency", player.mining_efficiency + 0.1
                    ),
                    "category": "ship",
                    "icon": "⛏️",
                    "max_level": 10,
                    "current_level": 0,
                    "cost_multiplier": 1.5,
                },
                {
                    "id": "mining_range",
                    "name": "Mining Range",
                    "cost": 100,
                    "description": "Increase mining radius by 1 cell.\nMine more asteroids with each operation.",
                    "action": lambda player, field: setattr(
                        player, "mining_range", player.mining_range + 1
                    ),
                    "category": "ship",
                    "icon": "🔍",
                    "max_level": 5,
                    "current_level": 0,
                    "cost_multiplier": 2.0,
                },
                {
                    "id": "movement_speed",
                    "name": "Movement Speed",
                    "cost": 75,
                    "description": "Increase ship movement speed.\nTravel through the asteroid field faster.",
                    "action": lambda player, field: setattr(
                        player, "move_speed", player.move_speed + 1
                    ),
                    "category": "ship",
                    "icon": "🚀",
                    "max_level": 5,
                    "current_level": 0,
                    "cost_multiplier": 1.8,
                },
                {
                    "id": "auto_miner",
                    "name": "Auto-Miner",
                    "cost": 200,
                    "description": "Add an automatic mining drone.\nCollects resources without manual input.",
                    "action": lambda player, field: setattr(
                        player, "auto_miners", player.auto_miners + 1
                    ),
                    "category": "ship",
                    "icon": "🤖",
                    "max_level": 5,
                    "current_level": 0,
                    "cost_multiplier": 1.8,
                },
                {
                    "id": "mining_ship",
                    "name": "Mining Fleet Ship",
                    "cost": 500,
                    "description": "Add another ship to your mining fleet.\nEach ship mines independently.",
                    "action": lambda player, field: player.add_ship(),
                    "category": "ship",
                    "icon": "🛸",
                    "max_level": 5,
                    "current_level": 0,
                    "cost_multiplier": 1.5,
                    "cost_function": lambda player: player.ship_cost,
                },
            ]
        )

        # Field upgrades
        self.options.extend(
            [
                {
                    "id": "asteroid_regen",
                    "name": "Asteroid Generation",
                    "cost": 150,
                    "description": "Increase asteroid regeneration rate.\nMore asteroids will spawn over time.",
                    "action": lambda player, field: setattr(
                        field, "regen_rate", field.regen_rate + 0.01
                    ),
                    "category": "field",
                    "icon": "🌑",
                    "max_level": 10,
                    "current_level": 0,
                    "cost_multiplier": 1.4,
                },
                {
                    "id": "rare_chance",
                    "name": "Rare Mineral Chance",
                    "cost": 200,
                    "description": "Increase chance of rare mineral spawns.\nRare minerals are worth 3x more.",
                    "action": lambda player, field: setattr(
                        field, "rare_chance", field.rare_chance + 0.05
                    ),
                    "category": "field",
                    "icon": "💎",
                    "max_level": 10,
                    "current_level": 0,
                    "cost_multiplier": 1.6,
                },
                {
                    "id": "energy_level",
                    "name": "Field Energy",
                    "cost": 100,
                    "description": "Increase asteroid field energy level.\nHigher energy leads to more complex structures.",
                    "action": lambda player, field: setattr(
                        field, "energy_grid", field.energy_grid + 0.5
                    ),
                    "category": "field",
                    "icon": "⚡",
                    "max_level": 10,
                    "current_level": 0,
                    "cost_multiplier": 1.5,
                },
                {
                    "id": "rare_bonus",
                    "name": "Rare Mineral Value",
                    "cost": 250,
                    "description": "Increase value of rare minerals.\nRare minerals provide more resources.",
                    "action": lambda player, field: setattr(
                        field,
                        "rare_bonus_multiplier",
                        field.rare_bonus_multiplier + 0.5,
                    ),
                    "category": "field",
                    "icon": "🔮",
                    "max_level": 5,
                    "current_level": 0,
                    "cost_multiplier": 1.8,
                },
            ]
        )

        # Race upgrades
        self.options.extend(
            [
                {
                    "id": "discover_race",
                    "name": "Discover New Race",
                    "cost": 300,
                    "description": "Discover a new symbiotic race.\nEach race has unique traits and behaviors.",
                    "action": lambda player, field, notifier: self.discover_race(
                        field, notifier
                    ),
                    "category": "race",
                    "icon": "👽",
                    "max_level": 3,
                    "current_level": 0,
                    "cost_multiplier": 3.0,
                    "requires": lambda field: len(field.races) < 3,
                },
                {
                    "id": "feed_amount_small",
                    "name": "Small Feeding",
                    "cost": 50,
                    "description": "Feed a small amount to symbiotic races.\nReduces race aggression temporarily.",
                    "action": lambda player, field, notifier: self._feed_races(
                        player, field, 50, notifier
                    ),
                    "category": "race",
                    "icon": "🍽️",
                    "max_level": 0,  # No level limit
                    "current_level": 0,
                    "cost_multiplier": 1.0,
                },
                {
                    "id": "feed_amount_medium",
                    "name": "Medium Feeding",
                    "cost": 150,
                    "description": "Feed a moderate amount to symbiotic races.\nReduces race aggression significantly.",
                    "action": lambda player, field, notifier: self._feed_races(
                        player, field, 150, notifier
                    ),
                    "category": "race",
                    "icon": "🍲",
                    "max_level": 0,  # No level limit
                    "current_level": 0,
                    "cost_multiplier": 1.0,
                },
                {
                    "id": "feed_amount_large",
                    "name": "Large Feeding",
                    "cost": 500,
                    "description": "Feed a large amount to symbiotic races.\nGreatly reduces race aggression and may trigger evolution.",
                    "action": lambda player, field, notifier: self._feed_races(
                        player, field, 500, notifier
                    ),
                    "category": "race",
                    "icon": "🍱",
                    "max_level": 0,  # No level limit
                    "current_level": 0,
                    "cost_multiplier": 1.0,
                },
            ]
        )

        # Special upgrades
        self.options.extend(
            [
                {
                    "id": "auto_upgrade",
                    "name": "Auto-Upgrade System",
                    "cost": 1000,
                    "description": "Enables automatic upgrades for your ship.\nPeriodically selects and purchases upgrades.",
                    "action": lambda player, field: setattr(
                        player, "auto_upgrade", True
                    ),
                    "category": "special",
                    "icon": "🔄",
                    "max_level": 1,
                    "current_level": 0,
                    "cost_multiplier": 1.0,
                    "requires": lambda player, field: not player.auto_upgrade,
                },
            ]
        )

    def _initialize_hidden_options(self) -> None:
        """Initialize hidden options that unlock based on game progression."""
        self.hidden_options = [
            {
                "id": "anomaly_detector",
                "name": "Anomaly Detector",
                "cost": 800,
                "description": "Detects spatial anomalies in the asteroid field.\nReveals hidden resources and phenomena.",
                "action": lambda player, field, notifier: self._unlock_anomaly_detection(
                    player, field, notifier
                ),
                "category": "special",
                "icon": "📡",
                "max_level": 1,
                "current_level": 0,
                "unlock_condition": lambda player, field: player.total_mined >= 5000,
                "unlock_message": "Anomaly Detector technology now available!",
            },
            {
                "id": "symbiote_communication",
                "name": "Symbiote Communication",
                "cost": 1200,
                "description": "Establish direct communication with symbiotic races.\nUnlocks trading and special missions.",
                "action": lambda player, field, notifier: self._unlock_symbiote_communication(
                    player, field, notifier
                ),
                "category": "race",
                "icon": "🔊",
                "max_level": 1,
                "current_level": 0,
                "unlock_condition": lambda player, field: len(field.races) >= 2
                and any(race.evolution_stage >= 2 for race in field.races),
                "unlock_message": "Symbiote Communication technology now available!",
            },
        ]

    def check_unlockable_options(
        self, player: Player, field: AsteroidField, notifier: NotificationManager
    ) -> List[Dict[str, Any]]:
        """
        Check if any hidden options can be unlocked based on game progress.

        Args:
            player: Player object
            field: AsteroidField object
            notifier: NotificationManager for notifications
        """
        newly_unlocked = []

        for option in list(self.hidden_options):
            if "unlock_condition" in option and option["unlock_condition"](
                player, field
            ):
                # Move from hidden to available
                self.options.append(option)
                self.hidden_options.remove(option)
                newly_unlocked.append(option)

                # Notify player
                if "unlock_message" in option and notifier:
                    notifier.add(
                        option["unlock_message"],
                        color=(255, 215, 0),
                        category="upgrade",
                        importance=2,
                    )

        return newly_unlocked

    def get_filtered_options(self, player: Player) -> List[Dict[str, Any]]:
        """
        Get options filtered by the current category.

        Args:
            player: Player object to check if options are affordable

        Returns:
            list: Filtered and processed shop options
        """
        filtered_options = [
            option
            for option in self.options
            if option["category"] == self.current_category
        ]

        # Sort by affordability and level
        filtered_options.sort(
            key=lambda x: (
                player.currency
                < self._get_option_cost(x, player),  # Affordable options first
                (
                    x["current_level"] >= x["max_level"]
                    if x["max_level"] > 0
                    else False
                ),  # Available levels first
                x["current_level"],  # Lower levels first
                x["cost"],  # Lower cost first
            )
        )

        return filtered_options

    def _get_option_cost(self, option: Dict[str, Any], player: Player) -> int:
        """
        Calculate the cost of an option considering current level.

        Args:
            option: Shop option dictionary
            player: Player object for custom cost functions

        Returns:
            int: Current cost of the option
        """
        # Check if option has a custom cost function
        if "cost_function" in option:
            return option["cost_function"](player)

        # Otherwise calculate based on level and multiplier
        base_cost = option["cost"]
        if option["current_level"] > 0:
            multiplier = option["cost_multiplier"] ** option["current_level"]
            return int(base_cost * multiplier)
        return base_cost

    def _feed_races(
        self,
        player: Player,
        field: AsteroidField,
        amount: int,
        notifier: NotificationManager,
    ) -> None:
        """
        Feed minerals to symbiotic races to reduce aggression.

        Args:
            player: Player object
            field: AsteroidField object
            amount: Amount to feed
            notifier: NotificationManager for notifications
        """
        if player.currency < amount:
            return

        # Store and utilize feeding results for notification and tracking
        feeding_results = player.feed_symbiotes(field, amount)
        
        # Process feeding results and send notification
        self._process_feeding_results(feeding_results, amount, notifier)
        
        # Check for and process any race evolutions
        self._check_for_race_evolution(field, notifier)

    def _process_feeding_results(self, feeding_results: Dict[str, Any], amount: int, notifier: NotificationManager) -> None:
        """
        Process feeding results and send appropriate notification.
        
        Args:
            feeding_results: Results from feeding operation
            amount: Amount of minerals fed
            notifier: NotificationManager for sending notifications
        """
        # Add detailed notification based on feeding results
        if feeding_results and isinstance(feeding_results, dict):
            # Extract feeding effectiveness if available
            effectiveness = feeding_results.get('effectiveness', 0)
            races_fed = feeding_results.get('races_fed', 0)
            
            # Add more detailed notification
            notifier.add(
                f"Fed {amount} minerals to {races_fed} symbiotic races (Effectiveness: {effectiveness:.1f}).", 
                category="race"
            )
            
            # Track feeding statistics for future enhancement
            self._track_feeding_statistics(feeding_results)
        else:
            # Simple notification if results format is different
            notifier.add(f"Fed {amount} minerals to symbiotic races.", category="race")
    
    def _check_for_race_evolution(self, field: AsteroidField, notifier: NotificationManager) -> None:
        """
        Check if any races in the field have evolved and process evolutions.
        
        Args:
            field: AsteroidField containing races
            notifier: NotificationManager for sending notifications
        """
        # Check if feeding triggered any evolution
        for race in field.races:
            if (
                race.fed_this_turn
                and race.hunger < 0.3
                and race.evolution_points >= race.evolution_threshold
            ):
                self._process_race_evolution(race, notifier)
    
    def _process_race_evolution(self, race: Any, notifier: NotificationManager) -> None:
        """
        Process a single race evolution and send notifications.
        
        Args:
            race: Race object that evolved
            notifier: NotificationManager for sending notifications
        """
        # Get evolution metrics and utilize them for enhanced notifications
        evolution_metrics = race.evolve()
        
        # Track evolution metrics for analysis and future visualizations
        self._track_evolution_metrics(race.race_id, evolution_metrics)
        
        # Create detailed evolution notification with metrics data
        evolution_message = f"Race {race.race_id} evolved to stage {race.evolution_stage}!"
        
        # Add detailed trait changes if available in metrics
        if evolution_metrics and isinstance(evolution_metrics, dict) and (trait_changes := evolution_metrics.get('trait_changes', {})):
            traits_msg = ', '.join([f"{trait}: {change:+.2f}" for trait, change in trait_changes.items()])
            evolution_message += f"\nTrait changes: {traits_msg}"
        
        notifier.add(
            evolution_message,
            color=(255, 100, 255),
            category="race",
            importance=3,
        )
    
    def _track_evolution_metrics(self, race_id: int, evolution_metrics: Dict[str, Any]) -> None:
        """
        Track and analyze symbiote evolution metrics for future visualization and analysis.
        
        Args:
            race_id: ID of the race that evolved
            evolution_metrics: Dictionary containing evolution metrics and trait changes
        """
        # Initialize evolution_data dictionary if it doesn't exist yet
        if not hasattr(self, "evolution_data"):
            self.evolution_data = {
                "total_evolutions": 0,
                "race_evolutions": {},
                "evolution_history": [],
            }
        
        # Update global evolution statistics
        self.evolution_data["total_evolutions"] += 1
        
        # Add to evolution history for timeline visualization
        self.evolution_data["evolution_history"].append({
            "timestamp": time.time(),
            "race_id": race_id,
            "metrics": evolution_metrics
        })
        
        # Update race-specific evolution data
        if race_id not in self.evolution_data["race_evolutions"]:
            self.evolution_data["race_evolutions"][race_id] = {
                "evolution_count": 0,
                "evolution_stages": [],
                "trait_history": {}
            }
        
        race_data = self.evolution_data["race_evolutions"][race_id]
        race_data["evolution_count"] += 1
        
        # Track evolution stage if available
        if evolution_metrics and isinstance(evolution_metrics, dict):
            if "stage" in evolution_metrics:
                race_data["evolution_stages"].append(evolution_metrics["stage"])
            
            # Track trait changes over time
            trait_changes = evolution_metrics.get("trait_changes", {})
            for trait, change in trait_changes.items():
                if trait not in race_data["trait_history"]:
                    race_data["trait_history"][trait] = []
                
                race_data["trait_history"][trait].append({
                    "timestamp": time.time(),
                    "change": change,
                    "absolute_value": evolution_metrics.get("traits", {}).get(trait)
                })
    
    def _track_feeding_statistics(self, feeding_results: Dict[str, Any]) -> None:
        """
        Track and analyze symbiote feeding statistics for future metrics.
        
        Args:
            feeding_results: Dictionary containing feeding operation results
        """
        # Initialize feeding_stats dictionary if it doesn't exist yet
        if not hasattr(self, "feeding_stats"):
            self.feeding_stats = {
                "total_feedings": 0,
                "total_minerals_fed": 0,
                "race_stats": {},
                "effectiveness_history": [],
            }
        
        # Update statistics
        self.feeding_stats["total_feedings"] += 1
        self.feeding_stats["total_minerals_fed"] += feeding_results.get("amount", 0)
        
        # Track effectiveness over time for potential UI graphs
        self.feeding_stats["effectiveness_history"].append({
            "timestamp": time.time(),
            "effectiveness": feeding_results.get("effectiveness", 0),
            "races_fed": feeding_results.get("races_fed", 0)
        })
        
        # Update per-race statistics
        for race_id, race_data in feeding_results.get("race_details", {}).items():
            if race_id not in self.feeding_stats["race_stats"]:
                self.feeding_stats["race_stats"][race_id] = {
                    "feedings": 0,
                    "minerals_fed": 0,
                    "evolutions_triggered": 0
                }
            
            race_stats = self.feeding_stats["race_stats"][race_id]
            race_stats["feedings"] += 1
            race_stats["minerals_fed"] += race_data.get("amount", 0)
            
            # Track if this feeding triggered an evolution
            if race_data.get("evolution_triggered", False):
                race_stats["evolutions_triggered"] += 1
    
    def discover_race(
        self, field: AsteroidField, notifier: NotificationManager
    ) -> bool:
        """
        Discover a new symbiotic race if slots are available.

        Args:
            field: AsteroidField object
            notifier: NotificationManager for notifications

        Returns:
            bool: True if a race was discovered, False otherwise
        """
        # Check if we're at max races
        if len(field.races) >= 3:
            notifier.add("No more undiscovered races in this sector.", category="race")
            return False

        # Check which race IDs are available
        used_ids = {race.race_id for race in field.races}
        available_ids = [id for id in [1, 2, 3] if id not in used_ids]

        if not available_ids:
            return False

        race_id = available_ids[0]
        race_colors = {1: COLOR_RACE_1, 2: COLOR_RACE_2, 3: COLOR_RACE_3}

        # Create new race with random traits
        birth_set = {random.randint(1, 8) for _ in range(random.randint(1, 3))}
        survival_set = {random.randint(1, 8) for _ in range(random.randint(1, 3))}

        from entities.miner_entity import MinerEntity

        new_race = MinerEntity(
            race_id=race_id,
            color=race_colors[race_id],
            birth_set=birth_set,
            survival_set=survival_set,
            initial_density=0.005,
        )

        # Add race to field
        field.races.append(new_race)
        new_race.populate(field)

        # Add notification
        notifier.add(
            f"Discovered new symbiotic race (ID: {race_id}) with trait: {new_race.trait}!",
            color=race_colors[race_id],
            duration=600,
            category="race",
            importance=3,
        )

        # Update shop option level
        for option in self.options:
            if option["id"] == "discover_race":
                option["current_level"] += 1
                break

        return True

    def purchase_upgrade(
        self,
        option: Dict[str, Any],
        player: Player,
        field: AsteroidField,
        notifier: NotificationManager,
    ) -> bool:
        """
        Purchase an upgrade if player can afford it.

        Args:
            option: Shop option dictionary
            player: Player object
            field: AsteroidField object
            notifier: NotificationManager for notifications

        Returns:
            bool: True if purchase was successful, False otherwise
        """
        # Check if upgrade can be purchased
        cost = self._get_option_cost(option, player)

        # Check max level
        if option["max_level"] > 0 and option["current_level"] >= option["max_level"]:
            notifier.add(
                f"Maximum level reached for {option['name']}.", category="upgrade"
            )
            return False

        # Check requirements
        if (
            "requires" in option
            and callable(option["requires"])
            and not option["requires"](player, field)
        ):
            notifier.add(
                f"Requirements not met for {option['name']}.",
                category="upgrade",
            )
            return False

        # Check if player can afford
        if player.currency < cost:
            notifier.add(
                f"Not enough resources for {option['name']}. Need {cost - player.currency} more.",
                category="upgrade",
            )
            return False

        # Purchase upgrade
        player.currency -= cost

        # Apply upgrade effects
        if "action" in option:
            if len(option["action"].__code__.co_varnames) == 3:
                option["action"](player, field, notifier)
            else:
                option["action"](player, field)

        # Increase level if needed
        if option["max_level"] != 0:
            option["current_level"] += 1

        # Notification
        notifier.add(
            f"Purchased {option['name']} for {cost} resources.",
            color=(0, 255, 100),
            category="upgrade",
        )

        return True

    def _unlock_anomaly_detection(
        self, player: Player, field: AsteroidField, notifier: NotificationManager
    ) -> None:
        """Unlock anomaly detection feature."""
        # Set field anomaly detection enabled
        field.anomaly_detection_enabled = True

        # Generate some initial anomalies
        field.generate_anomalies(5)

        # Notify player
        notifier.add(
            "Anomaly detector online! Strange energy signatures detected.",
            color=(0, 255, 255),
            duration=600,
            category="upgrade",
            importance=3,
        )

    def _unlock_symbiote_communication(
        self, player: Player, field: AsteroidField, notifier: NotificationManager
    ) -> None:
        """Unlock symbiote communication feature."""
        # Enable communication
        field.symbiote_communication_enabled = True

        # Let races know they can communicate
        for race in field.races:
            if hasattr(race, "communication_enabled"):
                race.communication_enabled = True

        # Notify player
        notifier.add(
            "Symbiote communication established! New interactions available.",
            color=(200, 100, 255),
            duration=600,
            category="upgrade",
            importance=3,
        )

    def toggle_expanded(self) -> None:
        """Toggle between expanded and collapsed shop view."""
        self.expanded = not self.expanded
        self.animation_progress = 0.0  # Start animation

    def update_animation(self, delta_time: float) -> None:
        """
        Update animation progress for expanding/collapsing.

        Args:
            delta_time: Time elapsed since last frame in seconds
        """
        if self.animation_progress < 1.0:
            # Animation speed (complete in 0.3 seconds)
            animation_speed = 3.0
            self.animation_progress = min(
                1.0, self.animation_progress + delta_time * animation_speed
            )

    def get_current_panel_rect(self) -> pygame.Rect:
        """
        Get the current panel rectangle based on animation state.

        Returns:
            pygame.Rect: Current panel rectangle
        """
        if self.animation_progress >= 1.0:
            return (
                pygame.Rect(self.position_expanded)
                if self.expanded
                else pygame.Rect(self.position_collapsed)
            )
        # Interpolate between collapsed and expanded positions
        progress = self.animation_progress
        if not self.expanded:
            progress = 1.0 - progress

        x1, y1, w1, h1 = self.position_collapsed
        x2, y2, w2, h2 = self.position_expanded

        x = int(x1 + (x2 - x1) * progress)
        y = int(y1 + (y2 - y1) * progress)
        w = int(w1 + (w2 - w1) * progress)
        h = int(h1 + (h2 - h1) * progress)

        return pygame.Rect(x, y, w, h)

    def _draw_panel_and_header(
        self, surface: pygame.Surface, panel_rect: pygame.Rect
    ) -> None:
        """Draw the shop panel background and header."""
        # Draw panel background
        draw_panel(
            surface,
            panel_rect,
            color=(30, 30, 40, self.panel_alpha),
            border_color=(100, 100, 150, self.panel_alpha),
            border_width=2,
            header="Upgrade Shop",
            header_height=self.title_height,
            header_color=(40, 40, 60, self.panel_alpha),
        )

        # Draw expand/collapse button
        toggle_button_rect = pygame.Rect(
            panel_rect.x + panel_rect.width - 30, panel_rect.y + 5, 20, 20
        )
        draw_button(
            surface, toggle_button_rect, "▼" if self.expanded else "▶", font_size=14
        )

    def _draw_player_stats(
        self, surface: pygame.Surface, panel_rect: pygame.Rect, player: Player
    ) -> None:
        """Draw player's currency and stats in collapsed mode."""
        # Draw player's currency
        currency_text = f"Resources: {player.currency}"
        draw_text(
            surface,
            currency_text,
            panel_rect.x + 10,
            panel_rect.y + self.title_height + 5,
            16,
            COLOR_TEXT,
        )

        # If collapsed, only show minimal info
        if not self.expanded and self.animation_progress >= 1.0:
            # Show a few stats/options in collapsed mode
            y_offset = panel_rect.y + self.title_height + 30

            # Show most important player stats
            stats = [
                f"Mining Eff: {player.mining_efficiency:.1f}",
                f"Ships: {player.mining_ships}",
                f"Range: {player.mining_range}",
                f"Speed: {player.move_speed}",
            ]

            for stat in stats:
                draw_text(surface, stat, panel_rect.x + 10, y_offset, 14, COLOR_TEXT)
                y_offset += 20

    def _draw_category_tabs(
        self, surface: pygame.Surface, panel_rect: pygame.Rect
    ) -> None:
        """Draw the category tabs."""
        tab_width = panel_rect.width / len(self.categories)
        for i, category in enumerate(self.categories):
            tab_rect = pygame.Rect(
                panel_rect.x + i * tab_width,
                panel_rect.y + self.title_height + 30,
                tab_width,
                self.category_height,
            )

            # Draw tab background
            color = (60, 60, 80) if category == self.current_category else (40, 40, 50)
            pygame.draw.rect(surface, color, tab_rect)

            # Draw tab border
            pygame.draw.rect(surface, (80, 80, 100), tab_rect, 1)

            # Draw tab text
            draw_text(
                surface,
                category.capitalize(),
                tab_rect.x + tab_width // 2,
                tab_rect.y + 5,
                14,
                COLOR_TEXT,
                align="center",
            )

    def _draw_scrollbar(
        self,
        surface: pygame.Surface,
        panel_rect: pygame.Rect,
        content_y: int,
        content_height: int,
        filtered_options: List[Dict[str, Any]],
    ) -> None:
        """Draw scrollbar if needed."""
        if len(filtered_options) <= self.max_visible_items:
            return

        scrollbar_height = content_height
        visible_ratio = self.max_visible_items / len(filtered_options)
        thumb_height = max(30, scrollbar_height * visible_ratio)

        # Calculate thumb position
        max_scroll = len(filtered_options) - self.max_visible_items
        scroll_ratio = self.scroll_offset / max_scroll if max_scroll > 0 else 0
        thumb_y = content_y + scroll_ratio * (scrollbar_height - thumb_height)

        # Draw track
        scrollbar_x = panel_rect.x + panel_rect.width - 15
        pygame.draw.rect(
            surface,
            (40, 40, 50),
            (scrollbar_x, content_y, 10, scrollbar_height),
            border_radius=5,
        )

        # Draw thumb
        pygame.draw.rect(
            surface,
            (80, 80, 100),
            (scrollbar_x, thumb_y, 10, thumb_height),
            border_radius=5,
        )

    def _get_item_colors(
        self, option: Dict[str, Any], player: Player
    ) -> Tuple[tuple, tuple, tuple]:
        """Get colors for item based on state (affordable, max level, requirements)."""
        # Get cost and check if affordable
        cost = self._get_option_cost(option, player)
        can_afford = player.currency >= cost

        # Check if max level reached
        max_level_reached = (
            option["max_level"] > 0 and option["current_level"] >= option["max_level"]
        )

        # Get requirements status
        requirements_met = True
        if "requires" in option and callable(option["requires"]):
            field = player.field if hasattr(player, "field") else None
            if len(option["requires"].__code__.co_varnames) == 1:
                requirements_met = option["requires"](player)
            else:
                requirements_met = option["requires"](player, field)

        # Determine colors based on state
        bg_color = (40, 40, 55)  # Default background

        if can_afford and not max_level_reached and requirements_met:
            border_color = (100, 200, 100)
            text_color = COLOR_TEXT
        elif max_level_reached or not requirements_met:
            border_color = (200, 100, 100)
            text_color = (200, 100, 100)
        else:
            border_color = (150, 150, 150)

            text_color = (150, 150, 150)

        return bg_color, border_color, text_color

    def _draw_item_icon(
        self, surface: pygame.Surface, item_rect: pygame.Rect, option: Dict[str, Any]
    ) -> None:
        """Draw the icon for a shop item."""
        icon_size = 24
        icon_rect = pygame.Rect(
            item_rect.x + 10,
            item_rect.y + (item_rect.height - icon_size) // 2,
            icon_size,
            icon_size,
        )

        # Draw icon background
        pygame.draw.rect(surface, (60, 60, 80), icon_rect, border_radius=5)

        # Draw icon text
        draw_text(
            surface,
            option["icon"],
            icon_rect.x + icon_rect.width // 2,
            icon_rect.y + 2,
            20,
            COLOR_TEXT,
            align="center",
        )

    def _draw_selection_arrow(
        self, surface: pygame.Surface, item_rect: pygame.Rect
    ) -> None:
        """Draw the selection arrow for a selected item."""
        arrow_size = 20
        arrow_rect = pygame.Rect(
            item_rect.x + item_rect.width - 10 - arrow_size,
            item_rect.y + (item_rect.height - arrow_size) // 2,
            arrow_size,
            arrow_size,
        )

        pygame.draw.polygon(
            surface,
            COLOR_TEXT,
            [
                (arrow_rect.x + arrow_size // 2, arrow_rect.y),
                (arrow_rect.x + arrow_size, arrow_rect.y + arrow_size // 2),
                (arrow_rect.x + arrow_size // 2, arrow_rect.y + arrow_size),
            ],
        )

    def _draw_shop_item(
        self,
        surface: pygame.Surface,
        item_rect: pygame.Rect,
        option: Dict[str, Any],
        player: Player,
        is_selected: bool,
        is_hovered: bool,
    ) -> None:
        """Draw a single shop item."""
        # Get colors based on item state
        default_bg, border_color, text_color = self._get_item_colors(option, player)

        # Background color based on selection/hover state
        if is_selected:
            bg_color = (60, 60, 80)
        elif is_hovered:
            bg_color = (50, 50, 70)
        else:
            bg_color = default_bg

        # Draw item background
        pygame.draw.rect(surface, bg_color, item_rect, border_radius=5)

        # Draw item border
        pygame.draw.rect(surface, border_color, item_rect, 2, border_radius=5)

        # Draw item icon if present
        if "icon" in option:
            self._draw_item_icon(surface, item_rect, option)

        # Get cost for display
        cost = self._get_option_cost(option, player)

        # Draw item name
        draw_text(
            surface,
            option["name"],
            item_rect.x + 50,
            item_rect.y + 10,
            20,
            text_color,
            align="left",
        )

        # Draw item cost
        draw_text(
            surface,
            f"Cost: {cost}",
            item_rect.x + item_rect.width - 100,
            item_rect.y + 10,
            20,
            COLOR_TEXT,
            align="right",
        )

        # Draw item description
        draw_text(
            surface,
            option["description"],
            item_rect.x + 10,
            item_rect.y + self.item_height - 10,
            15,
            text_color,
        )

        # Draw item level
        draw_text(
            surface,
            f"Level: {option['current_level']}",
            item_rect.x + item_rect.width - 100,
            item_rect.y + self.item_height - 10,
            15,
            text_color,
            align="right",
        )

        # Draw selection arrow if selected
        if is_selected:
            self._draw_selection_arrow(surface, item_rect)

    def _draw_buttons_and_summary(
        self, surface: pygame.Surface, panel_rect: pygame.Rect
    ) -> None:
        """Draw buttons and summary information."""
        # Draw expand/collapse button
        button_rect = pygame.Rect(
            panel_rect.x + 10,
            panel_rect.y + panel_rect.height - 30,
            panel_rect.width - 20,
            25,
        )

        button_text = "Collapse" if self.expanded else "Expand"
        draw_button(surface, button_text, button_rect, COLOR_BUTTON, COLOR_BUTTON_HOVER)

        # Draw category buttons
        for category in self.categories:
            button_rect = pygame.Rect(panel_rect.x + 10, panel_rect.y + 10, 100, 25)
            button_rect.x += 110 * self.categories.index(category)

            draw_button(
                surface,
                category.capitalize(),
                button_rect,
                COLOR_BUTTON,
                COLOR_BUTTON_HOVER,
            )

        # Draw category filter indicator
        indicator_rect = pygame.Rect(panel_rect.x + 10, panel_rect.y + 10, 100, 25)
        indicator_rect.x += 110 * self.categories.index(self.current_category)
        pygame.draw.rect(surface, COLOR_TEXT, indicator_rect, 2)

        # Draw close button
        close_rect = pygame.Rect(
            panel_rect.x + panel_rect.width - 30, panel_rect.y + 10, 25, 25
        )
        draw_button(surface, "X", close_rect, COLOR_BUTTON, COLOR_BUTTON_HOVER)

        # Draw help button
        help_rect = pygame.Rect(
            panel_rect.x + panel_rect.width - 60, panel_rect.y + 10, 25, 25
        )
        draw_button(surface, "?", help_rect, COLOR_BUTTON, COLOR_BUTTON_HOVER)

        # Draw title
        draw_text(
            surface,
            "Shop",
            panel_rect.x + panel_rect.width // 2,
            panel_rect.y + 10,
            25,
            COLOR_TEXT,
        )

    def _draw_summary_stats(
        self, surface: pygame.Surface, panel_rect: pygame.Rect
    ) -> None:
        """Draw summary statistics at the bottom of the shop."""
        # Draw total cost
        total_cost = sum(option["cost"] for option in self.options)
        draw_text(
            surface,
            f"Total cost: {total_cost}",
            panel_rect.x + 10,
            panel_rect.y + panel_rect.height - 10,
            20,
            COLOR_TEXT,
        )

        # Draw total level
        total_level = sum(option["current_level"] for option in self.options)
        draw_text(
            surface,
            f"Total level: {total_level}",
            panel_rect.x + panel_rect.width - 200,
            panel_rect.y + panel_rect.height - 10,
            20,
            COLOR_TEXT,
        )

    def _draw_tooltip(
        self, surface: pygame.Surface, name: str, description: str, level: int
    ) -> None:
        """Draw tooltip for selected item."""
        # Create tooltip rectangle
        tooltip_width = 300
        tooltip_height = 150
        tooltip_x = pygame.mouse.get_pos()[0] + 20
        tooltip_y = pygame.mouse.get_pos()[1] + 20

        # Ensure tooltip stays on screen
        if tooltip_x + tooltip_width > WINDOW_WIDTH:
            tooltip_x = WINDOW_WIDTH - tooltip_width - 10
        if tooltip_y + tooltip_height > WINDOW_HEIGHT:
            tooltip_y = WINDOW_HEIGHT - tooltip_height - 10

        tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_width, tooltip_height)

        # Draw tooltip background
        pygame.draw.rect(surface, (30, 30, 40), tooltip_rect, border_radius=5)
        pygame.draw.rect(surface, (80, 80, 100), tooltip_rect, 2, border_radius=5)

        # Draw tooltip content
        draw_text(
            surface, name, tooltip_rect.x + 10, tooltip_rect.y + 10, 20, COLOR_TEXT
        )
        draw_text(
            surface,
            f"Level: {level}",
            tooltip_rect.x + tooltip_width - 10,
            tooltip_rect.y + 10,
            16,
            COLOR_TEXT,
            align="right",
        )

        # Draw description with word wrap
        words = description.split()
        line = ""
        y_offset = tooltip_rect.y + 40
        for word in words:
            test_line = f"{line}{word} "
            if len(test_line) * 8 > tooltip_width - 20:  # Approximate character width
                draw_text(
                    surface,
                    line,
                    tooltip_rect.x + 10,
                    y_offset,
                    14,
                    COLOR_TEXT,
                    align="left",
                )
                y_offset += 20
                line = f"{word} "
            else:
                line = test_line
        if line:
            draw_text(
                surface,
                line,
                tooltip_rect.x + 10,
                y_offset,
                14,
                COLOR_TEXT,
                align="left",
            )

    def draw(self, surface: pygame.Surface, player: Player) -> None:
        """
        Draw the shop interface on the given surface.

        Args:
            surface: Pygame surface to draw on
            player: Player object to check affordability
        """
        # Get panel dimensions based on animation state
        panel_rect = self.get_current_panel_rect()

        # Draw panel and header
        self._draw_panel_and_header(surface, panel_rect)

        # Draw player stats
        self._draw_player_stats(surface, panel_rect, player)

        # If collapsed, return early
        if not self.expanded and self.animation_progress >= 1.0:
            return

        # In expanded view, draw category tabs
        self._draw_category_tabs(surface, panel_rect)

        # Draw shop items
        filtered_options = self.get_filtered_options(player)
        visible_count = min(self.max_visible_items, len(filtered_options))

        content_y = panel_rect.y + self.title_height + 30 + self.category_height + 10
        content_height = panel_rect.height - (content_y - panel_rect.y) - 20

        # Draw scrollbar if needed
        self._draw_scrollbar(
            surface, panel_rect, content_y, content_height, filtered_options
        )

        # Draw items
        for i in range(visible_count):
            item_idx = i + self.scroll_offset
            if item_idx >= len(filtered_options):
                break

            option = filtered_options[item_idx]
            item_y = content_y + i * self.item_height

            # Check if item is selected or hovered
            is_selected = item_idx == self.selected_item
            is_hovered = item_idx == self.hover_item

            # Create item rectangle
            item_rect = pygame.Rect(
                panel_rect.x + 10, item_y, panel_rect.width - 30, self.item_height - 5
            )

            # Draw the shop item
            self._draw_shop_item(
                surface, item_rect, option, player, is_selected, is_hovered
            )

            item_y += self.item_height + 5

        # Draw buttons and summary information
        self._draw_buttons_and_summary(surface, panel_rect)

        # Draw summary stats
        self._draw_summary_stats(surface, panel_rect)

        # Draw tooltip for selected item
        if self.selected_item is not None and self.show_tooltip:
            option = self.options[self.selected_item]
            self._draw_tooltip(
                surface, option["name"], option["description"], option["current_level"]
            )

        return surface

    def update(self, dt: float) -> None:
        """Update the shop and handle user input."""
        # Update animation with delta time
        self.update_animation(dt)

        # Handle input and update UI state
        self.handle_input()
        self.update_tooltip()
        self._extracted_from_update_16()
        self.update_expanded()
        self.update_category()
        self.update_max_visible_items()

        if self.show_tooltip:
            self.update_tooltip()

        if self.expanded:
            self._update_handler()

    def _update_handler(self):
        self.update_max_visible_items()
        self.update_category()
        self.update_expanded()
        self.update_scroll_offset()
        self.update_selected_item()
        self.update_hover_item()


# Forward references for type hints
class AsteroidField:
    """Type hint for AsteroidField class."""

    pass


class NotificationManager:
    """Type hint for NotificationManager class."""

    pass


class Player:
    """Type hint for Player class."""

    pass
