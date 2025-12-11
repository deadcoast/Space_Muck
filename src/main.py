"""
Space Muck - Advanced Procedural Generation Edition

A space mining game featuring:
- Evolving asteroid fields with cellular automaton mechanics
- Multiple symbiote races that adapt and evolve
- Dynamic mining and upgrade systems
- Advanced procedural generation with multiple algorithms
"""

import gc

# Standard library imports
import itertools
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Third-party library imports
import numpy as np
import pygame

# Local application imports
from config import (
    COLOR_BG,
    COLOR_PLAYER,
    COLOR_RACE_1,
    COLOR_RACE_2,
    COLOR_RACE_3,
    COLOR_TEXT,
    GRID_HEIGHT,
    GRID_WIDTH,
    MINIMAP_PADDING,
    MINIMAP_SIZE,
    RACE_INITIAL_DENSITY,
    STATE_PLAY,
    STATE_SHOP,
    VIEW_HEIGHT,
    VIEW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from entities.miner_entity import MinerEntity
from entities.player import Player
from events.event_bus import get_event_bus
from generators.asteroid_field import AsteroidField
from systems.combat_system import CombatSystem
from systems.encounter_generator import EncounterGenerator
from systems.event_system import get_event_batcher
from systems.game_loop import get_game_loop
from ui.draw_utils import (
    draw_button,
    draw_panel,
    draw_progress_bar,
    draw_text,
)
from ui.notification import NotificationManager
from ui.renderers import AsteroidFieldRenderer
from ui.shop import Shop
from utils.logging_setup import (
    LogContext,
    log_exception,
    log_memory_usage,
    log_performance_end,
    log_performance_metric,
    log_performance_start,
)

# Standard library imports

# Add the current directory to path to ensure proper importing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

# Constants for logging operations
LOG_FLEET_UPDATE = "Fleet update"

# Game configuration constants
GAME_CONFIG = {
    "version": "1.0.0",
    "intervals": {
        "update": 10,
        "auto_upgrade": 300,  # Every 5 seconds at 60 FPS
        "memory_check": 60,  # Every second at 60 FPS
    },
    "display": {
        "show_grid": True,
        "show_fps": True,
        "show_minimap": True,
        "show_tooltips": True,
        "show_debug": False,
    },
    "states": {
        "play": "PLAY",
        "shop": "SHOP",
        "paused": "PAUSED",
        "game_over": "GAME_OVER",
    },
    "state_transitions": {
        "PLAY": ["SHOP", "PAUSED", "GAME_OVER"],
        "SHOP": ["PLAY", "PAUSED"],
        "PAUSED": ["PLAY", "SHOP"],
        "GAME_OVER": ["PLAY"],
    },
    "race": {
        "initial_density": 0.005,
        "evolution_threshold": 100,
        "max_races": 3,
    },
    "resources": {
        "memory_warning_threshold": 90,  # Percentage
        "fps_history_size": 60,  # Number of frames to track
        "gc_threshold": 85,  # Percentage
    },
}


# State machine error types
class GameStateError(Exception):
    """Base class for game state related errors."""

    pass


class InvalidStateTransitionError(GameStateError):
    """Raised when attempting an invalid state transition."""

    pass


class StateValidationError(GameStateError):
    """Raised when state validation fails."""

    pass


class GameInitializationError(GameStateError):
    """Raised when game initialization fails."""

    pass


# Game components
(
    draw_text,
    draw_panel,
    draw_progress_bar,
    draw_button,
)

# Game systems
(
    log_exception,
    log_performance_start,
    log_performance_end,
    LogContext,
    log_memory_usage,
    log_performance_metric,
)

# Performance logging constants
LOG_DISCOVERY_CHECK = "Discovery check"
LOG_ENCOUNTER_PROCESSING = "Encounter processing"
LOG_FORCE_ENCOUNTER = "Force encounter generation"
LOG_COMBAT_RESULT = "Combat result processing"
LOG_AUTO_MINING_CHECK = "Auto-mining check"
LOG_ASTEROID_FIELD_UPDATE = "Asteroid field update"
LOG_RACE_EVOLUTION_CHECK = "Race evolution check"


class Game:
    """Main game class that orchestrates all game components.

    Following the standardized manager pattern, this class handles:
    - Resource tracking and capacity management
    - State management with proper logging
    - Performance monitoring and optimization
    - Event system integration

    Attributes:
        screen (pygame.Surface): Main display surface
        clock (pygame.time.Clock): Game clock for timing
        state (str): Current game state
        field (AsteroidField): Main asteroid field
        player (Player): Player instance
        stats (Dict[str, Any]): Game statistics tracking
    """

    def __init__(self) -> None:
        """Initialize the game and all its components.

        Follows initialization pattern:
        1. Core systems (pygame, display, clock)
        2. State and resource tracking
        3. UI components
        4. Game systems
        5. Entity initialization
        """
        # Initialize core systems with error handling
        with LogContext("Core System Initialization"):
            pygame.init()
            pygame.mixer.init()  # Initialize sound system
            pygame.font.init()  # Ensure fonts are initialized
            
            # Initialize game clock
            self.clock = pygame.time.Clock()

            # Create display surface
            self.screen: pygame.Surface = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            pygame.display.set_caption(
                f"Space Muck - Procedural Edition v{GAME_CONFIG['version']}"
            )

        # Initialize state tracking
        self._init_state_tracking()

        # Initialize resource tracking
        self._init_resource_tracking()

        # Initialize UI components with LogContext for error handling
        with LogContext("UI Initialization"):
            self.shop: Shop = Shop()
            self.notifier: NotificationManager = NotificationManager()
            self.renderer: AsteroidFieldRenderer = AsteroidFieldRenderer()

            # Initialize new ASCII UI components
            from ui.game_screen import ASCIIGameScreen
            from ui.minimap_panel import ASCIIMinimapPanel

            self.game_screen = ASCIIGameScreen(
                pygame.Rect(10, 10, WINDOW_WIDTH - 20, WINDOW_HEIGHT - 20), "SPACE MUCK"
            )

            # Initialize minimap panel in the bottom right corner
            minimap_rect = pygame.Rect(
                WINDOW_WIDTH - MINIMAP_SIZE - MINIMAP_PADDING,
                WINDOW_HEIGHT - MINIMAP_SIZE - MINIMAP_PADDING,
                MINIMAP_SIZE,
                MINIMAP_SIZE,
            )
            self.minimap_panel = ASCIIMinimapPanel(minimap_rect, "NAVIGATION")

    def _init_state_tracking(self) -> None:
        """Initialize game state tracking following state machine pattern.

        Sets up state tracking with validation and history:
        - Current state
        - State transition rules
        - State history with timestamps
        - Performance metrics
        """
        # Initialize state
        self.state = GAME_CONFIG["states"]["play"]
        self.previous_state = None

        # State history tracks transitions with timestamps
        self.state_history = []
        self.state_transition_counts = {
            state: 0 for state in GAME_CONFIG["states"].values()
        }
        self._record_state_transition(None, self.state)

        # Performance tracking
        self.state_metrics = {
            "transitions": 0,
            "invalid_attempts": 0,
            "avg_time_in_state": {},
        }

    # This method has been removed to fix the redefinition lint warning
    # The implementation at line ~480 is now the canonical version

    def _validate_state_transition(self, new_state: str) -> None:
        """Validate if a state transition is allowed.

        Args:
            new_state: The state to validate transition to

        Raises:
            InvalidStateTransitionError: If transition is not allowed
            StateValidationError: If state validation fails
        """
        if new_state not in GAME_CONFIG["state_transitions"].get(self.state, []):
            raise InvalidStateTransitionError(
                f"Cannot transition from {self.state} to {new_state}"
            )

    def _record_state_transition(
        self, old_state: Optional[str], new_state: str
    ) -> None:
        """Record a state transition in history.

        Args:
            old_state: The state transitioning from (None if initial state)
            new_state: The state transitioning to
        """
        timestamp = time.time()

        # Update state history
        self.state_history.append(
            {"from": old_state, "to": new_state, "timestamp": timestamp}
        )

        # Trim history if too long (keep last 100 transitions)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

        # Update average time in state
        if old_state:
            if last_transition := next(
                (t for t in reversed(self.state_history[:-1]) if t["to"] == old_state),
                None,
            ):
                time_in_state = timestamp - last_transition["timestamp"]
                avg = self.state_metrics["avg_time_in_state"].get(old_state, 0)
                if avg == 0:
                    self.state_metrics["avg_time_in_state"][old_state] = time_in_state
                else:
                    # Rolling average
                    self.state_metrics["avg_time_in_state"][old_state] = (avg * 0.9) + (
                        time_in_state * 0.1
                    )

        # Core timing
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.delta_time: float = 0.0

        # Initialize state machine components
        self.state = "menu"
        self.previous_state = None
        self._state_valid = True

        # Initialize state history tracking
        self._state_history = []
        self._state_timestamps = {state: 0.0 for state in GAME_CONFIG["states"]}
        self._state_transition_counts = {state: 0 for state in GAME_CONFIG["states"]}

        # Initialize display and automation settings
        self.show_debug = False
        self.auto_upgrade = False

        # Initialize event bus and subscribe to state changes
        self.event_bus = get_event_bus("default")
        self.event_bus.subscribe("state_change", self._handle_state_change)

        # Initialize frame tracking
        self.frame_counter: int = 0
        self.game_time: float = 0.0
        self.fps_history: List[float] = []

        # Initialize state machine
        self.state = GAME_CONFIG["states"]["play"]
        self.previous_state = self.state
        self._state_valid = True

        # Initialize state history tracking
        self._state_history = []
        self._state_timestamps = {
            state: 0.0 for state in GAME_CONFIG["states"].values()
        }
        self._state_transition_counts = {
            state: 0 for state in GAME_CONFIG["states"].values()
        }
        # State validation flags
        self._state_valid: bool = True
        self._last_validation_time: float = 0.0
        self._validation_interval: float = 1.0  # Validate state every second

        # Display settings
        self.zoom_level: float = 1.0
        self.show_minimap: bool = GAME_CONFIG["display"]["show_minimap"]
        self.show_debug: bool = GAME_CONFIG["display"]["show_debug"]
        self.show_tooltips: bool = GAME_CONFIG["display"]["show_tooltips"]
        self.show_grid: bool = GAME_CONFIG["display"]["show_grid"]

        # Automation flags
        self.auto_mine: bool = False
        self.auto_upgrade: bool = False

        # Register state change handler
        event_bus = get_event_bus("game")
        event_bus.subscribe("state_change", self._handle_state_change)
        # Register state change handler
        event_bus = get_event_bus("game")
        event_bus.subscribe("state_change", self._handle_state_change)

    def _validate_state_transition(self, new_state: str) -> bool:
        """Validate if a state transition is allowed.

        Args:
            new_state: The state to transition to

        Returns:
            bool: True if transition is valid

        Raises:
            InvalidStateTransitionError: If transition is not allowed
            StateValidationError: If state validation fails
        """
        try:
            # Validate state exists
            if new_state not in GAME_CONFIG["states"].values():
                raise InvalidStateTransitionError(f"Invalid state: {new_state}")

            # Validate transition is allowed
            allowed_transitions = GAME_CONFIG["state_transitions"].get(self.state, [])
            if new_state not in allowed_transitions:
                raise InvalidStateTransitionError(
                    f"Cannot transition from {self.state} to {new_state}"
                )

            # Validate state-specific conditions
            if new_state == GAME_CONFIG["states"]["shop"]:
                # Can only enter shop if player has enough currency
                min_currency = GAME_CONFIG.get("shop_entry_min_currency", 0)
                if self.player.currency < min_currency:
                    raise StateValidationError(
                        f"Need at least {min_currency} currency to enter shop"
                    )

            elif new_state == GAME_CONFIG["states"]["play"]:
                # Ensure player is in valid position before resuming
                if not (
                    0 <= self.player.x < self.field.width
                    and 0 <= self.player.y < self.field.height
                ):
                    raise StateValidationError("Player position invalid for play state")

            elif new_state == GAME_CONFIG["states"]["game_over"]:
                # Log game stats before transitioning
                logging.info(f"Game Over - Final Stats: {self.stats}")

            # Log successful validation
            logging.debug(f"Validated transition from {self.state} to {new_state}")
            return True

        except (InvalidStateTransitionError, StateValidationError) as e:
            # Log validation failure
            logging.warning(f"State transition validation failed: {str(e)}")
            raise

    def _record_state_transition(
        self, from_state: Optional[str], new_state: str
    ) -> None:
        """Record a state transition in history.

        Args:
            from_state: The state being transitioned from (None for initial state)
            new_state: The state being transitioned to
        """
        transition = {
            "from_state": from_state if from_state is not None else self.state,
            "to_state": new_state,
            "timestamp": time.time(),
        }

        # Add attributes that might not be initialized during the first state transition
        if hasattr(self, "game_time"):
            transition["game_time"] = self.game_time
        if hasattr(self, "frame_counter"):
            transition["frame"] = self.frame_counter

        # Use the correct attribute names
        self.state_history.append(transition)

        # Initialize state_transition_counts if needed
        if not hasattr(self, "state_transition_counts"):
            self.state_transition_counts = {}
        if new_state not in self.state_transition_counts:
            self.state_transition_counts[new_state] = 0
        self.state_transition_counts[new_state] += 1

        # Keep history size bounded
        if len(self.state_history) > 1000:
            self.state_history.pop(0)

    def change_state(self, new_state: str) -> None:
        """Change the game state with validation and history tracking.

        Args:
            new_state: The state to transition to

        Raises:
            InvalidStateTransitionError: If transition is not allowed
        """
        try:
            if self._validate_state_transition(new_state):
                self._perform_state_transition(new_state)
        except GameStateError as e:
            logging.error(f"State transition failed: {e}")
            self._state_valid = False
            raise

    def _perform_state_transition(self, new_state):
        self._record_state_transition(self.state, new_state)
        self.previous_state = self.state
        self.state = new_state

        # Initialize state_timestamps if needed
        if not hasattr(self, "state_timestamps"):
            self.state_timestamps = {}
        self.state_timestamps[new_state] = time.time()

        # Emit state change event
        get_event_bus().emit(
            "state_change",
            {
                "from_state": self.previous_state,
                "to_state": new_state,
                "timestamp": time.time(),
            },
        )

        logging.info(
            f"State changed: {self.previous_state} -> {new_state} "
            f"(Total transitions to {new_state}: "
            f"{self._state_transition_counts[new_state]})"
        )

    def _init_resource_tracking(self) -> None:
        """Initialize resource and performance tracking."""
        # Performance timers
        self.last_field_update_time: float = 0.0
        self.last_memory_check: float = 0.0

        # Intervals
        self.update_interval: int = GAME_CONFIG["intervals"]["update"]
        self.auto_upgrade_interval: int = GAME_CONFIG["intervals"]["auto_upgrade"]

        # Resource thresholds
        self._memory_warning_threshold: float = GAME_CONFIG["resources"][
            "memory_warning_threshold"
        ]
        self._gc_threshold: float = GAME_CONFIG["resources"]["gc_threshold"]

        # Field generation parameters
        self.seed: int = random.randint(1, 1000000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initialize field
        with LogContext("Field Initialization"):
            self.field: AsteroidField = AsteroidField(seed=self.seed)

        # Initialize player
        with LogContext("Player Initialization"):
            self.player: Player = Player()
            # Connect field and player
            self.field.player = self.player

        # Initialize game systems
        with LogContext("Systems Initialization"):
            self.combat_system: CombatSystem = CombatSystem(self.player)
            self.encounter_generator: EncounterGenerator = EncounterGenerator(
                self.player, self.combat_system
            )
            logging.info("Game systems initialized successfully")

        # Initialize races
        with LogContext("Race Initialization"):
            self.initialize_symbiote_races()

        # Initialize UI state
        self._init_ui_state()

        # Initialize statistics tracking
        self._init_statistics()

        # Register event handlers
        self._register_event_handlers()

    def _init_ui_state(self) -> None:
        """Initialize UI state variables with proper type hints."""
        self.selected_race_index: int = -1
        self.hover_position: Tuple[int, int] = (0, 0)
        self.tooltip_text: Optional[str] = None
        self.tooltip_position: Tuple[int, int] = (0, 0)
        self.show_race_details: bool = False
        self.cursor_over_ui: bool = False
        self.display_controls_help: bool = False

        # Initialize UI component states
        self.ui_components_initialized: bool = False
        self.ui_needs_update: bool = True

    def _init_statistics(self) -> None:
        """Initialize game statistics tracking."""
        self.stats: Dict[str, int] = {
            "total_mined": 0,
            "total_rare_mined": 0,
            "total_precious_mined": 0,
            "total_anomalies_mined": 0,
            "upgrades_purchased": 0,
            "ships_lost": 0,
            "time_played": 0,
            "races_encountered": 0,
            "race_evolutions": 0,
            "total_combats": 0,
            "combats_won": 0,
            "encounters": 0,
            "discoveries": 0,
        }

    def _register_event_handlers(self) -> None:
        """Register event handlers for game events."""
        event_bus = get_event_bus()
        event_bus.subscribe("state_change", self._handle_state_change)
        event_bus.subscribe("resource_update", self._handle_resource_update)
        event_bus.subscribe("performance_warning", self._handle_performance_warning)

    def _handle_state_change(self, event_data: Dict[str, Any]) -> None:
        """Handle game state change events.

        Args:
            event_data: Dictionary containing state change information
        """
        new_state = event_data.get("new_state")
        if new_state is None:
            logging.warning("State change event received without new_state")
            return

        if new_state not in GAME_CONFIG["states"].values():
            logging.error(f"Invalid state received: {new_state}")
            return

        self.previous_state = self.state
        self.state = new_state
        self._state_timestamps[new_state] = time.time()
        logging.info(f"Game state changed from {self.previous_state} to {self.state}")

    def _handle_resource_update(self, event_data: Dict[str, Any]) -> None:
        """Handle resource update events.

        Args:
            event_data: Dictionary containing resource update information
        """
        resource_type = event_data.get("type")
        if resource_type is None:
            logging.warning("Resource update event received without type")
            return

        # Update resource tracking
        if resource_type == "memory":
            self._check_memory_usage()
        elif resource_type == "performance":
            self._check_performance_metrics()

    def _handle_performance_warning(self, event_data: Dict[str, Any]) -> None:
        """Handle performance warning events.

        Args:
            event_data: Dictionary containing performance warning information
        """
        warning_type = event_data.get("type")
        if warning_type is None:
            logging.warning("Performance warning event received without type")
            return

        if warning_type == "memory":
            self._handle_memory_warning(event_data)
        elif warning_type == "fps":
            self._handle_fps_warning(event_data)

    def _check_memory_usage(self) -> None:
        """Check current memory usage and trigger garbage collection if needed."""
        current_memory = log_memory_usage()
        if current_memory > self._gc_threshold:
            logging.info("Memory usage high, triggering garbage collection")
            gc.collect()

    def _check_performance_metrics(self) -> None:
        """Check and log performance metrics."""
        current_fps = self.clock.get_fps()
        self.fps_history.append(current_fps)

        # Keep history size bounded
        if len(self.fps_history) > GAME_CONFIG["resources"]["fps_history_size"]:
            self.fps_history.pop(0)

        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        if avg_fps < 30:  # Warning threshold
            get_event_bus().emit(
                "performance_warning",
                {"type": "fps", "current_fps": avg_fps, "threshold": 30},
            )

    def _handle_memory_warning(self, event_data: Dict[str, Any]) -> None:
        """Handle memory warning events.

        Args:
            event_data: Dictionary containing memory warning information
        """
        current_usage = event_data.get("usage", 0)
        if current_usage > self._memory_warning_threshold:
            self.notifier.add_notification(
                "High memory usage detected. Some features may be disabled.",
                notification_type="warning",
            )

    def _handle_fps_warning(self, event_data: Dict[str, Any]) -> None:
        """Handle FPS warning events.

        Args:
            event_data: Dictionary containing FPS warning information
        """
        current_fps = event_data.get("current_fps", 0)
        threshold = event_data.get("threshold", 30)
        if current_fps < threshold:
            self.notifier.add_notification(
                f"Low FPS detected ({current_fps:.1f}). Optimizing performance...",
                notification_type="warning",
            )

        # Visual settings initialization
        self.initialize_visual_settings()

        # Add welcome notifications
        self.add_welcome_notifications()

        # Log successful initialization
        logging.info(f"Game initialized with seed: {self.seed}")
        log_memory_usage("After game initialization")

    def initialize_symbiote_races(self) -> None:
        """Initialize the symbiote races with different traits and behaviors."""
        self.available_races = [
            # Blue race - adaptive trait
            MinerEntity(
                1,
                COLOR_RACE_1,
                birth_set={2, 3},
                survival_set={3, 4},
                initial_density=RACE_INITIAL_DENSITY,
            ),
            # Magenta race - expansive trait
            MinerEntity(
                2,
                COLOR_RACE_2,
                birth_set={3, 4},
                survival_set={2, 3},
                initial_density=RACE_INITIAL_DENSITY,
            ),
            # Orange race - selective trait
            MinerEntity(
                3,
                COLOR_RACE_3,
                birth_set={1, 5},
                survival_set={1, 4},
                initial_density=RACE_INITIAL_DENSITY,
            ),
        ]

        # Set traits explicitly to ensure diversity
        self.available_races[0].trait = "adaptive"
        self.available_races[1].trait = "expansive"
        self.available_races[2].trait = "selective"

        # Log race creation
        for race in self.available_races:
            logging.info(f"Created race {race.race_id} with trait {race.trait}")

        # Add races to field
        for race in self.available_races:
            race.populate(self.field)
            self.field.races.append(race)

    def initialize_visual_settings(self) -> None:
        """Initialize visual settings and precompute resources following manager pattern.

        This method follows standardized initialization steps:
        1. Initialize visual configuration with type-safe defaults
        2. Set up UI elements with error handling
        3. Create and cache visual resources
        4. Initialize renderer components
        """
        # Initialize visual configuration
        self._init_visual_config()

        try:
            # Initialize UI fonts with type hints
            self._init_fonts()

            # Create and cache background resources
            self._init_background_resources()

            # Initialize cursor resources
            self._init_cursor_resources()

            # Initialize renderer caches
            with LogContext("Renderer Cache Initialization"):
                self.renderer.initialize(self.field)

        except Exception as e:
            logging.error(f"Error initializing visuals: {e}")
            log_exception(e)
            raise GameInitializationError("Failed to initialize visual settings") from e

    def _init_visual_config(self) -> None:
        """Initialize visual configuration with type-safe defaults."""
        self.visual_config: Dict[str, Any] = {
            "ui": {
                "font_size_normal": 16,
                "font_size_title": 24,
                "font_family": "Arial",
            },
            "background": {
                "pattern_size": 64,
                "noise_range": (-5, 5),
                "base_color": COLOR_BG,
            },
            "cursor": {
                "size": 12,
                "inner_color": COLOR_PLAYER,
                "outline_color": (255, 255, 255),
                "outline_width": 1,
            },
        }

    def _init_fonts(self) -> None:
        """Initialize UI fonts with proper error handling."""
        config = self.visual_config["ui"]
        try:
            self.ui_font = pygame.font.SysFont(
                config["font_family"], config["font_size_normal"]
            )
            self.title_font = pygame.font.SysFont(
                config["font_family"], config["font_size_title"], bold=True
            )
        except pygame.error as e:
            logging.error(f"Failed to initialize fonts: {e}")
            raise GameInitializationError("Font initialization failed") from e

    def _init_background_resources(self) -> None:
        """Create and cache background pattern resources."""
        config = self.visual_config["background"]
        pattern_size = config["pattern_size"]

        # Create background pattern surface
        self.bg_pattern = pygame.Surface((pattern_size, pattern_size))
        self.bg_pattern.fill(config["base_color"])

        # Add subtle noise to background
        noise_min, noise_max = config["noise_range"]
        for y, x in itertools.product(range(pattern_size), range(pattern_size)):
            noise = random.randint(noise_min, noise_max)
            color = max(0, min(255, config["base_color"][0] + noise))
            self.bg_pattern.set_at((x, y), (color, color, color + 2))

    def _init_cursor_resources(self) -> None:
        """Initialize cursor-related visual resources."""
        config = self.visual_config["cursor"]

        # Create cursor surface
        self.cursor_size = config["size"]
        self.cursor_surface = pygame.Surface(
            (self.cursor_size, self.cursor_size), pygame.SRCALPHA
        )

        # Draw cursor circle and outline
        center = (self.cursor_size // 2, self.cursor_size // 2)
        radius = self.cursor_size // 2

        pygame.draw.circle(self.cursor_surface, config["inner_color"], center, radius)
        pygame.draw.circle(
            self.cursor_surface,
            config["outline_color"],
            center,
            radius,
            config["outline_width"],
        )

    def add_welcome_notifications(self) -> None:
        """Add initial helpful notifications for new players."""
        self.notifier.add("Welcome to Space Muck!", duration=240)
        self.notifier.add(
            "Use ARROW KEYS to move your mining ship", category="system", duration=240
        )
        self.notifier.add(
            "Press SPACE to mine asteroids", category="system", duration=240
        )
        self.notifier.add("Press S to open the shop", category="system", duration=240)
        self.notifier.add(
            "Press N to toggle notification panel", category="system", duration=240
        )
        self.notifier.add("Press M to toggle minimap", category="system", duration=240)
        self.notifier.add("Use +/- to zoom in/out", category="system", duration=240)
        self.notifier.add("Press F to feed symbiotes", category="system", duration=240)
        self.notifier.add(
            "Press A to toggle auto-mining", category="system", duration=240
        )
        self.notifier.add(
            "Press H to display control help", category="system", duration=240
        )

        # Toggle notification panel on by default for first run
        self.notifier.show_full_panel = True

    def regenerate_field(self) -> None:
        """Regenerate the asteroid field with new parameters."""
        start_time = log_performance_start("Field regeneration")

        # Generate new seed
        self.seed = random.randint(1, 1000000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Create new field
        with LogContext("Field Regeneration"):
            self.setup_new_field()
        # Notify player
        self.notifier.add(
            f"Field regenerated with seed {self.seed}!", category="event", importance=2
        )

        log_performance_end("Field regeneration", start_time)

    def setup_new_field(self):
        old_field = self.field
        self.field = AsteroidField(seed=self.seed)

        # Transfer races
        for race in old_field.races:
            race.field = self.field
            race.populate(self.field)
            self.field.races.append(race)

        # Connect player to new field
        self.player.x = GRID_WIDTH // 2
        self.player.y = GRID_HEIGHT // 2
        self.field.player = self.player

        # Reset renderer caches
        self.renderer.clear_caches()
        self.renderer.initialize(self.field)

        # Clean up old field
        del old_field
        gc.collect()

    def _handle_ui_component_events(self, event) -> bool:
        """Handle events for UI components.

        Args:
            event: The pygame event to process

        Returns:
            bool: True if the event was handled by a UI component, False otherwise
        """
        # Check if notifier handled the event
        if self.notifier.handle_event(event):
            return True

        # Check if shop handled the event
        if self.state == STATE_SHOP and self.shop.handle_event(
            event, self.player, self.field
        ):
            return True

        # Check if ASCII UI components handled the event
        if hasattr(self, "game_screen") and self.game_screen.handle_event(event):
            return True

        # Check if minimap handled the event
        return bool(
            (
                hasattr(self, "minimap_panel")
                and self.show_minimap
                and self.minimap_panel.handle_event(event)
            )
        )

    # Constants for event handling
    EVENT_QUIT = "EVENT_QUIT"
    EVENT_KEYDOWN = "EVENT_KEYDOWN"
    EVENT_MOUSEMOTION = "EVENT_MOUSEMOTION"
    EVENT_MOUSEBUTTONDOWN = "EVENT_MOUSEBUTTONDOWN"

    def handle_events(self) -> bool:
        """Process all pygame events.

        Returns:
            bool: True if the game should continue running, False otherwise.
        """
        for event in pygame.event.get():
            # First, let UI components handle events
            if self._handle_ui_component_events(event):
                continue

            # Process event based on type
            event_result = self._process_event_by_type(event)
            if event_result == self.EVENT_QUIT:
                return False  # Stop the game

        return True

    def _process_event_by_type(self, event) -> Optional[str]:
        """Process an event based on its type.

        Args:
            event: The pygame event to process

        Returns:
            Optional[str]: Event result code if special handling is needed, None otherwise
        """
        # Map event types to their handler methods
        event_handlers = {
            pygame.QUIT: self._handle_quit_event,
            pygame.KEYDOWN: self._handle_keydown_event,
            pygame.MOUSEMOTION: self._handle_mousemotion_event,
            pygame.MOUSEBUTTONDOWN: self._handle_mousebuttondown_event,
        }

        # Call the appropriate handler if available
        if event.type in event_handlers:
            return event_handlers[event.type](event)

        return None

    def _handle_quit_event(self, event) -> str:
        """Handle quit event.

        Args:
            event: The pygame quit event

        Returns:
            str: EVENT_QUIT to signal game should stop
        """
        self.quit_game()
        return self.EVENT_QUIT

    def _handle_keydown_event(self, event) -> None:
        """Handle key down event.

        Args:
            event: The pygame keydown event
        """
        self.handle_key_press(event.key)
        return None

    def _handle_mousemotion_event(self, event) -> None:
        """Handle mouse motion event.

        Args:
            event: The pygame mouse motion event
        """
        self.handle_mouse_motion(event)
        return None

    def _handle_mousebuttondown_event(self, event) -> None:
        """Handle mouse button down event.

        Args:
            event: The pygame mouse button down event
        """
        self.handle_mouse_button_down(event)
        return None

    def quit_game(self) -> None:
        """Exit the game cleanly and perform cleanup operations."""
        try:
            # Start performance timing for exit process
            exit_start = log_performance_start("Game exit process")

            # Log all game statistics
            self._log_game_statistics()

            # Log performance metrics if available
            self._log_performance_metrics()

            # Save player data and clean up resources
            try:
                self._display_farewell_and_cleanup()
                # Close any open resources or connections
                # (Adding placeholder for future implementation)
            except Exception as cleanup_error:
                log_exception("Error during exit cleanup", cleanup_error)

            # Signal that the game should stop running
            logging.info("Quitting game")
            log_performance_end("Game exit process", exit_start)
        except Exception as e:
            log_exception("Error during game exit", e)

    def _log_game_statistics(self) -> None:
        """Log complete game statistics before quitting."""
        logging.info("===== COMPLETE GAME STATISTICS =====")

        # Define and log basic statistics
        basic_stats = [
            ("Game time", f"{self.game_time:.2f} seconds"),
            ("Resources collected", self.stats.get("resources_collected", 0)),
            ("Distance traveled", self.stats.get("distance_traveled", 0)),
            ("Encounters", self.stats.get("encounters", 0)),
            ("Combat victories", self.stats.get("combats_won", 0)),
            ("Total combats", self.stats.get("total_combats", 0)),
        ]

        for stat_name, stat_value in basic_stats:
            logging.info(f"  {stat_name}: {stat_value}")

        # Log detailed statistics
        logging.info("\n--- Detailed Statistics ---")
        basic_stat_names = [item[0] for item in basic_stats]
        for stat_name, stat_value in sorted(self.stats.items()):
            if stat_name not in basic_stat_names:
                logging.info(f"  {stat_name}: {stat_value}")

    def _log_performance_metrics(self) -> None:
        """Log detailed performance metrics if available."""
        if not hasattr(self, "performance_metrics") or not self.performance_metrics:
            return

        logging.info("\n--- Performance Metrics Summary ---")
        for metric_name, values in self.performance_metrics.items():
            if not values:
                continue

            # Calculate statistics
            avg = sum(values) / len(values)
            max_val = max(values)
            min_val = min(values)

            # Calculate standard deviation
            std_dev = (
                (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
                if len(values) > 1
                else 0
            )

            # Calculate 95th percentile
            p95 = (
                sorted(values)[int(len(values) * 0.95)]
                if len(values) >= 20
                else max_val
            )

            # Log the metrics
            logging.info(f"  {metric_name}:")
            logging.info(
                f"    avg={avg:.3f}ms, min={min_val:.3f}ms, max={max_val:.3f}ms"
            )
            logging.info(
                f"    std_dev={std_dev:.3f}ms, p95={p95:.3f}ms, samples={len(values)}"
            )

    def _display_farewell_and_cleanup(self):
        # Display a farewell message to the player
        play_time_min = self.game_time / 60

        farewell_msg = (
            f"Thanks for playing Space Muck! You played for {play_time_min:.1f} minutes."
            if play_time_min >= 1
            else "Thanks for playing Space Muck!"
        )
        # Add session highlights if available
        if self.stats.get("combats_won", 0) > 0:
            combat_ratio = self.stats.get("combats_won", 0) / max(
                1, self.stats.get("total_combats", 1)
            )
            farewell_msg += f"\nCombat success rate: {combat_ratio:.1%}"

        if self.stats.get("discoveries", 0) > 0:
            farewell_msg += f"\nUnique discoveries: {self.stats.get('discoveries', 0)}"

        self.notifier.add(farewell_msg, category="system", importance=3)

        # Perform system cleanup
        log_memory_usage("Before final cleanup")

        # Run garbage collection to free memory
        gc_start = time.time()
        gc.collect()
        logging.info(
            f"Final garbage collection completed in {time.time() - gc_start:.4f} seconds"
        )

    def handle_key_press(self, key: int) -> None:
        """Handle keyboard input with state validation."""
        try:
            # Handle global keys (work in all states)
            if key == pygame.K_ESCAPE:
                if self.state == GAME_CONFIG["states"]["play"]:
                    self.change_state(GAME_CONFIG["states"]["paused"])
                elif self.state == GAME_CONFIG["states"]["paused"]:
                    self.change_state(GAME_CONFIG["states"]["play"])
                elif self.state == GAME_CONFIG["states"]["shop"]:
                    self.change_state(GAME_CONFIG["states"]["play"])

            elif key == pygame.K_F3:
                # Toggle debug mode
                self.show_debug = not self.show_debug
                logging.info(
                    f"Debug mode {'enabled' if self.show_debug else 'disabled'}"
                )

            elif key == pygame.K_h:
                # Toggle control help
                self.display_controls_help = not self.display_controls_help
                logging.info(
                    f"Control help {'shown' if self.display_controls_help else 'hidden'}"
                )

            # Handle state-specific keys
            elif self.state == GAME_CONFIG["states"]["play"]:
                self.handle_play_state_keys(key)

        except GameStateError as e:
            logging.warning(f"Invalid key press in state {self.state}: {e}")
            self.notifier.add_notification(
                "Cannot perform that action in the current state",
                notification_type="warning",
            )

    def handle_mouse_motion(self, event) -> None:
        """Handle mouse motion events."""
        # Update hover position for tooltips
        self.hover_position = event.pos
        self.cursor_over_ui = self.check_cursor_over_ui()

    def handle_mouse_button_down(self, event) -> None:
        """Handle mouse button down events."""
        # Handle mouse clicks
        if event.button == 1:  # Left click
            self.handle_mouse_click(event.pos)
        elif event.button == 3:  # Right click
            if self.state == STATE_PLAY:
                self.handle_right_click_in_play_state(event.pos)

    def handle_right_click_in_play_state(self, pos) -> None:
        """Handle right click in play state."""
        # Select race under cursor
        grid_x, grid_y = self.screen_to_grid(pos[0], pos[1])
        if not (0 <= grid_x < self.field.width and 0 <= grid_y < self.field.height):
            return

        race_id = self.field.entity_grid[grid_y, grid_x]
        if race_id <= 0:
            return

        # Find the race
        for i, race in enumerate(self.field.races):
            if race.race_id == race_id:
                self.selected_race_index = i
                self.show_race_details = True
                break

    def handle_play_state_keys(self, key: int) -> None:
        """Handle keys specific to play state with validation.

        Args:
            key: The pygame key code that was pressed
        """
        try:
            # Validate we're in play state
            if self.state != GAME_CONFIG["states"]["play"]:
                raise StateValidationError(
                    f"Play state keys called in {self.state} state"
                )

            # Use a command dictionary to map keys to their handler methods
            key_handlers = {
                pygame.K_s: self._handle_shop_key,
                pygame.K_m: self._handle_minimap_toggle,
                pygame.K_g: self._handle_grid_toggle,
                pygame.K_a: self._handle_auto_mine_toggle,
                pygame.K_SPACE: self._handle_mine_action,
                pygame.K_r: self._handle_field_regeneration,
                pygame.K_f: self._handle_symbiote_feeding,
                pygame.K_u: self._handle_auto_upgrade_toggle,
                pygame.K_p: self._handle_add_ship,
            }

            # Handle zoom keys separately since they use multiple key mappings
            if key in [pygame.K_PLUS, pygame.K_EQUALS]:
                self._handle_zoom_in()
            elif key in [pygame.K_MINUS, pygame.K_UNDERSCORE]:
                self._handle_zoom_out()
            # Handle other mapped keys
            elif key in key_handlers:
                key_handlers[key]()

        except GameStateError as e:
            logging.warning(f"Invalid play state action: {e}")
            self.notifier.add_notification(
                "Action not allowed in current state", notification_type="warning"
            )
        except Exception as e:
            log_exception("Error handling play state key", e)

    def _handle_shop_key(self) -> None:
        """Open the shop state."""
        self.change_state(GAME_CONFIG["states"]["shop"])

    def _handle_minimap_toggle(self) -> None:
        """Toggle minimap visibility."""
        self.show_minimap = not self.show_minimap
        logging.info(f"Minimap {'shown' if self.show_minimap else 'hidden'}")

    def _handle_grid_toggle(self) -> None:
        """Toggle grid visibility."""
        self.show_grid = not self.show_grid
        logging.info(f"Grid {'shown' if self.show_grid else 'hidden'}")

    def _handle_auto_mine_toggle(self) -> None:
        """Toggle auto-mining feature."""
        self.auto_mine = not self.auto_mine
        status = "enabled" if self.auto_mine else "disabled"
        self.notifier.add(f"Auto-mining {status}", category="mining")
        logging.info(f"Auto-mining {status}")

    def _handle_mine_action(self) -> None:
        """Perform mining action."""
        self.mine()

    def _handle_field_regeneration(self) -> None:
        """Regenerate the asteroid field."""
        self.regenerate_field()
        logging.info("Field regenerated")

    def _handle_symbiote_feeding(self) -> None:
        """Feed symbiotes."""
        self.feed_symbiotes()
        logging.info("Symbiotes fed")

    def _handle_auto_upgrade_toggle(self) -> None:
        """Toggle auto-upgrade feature."""
        self.auto_upgrade = not self.auto_upgrade
        status = "enabled" if self.auto_upgrade else "disabled"
        self.notifier.add(f"Auto-upgrade {status}", category="upgrade")
        logging.info(f"Auto-upgrade {status}")

    def _handle_zoom_in(self) -> None:
        """Increase zoom level."""
        self.zoom_level = min(2.0, self.zoom_level + 0.1)
        logging.debug(f"Zoom level increased to {self.zoom_level:.1f}")

    def _handle_zoom_out(self) -> None:
        """Decrease zoom level."""
        self.zoom_level = max(0.5, self.zoom_level - 0.1)
        logging.debug(f"Zoom level decreased to {self.zoom_level:.1f}")

    def _handle_add_ship(self) -> None:
        """Add a ship to the player's fleet."""
        if self.player.add_ship():
            msg = "New mining ship added to your fleet!"
            self.notifier.add(msg, category="mining")
            logging.info(msg)
        else:
            msg = "Cannot add more ships (max reached or insufficient funds)"
            self.notifier.add(msg, category="mining")
            logging.warning(msg)

    def handle_mouse_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse clicks based on game state with validation."""
        try:
            # Handle click based on current game state
            if self.state == GAME_CONFIG["states"]["shop"]:
                self._handle_shop_state_click(pos)
            elif self.state == GAME_CONFIG["states"]["play"]:
                self._handle_play_state_click(pos)
            elif self.state == GAME_CONFIG["states"]["paused"]:
                self._handle_paused_state_click(pos)
            elif self.state == GAME_CONFIG["states"]["game_over"]:
                self._handle_game_over_state_click(pos)

        except GameStateError as e:
            logging.warning(f"Invalid mouse click in state {self.state}: {e}")
            self.notifier.add_notification(
                "Cannot interact in the current state", notification_type="warning"
            )

    def _handle_shop_state_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse clicks in shop state.

        Args:
            pos: The mouse position (x, y)
        """
        self.shop.handle_click(pos)
        logging.debug(f"Shop click at position {pos}")

    def _handle_play_state_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse clicks in play state.

        Args:
            pos: The mouse position (x, y)
        """
        # Check if click was on UI element
        if self.check_cursor_over_ui():
            logging.debug("Click ignored - over UI element")
            return

        # Try to move player to clicked position
        self._attempt_player_movement(pos)

    def _attempt_player_movement(self, pos: Tuple[int, int]) -> None:
        """Attempt to move the player to the clicked position.

        Args:
            pos: The mouse position (x, y)
        """
        # Convert screen position to grid position
        grid_x, grid_y = self.screen_to_grid(pos[0], pos[1])

        # Check if position is valid
        if 0 <= grid_x < self.field.width and 0 <= grid_y < self.field.height:
            # Calculate movement delta
            dx = grid_x - self.player.x
            dy = grid_y - self.player.y

            # Move player there
            self.player.move(dx, dy, self.field)
            self.player.has_moved = True  # For encounter checks
            logging.debug(f"Player moved to grid position ({grid_x}, {grid_y})")
        else:
            logging.debug(f"Invalid grid position: ({grid_x}, {grid_y})")

    @staticmethod
    def _handle_paused_state_click(_: Tuple[int, int]) -> None:
        """Handle mouse clicks in paused state.

        Args:
            _: The mouse position (x, y) - unused in this state
        """
        logging.debug("Click ignored in paused state")

    @staticmethod
    def _handle_game_over_state_click(_: Tuple[int, int]) -> None:
        """Handle mouse clicks in game over state.

        Args:
            _: The mouse position (x, y) - unused in this state
        """
        logging.debug("Click ignored in game over state")

    def check_cursor_over_ui(self) -> bool:
        """Check if the cursor is over a UI element."""
        # Check notification panel
        if self.notifier.show_full_panel:
            panel_rect = pygame.Rect(
                self.notifier.panel_x,
                self.notifier.panel_y,
                self.notifier.panel_width,
                self.notifier.panel_height,
            )
            if panel_rect.collidepoint(self.hover_position):
                return True

        # Check shop panel
        if self.state == STATE_SHOP:
            panel_rect = pygame.Rect(
                self.shop.panel_x,
                self.shop.panel_y,
                self.shop.current_width,
                self.shop.panel_height,
            )
            if panel_rect.collidepoint(self.hover_position):
                return True

        # Check minimap
        if self.show_minimap:
            minimap_rect = pygame.Rect(
                WINDOW_WIDTH - MINIMAP_SIZE - MINIMAP_PADDING,
                WINDOW_HEIGHT - MINIMAP_SIZE - MINIMAP_PADDING,
                MINIMAP_SIZE,
                MINIMAP_SIZE,
            )
            if minimap_rect.collidepoint(self.hover_position):
                return True

        return False

    def screen_to_grid(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Convert screen coordinates to grid coordinates."""
        # Calculate view center based on player position
        view_center_x = self.player.x
        view_center_y = self.player.y

        # Calculate view dimensions considering zoom
        view_width = VIEW_WIDTH / self.zoom_level
        view_height = VIEW_HEIGHT / self.zoom_level

        # Calculate view origin
        view_origin_x = view_center_x - view_width / 2
        view_origin_y = view_center_y - view_height / 2

        # Calculate pixel to cell ratio
        cell_width = WINDOW_WIDTH / view_width
        cell_height = WINDOW_HEIGHT / view_height

        # Convert screen position to grid position
        grid_x = int(view_origin_x + screen_x / cell_width)
        grid_y = int(view_origin_y + screen_y / cell_height)

        return grid_x, grid_y

    def grid_to_screen(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates."""
        # Calculate view center based on player position
        view_center_x = self.player.x
        view_center_y = self.player.y

        # Calculate view dimensions considering zoom
        view_width = VIEW_WIDTH / self.zoom_level
        view_height = VIEW_HEIGHT / self.zoom_level

        # Calculate view origin
        view_origin_x = view_center_x - view_width / 2
        view_origin_y = view_center_y - view_height / 2

        # Calculate pixel to cell ratio
        cell_width = WINDOW_WIDTH / view_width
        cell_height = WINDOW_HEIGHT / view_height

        # Convert grid position to screen position
        screen_x = int((grid_x - view_origin_x) * cell_width)
        screen_y = int((grid_y - view_origin_y) * cell_height)

        return screen_x, screen_y

    def mine(self) -> None:
        """Mine asteroids at the player's current position."""
        with LogContext("Mining operation"):
            # Mine the asteroid
            minerals_mined = self.player.mine(self.field)

            # Show notification if something was mined
            if minerals_mined > 0:
                self.notifier.add(f"Mined {minerals_mined} minerals", category="mining")
                self.stats["total_mined"] += minerals_mined

    def feed_symbiotes(self) -> None:
        """Feed the symbiotes to reduce aggression."""
        # Show dialog to select amount
        feed_amount = min(100, self.player.currency // 2)

        if feed_amount <= 0:
            self.notifier.add("Not enough currency to feed symbiotes", category="race")
            return

        # Feed symbiotes
        results = self.player.feed_symbiotes(self.field, feed_amount)

        if results["fed"] > 0:
            self.notifier.add(
                f"Fed symbiotes with {results['fed']} minerals", category="race"
            )
        else:
            self.notifier.add("Failed to feed symbiotes", category="race")

    def handle_auto_upgrade(self) -> None:
        """Handle auto-upgrade logic."""
        if (
            not self.auto_upgrade
            or self.frame_counter % self.auto_upgrade_interval != 0
        ):
            return

        # Find best upgrade based on cost/benefit ratio
        best_upgrade = None
        best_ratio = 0

        for upgrade in self.shop.get_available_upgrades(self.player, self.field):
            if upgrade["cost"] > self.player.currency:
                continue

            # Calculate benefit ratio (simplified)
            ratio = 1.0

            if "mining" in upgrade["name"].lower():
                ratio = 2.0  # Prioritize mining upgrades

            # Adjust for cost
            ratio /= upgrade["cost"]

            if ratio > best_ratio:
                best_ratio = ratio
                best_upgrade = upgrade

        # Purchase the upgrade
        if best_upgrade:
            self.shop.purchase_upgrade(
                best_upgrade, self.player, self.field, self.notifier
            )
            self.stats["upgrades_purchased"] += 1

    def update(self) -> None:
        """Update game state for the current frame."""
        try:
            # Skip updates if paused
            if self.paused:
                return

            # Start performance timing
            update_start = log_performance_start("Update frame")

            # Update time-related state
            self._update_time_tracking()

            # Update UI components
            self._update_ui_components()

            # State-specific updates
            self._update_current_game_state()

            # End performance timing
            log_performance_end("Update frame", update_start)
        except Exception as e:
            log_exception("Error in update process", e)
            self.notifier.add(f"Error: {str(e)}", category="error", importance=3)

    def _update_time_tracking(self) -> None:
        """Update time-related tracking and statistics."""
        # Update delta time
        self.delta_time = self.clock.get_time() / 1000.0  # Convert ms to seconds
        self.game_time += self.delta_time
        self.frame_counter += 1

        # Update game time in stats
        self.stats["time_played"] = self.game_time

        # Periodic memory check
        current_time = time.time()
        if current_time - self.last_memory_check > 60:  # Every minute
            log_memory_usage(f"Game running for {int(self.game_time)}s")
            self.last_memory_check = current_time

    def _update_ui_components(self) -> None:
        """Update UI components that need to be updated regardless of game state."""
        # Update notification system
        self.notifier.update(self.delta_time)

        # Update ASCII UI components animations
        if hasattr(self, "game_screen"):
            self.game_screen.update_animations(self.delta_time)
        if hasattr(self, "minimap_panel") and self.show_minimap:
            self.minimap_panel.update_animations(self.delta_time)

        # Clear tooltip if not updated this frame
        self.tooltip_text = None

    def _update_current_game_state(self) -> None:
        """Update the current game state based on the active state."""
        # State-specific updates
        if self.state == STATE_PLAY:
            self.update_play_state()
        elif self.state == STATE_SHOP:
            self.shop.update(self.delta_time)

        # Auto-upgrade if enabled (applies to multiple states)
        if self.auto_upgrade:
            self.handle_auto_upgrade()

    def update_play_state(self) -> None:
        """Update game elements during play state.

        This method orchestrates the execution of all game systems in the correct order,
        ensuring proper dependencies are respected and facilitating communication between
        components according to the system architecture integration plan.
        """
        try:
            # Start performance timing with detailed context
            play_state_start = log_performance_start("Play state update")
            update_context = {"frame": self.frame_counter, "game_time": self.game_time}
            logging.debug(f"Starting play state update for frame {self.frame_counter}")

            # System update pipeline - ordered by dependencies and data flow
            # Each stage has error isolation to prevent cascading failures

            # STAGE 1: Input Processing and Player State
            try:
                # Handle player movement and input
                movement_start = log_performance_start("Player movement processing")
                self.handle_player_movement()
                log_performance_end("Player movement processing", movement_start)

                # Calculate player context for other systems
                player_stats = {
                    "position": (self.player.x, self.player.y),
                    "velocity": getattr(self.player, "velocity", (0, 0)),
                    "resources": getattr(self.player, "resources", {}),
                    "health": getattr(self.player, "health", 100),
                    "fleet_size": getattr(self.player, "fleet_size", 1),
                }
                update_context["player"] = player_stats

            except Exception as stage1_error:
                log_exception(
                    "Error in Player State processing (Stage 1)", stage1_error
                )
                self.notifier.add(
                    "Movement system error", category="error", importance=2
                )

            # STAGE 2: World Systems and Environment
            try:
                # Update asteroid field and environmental elements
                world_start = log_performance_start("World systems update")
                self.update_asteroid_field()
                log_performance_end("World systems update", world_start)

                # Calculate environmental context for other systems
                field_stats = {
                    "asteroid_count": len(getattr(self.field, "asteroids", [])),
                    "field_density": getattr(self.field, "density", 0.5),
                    "resource_availability": getattr(
                        self.field, "resource_level", "medium"
                    ),
                }
                update_context["environment"] = field_stats

            except Exception as stage2_error:
                log_exception(
                    "Error in World Systems processing (Stage 2)", stage2_error
                )
                self.notifier.add(
                    "Environmental systems error", category="error", importance=2
                )

            # STAGE 3: Encounter and Combat Systems
            try:
                # Process encounters and potential combat
                encounter_start = log_performance_start("Encounter and combat systems")
                self.process_encounters()
                log_performance_end("Encounter and combat systems", encounter_start)

                # Track encounter contexts for statistics and future reference
                encounter_stats = {
                    "last_encounter_time": self.stats.get("last_encounter_time", 0),
                    "last_encounter_type": self.stats.get(
                        "last_encounter_type", "none"
                    ),
                    "encounter_frequency": self.stats.get("encounters", 0)
                    / max(1, self.game_time / 60),
                }
                update_context["encounters"] = encounter_stats

            except Exception as stage3_error:
                log_exception(
                    "Error in Encounter/Combat processing (Stage 3)", stage3_error
                )
                self.notifier.add(
                    "Encounter system error", category="error", importance=3
                )

            # STAGE 4: Entity Lifecycle and Simulation
            try:
                self._update_entity_lifecycle(update_context)
            except Exception as stage4_error:
                log_exception(
                    "Error in Entity Lifecycle processing (Stage 4)", stage4_error
                )
                self.notifier.add(
                    "Evolution system error", category="error", importance=2
                )

            # STAGE 5: Resource Management and Economy
            try:
                # Handle mining, trading, and economy updates
                economy_start = log_performance_start(
                    "Economy and resources processing"
                )
                self.handle_auto_mining()
                # Future: Add trading system update here
                log_performance_end("Economy and resources processing", economy_start)

                # Track economic context
                economy_stats = {
                    "player_currency": self.player.currency,
                    "resources_collected": self.stats.get("resources_collected", 0),
                    "trading_opportunities": getattr(self, "available_trades", 0),
                }
                update_context["economy"] = economy_stats

            except Exception as stage5_error:
                log_exception("Error in Economy processing (Stage 5)", stage5_error)
                self.notifier.add(
                    "Resource system error", category="error", importance=2
                )

            # STAGE 6: Fleet and Unit Management
            try:
                # Update player fleet and unit behaviors
                fleet_start = log_performance_start("Fleet management processing")
                self.update_player_fleet()
                log_performance_end("Fleet management processing", fleet_start)

                # Track fleet context
                fleet_stats = {
                    "fleet_size": getattr(self.player, "fleet_size", 1),
                    "fleet_strength": getattr(self.player, "fleet_strength", 100),
                    "formation": getattr(self.player, "formation", "standard"),
                }
                update_context["fleet"] = fleet_stats

            except Exception as stage6_error:
                log_exception(
                    "Error in Fleet Management processing (Stage 6)", stage6_error
                )
                self.notifier.add("Fleet system error", category="error", importance=2)

            # STAGE 7: Game State Evaluation
            # Evaluate game state and trigger events if necessary
            self.evaluate_game_state(update_context)

            # Performance tracking and logging for the entire update cycle
            update_duration = time.time() - play_state_start
            log_performance_metric("play_state_update_time", update_duration * 1000)

            # Log completion with statistics
            if (
                self.frame_counter % 300 == 0
            ):  # Detailed logging every 5 seconds (at 60 FPS)
                logging.info(
                    f"Play state update complete. Frame: {self.frame_counter}, Duration: {update_duration:.4f}s"
                )
                log_memory_usage("Regular play state memory check")

            # Complete performance tracking
            log_performance_end("Play state update", play_state_start, "complete")

        except Exception as e:
            log_exception("Critical error in play state update", e)
            self.notifier.add(
                f"Game system error: {str(e)}", category="error", importance=3
            )

    def _update_entity_lifecycle(self, update_context: Dict[str, Any]) -> None:
        """Update entity lifecycle including races, evolutions, and discoveries.

        This method orchestrates the lifecycle management of game entities,
        including race evolutions and resource discoveries.

        Args:
            update_context: Dictionary containing the current update context
        """
        # Start performance timing
        entity_start = log_performance_start("Entity lifecycle processing")

        try:
            # Process race evolutions
            self._process_race_evolutions()

            # Process resource discoveries
            self._process_discoveries()

            # Update entity-related context
            self._update_entity_context(update_context)

        except Exception as e:
            log_exception("Error in entity lifecycle processing", e)
            self.notifier.add("Entity system error", category="error", importance=2)
        finally:
            # End performance timing
            log_performance_end("Entity lifecycle processing", entity_start)

    def _process_race_evolutions(self) -> None:
        """Process race evolutions with performance tracking."""
        evolution_start = log_performance_start("Race evolution processing")
        try:
            self.check_race_evolutions()
            log_performance_metric("race_evolutions_checked", 1)
        except Exception as e:
            log_exception("Error processing race evolutions", e)
            log_performance_metric("race_evolution_errors", 1)
        finally:
            log_performance_end("Race evolution processing", evolution_start)

    def _process_discoveries(self) -> None:
        """Process resource discoveries with performance tracking."""
        discovery_start = log_performance_start("Resource discovery processing")
        try:
            self.check_for_discoveries()
            log_performance_metric("discoveries_checked", 1)
        except Exception as e:
            log_exception("Error processing discoveries", e)
            log_performance_metric("discovery_errors", 1)
        finally:
            log_performance_end("Resource discovery processing", discovery_start)

    def _update_entity_context(self, update_context: Dict[str, Any]) -> None:
        """Update the entity-related context in the update context dictionary.

        Args:
            update_context: Dictionary containing the current update context
        """
        # Generate entity-related context
        entity_stats = {
            "race_count": len(getattr(self.field, "races", [])),
            "active_races": getattr(self, "race_count", 0),
            "evolution_stage": getattr(self, "evolution_stage", 1),
            "discovery_count": self.stats.get("discoveries", 0),
        }
        update_context["entities"] = entity_stats

    def evaluate_game_state(self, context: Dict[str, Any]) -> None:
        """Evaluate the current game state and trigger appropriate events.

        This method analyzes the game context from all systems and determines
        if any special events, achievements, or state changes should occur.

        Args:
            context: Dictionary containing state information from all game systems
        """
        try:
            # Start performance timing
            eval_start = log_performance_start("Game state evaluation")

            # Evaluate different aspects of game state
            self._check_for_encounter_triggers(context)
            self._track_progression_milestones(context)
            self._analyze_environmental_dangers(context)

            # End performance timing
            log_performance_end("Game state evaluation", eval_start)

        except Exception as e:
            log_exception("Error in game state evaluation", e)
            # Non-critical, so just log but don't notify player

    def _check_for_encounter_triggers(self, context: Dict[str, Any]) -> None:
        """Check if conditions are met to trigger an encounter.

        Args:
            context: Dictionary containing state information from all game systems
        """
        try:
            # Get relevant context data
            player_pos = context.get("player", {}).get("position", (0, 0))
            last_encounter = context.get("encounters", {}).get("last_encounter_time", 0)
            current_time = self.game_time

            # Check for extended peaceful periods (no encounters for 2+ minutes)
            if (
                current_time - last_encounter > 120
                and self.stats.get("encounters", 0) > 0
            ):
                zone = self.get_current_zone(player_pos)
                if zone not in ["central_hub", "safe_zone"]:
                    # Trigger dynamic encounter after periods of inactivity
                    logging.info(
                        f"Extended peace period detected in {zone} ({current_time - last_encounter:.1f}s)"
                    )
                    self.force_encounter("dynamic", player_pos, zone)
        except Exception as e:
            log_exception("Error checking for encounter triggers", e)

    def _track_progression_milestones(self, context: Dict[str, Any]) -> None:
        """Track and notify about progression milestones.

        Args:
            context: Dictionary containing state information from all game systems
        """
        try:
            # Track currency milestones
            player_currency = context.get("economy", {}).get("player_currency", 0)
            currency_milestone = (player_currency // 1000) * 1000
            last_milestone = self.stats.get("last_currency_milestone", 0)

            if currency_milestone > last_milestone and currency_milestone > 0:
                self.stats["last_currency_milestone"] = currency_milestone
                self.notifier.add(
                    f"Milestone: Accumulated {currency_milestone} currency!",
                    category="achievement",
                    importance=2,
                )

            # Additional milestone checks can be added here
        except Exception as e:
            log_exception("Error tracking progression milestones", e)

    def _analyze_environmental_dangers(self, context: Dict[str, Any]) -> None:
        """Analyze environmental conditions for potential dangers.

        Args:
            context: Dictionary containing state information from all game systems
        """
        try:
            # Check asteroid density
            asteroid_count = context.get("environment", {}).get("asteroid_count", 0)
            if asteroid_count > 200:  # High density field
                player_health = context.get("player", {}).get("health", 100)
                if player_health < 50:  # Player in danger
                    self.notifier.add(
                        "Warning: High asteroid density detected, recommend seeking safer area",
                        category="warning",
                        importance=2,
                    )

            # Additional environmental danger checks can be added here
        except Exception as e:
            log_exception("Error analyzing environmental dangers", e)

    def handle_player_movement(self) -> None:
        """Handle player movement based on keyboard input."""
        try:
            # Start performance timing
            movement_start = log_performance_start("Player keyboard movement")

            # Get movement direction from keyboard input
            dx, dy = self._get_movement_direction()

            # Apply movement if direction is non-zero
            if dx != 0 or dy != 0:
                self._apply_player_movement(dx, dy)

            # End performance timing
            log_performance_end("Player keyboard movement", movement_start)

        except Exception as e:
            log_exception("Error handling player movement", e)

    @staticmethod
    def _get_movement_direction() -> Tuple[int, int]:
        """Get movement direction based on keyboard input.

        Returns:
            Tuple containing x and y direction (-1, 0, or 1 for each)
        """
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0

        # Horizontal movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx = -1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx = 1

        # Vertical movement
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy = -1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy = 1

        return dx, dy

    def _apply_player_movement(self, dx: int, dy: int) -> None:
        """Apply movement to player with the given direction.

        Args:
            dx: Horizontal direction (-1, 0, or 1)
            dy: Vertical direction (-1, 0, or 1)
        """
        # Move the player in the field
        self.player.move(dx, dy, self.field)

        # Set a flag to indicate the player has moved (for encounter checks)
        self.player.has_moved = True

        # Log movement for analytics
        log_performance_metric("player_movement", 1)

    def handle_auto_mining(self) -> None:
        """Handle auto-mining if enabled."""
        try:
            # Start performance timing
            mining_start = log_performance_start(LOG_AUTO_MINING_CHECK)

            # Check if auto-mining should be performed this frame
            if not self._should_perform_auto_mining():
                log_performance_end(LOG_AUTO_MINING_CHECK, mining_start, "skipped")
                return

            # Perform the auto-mining operation
            self._perform_auto_mining_operation()

            # End performance timing
            log_performance_end(LOG_AUTO_MINING_CHECK, mining_start, "complete")

        except Exception as e:
            log_exception("Error in auto-mining", e)

    def _should_perform_auto_mining(self) -> bool:
        """Check if auto-mining should be performed this frame.

        Returns:
            True if auto-mining should be performed, False otherwise
        """
        # Check if auto-mining is enabled
        return self.frame_counter % 30 == 0 if self.auto_mine else False

    def _perform_auto_mining_operation(self) -> None:
        """Perform the auto-mining operation and handle the results."""
        # Try to mine with error handling
        with LogContext("Auto-mining operation"):
            minerals_mined = self.player.mine(self.field)

            # Update statistics and notify if successful
            if minerals_mined > 0:
                self._process_successful_mining(minerals_mined)

    def _process_successful_mining(self, minerals_mined: int) -> None:
        """Process the results of a successful mining operation.

        Args:
            minerals_mined: The number of minerals mined
        """
        # Update statistics
        self.stats["total_mined"] += minerals_mined

        # Log performance metric
        log_performance_metric("minerals_mined", minerals_mined)

        # Only notify occasionally to avoid spamming
        if self.frame_counter % 120 == 0:  # Once every 2 seconds
            self.notifier.add(
                f"Auto-mined {minerals_mined} minerals",
                category="mining",
                importance=1,
            )

    def update_asteroid_field(self) -> None:
        """Update asteroid field at intervals."""
        try:
            # Start performance timing
            field_update_start = log_performance_start(LOG_ASTEROID_FIELD_UPDATE)

            # Check if it's time to update the field
            if not self._should_update_asteroid_field():
                log_performance_end(
                    LOG_ASTEROID_FIELD_UPDATE, field_update_start, "skipped"
                )
                return

            # Update the field and process results
            race_incomes = self._perform_asteroid_field_update()

            # Process significant race incomes
            self._process_significant_race_incomes(race_incomes)

            # End performance timing
            log_performance_end(
                LOG_ASTEROID_FIELD_UPDATE, field_update_start, "complete"
            )

        except Exception as e:
            log_exception("Error updating asteroid field", e)

    def _should_update_asteroid_field(self) -> bool:
        """Check if the asteroid field should be updated based on elapsed time.

        Returns:
            True if the field should be updated, False otherwise
        """
        current_time = time.time()
        return current_time - self.last_field_update_time > self.update_interval / 60

    def _perform_asteroid_field_update(self) -> Dict[int, int]:
        """Update the asteroid field and track the update time.

        Returns:
            Dictionary mapping race IDs to their income from the update
        """
        race_incomes = self.field.update()
        self.last_field_update_time = time.time()
        return race_incomes

    def _process_significant_race_incomes(self, race_incomes: Dict[int, int]) -> None:
        """Process race incomes and notify about significant events.

        Args:
            race_incomes: Dictionary mapping race IDs to their income
        """
        # Define threshold for significant income
        SIGNIFICANT_INCOME_THRESHOLD = 500

        # Process race incomes and notify about significant events
        for race_id, income in race_incomes.items():
            # Skip insignificant incomes
            if income <= SIGNIFICANT_INCOME_THRESHOLD:
                continue

            if race := next(
                (r for r in self.field.races if r.race_id == race_id), None
            ):
                # Notify about the significant income
                self.notifier.add(
                    f"Race {race_id} ({race.trait}) found a large mineral deposit!",
                    category="race",
                    importance=2,
                )

                # Log the event for analytics
                log_performance_metric("significant_race_income", income)

    def check_race_evolutions(self) -> None:
        """Check for race evolutions."""
        try:
            # Start performance timing
            evolution_start = log_performance_start(LOG_RACE_EVOLUTION_CHECK)

            # Check each race for potential evolution
            evolution_count = 0
            for race in self.field.races:
                if self._should_race_evolve(race):
                    self._process_race_evolution(race)
                    evolution_count += 1

            # Log evolution statistics
            if evolution_count > 0:
                log_performance_metric("race_evolutions_processed", evolution_count)

            # End performance timing
            log_performance_end(
                LOG_RACE_EVOLUTION_CHECK,
                evolution_start,
                "evolved" if evolution_count > 0 else "none",
            )

        except Exception as e:
            log_exception("Error checking race evolutions", e)

    def _should_race_evolve(self, race: Any) -> bool:
        """Determine if a race should evolve based on evolution points and timing.

        Args:
            race: The race to check for evolution eligibility

        Returns:
            True if the race should evolve, False otherwise
        """
        # Check if race has enough evolution points
        if race.evolution_points < race.evolution_threshold:
            return False

        # Stagger evolutions based on race ID to prevent all races evolving at once
        return self.frame_counter % 300 == race.race_id % 300

    def _process_race_evolution(self, race: Any) -> None:
        """Process the evolution of a race and handle related events.

        Args:
            race: The race to evolve
        """
        # Race evolves and get metrics
        metrics = race.evolve()

        # Notify the player
        self.notifier.notify_event(
            "race",
            f"Race {race.race_id} ({race.trait}) has evolved to stage {race.evolution_stage}!",
            importance=2,
        )

        # Track in stats
        self.stats["race_evolutions"] = self.stats.get("race_evolutions", 0) + 1

        # Analyze and log territory control
        self._log_territory_metrics(race, metrics)

    @staticmethod
    def _log_territory_metrics(
        race: Any, metrics: Optional[Dict[str, Any]]
    ) -> None:
        """Log territory metrics for a race if available.

        Args:
            race: The race whose territory to log
            metrics: Dictionary containing territory metrics, or None
        """
        if metrics and metrics.get("center"):
            logging.info(
                f"Race {race.race_id} territory: "
                + f"center={metrics['center']}, radius={metrics.get('radius', 0)}, "
                + f"density={metrics.get('density', 0):.4f}"
            )

            # Log metrics for performance tracking
            log_performance_metric(
                f"race_{race.race_id}_territory_radius", metrics.get("radius", 0)
            )
            log_performance_metric(
                f"race_{race.race_id}_territory_density", metrics.get("density", 0)
            )

    def check_for_discoveries(self) -> None:
        """Check for new resource discoveries by player."""
        try:
            # Start performance timing for discovery check
            discovery_start = log_performance_start(LOG_DISCOVERY_CHECK)

            # Check if discovery scan should be performed this frame
            if not self._should_perform_discovery_scan():
                log_performance_end(LOG_DISCOVERY_CHECK, discovery_start, "skipped")
                return

            # Perform the discovery scan and track results
            discoveries_made = self._perform_discovery_scan()

            # End performance timing with appropriate result status
            log_performance_end(
                LOG_DISCOVERY_CHECK,
                discovery_start,
                "found" if discoveries_made else "none",
            )

            # Track metrics for analytics
            log_performance_metric("discovery_scan_completed", 1)
            if discoveries_made:
                log_performance_metric("discoveries_made", 1)

        except Exception as e:
            log_exception("Error checking for discoveries", e)
            log_performance_metric("discovery_scan_errors", 1)

    def _should_perform_discovery_scan(self) -> bool:
        """Determine if a discovery scan should be performed this frame.

        Returns:
            bool: True if a discovery scan should be performed, False otherwise
        """
        # Only run this check periodically to avoid performance impact
        return self.frame_counter % 120 == 0  # Every 2 seconds at 60 FPS

    def _perform_discovery_scan(self) -> bool:
        """Perform a discovery scan around the player.

        Returns:
            bool: True if any discoveries were made, False otherwise
        """
        # Track whether anything was discovered in this scan
        discoveries_made = False

        # Use LogContext for proper error handling and tracking
        with LogContext("Resource discovery scan"):
            # Validate player and field before proceeding
            if not self._validate_discovery_prerequisites():
                return False

            # Check for nearby anomalies
            discoveries_made = self._scan_for_anomalies()

        return discoveries_made

    def _validate_discovery_prerequisites(self) -> bool:
        """Validate that all prerequisites for discovery scanning are met.

        Returns:
            bool: True if all prerequisites are valid, False otherwise
        """
        # Safety check for player position
        if not hasattr(self.player, "x") or not hasattr(self.player, "y"):
            logging.error("Player position attributes missing")
            return False

        # Ensure field dimensions are valid
        if not hasattr(self.field, "width") or not hasattr(self.field, "height"):
            logging.error("Field dimension attributes missing")
            return False

        # Validate player is within field bounds
        if not (
            0 <= self.player.x < self.field.width
            and 0 <= self.player.y < self.field.height
        ):
            logging.warning(
                f"Player outside field bounds: ({self.player.x}, {self.player.y})"
            )
            return False

        return True

    def _scan_for_anomalies(self) -> bool:
        """Scan the area around the player for anomalies.

        Returns:
            bool: True if any discoveries were made, False otherwise
        """
        discoveries_made = False
        discovery_range = 3  # How far to look for discoveries

        for dy, dx in itertools.product(
            range(-discovery_range, discovery_range + 1),
            range(-discovery_range, discovery_range + 1),
        ):
            nx, ny = self.player.x + dx, self.player.y + dy

            # Validate coordinates are within bounds
            if not (0 <= nx < self.field.width and 0 <= ny < self.field.height):
                continue

            # Check if this is an anomaly worth reporting
            if self._is_reportable_anomaly(nx, ny):
                discoveries_made = True
                # Process the discovery if it's new
                self._process_new_discovery(nx, ny)

        return discoveries_made

    def _is_reportable_anomaly(self, x: int, y: int) -> bool:
        """Check if the given coordinates contain a reportable anomaly.

        Args:
            x: X-coordinate to check
            y: Y-coordinate to check

        Returns:
            bool: True if the location contains a reportable anomaly
        """
        try:
            # Define anomaly type constant
            ANOMALY_TYPE = 3

            # Check if there's an asteroid at this location
            has_asteroid = self.field.grid[y, x] > 0

            # Check if there's an anomaly at this location
            is_anomaly = self.field.rare_grid[y, x] == ANOMALY_TYPE

            # Both conditions must be true for a reportable anomaly
            return has_asteroid and is_anomaly

        except (IndexError, AttributeError) as e:
            # Log the error and return False for safety
            logging.warning(f"Error checking anomaly at ({x}, {y}): {e}")
            return False

    def _process_new_discovery(self, x: int, y: int) -> None:
        """Process a newly discovered anomaly.

        Args:
            x: X-coordinate of the discovery
            y: Y-coordinate of the discovery
        """
        try:
            # Check if this anomaly has already been discovered
            if self._is_already_discovered(x, y):
                return

            # Record the new discovery
            self._record_discovery(x, y)

            # Notify the player about the discovery
            self._notify_discovery(x, y)

            # Update game statistics
            self._update_discovery_statistics()

        except Exception as e:
            logging.error(f"Error processing discovery at ({x}, {y}): {e}")

    def _is_already_discovered(self, x: int, y: int) -> bool:
        """Check if the given coordinates have already been discovered.

        Args:
            x: X-coordinate to check
            y: Y-coordinate to check

        Returns:
            bool: True if already discovered, False otherwise
        """
        # Initialize the set if it doesn't exist
        if not hasattr(self, "discovered_anomalies"):
            self.discovered_anomalies = set()

        # Check if the coordinates are in the discovered set
        return (x, y) in self.discovered_anomalies

    def _record_discovery(self, x: int, y: int) -> None:
        """Record a new discovery in the discovered set.

        Args:
            x: X-coordinate of the discovery
            y: Y-coordinate of the discovery
        """
        # Add to discovered set
        self.discovered_anomalies.add((x, y))

    def _notify_discovery(self, x: int, y: int) -> None:
        """Notify the player about a new discovery.

        Args:
            x: X-coordinate of the discovery
            y: Y-coordinate of the discovery
        """
        # Notify the player
        self.notifier.notify_event(
            "discovery",
            f"Found an anomaly at ({x}, {y})!",
            importance=3,
        )

        # Log the discovery
        logging.info(f"Player discovered anomaly at ({x}, {y})")

    def _update_discovery_statistics(self) -> None:
        """Update game statistics related to discoveries."""
        # Update total discoveries counter
        self.stats["total_anomalies_discovered"] = (
            self.stats.get("total_anomalies_discovered", 0) + 1
        )

        # Track for analytics
        log_performance_metric("anomaly_discovered", 1)

    def update_player_fleet(self) -> None:
        """Update the player's fleet status and process results."""
        try:
            # Start performance timing
            fleet_update_start = log_performance_start(LOG_FLEET_UPDATE)

            # Check if we should update the fleet this frame
            if not self._should_update_fleet():
                log_performance_end(LOG_FLEET_UPDATE, fleet_update_start, "skipped")
                return

            if fleet_results := self._perform_fleet_update():
                self._process_fleet_update_results(fleet_results)

            # End performance timing
            log_performance_end(LOG_FLEET_UPDATE, fleet_update_start, "complete")

        except Exception as e:
            log_exception("Error updating player fleet", e)
            self.notifier.add("Fleet system error", category="error", importance=2)

    def _should_update_fleet(self) -> bool:
        """Determine if the fleet should be updated this frame.

        Returns:
            True if the fleet should be updated, False otherwise
        """
        # Only update fleet periodically to avoid performance impact
        return self.frame_counter % 60 == 0  # Once per second at 60 FPS

    def _perform_fleet_update(self) -> Optional[Dict[str, Any]]:
        """Perform the fleet update operation.

        Returns:
            Dictionary containing fleet update results, or None if no update occurred
        """
        with LogContext(f"{LOG_FLEET_UPDATE} operation"):
            # Update the fleet and get results
            fleet_results = self.player.update_fleet(self.field)

            # Log the update for analytics
            log_performance_metric("fleet_update_performed", 1)

            if fleet_results is None:
                logging.debug("No fleet update results to process")

            return fleet_results

    def _process_fleet_update_results(self, fleet_results: Dict[str, Any]) -> None:
        """Process the results of a fleet update.

        Args:
            fleet_results: Dictionary containing fleet update results
        """
        # Process damaged ships
        self._handle_damaged_ships(fleet_results)

        # Process lost ships
        self._handle_lost_ships(fleet_results)

        # Log overall fleet status
        self._log_fleet_status(fleet_results)

    def _handle_damaged_ships(self, fleet_results: Dict[str, Any]) -> None:
        """Handle damaged ships from fleet update results.

        Args:
            fleet_results: Dictionary containing fleet update results
        """
        damaged_count = fleet_results.get("ships_damaged", 0)

        if damaged_count > 0:
            # Notify the player
            self.notifier.add(
                f"{damaged_count} ships damaged!",
                category="fleet",
                importance=2,
            )

            # Log the event
            logging.info(f"Player fleet update: {damaged_count} ships damaged")

            # Track for analytics
            log_performance_metric("ships_damaged", damaged_count)

    def _handle_lost_ships(self, fleet_results: Dict[str, Any]) -> None:
        """Handle lost ships from fleet update results.

        Args:
            fleet_results: Dictionary containing fleet update results
        """
        lost_count = fleet_results.get("ships_lost", 0)

        if lost_count > 0:
            # Notify the player
            self.notifier.add(
                f"{lost_count} ships lost in dangerous territory!",
                category="fleet",
                importance=3,
            )

            # Update statistics
            self.stats["ships_lost"] = self.stats.get("ships_lost", 0) + lost_count

            # Track for analytics
            log_performance_metric("ships_lost", lost_count)

    def _log_fleet_status(self, fleet_results: Dict[str, Any]) -> None:
        """Log the current fleet status for analytics and debugging.

        Args:
            fleet_results: Dictionary containing fleet update results
        """
        # Log overall fleet status
        current_fleet_size = getattr(self.player, "fleet_size", 1)
        fleet_health = fleet_results.get("fleet_health", 100)

        logging.debug(
            f"Fleet status: Size={current_fleet_size}, Health={fleet_health}%, "
            f"Damaged={fleet_results.get('ships_damaged', 0)}, "
            f"Lost={fleet_results.get('ships_lost', 0)}"
        )

        # Log specific fleet events
        if fleet_results.get("ships_damaged", 0) > 0:
            logging.warning(
                f"Player fleet update: {fleet_results.get('ships_damaged', 0)} ships damaged"
            )

        if fleet_results.get("ships_lost", 0) > 0:
            logging.warning(
                f"Player fleet update: {fleet_results.get('ships_lost', 0)} ships lost"
            )

        # Handle mining income
        if fleet_results.get("minerals_mined", 0) > 0:
            minerals_gained = fleet_results["minerals_mined"]
            self.stats["total_mined"] += minerals_gained

            # Log all mining activity but only notify player of significant amounts
            logging.info(f"Fleet mined {minerals_gained} minerals")
            if minerals_gained > 100:
                self.notifier.add(
                    f"Fleet mined {minerals_gained} minerals",
                    category="mining",
                    importance=1,
                )

                # Track performance metrics if available
                if "processing_time" in fleet_results:
                    log_performance_metric(
                        "fleet_update_processing", fleet_results["processing_time"]
                    )

    def process_encounters(self) -> None:
        """Process potential encounters based on player movement and location.

        This method checks for and handles all encounter types including combat,
        discovery, and trader encounters. It manages the encounter lifecycle from
        detection through resolution, with comprehensive performance tracking.
        """
        # Start performance timing for encounter processing
        encounter_start = log_performance_start(LOG_ENCOUNTER_PROCESSING)

        try:
            # Track last encounter check time for throttling
            current_time = time.time()
            last_check_time = getattr(self, "last_encounter_check_time", 0)
            check_interval = 1.5  # Seconds between checks

            # Only check for encounters if player has moved or if enough time has passed
            player_moved = getattr(
                self.player, "has_moved", False
            )  # More robust attribute check
            time_to_check = (current_time - last_check_time) >= check_interval

            if not (player_moved or time_to_check):
                log_performance_end(
                    LOG_ENCOUNTER_PROCESSING, encounter_start, "throttled"
                )
                return

            # Update last check time
            self.last_encounter_check_time = current_time

            # Reset player movement flag if it exists
            if hasattr(self.player, "has_moved"):
                self.player.has_moved = False

            # Get current player position and zone for context
            player_pos = (self.player.x, self.player.y)
            current_zone = self.get_current_zone(player_pos)

            # Log zone transition if it changed
            if hasattr(self, "last_zone") and self.last_zone != current_zone:
                logging.info(
                    f"Player transitioned from {self.last_zone} to {current_zone}"
                )
                self.notifier.add(
                    f"Entering {current_zone}", category="navigation", importance=1
                )
            self.last_zone = current_zone

            # Check for potential encounter with appropriate context
            with LogContext("Encounter check"):
                # Track checks by zone
                zone_checks_key = f"encounter_checks_{current_zone}"
                self.stats[zone_checks_key] = self.stats.get(zone_checks_key, 0) + 1

                # Pass current gameplay context to encounter generator
                encounter_context = {
                    "player_health": self.player.health,
                    "player_level": (
                        self.player.level if hasattr(self.player, "level") else 1
                    ),
                    "player_currency": self.player.currency,
                    "game_time": self.game_time,
                    "zone": current_zone,
                }

                # Enhanced encounter check with more context
                encounter_check_start = time.time()
                encounter_result = self.encounter_generator.check_for_encounter(
                    player_pos, encounter_context
                )
                log_performance_metric(
                    "encounter_check_time", (time.time() - encounter_check_start) * 1000
                )

                if not encounter_result:
                    log_performance_end(
                        "Encounter processing", encounter_start, "no_encounter"
                    )
                    return

                # Process the encounter that was triggered
                encounter_type = encounter_result.get("type", "unknown")
                encounter_zone = encounter_result.get("zone", current_zone)
                encounter_rarity = encounter_result.get("rarity", "common")

                # Track encounter by zone and type
                self.stats[f"encounters_in_{encounter_zone}"] = (
                    self.stats.get(f"encounters_in_{encounter_zone}", 0) + 1
                )
                self.stats[f"encounters_{encounter_type}"] = (
                    self.stats.get(f"encounters_{encounter_type}", 0) + 1
                )
                self.stats[f"encounters_{encounter_rarity}"] = (
                    self.stats.get(f"encounters_{encounter_rarity}", 0) + 1
                )

                # Process specific encounter types
                if encounter_type == "combat":
                    self._handle_combat_encounter(encounter_result)

                elif encounter_type == "discovery":
                    self._handle_discovery_encounter(encounter_result)

                elif encounter_type == "trader":
                    self._handle_trader_encounter(encounter_result)

                else:  # Handle unknown encounter types gracefully
                    logging.warning(f"Unknown encounter type: {encounter_type}")
                    self.notifier.add(
                        "Strange phenomenon detected",
                        category="encounter",
                        importance=2,
                    )

                # Update global encounter statistics
                self.stats["encounters"] = self.stats.get("encounters", 0) + 1
                self.stats["last_encounter_time"] = self.game_time
                self.stats["last_encounter_type"] = encounter_type
                self.stats["last_encounter_zone"] = encounter_zone

                # Log the encounter details
                logging.info(
                    f"Player encountered {encounter_type} ({encounter_rarity}) at {player_pos} in {encounter_zone}"
                )

            # End performance timing with encounter details
            log_performance_end(
                LOG_ENCOUNTER_PROCESSING,
                encounter_start,
                f"{encounter_type}_{encounter_rarity}",
            )

        except Exception as e:
            logging.error(f"Error processing encounters: {str(e)}")
            log_exception("Encounter processing error", e)
            log_performance_end(LOG_ENCOUNTER_PROCESSING, encounter_start, "error")

    @staticmethod
    def get_current_zone(position: Tuple[int, int]) -> str:
        """Determine which zone the player is currently in based on position.

        Args:
            position: Tuple containing (x, y) coordinates

        Returns:
            str: The name of the current zone
        """
        x, y = position

        # Simple zone determination based on distance from center
        # Could be enhanced with more sophisticated zone mapping
        distance = math.sqrt(x**2 + y**2)

        if distance < 500:
            return "central_hub"
        elif distance < 1500:
            return "inner_zone"
        elif distance < 3000:
            return "middle_zone"
        elif distance < 5000:
            return "outer_zone"
        else:
            return "deep_space"

    def _handle_combat_encounter(self, encounter_result: Dict[str, Any]) -> None:
        """Handle a combat-type encounter.

        Args:
            encounter_result: Dictionary containing encounter information
        """
        combat_start = time.time()

        enemy = encounter_result.get("enemy")
        if not enemy:
            logging.warning("Combat encounter missing enemy information")
            return

        # Notify the player
        self.notifier.add(
            f"Encountered hostile {enemy.name}!", category="encounter", importance=3
        )

        # Initiate combat and process results
        try:
            # Track combat initiation performance
            combat_initiation_start = time.time()
            combat_result = self.combat_system.initiate_combat(enemy)
            log_performance_metric(
                "combat_initiation_time", (time.time() - combat_initiation_start) * 1000
            )

            # Process the combat result
            self.process_combat_result(combat_result)

            # Track overall combat handling time
            log_performance_metric(
                "total_combat_handling_time", (time.time() - combat_start) * 1000
            )

        except Exception as e:
            log_exception("Error during combat processing", e)
            self.notifier.add(
                "Combat system malfunction", category="error", importance=3
            )

    def _handle_discovery_encounter(self, encounter_result: Dict[str, Any]) -> None:
        """Handle a discovery-type encounter.

        Args:
            encounter_result: Dictionary containing encounter information
        """
        discovery = encounter_result.get("discovery")
        if not discovery:
            logging.warning("Discovery encounter missing discovery information")
            return

        # Get discovery details
        discovery_name = discovery.get("name", "Unknown object")
        discovery_type = discovery.get("type", "general")
        discovery_rarity = discovery.get("rarity", "common")

        # Show appropriate notification based on rarity
        importance = 1
        if discovery_rarity == "rare":
            importance = 2
        elif discovery_rarity in ["epic", "legendary"]:
            importance = 3

        # Notify the player
        self.notifier.add(
            f"Discovered {discovery_name}!", category="discovery", importance=importance
        )

        # Apply discovery effects
        rewards_text = []

        if "currency" in discovery:
            currency_amount = discovery["currency"]
            self.player.currency += currency_amount
            rewards_text.append(f"{currency_amount} currency")

            # Track discovery rewards by type
            self.stats["discovery_currency_gained"] = (
                self.stats.get("discovery_currency_gained", 0) + currency_amount
            )

        if "items" in discovery and discovery["items"]:
            items_count = len(discovery["items"])
            rewards_text.append(f"{items_count} items")

            # Track items discovered
            self.stats["discovery_items_found"] = (
                self.stats.get("discovery_items_found", 0) + items_count
            )

        if "research" in discovery:
            research_points = discovery["research"]
            # If we implement a research system later
            rewards_text.append(f"{research_points} research points")

        # Comprehensive reward notification if multiple rewards
        if rewards_text:
            self.notifier.add(
                f"Obtained: {', '.join(rewards_text)}",
                category="discovery",
                importance=1,
            )

        # Update discovery statistics
        self.stats["discoveries"] = self.stats.get("discoveries", 0) + 1
        self.stats[f"discoveries_{discovery_type}"] = (
            self.stats.get(f"discoveries_{discovery_type}", 0) + 1
        )
        self.stats[f"discoveries_{discovery_rarity}"] = (
            self.stats.get(f"discoveries_{discovery_rarity}", 0) + 1
        )

    def _handle_trader_encounter(self, encounter_result: Dict[str, Any]) -> None:
        """Handle a trader-type encounter.

        Args:
            encounter_result: Dictionary containing encounter information
        """
        trader = encounter_result.get("trader", {})
        trader_name = trader.get("name", "Wandering Trader")
        trader_type = trader.get("type", "general")

        # Notify the player
        self.notifier.add(
            f"Encountered {trader_name}", category="encounter", importance=2
        )

        # Add trader-specific details if available
        if "specialty" in trader:
            self.notifier.add(
                f"Specializes in {trader['specialty']}", category="trader", importance=1
            )

        # Could transition to a trading state here in the future
        # self.state = "TRADE"
        # self.current_trader = trader

        # Update trader statistics
        self.stats["trader_encounters"] = self.stats.get("trader_encounters", 0) + 1
        self.stats[f"trader_{trader_type}_encounters"] = (
            self.stats.get(f"trader_{trader_type}_encounters", 0) + 1
        )

    def force_encounter(
        self, encounter_type: str, position: Tuple[int, int], zone: str = None
    ) -> None:
        """Force a specific type of encounter to occur at the given position.

        This method provides direct integration between the game loop and encounter
        generator, allowing the game to dynamically create encounters based on
        gameplay needs rather than just random chance.

        Args:
            encounter_type: Type of encounter to generate ("combat", "discovery", "trader", or "dynamic")
            position: Position (x, y) where the encounter should occur
            zone: Optional zone identifier to contextualize the encounter

        Returns:
            None: The encounter is processed immediately
        """
        try:
            # Start performance timing
            force_encounter_start = log_performance_start(LOG_FORCE_ENCOUNTER)

            # Log the forced encounter request
            logging.info(
                f"Forcing {encounter_type} encounter at {position} in {zone or 'current zone'}"
            )

            # If zone not specified, determine it from the position
            if zone is None:
                zone = self.get_current_zone(position)

            # Create encounter context and determine the actual encounter type
            context = self._create_encounter_context(encounter_type, zone)
            chosen_type = self._resolve_encounter_type(encounter_type, context)

            # Generate and process the encounter
            if not self._generate_and_process_encounter(
                chosen_type, position, context, force_encounter_start
            ):
                return

            # Update statistics for forced encounters
            self._update_forced_encounter_stats(chosen_type)

            # Complete performance timing
            log_performance_end(LOG_FORCE_ENCOUNTER, force_encounter_start, "success")

        except Exception as e:
            log_exception("Error forcing encounter", e)
            log_performance_end(LOG_FORCE_ENCOUNTER, force_encounter_start, "error")
            self.notifier.add(
                "Error generating encounter", category="error", importance=2
            )
            log_exception("Error processing encounters", e)

    def _create_encounter_context(
        self, encounter_type: str, zone: str
    ) -> Dict[str, Any]:
        """Create the context dictionary for encounter generation.

        Args:
            encounter_type: The type of encounter to generate
            zone: The zone where the encounter will occur

        Returns:
            Dict containing the encounter context
        """
        return {
            "player_health": self.player.health,
            "player_level": getattr(self.player, "level", 1),
            "player_currency": self.player.currency,
            "game_time": self.game_time,
            "zone": zone,
            "forced": True,  # Flag that this was a forced encounter
            "encounter_type": encounter_type,
        }

    def _resolve_encounter_type(
        self, encounter_type: str, context: Dict[str, Any]
    ) -> str:
        """Resolve the actual encounter type, handling 'dynamic' type specially.

        Args:
            encounter_type: The requested encounter type
            context: The encounter context dictionary

        Returns:
            str: The resolved encounter type
        """
        if encounter_type != "dynamic":
            return encounter_type

        # Dynamic selection based on current game state
        hour = int(self.game_time / 60) % 24  # Game time hour (0-23)
        player_health_pct = self.player.health / 100.0

        # Evening hours tend toward combat, daytime toward discoveries
        if 18 <= hour <= 23 or 0 <= hour <= 5:  # Night hours
            combat_weight = 0.7
            discovery_weight = 0.1
        else:  # Day hours
            combat_weight = 0.3
            discovery_weight = 0.5
        trader_weight = 0.2

        # Adjust weights based on player health
        if player_health_pct < 0.3:  # Low health, reduce combat chance
            combat_weight *= 0.5
            trader_weight *= 1.5

        # Normalize weights
        total_weight = combat_weight + discovery_weight + trader_weight
        combat_weight /= total_weight
        discovery_weight /= total_weight
        trader_weight /= total_weight

        # Determine encounter type based on weights
        rand_val = random.random()
        if rand_val < combat_weight:
            chosen_type = "combat"
        elif rand_val < combat_weight + discovery_weight:
            chosen_type = "discovery"
        else:
            chosen_type = "trader"

        # Update context with chosen type
        context["encounter_type"] = chosen_type
        logging.info(f"Dynamic encounter resolved to {chosen_type} encounter")
        return chosen_type

    def _generate_and_process_encounter(
        self,
        encounter_type: str,
        position: Tuple[int, int],
        context: Dict[str, Any],
        timing_start: float,
    ) -> bool:
        """Generate and process an encounter of the specified type.

        Args:
            encounter_type: The type of encounter to generate
            position: The position where the encounter occurs
            context: The encounter context
            timing_start: The start time for performance logging

        Returns:
            bool: True if encounter was successfully generated and processed, False otherwise
        """
        # Generate the encounter through the encounter generator
        encounter_result = self.encounter_generator.generate_encounter(
            position, encounter_type, context
        )

        if not encounter_result:
            logging.warning(f"Failed to generate forced {encounter_type} encounter")
            log_performance_end(LOG_FORCE_ENCOUNTER, timing_start, "failed")
            return False

        # Process the generated encounter immediately
        if encounter_type == "combat":
            self._handle_combat_encounter(encounter_result)
        elif encounter_type == "discovery":
            self._handle_discovery_encounter(encounter_result)
        elif encounter_type == "trader":
            self._handle_trader_encounter(encounter_result)

        return True

    def _update_forced_encounter_stats(self, encounter_type: str) -> None:
        """Update statistics for forced encounters.

        Args:
            encounter_type: The type of encounter that was forced
        """
        self.stats["forced_encounters"] = self.stats.get("forced_encounters", 0) + 1
        self.stats[f"forced_{encounter_type}_encounters"] = (
            self.stats.get(f"forced_{encounter_type}_encounters", 0) + 1
        )

    def process_combat_result(self, combat_result: Dict[str, Any]) -> None:
        """Process the results of a combat encounter.

        Args:
            combat_result: Dictionary containing combat results
        """
        # Start performance monitoring for combat result processing
        combat_processing_start = log_performance_start(LOG_COMBAT_RESULT)

        try:
            if not combat_result:
                log_performance_end(
                    LOG_COMBAT_RESULT, combat_processing_start, "empty_result"
                )
                return

            # Extract and log combat details
            combat_info = self._extract_combat_info(combat_result)
            self._log_combat_outcome(combat_info)

            # Process based on outcome
            outcome = combat_info["outcome"]
            if outcome == "victory":
                self._process_victory(combat_info)
            elif outcome == "defeat":
                self._process_defeat(combat_info)
            elif outcome == "escape":
                self._process_escape(combat_info)

            # Update comprehensive combat statistics
            self._update_combat_statistics(combat_info)

            # Performance tracking for specific combat types
            log_performance_metric(
                f"combat_processing_{outcome}", time.time() - combat_processing_start
            )

            # Log combat result
            logging.info(
                f"Combat result: {outcome} with {len(combat_result.get('log', []))} actions"
            )

            log_performance_end(LOG_COMBAT_RESULT, combat_processing_start)

        except Exception as e:
            log_exception("Error processing combat result", e)
            log_performance_end(LOG_COMBAT_RESULT, combat_processing_start, "error")
            # Attempt recovery
            try:
                self.player.reset_fleet_state()
                logging.info("Reset fleet state after error")
            except Exception as recovery_error:
                log_exception("Failed to recover from fleet error", recovery_error)

    @staticmethod
    def _extract_combat_info(combat_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant information from combat result.

        Args:
            combat_result: Raw combat result dictionary

        Returns:
            Dictionary with extracted and normalized combat information
        """
        return {
            "outcome": combat_result.get("outcome", "unknown"),
            "enemy_type": combat_result.get("enemy_type", "unknown"),
            "zone": combat_result.get("zone", "unknown"),
            "difficulty": combat_result.get("difficulty", 1),
            "rewards": combat_result.get("rewards", {}),
            "penalties": combat_result.get("penalties", {}),
            "escape_cost": combat_result.get("escape_cost", 0),
            "stats": combat_result.get("stats", {}),
            "raw_result": combat_result,  # Keep original data for reference
        }

    @staticmethod
    def _log_combat_outcome(combat_info: Dict[str, Any]) -> None:
        """Log details about the combat outcome.

        Args:
            combat_info: Processed combat information
        """
        logging.info(
            f"Combat outcome: {combat_info['outcome']} against {combat_info['enemy_type']} "
            f"in zone {combat_info['zone']} (difficulty: {combat_info['difficulty']})"
        )

    def _process_victory(self, combat_info: Dict[str, Any]) -> None:
        """Process a victorious combat outcome.

        Args:
            combat_info: Processed combat information
        """
        # Notify player of victory
        self.notifier.add(
            f"Victory! {combat_info['enemy_type']} defeated",
            category="combat",
            importance=2,
        )

        # Process rewards
        total_reward_value = self._process_combat_rewards(combat_info["rewards"])

        # Track reward value by difficulty
        difficulty = combat_info["difficulty"]
        difficulty_key = f"reward_value_diff_{int(difficulty)}"
        self.stats[difficulty_key] = (
            self.stats.get(difficulty_key, 0) + total_reward_value
        )

    def _process_defeat(self, combat_info: Dict[str, Any]) -> None:
        """Process a defeat combat outcome.

        Args:
            combat_info: Processed combat information
        """
        # Notify player of defeat
        self.notifier.add(
            f"Defeat! Your ships were overwhelmed by {combat_info['enemy_type']}",
            category="combat",
            importance=3,
        )

        # Process penalties
        self._process_combat_penalties(
            combat_info["penalties"], combat_info["difficulty"]
        )

    def _process_escape(self, combat_info: Dict[str, Any]) -> None:
        """Process an escape combat outcome.

        Args:
            combat_info: Processed combat information
        """
        # Notify player of escape
        self.notifier.add(
            f"Escaped from {combat_info['enemy_type']}", category="combat", importance=2
        )

        # Process escape costs
        escape_cost = combat_info["escape_cost"]
        if escape_cost > 0:
            self.notifier.add(
                f"Escape cost: {escape_cost} damage to ships", category="combat"
            )
            # Track escape costs
            self.stats["escape_costs"] = self.stats.get("escape_costs", 0) + escape_cost

        # Track escapes by difficulty
        difficulty = combat_info["difficulty"]
        escape_key = f"escapes_diff_{int(difficulty)}"
        self.stats[escape_key] = self.stats.get(escape_key, 0) + 1

    def _update_combat_statistics(self, combat_info: Dict[str, Any]) -> None:
        """Update comprehensive combat statistics.

        Args:
            combat_info: Processed combat information
        """
        # Track basic combat metrics
        self.stats["total_combats"] = self.stats.get("total_combats", 0) + 1

        # Track outcome-specific stats
        outcome = combat_info["outcome"]
        if outcome == "defeat":
            self.stats["combats_lost"] = self.stats.get("combats_lost", 0) + 1
        elif outcome == "escape":
            self.stats["combats_escaped"] = self.stats.get("combats_escaped", 0) + 1

        elif outcome == "victory":
            self.stats["combats_won"] = self.stats.get("combats_won", 0) + 1
        self._track_combat_by_category(combat_info, "zone", "combats_in_")
        self._track_combat_by_category(combat_info, "enemy_type", "combats_against_")
        if combat_stats := combat_info["stats"]:
            for stat_name, stat_value in combat_stats.items():
                self.stats[f"combat_{stat_name}"] = (
                    self.stats.get(f"combat_{stat_name}", 0) + stat_value
                )

    def _track_combat_by_category(self, combat_info, category_key, stat_prefix):
        # Track combat by category (zone, enemy_type, etc.)
        category_value = combat_info[category_key]
        stat_key = f"{stat_prefix}{category_value}"
        self.stats[stat_key] = self.stats.get(stat_key, 0) + 1

    def _process_combat_rewards(self, rewards: Dict[str, Any]) -> int:
        """Process rewards from combat.

        Args:
            rewards: Dictionary of rewards

        Returns:
            Total value of rewards processed
        """
        total_reward_value = 0

        # Process currency rewards
        if "currency" in rewards:
            currency = rewards["currency"]
            self.player.currency += currency
            total_reward_value += currency

            # Notify player
            self.notifier.add(f"Gained {currency} currency", category="combat")

            # Update stats
            self.stats["combat_currency_gained"] = (
                self.stats.get("combat_currency_gained", 0) + currency
            )

        # Process item rewards
        if "items" in rewards and rewards["items"]:
            items_count = len(rewards["items"])

            # Notify player
            self.notifier.add(f"Acquired {items_count} items", category="combat")

            # Update stats
            self.stats["combat_items_gained"] = (
                self.stats.get("combat_items_gained", 0) + items_count
            )
            # Future: Add items to player inventory

        return total_reward_value

    def _process_combat_penalties(
        self, penalties: Dict[str, Any], difficulty: int = 1
    ) -> None:
        """Process penalties from combat defeat.

        Args:
            penalties: Dictionary of penalties
            difficulty: Combat difficulty level
        """
        total_penalty_value = 0

        # Process currency penalties
        if "currency" in penalties:
            loss = min(self.player.currency, penalties["currency"])
            self.player.currency -= loss
            total_penalty_value += loss

            # Notify player
            if loss > 0:
                self.notifier.add(f"Lost {loss} currency", category="combat")

            # Update stats
            self.stats["combat_currency_lost"] = (
                self.stats.get("combat_currency_lost", 0) + loss
            )

        # Process ship losses
        if "ships" in penalties:
            ships_lost = penalties["ships"]
            self.player.fleet_size = max(1, self.player.fleet_size - ships_lost)

            # Notify player
            if ships_lost > 0:
                self.notifier.add(
                    f"Lost {ships_lost} ships in battle", category="combat"
                )

            # Update stats
            self.stats["ships_lost"] = self.stats.get("ships_lost", 0) + ships_lost

        # Track penalty value by difficulty
        difficulty_key = f"penalty_value_diff_{difficulty}"
        self.stats[difficulty_key] = (
            self.stats.get(difficulty_key, 0) + total_penalty_value
        )

    def draw(self) -> None:
        """Render the current game state."""
        try:
            # Start performance timing
            render_start = log_performance_start("Render frame")

            # Draw the core elements
            self._draw_background()
            self._draw_current_state()
            self._draw_ui_overlays()

            # End performance timing
            log_performance_end("Render frame", render_start)
            
            # Update the display to make the rendered content visible
            pygame.display.flip()

        except Exception as e:
            log_exception("Error in render process", e)

    def _draw_background(self) -> None:
        """Draw the background elements."""
        # Fill background with base color
        self.screen.fill(COLOR_BG)

        # Draw tiled background pattern
        for y in range(0, WINDOW_HEIGHT, self.bg_pattern.get_height()):
            for x in range(0, WINDOW_WIDTH, self.bg_pattern.get_width()):
                self.screen.blit(self.bg_pattern, (x, y))

    def _draw_current_state(self) -> None:
        """Draw the current game state content."""
        # Render content based on current game state
        if self.state == STATE_PLAY:
            self.draw_play_state()
        elif self.state == STATE_SHOP:
            self.draw_shop_state()

    def _draw_ui_overlays(self) -> None:
        """Draw UI overlays that appear on top of the game content."""
        # Draw notifications (always shown)
        self.notifier.draw(self.screen)

        # Draw optional UI elements
        self._draw_tooltip()
        self._draw_controls_help()
        self._draw_performance_info()
        self._draw_pause_indicator()

    def _draw_tooltip(self) -> None:
        """Draw the active tooltip if enabled."""
        if self.tooltip_text and self.show_tooltips:
            from ui.draw_utils import draw_tooltip

            draw_tooltip(
                self.screen,
                self.tooltip_text,
                self.tooltip_position[0],
                self.tooltip_position[1],
            )

    def _draw_controls_help(self) -> None:
        """Draw the controls help if requested."""
        if self.display_controls_help:
            self.notifier.draw_tooltips(self.screen, WINDOW_WIDTH - 270, 50)

    def _draw_performance_info(self) -> None:
        """Draw performance information (FPS counter and debug info)."""
        # Draw FPS counter if enabled and debug info is off
        if self.show_fps and not self.show_debug:
            self._draw_fps_counter()

        # Draw detailed debug info if enabled
        if self.show_debug:
            self.draw_debug_info()

    def _draw_fps_counter(self) -> None:
        """Draw the FPS counter with averaged values."""
        # Calculate current FPS and update history
        fps = self.clock.get_fps()
        self.fps_history.append(fps)

        # Keep history at a reasonable size
        if len(self.fps_history) > 60:
            self.fps_history.pop(0)

        # Calculate average FPS
        avg_fps = (
            sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        )

        # Draw the FPS text
        draw_text(self.screen, f"FPS: {avg_fps:.1f}", 10, 10, 14, COLOR_TEXT)

    def _draw_pause_indicator(self) -> None:
        """Draw the pause indicator when the game is paused."""
        if self.paused:
            # Draw pause panel
            draw_panel(
                self.screen,
                pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 30, 200, 60),
                color=(30, 30, 40, 200),
                header="PAUSED",
            )

            # Draw pause instructions
            draw_text(
                self.screen,
                "Press ESC to resume",
                WINDOW_WIDTH // 2,
                WINDOW_HEIGHT // 2 + 5,
                16,
                COLOR_TEXT,
                align="center",
            )

    def draw_play_state(self) -> None:
        """Render the play state elements."""
        # Draw the asteroid field using the renderer
        self.renderer.render(
            self.screen,
            self.field,
            self.player.x,
            self.player.y,
            self.zoom_level,
            show_grid=self.show_grid,
        )

        # Draw the player's ships
        self.player.draw_ships(
            self.screen,
            self.renderer.field_offset_x,
            self.renderer.field_offset_y,
            self.zoom_level,
        )

        # Draw player ship
        self.player.draw(self.screen, self.zoom_level)

        # Get the default font for UI components
        font = pygame.font.Font(None, 24)

        # Draw the main game screen UI
        self.game_screen.update(
            player_position=(self.player.x, self.player.y),
            resources=self.player.resources,
            health=self.player.health,
            energy=self.player.energy,
        )
        self.game_screen.draw(self.screen, font)

        # Draw minimap if enabled
        if self.show_minimap:
            # Update minimap data
            self.minimap_panel.update(
                player_position=(self.player.x, self.player.y),
                grid=self.field.grid,
                entity_grid=self.field.entity_grid,
                field_offset_x=self.renderer.field_offset_x,
                field_offset_y=self.renderer.field_offset_y,
                zoom_level=self.zoom_level,
            )
            self.minimap_panel.draw(self.screen, font)

        # Draw progress bar if enabled
        if self.show_progress_bar:
            draw_progress_bar(
                self.screen,
                self.player.x,
                self.player.y,
                self.field.grid,
                self.field.entity_grid,
                self.renderer.field_offset_x,
                self.renderer.field_offset_y,
                self.zoom_level,
            )

        # Draw button if enabled
        if self.show_button:
            draw_button(
                self.screen,
                self.player.x,
                self.player.y,
                self.field.grid,
                self.field.entity_grid,
                self.renderer.field_offset_x,
                self.renderer.field_offset_y,
                self.zoom_level,
            )

        # Draw player's ship info if enabled
        if self.show_ship_info:
            self.player.draw_ship_info(self.screen)

    def draw_debug_info(self) -> None:
        """Draw debug information on the screen with enhanced state tracking."""
        # Get game loop instance for its metrics
        game_loop = get_game_loop()
        event_batcher = get_event_batcher()
        game_event_bus = get_event_bus("GameEventBus")

        # Calculate frame timing information
        fps = self.clock.get_fps()
        self.fps_history.append(fps)
        if len(self.fps_history) > 100:
            self.fps_history.pop(0)

        avg_fps = (
            sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        )

        # Get state timing info
        current_time = time.time()
        time_in_state = current_time - self._state_timestamps[self.state]
        transitions_this_state = self._state_transition_counts[self.state]

        # Debug info sections
        state_text = [
            "=== State Info ===",
            f"Current: {self.state}",
            f"Previous: {self.previous_state}",
            f"Time in State: {time_in_state:.1f}s",
            f"Transitions: {transitions_this_state}",
            f"State Valid: {self._state_valid}",
            "Recent History:",
        ]

        # Add last 3 state transitions
        state_text.extend(
            f"  {transition['from_state']} -> {transition['to_state']} ({transition['game_time']:.1f}s)"
            for transition in self._state_history[-3:]
        )
        performance_text = [
            "=== Performance ===",
            f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})",
            f"Frame: {self.frame_counter} (GL: {game_loop.frame_counter})",
            f"Game Time: {self.game_time:.1f}s",
            f"Delta: {self.delta_time * 1000:.1f}ms",
            f"Memory: {log_memory_usage(return_value=True):.1f} MB",
        ]

        player_text = [
            "=== Player Info ===",
            f"Position: ({self.player.x}, {self.player.y})",
            f"Mining Power: {self.player.mining_power}",
            f"Rare Mining: {self.player.rare_mining_power}",
            f"Currency: {self.player.currency}",
            f"Health: {self.player.health}",
        ]

        field_text = [
            "=== Field Info ===",
            f"Grid Size: {GRID_WIDTH}x{GRID_HEIGHT}",
            f"Zoom: {self.zoom_level:.2f}x",
            f"Entities: {len(self.field.entities)}",
            f"Races: {len(self.field.races)}",
        ]

        game_loop_text = [
            "=== Game Loop ===",
            f"Update funcs: {len(game_loop.update_functions)}",
            f"Render funcs: {len(game_loop.render_functions)}",
            f"Event handlers: {len(game_loop.event_handlers)}",
            f"Interval updates: {len(game_loop.interval_updates)}",
        ]

        event_system_text = [
            "=== Event System ===",
            f"Game Events: {len(game_event_bus.event_history) if game_event_bus else 0}",
            f"Bus Subscribers: {len(game_event_bus.subscriptions) if game_event_bus else 0}",
            f"Event Batching: {event_batcher.is_batching}",
        ]

        # Draw debug sections in columns
        left_x = 10
        right_x = WINDOW_WIDTH // 2 + 10
        line_height = 20

        # Left column (State and Performance)
        y = 10
        for section in [state_text, performance_text]:
            for line in section:
                text = self.debug_font.render(line, True, (255, 255, 255))
                self.screen.blit(text, (left_x, y))
                y += line_height
            y += 10  # Section spacing

        # Right column (Player, Field, Game Loop, Events)
        y = 10
        for section in [player_text, field_text, game_loop_text, event_system_text]:
            for line in section:
                text = self.debug_font.render(line, True, (255, 255, 255))
                self.screen.blit(text, (right_x, y))
                y += line_height
            y += 10  # Section spacing

        # Draw state validation warning if needed
        if not self._state_valid:
            warning_text = "WARNING: Invalid State Transition Detected!"
            text = self.debug_font.render(warning_text, True, (255, 100, 100))
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
            self.screen.blit(text, text_rect)

    def get_state_debug_info(self) -> Dict[str, Any]:
        """Get detailed state debugging information.

        Returns:
            Dict containing state debug info including:
            - Current and previous states
            - Time spent in each state
            - Transition counts
            - Validation status
            - Performance metrics
        """
        current_time = time.time()

        state_times = {
            state: current_time - timestamp if state == self.state else 0.0
            for state, timestamp in self._state_timestamps.items()
        }
        return {
            "current_state": self.state,
            "previous_state": self.previous_state,
            "state_valid": self._state_valid,
            "last_validation_time": self._last_validation_time,
            "state_times": state_times,
            "transition_counts": self._state_transition_counts,
            "state_history": self._state_history[-10:],  # Last 10 transitions
            "performance": {
                "current_fps": self.clock.get_fps(),
                "avg_fps": (
                    sum(self.fps_history) / len(self.fps_history)
                    if self.fps_history
                    else 0
                ),
                "frame_counter": self.frame_counter,
                "game_time": self.game_time,
            },
        }

    def log_state_debug_info(self) -> None:
        """Log detailed state debugging information."""
        debug_info = self.get_state_debug_info()

        logging.debug("=== State Debug Information ===")
        logging.debug(f"Current State: {debug_info['current_state']}")
        logging.debug(f"Previous State: {debug_info['previous_state']}")
        logging.debug(f"State Valid: {debug_info['state_valid']}")

        # Log time in states
        logging.debug("\nTime in States:")
        for state, time_spent in debug_info["state_times"].items():
            logging.debug(f"  {state}: {time_spent:.1f}s")

        # Log transition counts
        logging.debug("\nState Transition Counts:")
        for state, count in debug_info["transition_counts"].items():
            logging.debug(f"  {state}: {count}")

        # Log recent state history
        logging.debug("\nRecent State Transitions:")
        for transition in debug_info["state_history"]:
            logging.debug(
                f"  {transition['from_state']} -> {transition['to_state']} "
                f"at frame {transition['frame']}"
            )

        # Log performance metrics
        perf = debug_info["performance"]
        logging.debug("\nPerformance Metrics:")
        logging.debug(f"  Current FPS: {perf['current_fps']:.1f}")
        logging.debug(f"  Average FPS: {perf['avg_fps']:.1f}")
        logging.debug(f"  Frame: {perf['frame_counter']}")
        logging.debug(f"  Game Time: {perf['game_time']:.1f}s")

        # The following code appears to be for drawing debug info on screen
        # but contains undefined variables. Commenting out for now.
        # If this functionality is needed, it should be implemented properly.

        # Starting y position for debug text
        # y = 50  # Initial y position

        # # Player section

        # # Field section

        # # Game loop section

        # # Event system section

    # Wrapper methods for the new game loop architecture
    def handle_events_wrapper(self) -> bool:
        """Wrapper around handle_events for the new GameLoop system.

        Returns:
            bool: True if the game should exit, False otherwise
        """
        return not self.handle_events()

    def update_wrapper(self, delta_time: float) -> None:
        """Wrapper around update for the new GameLoop system.

        Args:
            delta_time: Time since last frame in seconds
        """
        # Update our internal time tracking
        self.delta_time = delta_time
        self.game_time += delta_time

        # Process the update if not paused
        if not self.paused:
            self.update()

        # Start event batching for this frame
        event_batcher = get_event_batcher()
        event_batcher.start_batch()

        if game_event_bus := get_event_bus("GameEventBus"):
            game_event_bus.dispatch(
                "frame_update",
                {
                    "frame": self.frame_counter,
                    "delta_time": delta_time,
                    "game_time": self.game_time,
                },
            )

        # End event batching and dispatch all events
        event_batcher.dispatch_batch()


def _log_game_completion_metrics(game):
    """
    Log metrics about a successfully completed game session.
    
    Args:
        game: The Game instance containing session data
    """
    # Log basic session information
    logging.info(f"Game session completed. Session duration: {game.session_time:.2f} seconds")
    
    # Log player statistics
    if hasattr(game, "player") and game.player:
        logging.info(f"Final player stats - Currency: {game.player.currency}, Level: {game.player.level}")
        
        # Log discovered races if applicable
        if hasattr(game.player, "discovered_races"):
            logging.info(f"Discovered races: {len(game.player.discovered_races)}")
    
    # Log game world statistics if available
    if hasattr(game, "world") and game.world:
        logging.info(f"World generation seed: {game.world.seed}")
        logging.info(f"Generated regions: {len(game.world.regions) if hasattr(game.world, 'regions') else 'N/A'}")


def _log_game_error_metrics(game):
    """
    Log metrics about game session that terminated with errors.
    
    Args:
        game: The Game instance containing session data
    """
    # Log basic error session information
    logging.warning(f"Game session terminated with errors. Session duration: {game.session_time:.2f} seconds")
    
    # Log system metrics that might help diagnose issues
    if hasattr(game, "performance_metrics"):
        logging.warning(f"Performance metrics: {game.performance_metrics}")
    
    # Log last known game state before failure
    if hasattr(game, "last_state"):
        logging.warning(f"Last game state before failure: {game.last_state}")
    
    # Check for exception information
    if hasattr(game, "last_exception"):
        logging.error(f"Last exception: {game.last_exception}")


def run_game_loop(game):
    """Run the main game loop.

    Args:
        game (Game): The game instance to run.

    Returns:
        bool: True if the game completed successfully, False otherwise.
    """
    try:
        # Get the game loop instance
        game_loop = get_game_loop()

        # Register event handlers, update functions, and render functions
        game_loop.register_event_handler(game.handle_events_wrapper)
        game_loop.register_update_function(game.update_wrapper)
        game_loop.register_render_function(game.draw)

        # Register interval updates
        game_loop.register_interval_update(
            "field_update", game.update_interval / 60.0
        )  # Convert frames to seconds

        # Define a clock function that uses pygame's clock
        def clock_func():
            game_loop.delta_time = (
                game.clock.get_time() / 1000.0
            )  # Convert ms to seconds
            return time.time()

        # Run the game loop
        game_loop.run(clock_func)

        return True
    except Exception as e:
        log_exception("Unhandled exception in main game loop", e)
        return False


def main():
    """Main entry point for the game."""
    try:
        _initialize_game_and_run()
    except Exception as e:
        log_exception("Critical error in main function", e)
        raise
    finally:
        # Ensure pygame is properly shut down
        pygame.quit()
        logging.info("Game shutdown complete")


def _initialize_game_and_run():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting Space Muck game...")

    # Initialize event system
    # Prefix unused variables with underscore
    _ = get_event_bus("GameEventBus")  # Initialize game event bus
    _ = get_event_bus("ResourceEventBus")  # Initialize resource event bus
    _ = get_event_bus("ModuleEventBus")  # Initialize module event bus
    _ = get_event_batcher()  # Initialize event batcher

    logging.info("Event system initialized")

    # Create the game instance
    game = Game()

    if game_success := run_game_loop(game):  # noqa: F841
        logging.info("Game completed successfully")
        _log_game_completion_metrics(game)
    else:
        logging.warning("Game loop terminated with errors")
        _log_game_error_metrics(game)


if __name__ == "__main__":
    main()
