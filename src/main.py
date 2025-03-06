#!/usr/bin/env python3
# /src/main.py
"""

Space Muck - Advanced Procedural Generation Edition

A space mining game featuring:
- Evolving asteroid fields with cellular automaton mechanics
- Multiple symbiote races that adapt and evolve
- Dynamic mining and upgrade systems
- Advanced procedural generation with multiple algorithms
"""


# Standard library imports
import gc
import itertools
import logging
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the current directory to path to ensure proper importing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

# Third-party library imports
import numpy as np  # noqa: E402
import pygame  # noqa: E402

# Local application imports
# Game constants and configuration
from config import (  # noqa: E402
    # Window and display settings
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    GRID_WIDTH,
    GRID_HEIGHT,
    MINIMAP_SIZE,
    MINIMAP_PADDING,
    VIEW_WIDTH,
    VIEW_HEIGHT,
    # Colors
    COLOR_BG,
    COLOR_TEXT,
    COLOR_RACE_1,
    COLOR_RACE_2,
    COLOR_RACE_3,
    COLOR_PLAYER,
)

# Manual definitions for missing constants
VERSION = "1.0.0"
UPDATE_INTERVAL = 10
DEBUG_MODE = False
SHOW_GRID = True
SHOW_FPS = True
STATE_PLAY = "PLAY"
STATE_SHOP = "SHOP"
RACE_INITIAL_DENSITY = 0.005

# Game components
from generators.asteroid_field import AsteroidField  # noqa: E402
from entities.player import Player  # noqa: E402
from entities.miner_entity import MinerEntity  # noqa: E402
from ui.shop import Shop  # noqa: E402
from ui.notification import NotificationManager  # noqa: E402
from ui.renderers import AsteroidFieldRenderer  # noqa: E402
from ui.draw_utils import (  # noqa: E402
    draw_text,
    draw_panel,
    draw_minimap,
    draw_progress_bar,
    draw_button,
)

# Game systems
from systems.combat_system import CombatSystem  # noqa: E402
from systems.encounter_generator import EncounterGenerator  # noqa: E402
from utils.logging_setup import (  # noqa: E402
    log_exception,
    log_performance_start,
    log_performance_end,
    LogContext,
    log_memory_usage
)


class Game:
    """Main game class that orchestrates all game components."""

    def __init__(self) -> None:
        """Initialize the game and all its components."""
        # Initialize Pygame
        pygame.init()
        pygame.mixer.init()  # Initialize sound system
        pygame.font.init()  # Ensure fonts are initialized

        # Create display surface
        self.screen: pygame.Surface = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        pygame.display.set_caption(f"Space Muck - Procedural Edition v{VERSION}")

        # Set up game clock and timing
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.delta_time: float = 0.0  # Time since last frame in seconds
        self.frame_counter: int = 0
        self.game_time: float = 0.0  # Total game time in seconds
        self.fps_history: List[float] = []  # Store recent FPS values

        # Initialize UI components with LogContext for error handling
        with LogContext("UI Initialization"):
            self.shop: Shop = Shop()
            self.notifier: NotificationManager = NotificationManager()
            self.renderer: AsteroidFieldRenderer = AsteroidFieldRenderer()

        # Game state
        self.state: str = STATE_PLAY
        self.previous_state: str = STATE_PLAY
        self.paused: bool = False
        self.game_over: bool = False

        # Performance settings
        self.update_interval: int = UPDATE_INTERVAL
        self.last_field_update_time: float = 0
        self.auto_upgrade_interval: int = 300  # Every 5 seconds at 60 FPS
        self.last_memory_check: float = 0

        # Game options and toggles
        self.show_minimap: bool = True
        self.show_debug: bool = DEBUG_MODE
        self.auto_mine: bool = False
        self.auto_upgrade: bool = False
        self.show_tooltips: bool = True
        self.show_grid: bool = SHOW_GRID
        self.zoom_level: float = 1.0

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

        # UI state
        self.selected_race_index: int = -1
        self.hover_position: Tuple[int, int] = (0, 0)
        self.tooltip_text: Optional[str] = None
        self.tooltip_position: Tuple[int, int] = (0, 0)
        self.show_race_details: bool = False
        self.cursor_over_ui: bool = False
        self.display_controls_help: bool = False

        # Statistics tracking
        self.stats: Dict[str, Any] = {
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
        """Initialize visual settings and precompute resources."""
        # Load and scale UI elements
        try:
            self.ui_font = pygame.font.SysFont("Arial", 16)
            self.title_font = pygame.font.SysFont("Arial", 24, bold=True)

            # Create background pattern for performance
            bg_pattern_size = 64
            self.bg_pattern = pygame.Surface((bg_pattern_size, bg_pattern_size))
            self.bg_pattern.fill(COLOR_BG)

            # Add subtle noise to background
            for y, x in itertools.product(
                range(bg_pattern_size), range(bg_pattern_size)
            ):
                noise = random.randint(-5, 5)
                color = max(0, min(255, COLOR_BG[0] + noise))
                self.bg_pattern.set_at((x, y), (color, color, color + 2))

            # Create cursor surface
            self.cursor_size = 12
            self.cursor_surface = pygame.Surface(
                (self.cursor_size, self.cursor_size), pygame.SRCALPHA
            )
            pygame.draw.circle(
                self.cursor_surface,
                COLOR_PLAYER,
                (self.cursor_size // 2, self.cursor_size // 2),
                self.cursor_size // 2,
            )
            pygame.draw.circle(
                self.cursor_surface,
                (255, 255, 255),
                (self.cursor_size // 2, self.cursor_size // 2),
                self.cursor_size // 2,
                1,
            )

            # Pre-compute asteroid field renderer caches
            self.renderer.initialize(self.field)

        except Exception as e:
            logging.error(f"Error initializing visuals: {e}")
            log_exception(e)

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

    def handle_events(self) -> bool:
        """Process all pygame events.
        
        Returns:
            bool: True if the game should continue running, False otherwise.
        """
        for event in pygame.event.get():
            # First, let UI components handle events
            if self.notifier.handle_event(event):
                continue

            if self.state == STATE_SHOP and self.shop.handle_event(
                event, self.player, self.field
            ):
                continue

            # Basic window events
            if event.type == pygame.QUIT:
                return self.quit_game()
            elif event.type == pygame.KEYDOWN:
                self.handle_key_press(event.key)
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_button_down(event)
                
        return True
        
    def quit_game(self) -> bool:
        """Exit the game cleanly.
        
        Returns:
            bool: Always returns False to indicate the game should stop running.
        """
        try:
            # Start performance timing for exit process
            exit_start = log_performance_start("Game exit process")
            
            # Log complete game statistics before quitting
            logging.info("===== COMPLETE GAME STATISTICS =====")
            
            # Log basic statistics
            basic_stats = [
                ("Game time", f"{self.game_time:.2f} seconds"),
                ("Resources collected", self.stats.get("resources_collected", 0)),
                ("Distance traveled", self.stats.get("distance_traveled", 0)),
                ("Encounters", self.stats.get("encounters", 0)),
                ("Combat victories", self.stats.get("combats_won", 0)),
                ("Total combats", self.stats.get("total_combats", 0))
            ]
            
            for stat_name, stat_value in basic_stats:
                logging.info(f"  {stat_name}: {stat_value}")
            
            # Log detailed statistics
            logging.info("\n--- Detailed Statistics ---")
            for stat_name, stat_value in sorted(self.stats.items()):
                if stat_name not in [item[0] for item in basic_stats]:
                    logging.info(f"  {stat_name}: {stat_value}")
                
            # Log performance summary with more detailed analysis
            if hasattr(self, 'performance_metrics') and self.performance_metrics:
                logging.info("\n--- Performance Metrics Summary ---")
                for metric_name, values in self.performance_metrics.items():
                    if values:
                        avg = sum(values) / len(values)
                        max_val = max(values)
                        min_val = min(values)
                        std_dev = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5 if len(values) > 1 else 0
                        p95 = sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max_val
                        
                        logging.info(f"  {metric_name}:")
                        logging.info(f"    avg={avg:.3f}ms, min={min_val:.3f}ms, max={max_val:.3f}ms")
                        logging.info(f"    std_dev={std_dev:.3f}ms, p95={p95:.3f}ms, samples={len(values)}")
            
            # Save player data if needed
            # This could be expanded to save game state for future sessions
            try:
                # Display a farewell message to the player
                play_time_min = self.game_time / 60
                
                if play_time_min >= 1:
                    farewell_msg = f"Thanks for playing Space Muck! You played for {play_time_min:.1f} minutes."
                else:
                    farewell_msg = "Thanks for playing Space Muck!"
                    
                # Add session highlights if available
                if self.stats.get("combats_won", 0) > 0:
                    combat_ratio = self.stats.get("combats_won", 0) / max(1, self.stats.get("total_combats", 1))
                    farewell_msg += f"\nCombat success rate: {combat_ratio:.1%}"
                
                if self.stats.get("discoveries", 0) > 0:
                    farewell_msg += f"\nUnique discoveries: {self.stats.get('discoveries', 0)}"
                
                self.notifier.add(farewell_msg, category="system", importance=3)
                
                # Perform system cleanup
                log_memory_usage("Before final cleanup")
                
                # Run garbage collection to free memory
                gc_start = time.time()
                gc.collect()
                logging.info(f"Final garbage collection completed in {time.time() - gc_start:.4f} seconds")
                
                # Close any open resources or connections
                # (Adding placeholder for future implementation)
                
            except Exception as cleanup_error:
                log_exception("Error during exit cleanup", cleanup_error)
            
            # Signal that the game should stop running
            logging.info("Quitting game")
            log_performance_end("Game exit process", exit_start)
            return False
        except Exception as e:
            log_exception("Error during game exit", e)
            return False  # Still exit even if there's an error

    def handle_key_press(self, key: int) -> None:
        """Handle keyboard input."""
        if key == pygame.K_n:
            return
        # Global keys (work in all states)
        if key == pygame.K_ESCAPE:
            if self.state != STATE_PLAY:
                # Return to play state
                self.state = STATE_PLAY
            else:
                # Toggle pause
                self.paused = not self.paused

        elif key == pygame.K_F1:
            # Toggle debug mode
            self.show_debug = not self.show_debug

        elif key == pygame.K_h:
            # Toggle control help
            self.display_controls_help = not self.display_controls_help

        elif self.state == STATE_PLAY:
            self.handle_play_state_keys(key)

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
        """Handle keys specific to play state."""
        if key == pygame.K_s:
            # Open shop
            self.state = STATE_SHOP

        elif key == pygame.K_m:
            # Toggle minimap
            self.show_minimap = not self.show_minimap

        elif key == pygame.K_g:
            # Toggle grid
            self.show_grid = not self.show_grid

        elif key == pygame.K_a:
            # Toggle auto-mine
            self.auto_mine = not self.auto_mine
            self.notifier.add(
                f"Auto-mining {'enabled' if self.auto_mine else 'disabled'}",
                category="mining",
            )

        elif key == pygame.K_SPACE:
            # Mine asteroids
            if not self.paused:
                self.mine()

        elif key == pygame.K_r:
            # Regenerate field
            self.regenerate_field()

        elif key == pygame.K_f:
            # Feed symbiotes
            self.feed_symbiotes()

        elif key == pygame.K_u:
            # Toggle auto-upgrade
            self.auto_upgrade = not self.auto_upgrade
            self.notifier.add(
                f"Auto-upgrade {'enabled' if self.auto_upgrade else 'disabled'}",
                category="upgrade",
            )

        elif key in [pygame.K_PLUS, pygame.K_EQUALS]:
            # Zoom in
            self.zoom_level = min(2.0, self.zoom_level + 0.1)

        elif key in [pygame.K_MINUS, pygame.K_UNDERSCORE]:
            # Zoom out
            self.zoom_level = max(0.5, self.zoom_level - 0.1)

        elif key == pygame.K_p:
            # Add ship to fleet
            if self.player.add_ship():
                self.notifier.add(
                    "New mining ship added to your fleet!", category="mining"
                )
            else:
                self.notifier.add(
                    "Cannot add more ships (max reached or insufficient funds)",
                    category="mining",
                )

    def handle_mouse_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse clicks based on game state."""
        if self.state == STATE_PLAY:
            # Check if click was on UI element
            if self.check_cursor_over_ui():
                return

            # Convert screen position to grid position
            grid_x, grid_y = self.screen_to_grid(pos[0], pos[1])

            # Check if position is valid
            if 0 <= grid_x < self.field.width and 0 <= grid_y < self.field.height:
                # Move player there
                dx = grid_x - self.player.x
                dy = grid_y - self.player.y
                self.player.move(dx, dy, self.field)
                # Set a flag to indicate the player has moved (for encounter checks)
                self.player.has_moved = True

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

            # Update UI components (must be done regardless of game state)
            self.notifier.update(self.delta_time)

            # Clear tooltip if not updated this frame
            self.tooltip_text = None

            # State-specific updates
            if self.state == STATE_PLAY:
                self.update_play_state()
            elif self.state == STATE_SHOP:
                self.shop.update(self.delta_time)

            # Auto-upgrade if enabled
            if self.auto_upgrade:
                self.handle_auto_upgrade()

            # End performance timing
            log_performance_end("Update frame", update_start)
        except Exception as e:
            log_exception("Error in update process", e)
            self.notifier.add(f"Error: {str(e)}", category="error", importance=3)

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
                    "velocity": getattr(self.player, 'velocity', (0, 0)),
                    "resources": getattr(self.player, 'resources', {}),
                    "health": getattr(self.player, 'health', 100),
                    "fleet_size": getattr(self.player, 'fleet_size', 1)
                }
                update_context["player"] = player_stats
                
            except Exception as stage1_error:
                log_exception("Error in Player State processing (Stage 1)", stage1_error)
                self.notifier.add("Movement system error", category="error", importance=2)
            
            # STAGE 2: World Systems and Environment
            try:
                # Update asteroid field and environmental elements
                world_start = log_performance_start("World systems update")
                self.update_asteroid_field()
                log_performance_end("World systems update", world_start)
                
                # Calculate environmental context for other systems
                field_stats = {
                    "asteroid_count": len(getattr(self.field, 'asteroids', [])),
                    "field_density": getattr(self.field, 'density', 0.5),
                    "resource_availability": getattr(self.field, 'resource_level', "medium")
                }
                update_context["environment"] = field_stats
                
            except Exception as stage2_error:
                log_exception("Error in World Systems processing (Stage 2)", stage2_error)
                self.notifier.add("Environmental systems error", category="error", importance=2)
            
            # STAGE 3: Encounter and Combat Systems
            try:
                # Process encounters and potential combat
                encounter_start = log_performance_start("Encounter and combat systems")
                self.process_encounters()
                log_performance_end("Encounter and combat systems", encounter_start)
                
                # Track encounter contexts for statistics and future reference
                encounter_stats = {
                    "last_encounter_time": self.stats.get("last_encounter_time", 0),
                    "last_encounter_type": self.stats.get("last_encounter_type", "none"),
                    "encounter_frequency": self.stats.get("encounters", 0) / max(1, self.game_time / 60)
                }
                update_context["encounters"] = encounter_stats
                
            except Exception as stage3_error:
                log_exception("Error in Encounter/Combat processing (Stage 3)", stage3_error)
                self.notifier.add("Encounter system error", category="error", importance=3)
            
            # STAGE 4: Entity Lifecycle and Simulation
            try:
                # Update symbiote races, evolutions, and discoveries
                entity_start = log_performance_start("Entity lifecycle processing")
                self.check_race_evolutions()
                self.check_for_discoveries()
                log_performance_end("Entity lifecycle processing", entity_start)
                
                # Generate entity-related context
                entity_stats = {
                    "active_races": getattr(self, 'race_count', 0),
                    "evolution_stage": getattr(self, 'evolution_stage', 1),
                    "discovery_count": self.stats.get("discoveries", 0)
                }
                update_context["entities"] = entity_stats
                
            except Exception as stage4_error:
                log_exception("Error in Entity Lifecycle processing (Stage 4)", stage4_error)
                self.notifier.add("Evolution system error", category="error", importance=2)
            
            # STAGE 5: Resource Management and Economy
            try:
                # Handle mining, trading, and economy updates
                economy_start = log_performance_start("Economy and resources processing")
                self.handle_auto_mining()
                # Future: Add trading system update here
                log_performance_end("Economy and resources processing", economy_start)
                
                # Track economic context
                economy_stats = {
                    "player_currency": self.player.currency,
                    "resources_collected": self.stats.get("resources_collected", 0),
                    "trading_opportunities": getattr(self, 'available_trades', 0)
                }
                update_context["economy"] = economy_stats
                
            except Exception as stage5_error:
                log_exception("Error in Economy processing (Stage 5)", stage5_error)
                self.notifier.add("Resource system error", category="error", importance=2)
            
            # STAGE 6: Fleet and Unit Management
            try:
                # Update player fleet and unit behaviors
                fleet_start = log_performance_start("Fleet management processing")
                self.update_player_fleet()
                log_performance_end("Fleet management processing", fleet_start)
                
                # Track fleet context
                fleet_stats = {
                    "fleet_size": getattr(self.player, 'fleet_size', 1),
                    "fleet_strength": getattr(self.player, 'fleet_strength', 100),
                    "formation": getattr(self.player, 'formation', "standard")
                }
                update_context["fleet"] = fleet_stats
                
            except Exception as stage6_error:
                log_exception("Error in Fleet Management processing (Stage 6)", stage6_error)
                self.notifier.add("Fleet system error", category="error", importance=2)
                
            # STAGE 7: Game State Evaluation
            # Evaluate game state and trigger events if necessary
            self.evaluate_game_state(update_context)
            
            # Performance tracking and logging for the entire update cycle
            update_duration = time.time() - play_state_start
            log_performance_metric("play_state_update_time", update_duration * 1000)
            
            # Log completion with statistics
            if self.frame_counter % 300 == 0:  # Detailed logging every 5 seconds (at 60 FPS)
                logging.info(f"Play state update complete. Frame: {self.frame_counter}, Duration: {update_duration:.4f}s")
                log_memory_usage("Regular play state memory check")
            
            # Complete performance tracking
            log_performance_end("Play state update", play_state_start, "complete")
            
        except Exception as e:
            log_exception("Critical error in play state update", e)
            self.notifier.add(f"Game system error: {str(e)}", category="error", importance=3)
            
    def evaluate_game_state(self, context: Dict[str, Any]) -> None:
        """Evaluate the current game state and trigger appropriate events.
        
        This method analyzes the game context from all systems and determines
        if any special events, achievements, or state changes should occur.
        
        Args:
            context: Dictionary containing state information from all game systems
        """
        try:
            # Example of cross-system state evaluation
            player_pos = context.get("player", {}).get("position", (0, 0))
            last_encounter = context.get("encounters", {}).get("last_encounter_time", 0)
            current_time = self.game_time
            
            # Check for extended peaceful periods (no encounters for 2+ minutes)
            if current_time - last_encounter > 120 and self.stats.get("encounters", 0) > 0:
                zone = self.get_current_zone(player_pos)
                if zone not in ["central_hub", "safe_zone"]:
                    # Trigger dynamic encounter after periods of inactivity
                    logging.info(f"Extended peace period detected in {zone} ({current_time - last_encounter:.1f}s)")
                    self.force_encounter("dynamic", player_pos, zone)
            
            # Track progression milestones
            player_currency = context.get("economy", {}).get("player_currency", 0)
            currency_milestone = (player_currency // 1000) * 1000
            last_milestone = self.stats.get("last_currency_milestone", 0)
            
            if currency_milestone > last_milestone and currency_milestone > 0:
                self.stats["last_currency_milestone"] = currency_milestone
                self.notifier.add(
                    f"Milestone: Accumulated {currency_milestone} currency!",
                    category="achievement",
                    importance=2
                )
                
            # Example: Analyze environmental dangers
            asteroid_count = context.get("environment", {}).get("asteroid_count", 0)
            if asteroid_count > 200:  # High density field
                player_health = context.get("player", {}).get("health", 100)
                if player_health < 50:  # Player in danger
                    self.notifier.add(
                        "Warning: High asteroid density detected, recommend seeking safer area",
                        category="warning",
                        importance=2
                    )
            
        except Exception as e:
            log_exception("Error in game state evaluation", e)
            # Non-critical, so just log but don't notify player

    def handle_player_movement(self) -> None:
        """Handle player movement based on keyboard input."""
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx = -1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx = 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy = -1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy = 1

        if dx != 0 or dy != 0:
            self.player.move(dx, dy, self.field)
            # Set a flag to indicate the player has moved (for encounter checks)
            self.player.has_moved = True

    def handle_auto_mining(self) -> None:
        """Handle auto-mining if enabled."""
        try:
            # Check if auto-mining is enabled and if it's time to mine
            if not (
                self.auto_mine and self.frame_counter % 30 == 0
            ):  # Every half second at 60 FPS
                return

            # Try to mine with error handling
            with LogContext("Auto-mining operation"):
                minerals_mined = self.player.mine(self.field)
                
                # Update statistics and notify if successful
                if minerals_mined > 0:
                    self.stats["total_mined"] += minerals_mined
                    
                    # Only notify occasionally to avoid spamming
                    if self.frame_counter % 120 == 0:  # Once every 2 seconds
                        self.notifier.add(
                            f"Auto-mined {minerals_mined} minerals", 
                            category="mining", 
                            importance=1
                        )
        except Exception as e:
            log_exception("Error in auto-mining", e)

    def update_asteroid_field(self) -> None:
        """Update asteroid field at intervals."""
        current_time = time.time()
        if current_time - self.last_field_update_time <= self.update_interval / 60:
            return

        race_incomes = self.field.update()
        self.last_field_update_time = current_time

        # Process race incomes and notify about significant events
        for race_id, income in race_incomes.items():
            if income <= 500:
                continue

            if race := next(
                (r for r in self.field.races if r.race_id == race_id), None
            ):
                self.notifier.add(
                    f"Race {race_id} ({race.trait}) found a large mineral deposit!",
                    category="race",
                    importance=2,
                )

    def check_race_evolutions(self) -> None:
        """Check for race evolutions."""
        for race in self.field.races:
            if (
                race.evolution_points >= race.evolution_threshold
                and self.frame_counter % 300 == race.race_id % 300
            ):  # Stagger evolutions
                # Race evolves
                metrics = race.evolve()

                # Notify the player
                self.notifier.notify_event(
                    "race",
                    f"Race {race.race_id} ({race.trait}) has evolved to stage {race.evolution_stage}!",
                    importance=2,
                )

                # Track in stats
                self.stats["race_evolutions"] += 1

                # Analyze territory control
                if metrics and metrics["center"]:
                    logging.info(
                        f"Race {race.race_id} territory: "
                        + f"center={metrics['center']}, radius={metrics['radius']}, "
                        + f"density={metrics['density']:.4f}"
                    )

    def check_for_discoveries(self) -> None:
        """Check for new resource discoveries by player."""
        try:
            # Start performance timing for discovery check
            discovery_start = log_performance_start("Discovery check")
            
            # Only run this check periodically to avoid performance impact
            if self.frame_counter % 120 != 0:  # Every 2 seconds at 60 FPS
                log_performance_end("Discovery check", discovery_start, "skipped")
                return

            # Track whether anything was discovered in this check
            discoveries_made = False
            
            # Use LogContext for proper error handling and tracking
            with LogContext("Resource discovery check"):
                # Safety check for player position
                if not hasattr(self.player, 'x') or not hasattr(self.player, 'y'):
                    logging.error("Player position attributes missing")
                    return
                    
                # Ensure field dimensions are valid
                if not hasattr(self.field, 'width') or not hasattr(self.field, 'height'):
                    logging.error("Field dimension attributes missing")
                    return
                    
                # Validate player is within field bounds
                if not (0 <= self.player.x < self.field.width and 0 <= self.player.y < self.field.height):
                    logging.warning(f"Player outside field bounds: ({self.player.x}, {self.player.y})")
                    return
                
                # Check for nearby anomalies in a square area around the player
                discovery_range = 3  # How far to look for discoveries
                for dy, dx in itertools.product(range(-discovery_range, discovery_range + 1), 
                                               range(-discovery_range, discovery_range + 1)):
                    nx, ny = self.player.x + dx, self.player.y + dy
                    
                    # Validate coordinates are within bounds
                    if not (0 <= nx < self.field.width and 0 <= ny < self.field.height):
                        continue
                        
                    # Check if this is an anomaly worth reporting
                    if (
                        self.field.grid[ny, nx] > 0 and
                        self.field.rare_grid[ny, nx] == 3  # Anomaly
                    ):
                        discoveries_made = True
                        # Only notify if not already discovered
                        if (nx, ny) not in getattr(self, 'discovered_anomalies', set()):
                            # Initialize the set if it doesn't exist
                            if not hasattr(self, 'discovered_anomalies'):
                                self.discovered_anomalies = set()
                                
                            # Add to discovered set
                            self.discovered_anomalies.add((nx, ny))
                            
                            # Notify the player
                            self.notifier.notify_event(
                                "discovery",
                                f"Found an anomaly at ({nx}, {ny})!",
                                importance=3,
                            )
                            logging.info(f"Player discovered anomaly at ({nx}, {ny})")
                            
                            # Update stats
                            self.stats["total_anomalies_discovered"] = \
                                self.stats.get("total_anomalies_discovered", 0) + 1
            
            # End performance timing
            log_performance_end("Discovery check", discovery_start, 
                               "found" if discoveries_made else "none")
                               
        except Exception as e:
            log_exception("Error checking for discoveries", e)
            
    def update_player_fleet(self) -> None:
        """Update the player's fleet status and process results."""
        try:
            # Only update fleet periodically to avoid performance impact
            if self.frame_counter % 60 != 0:  # Once per second at 60 FPS
                return
                
            # Update the fleet and get results
            with LogContext("Fleet update operation"):
                fleet_results = self.player.update_fleet(self.field)
                
                # Process fleet update results
                if fleet_results is None:
                    logging.debug("No fleet update results to process")
                    return
                    
                # Handle damaged ships
                if fleet_results.get("ships_damaged", 0) > 0:
                    damaged_count = fleet_results["ships_damaged"]
                    self.notifier.add(
                        f"{damaged_count} ships damaged!",
                        category="fleet",
                        importance=2
                    )
                    logging.info(f"Player fleet update: {damaged_count} ships damaged")
                    
                # Handle lost ships
                if fleet_results.get("ships_lost", 0) > 0:
                    lost_count = fleet_results["ships_lost"]
                    self.notifier.add(
                        f"{lost_count} ships lost in dangerous territory!",
                        category="fleet",
                        importance=3
                    )
                    self.stats["ships_lost"] += lost_count
                    logging.warning(f"Player lost {lost_count} ships in fleet update")

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
                            importance=1
                        )
                        
                # Track performance metrics if available
                if "processing_time" in fleet_results:
                    log_performance_metric("fleet_update_processing", fleet_results["processing_time"])
        except Exception as e:
            log_exception("Error during fleet update", e)
            
    def process_encounters(self) -> None:
        """Process potential encounters based on player movement and location.
        
        This method checks for and handles all encounter types including combat,
        discovery, and trader encounters. It manages the encounter lifecycle from
        detection through resolution, with comprehensive performance tracking.
        """
        # Start performance timing for encounter processing
        encounter_start = log_performance_start("Encounter processing")
        
        try:
            # Track last encounter check time for throttling
            current_time = time.time()
            last_check_time = getattr(self, 'last_encounter_check_time', 0)
            check_interval = 1.5  # Seconds between checks
            
            # Only check for encounters if player has moved or if enough time has passed
            player_moved = getattr(self.player, 'has_moved', False)  # More robust attribute check
            time_to_check = (current_time - last_check_time) >= check_interval
            
            if not (player_moved or time_to_check):
                log_performance_end("Encounter processing", encounter_start, "throttled")
                return
            
            # Update last check time
            self.last_encounter_check_time = current_time
                
            # Reset player movement flag if it exists
            if hasattr(self.player, 'has_moved'):
                self.player.has_moved = False
            
            # Get current player position and zone for context
            player_pos = (self.player.x, self.player.y)
            current_zone = self.get_current_zone(player_pos)
            
            # Log zone transition if it changed
            if hasattr(self, 'last_zone') and self.last_zone != current_zone:
                logging.info(f"Player transitioned from {self.last_zone} to {current_zone}")
                self.notifier.add(f"Entering {current_zone}", category="navigation", importance=1)
            self.last_zone = current_zone
            
            # Check for potential encounter with appropriate context
            with LogContext("Encounter check"):
                # Track checks by zone
                zone_checks_key = f"encounter_checks_{current_zone}"
                self.stats[zone_checks_key] = self.stats.get(zone_checks_key, 0) + 1
                
                # Pass current gameplay context to encounter generator
                encounter_context = {
                    "player_health": self.player.health,
                    "player_level": self.player.level if hasattr(self.player, 'level') else 1,
                    "player_currency": self.player.currency,
                    "game_time": self.game_time,
                    "zone": current_zone
                }
                
                # Enhanced encounter check with more context
                encounter_check_start = time.time()
                encounter_result = self.encounter_generator.check_for_encounter(player_pos, encounter_context)
                log_performance_metric("encounter_check_time", (time.time() - encounter_check_start) * 1000)
                
                if not encounter_result:
                    log_performance_end("Encounter processing", encounter_start, "no_encounter")
                    return
                    
                # Process the encounter that was triggered
                encounter_type = encounter_result.get("type", "unknown")
                encounter_zone = encounter_result.get("zone", current_zone)
                encounter_rarity = encounter_result.get("rarity", "common")
                
                # Track encounter by zone and type
                self.stats[f"encounters_in_{encounter_zone}"] = self.stats.get(f"encounters_in_{encounter_zone}", 0) + 1
                self.stats[f"encounters_{encounter_type}"] = self.stats.get(f"encounters_{encounter_type}", 0) + 1
                self.stats[f"encounters_{encounter_rarity}"] = self.stats.get(f"encounters_{encounter_rarity}", 0) + 1
                
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
                        importance=2
                    )
                
                # Update global encounter statistics
                self.stats["encounters"] = self.stats.get("encounters", 0) + 1
                self.stats["last_encounter_time"] = self.game_time
                self.stats["last_encounter_type"] = encounter_type
                self.stats["last_encounter_zone"] = encounter_zone
                
                # Log the encounter details
                logging.info(f"Player encountered {encounter_type} ({encounter_rarity}) at {player_pos} in {encounter_zone}")
                
            # End performance timing with encounter details
            log_performance_end("Encounter processing", encounter_start, f"{encounter_type}_{encounter_rarity}")
            
        except Exception as e:
            logging.error(f"Error processing encounters: {str(e)}")
            log_exception("Encounter processing error", e)
            log_performance_end("Encounter processing", encounter_start, "error")
    
    def get_current_zone(self, position: Tuple[int, int]) -> str:
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
            f"Encountered hostile {enemy.name}!",
            category="encounter",
            importance=3
        )
        
        # Initiate combat and process results
        try:
            # Track combat initiation performance
            combat_initiation_start = time.time()
            combat_result = self.combat_system.initiate_combat(enemy)
            log_performance_metric("combat_initiation_time", (time.time() - combat_initiation_start) * 1000)
            
            # Process the combat result
            self.process_combat_result(combat_result)
            
            # Track overall combat handling time
            log_performance_metric("total_combat_handling_time", (time.time() - combat_start) * 1000)
            
        except Exception as e:
            log_exception("Error during combat processing", e)
            self.notifier.add(
                "Combat system malfunction", 
                category="error",
                importance=3
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
        discovery_name = discovery.get('name', 'Unknown object')
        discovery_type = discovery.get('type', 'general')
        discovery_rarity = discovery.get('rarity', 'common')
        
        # Show appropriate notification based on rarity
        importance = 1
        if discovery_rarity == "rare":
            importance = 2
        elif discovery_rarity == "epic" or discovery_rarity == "legendary":
            importance = 3
            
        # Notify the player
        self.notifier.add(
            f"Discovered {discovery_name}!",
            category="discovery",
            importance=importance
        )
        
        # Apply discovery effects
        rewards_text = []
        
        if "currency" in discovery:
            currency_amount = discovery["currency"]
            self.player.currency += currency_amount
            rewards_text.append(f"{currency_amount} currency")
            
            # Track discovery rewards by type
            self.stats["discovery_currency_gained"] = self.stats.get("discovery_currency_gained", 0) + currency_amount
        
        if "items" in discovery and discovery["items"]:
            items_count = len(discovery["items"])
            rewards_text.append(f"{items_count} items")
            
            # Track items discovered
            self.stats["discovery_items_found"] = self.stats.get("discovery_items_found", 0) + items_count
            
        if "research" in discovery:
            research_points = discovery["research"]
            # If we implement a research system later
            rewards_text.append(f"{research_points} research points")
        
        # Comprehensive reward notification if multiple rewards
        if rewards_text:
            self.notifier.add(
                f"Obtained: {', '.join(rewards_text)}",
                category="discovery",
                importance=1
            )
        
        # Update discovery statistics
        self.stats["discoveries"] = self.stats.get("discoveries", 0) + 1
        self.stats[f"discoveries_{discovery_type}"] = self.stats.get(f"discoveries_{discovery_type}", 0) + 1
        self.stats[f"discoveries_{discovery_rarity}"] = self.stats.get(f"discoveries_{discovery_rarity}", 0) + 1
    
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
            f"Encountered {trader_name}",
            category="encounter",
            importance=2
        )
        
        # Add trader-specific details if available
        if "specialty" in trader:
            self.notifier.add(
                f"Specializes in {trader['specialty']}",
                category="trader",
                importance=1
            )
        
        # Could transition to a trading state here in the future
        # self.state = "TRADE"
        # self.current_trader = trader
        
        # Update trader statistics
        self.stats["trader_encounters"] = self.stats.get("trader_encounters", 0) + 1
        self.stats[f"trader_{trader_type}_encounters"] = self.stats.get(f"trader_{trader_type}_encounters", 0) + 1
        
    def force_encounter(self, encounter_type: str, position: Tuple[int, int], zone: str = None) -> None:
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
            force_encounter_start = log_performance_start("Force encounter generation")
            
            # Log the forced encounter request
            logging.info(f"Forcing {encounter_type} encounter at {position} in {zone or 'current zone'}")
            
            # If zone not specified, determine it from the position
            if zone is None:
                zone = self.get_current_zone(position)
            
            # Create appropriate context for encounter generation
            context = {
                "player_health": self.player.health,
                "player_level": getattr(self.player, 'level', 1),
                "player_currency": self.player.currency,
                "game_time": self.game_time,
                "zone": zone,
                "forced": True,  # Flag that this was a forced encounter
                "encounter_type": encounter_type
            }
            
            # Handle "dynamic" type by choosing based on game state
            if encounter_type == "dynamic":
                # Dynamic selection based on current game state
                hour = int(self.game_time / 60) % 24  # Game time hour (0-23)
                player_health_pct = self.player.health / 100.0
                
                # Evening hours tend toward combat, daytime toward discoveries
                if 18 <= hour <= 23 or 0 <= hour <= 5:  # Night hours
                    combat_weight = 0.7
                    discovery_weight = 0.1
                    trader_weight = 0.2
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
            else:
                chosen_type = encounter_type
            
            # Generate the encounter through the encounter generator
            encounter_result = self.encounter_generator.generate_encounter(position, chosen_type, context)
            
            if not encounter_result:
                logging.warning(f"Failed to generate forced {chosen_type} encounter")
                log_performance_end("Force encounter generation", force_encounter_start, "failed")
                return
            
            # Process the generated encounter immediately
            if chosen_type == "combat":
                self._handle_combat_encounter(encounter_result)
            elif chosen_type == "discovery":
                self._handle_discovery_encounter(encounter_result)
            elif chosen_type == "trader":
                self._handle_trader_encounter(encounter_result)
            
            # Update statistics for forced encounters
            self.stats["forced_encounters"] = self.stats.get("forced_encounters", 0) + 1
            self.stats[f"forced_{chosen_type}_encounters"] = self.stats.get(f"forced_{chosen_type}_encounters", 0) + 1
            
            # Complete performance timing
            log_performance_end("Force encounter generation", force_encounter_start, "success")
            
        except Exception as e:
            log_exception("Error forcing encounter", e)
            log_performance_end("Force encounter generation", force_encounter_start, "error")
            self.notifier.add(
                "Error generating encounter", 
                category="error",
                importance=2
            )
            log_exception("Error processing encounters", e)
    
    def process_combat_result(self, combat_result: Dict[str, Any]) -> None:
        """Process the results of a combat encounter.
        
        Args:
            combat_result: Dictionary containing combat results
        """
        # Start performance monitoring for combat result processing
        combat_processing_start = log_performance_start("Combat result processing")
        
        try:
            if not combat_result:
                log_performance_end("Combat result processing", combat_processing_start, "empty_result")
                return
                
            # Extract combat details for logging and stats
            outcome = combat_result.get("outcome", "unknown")
            enemy_type = combat_result.get("enemy_type", "unknown")
            encounter_zone = combat_result.get("zone", "unknown")
            difficulty = combat_result.get("difficulty", 1)
            
            # Log combat details
            logging.info(f"Combat outcome: {outcome} against {enemy_type} in zone {encounter_zone} (difficulty: {difficulty})")
            
            # Process based on outcome
            if outcome == "victory":
                # Player won the combat
                self.notifier.add(
                    f"Victory! {enemy_type} defeated",
                    category="combat",
                    importance=2
                )
                
                # Process rewards
                rewards = combat_result.get("rewards", {})
                total_reward_value = 0
                
                if rewards:
                    if "currency" in rewards:
                        self.player.currency += rewards["currency"]
                        total_reward_value += rewards["currency"]
                        self.notifier.add(
                            f"Gained {rewards['currency']} currency",
                            category="combat"
                        )
                        # Track currency gained from combat
                        self.stats["combat_currency_gained"] = self.stats.get("combat_currency_gained", 0) + rewards["currency"]
                    
                    if "items" in rewards and rewards["items"]:
                        items_count = len(rewards["items"])
                        self.notifier.add(
                            f"Acquired {items_count} items",
                            category="combat"
                        )
                        # Track items gained from combat
                        self.stats["combat_items_gained"] = self.stats.get("combat_items_gained", 0) + items_count
                        # Future: Add items to player inventory
                
                # Track reward value by difficulty
                difficulty_key = f"reward_value_diff_{int(difficulty)}"
                self.stats[difficulty_key] = self.stats.get(difficulty_key, 0) + total_reward_value
                
            elif outcome == "defeat":
                # Player lost the combat
                self.notifier.add(
                    f"Defeat! Your ships were overwhelmed by {enemy_type}",
                    category="combat",
                    importance=3
                )
                
                # Process penalties
                penalties = combat_result.get("penalties", {})
                total_penalty_value = 0
                
                if penalties:
                    if "currency" in penalties:
                        loss = min(self.player.currency, penalties["currency"])
                        self.player.currency -= loss
                        total_penalty_value += loss
                        self.notifier.add(
                            f"Lost {loss} currency",
                            category="combat"
                        )
                        # Track currency lost from combat
                        self.stats["combat_currency_lost"] = self.stats.get("combat_currency_lost", 0) + loss
                    
                    if "ships" in penalties:
                        ships_lost = penalties["ships"]
                        self.player.fleet_size = max(1, self.player.fleet_size - ships_lost)
                        self.stats["ships_lost"] = self.stats.get("ships_lost", 0) + ships_lost
                        self.notifier.add(
                            f"Lost {ships_lost} ships in battle",
                            category="combat"
                        )
                
                # Track penalty value by difficulty
                difficulty_key = f"penalty_value_diff_{int(difficulty)}"
                self.stats[difficulty_key] = self.stats.get(difficulty_key, 0) + total_penalty_value
            
            elif outcome == "escape":
                # Player escaped from combat
                self.notifier.add(
                    f"Escaped from {enemy_type}",
                    category="combat",
                    importance=2
                )
                
                # May have some minor penalties
                escape_cost = combat_result.get("escape_cost", 0)
                if escape_cost > 0:
                    self.notifier.add(
                        f"Escape cost: {escape_cost} damage to ships",
                        category="combat"
                    )
                    # Track escape costs
                    self.stats["escape_costs"] = self.stats.get("escape_costs", 0) + escape_cost
                
                # Track escapes by difficulty
                escape_key = f"escapes_diff_{int(difficulty)}"
                self.stats[escape_key] = self.stats.get(escape_key, 0) + 1
            
            # Update comprehensive combat statistics
            combat_stats = combat_result.get("stats", {})
            
            # Track basic combat metrics
            self.stats["total_combats"] = self.stats.get("total_combats", 0) + 1
            
            # Track outcome-specific stats
            if outcome == "victory":
                self.stats["combats_won"] = self.stats.get("combats_won", 0) + 1
            elif outcome == "defeat":
                self.stats["combats_lost"] = self.stats.get("combats_lost", 0) + 1
            elif outcome == "escape":
                self.stats["combats_escaped"] = self.stats.get("combats_escaped", 0) + 1
            
            # Track combat by zone
            zone_key = f"combats_in_{encounter_zone}"
            self.stats[zone_key] = self.stats.get(zone_key, 0) + 1
            
            # Track combat by enemy type
            enemy_key = f"combats_against_{enemy_type}"
            self.stats[enemy_key] = self.stats.get(enemy_key, 0) + 1
            
            # Track detailed combat metrics if available
            if combat_stats:
                for stat_name, stat_value in combat_stats.items():
                    self.stats[f"combat_{stat_name}"] = self.stats.get(f"combat_{stat_name}", 0) + stat_value
            
            # Performance tracking for specific combat types
            log_performance_metric(f"combat_processing_{outcome}", time.time() - combat_processing_start)
            
            # Log combat result
            logging.info(f"Combat result: {outcome} with {len(combat_result.get('log', []))} actions")
            
        except Exception as e:
            log_exception("Error processing combat result", e)
            log_exception("Error updating player fleet", e)
            # Attempt recovery
            try:
                self.player.reset_fleet_state()
                logging.info("Reset fleet state after error")
            except Exception as recovery_error:
                log_exception("Failed to recover from fleet error", recovery_error)

    def draw(self) -> None:
        """Render the current game state."""
        try:
            # Start performance timing
            render_start = log_performance_start("Render frame")

            # Fill background
            self.screen.fill(COLOR_BG)

            # Draw background pattern
            for y in range(0, WINDOW_HEIGHT, self.bg_pattern.get_height()):
                for x in range(0, WINDOW_WIDTH, self.bg_pattern.get_width()):
                    self.screen.blit(self.bg_pattern, (x, y))

            # State-specific rendering
            if self.state == STATE_PLAY:
                self.draw_play_state()
            elif self.state == STATE_SHOP:
                self.draw_shop_state()

            # Draw notifications (always shown)
            self.notifier.draw(self.screen)

            # Draw active tooltip if any
            if self.tooltip_text and self.show_tooltips:
                from ui.draw_utils import draw_tooltip

                draw_tooltip(
                    self.screen,
                    self.tooltip_text,
                    self.tooltip_position[0],
                    self.tooltip_position[1],
                )

            # Draw controls help if requested
            if self.display_controls_help:
                self.notifier.draw_tooltips(self.screen, WINDOW_WIDTH - 270, 50)

            # Draw FPS counter if enabled
            if SHOW_FPS:
                self.fps_history.append(self.clock.get_fps())
                if len(self.fps_history) > 60:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                draw_text(self.screen, f"FPS: {avg_fps:.1f}", 10, 10, 14, COLOR_TEXT)

            # Draw debug info if enabled
            if self.show_debug:
                self.draw_debug_info()

            # Draw paused indicator
            if self.paused:
                draw_panel(
                    self.screen,
                    pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 30, 200, 60),
                    color=(30, 30, 40, 200),
                    header="PAUSED",
                )
                draw_text(
                    self.screen,
                    "Press ESC to resume",
                    WINDOW_WIDTH // 2,
                    WINDOW_HEIGHT // 2 + 5,
                    16,
                    COLOR_TEXT,
                    align="center",
                )

            # End performance timing
            log_performance_end("Render frame", render_start)

            # Update display - moved to game loop
            # pygame.display.flip()
            
        except Exception as e:
            log_exception("Error in render process", e)

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

        # Draw minimap if enabled
        if self.show_minimap:
            minimap_rect = pygame.Rect(
                WINDOW_WIDTH - MINIMAP_SIZE - MINIMAP_PADDING,
                WINDOW_HEIGHT - MINIMAP_SIZE - MINIMAP_PADDING,
                MINIMAP_SIZE,
                MINIMAP_SIZE,
            )

            draw_minimap(
                self.screen,
                minimap_rect,
                self.field.grid,
                self.field.entity_grid,
                (self.player.x, self.player.y),
                self.renderer.field_offset_x,
                self.renderer.field_offset_y,
                self.zoom_level,
            )

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

        # Draw debug info if enabled
        if self.show_debug:
            self.draw_debug_info()

        # Draw player's ship info if enabled
        if self.show_ship_info:
            self.player.draw_ship_info(self.screen)


def run_game_loop(game):
    """Run the main game loop.
    
    Args:
        game (Game): The game instance to run.
    
    Returns:
        bool: True if the game completed successfully, False otherwise.
    """
    try:
        # Main game loop
        running = True
        while running:
            # Process input events
            running = game.handle_events()
            
            # Skip updates if paused
            if not game.paused:
                # Update game state
                game.update()
            
            # Always render (even when paused)
            game.draw()
            
            # Control frame rate
            game.clock.tick(60)  # Cap at 60 FPS
            
        return True
    except Exception as e:
        log_exception("Unhandled exception in main game loop", e)
        return False

def main():
    """Main entry point for the game."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.info("Starting Space Muck game...")
        
        # Create the game instance
        game = Game()
        
        # Run the game loop
        success = run_game_loop(game)
        
        if success:
            logging.info("Game completed successfully")
        else:
            logging.warning("Game loop terminated with errors")
            
    except Exception as e:
        log_exception("Critical error in main function", e)
        raise
    finally:
        # Ensure pygame is properly shut down
        pygame.quit()
        logging.info("Game shutdown complete")


if __name__ == "__main__":
    main()
