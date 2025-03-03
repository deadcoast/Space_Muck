#!/usr/bin/env python3
# filepath: /Users/deadcoast/PycharmProjects/Space_Muck/src/main.py
"""
Space Muck - Advanced Procedural Generation Edition

A space mining game featuring:
- Evolving asteroid fields with cellular automaton mechanics
- Multiple symbiote races that adapt and evolve
- Dynamic mining and upgrade systems
- Advanced procedural generation with multiple algorithms
"""

import gc
import logging
import os
import sys
import time
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pygame

from src.config import *
from src.world.asteroid_field import AsteroidField
from src.generators.procedural_generator import create_field_with_multiple_algorithms
from src.entities.player import Player
from src.entities.miner_entity import MinerEntity
from src.ui.shop import Shop
from src.ui.notification import NotificationManager
from src.ui.renderers import AsteroidFieldRenderer
from src.ui.draw_utils import (
    draw_text,
    draw_panel,
    draw_minimap,
    draw_progress_bar,
    draw_button,
)
from src.utils.logging_setup import (
    setup_logging,
    log_exception,
    LogContext,
    log_performance_start,
    log_performance_end,
    log_memory_usage,
)
import gc
import logging
import os
import sys
import time
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set, Union

import numpy as np
import pygame

from src.config import *
from src.world.asteroid_field import AsteroidField
from src.generators.procedural_generator import create_field_with_multiple_algorithms
from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
from src.entities.player import Player
from src.entities.miner_entity import MinerEntity
from src.ui.shop import Shop
from src.ui.notification import NotificationManager
from src.ui.renderers import AsteroidFieldRenderer, EffectsRenderer
from src.ui.draw_utils import (
    draw_text,
    draw_panel,
    draw_minimap,
    draw_progress_bar,
    draw_button,
    draw_histogram,
    draw_tooltip,
)
from src.utils.logging_setup import (
    setup_logging,
    log_exception,
    log_performance_start,
    log_performance_end,
    log_memory_usage,
    LogContext,
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
            for y in range(bg_pattern_size):
                for x in range(bg_pattern_size):
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

        # Notify player
        self.notifier.add(
            f"Field regenerated with seed {self.seed}!", category="event", importance=2
        )

        log_performance_end("Field regeneration", start_time)

    def handle_events(self) -> None:
        """Process all pygame events."""
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
                self.quit_game()

            elif event.type == pygame.KEYDOWN:
                self.handle_key_press(event.key)

            elif event.type == pygame.MOUSEMOTION:
                # Update hover position for tooltips
                self.hover_position = event.pos
                self.cursor_over_ui = self.check_cursor_over_ui()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle mouse clicks
                if event.button == 1:  # Left click
                    self.handle_mouse_click(event.pos)
                elif event.button == 3:  # Right click
                    if self.state == STATE_PLAY:
                        # Select race under cursor
                        grid_x, grid_y = self.screen_to_grid(event.pos[0], event.pos[1])
                        if (
                            0 <= grid_x < self.field.width
                            and 0 <= grid_y < self.field.height
                        ):
                            race_id = self.field.entity_grid[grid_y, grid_x]
                            if race_id > 0:
                                # Find the race
                                for i, race in enumerate(self.field.races):
                                    if race.race_id == race_id:
                                        self.selected_race_index = i
                                        self.show_race_details = True
                                        break

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
        if self.paused:
            return

        # Update timing
        self.frame_counter += 1
        current_time = time.time()

        # Update game time
        self.stats["time_played"] += self.delta_time

        # Periodic memory check in debug mode
        if self.show_debug and current_time - self.last_memory_check > 10:
            log_memory_usage("Periodic memory check")
            self.last_memory_check = current_time

        # State-specific updates
        if self.state == STATE_PLAY:
            self.update_play_state()

        # Update notifications regardless of state
        self.notifier.update()

        # Auto-upgrade if enabled
        if self.auto_upgrade:
            self.handle_auto_upgrade()

    def update_play_state(self) -> None:
        """Update game elements during play state."""
        # Handle player movement
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

        # Auto-mining if enabled
        if (
            self.auto_mine and self.frame_counter % 30 == 0
        ):  # Every half second at 60 FPS
            minerals_mined = self.player.mine(self.field)
            if minerals_mined > 0:
                self.stats["total_mined"] += minerals_mined

        # Update asteroid field at intervals
        current_time = time.time()
        if current_time - self.last_field_update_time > self.update_interval / 60:
            race_incomes = self.field.update()
            self.last_field_update_time = current_time

            # Process race incomes and notify about significant events
            for race_id, income in race_incomes.items():
                if income > 500:
                    if race := next(
                        (r for r in self.field.races if r.race_id == race_id), None
                    ):
                        self.notifier.add(
                            f"Race {race_id} ({race.trait}) found a large mineral deposit!",
                            category="race",
                            importance=2,
                        )

        # Update fleet status
        if self.frame_counter % 60 == 0:  # Once per second at 60 FPS
            fleet_results = self.player.update_fleet(self.field)

            # Notify about ship damage or loss
            if fleet_results["ships_lost"] > 0:
                self.notifier.add(
                    f"Lost {fleet_results['ships_lost']} mining ship(s) to symbiote attacks!",
                    category="event",
                    importance=3,
                )
                self.stats["ships_lost"] += fleet_results["ships_lost"]

            # Add mining income
            if fleet_results["minerals_mined"] > 0:
                self.stats["total_mined"] += fleet_results["minerals_mined"]

        # Check for race evolutions
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

        # Check for new resource discoveries by player
        if self.frame_counter % 120 == 0:  # Every 2 seconds at 60 FPS
            # Check for nearby anomalies
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = self.player.x + dx, self.player.y + dy
                    if (
                        0 <= nx < self.field.width
                        and 0 <= ny < self.field.height
                        and self.field.grid[ny, nx] > 0
                        and self.field.rare_grid[ny, nx] == 3
                    ):  # Anomaly
                        self.notifier.notify_event(
                            "discovery",
                            f"Found an anomaly at ({nx}, {ny})!",
                            importance=3,
                        )

    def draw(self) -> None:
        """Render the current game state."""
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
            from src.ui.draw_utils import draw_tooltip

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

        # Update display
        pygame.display.flip()

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
            for y in range(bg_pattern_size):
                for x in range(bg_pattern_size):
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

        # Notify player
        self.notifier.add(
            f"Field regenerated with seed {self.seed}!", category="event", importance=2
        )

        log_performance_end("Field regeneration", start_time)

    def handle_events(self) -> None:
        """Process all pygame events."""
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
                self.quit_game()

            elif event.type == pygame.KEYDOWN:
                self.handle_key_press(event.key)

            elif event.type == pygame.MOUSEMOTION:
                # Update hover position for tooltips
                self.hover_position = event.pos
                self.cursor_over_ui = self.check_cursor_over_ui()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle mouse clicks
                if event.button == 1:  # Left click
                    self.handle_mouse_click(event.pos)
                elif event.button == 3:  # Right click
                    if self.state == STATE_PLAY:
                        # Select race under cursor
                        grid_x, grid_y = self.screen_to_grid(event.pos[0], event.pos[1])
                        if (
                            0 <= grid_x < self.field.width
                            and 0 <= grid_y < self.field.height
                        ):
                            race_id = self.field.entity_grid[grid_y, grid_x]
                            if race_id > 0:
                                # Find the race
                                for i, race in enumerate(self.field.races):
                                    if race.race_id == race_id:
                                        self.selected_race_index = i
                                        self.show_race_details = True
                                        break

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
        if self.paused:
            return

        # Update timing
        self.frame_counter += 1
        current_time = time.time()

        # Update game time
        self.stats["time_played"] += self.delta_time

        # Periodic memory check in debug mode
        if self.show_debug and current_time - self.last_memory_check > 10:
            log_memory_usage("Periodic memory check")
            self.last_memory_check = current_time

        # State-specific updates
        if self.state == STATE_PLAY:
            self.update_play_state()

        # Update notifications regardless of state
        self.notifier.update()

        # Auto-upgrade if enabled
        if self.auto_upgrade:
            self.handle_auto_upgrade()

    def update_play_state(self) -> None:
        """Update game elements during play state."""
        # Handle player movement
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

        # Auto-mining if enabled
        if (
            self.auto_mine and self.frame_counter % 30 == 0
        ):  # Every half second at 60 FPS
            minerals_mined = self.player.mine(self.field)
            if minerals_mined > 0:
                self.stats["total_mined"] += minerals_mined

        # Update asteroid field at intervals
        current_time = time.time()
        if current_time - self.last_field_update_time > self.update_interval / 60:
            race_incomes = self.field.update()
            self.last_field_update_time = current_time

            # Process race incomes and notify about significant events
            for race_id, income in race_incomes.items():
                if income > 500:
                    if race := next(
                        (r for r in self.field.races if r.race_id == race_id), None
                    ):
                        self.notifier.add(
                            f"Race {race_id} ({race.trait}) found a large mineral deposit!",
                            category="race",
                            importance=2,
                        )

        # Update fleet status
        if self.frame_counter % 60 == 0:  # Once per second at 60 FPS
            fleet_results = self.player.update_fleet(self.field)

            # Notify about ship damage or loss
            if fleet_results["ships_lost"] > 0:
                self.notifier.add(
                    f"Lost {fleet_results['ships_lost']} mining ship(s) to symbiote attacks!",
                    category="event",
                    importance=3,
                )
                self.stats["ships_lost"] += fleet_results["ships_lost"]

            # Add mining income
            if fleet_results["minerals_mined"] > 0:
                self.stats["total_mined"] += fleet_results["minerals_mined"]

        # Check for race evolutions
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

        # Check for new resource discoveries by player
        if self.frame_counter % 120 == 0:  # Every 2 seconds at 60 FPS
            # Check for nearby anomalies
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = self.player.x + dx, self.player.y + dy
                    if (
                        0 <= nx < self.field.width
                        and 0 <= ny < self.field.height
                        and self.field.grid[ny, nx] > 0
                        and self.field.rare_grid[ny, nx] == 3
                    ):  # Anomaly
                        self.notifier.notify_event(
                            "discovery",
                            f"Found an anomaly at ({nx}, {ny})!",
                            importance=3,
                        )

    def draw(self) -> None:
        """Render the current game state."""
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
            from src.ui.draw_utils import draw_tooltip

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

        # Update display
        pygame.display.flip()

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

        # Draw UI elements
        self.notifier.draw_ui(self.screen, self.state, self.player)
        self.shop.draw(self.screen, self.state)

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

        # Update display
        pygame.display.flip()

    def draw_game_over(self) -> None:
        """Render the game over screen."""
        draw_panel(
            self.screen,
            pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 30, 200, 60),
            color=(30, 30, 40, 200),
            header="GAME OVER",
        )
        draw_text(
            self.screen,
            "Press ESC to quit",
            WINDOW_WIDTH // 2,
            WINDOW_HEIGHT // 2 + 5,
            16,
            COLOR_TEXT,
            align="center",
        )

        # End performance timing
        log_performance_end("Render frame", render_start)

        # Update display
        pygame.display.flip()


"""
Space Muck - Advanced Procedural Generation Edition

A space mining game featuring:
- Evolving asteroid fields with cellular automaton mechanics
- Multiple symbiote races that adapt and evolve
- Dynamic mining and upgrade systems
- Advanced procedural generation with multiple algorithms
"""


class Game:
    """
    Main game class that coordinates all game systems and manages the game loop.
    """

    def __init__(self) -> None:
        """Initialize the game environment, field, and entities."""
        # Initialize pygame and display
        pygame.init()
        self.screen: pygame.Surface = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        pygame.display.set_caption(
            f"Space Muck v{VERSION} - Procedural Generation Edition"
        )

        # Setup logging
        setup_logging(log_level=logging.INFO)
        logging.info(f"Starting Space Muck v{VERSION} ({BUILD_DATE})")

        # Create clock and initialize timing
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.frame_counter: int = 0
        self.update_interval: int = UPDATE_INTERVAL
        self.fps_history: List[int] = []
        self.last_time: float = time.time()

        # Set up random seed and initialize RNG
        self.seed: int = random.randint(1, 1000000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initialize field
        with LogContext("Field Initialization"):
            # Use advanced procedural generation
            self.field: AsteroidField = create_field_with_multiple_algorithms(
                width=GRID_WIDTH,
                height=GRID_HEIGHT,
                seed=self.seed,
                rare_chance=RARE_THRESHOLD,
                rare_bonus=RARE_BONUS_MULTIPLIER,
            )

        # Initialize player
        with LogContext("Player Initialization"):
            self.player: Player = Player()
            # Connect field and player
            self.field.player = self.player

        # Initialize races
        with LogContext("Race Initialization"):
            self.initialize_symbiote_races()

        # UI state
        self.selected_race_index: int = -1
        self.hover_position: Tuple[int, int] = (0, 0)
        self.tooltip_text: Optional[str] = None
        self.state: str = STATE_PLAY

        # UI components
        self.notifier: NotificationManager = NotificationManager()
        self.shop: Shop = Shop()
        self.field_renderer: AsteroidFieldRenderer = AsteroidFieldRenderer()
        self.effects_renderer: EffectsRenderer = EffectsRenderer()

        # Game state
        self.game_time: int = 0
        self.show_minimap: bool = True
        self.show_race_details: bool = False
        self.show_help: bool = False
        self.auto_mine: bool = False
        self.economy_stats: Dict[str, Any] = {
            "minerals_mined": 0,
            "rare_minerals_mined": 0,
            "ship_minerals": 0,
            "race_minerals": 0,
            "total_value": 0,
            "history": [],
        }

        # Initialize visual filters
        self.initialize_visual_settings()

        # Send initial notifications
        self.notifier.add(
            "Welcome to Space Muck! Press H for help.",
            duration=600,
            category="system",
            importance=3,
        )
        self.notifier.add(
            "Use WASD or arrow keys to move.", duration=600, category="system"
        )
        self.notifier.add(f"Seed: {self.seed}", duration=300, category="system")

        # Performance monitoring
        self.performance_metrics: Dict[str, List[float]] = {
            "update": [],
            "draw": [],
            "field_update": [],
            "entities_update": [],
            "ui_draw": [],
        }

        log_memory_usage("Game initialization complete")

    def initialize_symbiote_races(self) -> None:
        """Initialize available symbiote races with distinct traits and behaviors."""
        self.available_races: List[MinerEntity] = [
            # Blue race - adaptive
            MinerEntity(
                race_id=1,
                color=COLOR_RACE_1,
                birth_set={2, 3},
                survival_set={3, 4},
                initial_density=RACE_INITIAL_DENSITY,
            ),
            # Magenta race - expansive
            MinerEntity(
                race_id=2,
                color=COLOR_RACE_2,
                birth_set={3, 4},
                survival_set={2, 3},
                initial_density=RACE_INITIAL_DENSITY,
            ),
            # Orange race - selective
            MinerEntity(
                race_id=3,
                color=COLOR_RACE_3,
                birth_set={3},
                survival_set={2, 3, 4},
                initial_density=RACE_INITIAL_DENSITY,
            ),
        ]

        # Start with just one race
        starting_race = self.available_races[0]
        self.field.races = [starting_race]

        # Populate the field with the starting race
        with LogContext("Race Population Generation"):
            starting_race.populate(self.field)
            logging.info(
                f"Starting with race {starting_race.race_id} ({starting_race.trait})"
            )

    def initialize_visual_settings(self) -> None:
        """Initialize visual filter settings for the game."""
        self.visual_filters = {
            "show_grid": False,
            "show_energy": True,
            "show_all_asteroids": True,
            "show_symbiotes": True,
            "show_mining_radius": False,
            "show_symbiote_colonies": True,
            "show_mineral_values": False,
            "show_animations": True,
            "detail_level": "medium",  # low, medium, high
        }

        # Keyboard shortcuts for toggling filters
        self.filter_shortcuts = {
            pygame.K_g: "show_grid",
            pygame.K_e: "show_energy",
            pygame.K_y: "show_symbiotes",
            pygame.K_c: "show_symbiote_colonies",
            pygame.K_v: "show_mineral_values",
            pygame.K_d: "cycle_detail",  # Cycles through detail levels
        }

    def handle_filter_keys(self, key: int) -> None:
        """
        Handle keyboard shortcuts for visual filters.

        Args:
            key: Key code from Pygame
        """
        if key in self.filter_shortcuts:
            filter_name = self.filter_shortcuts[key]
            if filter_name == "cycle_detail":
                # Cycle through detail levels
                detail_levels = ["low", "medium", "high"]
                current_idx = detail_levels.index(self.visual_filters["detail_level"])
                next_idx = (current_idx + 1) % len(detail_levels)
                self.visual_filters["detail_level"] = detail_levels[next_idx]
                self.notifier.add(
                    f"Detail level: {self.visual_filters['detail_level']}",
                    category="system",
                )
            else:
                # Toggle boolean filter
                self.visual_filters[filter_name] = not self.visual_filters[filter_name]
                status = "On" if self.visual_filters[filter_name] else "Off"
                self.notifier.add(
                    f"{filter_name.replace('_', ' ').title()}: {status}",
                    category="system",
                )

    def auto_upgrade_logic(self) -> None:
        """Automatically purchase upgrades if auto-upgrade is enabled."""
        if not hasattr(self.player, "auto_upgrade") or not self.player.auto_upgrade:
            return

        affordable_upgrades = [
            option
            for option in self.shop.options
            if (
                option.get("category", "") == "ship"
                and self.player.currency >= option.get("cost", float("inf"))
                and not option.get("locked", False)
                and (
                    option.get("max_level", None) is None
                    or option.get("current_level", 0) < option.get("max_level", 0)
                )
            )
        ]
        # Sort by cost (cheapest first) for optimal resource usage
        affordable_upgrades.sort(key=lambda x: x.get("cost", float("inf")))

        # Buy the cheapest upgrade if we can afford it
        if affordable_upgrades:
            upgrade = affordable_upgrades[0]
            if success := self.shop.purchase_upgrade(
                upgrade, self.player, self.field, self.notifier
            ):
                self.notifier.add(
                    f"Auto-upgrade: Purchased {upgrade['name']} for {upgrade['cost']} credits",
                    category="upgrade",
                )

    def handle_events(self) -> None:
        """Process all game events."""
        for event in pygame.event.get():
            # Handle game exit
            if event.type == pygame.QUIT:
                self.quit_game()

            # Let the notification manager handle its events first
            if self.notifier.handle_event(event):
                continue

            # Handle keyboard input
            if event.type == pygame.KEYDOWN:
                # Global hotkeys (work in any state)
                if event.key == pygame.K_F1 or event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_m:
                    self.show_minimap = not self.show_minimap
                    self.notifier.add(
                        f"Minimap: {'On' if self.show_minimap else 'Off'}",
                        category="system",
                    )
                elif event.key == pygame.K_n:
                    # Handled by notification manager
                    pass
                elif event.key == pygame.K_F12:
                    # Take screenshot
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                    pygame.image.save(self.screen, filename)
                    self.notifier.add(
                        f"Screenshot saved: {filename}", category="system"
                    )

                # State-specific input
                if self.state == STATE_PLAY:
                    self.handle_play_state_input(event)
                elif self.state == STATE_SHOP:
                    # Handle shop state input
                    if event.key == pygame.K_ESCAPE:
                        self.state = STATE_PLAY
                    elif self.shop.handle_event(
                        event, self.player, self.field, self.notifier
                    ):
                        self.state = STATE_PLAY

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle clicking on the minimap
                if self.show_minimap:
                    minimap_rect = pygame.Rect(
                        10,
                        WINDOW_HEIGHT - MINIMAP_SIZE - 10,
                        MINIMAP_SIZE,
                        MINIMAP_SIZE,
                    )
                    if minimap_rect.collidepoint(event.pos):
                        # Convert click position to world coordinates
                        mx, my = event.pos
                        rel_x = (mx - minimap_rect.x) / minimap_rect.width
                        rel_y = (my - minimap_rect.y) / minimap_rect.height

                        # Move camera to clicked position
                        world_x = int(rel_x * self.field.width)
                        world_y = int(rel_y * self.field.height)

                        self.player.x = world_x
                        self.player.y = world_y
                        self.field.camera_x = world_x
                        self.field.camera_y = world_y

            elif event.type == pygame.MOUSEWHEEL:
                if self.state == STATE_PLAY:
                    zoom_factor = 1.1 if event.y > 0 else 0.9
                    self.player.zoom_camera(self.field, zoom_factor)

    def handle_play_state_input(self, event: pygame.event.Event) -> None:
        """
        Handle keyboard input when in play state.

        Args:
            event: Pygame event to process
        """
        if event.key == pygame.K_s and not pygame.key.get_mods() & pygame.KMOD_SHIFT:
            self.state = STATE_SHOP

        elif event.key == pygame.K_SPACE:
            # Mine asteroids at player location
            reward = self.player.mine(self.field)
            if reward > 0:
                self.notifier.add(
                    f"Mined asteroids: +{reward} credits", category="mining"
                )
                # Add visual effect
                if self.visual_filters["show_animations"]:
                    self.effects_renderer.add_effect(
                        "sparkle",
                        self.player.x,
                        self.player.y,
                        size=min(30, max(5, reward // 10)),
                        color=COLOR_SUCCESS,
                    )

        elif event.key == pygame.K_r:
            # Manual asteroid seeding
            self.field.manual_seed(self.player.x, self.player.y, radius=5)
            self.notifier.add("Manual seeding activated.", category="field")
            # Add visual effect
            if self.visual_filters["show_animations"]:
                self.effects_renderer.add_effect(
                    "bloom",
                    self.player.x,
                    self.player.y,
                    size=15,
                    color=COLOR_ASTEROID_RARE,
                )

        elif event.key in (pygame.K_LEFT, pygame.K_a):
            self.player.move(-self.player.move_speed, 0, self.field)
        elif event.key in (pygame.K_RIGHT, pygame.K_d):
            self.player.move(self.player.move_speed, 0, self.field)
        elif event.key in (pygame.K_UP, pygame.K_w):
            self.player.move(0, -self.player.move_speed, self.field)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.player.move(0, self.player.move_speed, self.field)

        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
            self.player.zoom_camera(self.field, 1.25)
            self.notifier.add("Zoomed in", category="system", duration=60)
        elif event.key == pygame.K_MINUS:
            self.player.zoom_camera(self.field, 0.8)
            self.notifier.add("Zoomed out", category="system", duration=60)

        elif event.key == pygame.K_w and pygame.key.get_mods() & pygame.KMOD_SHIFT:
            for _ in range(10):
                self.player.move(0, -self.player.move_speed, self.field)
            self.notifier.add("Fast move up", category="system", duration=30)
        elif event.key == pygame.K_a and pygame.key.get_mods() & pygame.KMOD_SHIFT:
            for _ in range(10):
                self.player.move(-self.player.move_speed, 0, self.field)
            self.notifier.add("Fast move left", category="system", duration=30)
        elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_SHIFT:
            for _ in range(10):
                self.player.move(0, self.player.move_speed, self.field)
            self.notifier.add("Fast move down", category="system", duration=30)
        elif event.key == pygame.K_d and pygame.key.get_mods() & pygame.KMOD_SHIFT:
            for _ in range(10):
                self.player.move(self.player.move_speed, 0, self.field)
            self.notifier.add("Fast move right", category="system", duration=30)

        elif event.key == pygame.K_m and pygame.key.get_mods() & pygame.KMOD_SHIFT:
            self.auto_mine = not self.auto_mine
            self.notifier.add(
                f"Auto-mining {'enabled' if self.auto_mine else 'disabled'}",
                category="mining",
                duration=120,
            )

        elif event.key == pygame.K_u:
            if hasattr(self.player, "auto_upgrade"):
                self.player.auto_upgrade = not self.player.auto_upgrade
                self.notifier.add(
                    f"Auto-upgrades {'enabled' if self.player.auto_upgrade else 'disabled'}",
                    category="upgrade",
                    duration=120,
                )
            else:
                self.player.auto_upgrade = True
                self.notifier.add(
                    "Auto-upgrades enabled", category="upgrade", duration=120
                )

        elif event.key == pygame.K_f:
            # Feed a small amount
            minerals_fed = self.player.feed_symbiotes(self.field, 50)
            if minerals_fed > 0:
                self.notifier.add(
                    f"Fed symbiotes: {minerals_fed} minerals",
                    category="race",
                    importance=2,
                )
                # Visual effect
                if self.visual_filters["show_animations"]:
                    self.effects_renderer.add_effect(
                        "spray",
                        self.player.x,
                        self.player.y,
                        size=10,
                        color=COLOR_RACE_1,
                    )
            else:
                self.notifier.add(
                    "Not enough minerals for feeding", category="race", importance=1
                )

        elif event.key == pygame.K_b:
            # Build a new mining ship if resources allow
            if (
                self.player.currency >= self.player.ship_cost
                and self.player.mining_ships < self.player.max_mining_ships
            ):
                if success := self.player.add_ship():
                    self.notifier.add(
                        f"New mining ship built! Fleet: {self.player.mining_ships}",
                        category="upgrade",
                        importance=2,
                    )
                    # Visual effect
                    if self.visual_filters["show_animations"]:
                        self.effects_renderer.add_effect(
                            "explosion",
                            self.player.x,
                            self.player.y,
                            size=5,
                            color=COLOR_PLAYER,
                        )
            elif self.player.mining_ships >= self.player.max_mining_ships:
                self.notifier.add(
                    "Maximum fleet size reached!",
                    color=COLOR_WARNING,
                    category="upgrade",
                )
            else:
                self.notifier.add(
                    f"Not enough credits! Need {self.player.ship_cost}",
                    color=COLOR_WARNING,
                    category="upgrade",
                )

        elif event.key == pygame.K_i:
            # Toggle race details panel
            self.show_race_details = not self.show_race_details

        else:
            self.handle_filter_keys(event.key)

    def update(self) -> None:
        """Update the game state for the current frame."""
        # Increment frame counter
        self.frame_counter += 1
        self.game_time += 1

        # Measure performance
        update_start = log_performance_start("game_update")

        # Update field periodically to control game speed
        if self.frame_counter % self.update_interval == 0:
            # Measure field update performance
            field_start = log_performance_start("field_update")

            self.field.update()

            field_time = log_performance_end("field_update", field_start)
            self.performance_metrics["field_update"].append(field_time)

            # Update player fleet
            fleet_results = self.player.update_fleet(self.field)

            # Report significant fleet events
            if fleet_results["minerals_mined"] > 0:
                self.economy_stats["ship_minerals"] += fleet_results["minerals_mined"]
                self.economy_stats["total_value"] += fleet_results["minerals_mined"]

                if fleet_results["minerals_mined"] > 100:
                    self.notifier.add(
                        f"Mining fleet income: +{fleet_results['minerals_mined']} minerals",
                        category="mining",
                        duration=60,
                    )

            if fleet_results["damage_taken"] > 0:
                self.notifier.add(
                    f"Fleet attacked! Damage taken: {fleet_results['damage_taken']}",
                    color=COLOR_WARNING,
                    category="event",
                    importance=2,
                    duration=90,
                )

            if fleet_results["ships_lost"] > 0:
                self.notifier.add(
                    f"ALERT: {fleet_results['ships_lost']} ships lost to symbiote attacks!",
                    color=COLOR_ERROR,
                    category="event",
                    importance=3,
                    duration=180,
                )

            # Add visual effect for ship destruction
            if (
                self.visual_filters["show_animations"]
                and fleet_results["ships_lost"] > 0
            ):
                for _ in range(min(fleet_results["ships_lost"], 3)):  # Limit effects
                    offset_x = random.randint(-5, 5)
                    offset_y = random.randint(-5, 5)
                    self.effects_renderer.add_effect(
                        "explosion",
                        self.player.x + offset_x,
                        self.player.y + offset_y,
                        size=15,
                        color=COLOR_ERROR,
                    )

            # Auto-mining
            if self.auto_mine and self.player.auto_miners > 0:
                mined = self.player.auto_mine(self.field)
                if mined > 0:
                    self.economy_stats["minerals_mined"] += mined
                    self.economy_stats["total_value"] += mined
                    if mined > 50:
                        self.notifier.add(
                            f"Auto-mining: +{mined} minerals",
                            category="mining",
                            duration=60,
                        )

            # Calculate metrics for each symbiote race
            entities_start = log_performance_start("entities_update")

            for race in self.field.races:
                race_count = np.sum(self.field.entity_grid == race.race_id)
                race.population = race_count

                # Only show race updates occasionally to avoid spam
                if self.frame_counter % 300 == 0:
                    hunger_status = (
                        "Starving"
                        if race.hunger > 0.8
                        else "Hungry"
                        if race.hunger > 0.4
                        else "Satiated"
                    )

                    behavior_desc = {
                        "feeding": "searching for food",
                        "expanding": "expanding territory",
                        "defensive": "defending territory",
                        "migrating": "migrating to new areas",
                        "aggressive": "attacking competitors",
                    }.get(race.current_behavior, race.current_behavior)

                    self.notifier.add(
                        f"Race {race.race_id}: {race_count} symbiotes, {hunger_status}, {behavior_desc}",
                        duration=120,
                        color=race.color,
                        category="race",
                    )

                # Track history
                race.population_history.append(race_count)
                if len(race.population_history) > 100:
                    race.population_history = race.population_history[-100:]

                # Process evolution
                race.process_evolution(self.field)

            entities_time = log_performance_end("entities_update", entities_start)
            self.performance_metrics["entities_update"].append(entities_time)

            # Auto-upgrade check if enabled
            self.auto_upgrade_logic()

            # Update economy stats
            if self.frame_counter % 60 == 0:
                self.update_economy()

        # Update animations and effects
        if self.visual_filters["show_animations"]:
            self.effects_renderer.update()

        # Update UI elements
        self.shop.update_animation(1.0 / FPS)
        self.notifier.update()

        # Track performance
        update_time = log_performance_end("game_update", update_start)
        self.performance_metrics["update"].append(update_time)

        # Calculate FPS every second
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.clock.get_fps()
            self.fps_history.append(int(self.fps))
            if len(self.fps_history) > 60:
                self.fps_history = self.fps_history[-60:]
            self.last_time = current_time

    def update_economy(self) -> None:
        """Update the game's economy based on mining and symbiote activity."""
        # Update history
        self.economy_stats["history"].append(
            {
                "player_minerals": self.player.total_mined,
                "race_minerals": sum(race.last_income for race in self.field.races),
                "ship_minerals": self.economy_stats["ship_minerals"],
                "rare_minerals": self.player.total_rare_mined,
                "time": self.game_time,
            }
        )

        if len(self.economy_stats["history"]) > 60:
            self.economy_stats["history"] = self.economy_stats["history"][-60:]

        # Reset counters for new period
        self.economy_stats["ship_minerals"] = 0

    def draw(self) -> None:
        """Render the game to the screen."""
        # Start performance measurement
        draw_start = log_performance_start("draw")

        # Fill screen with background color
        self.screen.fill(COLOR_BG)

        # Draw the asteroid field
        self.field.draw(self.screen)

        # Draw effects if enabled
        if self.visual_filters["show_animations"]:
            self.effects_renderer.render(
                self.screen,
                self.field.get_view_bounds(),
                int(CELL_SIZE * self.field.zoom),
            )

        # Draw player on top
        self.player.draw(self.screen, self.field.zoom)

        # Draw player's fleet
        self.player.draw_ships(
            self.screen, *self.field.get_view_bounds(), self.field.zoom
        )

        # Draw minimap if enabled
        if self.show_minimap:
            self.draw_minimap(self.screen)

        # Draw status panel
        self.draw_status_panel(self.screen)

        # Draw race details panel if toggled
        if self.show_race_details:
            self.draw_race_stats(self.screen)

        # Draw shop when in shop state
        if self.state == STATE_SHOP:
            self.shop.draw(self.screen, self.player)

        # Draw help overlay when active
        if self.show_help:
            self.draw_help_overlay(self.screen)

        # Draw notification panel last (so it's on top)
        self.notifier.draw(self.screen)

        # Draw tooltip at cursor location if present
        if self.tooltip_text:
            draw_tooltip(
                self.screen, self.tooltip_text, *pygame.mouse.get_pos(), max_width=250
            )

        # Show welcome message for the first 10 seconds
        if self.frame_counter < 300:
            self.draw_welcome_panel()

        # Show FPS counter and coordinates
        if SHOW_FPS:
            draw_text(
                self.screen,
                f"FPS: {int(self.clock.get_fps())}",
                10,
                10,
                size=16,
                color=(
                    COLOR_TEXT if self.clock.get_fps() >= FPS * 0.9 else COLOR_WARNING
                ),
            )

        # Show coordinate info
        draw_text(
            self.screen,
            f"Position: ({self.player.x}, {self.player.y})",
            10,
            WINDOW_HEIGHT - 30,
            size=16,
            color=COLOR_TEXT,
        )

        # Record performance
        draw_time = log_performance_end("draw", draw_start)
        self.performance_metrics["draw"].append(draw_time)

    def draw_minimap(self, surface: pygame.Surface) -> None:
        """
        Draw a minimap in the bottom left corner of the screen.

        Args:
            surface: Surface to draw on
        """
        # Define minimap dimensions
        minimap_size = MINIMAP_SIZE
        padding = MINIMAP_PADDING
        minimap_rect = pygame.Rect(
            padding, WINDOW_HEIGHT - minimap_size - padding, minimap_size, minimap_size
        )

        # Draw the minimap
        draw_minimap(
            surface,
            minimap_rect,
            self.field.grid,
            self.field.entity_grid,
            (self.player.x, self.player.y),
            self.field.get_view_bounds(),
        )

        # Draw legend below minimap
        legend_y = WINDOW_HEIGHT - padding - 18

        # Draw player indicator
        pygame.draw.rect(surface, COLOR_PLAYER, (padding, legend_y, 10, 10))
        draw_text(surface, "You", padding + 15, legend_y - 2, size=12)

        # Draw race indicators
        legend_x = padding + 60
        for race in self.field.races:
            pygame.draw.rect(surface, race.color, (legend_x, legend_y, 10, 10))
            draw_text(surface, race.name, legend_x + 15, legend_y - 2, size=12)
            legend_x += 30

        # Draw rare asteroid indicator
        pygame.draw.rect(surface, (255, 215, 0), (legend_x, legend_y, 10, 10))
        draw_text(surface, "Rare", legend_x + 15, legend_y - 2, size=12)

        # Draw normal asteroid indicator
        pygame.draw.rect(surface, (200, 200, 200), (legend_x + 30, legend_y, 10, 10))
        draw_text(surface, "Normal", legend_x + 45, legend_y - 2, size=12)

        # Draw ship indicator
        pygame.draw.rect(surface, COLOR_SHIP, (legend_x + 60, legend_y, 10, 10))
        draw_text(surface, "Ship", legend_x + 75, legend_y - 2, size=12)

        # Draw ship indicator
        pygame.draw.rect(surface, COLOR_SHIP, (legend_x + 90, legend_y, 10, 10))
        draw_text(surface, "Fleet", legend_x + 105, legend_y - 2, size=12)

        # Draw ship indicator
        pygame.draw.rect(surface, COLOR_SHIP, (legend_x + 120, legend_y, 10, 10))
        draw_text(surface, "Symbiote", legend_x + 135, legend_y - 2, size=12)

        # Draw ship indicator
        pygame.draw.rect(surface, COLOR_SHIP, (legend_x + 150, legend_y, 10, 10))
        draw_text(surface, "Symbiote fleet", legend_x + 165, legend_y - 2, size=12)
