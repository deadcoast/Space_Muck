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
import random
import time
from typing import Dict, List, Tuple, Any, Optional

# Third-party library imports
import numpy as np
import pygame

# Local application imports
# Game constants and configuration
from config import (
    # Version information
    VERSION,
    # Window and display settings
    WINDOW_WIDTH, WINDOW_HEIGHT, 
    GRID_WIDTH, GRID_HEIGHT, MINIMAP_SIZE, MINIMAP_PADDING,
    VIEW_WIDTH, VIEW_HEIGHT,
    # Performance settings
    UPDATE_INTERVAL, DEBUG_MODE, SHOW_GRID, SHOW_FPS,
    # Game states
    STATE_PLAY, STATE_SHOP,
    # Colors
    COLOR_BG, COLOR_TEXT,
    COLOR_RACE_1, COLOR_RACE_2, COLOR_RACE_3,
    COLOR_PLAYER,
    # Race settings
    RACE_INITIAL_DENSITY
)

# Game components
from generators.asteroid_field import AsteroidField
from entities.player import Player
from entities.miner_entity import MinerEntity
from ui.shop import Shop
from ui.notification import NotificationManager
from ui.renderers import AsteroidFieldRenderer
from ui.draw_utils import (
    draw_text,
    draw_panel,
    draw_minimap,
    draw_progress_bar,
    draw_button,
)
from utils.logging_setup import (
    log_exception,
    LogContext,
    log_performance_start,
    log_performance_end,
    log_memory_usage,
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
            for y, x in itertools.product(range(bg_pattern_size), range(bg_pattern_size)):
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
                self.handle_mouse_motion(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_button_down(event)

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
        self.handle_player_movement()
        self.handle_auto_mining()
        self.update_asteroid_field()
        self.check_race_evolutions()
        self.check_for_discoveries()
        
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
            
    def handle_auto_mining(self) -> None:
        """Handle auto-mining if enabled."""
        if not (self.auto_mine and self.frame_counter % 30 == 0):  # Every half second at 60 FPS
            return
            
        minerals_mined = self.player.mine(self.field)
        if minerals_mined > 0:
            self.stats["total_mined"] += minerals_mined
            
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
        if self.frame_counter % 120 != 0:  # Every 2 seconds at 60 FPS
            return
            
        # Check for nearby anomalies
        for dy, dx in itertools.product(range(-3, 4), range(-3, 4)):
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

