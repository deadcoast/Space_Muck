"""
Specialized rendering components for Space Muck.

This module contains classes and functions for rendering different game elements
in a consistent and optimized way, separating the rendering logic from the game objects.
"""

import itertools

# Standard library imports
import math
import random
from typing import List, Tuple

# Third-party library imports
import numpy as np
import pygame

# Local application imports
from config import (
    COLOR_ASTEROID_RARE,
    COLOR_BG,
    COLOR_GRID,
    COLOR_RACE_1,
    COLOR_RACE_2,
    COLOR_RACE_3,
)
from .draw_utils import draw_text, draw_minimap


class AsteroidFieldRenderer:
    """Handles rendering of the asteroid field with optimized performance."""

    def __init__(self) -> None:
        """Initialize the asteroid field renderer."""
        # Cache for pre-rendered surfaces
        self.cell_surfaces = {}  # Cache for asteroid surfaces
        self.cell_colors = {}  # Cache for asteroid colors
        self.entity_colors = {}  # Cache for entity colors
        self.rare_overlay = None  # Surface for rare asteroid overlay

        # Color gradient for asteroid values
        self.asteroid_gradient = [
            (80, 80, 80),  # Base color for low value
            (120, 120, 120),  # Medium value
            (180, 180, 180),  # High value
        ]

        # Create color cache for asteroid values
        for i in range(101):
            # Normalize value to 0-1 range
            normalized = i / 100.0

            # Calculate color from gradient
            if normalized <= 0.5:
                # Mix first and second colors
                t = normalized / 0.5
                color = self._interpolate_color(
                    self.asteroid_gradient[0], self.asteroid_gradient[1], t
                )
            else:
                # Mix second and third colors
                t = (normalized - 0.5) / 0.5
                color = self._interpolate_color(
                    self.asteroid_gradient[1], self.asteroid_gradient[2], t
                )

            self.cell_colors[i] = color

        # Set entity colors
        self.entity_colors[1] = COLOR_RACE_1
        self.entity_colors[2] = COLOR_RACE_2
        self.entity_colors[3] = COLOR_RACE_3

        # Rare minerals color
        self.rare_color = COLOR_ASTEROID_RARE

        # Initialize cell surfaces
        self._init_cell_surfaces()

    def _init_cell_surfaces(self) -> None:
        """Initialize pre-rendered cell surfaces for better performance."""
        # Base cell size for pre-rendering
        base_size = 16

        # Create asteroid surfaces for different values
        for i in range(0, 101, 5):  # Create every 5th value to save memory
            color = self.cell_colors[i]

            # Create surface for regular asteroid
            surface = pygame.Surface((base_size, base_size), pygame.SRCALPHA)

            # Draw asteroid with some texture/variation
            pygame.draw.circle(
                surface, color, (base_size // 2, base_size // 2), base_size // 2 - 1
            )

            # Add some shading/detail
            highlight = self._interpolate_color(color, (255, 255, 255), 0.3)
            shadow = self._interpolate_color(color, (0, 0, 0), 0.3)

            # Add highlight and shadow
            pygame.draw.circle(
                surface, highlight, (base_size // 3, base_size // 3), base_size // 6
            )
            pygame.draw.circle(
                surface,
                shadow,
                (2 * base_size // 3, 2 * base_size // 3),
                base_size // 5,
            )

            self.cell_surfaces[i] = surface

        # Create rare overlay
        self.rare_overlay = pygame.Surface((base_size, base_size), pygame.SRCALPHA)
        pygame.draw.circle(
            self.rare_overlay,
            (*self.rare_color, 180),
            (base_size // 2, base_size // 2),
            base_size // 2 - 2,
            2,
        )

        # Small sparkles
        for _ in range(4):
            x = random.randint(4, base_size - 4)
            y = random.randint(4, base_size - 4)
            size = random.randint(1, 3)
            pygame.draw.circle(self.rare_overlay, (*self.rare_color, 230), (x, y), size)

    def _interpolate_color(
        self, color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float
    ) -> Tuple[int, int, int]:
        """Interpolate between two colors."""
        r = int(color1[0] * (1 - t) + color2[0] * t)
        g = int(color1[1] * (1 - t) + color2[1] * t)
        b = int(color1[2] * (1 - t) + color2[2] * t)
        return (r, g, b)

    def get_cell_color(self, value: int, is_rare: bool = False) -> Tuple[int, int, int]:
        """Get color for an asteroid cell based on its value."""
        if value <= 0:
            return COLOR_BG

        # Use cached color or closest value
        color_key = min(self.cell_colors.keys(), key=lambda k: abs(k - value))
        base_color = self.cell_colors[color_key]

        # Adjust color for rare asteroids
        if is_rare:
            # Blend with gold/yellow for rare asteroids
            return self._interpolate_color(base_color, self.rare_color, 0.5)

        return base_color

    def get_cell_surface(self, value: int, cell_size: int) -> pygame.Surface:
        """Get a scaled surface for an asteroid cell."""
        if value <= 0:
            return None

        # Get closest pre-rendered surface
        surface_key = min(self.cell_surfaces.keys(), key=lambda k: abs(k - value))
        base_surface = self.cell_surfaces[surface_key]

        # Scale to current cell size if needed
        if base_surface.get_width() != cell_size:
            return pygame.transform.scale(base_surface, (cell_size, cell_size))

        return base_surface

    def render_field(
        self,
        surface: pygame.Surface,
        grid: np.ndarray,
        rare_grid: np.ndarray,
        entity_grid: np.ndarray,
        view_bounds: Tuple[int, int, int, int],
        cell_size: int,
        draw_grid_lines: bool = True,
    ) -> None:
        """
        Render the visible portion of the asteroid field.

        Args:
            surface: Pygame surface to render on
            grid: The asteroid value grid
            rare_grid: The rare status grid
            entity_grid: The entity grid
            view_bounds: (min_x, min_y, max_x, max_y) of the visible area
            cell_size: Size of each cell in pixels
            draw_grid_lines: Whether to draw grid lines
        """
        min_x, min_y, max_x, max_y = view_bounds

        # Fill background
        surface.fill(COLOR_BG)

        # Draw grid lines first if enabled
        if draw_grid_lines and cell_size >= 4:
            for x in range(min_x, max_x + 1):
                pygame.draw.line(
                    surface,
                    COLOR_GRID,
                    ((x - min_x) * cell_size, 0),
                    ((x - min_x) * cell_size, surface.get_height()),
                    1,
                )
            for y in range(min_y, max_y + 1):
                pygame.draw.line(
                    surface,
                    COLOR_GRID,
                    (0, (y - min_y) * cell_size),
                    (surface.get_width(), (y - min_y) * cell_size),
                    1,
                )

        # Draw entities (under asteroids)
        for y in range(min_y, min(max_y + 1, entity_grid.shape[0])):
            for x in range(min_x, min(max_x + 1, entity_grid.shape[1])):
                entity_id = entity_grid[y, x]
                if entity_id > 0:
                    color = self.entity_colors.get(entity_id, (255, 255, 255))
                    screen_x = (x - min_x) * cell_size
                    screen_y = (y - min_y) * cell_size

                    # Draw entity as small shapes based on race
                    if entity_id == 1:  # First race (blue)
                        # Draw as diamond
                        pygame.draw.polygon(
                            surface,
                            color,
                            [
                                (screen_x + cell_size // 2, screen_y),
                                (screen_x + cell_size, screen_y + cell_size // 2),
                                (screen_x + cell_size // 2, screen_y + cell_size),
                                (screen_x, screen_y + cell_size // 2),
                            ],
                        )
                    elif entity_id == 2:  # Second race (magenta)
                        # Draw as triangle
                        pygame.draw.polygon(
                            surface,
                            color,
                            [
                                (screen_x + cell_size // 2, screen_y),
                                (screen_x + cell_size, screen_y + cell_size),
                                (screen_x, screen_y + cell_size),
                            ],
                        )
                    else:  # Third race (orange)
                        # Draw as circle
                        pygame.draw.circle(
                            surface,
                            color,
                            (screen_x + cell_size // 2, screen_y + cell_size // 2),
                            cell_size // 2,
                        )

        # Draw asteroids
        for y in range(min_y, min(max_y + 1, grid.shape[0])):
            for x in range(min_x, min(max_x + 1, grid.shape[1])):
                value = grid[y, x]
                if value > 0:
                    screen_x = (x - min_x) * cell_size
                    screen_y = (y - min_y) * cell_size

                    is_rare = rare_grid[y, x] == 1

                    if cell_size >= 8:
                        if cell_surface := self.get_cell_surface(value, cell_size):
                            surface.blit(cell_surface, (screen_x, screen_y))

                            # Add rare overlay if needed
                            if is_rare:
                                rare_scaled = pygame.transform.scale(
                                    self.rare_overlay, (cell_size, cell_size)
                                )
                                surface.blit(rare_scaled, (screen_x, screen_y))
                    else:
                        # For small cell sizes, use simple rects for performance
                        color = self.get_cell_color(value, is_rare)
                        pygame.draw.rect(
                            surface, color, (screen_x, screen_y, cell_size, cell_size)
                        )


class PlayerRenderer:
    """Handles rendering of the player and player-related UI elements."""

    def __init__(self) -> None:
        """Initialize the player renderer."""
        self.ship_surface = None
        self.exhaust_animation = []
        self.ship_animation_frame = 0
        self.animation_speed = 0.2
        self.animation_counter = 0

        # Initialize ship graphics
        self._init_ship_graphics()

    def _init_ship_graphics(self) -> None:
        """Initialize ship and related graphics."""
        # Base ship size
        ship_size = 32

        # Create ship surface with alpha channel
        self.ship_surface = pygame.Surface((ship_size, ship_size), pygame.SRCALPHA)

        # Draw ship body (triangle shape)
        ship_color = (0, 220, 0)
        pygame.draw.polygon(
            self.ship_surface,
            ship_color,
            [
                (ship_size // 2, 0),  # Nose
                (ship_size, ship_size),  # Bottom right
                (0, ship_size),  # Bottom left
            ],
        )

        # Add cockpit
        cockpit_color = (150, 255, 150)
        pygame.draw.ellipse(
            self.ship_surface,
            cockpit_color,
            (ship_size // 4, ship_size // 3, ship_size // 2, ship_size // 3),
        )

        # Create exhaust animation frames
        for i in range(4):
            exhaust_surf = pygame.Surface((ship_size, ship_size // 2), pygame.SRCALPHA)

            # Variation in exhaust size
            size_var = i / 4.0

            # Draw exhaust flames
            flame_length = int(ship_size // 3 * (1 + size_var))

            # Main flame
            pygame.draw.polygon(
                exhaust_surf,
                (255, 100 + i * 40, 0, 200),
                [
                    (ship_size // 2, 0),
                    (ship_size // 2 + ship_size // 6, flame_length),
                    (ship_size // 2 - ship_size // 6, flame_length),
                ],
            )

            # Side flames
            side_flame_length = int(flame_length * 0.7)
            pygame.draw.polygon(
                exhaust_surf,
                (255, 150 + i * 20, 0, 150),
                [
                    (ship_size // 4, 0),
                    (ship_size // 4 + ship_size // 8, side_flame_length),
                    (ship_size // 4 - ship_size // 8, side_flame_length),
                ],
            )

            pygame.draw.polygon(
                exhaust_surf,
                (255, 150 + i * 20, 0, 150),
                [
                    (3 * ship_size // 4, 0),
                    (3 * ship_size // 4 + ship_size // 8, side_flame_length),
                    (3 * ship_size // 4 - ship_size // 8, side_flame_length),
                ],
            )

            self.exhaust_animation.append(exhaust_surf)

    def render_player(
        self,
        surface: pygame.Surface,
        player_x: int,
        player_y: int,
        view_bounds: Tuple[int, int, int, int],
        cell_size: int,
        moving: bool = False,
    ) -> None:
        """
        Render the player ship.

        Args:
            surface: Pygame surface to render on
            player_x: Player x position in grid coordinates
            player_y: Player y position in grid coordinates
            view_bounds: (min_x, min_y, max_x, max_y) of the visible area
            cell_size: Size of each cell in pixels
            moving: Whether the player is currently moving
        """
        min_x, min_y, max_x, max_y = view_bounds

        # Calculate screen position
        screen_x = (player_x - min_x) * cell_size
        screen_y = (player_y - min_y) * cell_size

        # Scale ship surface to match cell size
        ship_size = max(cell_size * 2, 8)  # Make ship at least 2 cells big
        scaled_ship = pygame.transform.scale(self.ship_surface, (ship_size, ship_size))

        # Animation logic
        if moving:
            self._render_player_exhaust_animation(
                ship_size, screen_y, surface, screen_x
            )
        # Draw ship
        surface.blit(scaled_ship, (screen_x, screen_y))

    def _render_player_exhaust_animation(self, ship_size, screen_y, surface, screen_x):
        self.animation_counter += self.animation_speed
        if self.animation_counter >= len(self.exhaust_animation):
            self.animation_counter = 0

        exhaust_surf = self.exhaust_animation[int(self.animation_counter)]

        # Scale exhaust to match ship
        scaled_exhaust = pygame.transform.scale(
            exhaust_surf, (ship_size, ship_size // 2)
        )

        # Draw exhaust beneath ship
        exhaust_y = screen_y + ship_size
        surface.blit(scaled_exhaust, (screen_x, exhaust_y))

    def render_fleet(
        self,
        surface: pygame.Surface,
        ship_positions: List[Tuple[int, int]],
        ship_health: List[int],
        view_bounds: Tuple[int, int, int, int],
        cell_size: int,
    ) -> None:
        """
        Render the player's fleet.

        Args:
            surface: Pygame surface to render on
            ship_positions: List of (x,y) positions for each ship
            ship_health: List of health values for each ship
            view_bounds: (min_x, min_y, max_x, max_y) of the visible area
            cell_size: Size of each cell in pixels
        """
        min_x, min_y, max_x, max_y = view_bounds

        # Skip main ship (already rendered by render_player)
        for i in range(1, len(ship_positions)):
            ship_x, ship_y = ship_positions[i]
            # Check if ship is in view
            if min_x <= ship_x <= max_x and min_y <= ship_y <= max_y:
                # Calculate screen position
                screen_x = (ship_x - min_x) * cell_size
                screen_y = (ship_y - min_y) * cell_size

                health = ship_health[i]

                # Scale ship based on health
                health_factor = max(0.5, health / 100.0)
                # Calculate ship size based on health (used for scaling)
                ship_scale = max(int(cell_size * 1.5 * health_factor), 6)

                # Draw smaller ships for fleet
                ship_color = (0, int(220 * health_factor), 0)

                # Use ship_scale for the ship size
                pygame.draw.polygon(
                    surface,
                    ship_color,
                    [
                        (screen_x + ship_scale // 2, screen_y),
                        (screen_x + ship_scale, screen_y + ship_scale),
                        (screen_x, screen_y + ship_scale),
                    ],
                )

                # Health indicator
                if health < 100:
                    # Draw health bar
                    bar_width = cell_size
                    bar_height = 3
                    pygame.draw.rect(
                        surface,
                        (50, 50, 50),
                        (screen_x, screen_y - bar_height - 1, bar_width, bar_height),
                    )
                    pygame.draw.rect(
                        surface,
                        (
                            (0, 255, 0)
                            if health > 50
                            else (255, 255, 0) if health > 25 else (255, 0, 0)
                        ),
                        (
                            screen_x,
                            screen_y - bar_height - 1,
                            int(bar_width * health / 100),
                            bar_height,
                        ),
                    )


class EffectsRenderer:
    """Handles rendering of special effects like explosions, particles, etc."""

    def __init__(self) -> None:
        """Initialize the effects renderer."""
        self.effects = []  # List of active effects
        self.particle_images = []  # Pre-rendered particle images

        # Initialize particle images
        self._init_particle_images()

    def _init_particle_images(self) -> None:
        """Initialize pre-rendered particle images."""
        # Create different sizes and colors
        sizes = [2, 3, 4, 6, 8]
        colors = [
            (255, 100, 0),  # Orange
            (255, 200, 0),  # Yellow
            (200, 200, 200),  # White
            (100, 100, 255),  # Blue
            (255, 100, 255),  # Pink
        ]

        for size, color in itertools.product(sizes, colors):
            # Create particle surface
            particle = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)

            # Draw circle
            pygame.draw.circle(particle, color, (size, size), size)

            # Add glow effect
            glow_color = (*color[:3], 100)  # Semi-transparent
            pygame.draw.circle(particle, glow_color, (size, size), size + 1)

            self.particle_images.append(particle)

    def add_effect(self, effect_type: str, x: int, y: int, **kwargs) -> None:
        """
        Add a new effect to be rendered.

        Args:
            effect_type: Type of effect ("explosion", "sparkle", etc.)
            x, y: Position in grid coordinates
            **kwargs: Additional parameters for the effect
        """
        if effect_type == "explosion":
            num_particles = kwargs.get("size", 20)

            particles = []
            for _ in range(num_particles):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 5)
                velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
                lifetime = random.uniform(20, 50)
                size = random.uniform(0.5, 2.0)
                color_idx = random.randint(0, len(self.particle_images) - 1)

                particles.append(
                    {
                        "pos": [x, y],
                        "velocity": velocity,
                        "lifetime": lifetime,
                        "max_lifetime": lifetime,
                        "color_idx": color_idx,
                        "size": size,
                    }
                )

            self.effects.append(
                {
                    "type": "explosion",
                    "particles": particles,
                    "duration": 60,  # Effect lasts for 60 frames
                    "x": x,
                    "y": y,
                }
            )

        elif effect_type == "sparkle":
            num_particles = kwargs.get("size", 10)

            particles = []
            for _ in range(num_particles):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.5, 2)
                velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
                lifetime = random.uniform(10, 30)
                size = random.uniform(0.3, 1.0)
                color_idx = random.randint(0, len(self.particle_images) - 1)

                particles.append(
                    {
                        "pos": [x, y],
                        "velocity": velocity,
                        "lifetime": lifetime,
                        "max_lifetime": lifetime,
                        "color_idx": color_idx,
                        "size": size,
                    }
                )

            self.effects.append(
                {
                    "type": "sparkle",
                    "particles": particles,
                    "duration": 30,  # Effect lasts for 30 frames
                    "x": x,
                    "y": y,
                }
            )

    def update(self) -> None:
        """Update all active effects."""
        # Update each effect
        for effect in self.effects[:]:
            # Update effect duration
            effect["duration"] -= 1

            # Remove expired effects
            if effect["duration"] <= 0:
                self.effects.remove(effect)
                continue

            # Update particles
            if "particles" in effect:
                for particle in effect["particles"][:]:
                    # Update position
                    particle["pos"][0] += particle["velocity"][0]
                    particle["pos"][1] += particle["velocity"][1]

                    # Apply gravity or drag if needed
                    if effect["type"] == "explosion":
                        # Slow down particles over time
                        particle["velocity"] = (
                            particle["velocity"][0] * 0.95,
                            particle["velocity"][1] * 0.95 + 0.1,  # Add gravity
                        )

                    # Update lifetime
                    particle["lifetime"] -= 1

                    # Remove expired particles
                    if particle["lifetime"] <= 0:
                        effect["particles"].remove(particle)

    def render(
        self,
        surface: pygame.Surface,
        view_bounds: Tuple[int, int, int, int],
        cell_size: int,
    ) -> None:
        """
        Render all active effects.

        Args:
            surface: Pygame surface to render on
            view_bounds: (min_x, min_y, max_x, max_y) of the visible area
            cell_size: Size of each cell in pixels
        """
        min_x, min_y, max_x, max_y = view_bounds

        for effect in self.effects:
            # Check if effect is in view
            effect_x, effect_y = effect["x"], effect["y"]
            if not (min_x <= effect_x <= max_x and min_y <= effect_y <= max_y):
                continue

            # Calculate screen position
            screen_effect_x = (effect_x - min_x) * cell_size
            screen_effect_y = (effect_y - min_y) * cell_size

            # Render particles
            if "particles" in effect:
                for particle in effect["particles"]:
                    # Get relative position
                    rel_x = particle["pos"][0] - effect_x
                    rel_y = particle["pos"][1] - effect_y

                    # Calculate screen position
                    screen_x = screen_effect_x + rel_x * cell_size
                    screen_y = screen_effect_y + rel_y * cell_size

                    # Get particle image based on color index
                    img = self.particle_images[
                        particle["color_idx"] % len(self.particle_images)
                    ]

                    # Scale based on size and lifetime
                    fade = particle["lifetime"] / particle["max_lifetime"]
                    size_factor = particle["size"] * fade * cell_size / 8

                    if size_factor < 0.2:
                        continue

                    # Scale particle
                    particle_size = max(2, int(img.get_width() * size_factor))
                    scaled_img = pygame.transform.scale(
                        img, (particle_size, particle_size)
                    )

                    # Apply fade with alpha
                    scaled_img.set_alpha(int(255 * fade))

                    # Draw particle
                    surface.blit(
                        scaled_img,
                        (
                            int(screen_x - particle_size / 2),
                            int(screen_y - particle_size / 2),
                        ),
                    )


class UIStateRenderer:
    """Handles rendering of UI state overlays and transitions."""

    def __init__(self) -> None:
        """Initialize the UI state renderer."""
        self.transitions = {}  # Active transitions
        self.notifications = []  # Visual notifications
        self.fade_overlay = None  # Surface for screen fades
        self.fade_alpha = 0  # Current fade level

    def start_transition(self, transition_type: str, duration: int = 30) -> None:
        """
        Start a screen transition effect.

        Args:
            transition_type: Type of transition ("fade_in", "fade_out", etc.)
            duration: Duration in frames
        """
        self.transitions[transition_type] = {
            "progress": 0,
            "duration": duration,
            "active": True,
        }

    def update(self) -> None:
        """Update all active UI transitions and notifications."""
        # Update transitions
        for trans_type, trans in list(self.transitions.items()):
            if trans["active"]:
                trans["progress"] += 1

                # Update fade overlay alpha
                if trans_type == "fade_in":
                    self.fade_alpha = 255 - int(
                        255 * (trans["progress"] / trans["duration"])
                    )
                elif trans_type == "fade_out":
                    self.fade_alpha = int(255 * (trans["progress"] / trans["duration"]))

                # Check if transition is complete
                if trans["progress"] >= trans["duration"]:
                    trans["active"] = False

                    # Clean up completed transitions
                    if trans_type == "fade_in":
                        self.fade_alpha = 0
                    elif trans_type == "fade_out":
                        self.fade_alpha = 255

        # Update notifications
        for notification in self.notifications[:]:
            notification["lifetime"] -= 1

            if notification["lifetime"] <= 0:
                self.notifications.remove(notification)

    def add_notification(
        self,
        text: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        lifetime: int = 180,
        size: int = 20,
        position: str = "center",
    ) -> None:
        """
        Add a visual notification to the screen.

        Args:
            text: Notification text
            color: Text color
            lifetime: How long to display (in frames)
            size: Font size
            position: Where to display ("center", "top", "bottom")
        """
        self.notifications.append(
            {
                "text": text,
                "color": color,
                "lifetime": lifetime,
                "size": size,
                "position": position,
                "max_lifetime": lifetime,
            }
        )

    def render(self, surface: pygame.Surface) -> None:
        """
        Render all UI state effects.

        Args:
            surface: Pygame surface to render on
        """

        # Render fade overlay if needed
        if self.fade_alpha > 0:
            self._render_surface_handler(surface)
        # Render notifications
        for notification in self.notifications:
            # Calculate position
            text = notification["text"]
            color = notification["color"]
            size = notification["size"]
            position = notification["position"]
            lifetime_ratio = notification["lifetime"] / notification["max_lifetime"]

            # Fade in/out
            alpha = 255
            if lifetime_ratio > 0.8:  # Fade in
                alpha = int(255 * (1 - (lifetime_ratio - 0.8) * 5))
            elif lifetime_ratio < 0.2:  # Fade out
                alpha = int(255 * lifetime_ratio * 5)

            # Position
            if position == "bottom":
                x = surface.get_width() // 2
                y = surface.get_height() - 50
                align = "center"
            elif position == "center":
                x = surface.get_width() // 2
                y = surface.get_height() // 2
                align = "center"
            elif position == "top":
                x = surface.get_width() // 2
                y = 50
                align = "center"
            else:
                x, y = 50, 50
                align = "left"

            # Draw with shadow for visibility
            draw_text(
                surface,
                text,
                x,
                y,
                size,
                color,
                align,
                shadow=True,
                shadow_color=(0, 0, 0),
                alpha=alpha,
            )

            # Update lifetime
            notification["lifetime"] -= 1

            if notification["lifetime"] <= 0:
                self.notifications.remove(notification)

        # Render transitions
        for trans_type, trans in self.transitions.items():
            if trans["active"]:
                if trans_type == "fade_in":
                    self.fade_alpha = 255 - int(
                        255 * (trans["progress"] / trans["duration"])
                    )
                elif trans_type == "fade_out":
                    self.fade_alpha = int(255 * (trans["progress"] / trans["duration"]))
                if self.fade_overlay:
                    self.fade_overlay.fill((0, 0, 0, self.fade_alpha))
                    surface.blit(self.fade_overlay, (0, 0))

            if trans["progress"] >= trans["duration"]:
                trans["active"] = False

                # Clean up completed transitions
                if trans_type == "fade_in":
                    self.fade_alpha = 0
                elif trans_type == "fade_out":
                    self.fade_alpha = 255

        # Update transitions
        for trans_type, trans in self.transitions.items():
            if trans["active"]:
                trans["progress"] += 1

            if trans["progress"] >= trans["duration"]:
                trans["active"] = False

                # Clean up completed transitions
                if trans_type == "fade_in":
                    self.fade_alpha = 0
                elif trans_type == "fade_out":
                    self.fade_alpha = 255

        # Update notifications
        for notification in self.notifications[:]:
            notification["lifetime"] -= 1

            if notification["lifetime"] <= 0:
                self.notifications.remove(notification)

        # Render minimap
        if self.show_minimap:
            self.minimap.render(surface)

        # Render shop
        if self.show_shop:
            self.shop.render(surface)

        # Render cursor
        if self.cursor:
            self.cursor.render(surface)

        # Render crosshair
        if self.crosshair:
            self.crosshair.render(surface)

        # Render tooltip
        if self.tooltip:
            self.tooltip.render(surface)

        # Render notification
        if self.notification:
            self.notification.render(surface)

        # Render fade overlay if needed
        if self.fade_alpha > 0:
            self._render_surface_handler(surface)
        # Render notifications
        for notification in self.notifications:
            # Calculate position
            text = notification["text"]
            color = notification["color"]
            size = notification["size"]
            position = notification["position"]
            lifetime_ratio = notification["lifetime"] / notification["max_lifetime"]

            # Fade in/out
            alpha = 255
            if lifetime_ratio > 0.8:  # Fade in
                alpha = int(255 * (1 - (lifetime_ratio - 0.8) * 5))
            elif lifetime_ratio < 0.2:  # Fade out
                alpha = int(255 * lifetime_ratio * 5)

            # Position
            if position == "bottom":
                x = surface.get_width() // 2
                y = surface.get_height() - 50
                align = "center"
            elif position == "center":
                x = surface.get_width() // 2
                y = surface.get_height() // 2
                align = "center"
            elif position == "top":
                x = surface.get_width() // 2
                y = 50
                align = "center"
            else:
                x, y = 50, 50
                align = "left"

            # Draw with shadow for visibility
            draw_text(surface, text, x, y, size, color)

            # Draw with shadow for visibility
            draw_text(surface, text, x + 2, y + 2, size, (0, 0, 0), 1)

            # Draw with transparency
            draw_text(surface, text, x, y, size, color, alpha)

            # Draw with transparency and shadow for visibility
            draw_text(surface, text, x + 2, y + 2, size, (0, 0, 0), 1, alpha)

            # Update lifetime
            notification["lifetime"] = max(0, notification["lifetime"] - 1)

        # Render minimap
        draw_minimap(surface, self.minimap, self.player)

        # Render debug info
        draw_text(
            surface, f"FPS: {int(self.clock.get_fps())}", 10, 10, 16, (255, 255, 255)
        )
        draw_text(
            surface,
            f"Player position: {self.player.position}",
            10,
            30,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player velocity: {self.player.velocity}",
            10,
            50,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface, f"Player angle: {self.player.angle}", 10, 70, 16, (255, 255, 255)
        )
        draw_text(
            surface,
            f"Player rotation: {self.player.rotation}",
            10,
            90,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player mining range: {self.player.mining_range}",
            10,
            110,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player mining efficiency: {self.player.mining_efficiency}",
            10,
            130,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player mining speed: {self.player.mining_speed}",
            10,
            150,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player mining cooldown: {self.player.mining_cooldown}",
            10,
            170,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player inventory: {self.player.inventory}",
            10,
            190,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player currency: {self.player.currency}",
            10,
            210,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player health: {self.player.health}",
            10,
            230,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player energy: {self.player.energy}",
            10,
            250,
            16,
            (255, 255, 255),
        )
        draw_text(
            surface,
            f"Player shield: {self.player.shield}",
            10,
            270,
            16,
            (255, 255, 255),
        )

    def _render_surface_handler(self, surface):
        if not self.fade_overlay or self.fade_overlay.get_size() != surface.get_size():
            self.fade_overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        self.fade_overlay.fill((0, 0, 0, self.fade_alpha))
        surface.blit(self.fade_overlay, (0, 0))
