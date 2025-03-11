import contextlib
import time
import random
import curses
import math
import logging
import pygame
# Removed unused imports: from enum import Enum, auto
from typing import List, Dict, Optional, Any, Tuple
from ui.ui_element.ui_style import UIStyle
from ui.ui_element.animation_style import AnimationStyle
from ui.ui_helpers.render_helper import RenderHelper


# Define standard colors for UI components
COLOR_TEXT = (220, 220, 220)  # Standard text color
COLOR_BG = (20, 20, 30)  # Standard background color
COLOR_HIGHLIGHT = (180, 180, 255)  # Standard highlight color


class UIElement:
    """Base class for all UI elements with cellular automaton-inspired behaviors

    This class provides a standardized foundation for all UI components with consistent
    animation handling, style application, and error management. All UI components should
    inherit from this class to ensure a unified approach to UI rendering and behavior.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        style: UIStyle,
        title: Optional[str] = None,
    ):
        """Initialize a UI element.

        Args:
            x: X coordinate of the top-left corner
            y: Y coordinate of the top-left corner
            width: Width of the element in characters
            height: Height of the element in characters
            style: Visual style for the element
            title: Optional title for the element
        """
        self.x = x
        self.y = y
        self.width = max(1, width)  # Ensure positive dimensions
        self.height = max(1, height)
        self.style = style
        self.title = title
        self.visible = True
        self.enabled = True
        self.hover = False
        self.border_chars = UIStyle.get_border_chars(style)
        self.animation_history: List[Any] = []  # Tracks cellular automaton states
        self.animation_style = AnimationStyle.get_animation_for_style(style)

        # Standardized animation state structure for all UI components
        self.animation: Dict[str, Any] = {
            "active": False,  # Whether animation is currently active
            "progress": 0.0,  # Progress from 0.0 to 1.0
            "duration": 0.5,  # Animation duration in seconds
            "start_time": 0,  # Start time of the animation
            "cells": [
                [False for _ in range(self.width)] for _ in range(self.height)
            ],  # Cell grid for cellular animations
            "phase": 0,  # Current animation phase/step
            "style_data": {},  # Style-specific animation data
            "target_progress": 1.0,  # Target progress value (for progress bars)
            "easing": "linear",  # Easing function for animations
        }

    def start_animation(self, duration: float = 0.5) -> None:
        """Start an animation sequence.

        Args:
            duration: Animation duration in seconds
        """
        try:
            self.animation["active"] = True
            self.animation["progress"] = 0.0
            self.animation["duration"] = max(0.1, duration)  # Ensure positive duration
            self.animation["start_time"] = time.time()
            self.animation["phase"] = 0

            # Initialize animation pattern based on style
            self._initialize_animation_pattern()

        except Exception as e:
            logging.error(f"Error starting animation: {e}")
            # Reset to safe state
            self.animation["active"] = False

    def _initialize_animation_pattern(self) -> None:
        """Initialize the animation pattern based on the UI style.

        Each style has its own unique animation pattern initialization.
        This method centralizes the pattern creation logic for consistency.
        """
        try:
            # Create a standardized pattern structure for all styles
            self.animation["pattern"] = []
            self.animation["phase"] = 0.0
            
            # Initialize pattern based on style
            style_initializers = {
                UIStyle.QUANTUM: self._init_quantum_pattern,
                UIStyle.SYMBIOTIC: self._init_symbiotic_pattern,
                UIStyle.ASTEROID: self._init_asteroid_pattern,
                UIStyle.MECHANICAL: self._init_mechanical_pattern,
                UIStyle.FLEET: self._init_fleet_pattern
            }
            
            # Call the appropriate initializer or default
            initializer = style_initializers.get(self.style, self._init_default_pattern)
            initializer()
            
        except Exception as e:
            logging.error(f"Error initializing animation pattern: {e}")
            # Fallback to simple random pattern
            self._init_fallback_pattern()
    
    def _init_quantum_pattern(self) -> None:
        """Initialize quantum wave interference pattern with particles."""
        num_particles = max(3, int(self.width * self.height * 0.05))
        self.animation["pattern"] = [
            {
                "pos": (random.uniform(0, self.width), random.uniform(0, self.height)),
                "angle": random.uniform(0, 2 * math.pi),
                "speed": random.uniform(0.1, 0.3),
                "char": random.choice([".", "*", "o", "O"]),
            }
            for _ in range(num_particles)
        ]
        # Add wave properties
        self.animation["wave_frequency"] = 0.5
    
    def _init_symbiotic_pattern(self) -> None:
        """Initialize organic cellular automaton pattern."""
        num_cells = max(2, int(self.width * self.height * 0.03))
        self.animation["pattern"] = [
            {
                "pos": (random.uniform(1, self.width - 2), random.uniform(1, self.height - 2)),
                "size": 0.0,  # Will grow during animation
                "max_size": random.uniform(1.0, 2.5),
                "growth_rate": random.uniform(0.5, 1.5),
                "char": random.choice(["o", "O", "*", "#"]),
            }
            for _ in range(num_cells)
        ]
        self.animation["growth_rate"] = 0.2
        self.animation["mutation_chance"] = 0.05
    
    def _init_asteroid_pattern(self) -> None:
        """Initialize asteroid/mineral growth pattern."""
        # Create fractal-like growth points
        start_points = [
            (random.randint(1, self.width - 2), random.randint(1, self.height - 2))
            for _ in range(max(1, min(3, self.width // 5)))
        ]

        growth_points = []
        for x, y in start_points:
            # Add seed point
            growth_points.append({
                "pos": (x, y),
                "char": "#",
                "growth_stage": 1.0,
                "is_seed": True,
            })
            
            # Add branching points
            self._add_branching_points(growth_points, x, y)

        self.animation["pattern"] = growth_points
    
    def _add_branching_points(self, growth_points: List[Dict], x: int, y: int) -> None:
        """Add branching points to the asteroid pattern.
        
        Args:
            growth_points: List to add the branching points to
            x: X coordinate of the seed point
            y: Y coordinate of the seed point
        """
        for _ in range(random.randint(3, 8)):
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            new_x = max(0, min(self.width - 1, x + dx))
            new_y = max(0, min(self.height - 1, y + dy))
            growth_points.append({
                "pos": (new_x, new_y),
                "char": random.choice(["*", ".", "+"]),
                "growth_stage": 0.0,
                "is_seed": False,
            })
    
    def _init_mechanical_pattern(self) -> None:
        """Initialize mechanical grid-like pattern."""
        # Create a grid pattern where either x or y is even
        grid_points = [
            (x, y)
            for x in range(1, self.width - 1)
            for y in range(1, self.height - 1)
            if x % 2 == 0 or y % 2 == 0
        ]

        # Randomize the order for sequential activation
        random.shuffle(grid_points)

        self.animation["pattern"] = [
            {
                "pos": pos,
                "activation_time": i / len(grid_points),  # Staggered activation
                "char": random.choice([".", ":", "+", "=", "-", "|", "#"]),
                "active": False,
            }
            for i, pos in enumerate(grid_points)
        ]
    
    def _init_fleet_pattern(self) -> None:
        """Initialize fleet style - ordered, military pattern."""
        pattern = []

        # Create horizontal scan lines
        for y in range(1, self.height - 1):
            if y % 2 == 0:
                self._add_scan_line(pattern, y, left_to_right=True)
            else:
                self._add_scan_line(pattern, y, left_to_right=False)

        self.animation["pattern"] = pattern
    
    def _add_scan_line(self, pattern: List[Dict], y: int, left_to_right: bool = True) -> None:
        """Add a horizontal scan line to the fleet pattern.
        
        Args:
            pattern: List to add the scan line to
            y: Y coordinate of the scan line
            left_to_right: Direction of the scan line
        """
        if left_to_right:
            for x in range(1, self.width - 1):
                delay = x / self.width
                pattern.append({
                    "pos": (x, y),
                    "activation_time": delay,
                    "char": "-",
                    "active": False,
                })
        else:
            for x in range(self.width - 2, 0, -1):
                delay = (self.width - x) / self.width
                pattern.append({
                    "pos": (x, y),
                    "activation_time": delay,
                    "char": "-",
                    "active": False,
                })
    
    def _init_default_pattern(self) -> None:
        """Initialize default cellular pattern."""
        # Simple cellular automaton pattern with 20% chance at each position
        cells = [
            {
                "pos": (x, y),
                "alive": True,
                "char": "*",
                "age": 0,
                "energy": random.uniform(0.5, 1.0),  # Initial energy level
            }
            for y in range(self.height)
            for x in range(self.width)
            if random.random() < 0.2  # 20% chance of initial cell
        ]

        self.animation["pattern"] = cells
        self.animation["style_data"] = {}
    
    def _init_fallback_pattern(self) -> None:
        """Initialize fallback pattern when an error occurs."""
        self.animation["cells"] = [
            [random.random() < 0.5 for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def update_animation(self, dt: Optional[float] = None) -> None:
        """Update animation state based on elapsed time

        Args:
            dt: Optional time delta in seconds. If None, will calculate based on current time.
        """
        if not self.animation["active"]:
            return

        try:
            current_time = time.time()

            elapsed = current_time - self.animation["start_time"] if dt is None else dt
            # Avoid division by zero
            duration = max(0.001, self.animation["duration"])

            # Apply easing function based on animation settings
            progress = min(1.0, elapsed / duration)
            self.animation["progress"] = self._apply_easing(
                progress, self.animation.get("easing", "linear")
            )

            if self.animation["progress"] >= 1.0:
                self.animation["active"] = False
                return

            # Update animation pattern based on style with proper timing
            self._update_animation_pattern(elapsed)

        except Exception as e:
            logging.error(f"Error updating animation: {e}")
            # Reset to safe state
            self.animation["active"] = False

    def _apply_easing(self, progress: float, easing_type: str) -> float:
        """Apply easing function to the progress value.

        Args:
            progress: Raw progress value from 0.0 to 1.0
            easing_type: Type of easing function to apply

        Returns:
            Eased progress value
        """
        try:
            if easing_type == "bounce":
                return (
                    4 * progress * progress
                    if progress < 0.5
                    else 1 - math.pow(-2 * progress + 2, 2) / 2
                )
            elif easing_type == "ease_in":
                return progress**2
            elif easing_type == "ease_in_out":
                return 0.5 - 0.5 * math.cos(math.pi * progress)
            elif easing_type == "ease_out":
                return 1 - (1 - progress) * (1 - progress)
            elif easing_type == "elastic":
                c = (2 * math.pi) / 3
                return (
                    progress
                    if progress in {0, 1}
                    else math.pow(2, -10 * progress)
                    * math.sin((progress * 10 - 0.75) * c)
                    + 1
                )
            else:
                return progress  # Default to linear if unknown
        except Exception as e:
            logging.error(f"Error applying easing: {e}")
            return progress  # Fallback to linear

    def _update_animation_pattern(self, elapsed: float) -> None:
        """Update the animation pattern based on the UI style.

        Args:
            elapsed: Time elapsed since animation start
        """
        try:
            # Only update at appropriate intervals based on style
            phase_elapsed = elapsed - self.animation["phase"]

            if self.style == UIStyle.QUANTUM and phase_elapsed > 0.05:
                self._update_quantum_pattern(elapsed)
                self.animation["phase"] = elapsed

            elif self.style == UIStyle.SYMBIOTIC and phase_elapsed > 0.1:
                self._update_cellular_pattern()
                self.animation["phase"] = elapsed

            elif self.style == UIStyle.ASTEROID and phase_elapsed > 0.15:
                self._update_mineral_growth()
                self.animation["phase"] = elapsed

            elif self.style == UIStyle.MECHANICAL and phase_elapsed > 0.12:
                self._update_mechanical_pattern(elapsed)
                self.animation["phase"] = elapsed

            elif self.style == UIStyle.FLEET and phase_elapsed > 0.08:
                self._update_fleet_pattern(elapsed)
                self.animation["phase"] = elapsed

        except Exception as e:
            logging.error(f"Error updating animation pattern: {e}")
            # Don't reset animation state here to allow it to continue

    def _update_quantum_pattern(self, elapsed: float) -> None:
        """Update quantum flux animation pattern with wave interference effects

        Creates a dynamic pattern that resembles quantum wave interference,
        with adjustments based on animation progress.

        Args:
            elapsed: Time elapsed since animation start
        """
        try:
            freq = 4.0  # Base frequency
            for y in range(self.height):
                for x in range(self.width):
                    phase = elapsed * freq
                    value = math.sin(x / 2 + y / 2 + phase)
                    self.animation_state["cells"][y][x] = value > 0
        except Exception as e:
            logging.error(f"Error updating quantum pattern: {e}")

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the UI element with animation effects.

        Args:
            surface: Pygame surface to draw on
            font: Font to use for text

        Returns:
            pygame.Rect: The area that was drawn
        """
        try:
            if not self.visible:
                return pygame.Rect(self.x, self.y, self.width, self.height)

            # Create element surface
            char_width, char_height = font.size("X")
            element_width = self.width * char_width
            element_height = self.height * char_height
            element_surf = pygame.Surface(
                (element_width, element_height), pygame.SRCALPHA
            )

            # Get style-specific colors
            base_color = RenderHelper.get_style_color(self.style, COLOR_TEXT)

            # Apply animation effects to colors if active
            if self.animation["active"]:
                self.update_animation()
                base_color = RenderHelper.apply_animation_effect(
                    base_color,
                    self.style,
                    self.animation["progress"],
                    self.animation["phase"],
                )
                self._draw_animation_effects(element_surf, font, base_color)

            # Draw border with style-specific characters
            for i in range(self.width):
                if i == 0:
                    RenderHelper.draw_char(
                        element_surf, font, 0, 0, self.border_chars["tl"], base_color
                    )
                    RenderHelper.draw_char(
                        element_surf,
                        font,
                        0,
                        self.height - 1,
                        self.border_chars["bl"],
                        base_color,
                    )
                elif i == self.width - 1:
                    RenderHelper.draw_char(
                        element_surf, font, i, 0, self.border_chars["tr"], base_color
                    )
                    RenderHelper.draw_char(
                        element_surf,
                        font,
                        i,
                        self.height - 1,
                        self.border_chars["br"],
                        base_color,
                    )
                else:
                    RenderHelper.draw_char(
                        element_surf, font, i, 0, self.border_chars["h"], base_color
                    )
                    RenderHelper.draw_char(
                        element_surf,
                        font,
                        i,
                        self.height - 1,
                        self.border_chars["h"],
                        base_color,
                    )

            for i in range(1, self.height - 1):
                RenderHelper.draw_char(
                    element_surf, font, 0, i, self.border_chars["v"], base_color
                )
                RenderHelper.draw_char(
                    element_surf,
                    font,
                    self.width - 1,
                    i,
                    self.border_chars["v"],
                    base_color,
                )

            # Draw title if provided
            if self.title:
                title_x = max(1, (self.width - len(self.title)) // 2)
                RenderHelper.draw_text(
                    element_surf, font, title_x, 0, self.title, base_color
                )

            return surface.blit(element_surf, (self.x, self.y))
        except Exception as e:
            logging.error(f"Error drawing UI element: {e}")
            # Return a safe default rect
            return pygame.Rect(self.x, self.y, self.width, self.height)

    def _draw_char(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        char: str,
        color: Tuple[int, int, int] = COLOR_TEXT,
    ) -> None:
        """Draw a single character at the specified position.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X position in character coordinates
            y: Y position in character coordinates
            char: Character to draw
            color: RGB color tuple for the character
        """
        # Use the standardized RenderHelper method
        RenderHelper.draw_char(surface, font, x, y, char, color, char_coords=True)

    def _init_style_data(self) -> Dict[str, Any]:
        """Initialize style-specific animation data.

        Returns:
            Dictionary containing style-specific animation parameters
        """
        try:
            width = max(1, self.width - 1)  # Avoid index errors
            height = max(1, self.height - 1)

            if self.style == UIStyle.QUANTUM:
                return {
                    "particles": [
                        (
                            random.randint(0, width),
                            random.randint(0, height),
                            random.random() * 2 * math.pi,
                        )
                        for _ in range(5)
                    ],
                    "wave_phase": 0.0,
                }
            elif self.style == UIStyle.SYMBIOTIC:
                return {
                    "growth_points": [
                        (
                            random.randint(0, width),
                            random.randint(0, height),
                        )
                        for _ in range(3)
                    ],
                    "tendrils": [],
                }
            elif self.style == UIStyle.MECHANICAL:
                return {
                    "gears": [
                        (
                            random.randint(0, width),
                            random.randint(0, height),
                            random.choice([4, 6, 8]),
                        )
                        for _ in range(3)
                    ],
                    "rotation": 0.0,
                }
            elif self.style == UIStyle.ASTEROID:
                return {
                    "asteroids": [
                        (
                            random.randint(0, width),
                            random.randint(0, height),
                            random.random() * 2 * math.pi,
                            random.random() * 0.5 + 0.5,
                        )
                        for _ in range(5)
                    ]
                }
            elif self.style == UIStyle.FLEET:
                return {
                    "ships": [
                        (
                            random.randint(0, width),
                            random.randint(0, height),
                            random.choice(["<", ">", "^", "v"]),
                        )
                        for _ in range(4)
                    ],
                    "formation_phase": 0.0,
                }
            return {}
        except Exception as e:
            logging.error(f"Error initializing style data: {e}")
            return {}  # Return empty dict on error

    def _draw_animation_effects(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        color: Tuple[int, int, int] = COLOR_TEXT,
    ) -> None:
        """Draw current animation frame effects.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            color: Base color for animation effects (already processed for style)
        """
        if not self.animation["active"]:
            return

        try:
            # Get animation pattern based on style
            pattern = self.animation["pattern"]
            progress = self.animation["progress"]

            # Draw animation pattern based on style
            if self.style == UIStyle.QUANTUM:
                self._draw_quantum_animation(surface, font, color, pattern, progress)
            elif self.style == UIStyle.SYMBIOTIC:
                self._draw_symbiotic_animation(surface, font, color, pattern, progress)
            elif self.style == UIStyle.MECHANICAL:
                self._draw_mechanical_animation(surface, font, color, pattern, progress)
            elif self.style == UIStyle.ASTEROID:
                self._draw_asteroid_animation(surface, font, color, pattern, progress)
            elif self.style == UIStyle.FLEET:
                self._draw_fleet_animation(surface, font, color, pattern, progress)
            else:
                self._draw_cellular_animation(surface, font, color, pattern, progress)
        except Exception as e:
            logging.error(f"Error drawing animation effects: {e}")

    def _evolve_cellular_grid(self) -> None:
        """Evolve the cellular automaton grid using Game of Life rules"""
        try:
            current_grid = self.animation_state["cells"]
            new_grid = [[False for _ in range(self.width)] for _ in range(self.height)]

            # Apply Conway's Game of Life rules
            for y in range(self.height):
                for x in range(self.width):
                    # Count live neighbors (Moore neighborhood)
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = (x + dx) % self.width, (y + dy) % self.height
                            if current_grid[ny][nx]:
                                neighbors += 1

                    # Apply Game of Life rules
                    new_grid[y][x] = (
                        neighbors in [2, 3] if current_grid[y][x] else neighbors == 3
                    )
                    # Apply random mutations (5% chance)
                    if random.random() < 0.05:
                        new_grid[y][x] = not new_grid[y][x]

            self.animation_state["cells"] = new_grid

            # Store history for analysis if needed
            if len(self.animation_history) > 10:
                self.animation_history.pop(0)
            self.animation_history.append(new_grid)
        except Exception as e:
            logging.error(f"Error evolving cellular grid: {e}")

    def _update_mineral_growth(self) -> None:
        """Update mineral growth animation pattern"""
        try:
            current_grid = self.animation_state["cells"]
            new_grid = [row[:] for row in current_grid]  # Copy current grid

            # Growth directions (4-connected)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            # For each cell that's already a crystal
            for y in range(self.height):
                for x in range(self.width):
                    if not current_grid[y][x]:
                        continue

                    # Try to grow in each direction
                    for dx, dy in directions:
                        nx, ny = (x + dx) % self.width, (y + dy) % self.height

                        # If the target cell is empty, it might grow
                        if not current_grid[ny][nx]:
                            # Higher chance to grow in the same direction as neighbors
                            neighbor_bias = 0
                            for ndx, ndy in directions:
                                nnx, nny = (nx + ndx) % self.width, (
                                    ny + ndy
                                ) % self.height
                                if current_grid[nny][nnx]:
                                    neighbor_bias += 0.1

                            # Apply growth probability
                            growth_chance = min(0.95, 0.15 + neighbor_bias)
                            if random.random() < growth_chance:
                                new_grid[ny][nx] = True

            self.animation_state["cells"] = new_grid
        except Exception as e:
            logging.error(f"Error updating mineral growth: {e}")

    def _draw_asteroid_animation(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = COLOR_TEXT,
        pattern: List[Dict[str, Any]] = None,
        progress: float = 0.0,
    ) -> None:
        """Draw asteroid field animation frame.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            base_color: Base color for animation effects
            pattern: Animation pattern data
            progress: Animation progress from 0.0 to 1.0
        """
        if pattern is None:
            return

        # Get phase from animation data and update it
        phase = self.animation.get("phase", 0.0)
        self.animation["phase"] = phase + 0.1  # Update phase for next frame

        # Draw and update each growth point
        for point in pattern:
            x, y = point.get("pos", (0, 0))
            is_seed = point.get("is_seed", False)
            growth_stage = point.get("growth_stage", 0.0)
            display_char = point.get("char", "#")

            # Update growth stage based on progress and whether it's a seed
            new_growth = (
                1.0 if is_seed else min(growth_stage + 0.1 * progress, progress)
            )
            point["growth_stage"] = new_growth

            # Only draw if the growth stage is positive
            if new_growth > 0:
                # Calculate color with mineral/asteroid tint
                r, g, b = base_color
                # Adjust color based on growth stage and seed status
                if is_seed:
                    # Seeds are more vibrant
                    color = (int(r * 0.9), int(g * 0.8), int(b * 0.7))
                else:
                    # Non-seeds have color based on growth stage
                    color = (
                        int(r * 0.7 * new_growth),
                        int(g * 0.6 * new_growth),
                        int(b * 0.5 * new_growth),
                    )

                # Draw the growth point
                RenderHelper.draw_char(
                    surface, font, int(x), int(y), display_char, color
                )

    def _draw_quantum_animation(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = COLOR_TEXT,
        pattern: List[Dict[str, Any]] = None,
        progress: float = 0.0,
    ) -> None:
        """Draw quantum flux animation frame.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            base_color: Base color for animation effects
            pattern: Animation pattern data
            progress: Animation progress from 0.0 to 1.0
        """
        if pattern is None:
            return

        # Get wave phase from animation data
        wave_phase = self.animation.get("phase", 0.0)

        # Draw quantum particles from pattern
        for particle in pattern:
            x, y = particle.get("pos", (0, 0))

            # Apply wave motion to particle position
            new_x = x + math.cos(wave_phase) * 0.2 * progress
            new_y = y + math.sin(wave_phase) * 0.2 * progress

            # Wrap around edges
            new_x = (new_x + self.width) % self.width
            new_y = (new_y + self.height) % self.height

            # Select character based on progress
            chars = [".", "*", "o", "O", "0", "@", "#", "%", "&"]
            idx = min(int(progress * len(chars)), len(chars) - 1)
            char = chars[idx]

            # Calculate color with quantum interference pattern
            r, g, b = base_color
            wave_factor = 0.7 + 0.3 * math.sin(wave_phase * 2)
            color = (int(r * wave_factor), int(g * wave_factor), int(b))

            # Draw the particle
            RenderHelper.draw_char(surface, font, int(new_x), int(new_y), char, color)

    def animate(self, stdscr, style: AnimationStyle = AnimationStyle.CELLULAR):
        """Animate the appearance of the element"""
        if style == AnimationStyle.CELLULAR:
            self._cellular_animation(stdscr)
        elif style == AnimationStyle.FRACTAL:
            self._fractal_animation(stdscr)
        elif style == AnimationStyle.WARP:
            self._warp_animation(stdscr)
        elif style == AnimationStyle.QUANTUM_FLUX:
            self._quantum_animation(stdscr)
        elif style == AnimationStyle.MINERAL_GROWTH:
            self._mineral_growth_animation(stdscr)

    def _draw_fleet_animation(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = COLOR_TEXT,
        pattern: List[Dict[str, Any]] = None,
        progress: float = 0.0,
    ) -> None:
        """Draw fleet movement animation frame.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            base_color: Base color for animation effects
            pattern: Animation pattern data
            progress: Animation progress from 0.0 to 1.0
        """
        if pattern is None:
            return

        # Get phase from animation data
        phase = self.animation.get("phase", 0.0)
        self.animation["phase"] = phase + 0.1  # Update phase for next frame

        # Draw each fleet component with ordered activation
        for component in pattern:
            x, y = component.get("pos", (0, 0))
            activation_time = component.get("activation_time", 0.0)
            display_char = component.get("char", "-")

            # Only draw components that should be active at current progress
            if progress >= activation_time:
                # Calculate color with fleet blue tint
                r, g, b = base_color
                # Fleet components have a blue-ish tint
                blue_factor = 1.2
                color = (int(r * 0.8), int(g * 0.9), int(b * blue_factor))

                # Mark component as active
                component["active"] = True

                # Draw the component
                RenderHelper.draw_char(
                    surface, font, int(x), int(y), display_char, color
                )

    def _draw_symbiotic_animation(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = COLOR_TEXT,
        pattern: List[Dict[str, Any]] = None,
        progress: float = 0.0,
    ) -> None:
        """Draw symbiotic growth animation frame.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            base_color: Base color for animation effects
            pattern: Animation pattern data
            progress: Animation progress from 0.0 to 1.0
        """
        if pattern is None:
            return

        # Get growth rate from animation data
        growth_rate = self.animation.get("growth_rate", 0.2)

        # Draw and update each growth cell
        for cell in pattern:
            x, y = cell.get("pos", (0, 0))
            size = cell.get("size", 0.0)
            max_size = cell.get("max_size", 2.0)
            # We don't use the original char from pattern, we select based on size

            # Grow the cell based on progress
            new_size = min(size + growth_rate * progress, max_size * progress)
            cell["size"] = new_size

            # Calculate color with organic green tint
            r, g, b = base_color
            green_factor = 1.5  # Increase green component for organic look
            color = (int(r * 0.7), int(g * green_factor), int(b * 0.7))

            # Draw the cell with size-based character
            size_chars = [".", ",", "o", "O", "@", "#"]
            size_idx = min(int(new_size * len(size_chars)), len(size_chars) - 1)
            display_char = size_chars[size_idx]

            # Draw the cell
            RenderHelper.draw_char(surface, font, int(x), int(y), display_char, color)

    def _draw_mechanical_animation(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = COLOR_TEXT,
        pattern: List[Dict[str, Any]] = None,
        progress: float = 0.0,
    ) -> None:
        """Draw mechanical gear animation frame.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            base_color: Base color for animation effects
            pattern: Animation pattern data
            progress: Animation progress from 0.0 to 1.0
        """
        if pattern is None:
            return

        # Get phase from animation data
        phase = self.animation.get("phase", 0.0)
        self.animation["phase"] = phase + 0.1  # Update phase for next frame

        # Draw each mechanical component
        for component in pattern:
            x, y = component.get("pos", (0, 0))
            activation_time = component.get("activation_time", 0.0)
            char = component.get("char", "+")

            # Only draw components that should be active at current progress
            if progress >= activation_time:
                # Calculate color with metallic tint
                r, g, b = base_color
                metal_factor = 0.8 + 0.2 * math.sin(phase * 2)  # Subtle pulsing
                color = (
                    int(r * metal_factor),
                    int(g * metal_factor),
                    int(b * metal_factor),
                )

                # Mark component as active
                component["active"] = True

                # Draw the component
                RenderHelper.draw_char(surface, font, int(x), int(y), char, color)

    def _draw_cellular_animation(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = COLOR_TEXT,
        pattern: List[Dict[str, Any]] = None,
        progress: float = 0.0,
    ) -> None:
        """Draw cellular automaton animation frame.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            base_color: Base color for animation effects
            pattern: Animation pattern data
            progress: Animation progress from 0.0 to 1.0
        """
        if pattern is None:
            return

        # Get phase from animation data
        phase = self.animation.get("phase", 0.0)
        self.animation["phase"] = phase + 0.1  # Update phase for next frame

        # Draw each cell in the cellular automaton
        for cell in pattern:
            x, y = cell.get("pos", (0, 0))
            alive = cell.get("alive", False)
            age = cell.get("age", 0)
            energy = cell.get("energy", 1.0)

            # Only draw living cells
            if alive:
                # Calculate color based on cell age and energy
                r, g, b = base_color

                # Newer cells are brighter, higher energy cells are more saturated
                age_factor = min(1.0, age / 10.0)  # Normalize age effect
                energy_factor = min(1.0, energy)

                # Adjust color based on age and energy
                color = (
                    int(r * (1.0 - age_factor * 0.5) * energy_factor),
                    int(g * (1.0 - age_factor * 0.3) * energy_factor),
                    int(b * (1.0 - age_factor * 0.1) * energy_factor),
                )

                # Select character based on energy level
                chars = ["·", ":", "o", "O", "@", "#"]
                idx = min(int(energy * (len(chars) - 1)), len(chars) - 1)
                display_char = chars[idx]

                # Draw the cell
                RenderHelper.draw_char(
                    surface, font, int(x), int(y), display_char, color
                )

                # Age the cell for next frame
                cell["age"] = age + 1

                # Reduce energy over time
                cell["energy"] = max(0.0, energy - 0.05)

                # If energy depleted, mark cell as dead
                if cell["energy"] <= 0:
                    cell["alive"] = False

    def _fractal_animation(self, stdscr):
        """Recursive revealing animation"""

        def draw_segment(start_x, start_y, segment_width, segment_height):
            if segment_width <= 1 or segment_height <= 1:
                return

            # Draw the outline of this segment
            for i in range(segment_width):
                with contextlib.suppress(curses.error):
                    stdscr.addstr(start_y, start_x + i, self.border_chars["h"])
                    stdscr.addstr(
                        start_y + segment_height - 1,
                        start_x + i,
                        self.border_chars["h"],
                    )
            for i in range(segment_height):
                with contextlib.suppress(curses.error):
                    stdscr.addstr(start_y + i, start_x, self.border_chars["v"])
                    stdscr.addstr(
                        start_y + i, start_x + segment_width - 1, self.border_chars["v"]
                    )
            stdscr.refresh()
            time.sleep(0.05)

            # Split into quadrants and recurse
            mid_x = segment_width // 2
            mid_y = segment_height // 2

            draw_segment(start_x, start_y, mid_x, mid_y)
            draw_segment(start_x + mid_x, start_y, segment_width - mid_x, mid_y)
            draw_segment(start_x, start_y + mid_y, mid_x, segment_height - mid_y)
            draw_segment(
                start_x + mid_x,
                start_y + mid_y,
                segment_width - mid_x,
                segment_height - mid_y,
            )

        draw_segment(self.x, self.y, self.width, self.height)
        # Final reveal
        self.draw(stdscr)

    def _warp_animation(self, stdscr):
        """Space-warp inspired expanding animation"""
        # Start from center and expand outward
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2

        max_radius = max(self.width, self.height)

        for radius in range(1, max_radius, 2):
            # Clear previous
            for y in range(
                max(0, self.y), min(self.y + self.height, stdscr.getmaxyx()[0])
            ):
                for x in range(
                    max(0, self.x), min(self.x + self.width, stdscr.getmaxyx()[1])
                ):
                    with contextlib.suppress(curses.error):
                        stdscr.addstr(y, x, " ")
            # Draw expanding circles
            for angle in range(0, 360, 10):
                angle_rad = angle * 3.14159 / 180
                x = int(center_x + radius * 0.5 * math.cos(angle_rad))
                y = int(center_y + radius * 0.25 * math.sin(angle_rad))

                if 0 <= x < stdscr.getmaxyx()[1] and 0 <= y < stdscr.getmaxyx()[0]:
                    with contextlib.suppress(curses.error):
                        stdscr.addstr(y, x, "·")
            stdscr.refresh()
            time.sleep(0.05)

        # Final reveal
        self.draw(stdscr)

    def _quantum_animation(self, stdscr):
        """Quantum-inspired probability wave collapse animation"""
        # Create a grid of probabilities
        probs = [
            [random.random() for _ in range(self.width)] for _ in range(self.height)
        ]

        # Animation phases
        for _ in range(5):
            # Update probabilities based on neighbors (quantum-like interference)
            new_probs = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
            for y in range(self.height):
                for x in range(self.width):
                    # Average with neighbors
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                neighbors.append(probs[ny][nx])

                    # Calculate new probability with quantum-like randomness
                    avg = sum(neighbors) / len(neighbors) if neighbors else 0.5
                    new_probs[y][x] = (avg + probs[y][x]) / 2 + (
                        random.random() - 0.5
                    ) * 0.1
                    new_probs[y][x] = max(0, min(1, new_probs[y][x]))  # Clamp to [0,1]

            # Update grid
            probs = new_probs

            # Visualize the current state - higher prob = more visible
            for y in range(self.height):
                for x in range(self.width):
                    if random.random() < probs[y][x]:  # Probabilistic rendering
                        with contextlib.suppress(curses.error):
                            char = " ·:+*#@"[min(6, int(probs[y][x] * 7))]
                            stdscr.addstr(self.y + y, self.x + x, char)
            stdscr.refresh()
            time.sleep(0.1)

        # Final "collapse" to actual UI
        self.draw(stdscr)

    def _mineral_growth_animation(self, stdscr):
        """Crystalline growth pattern animation"""
        # Start with seed points
        active_points = []
        crystal = [[False for _ in range(self.width)] for _ in range(self.height)]

        # Seed the border
        for i in range(self.width):
            active_points.extend(((i, 0), (i, self.height - 1)))
            crystal[0][i] = True
            crystal[self.height - 1][i] = True

        for i in range(1, self.height - 1):
            active_points.extend(((0, i), (self.width - 1, i)))
            crystal[i][0] = True
            crystal[i][self.width - 1] = True

        # Crystal growth patterns
        growth_chars = ["·", ":", "▪", "▫", "▣", "▨", "◆", "◇", "◈"]

        # Animation steps
        for _ in range(min(self.width, self.height) // 2):
            if not active_points:
                break

            # Select random active points to grow from
            growth_points = random.sample(active_points, min(len(active_points), 5))
            new_active_points = []

            for x, y in growth_points:
                # Try to grow in random directions
                directions = [
                    (0, 1),
                    (1, 0),
                    (0, -1),
                    (-1, 0),
                    (1, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                ]
                random.shuffle(directions)

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and not crystal[ny][nx]
                    ):
                        # Grow in this direction
                        crystal[ny][nx] = True
                        new_active_points.append((nx, ny))

                        # Draw the crystal growth
                        char = random.choice(growth_chars)
                        with contextlib.suppress(curses.error):
                            stdscr.addstr(self.y + ny, self.x + nx, char)
            # Update active points
            active_points = list(set(active_points + new_active_points))
            # Remove points that can't grow anymore
            active_points = [
                (x, y)
                for x, y in active_points
                if any(
                    0 <= x + dx < self.width
                    and 0 <= y + dy < self.height
                    and not crystal[y + dy][x + dx]
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                )
            ]

            stdscr.refresh()
            time.sleep(0.05)

        # Final draw
        self.draw(stdscr)
