"""
ASCII Minimap Panel for Space Muck.

This module provides a minimap component showing the player's position in the game world.
"""

# Standard library imports
import math
import random
import time
from typing import Any, Dict, List, Tuple

import pygame

# Local application imports
from config import COLOR_TEXT
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_element.ascii_panel import ASCIIPanel

# Third-party library imports

class ASCIIMinimapPanel:
    """
    A minimap showing the player's position in the game world.

    This component visualizes the surrounding area with ASCII characters,
    highlighting points of interest and the player's current location.
    """

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "LOCAL MAP",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """
        Initialize the minimap panel.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style for the UI
        """
        self.rect = rect
        self.title = title
        self.style = style
        self.panel = ASCIIPanel(rect, title, style)

        # Map data
        self.map_width = max(10, rect.width - 10)
        self.map_height = max(10, rect.height - 10)
        self.map_data = self._generate_empty_map()
        self.player_pos = (self.map_width // 2, self.map_height // 2)
        self.points_of_interest = []

        # Style-based characters
        style_chars = {
            UIStyle.QUANTUM: {
                "player": "◉",
                "empty": "·",
                "asteroid": "○",
                "station": "◇",
                "planet": "◎",
                "ship": "◈",
                "unknown": "?",
            },
            UIStyle.SYMBIOTIC: {
                "player": "*",
                "empty": ".",
                "asteroid": "°",
                "station": "□",
                "planet": "◯",
                "ship": "⬠",
                "unknown": "?",
            },
            UIStyle.MECHANICAL: {
                "player": "X",
                "empty": ".",
                "asteroid": "#",
                "station": "[]",
                "planet": "O",
                "ship": ">",
                "unknown": "?",
            },
            UIStyle.ASTEROID: {
                "player": "@",
                "empty": " ",
                "asteroid": "*",
                "station": "H",
                "planet": "O",
                "ship": "^",
                "unknown": "?",
            },
            UIStyle.FLEET: {
                "player": "P",
                "empty": ".",
                "asteroid": "a",
                "station": "S",
                "planet": "O",
                "ship": "^",
                "unknown": "?",
            },
        }
        self.chars = style_chars.get(self.style, style_chars[UIStyle.MECHANICAL])

        # Animation state
        self.animation_state = {
            "active": False,
            "progress": 0.0,
            "start_time": 0,
            "duration": 0.5,
            "pulse": 0.0,
        }
        self.last_update_time = time.time()

    def _generate_empty_map(self) -> List[List[str]]:
        """Generate an empty map grid."""
        return [
            ["empty" for _ in range(self.map_width)] for _ in range(self.map_height)
        ]

    def update_map(
        self,
        map_data: List[List[str]],
        player_pos: Tuple[int, int],
        points_of_interest: List[Dict[str, Any]],
    ) -> None:
        """
        Update the map data.

        Args:
            map_data: 2D grid of map cell types
            player_pos: (x, y) position of the player
            points_of_interest: List of POI dictionaries with position and type
        """
        if map_data and map_data[0]:
            self.map_data = map_data
            self.map_height = len(map_data)
            self.map_width = len(map_data[0])

        self.player_pos = player_pos
        self.points_of_interest = points_of_interest

        # Start a brief animation when the map updates
        self._start_animation()

    def _start_animation(self) -> None:
        """Start an animation sequence."""
        self.animation_state["active"] = True
        self.animation_state["progress"] = 0.0
        self.animation_state["start_time"] = time.time()

    def update(
        self,
        player_position=None,
        grid=None,
        entity_grid=None,
        field_offset_x=None,
        field_offset_y=None,
        zoom_level=None,
    ) -> None:
        """Update animation state and minimap data.

        Args:
            player_position: Optional tuple of (x, y) coordinates
            grid: Optional game grid data
            entity_grid: Optional entity grid data
            field_offset_x: Optional field offset X
            field_offset_y: Optional field offset Y
            zoom_level: Optional zoom level
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update animation state
        if self.animation_state["active"]:
            elapsed = current_time - self.animation_state["start_time"]
            self.animation_state["progress"] = min(
                1.0, elapsed / self.animation_state["duration"]
            )
            if self.animation_state["progress"] >= 1.0:
                self.animation_state["active"] = False

        # Update pulse animation (for player position)
        self.animation_state["pulse"] = (self.animation_state["pulse"] + dt * 2.0) % (
            2.0 * math.pi
        )

        # Update minimap data if provided
        if player_position is not None:
            self.player_position = player_position

        if grid is not None:
            self.grid = grid

        if entity_grid is not None:
            self.entity_grid = entity_grid

        if field_offset_x is not None:
            self.field_offset_x = field_offset_x

        if field_offset_y is not None:
            self.field_offset_y = field_offset_y

        if zoom_level is not None:
            self.zoom_level = zoom_level

    def update_animations(self, delta_time: float) -> None:
        """Update animations based on elapsed time.

        Args:
            delta_time: Time elapsed since last frame in seconds
        """
        # Update animation state
        if self.animation_state["active"]:
            self.animation_state["progress"] = min(
                1.0,
                self.animation_state["progress"]
                + delta_time / self.animation_state["duration"],
            )
            if self.animation_state["progress"] >= 1.0:
                self.animation_state["active"] = False

        # Update pulse animation (for player position)
        self.animation_state["pulse"] = (
            self.animation_state["pulse"] + delta_time * 2.0
        ) % (2.0 * math.pi)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame events for this panel.

        Args:
            event: Pygame event to handle

        Returns:
            bool: True if the event was consumed
        """
        # Handle mouse hover for showing details about map elements
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            if self.rect.collidepoint(mouse_pos):
                # Could show tooltip or highlight on hover
                return True

        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """
        Draw the minimap panel.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        # Draw the panel background
        panel_rect = self.panel.draw(surface, font)

        # Calculate map drawing area
        char_width = font.size("X")[0]
        char_height = font.get_height()
        map_start_x = self.rect.x + 5
        map_start_y = self.rect.y + 25  # Leave room for title

        # Draw the map grid
        for y in range(self.map_height):
            for x in range(self.map_width):
                # Get the character to draw based on cell type
                cell_type = self.map_data[y][x]
                char = self.chars.get(cell_type, self.chars["unknown"])

                # Calculate position
                pos_x = map_start_x + (x * char_width)
                pos_y = map_start_y + (y * char_height)

                # Determine color based on cell type
                color = self._get_cell_color(cell_type)

                # Draw the cell
                text_surface = font.render(char, True, color)
                surface.blit(text_surface, (pos_x, pos_y))

        # Draw points of interest
        for poi in self.points_of_interest:
            poi_x, poi_y = poi["position"]
            poi_type = poi["type"]
            poi_char = self.chars.get(poi_type, self.chars["unknown"])

            # Calculate position
            pos_x = map_start_x + (poi_x * char_width)
            pos_y = map_start_y + (poi_y * char_height)

            # Determine color
            color = self._get_cell_color(poi_type)

            # Draw the POI
            text_surface = font.render(poi_char, True, color)
            surface.blit(text_surface, (pos_x, pos_y))

        # Draw player position with pulsing effect
        player_x, player_y = self.player_pos
        pos_x = map_start_x + (player_x * char_width)
        pos_y = map_start_y + (player_y * char_height)

        # Pulse effect for player marker
        pulse_factor = 0.5 + 0.5 * math.sin(self.animation_state["pulse"])
        player_color = self._get_pulsing_color(pulse_factor)

        # Draw player marker
        player_char = self.chars["player"]
        text_surface = font.render(player_char, True, player_color)
        surface.blit(text_surface, (pos_x, pos_y))

        # Draw coordinates
        coord_text = f"X:{player_x} Y:{player_y}"
        coord_x = self.rect.x + 5
        coord_y = self.rect.y + self.rect.height - 20
        text_surface = font.render(coord_text, True, COLOR_TEXT)
        surface.blit(text_surface, (coord_x, coord_y))

        return panel_rect

    def _get_cell_color(self, cell_type: str) -> Tuple[int, int, int]:
        """Get the appropriate color for a cell type."""
        colors = {
            "empty": (100, 100, 100),  # Gray
            "asteroid": (150, 150, 150),  # Light gray
            "station": (100, 200, 255),  # Light blue
            "planet": (100, 255, 100),  # Green
            "ship": (255, 100, 100),  # Red
            "unknown": (255, 255, 100),  # Yellow
        }
        return colors.get(cell_type, COLOR_TEXT)

    def _get_pulsing_color(self, pulse_factor: float) -> Tuple[int, int, int]:
        """Get a pulsing color based on the animation state."""
        base_color = (0, 255, 255)  # Cyan
        return tuple(int(c * pulse_factor) for c in base_color)

    def generate_random_map(self, density: float = 0.3) -> None:
        """
        Generate a random map for testing.

        Args:
            density: Density of objects (0.0 to 1.0)
        """
        # Create empty map
        new_map = self._generate_empty_map()

        # Add random elements
        cell_types = ["empty", "asteroid", "station", "planet", "ship"]
        weights = [
            1.0 - density,
            density * 0.7,
            density * 0.1,
            density * 0.1,
            density * 0.1,
        ]

        for y in range(self.map_height):
            for x in range(self.map_width):
                # Skip player position
                if (x, y) == self.player_pos:
                    continue

                # Randomly select cell type based on weights
                cell_type = random.choices(cell_types, weights=weights, k=1)[0]
                new_map[y][x] = cell_type

        # Generate some points of interest
        poi_count = int(self.map_width * self.map_height * density * 0.05)
        points_of_interest = []

        for _ in range(poi_count):
            poi_x = random.randint(0, self.map_width - 1)
            poi_y = random.randint(0, self.map_height - 1)
            poi_type = random.choice(["station", "planet", "ship"])

            points_of_interest.append(
                {
                    "position": (poi_x, poi_y),
                    "type": poi_type,
                    "name": f"{poi_type.capitalize()}-{random.randint(1, 999)}",
                }
            )

        # Update the map
        self.update_map(new_map, self.player_pos, points_of_interest)
