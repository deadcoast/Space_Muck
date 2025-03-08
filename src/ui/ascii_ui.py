"""
ASCII UI components for Space Muck.

This module provides components and utilities for creating ASCII-style UI interfaces
for the converter management system and other game elements.
"""

from typing import Tuple, List, Dict, Optional, Any, Callable
import pygame
import time
import math
import random
import itertools
from enum import Enum, auto

from ui.draw_utils import draw_text, draw_panel
from config import COLOR_TEXT, COLOR_BG


class UIStyle(Enum):
    """Different visual styles for UI components"""

    SYMBIOTIC = auto()  # Organic, evolving patterns
    ASTEROID = auto()  # Rough, mineral-like patterns
    MECHANICAL = auto()  # Ship/tech inspired rigid patterns
    QUANTUM = auto()  # Probability-wave inspired patterns
    FLEET = auto()  # Organized, military-like patterns

    @classmethod
    def get_border_chars(cls, style) -> Dict[str, str]:
        """Get border characters based on the selected style"""
        borders = {
            cls.SYMBIOTIC: {
                "tl": "╭",
                "tr": "╮",
                "bl": "╰",
                "br": "╯",
                "h": "─",
                "v": "│",
            },
            cls.ASTEROID: {
                "tl": "┌",
                "tr": "┐",
                "bl": "└",
                "br": "┘",
                "h": "═",
                "v": "║",
            },
            cls.MECHANICAL: {
                "tl": "╔",
                "tr": "╗",
                "bl": "╚",
                "br": "╝",
                "h": "═",
                "v": "║",
            },
            cls.QUANTUM: {
                "tl": "╒",
                "tr": "╕",
                "bl": "╘",
                "br": "╛",
                "h": "╌",
                "v": "╎",
            },
            cls.FLEET: {"tl": "┏", "tr": "┓", "bl": "┗", "br": "┛", "h": "━", "v": "┃"},
        }
        return borders.get(style, borders[cls.MECHANICAL])

    @classmethod
    def get_style_for_converter(cls, converter_type: str) -> "UIStyle":
        """Get appropriate style based on converter type"""
        style_map = {
            "BASIC": cls.MECHANICAL,
            "ADVANCED": cls.QUANTUM,
            "ORGANIC": cls.SYMBIOTIC,
            "FLEET": cls.FLEET,
            "MINING": cls.ASTEROID,
        }
        return style_map.get(converter_type.upper(), cls.MECHANICAL)


class ASCIIBox:
    """A box drawn with ASCII characters for borders."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: Optional[str] = None,
        style: UIStyle = UIStyle.MECHANICAL,
        converter_type: Optional[str] = None,
    ):
        """
        Initialize an ASCII box.

        Args:
            x: X coordinate of the top-left corner
            y: Y coordinate of the top-left corner
            width: Width of the box in characters
            height: Height of the box in characters
            title: Optional title to display at the top of the box
            border_style: Style of border ('single', 'double', 'heavy')
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title = title
        self.content: List[Tuple[int, int, str, Optional[Dict[str, Any]]]] = []
        self.style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )
        self.borders = UIStyle.get_border_chars(self.style)

        # Animation state
        self.animation = {
            "active": False,
            "progress": 0.0,
            "start_time": 0,
            "duration": 0.3,
            "cells": [[False for _ in range(width)] for _ in range(height)],
        }

    def start_animation(self, duration: float = 0.3) -> None:
        """Start an animation sequence.

        Args:
            duration: Animation duration in seconds
        """
        self.animation["active"] = True
        self.animation["progress"] = 0.0
        self.animation["start_time"] = time.time()
        self.animation["duration"] = duration

        # Initialize with quantum-inspired pattern
        for y in range(self.height):
            for x in range(self.width):
                if self.style == UIStyle.QUANTUM:
                    self.animation["cells"][y][x] = math.sin(x / 2 + y / 2) > 0
                elif self.style == UIStyle.SYMBIOTIC:
                    self.animation["cells"][y][x] = random.random() < 0.3
                else:
                    self.animation["cells"][y][x] = random.random() < 0.5

    def update_animation(self) -> None:
        """Update animation state based on elapsed time"""
        if not self.animation["active"]:
            return

        current_time = time.time()
        elapsed = current_time - self.animation["start_time"]
        self.animation["progress"] = min(1.0, elapsed / self.animation["duration"])

        if self.animation["progress"] >= 1.0:
            self.animation["active"] = False
            return

        # Update cells based on style
        if self.style == UIStyle.QUANTUM:
            self._update_quantum_pattern(elapsed)
        elif self.style == UIStyle.SYMBIOTIC:
            self._update_cellular_pattern()

    def _update_quantum_pattern(self, elapsed: float) -> None:
        """Update quantum-style animation pattern"""
        freq = 4.0
        for y in range(self.height):
            for x in range(self.width):
                phase = elapsed * freq
                value = math.sin(x / 2 + y / 2 + phase)
                self.animation["cells"][y][x] = value > 0

    def _update_cellular_pattern(self) -> None:
        """Update cellular automaton pattern using Game of Life rules"""
        new_cells = [[False for _ in range(self.width)] for _ in range(self.height)]

        for y, x in itertools.product(range(self.height), range(self.width)):
            # Count live neighbors
            neighbors = sum(
                self.animation["cells"][ny][nx]
                for dx, dy in itertools.product([-1, 0, 1], repeat=2)
                if (dx, dy) != (0, 0)
                and 0 <= (nx := x + dx) < self.width
                and 0 <= (ny := y + dy) < self.height
            )

            # Apply Game of Life rules
            current = self.animation["cells"][y][x]
            new_cells[y][x] = neighbors in [2, 3] if current else neighbors == 3

        self.animation["cells"] = new_cells

    def add_text(
        self, x: int, y: int, text: str, props: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add text to the box content.

        Args:
            x: Relative X position within the box
            y: Relative Y position within the box
            text: Text to display
            props: Optional properties for styling
        """
        self.content.append((x, y, text, props))

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = COLOR_TEXT,
        bg_color: Tuple[int, int, int] = COLOR_BG,
        alpha: int = 255,
    ) -> pygame.Rect:
        """
        Draw the ASCII box on the surface.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            base_color: Base color for drawing
            bg_color: Background color
            alpha: Transparency (0-255)

        Returns:
            pygame.Rect: The drawn area
        """
        # Calculate dimensions based on font size
        char_width, char_height = font.size("X")
        box_width = self.width * char_width
        box_height = self.height * char_height

        # Create box surface with alpha
        box_surf = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        box_surf.fill((bg_color[0], bg_color[1], bg_color[2], alpha))

        # Update and draw animation if active
        if self.animation["active"]:
            self._draw_animation(box_surf, font, base_color, char_width, char_height)

        # Draw borders
        self._draw_borders(box_surf, font, base_color, char_width, char_height)

        # Draw title if provided
        if self.title:
            self._draw_title(box_surf, font, base_color, char_width)

        # Draw content
        self._draw_content(box_surf, font, base_color, char_width, char_height)
        return surface.blit(box_surf, (self.x, self.y))

    def _draw_animation(self, surface: pygame.Surface, font: pygame.font.Font,
                      base_color: Tuple[int, int, int], char_width: int, char_height: int) -> None:
        """Draw animation effects on the surface."""
        self.update_animation()
        alpha = int(255 * self.animation["progress"])

        for y, x in itertools.product(range(self.height), range(self.width)):
            if not self.animation["cells"][y][x]:
                continue

            char, color = self._get_animation_char_and_color(alpha, base_color)
            draw_text(
                surface,
                char,
                x * char_width,
                y * char_height,
                size=font.get_height(),
                color=color,
            )

    def _get_animation_char_and_color(self, alpha: int, base_color: Tuple[int, int, int]) -> Tuple[str, Tuple]:
        """Get the appropriate character and color for animation based on style."""
        if self.style == UIStyle.QUANTUM:
            chars = ["·", "∙", "•", "◦", "○", "◌", "◍", "◎", "●"]
            idx = int(self.animation["progress"] * (len(chars) - 1))
            return chars[idx], (200, 200, 255, alpha)
        return "█", (*base_color, alpha)

    def _draw_borders(self, surface: pygame.Surface, font: pygame.font.Font,
                     base_color: Tuple[int, int, int], char_width: int, char_height: int) -> None:
        """Draw box borders on the surface."""
        # Top border
        top_border = self.borders["tl"] + self.borders["h"] * (self.width - 2) + self.borders["tr"]
        draw_text(surface, top_border, 0, 0, size=font.get_height(), color=base_color)

        # Bottom border
        bottom_border = self.borders["bl"] + self.borders["h"] * (self.width - 2) + self.borders["br"]
        draw_text(
            surface,
            bottom_border,
            0,
            (self.height - 1) * char_height,
            size=font.get_height(),
            color=base_color,
        )

        # Side borders
        for i in range(1, self.height - 1):
            draw_text(
                surface,
                self.borders["v"],
                0,
                i * char_height,
                size=font.get_height(),
                color=base_color,
            )
            draw_text(
                surface,
                self.borders["v"],
                (self.width - 1) * char_width,
                i * char_height,
                size=font.get_height(),
                color=base_color,
            )

    def _draw_title(self, surface: pygame.Surface, font: pygame.font.Font,
                    base_color: Tuple[int, int, int], char_width: int) -> None:
        """Draw the title on the surface if one is set."""
        title_x = max((self.width - len(self.title)) // 2, 1)
        draw_text(
            surface,
            self.title,
            title_x * char_width,
            0,
            size=font.get_height(),
            color=base_color,
        )

    def _draw_content(self, surface: pygame.Surface, font: pygame.font.Font,
                      base_color: Tuple[int, int, int], char_width: int, char_height: int) -> None:
        """Draw the box content on the surface."""
        for x, y, text, props in self.content:
            color = props.get("color", base_color) if props else base_color
            draw_text(
                surface, text, x * char_width, y * char_height, size=font.get_height(), color=color
            )


class ASCIIPanel:
    """A panel with ASCII styling for complex UI layouts."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: Optional[str] = None,
        style: UIStyle = UIStyle.MECHANICAL,
        converter_type: Optional[str] = None,
    ):
        """
        Initialize an ASCII panel.

        Args:
            rect: Rectangle defining position and size
            title: Optional title to display at the top
            border_style: Style of border ('single', 'double', 'heavy')
        """
        self.rect = rect
        self.title = title
        self.style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )
        self.components: List[Any] = []

        self._initialize_colors()

    def add_component(self, component: Any) -> None:
        """
        Add a component to the panel.

        Args:
            component: Component to add
        """
        self.components.append(component)

    def _initialize_colors(self) -> None:
        """Initialize style-based background colors."""
        style_colors = {
            UIStyle.QUANTUM: (20, 20, 30, 200),
            UIStyle.SYMBIOTIC: (20, 30, 20, 200),
            UIStyle.MECHANICAL: (30, 30, 40, 200),
        }
        self.background_color = style_colors.get(self.style, (25, 25, 35, 200))

    def _create_ascii_box(self, font: pygame.font.Font) -> ASCIIBox:
        """Create an ASCII box for the panel border."""
        char_width, char_height = font.size("X")
        width_chars = self.rect.width // char_width
        height_chars = self.rect.height // char_height

        return ASCIIBox(
            self.rect.x,
            self.rect.y,
            width_chars,
            height_chars,
            title=None,  # Already handled by panel
            style=self.style,
        )

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
    ) -> pygame.Rect:
        """
        Draw the ASCII panel and its components.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        # Create and draw background panel
        panel_rect = draw_panel(
            surface,
            self.rect,
            color=self.background_color,
            border_color=(100, 100, 140),
            header=self.title,
            header_color=(40, 40, 60, 220),
        )

        # Draw ASCII border
        box = self._create_ascii_box(font)
        box.draw(surface, font, COLOR_TEXT, bg_color=(0, 0, 0, 0))

        # Draw components
        for component in self.components:
            component.draw(surface, font)

        return panel_rect


class ASCIIProgressBar:
    """ASCII-style progress bar."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        progress: float = 0.0,
        style: UIStyle = UIStyle.MECHANICAL,
        converter_type: Optional[str] = None,
    ):
        """
        Initialize an ASCII progress bar.

        Args:
            x: X position
            y: Y position
            width: Width in characters
            progress: Initial progress (0.0 to 1.0)
            style: Visual style ('block', 'line', 'equal')
        """
        self.x = x
        self.y = y
        self.width = width
        self.progress = max(0.0, min(1.0, progress))
        self.style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )
        self._set_progress_chars()

    def _set_progress_chars(self) -> None:
        """Set the progress bar characters based on UI style."""
        style_chars = {
            UIStyle.QUANTUM: ("◈", "◇"),      # ◈, ◇
            UIStyle.SYMBIOTIC: ("█", "░"),   # █, ░
            UIStyle.MECHANICAL: ("█", "▒"),  # █, ▒
            UIStyle.ASTEROID: ("◆", "◇"),   # ◆, ◇
            UIStyle.FLEET: ("■", "□"),      # ■, □
        }
        self.fill_char, self.empty_char = style_chars.get(self.style, ("#", " "))

    def set_progress(self, progress: float) -> None:
        """
        Set the current progress.

        Args:
            progress: Progress value (0.0 to 1.0)
        """
        self.progress = max(0.0, min(1.0, progress))

    def _draw_filled_portion(self, surface: pygame.Surface, font: pygame.font.Font,
                           filled_width: int, fill_color: Tuple[int, int, int]) -> pygame.Rect:
        """Draw the filled portion of the progress bar."""
        if filled_width <= 0:
            return pygame.Rect(self.x, self.y, 0, 0)

        filled_text = self.fill_char * filled_width
        return draw_text(surface, filled_text, self.x, self.y, size=font.get_height(), color=fill_color)

    def _draw_empty_portion(self, surface: pygame.Surface, font: pygame.font.Font,
                           filled_width: int, empty_width: int, filled_rect: pygame.Rect,
                           empty_color: Tuple[int, int, int]) -> pygame.Rect:
        """Draw the empty portion of the progress bar."""
        if empty_width <= 0:
            return filled_rect

        empty_text = self.empty_char * empty_width
        return draw_text(
            surface,
            empty_text,
            self.x + filled_rect.width,
            self.y,
            size=font.get_height(),
            color=empty_color,
        )

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        fill_color: Tuple[int, int, int] = (100, 255, 100),
        empty_color: Tuple[int, int, int] = (100, 100, 100),
    ) -> pygame.Rect:
        """
        Draw the progress bar.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            fill_color: Color for the filled portion
            empty_color: Color for the empty portion

        Returns:
            pygame.Rect: The drawn area
        """
        filled_width = int(self.width * self.progress)
        empty_width = self.width - filled_width

        # Draw both portions
        filled_rect = self._draw_filled_portion(surface, font, filled_width, fill_color)
        empty_rect = self._draw_empty_portion(surface, font, filled_width, empty_width,
                                             filled_rect, empty_color)

        return filled_rect.union(empty_rect) if empty_width > 0 else filled_rect


class ASCIIButton:
    """Interactive button with ASCII styling."""

    def __init__(
        self,
        x: int,
        y: int,
        text: str,
        callback: Optional[Callable[[], None]] = None,
        style: UIStyle = UIStyle.MECHANICAL,
        converter_type: Optional[str] = None,
    ):
        """
        Initialize an ASCII button.

        Args:
            x: X position
            y: Y position
            text: Button text
            callback: Function to call when clicked
            style: Button style ('bracket', 'block', 'underline')
        """
        self.x = x
        self.y = y
        self.text = text
        self.callback = callback
        self.hover = False
        self.style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )
        self._set_button_chars()
        self.rect = pygame.Rect(0, 0, 0, 0)  # Will be set properly when drawn

        # Animation state
        self.animation = {
            "active": False,
            "progress": 0.0,
            "start_time": 0,
            "duration": 0.2,
        }

    def _set_button_chars(self) -> None:
        """Set button characters based on UI style."""
        style_chars = {
            UIStyle.QUANTUM: ("◧", "◨"),      # ◧, ◨
            UIStyle.SYMBIOTIC: ("▐", "▌"),   # ▐, ▌
            UIStyle.MECHANICAL: ("[", "]"),
            UIStyle.ASTEROID: ("◄", "►"),    # ◄, ►
            UIStyle.FLEET: ("◀", "▶"),      # ◀, ▶
        }
        self.prefix, self.suffix = style_chars.get(self.style, ("<", ">"))

    def is_hover(self, mouse_pos: Tuple[int, int]) -> bool:
        """Check if the mouse is hovering over this button."""
        return self.rect.collidepoint(mouse_pos)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame events for this button.

        Returns:
            bool: True if the event was consumed
        """
        if event.type == pygame.MOUSEMOTION:
            was_hover = self.hover
            self.hover = self.is_hover(event.pos)

            # Start hover animation
            if self.hover and not was_hover:
                self.animation["active"] = True
                self.animation["start_time"] = time.time()

            return self.hover

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and (self.is_hover(event.pos) and self.callback)
        ):
            # Start click animation
            self.animation["active"] = True
            self.animation["start_time"] = time.time()
            self.animation["duration"] = 0.2

            self.callback()
            return True

        return False

    def _get_hover_color(self, color: Tuple[int, int, int],
                         hover_color: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Get the appropriate hover color based on style."""
        if hover_color is not None:
            return hover_color

        style_colors = {
            UIStyle.QUANTUM: (180, 180, 255),
            UIStyle.SYMBIOTIC: (180, 255, 180),
            UIStyle.MECHANICAL: (200, 200, 255),
            UIStyle.ASTEROID: (255, 200, 180),
            UIStyle.FLEET: (180, 220, 255),
        }
        return style_colors.get(self.style, (200, 200, 255))

    def _get_animation_alpha(self) -> int:
        """Calculate alpha value based on animation state."""
        if not self.animation["active"]:
            return 255

        current_time = time.time()
        elapsed = current_time - self.animation["start_time"]
        progress = min(1.0, elapsed / self.animation["duration"])

        if progress < 1.0:
            return int(255 * (0.7 + 0.3 * math.sin(progress * math.pi * 4)))

        self.animation["active"] = False
        return 255

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        color: Tuple[int, int, int] = COLOR_TEXT,
        hover_color: Optional[Tuple[int, int, int]] = None,
    ) -> pygame.Rect:
        """
        Draw the button.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            color: Normal text color
            hover_color: Color when hovering

        Returns:
            pygame.Rect: The drawn area
        """
        hover_color = self._get_hover_color(color, hover_color)
        draw_color = hover_color if self.hover else color
        alpha = self._get_animation_alpha()

        button_text = f"{self.prefix}{self.text}{self.suffix}"
        self.rect = draw_text(
            surface, button_text, self.x, self.y, size=font.get_height(), color=(*draw_color, alpha)
        )

        return self.rect


class ASCIIMetricsPanel:
    """A specialized panel for displaying production metrics and performance indicators."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Production Metrics",
        style: UIStyle = UIStyle.MECHANICAL,
        converter_type: Optional[str] = None,
    ):
        """Initialize a metrics panel.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
            converter_type: Optional converter type to determine style
        """
        self.rect = rect
        self.title = title
        self.style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )
        self.metrics: Dict[str, Any] = {
            "throughput": 0.0,  # units/min
            "energy_usage": 0.0,  # kW
            "queue_size": 0,  # items in process queue
            "utilization": 0.0,  # percentage
            "uptime": 0.0,  # hours
            "efficiency": 0.0,  # percentage
        }
        self.history: Dict[str, List[float]] = {
            "throughput": [],
            "energy_usage": [],
            "efficiency": [],
        }
        self.max_history = 60  # Keep 1 hour of minute-by-minute data

        # Base colors for different styles
        style_colors = {
            UIStyle.QUANTUM: {
                'high': (100, 200, 255),    # Cyan
                'medium': (255, 200, 100),  # Orange
                'low': (255, 100, 100),     # Red
                'primary': (100, 200, 255),  # Cyan
                'accent1': (255, 100, 100),  # Red
                'accent2': (100, 255, 100),  # Green
                'accent3': (200, 100, 255),  # Purple
            },
            UIStyle.SYMBIOTIC: {
                'high': (150, 255, 150),    # Light green
                'medium': (255, 255, 150),  # Light yellow
                'low': (255, 150, 150),     # Light red
                'primary': (150, 255, 150),  # Light green
                'accent1': (255, 150, 150),  # Light red
                'accent2': (150, 150, 255),  # Light blue
                'accent3': (255, 150, 255),  # Light purple
            },
            UIStyle.MECHANICAL: {
                'high': (100, 255, 100),    # Green
                'medium': (255, 255, 100),  # Yellow
                'low': (255, 100, 100),     # Red
                'primary': (100, 255, 100),  # Green
                'accent1': (255, 100, 100),  # Red
                'accent2': (100, 100, 255),  # Blue
                'accent3': (255, 100, 255),  # Magenta
            }
        }
        
        # Get base colors for current style
        base_colors = style_colors.get(self.style, style_colors[UIStyle.MECHANICAL])
        
        # Map colors to metrics
        self.colors = {
            **base_colors,  # Include high/medium/low and base colors
            'throughput': base_colors['primary'],
            'energy': base_colors['accent1'],
            'queue': base_colors['accent2'],
            'efficiency': base_colors['medium'],
            'uptime': base_colors['accent3'],
        }

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update current metrics and historical data.

        Args:
            metrics: Dictionary of current metric values
        """
        # Update current metrics
        self.metrics.update(metrics)

        # Update historical data
        for key in ["throughput", "energy_usage", "efficiency"]:
            if key in metrics:
                self.history[key].append(metrics[key])
                if len(self.history[key]) > self.max_history:
                    self.history[key].pop(0)

    def _get_trend_indicator(self, values: List[float]) -> str:
        """Get trend indicator based on recent values."""
        if len(values) < 2:
            return "-"

        recent = values[-5:] if len(values) >= 5 else values
        old_values = values[:-5] if len(values) > 5 else [values[0]]
        
        avg_old = sum(old_values) / len(old_values)
        avg_new = sum(recent) / len(recent)

        trend_indicators = {
            lambda x: x > 1.05: "↗",  # Significant increase
            lambda x: x < 0.95: "↘",  # Significant decrease
            lambda x: True: "→",      # Stable
        }

        ratio = avg_new / avg_old
        return next(indicator for condition, indicator in trend_indicators.items()
                   if condition(ratio))

    def _draw_metric_label(self, surface: pygame.Surface, font: pygame.font.Font,
                          x: int, y: int, label: str, color: Tuple[int, int, int]) -> int:
        """Draw the metric label and return its width."""
        text = f"{label}: "
        draw_text(surface, text, x, y, size=font.get_height(), color=color)
        return font.size(text)[0]

    def _get_bar_chars(self, fill_width: int, bar_width: int) -> str:
        """Generate the bar string with appropriate characters."""
        chars = ["=", "#", "%", "@"]
        return "".join(
            chars[min(3, int(4 * i / fill_width))] if i < fill_width else "="
            for i in range(bar_width)
        )

    def _draw_metric_bar(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
        value: float,
        max_value: float,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a metric bar with label and value."""
        # Draw label and get its width
        label_width = self._draw_metric_label(surface, font, x, y, label, color)

        # Calculate bar dimensions
        bar_width = int((width - label_width) * 0.6)
        fill_width = int(bar_width * (value / max_value))

        # Draw bar
        bar_str = self._get_bar_chars(fill_width, bar_width)
        draw_text(surface, bar_str, x + label_width, y, size=font.get_height(), color=color)

        # Draw value
        value_str = f"{value:.1f}"
        draw_text(
            surface,
            value_str,
            x + label_width + bar_width + font.size(" ")[0],
            y,
            size=font.get_height(),
            color=color,
        )

    def _get_metric_color(self, metric_name: str) -> Tuple[int, int, int]:
        """Get the appropriate color for a metric based on its value."""
        if metric_name == "utilization":
            value = self.metrics["utilization"]
            thresholds = [(80, "high"), (50, "medium"), (0, "low")]
        elif metric_name == "efficiency":
            value = self.metrics["efficiency"]
            thresholds = [(90, "high"), (70, "medium"), (0, "low")]
        else:
            return self.colors["medium"]

        return next(self.colors[level] for threshold, level in thresholds
                   if value > threshold)

    def _draw_throughput(self, surface: pygame.Surface, font: pygame.font.Font,
                        x: int, y: int) -> None:
        """Draw throughput metrics."""
        throughput = self.metrics["throughput"]
        trend = self._get_trend_indicator(self.history["throughput"])
        draw_text(
            surface,
            f"Throughput: {throughput:.1f} units/min {trend}",
            x, y,
            size=font.get_height(),
            color=self.colors["high"]
        )

    def _draw_uptime(self, surface: pygame.Surface, font: pygame.font.Font,
                     x: int, y: int) -> None:
        """Draw uptime information."""
        hours = int(self.metrics["uptime"])
        minutes = int((self.metrics["uptime"] - hours) * 60)
        draw_text(
            surface,
            f"Uptime: {hours}h {minutes}m",
            x, y,
            size=font.get_height(),
            color=self.colors["low"]
        )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the metrics panel.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)

        # Calculate layout
        margin = font.get_height()
        x = self.rect.x + margin
        y = self.rect.y + margin * 2  # Extra margin for title
        width = self.rect.width - margin * 2

        # Draw metrics
        self._draw_throughput(surface, font, x, y)

        # Draw energy usage bar
        y += margin
        self._draw_metric_bar(
            surface, font, x, y, width,
            self.metrics["energy_usage"], 100.0,
            "Energy", self.colors["medium"]
        )

        # Draw queue size
        y += margin
        draw_text(
            surface,
            f"Queue: {self.metrics['queue_size']} items",
            x, y,
            size=font.get_height(),
            color=self.colors["medium"]
        )

        # Draw utilization
        y += margin
        self._draw_metric_bar(
            surface, font, x, y, width,
            self.metrics["utilization"], 100.0,
            "Util%", self._get_metric_color("utilization")
        )

        # Draw uptime
        y += margin
        self._draw_uptime(surface, font, x, y)

        # Draw efficiency with trend
        y += margin
        eff_trend = self._get_trend_indicator(self.history["efficiency"])
        self._draw_metric_bar(
            surface, font, x, y, width,
            self.metrics["efficiency"], 100.0,
            f"Eff%{eff_trend}", self._get_metric_color("efficiency")
        )

        return panel_rect


class ASCIIChainVisualizer:
    """Visualizer for displaying converter chains and resource flows."""
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Chain Flow",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a chain visualizer.
        
        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        self.converters: List[Dict[str, Any]] = []
        self.connections: List[Tuple[int, int]] = []  # (from_idx, to_idx)
        self.flow_rates: Dict[Tuple[int, int], float] = {}  # (from, to) -> rate
        
        # Enhanced tracking of converter states
        self.converter_states: Dict[int, Dict[str, Any]] = {}
        self.efficiency_history: Dict[int, List[float]] = {}
        self.max_history_points = 60  # Keep one hour of minute-by-minute data
        
        # Animation and visual settings
        self.animation_settings = {
            'base_speed': 8.0,
            'speed_scale': 0.1,  # For flow rate scaling
            'max_speed': 24.0,   # Maximum animation speed
            'pulse_frequency': 0.5,  # Pulses per second
            'glow_intensity': 0.8,   # For status indicators
        }
        
        # Performance optimization
        self.cache_timeout = 1.0  # Seconds before refreshing cached calculations
        self._cached_layouts: Dict[str, Any] = {}
        self._last_cache_time = 0.0
        
        # Style-based characters
        style_chars = {
            UIStyle.QUANTUM: {
                'node': '@',
                'flow': '~',
                'arrow': '>',
                'corner': '+',
                'anim_chars': ['⋅', '∙', '•', '∙'],  # Quantum dots
                'resource_chars': {
                    'energy': '⚡',
                    'matter': '⬢',
                    'fluid': '⬡',
                    'data': '⬟',
                    'default': '⬡'
                }
            },
            UIStyle.SYMBIOTIC: {
                'node': '*',
                'flow': '.',
                'arrow': '>',
                'corner': '+',
                'anim_chars': ['·', '•', '○', '•'],  # Organic shapes
                'resource_chars': {
                    'energy': '✧',
                    'matter': '✦',
                    'fluid': '❋',
                    'data': '✺',
                    'default': '✧'
                }
            },
            UIStyle.MECHANICAL: {
                'node': '#',
                'flow': '-',
                'arrow': '>',
                'corner': '+',
                'anim_chars': ['.', 'o', 'O', 'o'],  # Mechanical progression
                'resource_chars': {
                    'energy': '⚊',
                    'matter': '⚌',
                    'fluid': '⚏',
                    'data': '⚎',
                    'default': '⚊'
                }
            }
        }
        self.chars = style_chars.get(self.style, style_chars[UIStyle.MECHANICAL])
        self.animation_offset = 0.0
        self.last_animation_time = time.time()
    
    def set_chain(self, converters: List[Dict[str, Any]], connections: List[Tuple[int, int]],
                  flow_rates: Optional[Dict[Tuple[int, int], float]] = None) -> None:
        """Set the current chain configuration.
        
        Args:
            converters: List of converter info dictionaries
            connections: List of (from_idx, to_idx) connections
            flow_rates: Optional dict of flow rates between converters
        """
        self.converters = converters
        self.connections = connections
        self.flow_rates = flow_rates or {}
        
        # Update converter states and initialize history if needed
        current_time = time.time()
        for idx, converter in enumerate(converters):
            if idx not in self.converter_states:
                self.converter_states[idx] = {
                    'last_update': current_time,
                    'status_duration': 0.0,
                    'total_uptime': 0.0,
                    'status_changes': [],
                    'peak_efficiency': 0.0,
                    'min_efficiency': 100.0,
                }
            if idx not in self.efficiency_history:
                self.efficiency_history[idx] = []
            
            # Update state tracking
            state = self.converter_states[idx]
            efficiency = converter.get('efficiency', 0.0)
            state['peak_efficiency'] = max(state['peak_efficiency'], efficiency)
            state['min_efficiency'] = min(state['min_efficiency'], efficiency)
            
            # Track status changes
            current_status = converter.get('status', 'idle')
            if state['status_changes'] and state['status_changes'][-1][0] != current_status:
                state['status_changes'].append((current_status, current_time))
                if len(state['status_changes']) > 10:  # Keep last 10 status changes
                    state['status_changes'] = state['status_changes'][-10:]
            
            # Update efficiency history
            self.efficiency_history[idx].append(efficiency)
            if len(self.efficiency_history[idx]) > self.max_history_points:
                self.efficiency_history[idx] = self.efficiency_history[idx][-self.max_history_points:]
        
        # Clear cache to force layout recalculation
        self._cached_layouts.clear()
        self._last_cache_time = current_time
    

        

        
        # Status colors based on style
        status_colors = {
            UIStyle.QUANTUM: {
                'active': (100, 200, 255),  # Cyan
                'error': (255, 100, 100),   # Red
                'idle': (255, 200, 100),    # Orange
                'paused': (200, 100, 255)   # Purple
            },
            UIStyle.SYMBIOTIC: {
                'active': (150, 255, 150),  # Light green
                'error': (255, 150, 150),   # Light red
                'idle': (255, 255, 150),    # Light yellow
                'paused': (150, 150, 255)   # Light blue
            },
            UIStyle.MECHANICAL: {
                'active': (100, 255, 100),  # Green
                'error': (255, 100, 100),   # Red
                'idle': (200, 200, 100),    # Yellow
                'paused': (100, 100, 255)   # Blue
            }
        }
        self.status_colors = status_colors.get(self.style, status_colors[UIStyle.MECHANICAL])

    def _get_node_color(self, converter: Dict[str, Any], idx: int) -> Tuple[int, int, int]:
        """Get the appropriate color for a converter node based on its status and efficiency."""
        base_color = self.status_colors.get(converter.get('status', ''), COLOR_TEXT)
        
        # Get state information
        state = self.converter_states.get(idx, {})
        efficiency = converter.get('efficiency', 0.0)
        peak_efficiency = state.get('peak_efficiency', 0.0)
        
        # Calculate color intensity based on current vs peak efficiency
        if peak_efficiency > 0:
            intensity = min(1.0, (efficiency / peak_efficiency) * 1.2)  # Allow 20% boost for visual pop
            return tuple(int(c * intensity) for c in base_color)
        
        return base_color
    
    def _calculate_animation_speed(self, flow_rate: float) -> float:
        """Calculate animation speed based on flow rate and settings."""
        base_speed = self.animation_settings['base_speed']
        speed_scale = self.animation_settings['speed_scale']
        max_speed = self.animation_settings['max_speed']
        
        # Scale speed with flow rate, but cap it
        speed = base_speed + (flow_rate * speed_scale)
        return min(speed, max_speed)
    
    def _get_efficiency_trend(self, idx: int) -> float:
        """Calculate efficiency trend for a converter."""
        history = self.efficiency_history.get(idx, [])
        if len(history) < 2:
            return 0.0
        
        # Calculate trend over last 5 points or all points if fewer
        window = min(5, len(history))
        recent = history[-window:]
        if not recent:
            return 0.0
        
        # Simple linear regression for trend
        x = list(range(len(recent)))
        y = recent
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        
        return numerator / denominator if denominator != 0 else 0.0

    def _draw_node_stats(self, surface: pygame.Surface, font: pygame.font.Font,
                        x: int, y: int, idx: int, converter: Dict[str, Any], color: Tuple[int, int, int]) -> None:
        """Draw converter statistics below the node."""
        # Get state information
        state = self.converter_states.get(idx, {})
        efficiency_trend = self._get_efficiency_trend(idx)
        
        # Get tier indicator character based on tier level
        tier_chars = {
            1: '⚍', 2: '⚎', 3: '⚏', 4: '⚌', 5: '⚋'
        }
        tier = converter.get('tier', 1)
        tier_char = tier_chars.get(tier, '⚍')
        
        # Get status indicator
        status_chars = {
            'active': '▶',
            'error': '⚠',
            'idle': '⏸',
            'paused': '⏹'
        }
        status = converter.get('status', 'idle')
        status_char = status_chars.get(status, '?')
        
        # Format efficiency with trend indicator
        efficiency = converter.get('efficiency', 0)
        trend_char = '↗' if efficiency_trend > 0 else '↘' if efficiency_trend < 0 else '→'
        peak_efficiency = state.get('peak_efficiency', 0.0)
        
        # Calculate utilization and uptime
        utilization = converter.get('utilization', 0)
        uptime = state.get('total_uptime', 0.0)
        uptime_hours = uptime / 3600  # Convert seconds to hours
        
        # Get color based on utilization
        util_color = (
            self.status_colors['active'] if utilization > 80 else
            self.status_colors['idle'] if utilization > 40 else
            self.status_colors['error']
        )
        
        # Format stats with enhanced information
        stats = [
            f"{tier_char} T{tier} {status_char}",
            f"Eff: {efficiency:.1f}% {trend_char} (Peak: {peak_efficiency:.1f}%)",
            f"Util: {utilization:.1f}% (Up: {uptime_hours:.1f}h)",
            f"Rate: {converter.get('rate', 0):.1f}/s",
            f"Queue: {converter.get('queue_size', 0)}"
        ]
        
        # Draw each stat with appropriate color
        draw_text(surface, stats[0], x, y, size=font.get_height(), color=color)  # Tier and status
        draw_text(surface, stats[1], x, y + 1, size=font.get_height(), color=color)  # Efficiency
        draw_text(surface, stats[2], x, y + 2, size=font.get_height(), color=util_color)  # Utilization
        draw_text(surface, stats[3], x, y + 3, size=font.get_height(), color=color)  # Rate
        draw_text(surface, stats[4], x, y + 4, size=font.get_height(), color=color)  # Queue

    def _draw_node(self, surface: pygame.Surface, font: pygame.font.Font,
                   x: int, y: int, idx: int, converter: Dict[str, Any], color: Tuple[int, int, int]) -> None:
        """Draw a converter node with detailed information."""
        # Calculate dimensions
        name_width = len(converter['name']) + 4
        stats_width = max(
            len("Eff: 100.0% (Peak: 100.0%)"),
            len("Rate: 100.0/s (Max: 100.0/s)"),
            len("Queue: 100 Uptime: 100%")
        )
        width = max(name_width, stats_width) + 2
        height = 8  # Added a line for extended stats

        # Draw borders
        top_border = f"{self.chars['corner']}{self.chars['node'] * (width - 2)}{self.chars['corner']}"
        draw_text(surface, top_border, x, y, size=font.get_height(), color=color)
        
        # Draw name
        name_padding = (width - 2 - len(converter['name'])) // 2
        name_line = f"{self.chars['node']}{' ' * name_padding}{converter['name']}{' ' * name_padding}{self.chars['node']}"
        if (width - 2 - len(converter['name'])) % 2 == 1:
            name_line = f"{name_line[:-1]} {name_line[-1]}"
        draw_text(surface, name_line, x, y + 1, size=font.get_height(), color=color)

        # Draw separator
        separator = f"{self.chars['corner']}{self.chars['node'] * (width - 2)}{self.chars['corner']}"
        draw_text(surface, separator, x, y + 2, size=font.get_height(), color=color)

        # Draw stats
        self._draw_node_stats(surface, font, x + 1, y + 3, idx, converter, color)

        # Draw bottom border
        draw_text(surface, top_border, x, y + height, size=font.get_height(), color=color)
    
    def _draw_connection(self, surface: pygame.Surface, font: pygame.font.Font,
                        start_x: int, start_y: int, end_x: int, end_y: int,
                        flow_rate: Optional[float], color: Tuple[int, int, int],
                        source_converter: Optional[Dict[str, Any]] = None) -> None:
        """Draw a connection between nodes with optional flow rate and resource type.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            flow_rate: Optional flow rate between nodes
            color: Color to draw with
            source_converter: Optional source converter info for resource type
        """
        # Calculate connection points
        dx = end_x - start_x

        # Calculate vertical offset if nodes are on different rows
        dy = end_y - start_y
        needs_vertical = dy != 0

        if needs_vertical:
            # Draw vertical lines
            v_line = '|'
            mid_y = start_y + dy // 2

            # Draw first vertical segment
            for y in range(start_y + 1, mid_y):
                draw_text(surface, v_line, start_x, y, size=font.get_height(), color=color)

            # Draw second vertical segment
            for y in range(mid_y, end_y):
                draw_text(surface, v_line, end_x, y, size=font.get_height(), color=color)

            # Draw horizontal line
            h_line = self.chars['flow'] * (abs(dx) - 1)
            if dx > 0:
                prefix = self.chars['corner'] if needs_vertical else ''
                suffix = self.chars['arrow']
                x_pos = start_x + (0 if needs_vertical else 1)
            else:
                prefix = self.chars['arrow']
                suffix = self.chars['corner'] if needs_vertical else ''
                x_pos = end_x + (0 if needs_vertical else 1)

            y_pos = mid_y if needs_vertical else start_y
            line_text = prefix + h_line + suffix
            draw_text(surface, line_text, x_pos, y_pos, size=font.get_height(), color=color)

        # Draw flow information if provided
        if flow_rate is not None:
            # Get resource type and its visual indicator
            resource_type = source_converter.get('output_type', '') if source_converter else ''
            resource_char = self.chars['resource_chars'].get(resource_type.lower(), self.chars['resource_chars']['default'])

            # Format flow rate with units and resource indicator
            display_str = (
                f"{resource_char} {flow_rate/1000:.1f}k/s" if flow_rate >= 1000 else
                f"{resource_char} {flow_rate:.1f}/s"
            )

            # Add resource type name if space allows
            if len(display_str) + len(resource_type) + 2 <= abs(dx):
                display_str = f"{resource_type}: {display_str}"

            # Calculate display position
            rate_x = start_x + (dx - len(display_str)) // 2
            rate_y = mid_y - 1 if needs_vertical else start_y - 1

            # Draw with slightly dimmed color for better readability
            dimmed_color = tuple(max(0, c - 40) for c in color)
            draw_text(surface, display_str, rate_x, rate_y, size=font.get_height(), color=dimmed_color)

            # Update animation timing with speed based on flow rate
            self.last_animation_time, current_time = self.last_animation_time, time.time()
            animation_speed = min(flow_rate / 10.0, 3.0) * 8  # Scale speed with flow rate, capped for readability
            self.animation_offset = (self.animation_offset + (current_time - self.last_animation_time) * animation_speed) % 1000
            self.last_animation_time = current_time

            # Resource-specific animation patterns
            anim_patterns = {
                'energy': [
                    '⚊⚋⚌⚍⚎⚏',  # Energy pulse
                    '⚏⚎⚍⚌⚋⚊',  # Reverse pulse
                ],
                'matter': [
                    '⬡⬢⬣',     # Matter flow
                    '⬣⬢⬡',     # Reverse flow
                ],
                'fluid': [
                    '⎾⏋⏌⎿',   # Fluid wave
                    '⎿⏌⏋⎾',   # Reverse wave
                ],
                'data': [
                    '⬒⬓⬔⬕',   # Data stream
                    '⬕⬔⬓⬒',   # Reverse stream
                ]
            }

            # Get animation pattern based on resource type
            resource_type = source_converter.get('output_type', '').lower() if source_converter else ''
            pattern_set = anim_patterns.get(resource_type, [self.chars['anim_chars']])

            # Select pattern based on flow direction
            pattern_idx = 0 if dx > 0 else 1 if len(pattern_set) > 1 else 0
            pattern = pattern_set[pattern_idx]

            h_spacing = len(pattern[0]) if isinstance(pattern[0], str) else 3
            offset = int(self.animation_offset * 3)

            # Get current animation character
            pattern_pos = int(self.animation_offset * 4) % len(pattern)
            anim_char = pattern[pattern_pos]

            # Calculate animation points with improved spacing
            points = []

            if needs_vertical:
                # Calculate animation parameters
                v_spacing = 2  # Tighter vertical spacing for better visuals
                # Vertical animation with resource-specific pattern
                points.extend(
                    (x, y) for x, y_range in [
                        (start_x, range(start_y + 1, mid_y)),
                        (end_x, range(mid_y, end_y))
                    ] for y in y_range if (y + offset) % v_spacing == 0
                )

            # Horizontal animation with pattern repetition
            y_pos = mid_y if needs_vertical else start_y
            h_offset = offset % h_spacing

            # Create flowing pattern effect
            h_points = [
                (start_x + (i * h_spacing) + h_offset, y_pos)
                for i in range(abs(dx) // h_spacing)
                if start_x + (i * h_spacing) + h_offset < end_x - 1
            ]

            # Add horizontal points with pattern cycling
            for idx, point in enumerate(h_points):
                pattern_char = pattern[(pattern_pos + idx) % len(pattern)]
                draw_text(surface, pattern_char, point[0], point[1], size=font.get_height(), color=color)

            # Draw vertical points
            for point in points:
                draw_text(surface, anim_char, point[0], point[1], size=font.get_height(), color=color)
    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the chain visualization.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            
        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)
        
        if not self.converters:
            # Draw empty state
            x = self.rect.x + self.rect.width // 4
            y = self.rect.y + self.rect.height // 2
            draw_text(surface, "No converters in chain", x, y, size=font.get_height(), color=COLOR_TEXT)
            return panel_rect
        
        # Calculate layout
        margin = font.get_height() * 2
        node_spacing = font.get_height() * 4
        start_x = self.rect.x + margin
        start_y = self.rect.y + margin * 2  # Extra margin for title
        
        # Calculate optimal layout
        total_width = self.rect.width - (margin * 2)
        node_width = max(len(conv['name']) for conv in self.converters) + 8
        max_nodes_per_row = max(1, total_width // (node_width + 4))
        
        # Update animation state
        current_time = time.time()
        dt = current_time - self.last_animation_time
        self.last_animation_time = current_time
        
        # Position nodes in a grid layout and update states
        node_positions = {}  # idx -> (x, y)
        for i, conv in enumerate(self.converters):
            row = i // max_nodes_per_row
            col = i % max_nodes_per_row
            
            x = start_x + (col * (node_width + 4))
            y = start_y + (row * node_spacing)
            node_positions[i] = (x, y)
            
            # Update state tracking
            state = self.converter_states.setdefault(i, {})
            if conv.get('status') == 'active':
                state['total_uptime'] = state.get('total_uptime', 0.0) + dt
            
            # Update efficiency history
            if 'efficiency' in conv:
                history = self.efficiency_history.setdefault(i, [])
                history.append(conv['efficiency'])
                if len(history) > self.max_history_points:
                    history.pop(0)
                
                # Update peak efficiency
                state['peak_efficiency'] = max(
                    state.get('peak_efficiency', 0.0),
                    conv['efficiency']
                )
            
            # Draw node with appropriate color
            color = self._get_node_color(conv, i)
            self._draw_node(surface, font, x, y, i, conv, color)
        
        # Draw connections with improved routing and animation
        for from_idx, to_idx in self.connections:
            if from_idx in node_positions and to_idx in node_positions:
                start = node_positions[from_idx]
                end = node_positions[to_idx]
                flow_rate = self.flow_rates.get((from_idx, to_idx))
                
                # Get style-based color with alpha for active flows
                base_color = {
                    UIStyle.QUANTUM: (100, 200, 255),
                    UIStyle.SYMBIOTIC: (100, 255, 100),
                }.get(self.style, (200, 200, 200))
                
                # Calculate animation speed based on flow rate
                if flow_rate is not None:
                    animation_speed = self._calculate_animation_speed(flow_rate)
                    alpha = min(255, int(128 + (animation_speed * 127)))
                    color = tuple(min(255, c * alpha // 255) for c in base_color)
                else:
                    color = base_color
                
                # Draw connection with animated flow
                self._draw_connection(surface, font,
                                    start[0], start[1] + 6,  # Adjusted for node height
                                    end[0], end[1] + 6,
                                    flow_rate, color,
                                    self.converters[from_idx])
        
        return panel_rect


class ASCIIRecipePanel:
    """Panel for displaying and managing converter recipes."""
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Available Recipes",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a recipe panel.
        
        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        self.recipes: List[Dict[str, Any]] = []
        self.selected_idx: Optional[int] = None
        self.scroll_offset = 0
        self.max_visible_recipes = 8
        
        # Button height will be set when drawn
        self.button_height = 0
        self.start_button = ASCIIButton(
            rect.x + rect.width // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Start Recipe",
            self._on_start_recipe,
            style
        )
        self.stop_button = ASCIIButton(
            rect.x + rect.width * 3 // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Stop Recipe",
            self._on_stop_recipe,
            style
        )
    
    def set_recipes(self, recipes: List[Dict[str, Any]]) -> None:
        """Set the list of available recipes.
        
        Args:
            recipes: List of recipe dictionaries containing name, inputs, outputs,
                    and efficiency metrics
        """
        self.recipes = recipes
        self.selected_idx = 0 if recipes else None
        self.scroll_offset = 0
    
    def _on_start_recipe(self) -> None:
        """Callback for starting the selected recipe."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.recipes):
            recipe = self.recipes[self.selected_idx]
            # Trigger recipe start in the converter system
            # This would be handled by the main game logic
            print(f"Starting recipe: {recipe['name']}")
    
    def _on_stop_recipe(self) -> None:
        """Callback for stopping the current recipe."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.recipes):
            recipe = self.recipes[self.selected_idx]
            # Trigger recipe stop in the converter system
            # This would be handled by the main game logic
            print(f"Stopping recipe: {recipe['name']}")
    
    def _draw_recipe_entry(self, surface: pygame.Surface, font: pygame.font.Font,
                          x: int, y: int, recipe: Dict[str, Any], is_selected: bool) -> None:
        """Draw a single recipe entry."""
        # Draw selection indicator
        if is_selected:
            draw_text(surface, ">", x - 2, y, size=font.get_height(), color=(255, 255, 100))
        
        # Draw recipe name and base efficiency
        name_color = (100, 255, 100) if recipe.get('active', False) else COLOR_TEXT
        draw_text(surface, recipe['name'], x, y, size=font.get_height(), color=name_color)
        
        # Draw efficiency bar
        eff = recipe.get('efficiency', 0.0)
        bar_x = x + font.size(recipe['name'])[0] + 4
        bar_width = 20
        bar = '=' * int(bar_width * eff)
        bar = bar.ljust(bar_width, '.')
        
        eff_color = (100, 255, 100) if eff > 0.8 else \
                    (255, 255, 100) if eff > 0.5 else \
                    (255, 100, 100)
        draw_text(surface, f"[{bar}] {eff*100:.1f}%", bar_x, y, size=font.get_height(), color=eff_color)
        
        # Draw input/output summary
        y += font.get_height()
        inputs = ' + '.join(str(item) for item in recipe.get('inputs', []))
        outputs = ' + '.join(str(item) for item in recipe.get('outputs', []))
        draw_text(surface, f"  {inputs} -> {outputs}", x, y, size=font.get_height(), color=(160, 160, 160))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events.
        
        Returns:
            bool: True if the event was handled
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.selected_idx is not None:
                self.selected_idx = max(0, self.selected_idx - 1)
                if self.selected_idx < self.scroll_offset:
                    self.scroll_offset = self.selected_idx
                return True
            elif event.key == pygame.K_DOWN and self.selected_idx is not None:
                self.selected_idx = min(len(self.recipes) - 1, self.selected_idx + 1)
                if self.selected_idx >= self.scroll_offset + self.max_visible_recipes:
                    self.scroll_offset = self.selected_idx - self.max_visible_recipes + 1
                return True

        # Handle button events
        if self.start_button.handle_event(event):
            return True
        return bool(self.stop_button.handle_event(event))
    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the recipe panel.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            
        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)
        
        if not self.recipes:
            # Draw empty state
            x = self.rect.x + self.rect.width // 4
            y = self.rect.y + self.rect.height // 2
            draw_text(surface, "No recipes available", x, y, size=font.get_height(), color=COLOR_TEXT)
            return panel_rect
        
        # Calculate layout
        margin = font.get_height()
        x = self.rect.x + margin
        y = self.rect.y + margin * 2  # Extra margin for title
        recipe_height = font.get_height() * 2 + 1  # 2 lines + spacing
        
        # Draw visible recipes
        end_idx = min(self.scroll_offset + self.max_visible_recipes, len(self.recipes))
        for i in range(self.scroll_offset, end_idx):
            recipe = self.recipes[i]
            self._draw_recipe_entry(surface, font, x, y, recipe, i == self.selected_idx)
            y += recipe_height
        
        # Draw scroll indicators if needed
        if self.scroll_offset > 0:
            draw_text(surface, font, "^ More ^", 
                     (self.rect.centerx - 4, self.rect.y + margin),
                     COLOR_TEXT)
        if end_idx < len(self.recipes):
            draw_text(surface, "v More v", self.rect.centerx - 4, self.rect.bottom - margin * 4, size=font.get_height(), color=COLOR_TEXT)
        
        # Update and draw control buttons
        self.button_height = font.get_height() * 3
        self.start_button.y = self.rect.bottom - self.button_height
        self.stop_button.y = self.rect.bottom - self.button_height
        self.start_button.draw(surface, font)
        self.stop_button.draw(surface, font)
        
        return panel_rect


class ASCIIEfficiencyMonitor:
    """Panel for monitoring and displaying efficiency metrics over time."""
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Efficiency Monitor",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize an efficiency monitor panel.
        
        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        self.history: List[float] = []
        self.max_history = 24  # Keep 24 hours of hourly data
        self.current_efficiency = 0.0
        self.daily_average = 0.0
        self.trend_direction = 0  # -1: down, 0: stable, 1: up
        
        # Style-based characters
        style_chars = {
            UIStyle.QUANTUM: {
                'graph_bg': '·',
                'graph_point': '◆',
                'trend_up': '↗',
                'trend_down': '↘',
                'trend_stable': '→'
            },
            UIStyle.SYMBIOTIC: {
                'graph_bg': '.',
                'graph_point': '*',
                'trend_up': '↑',
                'trend_down': '↓',
                'trend_stable': '-'
            },
            UIStyle.MECHANICAL: {
                'graph_bg': '.',
                'graph_point': '#',
                'trend_up': '^',
                'trend_down': 'v',
                'trend_stable': '-'
            }
        }
        self.chars = style_chars.get(self.style, style_chars[UIStyle.MECHANICAL])
    
    def update_efficiency(self, current: float) -> None:
        """Update efficiency metrics with a new value.
        
        Args:
            current: Current efficiency value (0.0 to 1.0)
        """
        self.current_efficiency = max(0.0, min(1.0, current))
        
        # Update history
        self.history.append(current)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Calculate daily average
        self.daily_average = sum(self.history) / len(self.history)
        
        # Calculate trend
        if len(self.history) >= 2:
            recent_change = self.history[-1] - self.history[-2]
            self.trend_direction = (
                1 if recent_change > 0.05 else
                -1 if recent_change < -0.05 else
                0
            )
    
    def _draw_graph(self, surface: pygame.Surface, font: pygame.font.Font,
                    x: int, y: int, width: int, height: int) -> None:
        """Draw efficiency history graph.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x, y: Position to draw at
            width, height: Size of the graph area
        """
        # Draw background grid
        for row in range(height):
            line = self.chars['graph_bg'] * width
            draw_text(surface, line, x, y + row * font.get_height(), size=font.get_height(), color=(60, 60, 60))
        
        # Draw data points
        if self.history:
            max_val = max(self.history)
            min_val = min(self.history)
            val_range = max(0.001, max_val - min_val)  # Avoid division by zero
            
            for i, val in enumerate(self.history):
                # Calculate position
                px = x + int(i * width / len(self.history))
                py = y + int((1 - (val - min_val) / val_range) * (height - 1)) * font.get_height()
                
                # Draw point
                color = (
                    (100, 255, 100) if val > 0.8 else
                    (255, 255, 100) if val > 0.5 else
                    (255, 100, 100)
                )
                draw_text(surface, self.chars['graph_point'], px, py, size=font.get_height(), color=color)
    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the efficiency monitor panel.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            
        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)
        
        # Calculate layout
        margin = font.get_height()
        content_x = self.rect.x + margin
        content_y = self.rect.y + margin * 2  # Extra margin for title
        content_width = self.rect.width - margin * 2
        
        # Draw current efficiency
        eff_color = (
            (100, 255, 100) if self.current_efficiency > 0.8 else
            (255, 255, 100) if self.current_efficiency > 0.5 else
            (255, 100, 100)
        )
        current_text = f"Current: {self.current_efficiency*100:.1f}%"
        draw_text(surface, current_text, content_x, content_y, size=font.get_height(), color=eff_color)
        
        # Draw trend indicator
        trend_char = {
            1: self.chars['trend_up'],
            0: self.chars['trend_stable'],
            -1: self.chars['trend_down']
        }[self.trend_direction]
        trend_color = {
            1: (100, 255, 100),
            0: (200, 200, 200),
            -1: (255, 100, 100)
        }[self.trend_direction]
        trend_x = content_x + font.size(current_text)[0] + margin
        draw_text(surface, trend_char, trend_x, content_y, size=font.get_height(), color=trend_color)
        
        # Draw daily average
        avg_y = content_y + margin
        draw_text(surface, f"24h Average: {self.daily_average*100:.1f}%",
                 content_x, avg_y, size=font.get_height(), color=(200, 200, 200))
        
        # Draw efficiency history graph
        graph_y = avg_y + margin * 2
        graph_height = 8  # lines
        self._draw_graph(
            surface, font,
            content_x, graph_y,
            content_width - margin,
            graph_height
        )
        
        return panel_rect


class ASCIIChainBuilder:
    """Interface for step-by-step creation of converter chains."""
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Chain Builder",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a chain builder interface.
        
        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        
        # Chain building state
        self.available_converters: List[Dict[str, Any]] = []
        self.selected_converters: List[Dict[str, Any]] = []
        self.connections: List[Tuple[int, int]] = []  # (from_idx, to_idx)
        self.selected_idx: Optional[int] = None
        self.connecting_from: Optional[int] = None
        
        # UI state
        self.scroll_offset = 0
        self.max_visible_items = 8
        self.mode = 'select'  # 'select' or 'connect'
        
        # Create control buttons with initial positions
        self.mode_button = ASCIIButton(
            rect.x + rect.width // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Switch Mode",
            self._toggle_mode,
            style
        )
        self.save_button = ASCIIButton(
            rect.x + rect.width * 3 // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Save Chain",
            self._save_chain,
            style
        )
    
    def set_available_converters(self, converters: List[Dict[str, Any]]) -> None:
        """Set the list of available converters.
        
        Args:
            converters: List of converter info dictionaries
        """
        self.available_converters = converters
        self.selected_idx = 0 if converters else None
        self.scroll_offset = 0
    
    def _toggle_mode(self) -> None:
        """Toggle between selection and connection modes."""
        self.mode = 'connect' if self.mode == 'select' else 'select'
        self.selected_idx = None
        self.connecting_from = None
        self.mode_button.text = "Connect Mode" if self.mode == 'select' else "Select Mode"
    
    def _save_chain(self) -> None:
        """Save the current chain configuration."""
        if self.selected_converters and self.connections:
            print(f"Saving chain with {len(self.selected_converters)} converters")
            # This would be handled by the main game logic
    
    def _check_compatibility(self, from_conv: Dict[str, Any], to_conv: Dict[str, Any]) -> bool:
        """Check if two converters can be connected.
        
        Args:
            from_conv: Source converter info
            to_conv: Target converter info
            
        Returns:
            bool: True if converters are compatible
        """
        # This is a simplified check - the actual implementation would need to
        # verify resource types, throughput compatibility, etc.
        outputs = from_conv.get('outputs', {})
        inputs = to_conv.get('inputs', {})
        return bool(set(outputs) & set(inputs))
    
    def _draw_converter_entry(self, surface: pygame.Surface, font: pygame.font.Font,
                             x: int, y: int, converter: Dict[str, Any],
                             is_selected: bool, is_connecting: bool) -> None:
        """Draw a single converter entry."""
        # Draw selection/connection indicator
        if is_selected:
            draw_text(surface, ">", x - 2, y, size=font.get_height(), color=(255, 255, 100))
        elif is_connecting:
            draw_text(surface, "*", x - 2, y, size=font.get_height(), color=(100, 255, 100))
        
        # Draw converter name and type
        draw_text(surface, converter['name'], x, y, size=font.get_height(), color=COLOR_TEXT)
        
        # Draw input/output summary
        y += font.get_height()
        inputs = ' + '.join(f"{amt}{res}" for res, amt in converter.get('inputs', {}).items())
        outputs = ' + '.join(f"{amt}{res}" for res, amt in converter.get('outputs', {}).items())
        draw_text(surface, f"  {inputs} -> {outputs}", x, y, size=font.get_height(), color=(160, 160, 160))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events.
        
        Returns:
            bool: True if the event was handled
        """
        if event.type == pygame.KEYDOWN:
            if self.mode == 'select':
                if event.key == pygame.K_UP and self.selected_idx is not None:
                    self.selected_idx = max(0, self.selected_idx - 1)
                    if self.selected_idx < self.scroll_offset:
                        self.scroll_offset = self.selected_idx
                    return True
                elif event.key == pygame.K_DOWN and self.selected_idx is not None:
                    max_idx = len(self.available_converters) - 1
                    self.selected_idx = min(max_idx, self.selected_idx + 1)
                    if self.selected_idx >= self.scroll_offset + self.max_visible_items:
                        self.scroll_offset = self.selected_idx - self.max_visible_items + 1
                    return True
                elif event.key == pygame.K_RETURN and self.selected_idx is not None:
                    # Add selected converter to chain
                    converter = self.available_converters[self.selected_idx]
                    if converter not in self.selected_converters:
                        self.selected_converters.append(converter)
                    return True
            elif event.key == pygame.K_UP and self.selected_idx is not None:
                self.selected_idx = max(0, self.selected_idx - 1)
                return True
            elif event.key == pygame.K_DOWN and self.selected_idx is not None:
                max_idx = len(self.selected_converters) - 1
                self.selected_idx = min(max_idx, self.selected_idx + 1)
                return True
            elif event.key == pygame.K_RETURN and self.selected_idx is not None:
                if self.connecting_from is None:
                    self.connecting_from = self.selected_idx
                else:
                    # Try to connect converters
                    from_conv = self.selected_converters[self.connecting_from]
                    to_conv = self.selected_converters[self.selected_idx]
                    if self._check_compatibility(from_conv, to_conv):
                        self.connections.append((self.connecting_from, self.selected_idx))
                    self.connecting_from = None
                return True

        # Handle button events
        if self.mode_button.handle_event(event):
            return True
        return bool(self.save_button.handle_event(event))
    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the chain builder interface.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            
        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)
        
        # Calculate layout
        margin = font.get_height()
        content_x = self.rect.x + margin
        content_y = self.rect.y + margin * 2  # Extra margin for title
        
        # Draw mode indicator
        mode_text = "Mode: " + ("Connect" if self.mode == 'connect' else "Select")
        draw_text(surface, font, mode_text, (content_x, content_y), COLOR_TEXT)
        
        # Draw converter list
        list_y = content_y + margin * 2
        
        converters = (
            self.selected_converters if self.mode == 'connect'
            else self.available_converters
        )
        
        if not converters:
            draw_text(surface, font,
                     "No converters available" if self.mode == 'select'
                     else "No converters in chain",
                     (content_x + margin, list_y), COLOR_TEXT)
        else:
            # Draw visible converters
            converter_height = font.get_height() * 2 + 1  # 2 lines + spacing
            end_idx = min(self.scroll_offset + self.max_visible_items, len(converters))
            
            for i in range(self.scroll_offset, end_idx):
                converter = converters[i]
                y = list_y + (i - self.scroll_offset) * converter_height
                self._draw_converter_entry(
                    surface, font, content_x + margin, y,
                    converter,
                    i == self.selected_idx,
                    i == self.connecting_from
                )
        
        # Draw connection preview in connect mode
        if self.mode == 'connect' and self.connecting_from is not None:
            preview_y = list_y + converter_height * len(converters) + margin
            from_conv = self.selected_converters[self.connecting_from]
            draw_text(surface, font,
                     f"Connecting from: {from_conv['name']}",
                     (content_x + margin, preview_y), (100, 255, 100))
        
        # Draw scroll indicators if needed
        if self.scroll_offset > 0:
            draw_text(surface, "^ More ^", self.rect.centerx - 4, self.rect.y + margin, size=font.get_height(), color=COLOR_TEXT)
        if end_idx < len(converters):
            draw_text(surface, "v More v", self.rect.centerx - 4, self.rect.bottom - margin * 4, size=font.get_height(), color=COLOR_TEXT)
        
        # Update and draw control buttons
        button_height = font.get_height() * 3
        self.mode_button.y = self.rect.bottom - button_height
        self.save_button.y = self.rect.bottom - button_height
        self.mode_button.draw(surface, font)
        self.save_button.draw(surface, font)
        
        return panel_rect


class ASCIIChainTemplateManager:
    """Interface for managing converter chain templates."""
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Chain Templates",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a chain template manager interface.
        
        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        
        # Template data
        self.templates: List[Dict[str, Any]] = []
        self.selected_idx: Optional[int] = None
        self.scroll_offset = 0
        self.max_visible_items = 8
        
        # Create control buttons
        self.save_button = ASCIIButton(
            rect.x + rect.width // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Save Template",
            self._save_template,
            style
        )
        self.load_button = ASCIIButton(
            rect.x + rect.width * 3 // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Load Template",
            self._load_template,
            style
        )
    
    def set_templates(self, templates: List[Dict[str, Any]]) -> None:
        """Set the list of available templates.
        
        Args:
            templates: List of template info dictionaries
        """
        self.templates = templates
        self.selected_idx = 0 if templates else None
        self.scroll_offset = 0
    
    def _save_template(self) -> None:
        """Save the current chain as a template."""
        # This would be handled by the main game logic
        pass
    
    def _load_template(self) -> None:
        """Load the selected template."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.templates):
            template = self.templates[self.selected_idx]
            print(f"Loading template: {template['name']}")
            # This would be handled by the main game logic
    
    def _draw_template_entry(self, surface: pygame.Surface, font: pygame.font.Font,
                           x: int, y: int, template: Dict[str, Any],
                           is_selected: bool) -> None:
        """Draw a single template entry."""
        # Draw selection indicator
        if is_selected:
            draw_text(surface, ">", x - 2, y, size=font.get_height(), color=(255, 255, 100))
        
        # Draw template name and details
        draw_text(surface, template['name'], x, y, size=font.get_height(), color=COLOR_TEXT)
        
        # Draw template info
        y += font.get_height()
        info = f"  {len(template['converters'])} converters, {len(template['connections'])} connections"
        draw_text(surface, info, x, y, size=font.get_height(), color=(160, 160, 160))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events.
        
        Returns:
            bool: True if the event was handled
        """
        if event.type == pygame.KEYDOWN and self.selected_idx is not None:
            if event.key == pygame.K_UP:
                self.selected_idx = max(0, self.selected_idx - 1)
                if self.selected_idx < self.scroll_offset:
                    self.scroll_offset = self.selected_idx
                return True
            if event.key == pygame.K_DOWN:
                max_idx = len(self.templates) - 1
                self.selected_idx = min(max_idx, self.selected_idx + 1)
                if self.selected_idx >= self.scroll_offset + self.max_visible_items:
                    self.scroll_offset = self.selected_idx - self.max_visible_items + 1
                return True
            if event.key == pygame.K_RETURN:
                self._load_template()
                return True
        
        # Handle button events
        return (
            self.save_button.handle_event(event) or
            self.load_button.handle_event(event)
        )
    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the template manager interface.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            
        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)
        
        # Calculate layout
        margin = font.get_height()
        content_x = self.rect.x + margin
        content_y = self.rect.y + margin * 2  # Extra margin for title
        
        # Draw template list
        list_y = content_y + margin
        
        if self.templates:
            # Draw visible templates
            template_height = font.get_height() * 2 + 1  # 2 lines + spacing
            end_idx = min(self.scroll_offset + self.max_visible_items, len(self.templates))
            
            for i in range(self.scroll_offset, end_idx):
                template = self.templates[i]
                y = list_y + (i - self.scroll_offset) * template_height
                self._draw_template_entry(
                    surface, font, content_x + margin, y,
                    template, i == self.selected_idx
                )
        else:
            draw_text(surface, "No templates available", content_x + margin, list_y, size=font.get_height(), color=COLOR_TEXT)
        
        # Draw scroll indicators if needed
        if self.scroll_offset > 0:
            draw_text(surface, "^ More ^", self.rect.centerx - 4, self.rect.y + margin, size=font.get_height(), color=COLOR_TEXT)
        if end_idx < len(self.templates):
            draw_text(surface, "v More v", self.rect.centerx - 4, self.rect.bottom - margin * 4, size=font.get_height(), color=COLOR_TEXT)
        
        # Update and draw control buttons
        button_height = font.get_height() * 3
        self.save_button.y = self.rect.bottom - button_height
        self.load_button.y = self.rect.bottom - button_height
        self.save_button.draw(surface, font)
        self.load_button.draw(surface, font)
        
        return panel_rect


class ASCIITemplateListView:
    """Interface for displaying and managing a list of chain templates."""
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Template List",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a template list view interface.
        
        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        
        # Template data
        self.templates: List[Dict[str, Any]] = []
        self.selected_idx: Optional[int] = None
        self.scroll_offset = 0
        self.max_visible_items = 10
        
        # Create control buttons
        self.delete_button = ASCIIButton(
            rect.x + rect.width // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Delete Template",
            self._delete_template,
            style
        )
        self.duplicate_button = ASCIIButton(
            rect.x + rect.width * 3 // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Duplicate Template",
            self._duplicate_template,
            style
        )
    
    def set_templates(self, templates: List[Dict[str, Any]]) -> None:
        """Set the list of available templates.
        
        Args:
            templates: List of template info dictionaries
        """
        self.templates = templates
        self.selected_idx = 0 if templates else None
        self.scroll_offset = 0
    
    def _delete_template(self) -> None:
        """Delete the selected template."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.templates):
            template = self.templates[self.selected_idx]
            print(f"Deleting template: {template['name']}")
            # This would be handled by the main game logic
    
    def _duplicate_template(self) -> None:
        """Create a copy of the selected template."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.templates):
            template = self.templates[self.selected_idx]
            print(f"Duplicating template: {template['name']}")
            # This would be handled by the main game logic
    
    def _draw_template_entry(self, surface: pygame.Surface, font: pygame.font.Font,
                           x: int, y: int, template: Dict[str, Any],
                           is_selected: bool) -> None:
        """Draw a single template entry."""
        # Draw selection indicator
        if is_selected:
            draw_text(surface, ">", x - 2, y, size=font.get_height(), color=(255, 255, 100))
        
        # Draw template name
        draw_text(surface, template['name'], x, y, size=font.get_height(), color=COLOR_TEXT)
        
        # Draw template details
        y += font.get_height()
        details = f"  Created: {template.get('created_at', 'Unknown')}"
        draw_text(surface, details, x, y, size=font.get_height(), color=(160, 160, 160))
        
        # Draw template stats
        y += font.get_height()
        stats = f"  Converters: {len(template['converters'])}, Efficiency: {template.get('efficiency', 0.0):.1%}"
        draw_text(surface, stats, x, y, size=font.get_height(), color=(160, 160, 160))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events.
        
        Returns:
            bool: True if the event was handled
        """
        if event.type == pygame.KEYDOWN and self.selected_idx is not None:
            if event.key == pygame.K_UP:
                self.selected_idx = max(0, self.selected_idx - 1)
                if self.selected_idx < self.scroll_offset:
                    self.scroll_offset = self.selected_idx
                return True
            if event.key == pygame.K_DOWN:
                max_idx = len(self.templates) - 1
                self.selected_idx = min(max_idx, self.selected_idx + 1)
                if self.selected_idx >= self.scroll_offset + self.max_visible_items:
                    self.scroll_offset = self.selected_idx - self.max_visible_items + 1
                return True
        
        # Handle button events
        return (
            self.delete_button.handle_event(event) or
            self.duplicate_button.handle_event(event)
        )
    
    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the template list interface.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            
        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)
        
        # Calculate layout
        margin = font.get_height()
        content_x = self.rect.x + margin
        content_y = self.rect.y + margin * 2  # Extra margin for title
        
        # Draw template list
        list_y = content_y + margin
        
        if self.templates:
            # Draw visible templates
            template_height = font.get_height() * 3 + 2  # 3 lines + spacing
            end_idx = min(self.scroll_offset + self.max_visible_items, len(self.templates))
            
            for i in range(self.scroll_offset, end_idx):
                template = self.templates[i]
                y = list_y + (i - self.scroll_offset) * template_height
                self._draw_template_entry(
                    surface, font, content_x + margin, y,
                    template, i == self.selected_idx
                )
        else:
            draw_text(surface, "No templates available", content_x + margin, list_y, size=font.get_height(), color=COLOR_TEXT)
        
        # Draw scroll indicators if needed
        if self.scroll_offset > 0:
            draw_text(surface, "^ More ^", self.rect.centerx - 4, self.rect.y + margin, size=font.get_height(), color=COLOR_TEXT)
        if end_idx < len(self.templates):
            draw_text(surface, "v More v", self.rect.centerx - 4, self.rect.bottom - margin * 4, size=font.get_height(), color=COLOR_TEXT)
        
        # Update and draw control buttons
        button_height = font.get_height() * 3
        self.delete_button.y = self.rect.bottom - button_height
        self.duplicate_button.y = self.rect.bottom - button_height
        self.delete_button.draw(surface, font)
        self.duplicate_button.draw(surface, font)
        
        return panel_rect


def draw_ascii_table(
    surface: pygame.Surface,
    x: int,
    y: int,
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[int]] = None,
    font: Optional[pygame.font.Font] = None,
    color: Tuple[int, int, int] = COLOR_TEXT,
    border_style: str = "single",
) -> pygame.Rect:
    """
    Draw an ASCII-styled table.

    Args:
        surface: Surface to draw on
        x, y: Position coordinates
        headers: Column headers
        rows: Table data rows
        col_widths: Optional fixed column widths
        font: Font to use (or None to use default)
        color: Text color
        border_style: Border style ('single', 'double', 'heavy')

    Returns:
        pygame.Rect: Bounding rectangle of the table
    """
    if not font:
        try:
            font = pygame.font.SysFont(
                "Courier New", 16
            )  # Monospace font works best for ASCII art
        except Exception:
            font = pygame.font.Font(None, 16)

    # Determine column widths if not provided
    if not col_widths:
        col_widths = []
        for i in range(len(headers)):
            # Calculate max width needed for this column
            col_width = len(headers[i])
            for row in rows:
                if i < len(row):
                    col_width = max(col_width, len(row[i]))
            col_widths.append(col_width + 2)  # Add padding

    # Border characters based on style
    borders = {
        "single": {
            "tl": "+",
            "tr": "+",
            "bl": "+",
            "br": "+",
            "h": "-",
            "v": "|",
            "lc": "+",
            "rc": "+",
            "tc": "+",
            "bc": "+",
        },
        "double": {
            "tl": "╔",
            "tr": "╗",
            "bl": "╚",
            "br": "╝",
            "h": "═",
            "v": "║",
            "lc": "╠",
            "rc": "╣",
            "tc": "╦",
            "bc": "╩",
        },
        "heavy": {
            "tl": "┏",
            "tr": "┓",
            "bl": "┗",
            "br": "┛",
            "h": "━",
            "v": "┃",
            "lc": "┣",
            "rc": "┫",
            "tc": "┳",
            "bc": "┻",
        },
    }

    b = borders.get(border_style, borders["single"])

    # Top border
    top_border = b["tl"]
    for i, width in enumerate(col_widths):
        top_border += b["h"] * width
        top_border += b["tc"] if i < len(col_widths) - 1 else b["tr"]
    table_strings = [top_border]
    # Headers
    header_row = b["v"]
    for i, header in enumerate(headers):
        header_row += header.ljust(col_widths[i]) + b["v"]
    table_strings.append(header_row)

    # Separator
    separator = b["lc"]
    for i, width in enumerate(col_widths):
        separator += b["h"] * width
        separator += b["rc"] if i < len(col_widths) - 1 else b["rc"]
    table_strings.append(separator)

    # Data rows
    for row in rows:
        data_row = b["v"]
        for i in range(len(col_widths)):
            cell = row[i] if i < len(row) else ""
            data_row += cell.ljust(col_widths[i]) + b["v"]
        table_strings.append(data_row)

    # Bottom border
    bottom_border = b["bl"]
    for i, width in enumerate(col_widths):
        bottom_border += b["h"] * width
        bottom_border += b["bc"] if i < len(col_widths) - 1 else b["br"]
    table_strings.append(bottom_border)

    # Render all table strings
    char_height = font.size("X")[1]
    table_rect = None

    for i, line in enumerate(table_strings):
        line_rect = draw_text(
            surface, line, x, y + i * char_height, size=font.get_height(), color=color
        )

        table_rect = table_rect.union(line_rect) if table_rect else line_rect
    return table_rect or pygame.Rect(x, y, 0, 0)
