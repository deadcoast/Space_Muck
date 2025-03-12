import time
import math
import logging
import pygame
from enum import Enum, auto
from typing import Dict, Optional, Tuple

# Define color constants
COLOR_TEXT = (200, 200, 200)  # Light gray
COLOR_BG = (10, 10, 10)  # Near black
COLOR_HIGHLIGHT = (255, 255, 100)  # Yellow-ish highlight


class UIStyle(Enum):
    """Enum for UI visual styles."""

    SYMBIOTIC = auto()  # Organic, flowing style
    ASTEROID = auto()  # Rocky, rough style
    MECHANICAL = auto()  # Industrial, mechanical style
    QUANTUM = auto()  # High-tech, digital style
    FLEET = auto()  # Military, utilitarian style

    @classmethod
    def get_border_chars(cls, style: "UIStyle") -> Dict[str, str]:
        """Get border characters for the specified UI style."""
        try:
            # Use style-specific border characters
            if style == UIStyle.SYMBIOTIC:
                return {
                    "top_left": "╭",
                    "top_right": "╮",
                    "bottom_left": "╰",
                    "bottom_right": "╯",
                    "horizontal": "─",
                    "vertical": "│",
                }
            elif style in (UIStyle.ASTEROID, UIStyle.FLEET):
                # Simple single-line border for asteroid and fleet styles
                return {
                    "top_left": "┌",
                    "top_right": "┐",
                    "bottom_left": "└",
                    "bottom_right": "┘",
                    "horizontal": "─",
                    "vertical": "│",
                }
            elif style == UIStyle.MECHANICAL:
                # Double-line border for mechanical style
                return {
                    "top_left": "╔",
                    "top_right": "╗",
                    "bottom_left": "╚",
                    "bottom_right": "╝",
                    "horizontal": "═",
                    "vertical": "║",
                }
            elif style == UIStyle.QUANTUM:
                # Bold single-line border for quantum style
                return {
                    "top_left": "┏",
                    "top_right": "┓",
                    "bottom_left": "┗",
                    "bottom_right": "┛",
                    "horizontal": "━",
                    "vertical": "┃",
                }
            else:
                return {
                    "top_left": "+",
                    "top_right": "+",
                    "bottom_left": "+",
                    "bottom_right": "+",
                    "horizontal": "-",
                    "vertical": "|",
                }
        except Exception as e:
            logging.error(f"Error getting border chars: {e}")
            return {
                "top_left": "+",
                "top_right": "+",
                "bottom_left": "+",
                "bottom_right": "+",
                "horizontal": "-",
                "vertical": "|",
            }


class AnimationStyle(Enum):
    """Enum for animation styles."""

    NONE = auto()  # No animation
    FADE_IN = auto()  # Fade in from transparent
    FADE_OUT = auto()  # Fade out to transparent
    SLIDE_IN = auto()  # Slide in from edge
    SLIDE_OUT = auto()  # Slide out to edge
    PULSE = auto()  # Pulsing animation
    BLINK = auto()  # Blinking animation

    @classmethod
    def get_animation_style(cls, ui_style: UIStyle) -> "AnimationStyle":
        """Get the appropriate animation style for a UI style."""
        try:
            if ui_style == UIStyle.SYMBIOTIC:
                return AnimationStyle.FADE_IN
            elif ui_style == UIStyle.ASTEROID:
                return AnimationStyle.SLIDE_IN
            elif ui_style == UIStyle.MECHANICAL:
                return AnimationStyle.NONE
            elif ui_style == UIStyle.QUANTUM:
                return AnimationStyle.PULSE
            elif ui_style == UIStyle.FLEET:
                return AnimationStyle.SLIDE_IN
            else:
                return AnimationStyle.NONE
        except Exception as e:
            logging.error(f"Error getting animation style: {e}")
            return AnimationStyle.NONE


class UIElement:
    """Base class for all UI elements."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        style: UIStyle,
        title: Optional[str] = None,
    ):
        """Initialize a UI element."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.style = style
        self.title = title
        self.visible = True
        self.hover = False

        # Animation state
        self.animation = {
            "active": False,
            "style": AnimationStyle.NONE,
            "progress": 0.0,
            "duration": 0.0,
            "start_time": 0.0,
        }

        # Initialize border characters based on style
        self.border_chars = UIStyle.get_border_chars(style)

    def start_animation(self, duration: float = 0.5) -> None:
        """Start an animation sequence."""
        try:
            # Get animation style based on UI style
            animation_style = AnimationStyle.get_animation_style(self.style)

            # Set animation state
            self.animation = {
                "active": True,
                "style": animation_style,
                "progress": 0.0,
                "duration": duration,
                "start_time": time.time(),
            }
        except Exception as e:
            logging.error(f"Error starting animation: {e}")

    def update_animation(self, delta_time: float = 0.0) -> bool:
        """Update animation state based on elapsed time."""
        try:
            return self._process_animation_progress(delta_time)
        except Exception as e:
            logging.error(f"Error updating animation: {e}")
            self.animation["active"] = False
            return False

    def _process_animation_progress(self, delta_time):
        if not self.animation["active"]:
            return False

        # Calculate elapsed time
        if delta_time <= 0:
            current_time = time.time()
            elapsed = current_time - self.animation["start_time"]
        else:
            elapsed = delta_time

        # Update progress
        duration = max(0.001, self.animation["duration"])
        self.animation["progress"] = min(1.0, elapsed / duration)

        # Check if animation is complete
        if self.animation["progress"] >= 1.0:
            self.animation["active"] = False
            return False

        return True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the UI element."""
        if not self.visible:
            return

        try:
            # Draw border and background
            self._draw_border(surface, font)

            # Draw title if provided
            if self.title:
                self._draw_title(surface, font)

        except Exception as e:
            logging.error(f"Error drawing UI element: {e}")

    def _draw_border(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the border of the UI element."""
        try:
            # Get border characters
            tl = self.border_chars["top_left"]
            tr = self.border_chars["top_right"]
            bl = self.border_chars["bottom_left"]
            br = self.border_chars["bottom_right"]
            h = self.border_chars["horizontal"]
            v = self.border_chars["vertical"]

            # Draw corners
            self._draw_char(surface, font, self.x, self.y, tl)
            self._draw_char(surface, font, self.x + self.width - 1, self.y, tr)
            self._draw_char(surface, font, self.x, self.y + self.height - 1, bl)
            self._draw_char(
                surface, font, self.x + self.width - 1, self.y + self.height - 1, br
            )

            # Draw horizontal borders
            for i in range(1, self.width - 1):
                self._draw_char(surface, font, self.x + i, self.y, h)
                self._draw_char(surface, font, self.x + i, self.y + self.height - 1, h)

            # Draw vertical borders
            for i in range(1, self.height - 1):
                self._draw_char(surface, font, self.x, self.y + i, v)
                self._draw_char(surface, font, self.x + self.width - 1, self.y + i, v)

            # Fill background
            for i in range(1, self.width - 1):
                for j in range(1, self.height - 1):
                    self._draw_char(surface, font, self.x + i, self.y + j, " ")

        except Exception as e:
            logging.error(f"Error drawing border: {e}")

    def _draw_title(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the title of the UI element."""
        try:
            # Ensure title fits within element width
            max_title_width = self.width - 4
            title = self.title

            if len(title) > max_title_width:
                title = f"{title[:max_title_width - 3]}..."

            # Calculate title position (centered)
            title_x = self.x + (self.width // 2) - (len(title) // 2)
            title_y = self.y

            # Draw title
            self._draw_text(surface, font, title_x, title_y, title)

        except Exception as e:
            logging.error(f"Error drawing title: {e}")

    def _draw_char(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        char: str,
        color: Tuple[int, int, int] = COLOR_TEXT,
    ) -> None:
        """Draw a single character."""
        try:
            # Skip if character is empty
            if not char:
                return

            # Calculate pixel position
            char_width, char_height = font.size(" ")
            pixel_x = x * char_width
            pixel_y = y * char_height

            # Apply animation effects if active
            if self.animation["active"]:
                color = self._apply_animation_effect(color)

            # Render character
            char_surface = font.render(char, True, color)
            surface.blit(char_surface, (pixel_x, pixel_y))

        except Exception as e:
            logging.error(f"Error drawing character: {e}")

    def _draw_text(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int] = COLOR_TEXT,
    ) -> None:
        """Draw a text string."""
        try:
            # Draw each character individually
            for i, char in enumerate(text):
                self._draw_char(surface, font, x + i, y, char, color)

        except Exception as e:
            logging.error(f"Error drawing text: {e}")

    def _apply_animation_effect(
        self, color: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Apply animation effect to a color based on current animation state."""
        try:
            if not self.animation["active"]:
                return color

            r, g, b = color
            progress = self.animation["progress"]

            if self.animation["style"] == AnimationStyle.FADE_IN:
                # Fade in: increase alpha from 0 to 1
                factor = progress
                return (int(r * factor), int(g * factor), int(b * factor))

            elif self.animation["style"] == AnimationStyle.FADE_OUT:
                # Fade out: decrease alpha from 1 to 0
                factor = 1.0 - progress
                return (int(r * factor), int(g * factor), int(b * factor))

            elif self.animation["style"] == AnimationStyle.PULSE:
                # Pulse: oscillate brightness
                factor = 0.7 + 0.3 * math.sin(progress * math.pi * 2)
                return (
                    min(255, int(r * factor)),
                    min(255, int(g * factor)),
                    min(255, int(b * factor)),
                )

            elif self.animation["style"] == AnimationStyle.BLINK:
                # Blink: on/off based on progress
                blink_state = (progress * 4) % 1.0 > 0.5
                factor = 1.0 if blink_state else 0.3
                return (int(r * factor), int(g * factor), int(b * factor))

            else:
                return color

        except Exception as e:
            logging.error(f"Error applying animation effect: {e}")
            return color

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is within the UI element's bounds."""
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def handle_input(self) -> Optional[str]:
        """Handle input for the UI element.
        
        Returns:
            Optional[str]: Response from handling input, if any
        """
        return None
