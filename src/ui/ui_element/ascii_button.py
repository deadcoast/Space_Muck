import logging
from typing import Tuple, Optional, Callable, TypeVar, cast
import pygame
import time
import math

from ui.draw_utils import draw_text
from ui.ascii_base import UIStyle, UIElement
from config import COLOR_TEXT

# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height

class ASCIIButton(UIElement):
    """Interactive button with ASCII styling.

    This class provides a clickable button with customizable styling based on the
    selected UIStyle. It supports hover effects, animations, and callback functions.

    Inherits from UIElement to leverage standardized animation framework and styling.
    """

    def __init__(
        self,
        x: int,
        y: int,
        text: str,
        callback: Optional[Callable[[], None]] = None,
        style: UIStyle = UIStyle.MECHANICAL,
        converter_type: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize an ASCII button.

        Args:
            x: X position
            y: Y position
            text: Button text
            callback: Function to call when clicked
            style: Visual style for the button
            converter_type: Optional converter type to determine style
            enabled: Whether the button is initially enabled
        """
        # Determine the style based on converter type if provided
        actual_style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )

        # Calculate width and height based on text length
        width = len(text) + 4  # Add padding for button characters
        height = 3  # Standard button height

        # Initialize the parent UIElement class
        super().__init__(x, y, width, height, actual_style, text)

        # ASCIIButton specific attributes
        self.callback = callback
        self.hover = False
        self.enabled = enabled
        self._set_button_chars()
        self.rect = pygame.Rect(0, 0, 0, 0)  # Will be set properly when drawn

    def _set_button_chars(self) -> None:
        """Set button characters based on UI style."""
        style_chars = {
            UIStyle.QUANTUM: ("◧", "◨"),  # ◧, ◨
            UIStyle.SYMBIOTIC: ("▐", "▌"),  # ▐, ▌
            UIStyle.MECHANICAL: ("[", "]"),
            UIStyle.ASTEROID: ("◄", "►"),  # ◄, ►
            UIStyle.FLEET: ("◀", "▶"),  # ◀, ▶
        }
        self.prefix, self.suffix = style_chars.get(self.style, ("<", ">"))

    def is_hover(self, mouse_pos: Tuple[int, int]) -> bool:
        """Check if the mouse is hovering over this button."""
        return self.rect.collidepoint(mouse_pos)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame events for this button.

        Args:
            event: The pygame event to handle

        Returns:
            bool: True if the event was consumed
        """
        try:
            if not self.enabled:
                return False

            if event.type == pygame.MOUSEMOTION:
                was_hover = self.hover
                self.hover = self.is_hover(event.pos)

                # Start hover animation when mouse enters the button
                if self.hover and not was_hover:
                    self._button_hover()
                return self.hover

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.is_hover(event.pos)
                and self.callback is not None
            ):
                self._button_hover()
                self.animation["duration"] = 0.2

                # Execute callback with error handling
                try:
                    self.callback()
                except Exception as e:
                    logging.error(f"Error executing button callback: {e}")
                return True

            return False
        except Exception as e:
            logging.error(f"Error handling button event: {e}")
            return False

    def _button_hover(self):
        """Start button hover animation with error handling"""
        try:
            # Use the standardized animation framework from UIElement
            # but with a shorter duration for hover effects
            self.start_animation(0.2)
            # Store that this is a hover animation
            self.animation["style_data"]["hover"] = True
        except Exception as e:
            logging.error(f"Error starting button hover animation: {e}")

    def _get_hover_color(
        self, color: Tuple[int, int, int], hover_color: Optional[Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
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
        """Calculate alpha value based on animation state with error handling."""
        if not self.animation["active"]:
            return 255

        try:
            return self._time_manager()
        except Exception as e:
            logging.error(f"Error calculating animation alpha: {e}")
            self.animation["active"] = False
            return 255

    def _time_manager(self):
        current_time = time.time()
        elapsed = current_time - self.animation["start_time"]

        # Avoid division by zero
        duration = max(0.001, self.animation["duration"])
        progress = min(1.0, elapsed / duration)

        if progress < 1.0:
            # Use a smoother sine wave animation based on style
            if self.style == UIStyle.QUANTUM:
                # Faster oscillation for quantum style
                return int(255 * (0.7 + 0.3 * math.sin(progress * math.pi * 6)))
            elif self.style == UIStyle.SYMBIOTIC:
                # Organic pulsing for symbiotic style
                return int(255 * (0.8 + 0.2 * math.sin(progress * math.pi * 3)))
            else:
                # Default animation
                return int(255 * (0.7 + 0.3 * math.sin(progress * math.pi * 4)))

        self.animation["active"] = False
        return 255

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        color: Color = COLOR_TEXT,
        hover_color: Optional[Color] = None,
        disabled_color: Optional[Color] = None,
    ) -> pygame.Rect:
        """
        Draw the button.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            color: Normal text color
            hover_color: Color when hovering
            disabled_color: Color when button is disabled

        Returns:
            pygame.Rect: The drawn area
        """
        if not self.enabled:
            # Use disabled color or dimmed regular color
            if disabled_color is None:
                disabled_color = tuple(max(0, c // 2) for c in color)  # type: ignore
            draw_color = cast(Color, disabled_color)
            alpha = 200
        else:
            hover_color = self._get_hover_color(color, hover_color)
            draw_color = hover_color if self.hover else color
            alpha = self._get_animation_alpha()

        button_text = f"{self.prefix}{self.text}{self.suffix}"
        self.rect = draw_text(
            surface,
            button_text,
            self.x,
            self.y,
            size=font.get_height(),
            color=(*draw_color, alpha),
        )

        return self.rect
