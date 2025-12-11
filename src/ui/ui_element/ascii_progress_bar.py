"""
ascii_progress_bar.py

Provides a specialized progress bar for displaying ASCII-style progress indicators.
"""

# Standard library imports
import logging
import math
import time

# Local application imports
from typing import Optional, Tuple, TypeVar

import pygame

from ui.draw_utils import draw_text
from ui.ui_base.ascii_base import UIStyle
from ui.ui_base.ui_element import UIElement

# Third-party library imports


# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height


class ASCIIProgressBar(UIElement):
    """ASCII-style progress bar with animation support.

    This class provides a customizable progress bar with different visual styles
    based on the selected UIStyle. It supports smooth animations when progress
    changes and style-specific characters for filled and empty portions.

    Inherits from UIElement to leverage standardized animation framework and styling.
    """

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
            style: Visual style for the progress bar
            converter_type: Optional converter type to determine style
        """
        # Determine the style based on converter type if provided
        actual_style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )

        # Standard height for progress bar is 1
        height = 1

        # Initialize the parent UIElement class
        super().__init__(x, y, width, height, actual_style)

        # ASCIIProgressBar specific attributes
        self.progress = max(0.0, min(1.0, progress))
        self._set_progress_chars()

        # Add progress bar specific animation data
        self.animation["style_data"]["target_progress"] = self.progress

    def _set_progress_chars(self) -> None:
        """Set the progress bar characters based on UI style."""
        style_chars = {
            UIStyle.QUANTUM: ("◈", "◇"),  # ◈, ◇
            UIStyle.SYMBIOTIC: ("█", "░"),  # █, ░
            UIStyle.MECHANICAL: ("█", "▒"),  # █, ▒
            UIStyle.ASTEROID: ("◆", "◇"),  # ◆, ◇
            UIStyle.FLEET: ("■", "□"),  # ■, □
        }
        self.fill_char, self.empty_char = style_chars.get(self.style, ("#", " "))

    def set_progress(
        self, progress: float, animate: bool = False, duration: float = 0.5
    ) -> None:
        """
        Set the current progress with optional animation.

        Args:
            progress: Progress value (0.0 to 1.0)
            animate: Whether to animate the progress change
            duration: Duration of the animation in seconds
        """
        try:
            target_progress = max(0.0, min(1.0, progress))

            if animate:
                # Store current progress as starting point and target progress
                self.animation["style_data"]["start_progress"] = self.progress
                self.animation["style_data"]["target_progress"] = target_progress

                # Use the standardized animation framework from UIElement
                self.start_animation(duration)
            else:
                self.progress = target_progress
        except Exception as e:
            logging.error(f"Error setting progress: {e}")
            self.progress = target_progress  # Fallback to direct setting

    @staticmethod
    def _get_quantum_filled_text(filled_width: int) -> str:
        """Generate filled text for quantum style.

        Args:
            filled_width: Width of the filled portion

        Returns:
            str: The generated text pattern
        """
        filled_text = ""
        time_factor = time.time() * 3.0
        for i in range(filled_width):
            # Alternate characters based on position and time
            wave = math.sin(i / 2.0 + time_factor)
            char_idx = 0 if wave > 0 else 1
            chars = ["◈", "◆"]
            filled_text += chars[char_idx]
        return filled_text

    def _get_symbiotic_filled_text(self, filled_width: int) -> str:
        """Generate filled text for symbiotic style.

        Args:
            filled_width: Width of the filled portion

        Returns:
            str: The generated text pattern
        """
        return "".join(
            "▓" if i / self.width > self.progress - 0.1 else "█"
            for i in range(filled_width)
        )

    def _draw_filled_portion(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        filled_width: int,
        fill_color: Tuple[int, int, int],
    ) -> pygame.Rect:
        """Draw the filled portion of the progress bar with style-specific rendering."""
        if filled_width <= 0:
            return pygame.Rect(self.x, self.y, 0, 0)

        try:
            # Get the appropriate filled text based on style
            if self.style == UIStyle.QUANTUM and self.animation["active"]:
                filled_text = self._get_quantum_filled_text(filled_width)
            elif self.style == UIStyle.SYMBIOTIC and self.animation["active"]:
                filled_text = self._get_symbiotic_filled_text(filled_width)
            else:
                # Default rendering
                filled_text = self.fill_char * filled_width

            return draw_text(
                surface,
                filled_text,
                self.x,
                self.y,
                size=font.get_height(),
                color=fill_color,
            )
        except Exception as e:
            logging.error(f"Error drawing filled portion: {e}")
            # Fallback to simple rendering
            filled_text = self.fill_char * filled_width
            return draw_text(
                surface,
                filled_text,
                self.x,
                self.y,
                size=font.get_height(),
                color=fill_color,
            )

    def _get_quantum_empty_text(self, filled_width: int, empty_width: int) -> str:
        """Generate empty text for quantum style.

        Args:
            filled_width: Width of the filled portion
            empty_width: Width of the empty portion

        Returns:
            str: The generated text pattern
        """
        empty_text = ""
        time_factor = time.time() * 2.0

        for i in range(empty_width):
            # Subtle variation in empty space
            pos = filled_width + i
            wave = math.sin(pos / 4.0 + time_factor) * 0.5 + 0.5
            empty_text += "·" if wave < 0.3 else self.empty_char

        return empty_text

    def _get_symbiotic_empty_text(self, filled_width: int, empty_width: int) -> str:
        """Generate empty text for symbiotic style.

        Args:
            filled_width: Width of the filled portion
            empty_width: Width of the empty portion

        Returns:
            str: The generated text pattern
        """
        return "".join(
            (
                "░"
                if (filled_width + i) / self.width < self.progress + 0.1
                else self.empty_char
            )
            for i in range(empty_width)
        )

    def _draw_empty_portion(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        filled_width: int,
        empty_width: int,
        filled_rect: pygame.Rect,
        empty_color: Tuple[int, int, int],
    ) -> pygame.Rect:
        """Draw the empty portion of the progress bar with style-specific rendering."""
        if empty_width <= 0:
            return filled_rect

        try:
            # Get the appropriate empty text based on style
            if self.style == UIStyle.QUANTUM and self.animation["active"]:
                empty_text = self._get_quantum_empty_text(filled_width, empty_width)
            elif self.style == UIStyle.SYMBIOTIC and self.animation["active"]:
                empty_text = self._get_symbiotic_empty_text(filled_width, empty_width)
            else:
                # Default rendering
                empty_text = self.empty_char * empty_width

            return draw_text(
                surface,
                empty_text,
                self.x + filled_rect.width,
                self.y,
                size=font.get_height(),
                color=empty_color,
            )
        except Exception as e:
            logging.error(f"Error drawing empty portion: {e}")
            # Fallback to simple rendering
            empty_text = self.empty_char * empty_width
            return draw_text(
                surface,
                empty_text,
                self.x + filled_rect.width,
                self.y,
                size=font.get_height(),
                color=empty_color,
            )

    def _init_animation_pattern(self) -> None:
        """Initialize animation pattern based on style.

        Override the parent method to add progress bar specific initialization.
        """
        # Call the parent method first
        super()._init_animation_pattern()

        # Add progress bar specific initialization if needed
        if "start_progress" not in self.animation["style_data"]:
            self.animation["style_data"]["start_progress"] = self.progress

    def _update_animation_style(self, progress: float) -> None:
        """Update animation based on style and progress.

        Override the parent method to implement progress bar specific animation.

        Args:
            progress: Animation progress from 0.0 to 1.0
        """
        try:
            # Get the start and target progress values
            start_progress = self.animation["style_data"].get("start_progress", 0.0)
            target_progress = self.animation["style_data"].get(
                "target_progress", self.progress
            )

            # Apply style-specific easing based on UI style
            t = progress
            if self.style == UIStyle.QUANTUM:
                # Quantum style: more abrupt changes with oscillation
                # Add small oscillation for quantum uncertainty effect
                oscillation = math.sin(t * math.pi * 4) * 0.05 * (1 - t)
                base_ease = math.sin(t * math.pi) * (1 - t) + t
                ease = base_ease + oscillation
            elif self.style == UIStyle.SYMBIOTIC:
                # Symbiotic style: organic, slower start and end with slight growth spurts
                base_ease = t**2 * (3.0 - 2.0 * t)
                # Add small growth spurts for organic feel
                if t > 0.3 and t < 0.7:
                    growth_spurt = math.sin(t * math.pi * 3) * 0.03
                    ease = base_ease + growth_spurt
                else:
                    ease = base_ease
            elif self.style == UIStyle.MECHANICAL:
                # Mechanical style: linear with slight steps
                steps = 10
                stepped_t = math.floor(t * steps) / steps
                # Blend between stepped and smooth
                ease = stepped_t * 0.7 + t * 0.3
            elif self.style == UIStyle.FLEET:
                # Fleet style: quick start, slower finish
                ease = math.sqrt(t)
            elif self.style == UIStyle.ASTEROID:
                # Asteroid style: bouncy with slight overshoot
                ease = 2 * t * t if t < 0.5 else 1 - math.pow(-2 * t + 2, 2) / 2
            else:
                # Default cubic ease out: smooth deceleration
                ease = t**2 * (3.0 - 2.0 * t)

            # Ensure ease is within valid range
            ease = max(0.0, min(1.0, ease))

            # Update the progress value based on the easing function
            self.progress = start_progress + (target_progress - start_progress) * ease

            # If animation is complete, set to final value
            if progress >= 1.0:
                self.progress = target_progress
        except Exception as e:
            logging.error(f"Error updating progress animation: {e}")

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        fill_color: Tuple[int, int, int] = (100, 255, 100),
        empty_color: Tuple[int, int, int] = (100, 100, 100),
    ) -> pygame.Rect:
        """
        Draw the progress bar with style-specific rendering.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            fill_color: Color for the filled portion
            empty_color: Color for the empty portion

        Returns:
            pygame.Rect: The drawn area
        """
        try:
            # Update animation if active
            if self.animation["active"]:
                self.update_animation()

            # Apply style-specific color adjustments
            adjusted_fill_color = fill_color
            adjusted_empty_color = empty_color

            # Style-specific color effects
            if self.animation["active"]:
                if self.style == UIStyle.QUANTUM:
                    # Quantum style: pulsing brightness with blue shift
                    pulse = 0.15 * math.sin(time.time() * 8.0) + 0.85
                    r = int(fill_color[0] * pulse * 0.8)
                    g = int(fill_color[1] * pulse * 0.9)
                    b = min(255, int(fill_color[2] * pulse * 1.2))
                    adjusted_fill_color = (r, g, b)

                    # Subtle glow in empty portion
                    empty_pulse = 0.05 * math.sin(time.time() * 6.0) + 0.95
                    adjusted_empty_color = tuple(
                        int(c * empty_pulse) for c in empty_color
                    )

                elif self.style == UIStyle.SYMBIOTIC:
                    # Symbiotic style: green shift with organic pulsing
                    anim_progress = self.animation["progress"]
                    # Intensity increases with animation progress
                    intensity = (0.1 * math.sin(time.time() * 3.0) + 0.9) * (
                        1.0 + anim_progress * 0.2
                    )
                    r = int(fill_color[0] * 0.7 * intensity)
                    g = min(255, int(fill_color[1] * 1.2 * intensity))
                    b = int(fill_color[2] * 0.7 * intensity)
                    adjusted_fill_color = (r, g, b)

                elif self.style == UIStyle.MECHANICAL:
                    # Mechanical style: consistent colors with slight variation
                    adjusted_fill_color = fill_color

                elif self.style == UIStyle.FLEET:
                    # Fleet style: brighter with slight yellow shift
                    r = min(255, int(fill_color[0] * 1.1))
                    g = min(255, int(fill_color[1] * 1.1))
                    b = int(fill_color[2] * 0.9)
                    adjusted_fill_color = (r, g, b)

                elif self.style == UIStyle.ASTEROID:
                    # Asteroid style: slight red shift
                    r = min(255, int(fill_color[0] * 1.2))
                    g = int(fill_color[1] * 0.9)
                    b = int(fill_color[2] * 0.9)
                    adjusted_fill_color = (r, g, b)

            filled_width = int(self.width * self.progress)
            empty_width = self.width - filled_width

            # Draw both portions
            filled_rect = self._draw_filled_portion(
                surface, font, filled_width, adjusted_fill_color
            )
            empty_rect = self._draw_empty_portion(
                surface,
                font,
                filled_width,
                empty_width,
                filled_rect,
                adjusted_empty_color,
            )

            return filled_rect.union(empty_rect) if empty_width > 0 else filled_rect
        except Exception as e:
            logging.error(f"Error drawing progress bar: {e}")
            # Return a minimal rect to avoid crashes
            return pygame.Rect(
                self.x, self.y, self.width * font.size("X")[0], font.get_height()
            )
