import math
import logging
import pygame
from typing import Tuple
from ui.ui_element.ui_style import UIStyle

# Define standard colors for UI components
COLOR_TEXT = (220, 220, 220)  # Standard text color
COLOR_BG = (20, 20, 30)  # Standard background color
COLOR_HIGHLIGHT = (180, 180, 255)  # Standard highlight color


class RenderHelper:
    """Helper class for standardized rendering across UI components.

    This class provides utility methods for rendering UI elements with consistent
    styling, animation effects, and error handling. It centralizes common rendering
    code to reduce duplication and ensure a unified visual approach.
    """

    @staticmethod
    def draw_char(
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        char: str,
        color: Tuple[int, int, int] = COLOR_TEXT,
        char_coords: bool = True,
    ) -> None:
        """Draw a single character at the specified position.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X position (in character coordinates if char_coords is True, else pixels)
            y: Y position (in character coordinates if char_coords is True, else pixels)
            char: Character to draw
            color: RGB color tuple for the character
            char_coords: If True, x and y are in character coordinates, else pixels
        """
        try:
            char_surf = font.render(char, True, color)
            if char_coords:
                char_width = font.size("X")[0]
                char_height = font.get_height()
                pixel_x = x * char_width
                pixel_y = y * char_height
            else:
                pixel_x = x
                pixel_y = y

            surface.blit(char_surf, (pixel_x, pixel_y))
        except Exception as e:
            logging.error(f"Error drawing character: {e}")

    @staticmethod
    def draw_text(
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int] = COLOR_TEXT,
        char_coords: bool = True,
    ) -> None:
        """Draw a text string at the specified position.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X position (in character coordinates if char_coords is True, else pixels)
            y: Y position (in character coordinates if char_coords is True, else pixels)
            text: Text to draw
            color: RGB color tuple for the text
            char_coords: If True, x and y are in character coordinates, else pixels
        """
        try:
            text_surf = font.render(text, True, color)
            if char_coords:
                char_width = font.size("X")[0]
                char_height = font.get_height()
                pixel_x = x * char_width
                pixel_y = y * char_height
            else:
                pixel_x = x
                pixel_y = y

            surface.blit(text_surf, (pixel_x, pixel_y))
        except Exception as e:
            logging.error(f"Error drawing text: {e}")

    @staticmethod
    def get_style_color(
        style: UIStyle, base_color: Tuple[int, int, int] = COLOR_TEXT
    ) -> Tuple[int, int, int]:
        """Get the appropriate color for a given UI style.

        Args:
            style: The UI style to get the color for
            base_color: The base color to adjust based on style

        Returns:
            RGB color tuple adjusted for the specified style
        """
        try:
            r, g, b = base_color

            if style == UIStyle.QUANTUM:
                # Quantum style: blue-shifted
                return (int(r * 0.8), int(g * 0.9), min(255, int(b * 1.2)))
            elif style == UIStyle.SYMBIOTIC:
                # Symbiotic style: green-shifted
                return (int(r * 0.8), min(255, int(g * 1.2)), int(b * 0.8))
            elif style == UIStyle.MECHANICAL:
                # Mechanical style: slight blue tint
                return (r, g, min(255, int(b * 1.1)))
            elif style == UIStyle.ASTEROID:
                # Asteroid style: orange/red tint
                return (min(255, int(r * 1.2)), int(g * 0.9), int(b * 0.8))
            elif style == UIStyle.FLEET:
                # Fleet style: cyan tint
                return (int(r * 0.8), min(255, int(g * 1.1)), min(255, int(b * 1.1)))
            else:
                return base_color
        except Exception as e:
            logging.error(f"Error getting style color: {e}")
            return base_color

    @staticmethod
    def apply_animation_effect(
        color: Tuple[int, int, int],
        style: UIStyle,
        progress: float,
        phase: float = 0.0,
    ) -> Tuple[int, int, int]:
        """Apply style-specific animation effects to a color.

        Args:
            color: Base RGB color tuple
            style: UI style to apply effects for
            progress: Animation progress from 0.0 to 1.0
            phase: Animation phase for cyclic effects

        Returns:
            Modified RGB color tuple with animation effects applied
        """
        try:
            r, g, b = color

            # Apply style-specific color effects
            if style == UIStyle.QUANTUM:
                # Quantum style: color shifts with subtle pulsing
                r_adj = int(min(255, r * (0.8 + 0.4 * math.sin(progress * math.pi))))
                b_adj = int(min(255, b * (0.8 + 0.4 * math.cos(progress * math.pi))))
                return (r_adj, g, b_adj)
            elif style == UIStyle.SYMBIOTIC:
                # Symbiotic style: gradual green enhancement
                g_boost = int(min(255, g * (1.0 + 0.3 * progress)))
                return (r, g_boost, b)
            elif style == UIStyle.ASTEROID:
                # Asteroid style: warming effect (red channel boost)
                r_boost = int(min(255, r * (1.0 + 0.3 * progress)))
                return (r_boost, g, b)
            elif style == UIStyle.FLEET:
                # Fleet style: blue/cyan pulsing
                b_boost = int(min(255, b * (1.0 + 0.2 * math.sin(phase))))
                g_boost = int(min(255, g * (1.0 + 0.1 * math.sin(phase))))
                return (r, g_boost, b_boost)
            elif style == UIStyle.MECHANICAL or style is None:
                # Mechanical style or default: subtle brightening of all channels
                factor = 1.0 + 0.2 * progress
                return (
                    int(min(255, r * factor)),
                    int(min(255, g * factor)),
                    int(min(255, b * factor)),
                )
        except Exception as e:
            logging.error(f"Error applying animation effect: {e}")
            return color
