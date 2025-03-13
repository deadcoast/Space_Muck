"""
ASCII Character Utilities for Space Muck.

This module provides utility functions for handling ASCII and box drawing characters
in a consistent and reliable way across different fonts and platforms.
"""

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from typing import Dict, Tuple, Optional
import pygame

# Mapping of box drawing characters to ASCII fallbacks
BOX_DRAWING_FALLBACKS: Dict[str, str] = {
    # Box drawing characters (single line)
    "┌": "+",
    "┐": "+",
    "└": "+",
    "┘": "+",  # Corners
    "─": "-",
    "│": "|",  # Lines
    "┬": "+",
    "┴": "+",
    "┤": "+",
    "├": "+",  # T-junctions
    "┼": "+",  # Cross
    # Box drawing characters (double line)
    "╔": "+",
    "╗": "+",
    "╚": "+",
    "╝": "+",  # Double corners
    "═": "=",
    "║": "|",  # Double lines
    "╦": "+",
    "╩": "+",
    "╣": "+",
    "╠": "+",  # Double T-junctions
    "╬": "+",  # Double cross
    # Box drawing characters (rounded)
    "╭": "+",
    "╮": "+",
    "╯": "+",
    "╰": "+",  # Rounded corners
}


def get_safe_char(char: str) -> str:
    """
    Get a safe character that can be rendered by any font.

    If the character is a box drawing character, returns an ASCII fallback.
    Otherwise, returns the original character.

    Args:
        char: Character to check

    Returns:
        Safe character for rendering
    """
    return BOX_DRAWING_FALLBACKS.get(char, char)


def render_safe_text(
    font: pygame.font.Font,
    text: str,
    color: Tuple[int, int, int],
    use_fallbacks: bool = True,
) -> pygame.Surface:
    """
    Render text safely, handling box drawing characters.

    Args:
        font: Font to use for rendering
        text: Text to render
        color: Color to render text with
        use_fallbacks: Whether to use fallbacks for box drawing characters

    Returns:
        Rendered text surface
    """
    if not text:
        # Return an empty surface if text is empty
        return pygame.Surface((0, 0), pygame.SRCALPHA)

    if not use_fallbacks:
        # Try to render directly
        try:
            return font.render(text, True, color)
        except Exception as e:
            logging.debug(f"Error rendering text '{text}': {e}")
            # Fall through to fallback rendering

    # Use fallbacks for box drawing characters
    safe_text = "".join(get_safe_char(char) for char in text)

    try:
        return font.render(safe_text, True, color)
    except Exception as e:
        logging.error(f"Error rendering safe text '{safe_text}': {e}")
        # Last resort: render a placeholder
        return font.render("?" * len(text), True, color)


def test_font_box_drawing_support(font: pygame.font.Font) -> bool:
    """
    Test if a font supports box drawing characters.

    Args:
        font: Font to test

    Returns:
        True if the font supports box drawing characters, False otherwise
    """
    test_chars = "┌─┐│└─┘"

    try:
        # Try to render the test characters
        test_surface = font.render(test_chars, True, (255, 255, 255))

        # Check if the width is reasonable (at least 7 pixels)
        return test_surface.get_width() >= len(test_chars) * 7
    except Exception:
        return False
