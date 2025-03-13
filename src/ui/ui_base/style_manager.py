

class StyleManager:
    """Centralized manager for UI style handling.

    This class provides a single source of truth for style-specific rendering logic,
    including colors, characters, and animation parameters. All UI components should
    use this manager to ensure consistent style application.
    """

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from typing import Dict, Tuple, Any
from src.ui.ui_base.ascii_base import UIStyle

    # Singleton instance
    _instance = None

    @classmethod
    def get_instance(cls) -> "StyleManager":
        """Get the singleton instance of StyleManager."""
        if cls._instance is None:
            cls._instance = StyleManager()
        return cls._instance

    def __init__(self):
        """Initialize the StyleManager."""
        # Define style-specific colors
        self.style_colors = {
            UIStyle.SYMBIOTIC: {
                "text": (180, 230, 180),  # Light green
                "highlight": (140, 255, 140),  # Bright green
                "border": (100, 200, 100),  # Medium green
                "background": (10, 30, 10),  # Dark green
            },
            UIStyle.ASTEROID: {
                "text": (230, 200, 150),  # Light brown
                "highlight": (255, 220, 120),  # Gold
                "border": (200, 170, 120),  # Medium brown
                "background": (30, 20, 10),  # Dark brown
            },
            UIStyle.MECHANICAL: {
                "text": (180, 200, 230),  # Light blue
                "highlight": (140, 180, 255),  # Bright blue
                "border": (100, 150, 200),  # Medium blue
                "background": (10, 20, 30),  # Dark blue
            },
            UIStyle.QUANTUM: {
                "text": (210, 180, 230),  # Light purple
                "highlight": (230, 140, 255),  # Bright purple
                "border": (180, 120, 200),  # Medium purple
                "background": (20, 10, 30),  # Dark purple
            },
            UIStyle.FLEET: {
                "text": (200, 200, 220),  # Light gray
                "highlight": (220, 220, 255),  # Bright gray
                "border": (150, 150, 170),  # Medium gray
                "background": (20, 20, 30),  # Dark gray
            },
        }

        # Define style-specific border characters
        self.border_chars = {
            UIStyle.SYMBIOTIC: {
                "top_left": "╭",
                "top_right": "╮",
                "bottom_left": "╰",
                "bottom_right": "╯",
                "horizontal": "─",
                "vertical": "│",
            },
            UIStyle.ASTEROID: {
                "top_left": "┌",
                "top_right": "┐",
                "bottom_left": "└",
                "bottom_right": "┘",
                "horizontal": "─",
                "vertical": "│",
            },
            UIStyle.MECHANICAL: {
                "top_left": "╔",
                "top_right": "╗",
                "bottom_left": "╚",
                "bottom_right": "╝",
                "horizontal": "═",
                "vertical": "║",
            },
            UIStyle.QUANTUM: {
                "top_left": "┏",
                "top_right": "┓",
                "bottom_left": "┗",
                "bottom_right": "┛",
                "horizontal": "━",
                "vertical": "┃",
            },
            UIStyle.FLEET: {
                "top_left": "┌",
                "top_right": "┐",
                "bottom_left": "└",
                "bottom_right": "┘",
                "horizontal": "─",
                "vertical": "│",
            },
        }

        # Define style-specific fill characters
        self.fill_chars = {
            UIStyle.SYMBIOTIC: {
                "empty": "·",
                "partial": ["░", "▒", "▓"],
                "full": "█",
            },
            UIStyle.ASTEROID: {
                "empty": "·",
                "partial": ["▪", "◆", "◈"],
                "full": "■",
            },
            UIStyle.MECHANICAL: {
                "empty": "·",
                "partial": ["▫", "▪", "◻"],
                "full": "◼",
            },
            UIStyle.QUANTUM: {
                "empty": "·",
                "partial": ["∴", "∷", "⋮"],
                "full": "⋯",
            },
            UIStyle.FLEET: {
                "empty": "·",
                "partial": ["◌", "◍", "◉"],
                "full": "●",
            },
        }

    def get_color(
        self, style: UIStyle, color_type: str = "text"
    ) -> Tuple[int, int, int]:
        """Get color for the specified style and color type.

        Args:
            style: UI style to get color for
            color_type: Type of color to get (text, highlight, border, background)

        Returns:
            RGB color tuple
        """
        try:
            if style in self.style_colors and color_type in self.style_colors[style]:
                return self.style_colors[style][color_type]
                # Fallback to default colors
            if color_type == "background":
                return COLOR_BG
            elif color_type == "highlight":
                return COLOR_HIGHLIGHT
            else:
                return COLOR_TEXT
        except Exception as e:
            logging.error(f"Error getting color: {e}")
            return COLOR_TEXT

    def get_border_chars(self, style: UIStyle) -> Dict[str, str]:
        """Get border characters for the specified style.

        Args:
            style: UI style to get border characters for

        Returns:
            Dictionary of border characters
        """
        try:
            if style in self.border_chars:
                return self.border_chars[style]
            else:
                # Fallback to basic ASCII
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

    def get_fill_chars(self, style: UIStyle) -> Dict[str, Any]:
        """Get fill characters for the specified style.

        Args:
            style: UI style to get fill characters for

        Returns:
            Dictionary of fill characters
        """
        try:
            if style in self.fill_chars:
                return self.fill_chars[style]
            else:
                # Fallback to basic ASCII
                return {
                    "empty": "·",
                    "partial": ["░", "▒", "▓"],
                    "full": "█",
                }
        except Exception as e:
            logging.error(f"Error getting fill chars: {e}")
            return {
                "empty": "·",
                "partial": ["░", "▒", "▓"],
                "full": "█",
            }

    def get_progress_color(
        self, style: UIStyle, position: float
    ) -> Tuple[int, int, int]:
        """Get color for progress bar based on position and style.

        Args:
            style: UI style to get color for
            position: Position along the progress bar (0.0 to 1.0)

        Returns:
            RGB color tuple
        """
        try:
            base_color = self.get_color(style, "text")
            r, g, b = base_color

            # Adjust color based on style
            if style == UIStyle.SYMBIOTIC:
                # Symbiotic style: green gradient
                g = min(255, int(g * (0.8 + 0.4 * position)))
                return (int(r * 0.8), g, int(b * 0.8))
            elif style == UIStyle.ASTEROID:
                # Asteroid style: orange/red gradient
                r = min(255, int(r * (0.8 + 0.4 * position)))
                return (r, int(g * 0.9), int(b * 0.7))
            elif style == UIStyle.MECHANICAL:
                # Mechanical style: blue gradient
                b = min(255, int(b * (0.8 + 0.4 * position)))
                return (int(r * 0.9), int(g * 0.9), b)
            elif style == UIStyle.QUANTUM:
                # Quantum style: purple gradient
                r = min(255, int(r * (0.8 + 0.3 * position)))
                b = min(255, int(b * (0.8 + 0.3 * position)))
                return (r, int(g * 0.8), b)
            elif style == UIStyle.FLEET:
                # Fleet style: cyan gradient
                g = min(255, int(g * (0.8 + 0.3 * position)))
                b = min(255, int(b * (0.8 + 0.3 * position)))
                return (int(r * 0.8), g, b)
            else:
                # Default: white gradient
                factor = 0.8 + 0.4 * position
                return (
                    min(255, int(r * factor)),
                    min(255, int(g * factor)),
                    min(255, int(b * factor)),
                )
        except Exception as e:
            logging.error(f"Error getting progress color: {e}")
            return COLOR_TEXT
