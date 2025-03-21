"""
ui_style.py

Contains the UIStyle enum and related functions.
"""

# Standard library imports
import logging
from enum import Enum, auto
from typing import Dict, Optional

# Local application imports
# Import config colors when needed

# Third-party library imports


class UIStyle(Enum):
    """Different visual styles for UI components"""

    SYMBIOTIC = auto()  # Organic, evolving patterns
    ASTEROID = auto()  # Rough, mineral-like patterns
    MECHANICAL = auto()  # Ship/tech inspired rigid patterns
    QUANTUM = auto()  # Probability-wave inspired patterns
    FLEET = auto()  # Organized, military-like patterns

    @classmethod
    def get_border_chars(cls, style: "UIStyle") -> Dict[str, str]:
        """Get border characters based on the selected style

        Args:
            style: UI style to get border characters for

        Returns:
            Dictionary of border characters for the specified style
        """
        try:
            borders = {
                cls.SYMBIOTIC: {
                    "tl": "╭",
                    "tr": "╮",
                    "bl": "╰",
                    "br": "╯",
                    "h": "─",
                    "v": "│",
                    "fill": " ",
                },
                cls.ASTEROID: {
                    "tl": "┌",
                    "tr": "┐",
                    "bl": "└",
                    "br": "┘",
                    "h": "═",
                    "v": "║",
                    "fill": "·",
                },
                cls.MECHANICAL: {
                    "tl": "╔",
                    "tr": "╗",
                    "bl": "╚",
                    "br": "╝",
                    "h": "═",
                    "v": "║",
                    "fill": " ",
                },
                cls.QUANTUM: {
                    "tl": "╒",
                    "tr": "╕",
                    "bl": "╘",
                    "br": "╛",
                    "h": "╌",
                    "v": "╎",
                    "fill": "·",
                },
                cls.FLEET: {
                    "tl": "┏",
                    "tr": "┓",
                    "bl": "┗",
                    "br": "┛",
                    "h": "━",
                    "v": "┃",
                    "fill": " ",
                },
            }
            return borders.get(style, borders[cls.MECHANICAL])
        except Exception as e:
            logging.error(f"Error getting border chars: {e}")
            # Return safe default on error
            return {
                "tl": "+",
                "tr": "+",
                "bl": "+",
                "br": "+",
                "h": "-",
                "v": "|",
                "fill": " ",
            }

    @classmethod
    def get_style_for_converter(cls, converter_type: Optional[str]) -> "UIStyle":
        """Get appropriate style based on converter type

        Args:
            converter_type: Type of converter to get style for

        Returns:
            UIStyle appropriate for the converter type
        """
        if converter_type is None:
            return cls.MECHANICAL

        try:
            style_map = {
                "BASIC": cls.MECHANICAL,
                "ADVANCED": cls.QUANTUM,
                "ORGANIC": cls.SYMBIOTIC,
                "FLEET": cls.FLEET,
                "MINING": cls.ASTEROID,
            }
            return style_map.get(converter_type.upper(), cls.MECHANICAL)
        except Exception as e:
            logging.error(f"Error getting style for converter: {e}")
            return cls.MECHANICAL
