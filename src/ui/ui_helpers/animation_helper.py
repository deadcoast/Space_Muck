import logging
from enum import Enum, auto
from ui.ui_element.ui_style import UIStyle

# Define standard colors for UI components
COLOR_TEXT = (220, 220, 220)  # Standard text color
COLOR_BG = (20, 20, 30)  # Standard background color
COLOR_HIGHLIGHT = (180, 180, 255)  # Standard highlight color


class AnimationStyle(Enum):
    """Animation styles for transitions"""

    CELLULAR = auto()  # Cell-by-cell reveal mimicking Game of Life
    FRACTAL = auto()  # Recursive splitting pattern
    WARP = auto()  # Space-warp style transitions
    QUANTUM_FLUX = auto()  # Probability wave collapse
    MINERAL_GROWTH = auto()  # Crystal-like growth patterns

    @classmethod
    def get_animation_for_style(cls, ui_style: UIStyle) -> "AnimationStyle":
        """Get appropriate animation style based on UI style

        Args:
            ui_style: UI style to get animation style for

        Returns:
            AnimationStyle appropriate for the UI style
        """
        try:
            animation_map = {
                UIStyle.SYMBIOTIC: cls.CELLULAR,
                UIStyle.ASTEROID: cls.MINERAL_GROWTH,
                UIStyle.MECHANICAL: cls.FRACTAL,
                UIStyle.QUANTUM: cls.QUANTUM_FLUX,
                UIStyle.FLEET: cls.WARP,
            }
            return animation_map.get(ui_style, cls.CELLULAR)
        except Exception as e:
            logging.error(f"Error getting animation for style: {e}")
            return cls.CELLULAR
