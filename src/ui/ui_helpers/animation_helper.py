# Use absolute imports for consistency

# Standard library imports
import logging

# Local application imports
from enum import Enum, auto

# Third-party library imports
# Color imports removed - add back if needed in the future
from src.ui.ui_base.ui_style import UIStyle

CELLULAR = auto()  # Cell-by-cell reveal mimicking Game of Life
FRACTAL = auto()  # Recursive splitting pattern
WARP = auto()  # Space-warp style transitions
QUANTUM_FLUX = auto()  # Probability wave collapse
MINERAL_GROWTH = auto()  # Crystal-like growth patterns


class AnimationStyle(Enum):
    """Animation styles for transitions"""

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
