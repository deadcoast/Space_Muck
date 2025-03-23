# Use absolute imports for consistency

# Standard library imports
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

# Proper handling of imports to avoid circular dependencies
if TYPE_CHECKING:
    # Use fully qualified import to avoid linting errors
    from src.ui.ui_base.ui_style import UIStyle

# Define a placeholder
UIStyle: Any = None

# Runtime import handling
try:
    # Use absolute import to avoid relative import beyond top-level package
    from ui.ui_base.ui_style import UIStyle
except ImportError:
    logging.warning("UIStyle could not be imported, using fallback")

CELLULAR = auto()  # Cell-by-cell reveal mimicking Game of Life
FRACTAL = auto()  # Recursive splitting pattern
WARP = auto()  # Space-warp style transitions
QUANTUM_FLUX = auto()  # Probability wave collapse
MINERAL_GROWTH = auto()  # Crystal-like growth patterns


class AnimationStyle(Enum):
    """Animation styles for transitions"""

    @classmethod
    def get_animation_for_style(cls, ui_style: Any) -> "AnimationStyle":
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
