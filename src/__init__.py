"""Space Muck game package.

This package contains all modules for the Space Muck game.
"""

# Standard library imports

# Third-party library imports

# Local application imports

# Import specific constants from config.py to make them available at src level
from .config import (
    GAME_CONFIG,
    PLAYER_CONFIG,
    WORLD_CONFIG,
    UI_CONFIG,
    DEBUG_CONFIG,
    LOG_LEVEL
)

# Import import_standards module to make it available at src level
from .utils import import_standards

__all__ = [
    "GAME_CONFIG",
    "PLAYER_CONFIG",
    "WORLD_CONFIG",
    "UI_CONFIG",
    "DEBUG_CONFIG",
    "LOG_LEVEL",
    "import_standards",
]

