"""
Space Muck game package.

This package contains all modules for the Space Muck game.
"""

# Standard library imports

# Third-party library imports

# Local application imports

# Import specific constants from config.py to make them available at src level
from .config import (
    DEBUG_CONFIG,
    GAME_CONFIG,
    LOG_LEVEL,
    PLAYER_CONFIG,
    UI_CONFIG,
    WORLD_CONFIG,
    SHOW_FPS,
    SHOW_MINIMAP,
    SHOW_DEBUG,
    SHOW_TOOLTIPS,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    VIEW_WIDTH,
    VIEW_HEIGHT,
    MINIMAP_SIZE,
    MINIMAP_PADDING,
    GRID_WIDTH,
    GRID_HEIGHT,
    COLOR_BG,
    COLOR_PLAYER,
    COLOR_RACE_1,
    COLOR_RACE_2,
    COLOR_RACE_3,
    COLOR_TEXT,
    RACE_INITIAL_DENSITY,
    COLOR_ENTITY_DEFAULT,
    COLOR_ENTITY_EXPANDING,
    COLOR_ENTITY_AGGRESSIVE,
    COLOR_RACE_4,
    COLOR_RACE_5
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
    "SHOW_FPS",
    "SHOW_MINIMAP",
    "SHOW_DEBUG",
    "SHOW_TOOLTIPS",
    "WINDOW_WIDTH",
    "WINDOW_HEIGHT",
    "VIEW_WIDTH",
    "VIEW_HEIGHT",
    "MINIMAP_SIZE",
    "MINIMAP_PADDING",
    "GRID_WIDTH",
    "GRID_HEIGHT",
    "COLOR_BG",
    "COLOR_PLAYER",
    "COLOR_RACE_1",
    "COLOR_RACE_2",
    "COLOR_RACE_3",
    "COLOR_TEXT",
    "RACE_INITIAL_DENSITY",
    "COLOR_ENTITY_DEFAULT",
    "COLOR_ENTITY_EXPANDING",
    "COLOR_ENTITY_AGGRESSIVE",
    "COLOR_RACE_4",
    "COLOR_RACE_5",
]
