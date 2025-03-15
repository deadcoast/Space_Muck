# Space Muck UI Package
# Define the public API for the ui module
# Only include components that are directly imported in this file

# Standard library imports

# Third-party library imports

from src..ui_element.ui_mining_status import MiningStatus
from src.ui.ui_base.ascii_base import UIStyle

# Local application imports
from .game_screen import ASCIIGameScreen

__all__ = [
    # Base components - these are directly imported below
    "UIStyle",
    "AnimationStyle",
    "UIElement",
    # Components that are still imported in this file
    "MiningStatus",
    "ASCIIGameScreen"
    # Other components should be imported directly where needed
    # rather than through this __init__ file to avoid circular imports
]

# Import only the base components to avoid circular imports
# Other components should be imported directly where needed

# Do not import ui_element modules here to avoid circular dependencies
# These should be imported directly in the files that need them

# Main UI component

# No need to import ASCII UI components here
# They should be imported directly where needed to avoid circular imports
