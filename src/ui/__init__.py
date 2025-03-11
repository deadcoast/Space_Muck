# Space Muck UI Package
# Define the public API for the ui module
__all__ = [
    # Base components
    'UIStyle', 'AnimationStyle', 'UIElement',
    
    # Specialized components
    'Menu', 'FleetDisplay', 'AsteroidFieldVisualizer',
    'SymbioteEvolutionMonitor', 'MiningStatus', 'SpaceMuckMainUI',
    
    # ASCII UI components
    'ASCIIBox', 'ASCIIPanel', 'ASCIIButton',
    'draw_text', 'draw_panel'
]

# Import specific components instead of using star imports
# Base components
from ui.ascii_base import (
    UIStyle,
    AnimationStyle,
    UIElement
)

# Specialized components from component_modules
from ui.component_modules import (
    Menu,
    FleetDisplay,
    AsteroidFieldVisualizer,
    SymbioteEvolutionMonitor,
    MiningStatus,
    SpaceMuckMainUI
)

# ASCII UI components
from ui.ascii_ui import (
    ASCIIBox,
    ASCIIPanel,
    ASCIIButton,
    draw_text,
    draw_panel
)
