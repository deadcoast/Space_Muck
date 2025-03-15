"""
Enhancement systems for Python Import Fixer.

This module provides enhancement capabilities for extending functionality
without modifying original code, including event handling and method enhancement.
"""

# Standard library imports
import importlib.util
from typing import TYPE_CHECKING

# For type checking only - these imports are not executed at runtime
if TYPE_CHECKING:
    try:
        import rich  # type: ignore
    except ImportError:
        pass

# Check for optional dependencies
RICH_AVAILABLE = importlib.util.find_spec("rich") is not None

# Import optional dependencies at runtime
rich = None  # Define at module level

if RICH_AVAILABLE:
    try:
        import rich
    except ImportError:
        pass

# Local application imports
from .enhancement_system import EnhancementSystem
from .event_system import EventSystem, EventType, Event

__all__ = [
    "EnhancementSystem",
    "EventSystem",
    "EventType",
    "Event",
]
