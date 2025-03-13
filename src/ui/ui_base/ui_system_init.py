"""
UI System Initialization.

This module provides a single entry point to initialize all UI subsystems.
It connects the component registry with the event system without modifying
either system directly, following strict principles of modularity and
optional integration.
"""

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from typing import Dict, Any
from src.ui.ui_base.component_event_bridge import connect_systems

# Track initialization status
_initialization_status = {"initialized": False, "error": None, "timestamp": None}


def initialize_ui_systems() -> bool:
    """
    Initialize all UI subsystems.

    This function sets up connections between UI subsystems to enable
    event handling and component lifecycle management. It should be
    called once during application startup.

    Returns:
        True if initialization was successful, False otherwise
    """
    import time

    global _initialization_status

    try:
        # Connect component registry with event system
        if connect_systems():
            _initialization_status = {
                "initialized": True,
                "error": None,
                "timestamp": time.time(),
            }
            logging.info("UI systems initialized successfully")
            return True
        else:
            _initialization_status = {
                "initialized": False,
                "error": "Failed to connect systems",
                "timestamp": time.time(),
            }
            logging.error("Failed to initialize UI systems: could not connect systems")
            return False
    except Exception as e:
        _initialization_status = {
            "initialized": False,
            "error": str(e),
            "timestamp": time.time(),
        }
        logging.error(f"Error initializing UI systems: {e}")
        return False


def get_initialization_status() -> Dict[str, Any]:
    """
    Get the current initialization status of UI systems.

    Returns:
        Dictionary with initialization status information
    """
    return _initialization_status.copy()
