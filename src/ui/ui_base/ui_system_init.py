"""
UI System Initialization.

This module provides a single entry point to initialize all UI subsystems.
It connects the component registry with the event system without modifying
either system directly, following strict principles of modularity and
optional integration.
"""

# Standard library imports
import logging
import time

# Local application imports
from typing import Any, Dict

from src.ui.ui_base.component_event_bridge import connect_systems

# Third-party library imports

# Track initialization status with detailed metrics
_initialization_status = {
    "initialized": False, 
    "error": None, 
    "timestamp": None,
    "subsystems": {},
    "performance": {}
}


def initialize_ui_systems() -> bool:
    """
    Initialize all UI subsystems.

    This function sets up connections between UI subsystems to enable
    event handling and component lifecycle management. It should be
    called once during application startup.

    Returns:
        True if initialization was successful, False otherwise
    """
    
    # Track start time for performance metrics
    start_time = time.time()
    
    # Create a fresh initialization status dictionary
    global _initialization_status
    _initialization_status = {
        "initialized": False,
        "error": None,
        "timestamp": start_time,
        "subsystems": {},
        "performance": {"start_time": start_time}
    }
    
    try:
        # Connect component registry with event system
        if connect_systems():
            # Update initialization status with success information
            _initialization_status.update({
                "initialized": True,
                "error": None,
                "performance": {
                    "start_time": start_time,
                    "end_time": time.time(),
                    "duration": time.time() - start_time
                }
            })
            logging.info("UI systems initialized successfully")
            return True
        else:
            # Update initialization status with connection failure information
            _initialization_status.update({
                "initialized": False,
                "error": "Failed to connect systems",
                "performance": {
                    "start_time": start_time,
                    "end_time": time.time(),
                    "duration": time.time() - start_time
                }
            })
            logging.error("Failed to initialize UI systems: could not connect systems")
            return False
    except Exception as e:
        # Update initialization status with exception information
        _initialization_status.update({
            "initialized": False,
            "error": str(e),
            "exception_type": type(e).__name__,
            "performance": {
                "start_time": start_time,
                "end_time": time.time(),
                "duration": time.time() - start_time
            }
        })
        logging.error(f"Error initializing UI systems: {e}")
        return False


def get_initialization_status() -> Dict[str, Any]:
    """
    Get the current initialization status of UI systems.

    Returns:
        Dictionary with initialization status information
    """
    return _initialization_status.copy()


def get_initialization_metrics() -> Dict[str, Any]:
    """
    Get detailed metrics about the UI system initialization process.
    
    This is particularly useful for diagnosing initialization issues
    and monitoring performance across different environments.
    
    Returns:
        Dictionary with detailed initialization metrics
    """
    metrics = {
        "success": _initialization_status["initialized"],
        "duration": _initialization_status["performance"].get("duration"),
        "subsystems": _initialization_status["subsystems"],
    }
    
    # Add error information if initialization failed
    if not _initialization_status["initialized"] and _initialization_status["error"]:
        metrics["error"] = {
            "message": _initialization_status["error"],
            "type": _initialization_status.get("exception_type"),
        }
    
    return metrics
