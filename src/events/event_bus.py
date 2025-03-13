"""
Event bus implementation for Space Muck.

This module provides a simple event bus system for handling game events,
following the observer pattern. It supports:
- Multiple event buses for different subsystems
- Event subscription and publishing
- Event history tracking for debugging
"""

# Standard library imports
from collections import defaultdict
import logging
import time

# Third-party library imports

# Local application imports
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class EventBus:
    """Event bus for handling game events."""

    def __init__(self, name: str):
        """Initialize event bus.

        Args:
            name: Name of the event bus for identification
        """
        self.name = name
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 100  # Maximum number of events to track

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            logger.debug(f"Added subscriber for {event_type} in {self.name}")

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            logger.debug(f"Removed subscriber for {event_type} in {self.name}")

    def publish(
        self, event_type: str, event_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publish an event to all subscribers.

        Args:
            event_type: Type of event to publish
            event_data: Optional data associated with the event
        """
        if event_data is None:
            event_data = {}

        # Record event in history
        event_record = {
            "type": event_type,
            "data": event_data,
            "timestamp": time.time(),
            "bus": self.name,
        }
        self._event_history.append(event_record)

        # Trim history if needed
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

        # Notify subscribers
        for callback in self._subscribers[event_type]:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get event history.

        Returns:
            List of event records
        """
        return self._event_history.copy()


# Global event bus registry
_event_buses: Dict[str, EventBus] = {}


def get_event_bus(name: str = "default") -> EventBus:
    """Get or create an event bus.

    Args:
        name: Name of the event bus

    Returns:
        EventBus instance
    """
    # Always return the default bus in test mode
    if name == "test":
        if "test" not in _event_buses:
            _event_buses["test"] = EventBus("test")
        return _event_buses["test"]

    # Normal operation
    if name not in _event_buses:
        _event_buses[name] = EventBus(name)
    return _event_buses[name]


# Create default event bus
_event_buses["default"] = EventBus("default")


def clear_event_buses() -> None:
    """Clear all event buses. Useful for testing."""
    _event_buses.clear()
