#!/usr/bin/env python3

# -----------------------------
# EVENT HANDLING SYSTEM
# -----------------------------
#
# Parent: analysis.enhancers
# Dependencies: typing, logging, enum
#
# MAP: /project_root/analysis/enhancers
# EFFECT: Provides event handling for method enhancements
# NAMING: Event[Type]Handler

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


class EventType(Enum):
    """Types of events that can be handled."""

    PRE_ENHANCEMENT = auto()
    POST_ENHANCEMENT = auto()
    STATE_CHANGE = auto()
    ENHANCEMENT_ERROR = auto()
    ENHANCEMENT_ROLLBACK = auto()


@dataclass
class Event:
    """Event data container."""

    type: EventType
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class EventSystem:
    """Event handling system for method enhancements.

    This system provides:
    1. Event registration and handling
    2. Pre/post enhancement hooks
    3. State change notifications
    4. Error handling events
    5. Event history tracking
    """

    def __init__(self):
        """Initialize the event system."""
        self.logger = logging.getLogger(__name__)
        self.handlers = {event_type: set() for event_type in EventType}
        self.event_history = []
        self.max_history = 1000

    def register_handler(
        self, event_type: EventType, handler: Callable[[Event], None]
    ) -> None:
        """Register an event handler.

        Args:
            event_type: Type of event to handle
            handler: Function to handle the event
        """
        self.handlers[event_type].add(handler)
        self.logger.debug(f"Registered handler for {event_type.name}")

    def unregister_handler(
        self, event_type: EventType, handler: Callable[[Event], None]
    ) -> None:
        """Unregister an event handler.

        Args:
            event_type: Type of event to unregister from
            handler: Handler function to remove
        """
        if handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
            self.logger.debug(f"Unregistered handler for {event_type.name}")

    def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers.

        Args:
            event: Event to emit
        """
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        # Notify handlers
        for handler in self.handlers[event.type]:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {str(e)}")

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """Get event history with optional filtering.

        Args:
            event_type: Filter by event type
            source: Filter by event source
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        events = self.event_history

        if event_type:
            events = [e for e in events if e.type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        if limit:
            events = events[-limit:]

        return events

    def clear_history(self) -> None:
        """Clear the event history."""
        self.event_history = []

    def pre_enhancement(self, source: str, **data) -> None:
        """Emit a pre-enhancement event.

        Args:
            source: Source of the enhancement
            **data: Additional event data
        """
        self.emit(Event(type=EventType.PRE_ENHANCEMENT, source=source, data=data))

    def post_enhancement(self, source: str, **data) -> None:
        """Emit a post-enhancement event.

        Args:
            source: Source of the enhancement
            **data: Additional event data
        """
        self.emit(Event(type=EventType.POST_ENHANCEMENT, source=source, data=data))

    def state_change(self, source: str, **data) -> None:
        """Emit a state change event.

        Args:
            source: Source of the state change
            **data: State change data
        """
        self.emit(Event(type=EventType.STATE_CHANGE, source=source, data=data))

    def enhancement_error(self, source: str, error: Exception, **data) -> None:
        """Emit an enhancement error event.

        Args:
            source: Source of the error
            error: The error that occurred
            **data: Additional error data
        """
        self.emit(
            Event(
                type=EventType.ENHANCEMENT_ERROR,
                source=source,
                data={"error": str(error), **data},
            )
        )

    def enhancement_rollback(self, source: str, **data) -> None:
        """Emit an enhancement rollback event.

        Args:
            source: Source of the rollback
            **data: Additional rollback data
        """
        self.emit(Event(type=EventType.ENHANCEMENT_ROLLBACK, source=source, data=data))
