#!/usr/bin/env python3
# /src/systems/event_system.py
"""
Event System

Implements an event-driven communication system for the Space Muck game.
Provides standardized event buses, subscription patterns, and event batching
to enable loose coupling between game components.
"""

# Standard library imports
import logging
import time
import uuid

# Third-party library imports

# Local application imports
from typing import (
    Dict,
    List,
    Callable,
    Any,
    Optional,
    TypeVar,
    Tuple,
)

# Type definitions
T = TypeVar("T")
EventData = Dict[str, Any]
EventHandler = Callable[[str, EventData], None]
EventFilter = Callable[[str, EventData], bool]
EventPriority = int
EventId = str


class EventSubscription:
    """Represents a subscription to an event bus."""

    def __init__(
        self,
        event_type: str,
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None,
        priority: EventPriority = 0,
    ) -> None:
        """
        Initialize a new event subscription.

        Args:
            event_type: The type of event to subscribe to, or "*" for all events
            handler: Function to call when an event is received
            filter_func: Optional function to filter events
            priority: Priority level (higher numbers execute first)
        """
        self.id: EventId = str(uuid.uuid4())
        self.event_type: str = event_type
        self.handler: EventHandler = handler
        self.filter_func: Optional[EventFilter] = filter_func
        self.priority: EventPriority = priority
        self.is_active: bool = True

    def matches(self, event_type: str) -> bool:
        """
        Check if this subscription matches the given event type.

        Args:
            event_type: The event type to check against

        Returns:
            True if the subscription matches, False otherwise
        """
        return self.event_type in ["*", event_type]

    def should_handle(self, event_type: str, data: EventData) -> bool:
        """
        Determine if this subscription should handle the given event.

        Args:
            event_type: The type of the event
            data: The event data

        Returns:
            True if the event should be handled, False otherwise
        """
        if not self.is_active:
            return False

        if not self.matches(event_type):
            return False

        return bool(not self.filter_func or self.filter_func(event_type, data))


class EventBus:
    """
    Event bus that manages event subscriptions and dispatches events to subscribers.

    Implements the observer pattern with support for event filtering,
    priority-based execution, and subscription management.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a new event bus.

        Args:
            name: Unique identifier for this event bus
        """
        self.name: str = name
        self.subscriptions: List[EventSubscription] = []
        self.event_history: List[Tuple[float, str, EventData]] = []
        self.max_history_size: int = 100
        self.is_dispatching: bool = False
        self.pending_subscriptions: List[EventSubscription] = []
        self.pending_unsubscriptions: List[EventId] = []

        logging.info(f"EventBus '{name}' initialized")

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None,
        priority: EventPriority = 0,
    ) -> EventId:
        """
        Subscribe to events on this bus.

        Args:
            event_type: The type of event to subscribe to, or "*" for all events
            handler: Function to call when an event is received
            filter_func: Optional function to filter events
            priority: Priority level (higher numbers execute first)

        Returns:
            Subscription ID that can be used to unsubscribe
        """
        subscription = EventSubscription(event_type, handler, filter_func, priority)

        if self.is_dispatching:
            # Defer subscription until after dispatch completes
            self.pending_subscriptions.append(subscription)
        else:
            self.subscriptions.append(subscription)
            # Sort by priority (descending)
            self.subscriptions.sort(key=lambda s: s.priority, reverse=True)

        return subscription.id

    def unsubscribe(self, subscription_id: EventId) -> bool:
        """
        Unsubscribe from events using the subscription ID.

        Args:
            subscription_id: The ID returned from subscribe()

        Returns:
            True if successfully unsubscribed, False if not found
        """
        if self.is_dispatching:
            # Defer unsubscription until after dispatch completes
            self.pending_unsubscriptions.append(subscription_id)
            return True

        for i, subscription in enumerate(self.subscriptions):
            if subscription.id == subscription_id:
                self.subscriptions.pop(i)
                return True

        return False

    def pause_subscription(self, subscription_id: EventId) -> bool:
        """
        Temporarily pause a subscription without removing it.

        Args:
            subscription_id: The ID returned from subscribe()

        Returns:
            True if successfully paused, False if not found
        """
        for subscription in self.subscriptions:
            if subscription.id == subscription_id:
                subscription.is_active = False
                return True

        return False

    def resume_subscription(self, subscription_id: EventId) -> bool:
        """
        Resume a paused subscription.

        Args:
            subscription_id: The ID returned from subscribe()

        Returns:
            True if successfully resumed, False if not found
        """
        for subscription in self.subscriptions:
            if subscription.id == subscription_id:
                subscription.is_active = True
                return True

        return False

    def dispatch(self, event_type: str, data: Optional[EventData] = None) -> None:
        """
        Dispatch an event to all matching subscribers.

        Args:
            event_type: The type of event to dispatch
            data: Optional data to include with the event
        """
        if data is None:
            data = {}

        # Add to event history
        self.event_history.append((time.time(), event_type, data))
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)

        # Set dispatching flag to defer subscription changes
        self.is_dispatching = True

        try:
            # Call each matching handler
            for subscription in self.subscriptions:
                if subscription.should_handle(event_type, data):
                    try:
                        subscription.handler(event_type, data)
                    except Exception as e:
                        logging.error(f"Error in event handler for '{event_type}': {e}")
        finally:
            self._process_pending_subscription_changes()

    def _process_pending_subscription_changes(self):
        # Clear dispatching flag
        self.is_dispatching = False

        # Process any pending subscription changes
        for subscription in self.pending_subscriptions:
            self.subscriptions.append(subscription)
        self.pending_subscriptions.clear()

        # Sort by priority (descending)
        if self.subscriptions:
            self.subscriptions.sort(key=lambda s: s.priority, reverse=True)

        # Process any pending unsubscriptions
        for sub_id in self.pending_unsubscriptions:
            self.unsubscribe(sub_id)
        self.pending_unsubscriptions.clear()

    def clear_all_subscriptions(self) -> None:
        """Remove all subscriptions from this event bus."""
        if self.is_dispatching:
            # Mark all subscriptions as inactive instead of removing
            for subscription in self.subscriptions:
                subscription.is_active = False
        else:
            self.subscriptions.clear()

    def get_recent_events(self, limit: int = 10) -> List[Tuple[float, str, EventData]]:
        """
        Get recent events from the history.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of (timestamp, event_type, data) tuples
        """
        return self.event_history[-limit:] if self.event_history else []


class EventBatcher:
    """
    Batches events and dispatches them efficiently to multiple event buses.

    Implements the event batching pattern to reduce overhead when many
    events are generated in a short period of time.
    """

    def __init__(self) -> None:
        """Initialize a new event batcher."""
        self.event_buses: Dict[str, EventBus] = {}
        self.batched_events: Dict[str, List[Tuple[str, EventData]]] = {}
        self.is_batching: bool = False

    def register_bus(self, name: str, bus: EventBus) -> None:
        """
        Register an event bus with this batcher.

        Args:
            name: Name of the bus
            bus: EventBus instance
        """
        self.event_buses[name] = bus
        self.batched_events[name] = []

    def start_batch(self) -> None:
        """Start batching events instead of dispatching immediately."""
        self.is_batching = True

    def dispatch_batch(self) -> None:
        """Dispatch all batched events and stop batching."""
        self.is_batching = False

        for bus_name, events in self.batched_events.items():
            if not events:
                continue

            bus = self.event_buses.get(bus_name)
            if not bus:
                continue

            for event_type, data in events:
                bus.dispatch(event_type, data)

            events.clear()

    def dispatch(
        self, bus_name: str, event_type: str, data: Optional[EventData] = None
    ) -> None:
        """
        Dispatch an event to the specified bus.

        If batching is active, the event will be queued for later.
        Otherwise, it will be dispatched immediately.

        Args:
            bus_name: Name of the bus to dispatch to
            event_type: Type of event
            data: Optional event data
        """
        if data is None:
            data = {}

        if bus_name not in self.event_buses:
            logging.warning(f"Attempted to dispatch to unknown event bus: {bus_name}")
            return

        if self.is_batching:
            # Queue the event for later
            self.batched_events[bus_name].append((event_type, data))
        else:
            # Dispatch immediately
            self.event_buses[bus_name].dispatch(event_type, data)

    def clear_batched_events(self) -> None:
        """Clear all batched events without dispatching them."""
        for events in self.batched_events.values():
            events.clear()


# Define standard event buses
module_event_bus = EventBus("ModuleEventBus")
game_event_bus = EventBus("GameEventBus")
resource_event_bus = EventBus("ResourceEventBus")

# Create the global event batcher
event_batcher = EventBatcher()

# Register standard buses with the batcher
event_batcher.register_bus("ModuleEventBus", module_event_bus)
event_batcher.register_bus("GameEventBus", game_event_bus)
event_batcher.register_bus("ResourceEventBus", resource_event_bus)


def get_event_bus(name: str) -> Optional[EventBus]:
    """
    Get an event bus by name.

    Args:
        name: Name of the event bus

    Returns:
        The EventBus instance or None if not found
    """
    if name == "ModuleEventBus":
        return module_event_bus
    elif name == "GameEventBus":
        return game_event_bus
    elif name == "ResourceEventBus":
        return resource_event_bus
    else:
        return None


def get_event_batcher() -> EventBatcher:
    """
    Get the global event batcher.

    Returns:
        The EventBatcher instance
    """
    return event_batcher
