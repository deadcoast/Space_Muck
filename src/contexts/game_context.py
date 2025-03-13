"""
Game Context: Central game state and event management system.

This module provides the core game context that manages game state,
coordinates between different managers, and handles event propagation.
"""

# Standard library imports
from datetime import datetime
import logging

# Third-party library imports

# Local application imports
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Tuple


# Game States
class GameState(Enum):
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    SAVING = auto()
    LOADING = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()


# Event Types
class GameEventType(Enum):
    STATE_CHANGED = auto()
    ERROR_OCCURRED = auto()

    # Resource Events
    RESOURCE_CREATED = auto()
    RESOURCE_UPDATED = auto()
    RESOURCE_DELETED = auto()
    RESOURCE_FLOW_STARTED = auto()
    RESOURCE_FLOW_STOPPED = auto()
    RESOURCE_THRESHOLD_REACHED = auto()

    # Module Events
    MODULE_CHANGED = auto()
    MODULE_ERROR = auto()

    # Threshold Events
    THRESHOLD_TRIGGERED = auto()
    THRESHOLD_CLEARED = auto()


@dataclass
class GameEvent:
    """Represents a game event."""

    type: GameEventType
    source: str
    data: Dict[str, any]
    timestamp: float
    priority: int = 1


class GameContext:
    """
    Central context for managing game state and coordinating between systems.
    """

    def __init__(self) -> None:
        """Initialize the game context."""
        # State management
        self.state = GameState.INITIALIZING
        self.previous_state = None
        self.state_history: List[Tuple[GameState, float]] = []

        # Event system
        self.event_queue: List[GameEvent] = []
        self.event_handlers: Dict[GameEventType, List[Callable[[GameEvent], None]]] = {
            event_type: [] for event_type in GameEventType
        }

        # Resource tracking
        self.active_resources: Set[str] = set()
        self.resource_states: Dict[str, str] = {}
        self.resource_thresholds: Dict[str, float] = {}
        self.resource_capacities: Dict[str, float] = {}
        self.resource_flow_rates: Dict[str, float] = {}

        # Flow tracking
        self.active_flows: Dict[str, Dict[str, float]] = {}  # source -> {dest -> rate}
        self.flow_history: List[Dict[str, any]] = []  # List of flow events
        self.max_flow_history = 1000  # Maximum number of flow events to keep

        # Module tracking
        self.active_modules: Set[str] = set()
        self.module_states: Dict[str, str] = {}
        self.module_dependencies: Dict[str, Set[str]] = {}

        # System state
        self.active = True
        self.error_count = 0
        self.last_update = 0.0
        self.update_interval = 1.0  # seconds

        logging.info("GameContext initialized")

    def register_event_handler(
        self, event_type: GameEventType, handler: Callable[[GameEvent], None]
    ) -> bool:
        """
        Register a handler for a specific event type.

        Args:
            event_type: Type of event to handle
            handler: Callback function for handling the event

        Returns:
            bool: True if registration successful
        """
        if event_type not in self.event_handlers:
            logging.error(f"Invalid event type: {event_type}")
            return False

        self.event_handlers[event_type].append(handler)
        return True

    def unregister_event_handler(
        self, event_type: GameEventType, handler: Callable[[GameEvent], None]
    ) -> bool:
        """
        Unregister an event handler.

        Args:
            event_type: Type of event
            handler: Handler to remove

        Returns:
            bool: True if unregistration successful
        """
        if event_type not in self.event_handlers:
            logging.error(f"Invalid event type: {event_type}")
            return False

        try:
            self.event_handlers[event_type].remove(handler)
            return True
        except ValueError:
            logging.warning(f"Handler not found for {event_type}")
            return False

    def dispatch_event(self, event: GameEvent) -> None:
        """
        Dispatch an event to registered handlers.

        Args:
            event: Event to dispatch
        """
        if event.type not in self.event_handlers:
            logging.error(f"Invalid event type: {event.type}")
            return

        # Add to queue for processing
        self.event_queue.append(event)
        self._process_events()

    def update(self, dt: float) -> None:
        """
        Update game context state.

        Args:
            dt: Time delta since last update
        """
        if not self.active:
            return

        self.last_update += dt
        if self.last_update < self.update_interval:
            return

        try:
            self._process_events()
            self._update_resource_states()
            self._update_module_states()
            self._check_thresholds()

            self.last_update = 0
            self.error_count = 0
        except Exception as e:
            self.error_count += 1
            logging.error(f"Error in GameContext update: {e}")

            if self.error_count >= 3:
                self.transition_state(GameState.ERROR)

    def transition_state(self, new_state: GameState) -> bool:
        """
        Transition to a new game state.

        Args:
            new_state: State to transition to

        Returns:
            bool: True if transition successful, False if transition not allowed or failed
        """
        # If already in the requested state, no transition needed
        if new_state == self.state:
            return True
            
        # Check if the transition is allowed based on current state
        if not self._is_valid_transition(self.state, new_state):
            logging.warning(f"Invalid state transition: {self.state} -> {new_state}")
            return False
            
        try:
            self._update_state_and_notify(new_state)
            return True
        except Exception as e:
            logging.error(f"Failed to transition state: {e}")
            return False
            
    def _update_state_and_notify(self, new_state: GameState) -> None:
        """
        Update the game state and notify observers about the change.
        
        Args:
            new_state: The new game state to transition to
        """
        # Record state change
        self.previous_state = self.state
        self.state = new_state
        self.state_history.append((new_state, datetime.now().timestamp()))

        # Notify observers
        self.dispatch_event(
            GameEvent(
                type=GameEventType.STATE_CHANGED,
                source="game_context",
                data={"old_state": self.previous_state, "new_state": new_state},
                timestamp=datetime.now().timestamp(),
            )
        )

        logging.info(f"Game state transitioned: {self.previous_state} -> {new_state}")
        
    def _is_valid_transition(self, current_state: GameState, new_state: GameState) -> bool:
        """
        Check if a state transition is valid.
        
        Args:
            current_state: Current game state
            new_state: Proposed new state
            
        Returns:
            bool: True if the transition is valid
        """
        # Define valid transitions between states
        valid_transitions = {
            GameState.INITIALIZING: [GameState.LOADING, GameState.ERROR],
            GameState.LOADING: [GameState.READY, GameState.ERROR],
            GameState.READY: [GameState.RUNNING, GameState.PAUSED, GameState.ERROR],
            GameState.RUNNING: [GameState.PAUSED, GameState.COMPLETED, GameState.ERROR],
            GameState.PAUSED: [GameState.RUNNING, GameState.READY, GameState.ERROR],
            GameState.COMPLETED: [GameState.READY, GameState.ERROR],
            GameState.ERROR: [GameState.INITIALIZING, GameState.READY],
        }
        
        # ERROR state can be reached from any state
        if new_state == GameState.ERROR:
            return True
            
        # Check if the transition is allowed
        return new_state in valid_transitions.get(current_state, [])

    def register_resource(
        self,
        resource_id: str,
        initial_state: str = "stable",
        threshold: Optional[float] = None,
        capacity: Optional[float] = None,
        flow_rate: Optional[float] = None,
    ) -> bool:
        """
        Register a resource for tracking.

        Args:
            resource_id: Resource identifier
            initial_state: Initial resource state
            threshold: Optional threshold value

        Returns:
            bool: True if registration successful
        """
        if resource_id in self.active_resources:
            logging.warning(f"Resource {resource_id} already registered")
            return False

        self.active_resources.add(resource_id)
        self.resource_states[resource_id] = initial_state

        # Store additional resource metadata
        if threshold is not None:
            self.resource_thresholds[resource_id] = threshold
        if capacity is not None:
            self.resource_capacities[resource_id] = capacity
        if flow_rate is not None:
            self.resource_flow_rates[resource_id] = flow_rate

        # Initialize flow tracking
        self.active_flows[resource_id] = {}

        # Notify observers of new resource
        self.dispatch_event(
            GameEvent(
                type=GameEventType.RESOURCE_CREATED,
                source=resource_id,
                data={
                    "state": initial_state,
                    "threshold": threshold,
                    "capacity": capacity,
                    "flow_rate": flow_rate,
                },
                timestamp=datetime.now().timestamp(),
            )
        )

        logging.info(f"Registered resource {resource_id}")
        return True

    def register_module(
        self,
        module_id: str,
        initial_state: str = "inactive",
        dependencies: Optional[Set[str]] = None,
    ) -> bool:
        """
        Register a module for tracking.

        Args:
            module_id: Module identifier
            initial_state: Initial module state
            dependencies: Optional set of module dependencies

        Returns:
            bool: True if registration successful
        """
        if module_id in self.active_modules:
            logging.warning(f"Module {module_id} already registered")
            return False

        self.active_modules.add(module_id)
        self.module_states[module_id] = initial_state
        self.module_dependencies[module_id] = dependencies or set()

        logging.info(f"Registered module {module_id}")
        return True

    def _process_events(self) -> None:
        """Process queued events."""
        # Sort by priority
        self.event_queue.sort(key=lambda e: e.priority, reverse=True)

        while self.event_queue:
            event = self.event_queue.pop(0)
            for handler in self.event_handlers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    logging.error(f"Error in event handler: {e}")

    def register_resource_flow(
        self, source_id: str, dest_id: str, flow_rate: float
    ) -> bool:
        """Register a resource flow between two resources.

        Args:
            source_id: Source resource ID
            dest_id: Destination resource ID
            flow_rate: Flow rate (units per second)

        Returns:
            bool: True if flow registered successfully
        """
        if source_id not in self.active_resources:
            logging.error(f"Source resource {source_id} not found")
            return False
        if dest_id not in self.active_resources:
            logging.error(f"Destination resource {dest_id} not found")
            return False

        # Register the flow
        self.active_flows[source_id][dest_id] = flow_rate

        # Record flow start event
        flow_event = {
            "type": "flow_start",
            "source": source_id,
            "destination": dest_id,
            "rate": flow_rate,
            "timestamp": datetime.now().timestamp(),
        }
        self.flow_history.append(flow_event)
        if len(self.flow_history) > self.max_flow_history:
            self.flow_history.pop(0)

        # Notify observers
        self.dispatch_event(
            GameEvent(
                type=GameEventType.RESOURCE_FLOW_STARTED,
                source=source_id,
                data={"destination": dest_id, "rate": flow_rate},
                timestamp=datetime.now().timestamp(),
            )
        )

        logging.info(
            f"Registered resource flow: {source_id} -> {dest_id} at {flow_rate}/s"
        )
        return True

    def stop_resource_flow(self, source_id: str, dest_id: str) -> bool:
        """Stop a resource flow.

        Args:
            source_id: Source resource ID
            dest_id: Destination resource ID

        Returns:
            bool: True if flow stopped successfully
        """
        if source_id not in self.active_flows:
            logging.error(f"Source resource {source_id} not found")
            return False
        if dest_id not in self.active_flows[source_id]:
            logging.error(f"No flow found from {source_id} to {dest_id}")
            return False

        # Record flow rate before removing
        flow_rate = self.active_flows[source_id][dest_id]

        # Remove the flow
        del self.active_flows[source_id][dest_id]

        # Record flow stop event
        flow_event = {
            "type": "flow_stop",
            "source": source_id,
            "destination": dest_id,
            "final_rate": flow_rate,
            "timestamp": datetime.now().timestamp(),
        }
        self.flow_history.append(flow_event)
        if len(self.flow_history) > self.max_flow_history:
            self.flow_history.pop(0)

        # Notify observers
        self.dispatch_event(
            GameEvent(
                type=GameEventType.RESOURCE_FLOW_STOPPED,
                source=source_id,
                data={"destination": dest_id, "final_rate": flow_rate},
                timestamp=datetime.now().timestamp(),
            )
        )

        logging.info(f"Stopped resource flow: {source_id} -> {dest_id}")
        return True

    def get_resource_flows(self, resource_id: str) -> Dict[str, float]:
        """Get all active flows for a resource.

        Args:
            resource_id: Resource to check

        Returns:
            Dict[str, float]: Map of destination IDs to flow rates
        """
        return self.active_flows.get(resource_id, {}).copy()

    def _update_resource_states(self) -> None:
        """Update resource states and check thresholds."""
        # Process active flows
        for source_id, flows in self.active_flows.items():
            for dest_id, rate in flows.items():
                # Update flow tracking
                flow_event = {
                    "type": "flow_update",
                    "source": source_id,
                    "destination": dest_id,
                    "rate": rate,
                    "timestamp": datetime.now().timestamp(),
                }
                self.flow_history.append(flow_event)
                if len(self.flow_history) > self.max_flow_history:
                    self.flow_history.pop(0)
        for resource_id in self.active_resources:
            if resource_id in self.resource_thresholds:
                threshold = self.resource_thresholds[resource_id]
                current_state = self.resource_states[resource_id]

                # Check if we need to trigger threshold events
                if current_state == "critical" and threshold > 0:
                    self.dispatch_event(
                        GameEvent(
                            type=GameEventType.RESOURCE_THRESHOLD_REACHED,
                            source=resource_id,
                            data={
                                "threshold": threshold,
                                "state": current_state,
                                "previous_state": self.previous_state,
                            },
                            timestamp=datetime.now().timestamp(),
                            priority=2,  # Higher priority for threshold events
                        )
                    )

    def _update_module_states(self) -> None:
        """Update module states and check dependencies."""
        for module_id in self.active_modules:
            dependencies = self.module_dependencies[module_id]
            if not dependencies.issubset(self.active_modules):
                logging.warning(f"Module {module_id} missing dependencies")

    def _check_thresholds(self) -> None:
        """Check and handle threshold triggers."""
        for resource_id, threshold in self.resource_thresholds.items():
            # Threshold checking logic would go here
            pass

    def shutdown(self) -> None:
        """Shutdown the game context."""
        self.active = False
        self.transition_state(GameState.SHUTTING_DOWN)
        logging.info("GameContext shut down")
