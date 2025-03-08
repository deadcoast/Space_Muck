"""
Threshold Context: Manages threshold monitoring and triggers.

This module provides a context for managing thresholds, triggers,
and notifications within the game architecture.
"""

import logging
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime
from .game_context import GameContext, GameEventType, GameEvent


# Threshold Types
class ThresholdType(Enum):
    RESOURCE = auto()  # Resource-based thresholds
    MODULE = auto()  # Module state thresholds
    TIME = auto()  # Time-based thresholds
    EVENT = auto()  # Event-based thresholds
    COMPOSITE = auto()  # Multiple condition thresholds


# Threshold States
class ThresholdState(Enum):
    INACTIVE = auto()  # Threshold not being monitored
    MONITORING = auto()  # Actively monitoring
    TRIGGERED = auto()  # Threshold condition met
    CLEARED = auto()  # Condition no longer met
    ERROR = auto()  # Error in monitoring


@dataclass
class ThresholdInfo:
    """Information about a threshold."""

    id: str
    type: ThresholdType
    target: str
    condition: str
    state: ThresholdState
    value: float
    last_check: float = 0.0
    trigger_count: int = 0
    error_count: int = 0


class ThresholdContext:
    """
    Context for managing thresholds and triggers.
    """

    # Valid threshold conditions
    VALID_CONDITIONS = frozenset([">", "<", ">=", "<=", "=="])

    # Valid state transitions
    VALID_STATE_TRANSITIONS = {
        ThresholdState.INACTIVE: {ThresholdState.MONITORING, ThresholdState.ERROR},
        ThresholdState.MONITORING: {
            ThresholdState.TRIGGERED,
            ThresholdState.CLEARED,
            ThresholdState.ERROR,
            ThresholdState.INACTIVE,
        },
        ThresholdState.TRIGGERED: {
            ThresholdState.CLEARED,
            ThresholdState.ERROR,
            ThresholdState.INACTIVE,
        },
        ThresholdState.CLEARED: {
            ThresholdState.MONITORING,
            ThresholdState.ERROR,
            ThresholdState.INACTIVE,
        },
        ThresholdState.ERROR: {ThresholdState.MONITORING, ThresholdState.INACTIVE},
    }

    def __init__(self, game_context: GameContext) -> None:
        """Initialize the threshold context.

        Args:
            game_context: The game context to integrate with
        """
        # Threshold tracking
        self.thresholds: Dict[str, ThresholdInfo] = {}
        self.active_thresholds: Set[str] = set()

        # Trigger tracking
        self.trigger_history: List[Dict[str, any]] = []
        self.max_history = 1000

        # Notification system
        self.notification_handlers: Dict[
            str, List[Callable[[str, Dict[str, any]], None]]
        ] = {}

        # Game context integration
        self.game_context = game_context

        logging.info("ThresholdContext initialized")

    def _validate_condition(self, condition: str) -> bool:
        """Validate a threshold condition.

        Args:
            condition: Condition to validate

        Returns:
            bool: True if condition is valid
        """
        if not isinstance(condition, str):
            logging.error(f"Condition must be a string, got {type(condition)}")
            return False

        if condition not in self.VALID_CONDITIONS:
            logging.error(
                f"Invalid condition: {condition}. Must be one of {self.VALID_CONDITIONS}"
            )
            return False

        return True

    def register_threshold(
        self,
        threshold_id: str,
        type: ThresholdType,
        target: str,
        condition: str,
        value: float,
    ) -> bool:
        """
        Register a new threshold.

        Args:
            threshold_id: Unique threshold identifier
            type: Type of threshold
            target: Target entity (resource, module, etc.)
            condition: Condition expression
            value: Threshold value

        Returns:
            bool: True if registration successful
        """
        # Validate inputs
        if not isinstance(threshold_id, str) or not threshold_id.strip():
            logging.error("Threshold ID must be a non-empty string")
            return False

        if not isinstance(type, ThresholdType):
            logging.error(
                f"Invalid threshold type: {type}. Must be a ThresholdType enum"
            )
            return False

        if not isinstance(target, str) or not target.strip():
            logging.error("Target must be a non-empty string")
            return False

        if not isinstance(value, (int, float)):
            logging.error(f"Invalid threshold value: {value}. Must be a number")
            return False

        # Check for existing threshold
        if threshold_id in self.thresholds:
            logging.warning(f"Threshold {threshold_id} already registered")
            return False

        # Validate condition
        if not self._validate_condition(condition):
            return False

        try:
            # Create threshold info
            threshold_info = ThresholdInfo(
                id=threshold_id,
                type=type,
                target=target,
                condition=condition,
                state=ThresholdState.INACTIVE,
                value=value,  # Value already validated as number
            )

            # Update tracking
            self.thresholds[threshold_id] = threshold_info
            self.notification_handlers[threshold_id] = []

            # Notify game context
            self.game_context.dispatch_event(
                GameEvent(
                    type=GameEventType.THRESHOLD_REGISTERED,  # Changed from TRIGGERED to REGISTERED
                    source=threshold_id,
                    data={
                        "type": type.name,
                        "target": target,
                        "condition": condition,
                        "value": value,
                    },
                    timestamp=datetime.now().timestamp(),
                )
            )

            logging.info(
                f"Registered threshold {threshold_id} for {target} "
                f"with condition: {condition} {value}"
            )
            return True

        except Exception as e:
            logging.error(f"Error registering threshold {threshold_id}: {e}")
            return False

    def _change_threshold_state(
        self, threshold_id: str, new_state: ThresholdState
    ) -> bool:
        """Change a threshold's state.

        Args:
            threshold_id: Threshold to change state for
            new_state: Target state

        Returns:
            bool: True if state change successful
        """
        threshold = self.thresholds.get(threshold_id)
        if not threshold:
            logging.error(f"Cannot change state for unknown threshold {threshold_id}")
            return False

        try:
            old_state = threshold.state
            return self._record_state_change(threshold_id, old_state, new_state)

        except Exception as e:
            logging.error(f"Error changing threshold state: {e}")
            return False

    def activate_threshold(self, threshold_id: str) -> bool:
        """
        Activate a threshold for monitoring.

        Args:
            threshold_id: Threshold to activate

        Returns:
            bool: True if activation successful
        """
        if threshold_id not in self.thresholds:
            logging.error(f"Threshold {threshold_id} not found")
            return False

        try:
            if not self._change_threshold_state(
                threshold_id, ThresholdState.MONITORING
            ):
                return False

            # Add to active set
            self.active_thresholds.add(threshold_id)

            logging.info(f"Activated threshold {threshold_id}")
            return True

        except Exception as e:
            logging.error(f"Error activating threshold {threshold_id}: {e}")
            return False

    def check_threshold(self, threshold_id: str, current_value: float) -> bool:
        """
        Check if a threshold has been triggered.

        Args:
            threshold_id: Threshold to check
            current_value: Current value to check against threshold

        Returns:
            bool: True if threshold state changed successfully
        """
        if threshold_id not in self.thresholds:
            logging.error(f"Threshold {threshold_id} not found")
            return False

        threshold = self.thresholds[threshold_id]

        # Update last check time
        threshold.last_check = datetime.now().timestamp()

        try:
            # Check if threshold triggered
            triggered = self._evaluate_condition(
                threshold.condition, current_value, threshold.value
            )

            # Handle state changes if needed
            if triggered and threshold.state != ThresholdState.TRIGGERED:
                return self._handle_threshold_triggered(
                    threshold_id, threshold, current_value
                )
            elif not triggered and threshold.state == ThresholdState.TRIGGERED:
                return self._handle_threshold_cleared(
                    threshold_id, threshold, current_value
                )

            return True  # No state change needed

        except ValueError as ve:
            logging.error(f"Validation error checking threshold {threshold_id}: {ve}")
            # Change to error state on validation failures
            return self._change_threshold_state(threshold_id, ThresholdState.ERROR)

        except Exception as e:
            logging.error(f"Unexpected error checking threshold {threshold_id}: {e}")
            # Change to error state on unexpected errors
            return self._change_threshold_state(threshold_id, ThresholdState.ERROR)

    def _handle_threshold_triggered(
        self, threshold_id: str, threshold: ThresholdInfo, current_value: float
    ) -> bool:
        """Handle a threshold being triggered.

        Args:
            threshold_id: ID of the triggered threshold
            threshold: Threshold info
            current_value: Current value that triggered threshold

        Returns:
            bool: True if handling successful
        """
        try:
            # Update trigger count
            threshold.trigger_count += 1

            # Change state
            if not self._change_threshold_state(threshold_id, ThresholdState.TRIGGERED):
                return False

            # Prepare notification data
            notification_data = {
                "action": "triggered",
                "current_value": current_value,
                "threshold_value": threshold.value,
                "trigger_count": threshold.trigger_count,
            }

            # Notify handlers
            handlers_notified = self._notify_handlers(threshold_id, notification_data)

            # Notify game context
            self.game_context.dispatch_event(
                GameEvent(
                    type=GameEventType.THRESHOLD_TRIGGERED,
                    source=threshold_id,
                    data={"target": threshold.target, **notification_data},
                    timestamp=datetime.now().timestamp(),
                    priority=2,  # Higher priority for threshold triggers
                )
            )

            # Return true only if state change and notifications were successful
            return handlers_notified

        except Exception as e:
            logging.error(f"Error handling threshold trigger: {e}")
            return False

    def _handle_threshold_cleared(
        self, threshold_id: str, threshold: ThresholdInfo, current_value: float
    ) -> bool:
        """Handle a threshold being cleared.

        Args:
            threshold_id: ID of the cleared threshold
            threshold: Threshold info
            current_value: Current value that cleared threshold

        Returns:
            bool: True if handling successful
        """
        try:
            # Change state
            if not self._change_threshold_state(threshold_id, ThresholdState.CLEARED):
                return False

            # Prepare notification data
            notification_data = {
                "action": "cleared",
                "current_value": current_value,
                "threshold_value": threshold.value,
            }

            # Notify handlers
            handlers_notified = self._notify_handlers(threshold_id, notification_data)

            # Notify game context
            self.game_context.dispatch_event(
                GameEvent(
                    type=GameEventType.THRESHOLD_CLEARED,
                    source=threshold_id,
                    data={"target": threshold.target, **notification_data},
                    timestamp=datetime.now().timestamp(),
                )
            )

            # Return true only if state change and notifications were successful
            return handlers_notified

        except Exception as e:
            logging.error(f"Error handling threshold clear: {e}")
            return False

    def _validate_notification_handler(
        self, threshold_id: str, handler: Callable[[str, Dict[str, any]], None]
    ) -> None:
        """Validate a notification handler registration.

        Args:
            threshold_id: Threshold to validate
            handler: Handler to validate

        Raises:
            ValueError: If validation fails
        """
        if threshold_id not in self.thresholds:
            raise ValueError(
                f"Cannot register handler for unknown threshold {threshold_id}"
            )

        if not callable(handler):
            raise ValueError(f"Handler must be callable, got {type(handler)}")

    def register_notification_handler(
        self, threshold_id: str, handler: Callable[[str, Dict[str, any]], None]
    ) -> bool:
        """
        Register a notification handler for a threshold.

        Args:
            threshold_id: Threshold to handle notifications for
            handler: Notification handler callback

        Returns:
            bool: True if registration successful

        Raises:
            ValueError: If threshold_id is invalid or handler is not callable
        """
        try:
            self._validate_notification_handler(threshold_id, handler)

            # Initialize handler list if needed
            if threshold_id not in self.notification_handlers:
                self.notification_handlers[threshold_id] = []
            elif handler in self.notification_handlers[threshold_id]:
                logging.warning(
                    f"Handler already registered for threshold {threshold_id}"
                )
                return True

            # Register handler
            self.notification_handlers[threshold_id].append(handler)
            logging.info(
                f"Registered notification handler {handler.__name__ if hasattr(handler, '__name__') else str(handler)} "
                f"for threshold {threshold_id}"
            )
            return True

        except ValueError as ve:
            logging.error(f"Validation error registering handler: {ve}")
            raise

        except Exception as e:
            logging.error(f"Unexpected error registering handler: {e}")
            return False

    def get_threshold_state(self, threshold_id: str) -> Optional[ThresholdState]:
        """
        Get the current state of a threshold.

        Args:
            threshold_id: Threshold to check

        Returns:
            Optional[ThresholdState]: Current threshold state or None if not found
        """
        return (
            self.thresholds[threshold_id].state
            if threshold_id in self.thresholds
            else None
        )

    def _evaluate_condition(
        self, condition: str, current_value: float, threshold_value: float
    ) -> bool:
        """
        Evaluate a threshold condition.

        Args:
            condition: Condition expression (pre-validated)
            current_value: Current value
            threshold_value: Threshold value

        Returns:
            bool: True if condition met

        Raises:
            ValueError: If condition is invalid or values cannot be compared
        """
        try:
            # Validate inputs
            if not isinstance(current_value, (int, float)) or not isinstance(
                threshold_value, (int, float)
            ):
                raise ValueError(
                    f"Invalid value types: current={type(current_value)}, threshold={type(threshold_value)}"
                )

            if condition not in self.VALID_CONDITIONS:
                raise ValueError(f"Invalid condition: {condition}")

            # Direct mapping with validated condition
            result = {
                ">": current_value > threshold_value,
                "<": current_value < threshold_value,
                ">=": current_value >= threshold_value,
                "<=": current_value <= threshold_value,
                "==": current_value == threshold_value,
            }[condition]

            logging.debug(
                f"Evaluated condition: {current_value} {condition} {threshold_value} = {result}"
            )
            return result

        except ValueError as ve:
            logging.error(f"Validation error in condition evaluation: {ve}")
            raise

        except Exception as e:
            logging.error(
                f"Unexpected error evaluating condition '{condition}' with values "
                f"current={current_value}, threshold={threshold_value}: {e}"
            )
            raise ValueError(f"Failed to evaluate condition: {e}") from e

    def _notify_handlers(self, threshold_id: str, data: Dict[str, any]) -> bool:
        """
        Notify handlers for a threshold.

        Args:
            threshold_id: Threshold that triggered notification
            data: Notification data

        Returns:
            bool: True if all handlers were notified successfully
        """
        if threshold_id not in self.notification_handlers:
            logging.debug(f"No handlers registered for threshold {threshold_id}")
            return True

        success = True
        handler_count = len(self.notification_handlers[threshold_id])
        failed_handlers = 0

        for handler in self.notification_handlers[threshold_id]:
            try:
                handler(threshold_id, data)
                logging.debug(
                    f"Successfully notified handler for threshold {threshold_id}"
                )

            except Exception as e:
                failed_handlers += 1
                success = False
                logging.error(
                    f"Error in notification handler for threshold {threshold_id}: {e}\n"
                    f"Handler: {handler.__name__ if hasattr(handler, '__name__') else str(handler)}"
                )

        if failed_handlers > 0:
            logging.warning(
                f"{failed_handlers}/{handler_count} handlers failed for threshold {threshold_id}"
            )

        return success

    def _validate_state_transition(
        self, old_state: ThresholdState, new_state: ThresholdState
    ) -> bool:
        """Validate a state transition.

        Args:
            old_state: Current state
            new_state: Target state

        Returns:
            bool: True if transition is valid
        """
        if old_state == new_state:
            return True

        valid_transitions = self.VALID_STATE_TRANSITIONS.get(old_state, set())
        if new_state not in valid_transitions:
            logging.error(
                f"Invalid state transition from {old_state.name} to {new_state.name}. "
                f"Valid transitions: {[s.name for s in valid_transitions]}"
            )
            return False

        return True

    def _record_state_change(
        self,
        threshold_id: str,
        old_state: Optional[ThresholdState],
        new_state: ThresholdState,
    ) -> bool:
        """
        Record a threshold state change.

        Args:
            threshold_id: Threshold that changed state
            old_state: Previous state
            new_state: New state

        Returns:
            bool: True if state change was recorded
        """
        # Validate state transition
        if old_state and not self._validate_state_transition(old_state, new_state):
            return False

        # Add to history with additional context
        threshold = self.thresholds.get(threshold_id)
        if not threshold:
            logging.error(
                f"Cannot record state change for unknown threshold {threshold_id}"
            )
            return False

        event = {
            "threshold_id": threshold_id,
            "old_state": old_state.name if old_state else None,
            "new_state": new_state.name,
            "timestamp": datetime.now().timestamp(),
            "target": threshold.target,
            "type": threshold.type.name,
            "trigger_count": threshold.trigger_count,
            "error_count": threshold.error_count,
        }

        self.trigger_history.append(event)
        if len(self.trigger_history) > self.max_history:
            self.trigger_history.pop(0)

        # Notify game context of state change
        self.game_context.dispatch_event(
            GameEvent(
                type=GameEventType.THRESHOLD_STATE_CHANGED,
                source=threshold_id,
                data={
                    "target": threshold.target,
                    "old_state": old_state.name if old_state else None,
                    "new_state": new_state.name,
                    "trigger_count": threshold.trigger_count,
                    "error_count": threshold.error_count,
                },
                timestamp=datetime.now().timestamp(),
            )
        )

        return True
