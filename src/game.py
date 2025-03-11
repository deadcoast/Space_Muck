"""
Game class implementation for Space Muck.

This module implements the core Game class with enhanced state management,
including state transitions, validation, and history tracking.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from config import GAME_CONFIG

@dataclass
class StateTransition:
    """Represents a state transition with timing information."""
    from_state: str
    to_state: str
    timestamp: datetime
    duration: float
    success: bool
    error: Optional[str] = None

class GameStateError(Exception):
    """Base exception for game state errors."""
    pass

class InvalidStateTransitionError(GameStateError):
    """Raised when an invalid state transition is attempted."""
    pass

class StateValidationError(GameStateError):
    """Raised when state validation fails."""
    pass

class Game:
    """Main game class with enhanced state management."""

    def __init__(self):
        """Initialize the game with state management."""
        self._state = GAME_CONFIG["initial_state"]
        self._state_history: List[StateTransition] = []
        self._state_valid = True
        self._transition_handlers: Dict[str, Dict[str, Callable]] = {}
        self._validation_handlers: Dict[str, List[Callable]] = {}
        self._entry_handlers: Dict[str, List[Callable]] = {}
        self._exit_handlers: Dict[str, List[Callable]] = {}
        self._setup_handlers()
        self._logger = logging.getLogger(__name__)

    def _setup_handlers(self) -> None:
        """Set up default state transition handlers."""
        # Example handler setup - to be expanded based on game needs
        self._validation_handlers = {
            "check_save_game": lambda: True,  # Placeholder
            "check_player_alive": lambda: self._state_valid,
            "check_game_initialized": lambda: True,  # Placeholder
            "check_game_active": lambda: self._state_valid,
            "check_shop_available": lambda: True,  # Placeholder
            "check_map_available": lambda: True,  # Placeholder
        }

        self._entry_handlers = {
            "reset_game": lambda: None,  # Placeholder
            "show_menu": lambda: None,  # Placeholder
            "resume_game": lambda: None,  # Placeholder
            "hide_menu": lambda: None,  # Placeholder
            "pause_game": lambda: None,  # Placeholder
            "show_pause_menu": lambda: None,  # Placeholder
            "show_shop": lambda: None,  # Placeholder
            "show_map": lambda: None,  # Placeholder
            "show_game_over": lambda: None,  # Placeholder
        }

        self._exit_handlers = {
            "hide_menu": lambda: None,  # Placeholder
            "save_game_state": lambda: None,  # Placeholder
            "hide_pause_menu": lambda: None,  # Placeholder
            "hide_shop": lambda: None,  # Placeholder
            "save_purchases": lambda: None,  # Placeholder
            "hide_map": lambda: None,  # Placeholder
            "save_high_score": lambda: None,  # Placeholder
            "reset_game": lambda: None,  # Placeholder
        }

    def _validate_state_transition(self, new_state: str) -> bool:
        """Validate if a state transition is allowed."""
        if new_state not in GAME_CONFIG["states"]:
            raise InvalidStateTransitionError(f"Invalid state: {new_state}")

        current_state_config = GAME_CONFIG["states"][self._state]
        if new_state not in current_state_config["allowed_transitions"]:
            raise InvalidStateTransitionError(
                f"Cannot transition from {self._state} to {new_state}"
            )

        # Run validation rules for the new state
        new_state_config = GAME_CONFIG["states"][new_state]
        for rule in new_state_config["validation_rules"]:
            if rule in self._validation_handlers and not self._validation_handlers[rule]():
                raise StateValidationError(f"Validation failed: {rule}")

        return True

    def _execute_state_actions(self, state: str, action_type: str) -> None:
        """Execute entry or exit actions for a state."""
        state_config = GAME_CONFIG["states"][state]
        actions = state_config.get(f"{action_type}_actions", [])
        handlers = self._entry_handlers if action_type == "entry" else self._exit_handlers

        for action in actions:
            if action in handlers:
                try:
                    start_time = time.time()
                    handlers[action]()
                    duration = time.time() - start_time
                    if duration > GAME_CONFIG["state_timing"]["action_timeout"]:
                        self._logger.warning(
                            f"{action_type.capitalize()} action {action} took {duration:.2f}s"
                        )
                except Exception as e:
                    self._logger.error(f"Error in {action_type} action {action}: {e}")
                    raise

    def change_state(self, new_state: str) -> None:
        """Change the game state with validation and history tracking."""
        start_time = time.time()
        transition = StateTransition(
            from_state=self._state,
            to_state=new_state,
            timestamp=datetime.now(),
            duration=0.0,
            success=False
        )

        try:
            # Validate the transition
            if self._validate_state_transition(new_state):
                self._extracted_from_change_state_16(new_state, start_time, transition)
        except GameStateError as e:
            self._extracted_from_change_state_42(
                e, transition, 'State transition failed: '
            )
        except Exception as e:
            self._extracted_from_change_state_42(
                e, transition, 'Unexpected error in state transition: '
            )

    # TODO Rename this here and in `change_state`
    def _extracted_from_change_state_16(self, new_state, start_time, transition):
        # Execute exit actions for current state
        self._execute_state_actions(self._state, "exit")

        # Update state
        old_state = self._state
        self._state = new_state

        # Execute entry actions for new state
        self._execute_state_actions(new_state, "entry")

        # Update transition record
        duration = time.time() - start_time
        transition.duration = duration
        transition.success = True

        # Trim history if needed
        if len(self._state_history) >= GAME_CONFIG["state_history_limit"]:
            self._state_history.pop(0)

        # Add to history
        self._state_history.append(transition)

        # Log transition
        self._logger.info(
            f"State transition: {old_state} -> {new_state} ({duration:.3f}s)"
        )

    # TODO Rename this here and in `change_state`
    def _extracted_from_change_state_42(self, e, transition, arg2):
        transition.error = str(e)
        self._logger.error(f"{arg2}{e}")
        self._state_valid = False
        raise

    @property
    def state(self) -> str:
        """Get the current game state."""
        return self._state

    @property
    def state_history(self) -> List[StateTransition]:
        """Get the state transition history."""
        return self._state_history.copy()

    @property
    def is_state_valid(self) -> bool:
        """Check if the current state is valid."""
        return self._state_valid

    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed information about the current state."""
        if self._state not in GAME_CONFIG["states"]:
            return {}
        
        state_config = GAME_CONFIG["states"][self._state]
        return {
            "name": state_config["name"],
            "current_state": self._state,
            "allowed_transitions": state_config["allowed_transitions"],
            "is_valid": self._state_valid,
            "history_count": len(self._state_history),
            "last_transition": self._state_history[-1] if self._state_history else None
        }
