"""
Game class implementation for Space Muck.

This module implements the core Game class with enhanced state management,
including state transitions, validation, and history tracking.
"""


import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass

# Standard library imports
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# Local application imports
from config import GAME_CONFIG

# Constants
ERROR_UI_DICT_NOT_FOUND = "UI elements dictionary not found"

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
            "check_save_game": self._check_save_game,
            "check_player_alive": self._check_player_alive,
            "check_game_initialized": self._check_game_initialized,
            "check_game_active": self._check_game_active,
            "check_shop_available": self._check_shop_available,
            "check_map_available": self._check_map_available,
        }

        self._entry_handlers = {
            "reset_game": self._reset_game,
            "show_menu": self._show_menu,
            "resume_game": self._resume_game,
            "hide_menu": self._hide_menu,
            "pause_game": self._pause_game,
            "show_pause_menu": self._show_pause_menu,
            "show_shop": self._show_shop,
            "show_map": self._show_map,
            "show_game_over": self._show_game_over,
        }

        self._exit_handlers = {
            "hide_menu": self._exit_hide_menu,
            "save_game_state": self._exit_save_game_state,
            "hide_pause_menu": self._exit_hide_pause_menu,
            "hide_shop": self._exit_hide_shop,
            "save_purchases": self._exit_save_purchases,
            "hide_map": self._exit_hide_map,
            "save_high_score": self._exit_save_high_score,
            "reset_game": self._exit_reset_game,
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
            if (
                rule in self._validation_handlers
                and not self._validation_handlers[rule]()
            ):
                raise StateValidationError(f"Validation failed: {rule}")

        return True

    def _execute_state_actions(self, state: str, action_type: str) -> None:
        """Execute entry or exit actions for a state."""
        state_config = GAME_CONFIG["states"][state]
        actions = state_config.get(f"{action_type}_actions", [])
        handlers = (
            self._entry_handlers if action_type == "entry" else self._exit_handlers
        )

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
            success=False,
        )

        try:
            # Validate the transition
            if self._validate_state_transition(new_state):
                self._perform_state_transition_and_log(
                    new_state, start_time, transition
                )
        except GameStateError as e:
            self._handle_state_transition_error(
                e, transition, "State transition failed: "
            )
        except Exception as e:
            self._handle_state_transition_error(
                e, transition, "Unexpected error in state transition: "
            )

    def _perform_state_transition_and_log(self, new_state, start_time, transition):
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

    def _handle_state_transition_error(self, e, transition, error_prefix):
        transition.error = str(e)
        self._logger.error(f"{error_prefix}{e}")
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
            "last_transition": self._state_history[-1] if self._state_history else None,
        }

    def _check_save_game(self) -> bool:
        """Check if save game data is available and valid.

        This validation handler verifies the existence and integrity of save game data
        before allowing transitions to states that require saved game data.

        Returns:
            bool: True if valid save game exists, False otherwise.
        """
        try:
            # Log the validation attempt
            self._logger.info("Checking save game availability and integrity")

            # Get save directory and file paths
            save_dir, save_files = self._get_save_game_paths()

            # Check if save directory exists
            if not os.path.exists(save_dir):
                self._logger.warning(f"Save directory not found: {save_dir}")
                return False

            # Check if at least one save file exists
            if not save_files:
                self._logger.warning("No save files found in save directory")
                return False

            # Validate the most recent save file
            most_recent_save = self._get_most_recent_save(save_files)
            if not most_recent_save:
                self._logger.warning("Could not determine most recent save file")
                return False

            # Validate save data integrity
            is_valid, error_msg = self._validate_save_data(most_recent_save)
            if not is_valid:
                self._logger.error(f"Save game validation failed: {error_msg}")
                return False

            self._logger.info(f"Save game validation successful: {most_recent_save}")
            return True
        except Exception as e:
            self._logger.error(f"Error during save game validation: {e}")
            return False

    def _get_save_game_paths(self) -> Tuple[str, List[str]]:
        """Get the save game directory and list of save files.

        Returns:
            Tuple[str, List[str]]: Save directory path and list of save file paths
        """
        # Determine save directory based on OS
        home_dir = os.path.expanduser("~")

        # Usually save games would be in a specific location like:
        # Windows: ~/AppData/Local/SpaceMuck/saves/
        # macOS: ~/Library/Application Support/SpaceMuck/saves/
        # Linux: ~/.local/share/SpaceMuck/saves/

        if os.name == "nt":  # Windows
            save_dir = os.path.join(home_dir, "AppData", "Local", "SpaceMuck", "saves")
        elif os.name == "posix":  # macOS/Linux
            if os.path.exists(os.path.join(home_dir, "Library", "Application Support")):
                # macOS
                save_dir = os.path.join(
                    home_dir, "Library", "Application Support", "SpaceMuck", "saves"
                )
            else:
                # Linux
                save_dir = os.path.join(
                    home_dir, ".local", "share", "SpaceMuck", "saves"
                )
        else:
            # Fallback
            save_dir = os.path.join(os.getcwd(), "saves")

        # For development/testing, if the save directory doesn't exist,
        # check in the current directory or the project root
        if not os.path.exists(save_dir):
            alt_save_dir = os.path.join(os.getcwd(), "saves")
            if os.path.exists(alt_save_dir):
                save_dir = alt_save_dir

        # Get list of save files (*.json)
        save_files = []
        if os.path.exists(save_dir):
            save_files = [
                os.path.join(save_dir, f)
                for f in os.listdir(save_dir)
                if f.endswith(".json") and os.path.isfile(os.path.join(save_dir, f))
            ]

        return save_dir, save_files

    def _get_most_recent_save(self, save_files: List[str]) -> Optional[str]:
        """Get the most recent save file based on modification time.

        Args:
            save_files: List of save file paths

        Returns:
            Optional[str]: Path to the most recent save file, or None if no files
        """
        if not save_files:
            return None

        # Sort files by modification time (newest first)
        recent_files = sorted(
            save_files, key=lambda x: os.path.getmtime(x), reverse=True
        )

        return recent_files[0] if recent_files else None

    def _validate_save_data(self, save_file_path: str) -> Tuple[bool, str]:
        """Validate the save data's integrity and format.

        Args:
            save_file_path: Path to the save file

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Check if file exists and is readable
            if not os.path.isfile(save_file_path):
                return False, f"Save file does not exist: {save_file_path}"

            # Check file size (empty files are invalid)
            if os.path.getsize(save_file_path) == 0:
                return False, "Save file is empty"

            # Read and parse the save file
            with open(save_file_path, "r") as file:
                save_data = json.load(file)

            # Validate required fields
            required_fields = [
                "version",
                "player",
                "game_state",
                "timestamp",
                "resources",
                "inventory",
            ]

            for field in required_fields:
                if field not in save_data:
                    return False, f"Save file missing required field: {field}"

            # Validate version
            if not isinstance(save_data["version"], (str, int, float)):
                return False, "Invalid version format"

            # Validate player data
            if not isinstance(save_data["player"], dict):
                return False, "Invalid player data format"

            player_required = ["name", "level", "health"]
            return next(
                (
                    (False, f"Player data missing field: {field}")
                    for field in player_required
                    if field not in save_data["player"]
                ),
                (
                    (True, "")
                    if isinstance(save_data["game_state"], str)
                    else (False, "Invalid game state format")
                ),
            )
        except json.JSONDecodeError:
            return False, "Save file contains invalid JSON"
        except Exception as e:
            return False, f"Save file validation error: {str(e)}"

    def _check_player_alive(self) -> bool:
        """Check if the player entity exists and is alive.

        This validation handler verifies that the player exists, has health
        above zero, and is not in a failed state that would prevent gameplay.

        Returns:
            bool: True if player exists and is alive, False otherwise.
        """
        try:
            # Log the validation attempt
            self._logger.info("Checking player health and status")

            # Check if player attribute exists
            if not hasattr(self, "player"):
                self._logger.warning("Player entity not found in game instance")
                return False

            # Get player reference
            player = getattr(self, "player")
            if player is None:
                self._logger.warning("Player entity is None")
                return False

            # Check health attribute
            if not hasattr(player, "health"):
                self._logger.warning("Player has no health attribute")
                return False

            # Check if player is alive (health > 0)
            player_health = getattr(player, "health")
            if not isinstance(player_health, (int, float)):
                self._logger.warning(
                    f"Invalid player health format: {type(player_health)}"
                )
                return False

            if player_health <= 0:
                self._logger.warning(f"Player is dead (health: {player_health})")
                return False

            # Check for status effects that might incapacitate the player
            if hasattr(player, "status_effects"):
                status_effects = getattr(player, "status_effects", [])
                fatal_effects = ["paralyzed", "terminated", "deactivated"]

                if any(effect in fatal_effects for effect in status_effects):
                    self._logger.warning(
                        f"Player has disabling status effect: {status_effects}"
                    )
                    return False

            self._logger.info(
                f"Player alive check successful (health: {player_health})"
            )
            return True

        except Exception as e:
            self._logger.error(f"Error during player alive validation: {e}")
            return False

    def _check_game_initialized(self) -> bool:
        """Verify that essential game resources and systems are loaded.

        This validation handler checks if critical game systems, resources, and
        components have been properly initialized and are ready for gameplay.

        Returns:
            bool: True if game is properly initialized, False otherwise.
        """
        try:
            # Log the validation attempt
            self._logger.info("Checking game initialization status")

            # Check core attributes and systems
            if not self._check_required_attributes():
                return False

            # Check resource manager
            if not self._check_resource_manager():
                return False

            # Check event system
            if not self._check_event_system():
                return False

            # Check game initialization flag
            if not self._check_initialization_flag():
                return False

            # Check UI components
            if not self._check_ui_components():
                return False

            self._logger.info("Game initialization check successful")
            return True

        except Exception as e:
            self._logger.error(f"Error during game initialization validation: {e}")
            return False

    def _check_required_attributes(self) -> bool:
        """Check if required game attributes are present.

        Returns:
            bool: True if all required attributes exist, False otherwise.
        """
        # Core game systems and attributes to verify
        required_attributes = [
            "player",
            "field",
            "combat_system",
            "resource_manager",
            "event_bus",
            "ui_manager",
            "ui_components_initialized",
            "seed",
            "initialized",
        ]

        # Use walrus operator to simplify assignment and conditional
        if missing_attributes := [
            attr for attr in required_attributes if not hasattr(self, attr)
        ]:
            self._logger.warning(
                f"Missing required game attributes: {', '.join(missing_attributes)}"
            )
            return False

        return True

    def _check_resource_manager(self) -> bool:
        """Check if resource manager is properly initialized.

        Returns:
            bool: True if resource manager is operational, False otherwise.
        """
        if not hasattr(self, "resource_manager"):
            return True  # Skip this check if attribute doesn't exist (will be caught in required attributes)

        if not (resource_manager := getattr(self, "resource_manager")) or not hasattr(
            resource_manager, "initialized"
        ):
            self._logger.warning("Resource manager not properly initialized")
            return False

        if not getattr(resource_manager, "initialized", False):
            self._logger.warning("Resources not loaded")
            return False

        return True

    def _check_event_system(self) -> bool:
        """Check if event system is operational.

        Returns:
            bool: True if event system is properly initialized, False otherwise.
        """
        if not hasattr(self, "event_bus"):
            return True  # Skip this check if attribute doesn't exist (will be caught in required attributes)

        if not (event_bus := getattr(self, "event_bus")) or not hasattr(
            event_bus, "publish"
        ):
            self._logger.warning("Event bus not properly initialized")
            return False

        return True

    def _check_initialization_flag(self) -> bool:
        """Check game's initialization flag.

        Returns:
            bool: True if initialization flag is set, False otherwise.
        """
        if hasattr(self, "initialized") and not getattr(self, "initialized", False):
            self._logger.warning("Game initialization flag is False")
            return False

        return True

    def _check_ui_components(self) -> bool:
        """Check if UI components are properly initialized.

        Returns:
            bool: True if UI is operational, False otherwise.
        """
        if (
            hasattr(self, "ui_manager")
            and hasattr(self, "ui_components_initialized")
            and not getattr(self, "ui_components_initialized", False)
        ):
            self._logger.warning("UI components not initialized")
            return False

        return True

    def _check_game_active(self) -> bool:
        """Verify that the game is in an active state.

        This validation handler checks if the game is currently active and not paused,
        which is required for gameplay, interactions, and certain UI operations.

        Returns:
            bool: True if game is active, False otherwise.
        """
        try:
            # Log the validation attempt
            self._logger.info("Checking game active status")

            # First verify the game is properly initialized
            if not self._check_game_initialized():
                self._logger.warning(
                    "Game is not active because it's not properly initialized"
                )
                return False

            # Check active flag
            if not self._check_active_flag():
                return False

            # Check pause state
            if not self._check_not_paused():
                return False

            self._logger.info("Game is active and ready for gameplay")
            return True

        except Exception as e:
            self._logger.error(f"Error during game active state validation: {e}")
            return False

    def _check_active_flag(self) -> bool:
        """Check if the game's active flag is set to True.

        Returns:
            bool: True if active flag is set, False otherwise.
        """
        if not hasattr(self, "active"):
            self._logger.warning("Game missing 'active' attribute")
            return False

        if not getattr(self, "active", False):
            self._logger.warning("Game is not in an active state")
            return False

        return True

    def _check_not_paused(self) -> bool:
        """Check if the game is not in a paused state.

        Returns:
            bool: True if game is not paused, False if paused.
        """
        if not hasattr(self, "paused"):
            self._logger.warning("Game missing 'paused' attribute")
            return True  # If no pause attribute exists, assume not paused

        if getattr(self, "paused", False):
            self._logger.warning("Game is currently paused")
            return False

        return True

    def _check_shop_available(self) -> bool:
        """Verify that the shop can be opened and accessed.

        This validation handler checks if the shop resources are loaded and available,
        and if the player meets the requirements (such as having sufficient currency)
        to access the shop.

        Returns:
            bool: True if shop is available, False otherwise.
        """
        try:
            # Log the validation attempt
            self._logger.info("Checking shop availability")

            # First verify the game is properly initialized and active
            if not self._check_game_initialized():
                self._logger.warning("Shop unavailable: Game not properly initialized")
                return False

            if not self._check_game_active():
                self._logger.warning("Shop unavailable: Game not in active state")
                return False

            # Check shop instance exists
            if not self._check_shop_instance():
                return False

            # Check player state and currency
            if not self._check_player_for_shop():
                return False

            # Check if required shop resources are loaded
            if not self._check_shop_resources():
                return False

            self._logger.info("Shop is available and ready for access")
            return True

        except Exception as e:
            self._logger.error(f"Error during shop availability validation: {e}")
            return False

    def _check_shop_instance(self) -> bool:
        """Check if the shop instance exists and is properly initialized.

        Returns:
            bool: True if shop instance exists, False otherwise.
        """
        if not hasattr(self, "shop"):
            self._logger.warning("Shop instance not found")
            return False

        if not (shop := getattr(self, "shop")):
            self._logger.warning("Shop instance is None or invalid")
            return False

        # Check for essential shop methods
        required_methods = ["get_available_upgrades", "purchase_upgrade", "update"]
        for method in required_methods:
            if not hasattr(shop, method):
                self._logger.warning(f"Shop instance missing required method: {method}")
                return False

        return True

    def _check_player_for_shop(self) -> bool:
        """Check if player exists and meets requirements for shop access.

        Returns:
            bool: True if player meets requirements, False otherwise.
        """
        if not hasattr(self, "player"):
            self._logger.warning("Player instance not found")
            return False

        player = getattr(self, "player")
        if not player:
            self._logger.warning("Player instance is None or invalid")
            return False

        # Check if player has currency attribute
        if not hasattr(player, "currency"):
            self._logger.warning("Player missing currency attribute")
            return False

        # Check minimum currency requirement
        # Default to 0 if GAME_CONFIG is not available or doesn't have the setting
        min_currency = 0
        if hasattr(self, "GAME_CONFIG") and isinstance(
            getattr(self, "GAME_CONFIG"), dict
        ):
            game_config = getattr(self, "GAME_CONFIG")
            min_currency = game_config.get("shop_entry_min_currency", 0)

        if player.currency < min_currency:
            self._logger.warning(
                f"Player has insufficient currency: {player.currency}/{min_currency}"
            )
            return False

        return True

    def _check_shop_resources(self) -> bool:
        """Check if required shop resources are loaded.

        Returns:
            bool: True if shop resources are loaded, False otherwise.
        """
        # Ensure resource manager is available
        if not hasattr(self, "resource_manager"):
            self._logger.warning("Resource manager not found for shop resources")
            return False

        if not (resource_manager := getattr(self, "resource_manager")):
            self._logger.warning("Resource manager is None or invalid")
            return False

        # If resource manager has a get_resource method, check for resource availability
        if hasattr(resource_manager, "get_resource"):
            # Check for shop-specific resources
            shop_resources = ["shop_icons", "upgrade_data", "item_textures"]

            for resource in shop_resources:
                # This is a soft check - we just log warnings but don't fail validation
                # for specific resources as some might be optional
                try:
                    if not resource_manager.get_resource(resource):
                        self._logger.warning(f"Shop resource not available: {resource}")
                except Exception as e:
                    self._logger.warning(
                        f"Error checking shop resource {resource}: {e}"
                    )

        return True

    def _check_map_available(self) -> bool:
        """Verify that the map/minimap can be displayed.

        This validation handler checks if the map resources and components are available,
        including the field data, rendering components, and player position.

        Returns:
            bool: True if map can be displayed, False otherwise.
        """
        try:
            # Log the validation attempt
            self._logger.info("Checking map availability")

            # First verify the game is properly initialized and active
            if not self._check_game_initialized():
                self._logger.warning("Map unavailable: Game not properly initialized")
                return False

            if not self._check_game_active():
                self._logger.warning("Map unavailable: Game not in active state")
                return False

            # Check if rendering components exist
            if not self._check_renderer_components():
                return False

            # Check player state and position data
            if not self._check_player_for_map():
                return False

            # Check if field data is available
            if not self._check_field_data():
                return False

            self._logger.info("Map is available and ready for display")
            return True

        except Exception as e:
            self._logger.error(f"Error during map availability validation: {e}")
            return False

    def _check_renderer_components(self) -> bool:
        """Check if required rendering components for the map exist.

        Returns:
            bool: True if rendering components exist, False otherwise.
        """
        # Check minimap panel
        if not hasattr(self, "minimap_panel"):
            self._logger.warning("Minimap panel not found")
            return False

        if not (minimap_panel := getattr(self, "minimap_panel")):
            self._logger.warning("Minimap panel is None or invalid")
            return False

        # Check if minimap panel has required methods
        required_methods = ["update", "draw"]
        for method in required_methods:
            if not hasattr(minimap_panel, method):
                self._logger.warning(f"Minimap panel missing required method: {method}")
                return False

        # Check renderer (used for field offsets and rendering)
        if not hasattr(self, "renderer"):
            self._logger.warning("Renderer component not found for map display")
            return False

        if not (renderer := getattr(self, "renderer")):
            self._logger.warning("Renderer is None or invalid")
            return False

        # Check renderer has required attributes for map display
        required_attrs = ["field_offset_x", "field_offset_y"]
        for attr in required_attrs:
            if not hasattr(renderer, attr):
                self._logger.warning(f"Renderer missing required attribute: {attr}")
                return False

        return True

    def _check_player_for_map(self) -> bool:
        """Check if player entity exists and has position data for map display.

        Returns:
            bool: True if player has valid position data, False otherwise.
        """
        if not hasattr(self, "player"):
            self._logger.warning("Player entity not found for map display")
            return False

        if not (player := getattr(self, "player")):
            self._logger.warning("Player entity is None or invalid")
            return False

        # Check player has position data
        required_attrs = ["x", "y"]
        for attr in required_attrs:
            if not hasattr(player, attr):
                self._logger.warning(
                    f"Player missing required position attribute: {attr}"
                )
                return False

        # Ensure coordinates are numeric values
        try:
            float(player.x)
            float(player.y)
        except (TypeError, ValueError):
            self._logger.warning(
                "Player position coordinates are not valid numeric values"
            )
            return False

        return True

    def _check_field_data(self) -> bool:
        """Check if field data required for map display is available.

        Returns:
            bool: True if field data is available, False otherwise.
        """
        if not hasattr(self, "field"):
            self._logger.warning("Field data not found for map display")
            return False

        if not (field := getattr(self, "field")):
            self._logger.warning("Field data is None or invalid")
            return False

        # Check if field has necessary grid data
        required_attrs = ["grid", "entity_grid"]
        for attr in required_attrs:
            if not hasattr(field, attr):
                self._logger.warning(f"Field missing required grid data: {attr}")
                return False

            # Check that grid data exists
            if not getattr(field, attr):
                self._logger.warning(f"Field {attr} is None or empty")
                return False

        return True

    def _reset_game(self) -> None:
        """Reset the game to its initial state.

        This state entry handler resets all game variables and components to their
        initial values, preparing for a new game session.
        """
        try:
            self._logger.info("Resetting game state to initial values")

            # Reset core game state
            self._reset_state_tracking()

            # Reset game components in sequence
            self._reset_player()
            self._reset_field()
            self._reset_ui_components()
            self._reset_game_counters()
            self._reset_event_system()

            self._logger.info("Game reset complete")

        except Exception as e:
            self._logger.error(f"Critical error during game reset: {e}")

    def _reset_state_tracking(self) -> None:
        """Reset state tracking variables to initial values."""
        self._state_history = []
        self._state_valid = True

    def _reset_player(self) -> None:
        """Reset player attributes, position, and inventory."""
        if not hasattr(self, "player"):
            return

        try:
            # Reset player position to starting coordinates
            if start_pos := GAME_CONFIG.get("player", {}).get("start_position"):
                self.player.x = start_pos.get("x", 0)
                self.player.y = start_pos.get("y", 0)

            # Reset player stats
            if start_stats := GAME_CONFIG.get("player", {}).get("initial_stats"):
                for stat, value in start_stats.items():
                    if hasattr(self.player, stat):
                        setattr(self.player, stat, value)

            self._reset_player_inventory()

        except Exception as e:
            self._logger.error(f"Error resetting player attributes: {e}")

    def _reset_player_inventory(self) -> None:
        """Reset player inventory and add starter items."""
        if not (
            hasattr(self, "player")
            and hasattr(self.player, "inventory")
            and hasattr(self.player.inventory, "clear")
        ):
            return

        try:
            # Clear inventory
            self.player.inventory.clear()

            # Add starter items
            if starter_items := GAME_CONFIG.get("player", {}).get("starter_items", []):
                if hasattr(self.player.inventory, "add_item"):
                    for item in starter_items:
                        self.player.inventory.add_item(item)
        except Exception as e:
            self._logger.error(f"Error resetting player inventory: {e}")

    def _reset_field(self) -> None:
        """Reset game field/map state."""
        if not (hasattr(self, "field") and self.field):
            return

        try:
            # Either regenerate the field or reset entities
            if hasattr(self.field, "reset"):
                self.field.reset()
            elif hasattr(self.field, "clear_entities"):
                self.field.clear_entities()

            self._logger.info("Field reset complete")
        except Exception as e:
            self._logger.error(f"Error resetting field: {e}")

    def _reset_ui_components(self) -> None:
        """Reset UI components to their default state."""
        ui_components = ["minimap_panel", "status_panel", "inventory_panel"]

        for ui_component in ui_components:
            if not hasattr(self, ui_component):
                continue

            component = getattr(self, ui_component)
            if hasattr(component, "reset"):
                try:
                    component.reset()
                except Exception as e:
                    self._logger.error(f"Error resetting {ui_component}: {e}")

    def _reset_game_counters(self) -> None:
        """Reset game time and turn counters."""
        if hasattr(self, "game_time"):
            self.game_time = 0
        if hasattr(self, "turn_counter"):
            self.turn_counter = 0

    def _reset_event_system(self) -> None:
        """Reset event system and active triggers."""
        if not (
            hasattr(self, "event_manager") and hasattr(self.event_manager, "reset")
        ):
            return

        try:
            self.event_manager.reset()
        except Exception as e:
            self._logger.error(f"Error resetting event manager: {e}")

    def _show_menu(self) -> None:
        """Display the main menu UI.

        This state entry handler activates and displays the main menu interface,
        ensuring proper visibility and focus for menu components.
        """
        try:
            self._logger.info("Displaying main menu interface")

            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return

            # Activate main menu if it exists
            if "main_menu" in self.ui_elements:
                self._main_menu_handler()
            else:
                self._logger.warning("Main menu component not found in UI elements")

            # Hide gameplay UI elements and update menu content
            self._hide_gameplay_elements()
            self._update_menu_content()

        except Exception as e:
            self._logger.error(f"Error displaying main menu: {e}")

    def _main_menu_handler(self) -> None:
        """Configure and activate the main menu UI component."""
        main_menu = self.ui_elements["main_menu"]

        # Set visibility and active state
        self._set_menu_visibility(main_menu, True)
        self._set_menu_active_state(main_menu)
        self._apply_menu_animation(main_menu)

        self._logger.info("Main menu activated successfully")
        
    def _set_menu_visibility(self, menu, is_visible: bool) -> None:
        """Set the visibility of a UI element."""
        if hasattr(menu, "visible"):
            menu.visible = is_visible
            
    def _set_menu_active_state(self, menu) -> None:
        """Set the active state of a menu and update active element tracking."""
        if hasattr(menu, "active"):
            menu.active = True
            
        # Update active element tracking if available
        if hasattr(self, "active_element"):
            self.active_element = "main_menu"
            
    def _apply_menu_animation(self, menu) -> None:
        """Apply animation effects to a menu if supported."""
        # Check if animation is supported
        if not (hasattr(menu, "animation") and 
                isinstance(menu.animation, dict) and 
                "style" in menu.animation):
            return
            
        # Check if fade-in animation is available
        if not (hasattr(self, "AnimationStyle") and 
                hasattr(self.AnimationStyle, "FADE_IN")):
            return
            
        # Apply fade-in animation
        menu.animation["style"] = self.AnimationStyle.FADE_IN
        menu.animation["active"] = True
        menu.animation["progress"] = 0.0
        menu.animation["start_time"] = time.time()
        menu.animation["duration"] = 0.5  # Half-second fade in
        
    def _hide_gameplay_elements(self) -> None:
        """Hide gameplay UI elements when showing the menu."""
        gameplay_elements = ["status_panel", "inventory_panel", "action_panel"]
        
        for element_name in gameplay_elements:
            if element_name not in self.ui_elements:
                continue
                
            element = self.ui_elements[element_name]
            if hasattr(element, "visible"):
                element.visible = False
                
    def _update_menu_content(self) -> None:
        """Update the menu content if the menu has an update method."""
        if "main_menu" not in self.ui_elements:
            return
            
        menu = self.ui_elements["main_menu"]
        if not hasattr(menu, "update"):
            return
            
        try:
            menu.update()
        except Exception as e:
            self._logger.error(f"Error updating main menu content: {e}")
            
    def _resume_game(self) -> None:
        """Resume game from paused state.
        
        This state entry handler restores the game from a paused state,
        reactivating all game systems and UI components needed for gameplay.
        """
        try:
            self._logger.info("Resuming game from paused state")
            
            # Resume core game modules
            self._resume_game_modules()
            
            # Restore UI components for gameplay
            self._restore_gameplay_ui()
            
            # Resume game timing if time tracking is used
            self._resume_game_timing()
            
            self._logger.info("Game resumed successfully")
            
        except Exception as e:
            self._logger.error(f"Error resuming game: {e}")
            
    def _resume_game_modules(self) -> None:
        """Resume all game modules that support pause/resume functionality."""
        modules_to_resume = [
            "event_manager", 
            "field", 
            "player", 
            "ai_manager",
            "physics_engine"
        ]
        
        for module_name in modules_to_resume:
            if not hasattr(self, module_name):
                continue
                
            module = getattr(self, module_name)
            if hasattr(module, "resume") and callable(module.resume):
                try:
                    module.resume()
                    self._logger.debug(f"Resumed module: {module_name}")
                except Exception as e:
                    self._logger.error(f"Error resuming {module_name}: {e}")
        
        # Set game paused state if tracking is available
        if hasattr(self, "paused"):
            self.paused = False
            
    def _restore_gameplay_ui(self) -> None:
        """Restore and show gameplay UI components."""
        if not hasattr(self, "ui_elements"):
            return
            
        # Show gameplay UI elements
        gameplay_elements = ["status_panel", "inventory_panel", "action_panel"]
        for element_name in gameplay_elements:
            if element_name not in self.ui_elements:
                continue
                
            element = self.ui_elements[element_name]
            if hasattr(element, "visible"):
                element.visible = True
                
        # Hide any menu components if they exist
        menu_elements = ["main_menu", "pause_menu"]
        for element_name in menu_elements:
            if element_name not in self.ui_elements:
                continue
                
            element = self.ui_elements[element_name]
            if hasattr(element, "visible"):
                element.visible = False
                
        # Update active element if tracking is available and status panel exists
        if hasattr(self, "active_element") and "status_panel" in self.ui_elements:
            self.active_element = "status_panel"
                
    def _resume_game_timing(self) -> None:
        """Resume game timing and update mechanisms."""
        # Resume any game timers or update mechanisms
        if hasattr(self, "game_clock") and hasattr(self.game_clock, "resume"):
            try:
                self.game_clock.resume()
            except Exception as e:
                self._logger.error(f"Error resuming game clock: {e}")
                
        # Resume any update loops or schedulers
        if hasattr(self, "update_scheduler") and hasattr(self.update_scheduler, "resume"):
            try:
                self.update_scheduler.resume()
            except Exception as e:
                self._logger.error(f"Error resuming update scheduler: {e}")
                
    def _hide_menu(self) -> None:
        """Hide the menu UI.
        
        This state entry handler hides any active menu interfaces and prepares
        for gameplay or other non-menu states.
        """
        try:
            self._logger.info("Hiding menu interface")
            
            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return
                
            # Hide main menu and related elements
            self._hide_menu_elements()
            
            # Reset any menu-related state
            self._reset_menu_state()
            
            self._logger.info("Menu hidden successfully")
            
        except Exception as e:
            self._logger.error(f"Error hiding menu: {e}")
            
    def _hide_menu_elements(self) -> None:
        """Hide all menu-related UI elements."""
        menu_elements = ["main_menu", "pause_menu", "settings_menu", "help_menu"]
        
        for element_name in menu_elements:
            if element_name not in self.ui_elements:
                continue
                
            element = self.ui_elements[element_name]
            # Deactivate the menu
            if hasattr(element, "active"):
                element.active = False
                
            # Hide the menu
            if hasattr(element, "visible"):
                element.visible = False
                
            # Stop any active animations
            if (hasattr(element, "animation") and isinstance(element.animation, dict) and
                    "active" in element.animation):
                element.animation["active"] = False
            
    def _reset_menu_state(self) -> None:
        """Reset menu-related state variables."""
        # Clear active element if it's a menu
        if (hasattr(self, "active_element") and
                self.active_element in ["main_menu", "pause_menu", "settings_menu"]):
                # Default to gameplay or null depending on what's available
                if "status_panel" in self.ui_elements:
                    self.active_element = "status_panel"
                else:
                    self.active_element = None
                    
        # Reset any menu-specific flags
        menu_flags = ["menu_active", "in_menu_transition", "menu_selection"]
        for flag in menu_flags:
            if hasattr(self, flag):
                setattr(self, flag, False)
                
    def _pause_game(self) -> None:
        """Pause game logic and timing.
        
        This state entry handler pauses all active game systems and modules,
        preparing for a pause menu or other interruption.
        """
        try:
            self._logger.info("Pausing game")
            
            # Pause all active game modules
            self._pause_game_modules()
            
            # Pause game timing mechanisms
            self._pause_game_timing()
            
            # Store current game state for potential resumption
            self._store_pause_state()
            
            self._logger.info("Game paused successfully")
            
        except Exception as e:
            self._logger.error(f"Error pausing game: {e}")
            
    def _pause_game_modules(self) -> None:
        """Pause all active game modules."""
        modules_to_pause = [
            "event_manager", 
            "field", 
            "player", 
            "ai_manager",
            "physics_engine"
        ]
        
        for module_name in modules_to_pause:
            if not hasattr(self, module_name):
                continue
                
            module = getattr(self, module_name)
            if hasattr(module, "pause") and callable(module.pause):
                try:
                    module.pause()
                    self._logger.debug(f"Paused module: {module_name}")
                except Exception as e:
                    self._logger.error(f"Error pausing {module_name}: {e}")
        
        # Set game paused state if tracking is available
        if hasattr(self, "paused"):
            self.paused = True
            
    def _pause_game_timing(self) -> None:
        """Pause game timing and update mechanisms."""
        # Pause game clock if available
        if hasattr(self, "game_clock") and hasattr(self.game_clock, "pause"):
            try:
                self.game_clock.pause()
                self._logger.debug("Paused game clock")
            except Exception as e:
                self._logger.error(f"Error pausing game clock: {e}")
                
        # Pause any update loops or schedulers
        if hasattr(self, "update_scheduler") and hasattr(self.update_scheduler, "pause"):
            try:
                self.update_scheduler.pause()
                self._logger.debug("Paused update scheduler")
            except Exception as e:
                self._logger.error(f"Error pausing update scheduler: {e}")
                
    def _store_pause_state(self) -> None:
        """Store current game state for resumption after pause."""
        # Create a snapshot of important game state if needed
        # This could include player position, active entities, etc.
        try:
            if hasattr(self, "pause_state"):
                # Prepare a state dictionary to hold pause information
                self.pause_state = {
                    "timestamp": time.time(),
                    "game_state": self._current_state
                }
                
                # Store player state if available
                if hasattr(self, "player"):
                    self.pause_state["player_position"] = getattr(self.player, "position", None)
                    self.pause_state["player_direction"] = getattr(self.player, "direction", None)
                    
                # Store any other state information that might be needed for proper resumption
                if hasattr(self, "active_entities"):
                    # Just store entity IDs or references, not deep copying the entities
                    self.pause_state["active_entity_ids"] = [
                        getattr(entity, "id", i) for i, entity in enumerate(self.active_entities)
                    ]
                    
            self._logger.debug("Stored game state for pause")
        except Exception as e:
            self._logger.error(f"Error storing pause state: {e}")
            
    def _show_pause_menu(self) -> None:
        """Display the pause menu overlay.
        
        This state entry handler displays the pause menu while keeping the game
        state preserved in the background. It handles the animation and
        activation of the pause menu UI elements.
        """
        try:
            self._logger.info("Showing pause menu")
            
            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return
                
            # Display and configure the pause menu
            self._setup_pause_menu()
            
            # Apply appropriate animations and transitions
            self._apply_pause_menu_effects()
            
            self._logger.info("Pause menu displayed successfully")
            
        except Exception as e:
            self._logger.error(f"Error showing pause menu: {e}")
            
    def _setup_pause_menu(self) -> None:
        """Set up and display the pause menu interface."""
        if "pause_menu" not in self.ui_elements:
            self._logger.warning("Pause menu UI element not found")
            return

        pause_menu = self.ui_elements["pause_menu"]

        # Activate the pause menu
        if hasattr(pause_menu, "active"):
            pause_menu.active = True
            self._logger.debug("Activated pause menu")

        # Make the pause menu visible
        if hasattr(pause_menu, "visible"):
            pause_menu.visible = True
            self._logger.debug("Made pause menu visible")

        # If the menu has a reset method, call it to ensure proper state
        if hasattr(pause_menu, "reset") and callable(pause_menu.reset):
            try:
                pause_menu.reset()
                self._logger.debug("Reset pause menu state")
            except Exception as e:
                self._logger.error(f"Error resetting pause menu: {e}")

        # Update pause menu content if method available
        if hasattr(pause_menu, "update") and callable(pause_menu.update):
            try:
                pause_info = self.pause_state if hasattr(self, "pause_state") else None
                pause_menu.update(pause_info)
                self._logger.debug("Updated pause menu content")
            except Exception as e:
                self._logger.error(f"Error updating pause menu content: {e}")

        # Track active UI element
        if hasattr(self, "active_element"):
            self.active_element = "pause_menu"
            self._logger.debug("Set pause menu as active element")
            
    def _apply_pause_menu_effects(self) -> None:
        """Apply visual effects and animations to the pause menu."""
        if "pause_menu" not in self.ui_elements:
            return
            
        pause_menu = self.ui_elements["pause_menu"]
        
        # If the menu supports animations, activate them
        if hasattr(pause_menu, "animation") and isinstance(pause_menu.animation, dict):
            if "active" in pause_menu.animation:
                pause_menu.animation["active"] = True
                self._logger.debug("Activated pause menu animation")
                
            # Set animation properties if available
            if "start_time" in pause_menu.animation:
                pause_menu.animation["start_time"] = time.time()
                
            if "duration" in pause_menu.animation:
                # Standard duration for pause menu animation
                pause_menu.animation["duration"] = 0.3  # 300ms animation
                
        # Add overlay effect to dim the background game elements
        if hasattr(self, "overlay") and hasattr(self.overlay, "activate"):
            try:
                # Typically we want a semi-transparent dark overlay for pause menus
                self.overlay.activate(color=(0, 0, 0), opacity=0.7)
                self._logger.debug("Activated background overlay for pause menu")
            except Exception as e:
                self._logger.error(f"Error activating overlay: {e}")
                
    def _show_shop(self) -> None:
        """Display the shop interface.
        
        This state entry handler shows the shop interface, allowing players
        to browse and purchase items, upgrades, or services.
        """
        try:
            self._logger.info("Showing shop interface")
            
            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return
                
            # Display and configure the shop interface
            self._setup_shop_interface()
            
            # Apply appropriate shop animations and transitions
            self._apply_shop_effects()
            
            # Update shop inventory with current available items
            self._update_shop_inventory()
            
            self._logger.info("Shop interface displayed successfully")
            
        except Exception as e:
            self._logger.error(f"Error showing shop interface: {e}")
            
    def _setup_shop_interface(self) -> None:
        """Configure and display the shop interface components."""
        if "shop_interface" not in self.ui_elements:
            self._logger.warning("Shop interface UI element not found")
            return
            
        shop = self.ui_elements["shop_interface"]
        
        # Activate the shop interface
        if hasattr(shop, "active"):
            shop.active = True
            self._logger.debug("Activated shop interface")
            
        # Make the shop interface visible
        if hasattr(shop, "visible"):
            shop.visible = True
            self._logger.debug("Made shop interface visible")
            
        # Hide gameplay elements while in shop
        self._hide_gameplay_for_shop()
            
        # Reset shop state if method available
        if hasattr(shop, "reset") and callable(shop.reset):
            try:
                shop.reset()
                self._logger.debug("Reset shop interface state")
            except Exception as e:
                self._logger.error(f"Error resetting shop interface: {e}")
                
        # Track active UI element
        if hasattr(self, "active_element"):
            self.active_element = "shop_interface"
            self._logger.debug("Set shop as active element")
            
    def _hide_gameplay_for_shop(self) -> None:
        """Hide gameplay elements while the shop is displayed."""
        # Elements to hide during shop interface
        gameplay_elements = ["action_panel", "minimap"]
        
        for element_name in gameplay_elements:
            if element_name not in self.ui_elements:
                continue
                
            element = self.ui_elements[element_name]
            if hasattr(element, "visible"):
                element.visible = False
                self._logger.debug(f"Hid {element_name} for shop display")
                
    def _apply_shop_effects(self) -> None:
        """Apply visual effects and animations to the shop interface."""
        if "shop_interface" not in self.ui_elements:
            return
            
        shop = self.ui_elements["shop_interface"]
        
        # If the shop supports animations, activate them
        if hasattr(shop, "animation") and isinstance(shop.animation, dict):
            if "active" in shop.animation:
                shop.animation["active"] = True
                self._logger.debug("Activated shop animation")
                
            # Set animation properties if available
            if "start_time" in shop.animation:
                shop.animation["start_time"] = time.time()
                
            if "duration" in shop.animation:
                # Standard duration for shop animation
                shop.animation["duration"] = 0.5  # 500ms animation
                
        # Add shop-specific visual effects if any
        if hasattr(shop, "apply_effects") and callable(shop.apply_effects):
            try:
                shop.apply_effects()
                self._logger.debug("Applied shop visual effects")
            except Exception as e:
                self._logger.error(f"Error applying shop effects: {e}")
                
    def _update_shop_inventory(self) -> None:
        """Update the shop inventory with current available items."""
        # Check if shop interface exists
        if "shop_interface" not in self.ui_elements:
            return
            
        shop = self.ui_elements["shop_interface"]
        
        # If shop has an update_inventory method, call it
        if hasattr(shop, "update_inventory") and callable(shop.update_inventory):
            try:
                # Get available items to display in the shop
                items = self._get_available_shop_items()
                
                # Update shop with items
                shop.update_inventory(items)
                self._logger.debug("Updated shop inventory")
            except Exception as e:
                self._logger.error(f"Error updating shop inventory: {e}")
                
    def _get_available_shop_items(self) -> List[Dict[str, Any]]:
        """Get the list of items available for purchase in the shop.
        
        Returns:
            List[Dict[str, Any]]: List of item dictionaries with properties
        """
        try:
            # Check if shop manager exists
            if hasattr(self, "shop_manager") and hasattr(self.shop_manager, "get_available_items"):
                return self.shop_manager.get_available_items()
                
            # Check if there's a shop data source in game config
            if "shop_items" in GAME_CONFIG:
                return GAME_CONFIG["shop_items"]
                
            # If no dynamic source exists, return empty list
            self._logger.warning("No shop item source found, displaying empty shop")
            return []
            
        except Exception as e:
            self._logger.error(f"Error retrieving shop items: {e}")
            return []
            
    def _show_map(self) -> None:
        """Display the game map/minimap.
        
        This state entry handler shows the map interface, allowing players
        to view the game world, their current position, and points of interest.
        """
        try:
            self._logger.info("Showing game map interface")
            
            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return
                
            # Check if field data is available
            if not self._check_field_data_available():
                self._logger.error("Field data not available for map display")
                return
                
            # Setup and display the map interface
            self._setup_map_interface()
            
            # Update map with current game state
            self._update_map_data()
            
            # Apply map-specific visual effects if any
            self._apply_map_effects()
            
            self._logger.info("Map interface displayed successfully")
            
        except Exception as e:
            self._logger.error(f"Error showing map interface: {e}")
            
    def _check_field_data_available(self) -> bool:
        """Check if field data required for map display is available.
        
        Returns:
            bool: True if field data is available, False otherwise
        """
        # Check if field exists
        if not hasattr(self, "field"):
            self._logger.warning("Field instance not found for map display")
            return False
            
        # Check if player has valid position data for map centering
        if not hasattr(self, "player") or not hasattr(self.player, "position"):
            self._logger.warning("Player position data not available for map display")
            return False
            
        return True
            
    def _setup_map_interface(self) -> None:
        """Set up and display the map interface components."""
        # Check for map interface component
        if "map_interface" not in self.ui_elements:
            self._logger.warning("Map interface UI element not found")
            return
            
        map_ui = self.ui_elements["map_interface"]
        
        # Activate the map interface
        if hasattr(map_ui, "active"):
            map_ui.active = True
            self._logger.debug("Activated map interface")
            
        # Make the map interface visible
        if hasattr(map_ui, "visible"):
            map_ui.visible = True
            self._logger.debug("Made map interface visible")
            
        # Track active UI element
        if hasattr(self, "active_element"):
            self.active_element = "map_interface"
            self._logger.debug("Set map interface as active element")
            
    def _update_map_data(self) -> None:
        """Update map with current game state and entity positions."""
        if "map_interface" not in self.ui_elements:
            return
            
        map_ui = self.ui_elements["map_interface"]
        
        # If map has update method, call it with current data
        if hasattr(map_ui, "update") and callable(map_ui.update):
            try:
                # Prepare map data
                map_data = self._prepare_map_data()
                
                # Update map with data
                map_ui.update(map_data)
                self._logger.debug("Updated map with current data")
            except Exception as e:
                self._logger.error(f"Error updating map data: {e}")
                
    def _prepare_map_data(self) -> Dict[str, Any]:
        """Prepare map data from current game state.
        
        Returns:
            Dict[str, Any]: Map data including field, entities, and POIs
        """
        map_data = {}
        
        # Add field data if available
        if hasattr(self, "field") and hasattr(self.field, "get_map_data"):
            try:
                map_data["field"] = self.field.get_map_data()
            except Exception as e:
                self._logger.error(f"Error getting field map data: {e}")
                
        # Add player position if available
        if hasattr(self, "player") and hasattr(self.player, "position"):
            map_data["player_position"] = self.player.position
            
        # Add entity positions if available
        if hasattr(self, "entities"):
            map_data["entities"] = [
                {
                    "position": entity.position,
                    "icon": entity.map_icon,
                    "entity_id": getattr(entity, "id", None)
                }
                for entity in self.entities
                if hasattr(entity, "position") and hasattr(entity, "map_icon")
            ]
                    
        # Add points of interest if available
        if hasattr(self, "points_of_interest"):
            map_data["poi"] = self.points_of_interest
            
        return map_data
                
    def _apply_map_effects(self) -> None:
        """Apply visual effects and animations to the map interface."""
        if "map_interface" not in self.ui_elements:
            return
            
        map_ui = self.ui_elements["map_interface"]
        
        # If the map supports animations, activate them
        if hasattr(map_ui, "animation") and isinstance(map_ui.animation, dict):
            if "active" in map_ui.animation:
                map_ui.animation["active"] = True
                self._logger.debug("Activated map animation")
                
            # Set animation properties if available
            if "start_time" in map_ui.animation:
                map_ui.animation["start_time"] = time.time()
                
            if "duration" in map_ui.animation:
                # Standard duration for map animation
                map_ui.animation["duration"] = 0.3  # 300ms animation
                
        # Apply specific map effects if available
        if hasattr(map_ui, "apply_effects") and callable(map_ui.apply_effects):
            try:
                map_ui.apply_effects("open")  # Pass the effect trigger/type
                self._logger.debug("Applied map open effects")
            except Exception as e:
                self._logger.error(f"Error applying map effects: {e}")
                
    def _show_game_over(self) -> None:
        """Display the game over screen.
        
        This state entry handler shows the game over screen, displaying the
        player's final stats, score, and options to restart or quit.
        """
        try:
            self._logger.info("Showing game over screen")
            
            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return
                
            # Pause all game systems before showing game over
            self._pause_game_systems_for_game_over()
                
            # Setup and display the game over interface
            self._setup_game_over_interface()
            
            # Update game over screen with final stats
            self._update_game_over_stats()
            
            # Apply game over animations and effects
            self._apply_game_over_effects()
            
            self._logger.info("Game over screen displayed successfully")
            
        except Exception as e:
            self._logger.error(f"Error showing game over screen: {e}")
            
    def _pause_game_systems_for_game_over(self) -> None:
        """Pause all active game systems when showing game over screen."""
        try:
            # Pause all modules that support pausing
            modules_to_pause = ["field", "combat_system", "quest_manager", "timer"]
            
            for module_name in modules_to_pause:
                if not hasattr(self, module_name):
                    continue
                    
                module = getattr(self, module_name)
                if hasattr(module, "pause") and callable(module.pause):
                    try:
                        module.pause()
                        self._logger.debug(f"Paused {module_name} for game over")
                    except Exception as e:
                        self._logger.error(f"Error pausing {module_name}: {e}")
                        
            self._logger.debug("Paused game systems for game over")
            
        except Exception as e:
            self._logger.error(f"Error pausing systems for game over: {e}")
            
    def _setup_game_over_interface(self) -> None:
        """Set up and display the game over interface."""
        # Check for game over interface component
        if "game_over_screen" not in self.ui_elements:
            self._logger.warning("Game over screen UI element not found")
            return
            
        game_over = self.ui_elements["game_over_screen"]
        
        # Deactivate other UI elements that shouldn't be visible
        self._hide_ui_elements_for_game_over()
        
        # Activate the game over screen
        if hasattr(game_over, "active"):
            game_over.active = True
            self._logger.debug("Activated game over screen")
            
        # Make the game over screen visible
        if hasattr(game_over, "visible"):
            game_over.visible = True
            self._logger.debug("Made game over screen visible")
            
        # Track active UI element
        if hasattr(self, "active_element"):
            self.active_element = "game_over_screen"
            self._logger.debug("Set game over screen as active element")
            
    def _hide_ui_elements_for_game_over(self) -> None:
        """Hide gameplay UI elements when showing game over screen."""
        # Elements to hide during game over screen
        elements_to_hide = [
            "action_panel", "minimap", "inventory_panel", 
            "status_bar", "game_hud", "shop_interface", "map_interface"
        ]
        
        for element_name in elements_to_hide:
            if element_name not in self.ui_elements:
                continue
                
            element = self.ui_elements[element_name]
            if hasattr(element, "visible"):
                element.visible = False
                self._logger.debug(f"Hid {element_name} for game over screen")
                
    def _update_game_over_stats(self) -> None:
        """Update game over screen with player's final stats and score."""
        # Check for game over screen
        if "game_over_screen" not in self.ui_elements:
            return
            
        game_over = self.ui_elements["game_over_screen"]
        
        # If game over screen has update_stats method, call it with player stats
        if hasattr(game_over, "update_stats") and callable(game_over.update_stats):
            try:
                # Get final game stats
                final_stats = self._get_final_game_stats()
                
                # Update game over screen with stats
                game_over.update_stats(final_stats)
                self._logger.debug("Updated game over screen with final stats")
            except Exception as e:
                self._logger.error(f"Error updating game over stats: {e}")
                
    def _get_final_game_stats(self) -> Dict[str, Any]:
        """Get the player's final stats for the game over screen.
        
        Returns:
            Dict[str, Any]: Dictionary containing final game statistics
        """
        stats = {}
        
        # Add player stats if available
        if hasattr(self, "player"):
            player = self.player
            
            # Get basic player attributes
            for attr in ["health", "max_health", "energy", "max_energy", "level"]:
                if hasattr(player, attr):
                    stats[attr] = getattr(player, attr)
                    
            # Get inventory summary if available
            if hasattr(player, "inventory") and hasattr(player.inventory, "get_summary"):
                try:
                    stats["inventory"] = player.inventory.get_summary()
                except Exception as e:
                    self._logger.error(f"Error getting inventory summary: {e}")
                    
        # Add game stats
        if hasattr(self, "game_stats"):
            stats["game_stats"] = self.game_stats
            
        # Add score if available
        if hasattr(self, "score"):
            stats["score"] = self.score
            
        # Add game duration if available
        if hasattr(self, "start_time"):
            stats["duration"] = time.time() - self.start_time
            
        return stats
                
    def _apply_game_over_effects(self) -> None:
        """Apply visual effects and animations to the game over screen."""
        if "game_over_screen" not in self.ui_elements:
            return
            
        game_over = self.ui_elements["game_over_screen"]
        
        # If the game over screen supports animations, activate them
        if hasattr(game_over, "animation") and isinstance(game_over.animation, dict):
            if "active" in game_over.animation:
                game_over.animation["active"] = True
                self._logger.debug("Activated game over animation")
                
            # Set animation properties if available
            if "start_time" in game_over.animation:
                game_over.animation["start_time"] = time.time()
                
            if "duration" in game_over.animation:
                # Longer duration for game over animation
                game_over.animation["duration"] = 1.5  # 1.5s animation
                
        # Apply specific game over effects if available
        if hasattr(game_over, "apply_effects") and callable(game_over.apply_effects):
            try:
                game_over.apply_effects("fade_in")  # Pass the effect trigger/type
                self._logger.debug("Applied game over screen effects")
            except Exception as e:
                self._logger.error(f"Error applying game over effects: {e}")
                
    # ======================================================================
    # State Exit Handlers
    # ======================================================================
                
    def _exit_hide_menu(self) -> None:
        """Clean up menu resources when exiting the menu state.
        
        This state exit handler ensures proper cleanup of menu resources,
        canceling animations, unloading unnecessary assets, and setting UI
        elements to appropriate states for the next state.
        """
        try:
            self._logger.info("Cleaning up menu resources")
            
            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return
                
            # Clean up menu animations and effects
            self._cancel_menu_animations()
            
            # Unload menu-specific assets if needed
            self._unload_menu_assets()
            
            # Clear any menu-related state
            self._clear_menu_state()
            
            self._logger.info("Menu resources cleaned up successfully")
            
        except Exception as e:
            self._logger.error(f"Error cleaning up menu resources: {e}")
            
    def _cancel_menu_animations(self) -> None:
        """Cancel any ongoing menu animations."""
        menu_elements = ["main_menu", "options_menu", "menu_background"]
        
        for element_name in menu_elements:
            if element_name not in self.ui_elements:
                continue
                
            element = self.ui_elements[element_name]
            
            # If element has animation, disable it
            if hasattr(element, "animation") and isinstance(element.animation, dict) and "active" in element.animation:
                element.animation["active"] = False
                self._logger.debug(f"Cancelled animation for {element_name}")
                    
            # If element has a cancel_animation method, call it
            if hasattr(element, "cancel_animation") and callable(element.cancel_animation):
                try:
                    element.cancel_animation()
                    self._logger.debug(f"Called cancel_animation for {element_name}")
                except Exception as e:
                    self._logger.error(f"Error cancelling animation for {element_name}: {e}")
                    
    def _unload_menu_assets(self) -> None:
        """Unload menu-specific assets to free memory."""
        # Check if resource manager exists and supports unloading
        if not self._has_resource_manager():
            return
            
        try:
            # Get list of menu resources to unload
            menu_resources = ["menu_background", "menu_music", "menu_effects"]
            
            # Unload each resource
            for resource in menu_resources:
                self._unload_single_resource(resource)
                
            self._logger.debug("Finished unloading menu assets")
            
        except Exception as e:
            self._logger.error(f"Error unloading menu assets: {e}")
            
    def _has_resource_manager(self) -> bool:
        """Check if resource manager exists and supports unloading."""
        return (hasattr(self, "resource_manager") and 
                hasattr(self.resource_manager, "unload"))
                
    def _unload_single_resource(self, resource_name: str) -> None:
        """Unload a single resource if it's loaded.
        
        Args:
            resource_name: Name of the resource to unload
        """
        if not hasattr(self, "resource_manager"):
            return

        if (hasattr(self.resource_manager, "is_loaded")
            and callable(self.resource_manager.is_loaded)):
            # Only unload if the resource is loaded
            if self.resource_manager.is_loaded(resource_name):
                self._perform_resource_unload(resource_name)
        else:
            # Try unloading directly if can't check loaded status
            self._attempt_direct_unload(resource_name)
            
    def _perform_resource_unload(self, resource_name: str) -> None:
        """Perform actual resource unloading and log result.
        
        Args:
            resource_name: Name of the resource to unload
        """
        try:
            self.resource_manager.unload(resource_name)
            self._logger.debug(f"Unloaded menu resource: {resource_name}")
        except Exception as e:
            self._logger.error(f"Failed to unload resource {resource_name}: {e}")
            
    def _attempt_direct_unload(self, resource_name: str) -> None:
        """Attempt to unload a resource without checking if it's loaded.
        
        Args:
            resource_name: Name of the resource to unload
        """
        with contextlib.suppress(Exception):
            self.resource_manager.unload(resource_name)
            self._logger.debug(f"Unloaded menu resource: {resource_name}")
                
    def _clear_menu_state(self) -> None:
        """Clear any menu-related state variables."""
        try:
            # Reset menu selection trackers if they exist
            menu_state_vars = ["menu_selection", "last_menu_state", "menu_history"]
            
            for var_name in menu_state_vars:
                if hasattr(self, var_name):
                    if isinstance(getattr(self, var_name), list):
                        setattr(self, var_name, [])
                    elif isinstance(getattr(self, var_name), dict):
                        setattr(self, var_name, {})
                    else:
                        setattr(self, var_name, None)
                    self._logger.debug(f"Cleared menu state variable: {var_name}")
                    
            # Clear active element if it's a menu
            if hasattr(self, "active_element") and self.active_element in ["main_menu", "options_menu"]:
                self.active_element = None
                self._logger.debug("Cleared active element")
                
            self._logger.debug("Finished clearing menu state")
            
        except Exception as e:
            self._logger.error(f"Error clearing menu state: {e}")
            
    def _exit_save_game_state(self) -> None:
        """Save current game progress when exiting a state.
        
        This state exit handler ensures that the current game state is saved,
        including player data, game world state, and other progress metrics.
        """
        try:
            self._logger.info("Saving current game state")

            # Create save data object with all relevant game state
            save_data = self._prepare_save_data()

            # Ensure save directory exists
            self._ensure_save_directory()

            if save_success := self._write_save_data(save_data):
                self._logger.info("Game state saved successfully")
            else:
                self._logger.warning("Failed to save game state")

        except Exception as e:
            self._logger.error(f"Error saving game state: {e}")
            
    def _prepare_save_data(self) -> Dict[str, Any]:
        """Collect all relevant game state data for saving.
        
        Returns:
            Dict[str, Any]: Dictionary containing game state data
        """
        save_data = {
            "version": GAME_CONFIG.get("version", "1.0.0"),
            "timestamp": time.time(),
            "state": self._state
        }

        # Add player data if available
        if hasattr(self, "player"):
            if player_data := self._collect_player_data():
                save_data["player"] = player_data

        # Add field/map data if available
        if hasattr(self, "field"):
            if field_data := self._collect_field_data():
                save_data["field"] = field_data

        # Add game progress metrics
        if hasattr(self, "game_stats"):
            save_data["stats"] = self.game_stats

        # Add game time tracking
        if hasattr(self, "start_time"):
            save_data["play_time"] = time.time() - self.start_time

        # Add entity data if available
        if hasattr(self, "entities"):
            if entities_data := self._collect_entities_data():
                save_data["entities"] = entities_data

        return save_data
        
    def _collect_player_data(self) -> Dict[str, Any]:
        """Collect player data for saving.
        
        Returns:
            Dict[str, Any]: Dictionary containing player state data
        """
        player_data = {}

        try:
            self._extracted_from__collect_player_data_10(player_data)
        except Exception as e:
            self._logger.error(f"Error collecting player data: {e}")

        return player_data

    # TODO Rename this here and in `_collect_player_data`
    def _extracted_from__collect_player_data_10(self, player_data):
        player = self.player

        # Collect basic player attributes
        basic_attrs = ["health", "max_health", "energy", "max_energy",
                      "level", "experience", "position", "direction"]

        for attr in basic_attrs:
            if hasattr(player, attr):
                player_data[attr] = getattr(player, attr)

        # Collect inventory if available
        if hasattr(player, "inventory"):
            if hasattr(player.inventory, "get_save_data") and callable(player.inventory.get_save_data):
                player_data["inventory"] = player.inventory.get_save_data()
            elif hasattr(player.inventory, "items"):
                player_data["inventory"] = {"items": player.inventory.items}

        # Collect equipment if available
        if hasattr(player, "equipment") and hasattr(player.equipment, "get_save_data"):
            player_data["equipment"] = player.equipment.get_save_data()

        # Collect skills/abilities if available
        if hasattr(player, "skills") and hasattr(player.skills, "get_save_data"):
            player_data["skills"] = player.skills.get_save_data()
        
    def _collect_field_data(self) -> Dict[str, Any]:
        """Collect field/map data for saving.
        
        Returns:
            Dict[str, Any]: Dictionary containing field state data
        """
        field_data = {}
        
        try:
            field = self.field
            
            # Check if field has save data method
            if hasattr(field, "get_save_data") and callable(field.get_save_data):
                return field.get_save_data()
                
            # Collect basic field properties
            field_data = self._collect_basic_field_properties(field)
            
            # Add objects data if available
            self._add_field_objects_data(field, field_data)
                        
        except Exception as e:
            self._logger.error(f"Error collecting field data: {e}")
            
        return field_data
        
    def _collect_basic_field_properties(self, field) -> Dict[str, Any]:
        """Collect basic field properties like dimensions and tiles.
        
        Args:
            field: The field object
            
        Returns:
            Dict[str, Any]: Basic field properties
        """
        field_data = {}
        
        # Get width and height
        if hasattr(field, "width"):
            field_data["width"] = field.width
        if hasattr(field, "height"):
            field_data["height"] = field.height
            
        # Get tiles data
        if hasattr(field, "tiles"):
            field_data["tiles"] = field.tiles
            
        return field_data
        
    def _add_field_objects_data(self, field, field_data: Dict[str, Any]) -> None:
        """Add field objects data to the field data dictionary.
        
        Args:
            field: The field object
            field_data: Dictionary to add objects data to
        """
        if not hasattr(field, "objects"):
            return
            
        # Get serializable objects if method available
        if hasattr(field, "get_serializable_objects") and callable(field.get_serializable_objects):
            field_data["objects"] = field.get_serializable_objects()
        else:
            field_data["objects"] = []
        
    def _collect_entities_data(self) -> List[Dict[str, Any]]:
        """Collect data for non-player entities.
        
        Returns:
            List[Dict[str, Any]]: List of entity data dictionaries
        """
        entities_data = []

        try:
            for entity in self.entities:
                if hasattr(entity, "get_save_data") and callable(entity.get_save_data):
                    entities_data.append(entity.get_save_data())
                else:
                    # Get essential attributes
                    basic_attrs = ["id", "type", "position", "health", "state"]
                    if entity_data := {
                        attr: getattr(entity, attr)
                        for attr in basic_attrs
                        if hasattr(entity, attr)
                    }:
                        entities_data.append(entity_data)

        except Exception as e:
            self._logger.error(f"Error collecting entities data: {e}")

        return entities_data
        
    def _ensure_save_directory(self) -> None:
        """Ensure the save directory exists, creating it if necessary."""
        save_dir = self._get_save_directory()
        
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                self._logger.debug(f"Created save directory: {save_dir}")
                
        except Exception as e:
            self._logger.error(f"Error creating save directory: {e}")
            
    def _get_save_directory(self) -> str:
        """Get the directory path for saving game data.
        
        Returns:
            str: Path to save directory
        """
        # Default save directory
        default_save_dir = os.path.join(os.path.expanduser("~"), ".space_muck", "saves")
        
        # Check if custom save directory is defined in config
        if "save_directory" in GAME_CONFIG:
            custom_save_dir = GAME_CONFIG["save_directory"]
            
            # If relative path, make it relative to user home
            if not os.path.isabs(custom_save_dir):
                custom_save_dir = os.path.join(os.path.expanduser("~"), custom_save_dir)
                
            return custom_save_dir
            
        return default_save_dir
        
    def _write_save_data(self, save_data: Dict[str, Any]) -> bool:
        """Write save data to file.
        
        Args:
            save_data: Dictionary containing game state data
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Get save directory and generate file path
            save_path = self._generate_save_path(save_data)
            
            # Write data to file
            self._write_data_to_file(save_path, save_data)
            
            # Create quick save reference
            self._create_quicksave_reference(save_path)
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error writing save data: {e}")
            return False
            
    def _generate_save_path(self, save_data: Dict[str, Any]) -> str:
        """Generate a path for saving the game data.
        
        Args:
            save_data: Dictionary containing game state data
            
        Returns:
            str: Complete path for the save file
        """
        save_dir = self._get_save_directory()

        # Generate timestamp
        save_timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Get player name if available
        player_name = "player"  # Default name
        if "player" in save_data and "name" in save_data["player"]:
            player_name = save_data["player"]["name"]

        # Create file name and full path
        save_file_name = f"{player_name}_{save_timestamp}.json"
        return os.path.join(save_dir, save_file_name)
        
    def _write_data_to_file(self, file_path: str, data: Dict[str, Any]) -> None:
        """Write data to a file in JSON format.
        
        Args:
            file_path: Path to save the file
            data: Data to save
        """
        with open(file_path, "w") as save_file:
            json.dump(data, save_file, indent=2)
            
        self._logger.debug(f"Saved game to: {file_path}")
            
    def _create_quicksave_reference(self, save_path: str) -> None:
        """Create a reference to the most recent save for quick loading.
        
        Args:
            save_path: Path to the most recent save file
        """
        try:
            save_dir = self._get_save_directory()
            quicksave_path = os.path.join(save_dir, "quicksave.json")
            
            # Create a quicksave reference file pointing to the latest save
            with open(quicksave_path, "w") as quicksave_file:
                quicksave_data = {
                    "latest_save": save_path,
                    "timestamp": time.time()
                }
                json.dump(quicksave_data, quicksave_file)
                
            self._logger.debug(f"Updated quicksave reference to: {save_path}")
            
        except Exception as e:
            self._logger.error(f"Error creating quicksave reference: {e}")
            
    def _exit_hide_pause_menu(self) -> None:
        """Clean up pause menu resources when exiting the pause state.
        
        This state exit handler ensures proper cleanup of pause menu resources,
        restoring game state and UI elements to continue gameplay seamlessly.
        """
        try:
            self._logger.info("Cleaning up pause menu resources")
            
            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return
                
            # Hide pause menu UI elements
            self._hide_pause_ui_elements()
            
            # Restore game state (unpause)
            self._resume_game_state()
            
            self._logger.info("Pause menu resources cleaned up successfully")
            
        except Exception as e:
            self._logger.error(f"Error cleaning up pause menu resources: {e}")
            
    def _hide_pause_ui_elements(self) -> None:
        """Hide all pause menu UI elements."""
        try:
            # List of pause menu related UI elements
            pause_elements = ["pause_menu", "pause_background", "pause_options"]
            
            for element_name in pause_elements:
                if element_name not in self.ui_elements:
                    continue
                    
                element = self.ui_elements[element_name]
                
                # Hide the element if it has visible property
                if hasattr(element, "visible"):
                    element.visible = False
                    self._logger.debug(f"Set {element_name} visibility to False")
                    
                # If element has specific hide method, call it
                if hasattr(element, "hide") and callable(element.hide):
                    try:
                        element.hide()
                        self._logger.debug(f"Called hide() for {element_name}")
                    except Exception as e:
                        self._logger.error(f"Error hiding {element_name}: {e}")
                        
            self._logger.debug("Finished hiding pause menu UI elements")
            
        except Exception as e:
            self._logger.error(f"Error hiding pause UI elements: {e}")
            
    def _resume_game_state(self) -> None:
        """Restore game state after exiting pause menu."""
        try:
            # Unpause any paused systems
            if hasattr(self, "_paused"):
                self._paused = False
                self._logger.debug("Unpaused game state")
                
            # Resume any paused animations or timers
            self._resume_animations_and_timers()
            
            # Restore input handling if it was disabled
            if hasattr(self, "_input_enabled"):
                self._input_enabled = True
                self._logger.debug("Restored input handling")
                
            # Resume sound if it was paused
            self._resume_audio()
            
            self._logger.debug("Finished resuming game state")
            
        except Exception as e:
            self._logger.error(f"Error resuming game state: {e}")
            
    def _resume_animations_and_timers(self) -> None:
        """Resume any animations or timers that were paused."""
        try:
            # Resume game clock
            self._resume_game_clock()
            
            # Resume entity animations
            self._resume_entity_animations()
            
            # Resume UI animations
            self._resume_ui_animations()
                        
        except Exception as e:
            self._logger.error(f"Error resuming animations and timers: {e}")
            
    def _resume_game_clock(self) -> None:
        """Resume the game clock if it exists."""
        if hasattr(self, "game_clock") and hasattr(self.game_clock, "resume"):
            self.game_clock.resume()
            self._logger.debug("Resumed game clock")
            
    def _resume_entity_animations(self) -> None:
        """Resume animations for all entities."""
        if not hasattr(self, "entities"):
            return
            
        for entity in self.entities:
            if hasattr(entity, "resume_animation") and callable(entity.resume_animation):
                entity.resume_animation()
                
    def _resume_ui_animations(self) -> None:
        """Resume animations for all UI elements."""
        if not hasattr(self, "ui_elements"):
            return
            
        for element_name, element in self.ui_elements.items():
            self._resume_element_animation(element_name, element)
                
    def _resume_element_animation(self, element_name: str, element) -> None:
        """Resume animation for a specific UI element.
        
        Args:
            element_name: Name of the UI element
            element: The UI element object
        """
        # Resume animation dictionary if it exists
        if hasattr(element, "animation") and isinstance(element.animation, dict) and "paused" in element.animation:
            element.animation["paused"] = False
            self._logger.debug(f"Resumed animation for {element_name}")
            
        # Call resume_animation method if it exists
        if hasattr(element, "resume_animation") and callable(element.resume_animation):
            try:
                element.resume_animation()
                self._logger.debug(f"Called resume_animation for {element_name}")
            except Exception as e:
                self._logger.error(f"Error resuming animation for {element_name}: {e}")
            
    def _resume_audio(self) -> None:
        """Resume any audio that was paused."""
        try:
            # Check if sound manager exists
            if not hasattr(self, "sound_manager"):
                return
                
            # Resume background music
            if hasattr(self.sound_manager, "resume_music") and callable(self.sound_manager.resume_music):
                self.sound_manager.resume_music()
                self._logger.debug("Resumed background music")
                
            # Resume sound effects if they were paused
            if hasattr(self.sound_manager, "resume_effects") and callable(self.sound_manager.resume_effects):
                self.sound_manager.resume_effects()
                self._logger.debug("Resumed sound effects")
                
            # If no specific methods, try a generic resume method
            elif hasattr(self.sound_manager, "resume") and callable(self.sound_manager.resume):
                self.sound_manager.resume()
                self._logger.debug("Resumed all audio")
                
        except Exception as e:
            self._logger.error(f"Error resuming audio: {e}")
            
    def _exit_hide_shop(self) -> None:
        """Clean up shop interface when exiting the shop state.
        
        This state exit handler ensures proper cleanup of shop UI elements,
        hiding shop panels, and restoring the game UI when exiting the shop.
        """
        try:
            self._logger.info("Cleaning up shop interface")
            
            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return
                
            # Hide shop UI elements
            self._hide_shop_ui_elements()
            
            # Reset shop state variables
            self._reset_shop_state()
            
            # Restore game UI visibility
            self._restore_game_ui_after_shop()
            
            self._logger.info("Shop interface cleaned up successfully")
            
        except Exception as e:
            self._logger.error(f"Error cleaning up shop interface: {e}")
            
    def _hide_shop_ui_elements(self) -> None:
        """Hide all shop-related UI elements."""
        try:
            # List of shop-related UI elements
            shop_elements = ["shop_panel", "shop_inventory", "shop_items", "shop_buttons"]
            
            for element_name in shop_elements:
                if element_name not in self.ui_elements:
                    continue
                    
                element = self.ui_elements[element_name]
                
                # Hide the element if it has visible property
                if hasattr(element, "visible"):
                    element.visible = False
                    self._logger.debug(f"Set {element_name} visibility to False")
                    
                # If element has hide method, call it
                if hasattr(element, "hide") and callable(element.hide):
                    try:
                        element.hide()
                        self._logger.debug(f"Called hide() for {element_name}")
                    except Exception as e:
                        self._logger.error(f"Error hiding {element_name}: {e}")
            
            self._logger.debug("Finished hiding shop UI elements")
            
        except Exception as e:
            self._logger.error(f"Error hiding shop UI elements: {e}")
            
    def _reset_shop_state(self) -> None:
        """Reset shop-related state variables."""
        try:
            # Reset shop selection and interaction state
            shop_state_vars = ["shop_selection", "shop_page", "shop_filter", "shop_sort_order"]
            
            for var_name in shop_state_vars:
                if hasattr(self, var_name):
                    if isinstance(getattr(self, var_name), list):
                        setattr(self, var_name, [])
                    elif isinstance(getattr(self, var_name), dict):
                        setattr(self, var_name, {})
                    elif isinstance(getattr(self, var_name), int):
                        setattr(self, var_name, 0)
                    else:
                        setattr(self, var_name, None)
                    self._logger.debug(f"Reset shop state variable: {var_name}")
                    
            # Clear active element if it's shop-related
            if hasattr(self, "active_element") and "shop" in str(self.active_element):
                self.active_element = None
                self._logger.debug("Cleared active element")
                
            self._logger.debug("Finished resetting shop state")
            
        except Exception as e:
            self._logger.error(f"Error resetting shop state: {e}")
            
    def _restore_game_ui_after_shop(self) -> None:
        """Restore game UI elements after exiting shop."""
        try:
            # List of game UI elements to restore
            game_ui_elements = ["player_hud", "inventory_button", "map_button"]
            
            for element_name in game_ui_elements:
                if element_name not in self.ui_elements:
                    continue
                    
                element = self.ui_elements[element_name]
                
                # Show the element if it has visible property
                if hasattr(element, "visible"):
                    element.visible = True
                    self._logger.debug(f"Set {element_name} visibility to True")
                    
                # If element has show method, call it
                if hasattr(element, "show") and callable(element.show):
                    try:
                        element.show()
                        self._logger.debug(f"Called show() for {element_name}")
                    except Exception as e:
                        self._logger.error(f"Error showing {element_name}: {e}")
                        
            self._logger.debug("Finished restoring game UI elements")
            
        except Exception as e:
            self._logger.error(f"Error restoring game UI: {e}")
            
    def _exit_save_purchases(self) -> None:
        """Save player's purchase data when exiting the shop state.
        
        This state exit handler processes pending purchases, updates player inventory,
        deducts currency or resources, and records purchase history.
        """
        try:
            self._logger.info("Processing and saving purchase data")
            
            # Validate that we have necessary data to process purchases
            if not self._validate_purchase_data():
                return
                
            # Process pending purchases
            processed_purchases = self._process_pending_purchases()
            
            # Update transaction history
            if processed_purchases:
                self._update_transaction_history(processed_purchases)
                
            # Update shop inventory if needed
            self._update_shop_inventory()
            
            self._logger.info("Purchase data processed and saved successfully")
            
        except Exception as e:
            self._logger.error(f"Error saving purchase data: {e}")
            
    def _validate_purchase_data(self) -> bool:
        """Validate that we have necessary data to process purchases.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        # Check if player exists
        if not hasattr(self, "player"):
            self._logger.warning("Cannot process purchases: player object not found")
            return False
            
        # Check if pending purchases exist
        if not hasattr(self, "pending_purchases") or not self.pending_purchases:
            self._logger.debug("No pending purchases to process")
            return False
            
        # Check if player has inventory
        if not hasattr(self.player, "inventory"):
            self._logger.warning("Cannot process purchases: player inventory not found")
            return False
            
        return True
        
    def _process_pending_purchases(self) -> List[Dict[str, Any]]:
        """Process all pending purchases and update player inventory and currency.
        
        Returns:
            List[Dict[str, Any]]: List of successfully processed purchases
        """
        processed_purchases = []
        
        try:
            # Get the list of pending purchases
            pending_purchases = getattr(self, "pending_purchases", [])
            
            for purchase in pending_purchases:
                # Make sure purchase has required data
                if self._is_valid_purchase_data(purchase):
                    # Process individual purchase
                    if self._process_single_purchase(purchase):
                        purchase["timestamp"] = time.time()
                        purchase["status"] = "completed"
                        processed_purchases.append(purchase)
                    else:
                        purchase["status"] = "failed"
                        self._logger.warning(f"Failed to process purchase: {purchase['item_id']}")
                else:
                    self._logger.warning(f"Invalid purchase data: {purchase}")
                
            # Clear pending purchases after processing
            self.pending_purchases = []
            
        except Exception as e:
            self._logger.error(f"Error processing pending purchases: {e}")
            
        return processed_purchases
        
    def _is_valid_purchase_data(self, purchase: Dict[str, Any]) -> bool:
        """Check if purchase data contains required fields.
        
        Args:
            purchase: Purchase data dictionary
            
        Returns:
            bool: True if purchase data is valid, False otherwise
        """
        required_fields = ["item_id", "quantity", "cost"]
        return all(field in purchase for field in required_fields)
        
    def _process_single_purchase(self, purchase: Dict[str, Any]) -> bool:
        """Process a single purchase, update inventory and deduct resources.
        
        Args:
            purchase: Purchase data dictionary
            
        Returns:
            bool: True if purchase was processed successfully, False otherwise
        """
        try:
            # Get purchase details
            item_id = purchase["item_id"]
            quantity = purchase["quantity"]
            cost = purchase["cost"]
            
            # Check if player has enough currency/resources
            if not self._can_afford_purchase(cost):
                self._logger.warning(f"Cannot afford purchase: {item_id}")
                return False
                
            # Deduct cost from player
            if not self._deduct_purchase_cost(cost):
                return False
                
            # Add items to player inventory
            if not self._add_purchased_items_to_inventory(item_id, quantity):
                # If adding to inventory fails, refund the cost
                self._refund_purchase_cost(cost)
                return False
                
            # Update game stats to track purchases
            self._update_purchase_stats(item_id, quantity, cost)
            
            self._logger.debug(f"Processed purchase: {quantity}x {item_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error processing purchase {purchase.get('item_id', 'unknown')}: {e}")
            return False
            
    def _can_afford_purchase(self, cost: Dict[str, int]) -> bool:
        """Check if player has enough resources to cover the purchase cost.
        
        Args:
            cost: Dictionary of resources and amounts required
            
        Returns:
            bool: True if player can afford the purchase, False otherwise
        """
        player = self.player
        # Return true only if player has enough currency AND resources
        return (self._has_enough_currency(player, cost) and 
                self._has_enough_resources(player, cost))
        
    def _has_enough_currency(self, player, cost: Dict[str, int]) -> bool:
        """Check if player has enough currency for the purchase.
        
        Args:
            player: Player object
            cost: Purchase cost dictionary
            
        Returns:
            bool: True if player has enough currency, False otherwise
        """
        # Return False only if currency is specified and player doesn't have enough
        return not ("currency" in cost and 
                  hasattr(player, "currency") and 
                  player.currency < cost["currency"])
        
    def _has_enough_resources(self, player, cost: Dict[str, int]) -> bool:
        """Check if player has enough resources for the purchase.
        
        Args:
            player: Player object
            cost: Purchase cost dictionary
            
        Returns:
            bool: True if player has enough resources, False otherwise
        """
        # No resources check needed if cost doesn't specify resources
        if "resources" not in cost or not isinstance(cost["resources"], dict):
            return True
            
        # Player must have resources attribute
        if not hasattr(player, "resources"):
            return False
            
        # Check if player has all required resources in sufficient amounts
        has_get_method = hasattr(player.resources, "get")
        return all(
            (player.resources.get(resource_id, 0) if has_get_method else 0) >= amount
            for resource_id, amount in cost["resources"].items()
        )
        
    def _deduct_purchase_cost(self, cost: Dict[str, int]) -> bool:
        """Deduct the purchase cost from player's currency/resources.
        
        Args:
            cost: Dictionary of resources and amounts to deduct
            
        Returns:
            bool: True if deduction was successful, False otherwise
        """
        try:
            player = self.player
            
            # Deduct currency if specified
            if "currency" in cost and hasattr(player, "currency"):
                player.currency -= cost["currency"]
                self._logger.debug(f"Deducted {cost['currency']} currency")
                
            # Deduct resources if specified
            if "resources" in cost and isinstance(cost["resources"], dict) and hasattr(player, "resources"):
                for resource_id, amount in cost["resources"].items():
                    if hasattr(player.resources, "get") and resource_id in player.resources:
                        player.resources[resource_id] -= amount
                        self._logger.debug(f"Deducted {amount}x {resource_id}")
                    else:
                        self._logger.warning(f"Resource not found: {resource_id}")
                        return False
                        
            return True
            
        except Exception as e:
            self._logger.error(f"Error deducting purchase cost: {e}")
            return False
            
    def _refund_purchase_cost(self, cost: Dict[str, int]) -> None:
        """Refund the purchase cost to player's currency/resources if purchase fails.
        
        Args:
            cost: Dictionary of resources and amounts to refund
        """
        try:
            player = self.player
            
            # Refund currency if specified
            if "currency" in cost and hasattr(player, "currency"):
                player.currency += cost["currency"]
                self._logger.debug(f"Refunded {cost['currency']} currency")
                
            # Refund resources if specified
            if "resources" in cost and isinstance(cost["resources"], dict) and hasattr(player, "resources"):
                for resource_id, amount in cost["resources"].items():
                    if hasattr(player.resources, "get") and resource_id in player.resources:
                        player.resources[resource_id] += amount
                        self._logger.debug(f"Refunded {amount}x {resource_id}")
                        
        except Exception as e:
            self._logger.error(f"Error refunding purchase cost: {e}")
            
    def _add_purchased_items_to_inventory(self, item_id: str, quantity: int) -> bool:
        """Add purchased items to player inventory.
        
        Args:
            item_id: ID of the purchased item
            quantity: Quantity of items to add
            
        Returns:
            bool: True if items were added successfully, False otherwise
        """
        try:
            player = self.player
            inventory = player.inventory
            
            # Try different inventory adding strategies
            if self._try_add_via_method(inventory, item_id, quantity):
                return True
                
            if self._try_add_via_dict(inventory, item_id, quantity):
                return True
                
            if self._try_add_via_list(inventory, item_id, quantity):
                return True
                
            # If no strategy worked
            self._logger.warning("Unknown inventory format, cannot add items")
            return False
                
        except Exception as e:
            self._logger.error(f"Error adding items to inventory: {e}")
            return False
            
    def _try_add_via_method(self, inventory, item_id: str, quantity: int) -> bool:
        """Try to add items to inventory using its add_item method.
        
        Args:
            inventory: Player inventory object
            item_id: ID of the item to add
            quantity: Quantity to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not hasattr(inventory, "add_item") or not callable(inventory.add_item):
            return False
            
        for _ in range(quantity):
            success = inventory.add_item(item_id)
            if not success:
                self._logger.warning(f"Failed to add item {item_id} to inventory")
                return False
                
        return True
        
    def _try_add_via_dict(self, inventory, item_id: str, quantity: int) -> bool:
        """Try to add items to inventory using its items dictionary.
        
        Args:
            inventory: Player inventory object
            item_id: ID of the item to add
            quantity: Quantity to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not hasattr(inventory, "items") or not isinstance(inventory.items, dict):
            return False
            
        if item_id in inventory.items:
            inventory.items[item_id] += quantity
        else:
            inventory.items[item_id] = quantity
            
        return True
        
    def _try_add_via_list(self, inventory, item_id: str, quantity: int) -> bool:
        """Try to add items to inventory using its append method (list).
        
        Args:
            inventory: Player inventory object
            item_id: ID of the item to add
            quantity: Quantity to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not hasattr(inventory, "append") or not callable(inventory.append):
            return False
            
        for _ in range(quantity):
            inventory.append(item_id)
            
        return True
            
    def _update_purchase_stats(self, item_id: str, quantity: int, cost: Dict[str, int]) -> None:
        """Update game statistics with purchase information.
        
        Args:
            item_id: ID of the purchased item
            quantity: Quantity of items purchased
            cost: Cost of the purchase
        """
        try:
            # Initialize game_stats if it doesn't exist
            if not hasattr(self, "game_stats"):
                self.game_stats = {}
                
            # Initialize purchases stats if they don't exist
            if "purchases" not in self.game_stats:
                self.game_stats["purchases"] = {
                    "total_items": 0,
                    "total_spent": 0,
                    "items": {}
                }
                
            stats = self.game_stats["purchases"]
            
            # Update totals
            stats["total_items"] += quantity
            
            # Calculate total cost in currency
            total_cost = cost.get("currency", 0)
            if "resources" in cost:
                # Convert resources to equivalent currency if resource values exist
                if hasattr(self, "resource_values") and isinstance(self.resource_values, dict):
                    for resource_id, amount in cost["resources"].items():
                        if resource_id in self.resource_values:
                            total_cost += amount * self.resource_values[resource_id]
                            
            stats["total_spent"] += total_cost
            
            # Update item-specific stats
            if item_id not in stats["items"]:
                stats["items"][item_id] = {
                    "quantity": 0,
                    "spent": 0
                }
                
            stats["items"][item_id]["quantity"] += quantity
            stats["items"][item_id]["spent"] += total_cost
            
        except Exception as e:
            self._logger.error(f"Error updating purchase stats: {e}")
            
    def _update_transaction_history(self, processed_purchases: List[Dict[str, Any]]) -> None:
        """Update player's transaction history with processed purchases.
        
        Args:
            processed_purchases: List of successfully processed purchases
        """
        try:
            player = self.player
            
            # Initialize transaction history if it doesn't exist
            if not hasattr(player, "transaction_history"):
                player.transaction_history = []
                
            # Add purchases to history
            for purchase in processed_purchases:
                transaction = {
                    "type": "purchase",
                    "item_id": purchase["item_id"],
                    "quantity": purchase["quantity"],
                    "cost": purchase["cost"],
                    "timestamp": purchase["timestamp"]
                }
                player.transaction_history.append(transaction)
                
            self._logger.debug(f"Added {len(processed_purchases)} transactions to history")
            
        except Exception as e:
            self._logger.error(f"Error updating transaction history: {e}")
            
    def _update_shop_inventory(self) -> None:
        """Update shop inventory after purchases if needed."""
        try:
            # Check if shop inventory tracking is enabled
            if not hasattr(self, "shop_inventory") or not hasattr(self, "shop_inventory_tracking"):
                return
                
            if not self.shop_inventory_tracking:
                return
                
            # Update inventory based on purchases
            if hasattr(self, "processed_purchases") and self.processed_purchases:
                for purchase in self.processed_purchases:
                    item_id = purchase["item_id"]
                    quantity = purchase["quantity"]
                    
                    if item_id in self.shop_inventory and self.shop_inventory[item_id] > 0:
                        self.shop_inventory[item_id] -= min(quantity, self.shop_inventory[item_id])
                        self._logger.debug(f"Updated shop inventory for {item_id}: {self.shop_inventory[item_id]} remaining")
                        
        except Exception as e:
            self._logger.error(f"Error updating shop inventory: {e}")
            
    def _exit_hide_map(self) -> None:
        """Clean up map interface when exiting the map state.
        
        This state exit handler ensures proper cleanup of the map UI elements
        and restores the game UI when exiting the map view.
        """
        try:
            self._logger.info("Cleaning up map interface")
            
            # Check if UI component dictionary exists
            if not hasattr(self, "ui_elements"):
                self._logger.warning(ERROR_UI_DICT_NOT_FOUND)
                return
                
            # Hide map UI elements
            self._hide_map_ui_elements()
            
            # Reset map state variables
            self._reset_map_state()
            
            # Restore game UI visibility
            self._restore_game_ui_after_map()
            
            self._logger.info("Map interface cleaned up successfully")
            
        except Exception as e:
            self._logger.error(f"Error cleaning up map interface: {e}")
            
    def _hide_map_ui_elements(self) -> None:
        """Hide all map-related UI elements."""
        try:
            # List of map-related UI elements
            map_elements = ["map_view", "map_panel", "map_controls", "map_legend"]
            
            for element_name in map_elements:
                if element_name not in self.ui_elements:
                    continue
                    
                element = self.ui_elements[element_name]
                
                # Hide the element if it has visible property
                if hasattr(element, "visible"):
                    element.visible = False
                    self._logger.debug(f"Set {element_name} visibility to False")
                    
                # If element has hide method, call it
                if hasattr(element, "hide") and callable(element.hide):
                    try:
                        element.hide()
                        self._logger.debug(f"Called hide() for {element_name}")
                    except Exception as e:
                        self._logger.error(f"Error hiding {element_name}: {e}")
            
            self._logger.debug("Finished hiding map UI elements")
            
        except Exception as e:
            self._logger.error(f"Error hiding map UI elements: {e}")
            
    def _reset_map_state(self) -> None:
        """Reset map-related state variables."""
        try:
            # Reset map selection and interaction state
            map_state_vars = ["map_zoom", "map_position", "map_selection", "highlighted_location"]
            
            for var_name in map_state_vars:
                if hasattr(self, var_name):
                    if isinstance(getattr(self, var_name), tuple) or isinstance(getattr(self, var_name), list):
                        setattr(self, var_name, [])
                    elif isinstance(getattr(self, var_name), dict):
                        setattr(self, var_name, {})
                    elif isinstance(getattr(self, var_name), (int, float)):
                        setattr(self, var_name, 0)
                    else:
                        setattr(self, var_name, None)
                    self._logger.debug(f"Reset map state variable: {var_name}")
            
            # Clear temporary map data if it exists
            if hasattr(self, "temp_map_data"):
                self.temp_map_data = None
                self._logger.debug("Cleared temporary map data")
                
            self._logger.debug("Finished resetting map state")
            
        except Exception as e:
            self._logger.error(f"Error resetting map state: {e}")
            
    def _restore_game_ui_after_map(self) -> None:
        """Restore game UI elements after exiting map."""
        try:
            # List of game UI elements to restore
            game_ui_elements = ["player_hud", "main_view", "game_controls"]
            
            for element_name in game_ui_elements:
                if element_name not in self.ui_elements:
                    continue
                    
                element = self.ui_elements[element_name]
                
                # Show the element if it has visible property
                if hasattr(element, "visible"):
                    element.visible = True
                    self._logger.debug(f"Set {element_name} visibility to True")
                    
                # If element has show method, call it
                if hasattr(element, "show") and callable(element.show):
                    try:
                        element.show()
                        self._logger.debug(f"Called show() for {element_name}")
                    except Exception as e:
                        self._logger.error(f"Error showing {element_name}: {e}")
                        
            self._logger.debug("Finished restoring game UI elements")
            
        except Exception as e:
            self._logger.error(f"Error restoring game UI: {e}")
            
    def _exit_save_high_score(self) -> None:
        """Save the player's high score when exiting game over state.
        
        This state exit handler records the player's score, updates the high score list
        if the current score qualifies, and saves the updated high scores to a file.
        """
        try:
            self._logger.info("Processing and saving high score")
            
            # Get current score
            current_score = self._get_current_score()
            if current_score <= 0:
                self._logger.debug("No valid score to save")
                return
                
            # Load existing high scores
            high_scores = self._load_high_scores()
            
            # Check if current score qualifies as a high score
            if self._is_high_score(current_score, high_scores):
                # Add current score to high scores
                player_name = self._get_player_name()
                self._add_to_high_scores(player_name, current_score, high_scores)
                
                # Save updated high scores
                self._save_high_scores(high_scores)
                self._logger.info(f"New high score saved: {current_score}")
            else:
                self._logger.debug(f"Score {current_score} not high enough to qualify")
                
        except Exception as e:
            self._logger.error(f"Error saving high score: {e}")
            
    def _get_current_score(self) -> int:
        """Get the player's current score.
        
        Returns:
            int: Current score, or 0 if not available
        """
        try:
            # Check game stats for score
            if hasattr(self, "game_stats") and isinstance(self.game_stats, dict):
                if "score" in self.game_stats:
                    return self.game_stats["score"]
                    
            # Check player object for score
            if hasattr(self, "player") and hasattr(self.player, "score"):
                return self.player.score
                
            # Try other possible score locations
            if hasattr(self, "score"):
                return self.score
                
            self._logger.warning("Could not find current score")
            return 0
            
        except Exception as e:
            self._logger.error(f"Error getting current score: {e}")
            return 0
            
    def _load_high_scores(self) -> List[Dict[str, Any]]:
        """Load existing high scores from file.
        
        Returns:
            List[Dict[str, Any]]: List of high score entries, empty if none exist
        """
        try:
            # Default high scores file path
            file_path = os.path.join(self.save_dir, "high_scores.json")
            
            # If file doesn't exist, return empty list
            if not os.path.exists(file_path):
                self._logger.debug(f"High scores file not found at {file_path}")
                return []
                
            # Load high scores from file
            with open(file_path, "r") as f:
                high_scores = json.load(f)
                
            # Validate that high_scores is a list
            if not isinstance(high_scores, list):
                self._logger.warning("Invalid high scores format, resetting")
                return []
                
            self._logger.debug(f"Loaded {len(high_scores)} high scores")
            return high_scores
            
        except Exception as e:
            self._logger.error(f"Error loading high scores: {e}")
            return []
            
    def _is_high_score(self, score: int, high_scores: List[Dict[str, Any]]) -> bool:
        """Check if the current score qualifies as a high score.
        
        Args:
            score: Current score to check
            high_scores: List of existing high scores
            
        Returns:
            bool: True if the score qualifies, False otherwise
        """
        try:
            # If we have fewer than 10 high scores, any positive score qualifies
            if len(high_scores) < 10:
                return score > 0
                
            # Check if the current score is higher than the lowest high score
            lowest_score = float('inf')
            for entry in high_scores:
                if "score" in entry and entry["score"] < lowest_score:
                    lowest_score = entry["score"]
                    
            return score > lowest_score
            
        except Exception as e:
            self._logger.error(f"Error checking if score qualifies: {e}")
            # Default to allowing the score to be saved if there's an error
            return True
            
    def _get_player_name(self) -> str:
        """Get the player's name for the high score entry.
        
        Returns:
            str: Player name, or "Player" if not available
        """
        try:
            # Check player object for name
            if hasattr(self, "player"):
                if hasattr(self.player, "name") and self.player.name:
                    return self.player.name
                    
            # Check game settings for player name
            if hasattr(self, "settings") and isinstance(self.settings, dict):
                if "player_name" in self.settings and self.settings["player_name"]:
                    return self.settings["player_name"]
                    
            # Default player name
            return "Player"
            
        except Exception as e:
            self._logger.error(f"Error getting player name: {e}")
            return "Player"
            
    def _add_to_high_scores(self, player_name: str, score: int, high_scores: List[Dict[str, Any]]) -> None:
        """Add the current score to the high scores list.
        
        Args:
            player_name: Name of the player
            score: Score to add
            high_scores: List of existing high scores to update
        """
        try:
            # Create new high score entry
            new_entry = {
                "name": player_name,
                "score": score,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "play_time": getattr(self, "play_time", 0)
            }
            
            # Add new entry to high scores
            high_scores.append(new_entry)
            
            # Sort high scores by score (descending)
            high_scores.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Keep only the top 10 scores
            if len(high_scores) > 10:
                high_scores = high_scores[:10]
                
            self._logger.debug(f"Added score {score} for {player_name} to high scores")
            
        except Exception as e:
            self._logger.error(f"Error adding to high scores: {e}")
            
    def _save_high_scores(self, high_scores: List[Dict[str, Any]]) -> None:
        """Save high scores to file.
        
        Args:
            high_scores: List of high score entries to save
        """
        try:
            # Create save directory if it doesn't exist
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                self._logger.debug(f"Created save directory: {self.save_dir}")
                
            # Save high scores to file
            file_path = os.path.join(self.save_dir, "high_scores.json")
            with open(file_path, "w") as f:
                json.dump(high_scores, f, indent=2)
                
            self._logger.debug(f"Saved {len(high_scores)} high scores to {file_path}")
            
        except Exception as e:
            self._logger.error(f"Error saving high scores: {e}")
            
    def _exit_reset_game(self) -> None:
        """Reset the game when exiting certain game states.
        
        This handler is typically called when transitioning from game over state
        back to the main menu or when starting a new game. It resets all game
        variables and state to their initial values.
        """
        try:
            self._logger.info("Resetting game state")
            self._perform_full_reset()
            self._logger.info("Game state successfully reset")
        except Exception as e:
            self._logger.error(f"Error resetting game: {e}")
            
    def _perform_full_reset(self) -> None:
        """Perform all reset operations in sequence."""
        # Reset game variables
        self._reset_game_variables()
        
        # Reset player state
        self._reset_player_state()
        
        # Reset game world
        self._reset_game_world()
        
        # Reset UI elements
        self._reset_ui_elements()
        
        # Reset game timer and stats
        self._reset_game_timer()
        self._reset_game_stats()
        
        # Reset any ongoing animations or effects
        self._clear_animations()
            
    def _reset_game_variables(self) -> None:
        """Reset core game variables to their initial values."""
        try:
            self._logger.debug("Resetting game variables")
            
            # Reset game flags
            self.paused = False
            self.game_over = False
            self.victory = False
            
            # Reset game progress tracking
            self.current_level = 1
            self.difficulty = self.initial_difficulty
            
            # Reset timers
            self.elapsed_time = 0
            self.play_time = 0
            
            # Reset any cached data
            self.cached_data = {}
            
        except Exception as e:
            self._logger.error(f"Error resetting game variables: {e}")
            
    def _reset_player_state(self) -> None:
        """Reset the player character to initial state."""
        try:
            self._logger.debug("Resetting player state")
            
            # Check if player exists and has reset method
            if hasattr(self, "player"):
                if hasattr(self.player, "reset") and callable(self.player.reset):
                    self.player.reset()
                    self._logger.debug("Called player.reset()")
                    return
                    
                # Otherwise, recreate player with initial values
                self._create_new_player()
                
        except Exception as e:
            self._logger.error(f"Error resetting player state: {e}")
            
    def _create_new_player(self) -> None:
        """Create a new player with initial values."""
        try:
            # Get initial player config
            player_config = self.GAME_CONFIG.get("player", {})
            
            # Create new player instance using factory if available
            if self._try_create_player_from_factory(player_config):
                return
                
            # Otherwise reset existing player attributes
            if hasattr(self, "player"):
                self._reset_player_attributes(player_config)
                
        except Exception as e:
            self._logger.error(f"Error creating new player: {e}")
            
    def _try_create_player_from_factory(self, player_config: Dict[str, Any]) -> bool:
        """Try to create a new player using player class constructor.
        
        Args:
            player_config: Configuration for the new player
            
        Returns:
            bool: True if player was created, False otherwise
        """
        # Check for any player class factory - try common naming conventions
        for class_name in ["PlayerClass", "Player", "player_factory"]:
            if hasattr(self, class_name) and callable(getattr(self, class_name)):
                factory = getattr(self, class_name)
                self.player = factory(**player_config)
                self._logger.debug(f"Created new player instance using {class_name}")
                return True
                
        return False
        
    def _reset_player_attributes(self, player_config: Dict[str, Any]) -> None:
        """Reset player attributes to initial values.
        
        Args:
            player_config: Configuration with initial values
        """
        player = self.player
        
        # Reset basic attributes
        self._reset_player_basic_stats(player, player_config)
        
        # Reset inventory
        self._reset_player_inventory(player)
        
        # Reset position and score
        self._reset_player_position_and_score(player, player_config)
            
        self._logger.debug("Reset player attributes manually")
        
    def _reset_player_basic_stats(self, player, player_config: Dict[str, Any]) -> None:
        """Reset player's basic stats like health and energy.
        
        Args:
            player: Player object
            player_config: Configuration with initial values
        """
        if hasattr(player, "health"):
            player.health = player_config.get("health", 100)
            
        if hasattr(player, "energy"):
            player.energy = player_config.get("energy", 100)
            
        if hasattr(player, "resources") and isinstance(player.resources, dict):
            player.resources = player_config.get("resources", {})
            
    def _reset_player_inventory(self, player) -> None:
        """Reset player's inventory.
        
        Args:
            player: Player object
        """
        if not hasattr(player, "inventory"):
            return
            
        inventory = player.inventory
        if hasattr(inventory, "clear") and callable(inventory.clear):
            inventory.clear()
        elif hasattr(inventory, "items") and isinstance(inventory.items, dict):
            inventory.items = {}
            
    def _reset_player_position_and_score(self, player, player_config: Dict[str, Any]) -> None:
        """Reset player's position and score.
        
        Args:
            player: Player object
            player_config: Configuration with initial values
        """
        if hasattr(player, "position"):
            player.position = player_config.get("start_position", (0, 0))
            
        if hasattr(player, "score"):
            player.score = 0
            
    def _reset_game_world(self) -> None:
        """Reset the game world and environment."""
        try:
            self._logger.debug("Resetting game world")
            
            # Reset world components
            self._reset_world_generator()
            self._reset_entities()
            self._regenerate_world()
                
            self._logger.debug("Game world reset")
            
        except Exception as e:
            self._logger.error(f"Error resetting game world: {e}")
            
    def _reset_world_generator(self) -> None:
        """Reset the world generator if it exists."""
        if hasattr(self, "world_generator"):
            if hasattr(self.world_generator, "reset") and callable(self.world_generator.reset):
                self.world_generator.reset()
                
    def _reset_entities(self) -> None:
        """Reset all game entities (enemies, collectibles, etc)."""
        # Reset enemy spawns
        self._clear_collection("enemies")
            
        # Reset collectibles
        self._clear_collection("collectibles")
        
    def _clear_collection(self, collection_name: str) -> None:
        """Clear a collection of entities.
        
        Args:
            collection_name: Name of the collection attribute
        """
        if not hasattr(self, collection_name):
            return
            
        collection = getattr(self, collection_name)
        # Handle both list-type collections and collections with clear() method
        if hasattr(collection, "clear") and callable(collection.clear):
            collection.clear()
        # If no clear method but it's iterable, try to empty it another way
        elif isinstance(collection, (list, dict, set)):
            if isinstance(collection, dict):
                collection.clear()  # dict.clear() always exists
            elif isinstance(collection, list):
                while collection:  # This is a fallback approach
                    collection.pop()
            elif isinstance(collection, set):
                collection.clear()  # set.clear() always exists
            
    def _regenerate_world(self) -> None:
        """Regenerate the initial world if possible."""
        if hasattr(self, "generate_world") and callable(self.generate_world):
            self.generate_world(level=1)
            
    def _reset_ui_elements(self) -> None:
        """Reset UI elements to their initial state."""
        try:
            self._logger.debug("Resetting UI elements")
            
            # Hide overlay UI elements
            self._hide_overlay_ui_elements()
                        
            # Reset and show main game UI elements
            self._reset_main_ui_elements()
                        
            self._logger.debug("UI elements reset")
            
        except Exception as e:
            self._logger.error(f"Error resetting UI elements: {e}")
            
    def _hide_overlay_ui_elements(self) -> None:
        """Hide all overlay UI elements."""
        overlay_elements = ["pause_menu", "shop_menu", "map_screen", "game_over_screen"]
        
        for element_name in overlay_elements:
            self._hide_ui_element(element_name)
            
    def _hide_ui_element(self, element_name: str) -> None:
        """Hide a specific UI element.
        
        Args:
            element_name: Name of the UI element to hide
        """
        if element_name not in self.ui_elements:
            return
            
        element = self.ui_elements[element_name]
        
        # Set visibility to false
        if hasattr(element, "visible"):
            element.visible = False
            
        # Call hide method if available
        if hasattr(element, "hide") and callable(element.hide):
            element.hide()
            
    def _reset_main_ui_elements(self) -> None:
        """Reset and show main game UI elements."""
        main_elements = ["main_view", "player_hud", "game_controls"]
        
        for element_name in main_elements:
            self._reset_and_show_ui_element(element_name)
            
    def _reset_and_show_ui_element(self, element_name: str) -> None:
        """Reset and show a specific UI element.
        
        Args:
            element_name: Name of the UI element to reset and show
        """
        if element_name not in self.ui_elements:
            return
            
        element = self.ui_elements[element_name]
        
        # Call reset method if available
        if hasattr(element, "reset") and callable(element.reset):
            element.reset()
            
        # Set visibility to true
        if hasattr(element, "visible"):
            element.visible = True
            
        # Call show method if available
        if hasattr(element, "show") and callable(element.show):
            element.show()
            
    def _reset_game_timer(self) -> None:
        """Reset game timers and time-based variables."""
        try:
            self._logger.debug("Resetting game timer")
            
            # Reset time tracking
            self.game_start_time = time.time()
            self.elapsed_time = 0
            self.last_update_time = time.time()
            
            # Reset any time-based cooldowns
            self.cooldowns = {}
            
            # Reset frame counters
            self.current_frame = 0
            
            self._logger.debug("Game timer reset")
            
        except Exception as e:
            self._logger.error(f"Error resetting game timer: {e}")
            
    def _reset_game_stats(self) -> None:
        """Reset game statistics and metrics."""
        try:
            self._logger.debug("Resetting game statistics")
            
            # Initialize empty stats dictionary if needed
            if not hasattr(self, "game_stats") or not isinstance(self.game_stats, dict):
                self.game_stats = {}
                
            # Reset all game stats
            self.game_stats = {
                "score": 0,
                "enemies_defeated": 0,
                "resources_collected": 0,
                "distance_traveled": 0,
                "items_purchased": 0,
                "deaths": 0,
                "levels_completed": 0
            }
            
            self._logger.debug("Game statistics reset")
            
        except Exception as e:
            self._logger.error(f"Error resetting game stats: {e}")
            
    def _clear_animations(self) -> None:
        """Clear any ongoing animations or visual effects."""
        try:
            self._logger.debug("Clearing animations and effects")
            
            # Clear animation collections
            self._clear_collection("animations")
            self._clear_collection("particles")
            
            # Reset sound effects
            self._stop_sounds()
                
            self._logger.debug("Animations and effects cleared")
            
        except Exception as e:
            self._logger.error(f"Error clearing animations: {e}")
            
    def _stop_sounds(self) -> None:
        """Stop all active sound effects."""
        if hasattr(self, "_stop_all_sounds") and callable(self._stop_all_sounds):
            self._stop_all_sounds()
