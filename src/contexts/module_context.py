"""
Module Context: Manages module lifecycle and dependencies.

This module provides a context for managing module states, dependencies,
and lifecycle events within the game architecture.
"""

# Standard library imports
from datetime import datetime
import logging

# Third-party library imports

# Local application imports
from .game_context import GameContext, GameEventType, GameEvent
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable


# Module States
class ModuleState(Enum):
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    ERROR = auto()
    RECOVERING = auto()
    SHUTTING_DOWN = auto()


# Module Categories
class ModuleCategory(Enum):
    CORE = auto()  # Essential system modules
    RESOURCE = auto()  # Resource management modules
    GAMEPLAY = auto()  # Game mechanics modules
    UI = auto()  # User interface modules
    UTILITY = auto()  # Helper/utility modules


@dataclass
class ModuleInfo:
    """Information about a module."""

    id: str
    category: ModuleCategory
    state: ModuleState
    dependencies: Set[str]
    error_count: int = 0
    last_error: Optional[str] = None
    recovery_attempts: int = 0


class ModuleContext:
    """
    Context for managing module lifecycle and dependencies.
    """

    def __init__(self, game_context: GameContext) -> None:
        """Initialize the module context.

        Args:
            game_context: The game context to integrate with
        """
        # Module tracking
        self.modules: Dict[str, ModuleInfo] = {}
        self.active_modules: Set[str] = set()

        # Game context integration
        self.game_context = game_context

        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = {}  # module -> dependencies
        self.reverse_deps: Dict[str, Set[str]] = {}  # module -> dependent modules

        # Error handling
        self.max_recovery_attempts = 3
        self.error_handlers: Dict[str, List[Callable[[str, Exception], None]]] = {}

        # State tracking
        self.state_history: List[Dict[str, any]] = []
        self.max_history = 1000

        logging.info("ModuleContext initialized")

    def register_module(
        self,
        module_id: str,
        category: ModuleCategory,
        dependencies: Optional[Set[str]] = None,
    ) -> bool:
        """
        Register a new module.

        Args:
            module_id: Unique module identifier
            category: Module category
            dependencies: Optional set of module dependencies

        Returns:
            bool: True if registration successful
        """
        if module_id in self.modules:
            logging.warning(f"Module {module_id} already registered")
            return False

        # Create module info
        module_info = ModuleInfo(
            id=module_id,
            category=category,
            state=ModuleState.UNINITIALIZED,
            dependencies=dependencies or set(),
        )

        # Update tracking
        self.modules[module_id] = module_info
        self.dependency_graph[module_id] = dependencies or set()
        self.error_handlers[module_id] = []

        # Update reverse dependencies
        for dep in module_info.dependencies:
            if dep not in self.reverse_deps:
                self.reverse_deps[dep] = set()
            self.reverse_deps[dep].add(module_id)

        # Record state change
        self._record_state_change(module_id, None, ModuleState.UNINITIALIZED)

        # Notify game context
        self.game_context.dispatch_event(
            GameEvent(
                type=GameEventType.MODULE_CHANGED,
                source=module_id,
                data={
                    "action": "registered",
                    "category": category.name,
                    "state": ModuleState.UNINITIALIZED.name,
                    "dependencies": list(dependencies) if dependencies else [],
                },
                timestamp=datetime.now().timestamp(),
            )
        )

        logging.info(f"Registered module {module_id} of category {category}")
        return True

    def initialize_module(self, module_id: str) -> bool:
        """
        Initialize a module.

        Args:
            module_id: Module to initialize

        Returns:
            bool: True if initialization successful
        """
        if module_id not in self.modules:
            logging.error(f"Module {module_id} not found")
            return False

        module = self.modules[module_id]

        # Check dependencies
        for dep in module.dependencies:
            if dep not in self.active_modules:
                logging.error(f"Dependency {dep} not active for {module_id}")
                return False

        # Update state
        old_state = module.state
        module.state = ModuleState.INITIALIZING
        self._record_state_change(module_id, old_state, ModuleState.INITIALIZING)

        # Activate module
        self.active_modules.add(module_id)
        module.state = ModuleState.ACTIVE
        self._record_state_change(
            module_id, ModuleState.INITIALIZING, ModuleState.ACTIVE
        )

        logging.info(f"Initialized module {module_id}")
        return True

    def handle_module_error(self, module_id: str, error: Exception) -> None:
        """
        Handle a module error.

        Args:
            module_id: Module that encountered error
            error: The error that occurred
        """
        if module_id not in self.modules:
            logging.error(f"Module {module_id} not found")
            return

        module = self.modules[module_id]

        # Update error tracking
        module.error_count += 1
        module.last_error = str(error)

        # Update state
        old_state = module.state
        module.state = ModuleState.ERROR
        self._record_state_change(module_id, old_state, ModuleState.ERROR)

        # Notify game context
        self.game_context.dispatch_event(
            GameEvent(
                type=GameEventType.MODULE_ERROR,
                source=module_id,
                data={
                    "error": str(error),
                    "error_count": module.error_count,
                    "recovery_attempts": module.recovery_attempts,
                },
                timestamp=datetime.now().timestamp(),
                priority=2,  # Higher priority for errors
            )
        )

        # Notify error handlers
        for handler in self.error_handlers[module_id]:
            try:
                handler(module_id, error)
            except Exception as e:
                logging.error(f"Error in module error handler: {e}")

        # Attempt recovery if within limits
        if module.recovery_attempts < self.max_recovery_attempts:
            self._attempt_recovery(module_id)
        else:
            logging.error(f"Max recovery attempts reached for {module_id}")

    def register_error_handler(
        self, module_id: str, handler: Callable[[str, Exception], None]
    ) -> bool:
        """
        Register an error handler for a module.

        Args:
            module_id: Module to handle errors for
            handler: Error handler callback

        Returns:
            bool: True if registration successful
        """
        if module_id not in self.modules:
            logging.error(f"Module {module_id} not found")
            return False

        self.error_handlers[module_id].append(handler)
        return True

    def get_dependent_modules(self, module_id: str) -> Set[str]:
        """
        Get modules that depend on the specified module.

        Args:
            module_id: Module to check

        Returns:
            Set[str]: Set of dependent module IDs
        """
        return self.reverse_deps.get(module_id, set()).copy()

    def get_module_state(self, module_id: str) -> Optional[ModuleState]:
        """
        Get the current state of a module.

        Args:
            module_id: Module to check

        Returns:
            Optional[ModuleState]: Current module state or None if not found
        """
        return self.modules[module_id].state if module_id in self.modules else None

    def _attempt_recovery(self, module_id: str) -> None:
        """
        Attempt to recover a failed module.

        Args:
            module_id: Module to recover
        """
        module = self.modules[module_id]
        module.recovery_attempts += 1

        # Update state
        old_state = module.state
        module.state = ModuleState.RECOVERING
        self._record_state_change(module_id, old_state, ModuleState.RECOVERING)

        # Attempt reinitialization
        if self.initialize_module(module_id):
            module.recovery_attempts = 0
            logging.info(f"Successfully recovered module {module_id}")
        else:
            module.state = ModuleState.ERROR
            self._record_state_change(
                module_id, ModuleState.RECOVERING, ModuleState.ERROR
            )
            logging.error(f"Failed to recover module {module_id}")

    def _record_state_change(
        self, module_id: str, old_state: Optional[ModuleState], new_state: ModuleState
    ) -> None:
        """
        Record a module state change.

        Args:
            module_id: Module that changed state
            old_state: Previous state
            new_state: New state
        """
        event = {
            "module_id": module_id,
            "old_state": old_state.name if old_state else None,
            "new_state": new_state.name,
            "timestamp": datetime.now().timestamp(),
        }

        self.state_history.append(event)

        # Notify game context of state change
        self.game_context.dispatch_event(
            GameEvent(
                type=GameEventType.MODULE_CHANGED,
                source=module_id,
                data={
                    "action": "state_changed",
                    "old_state": old_state.name if old_state else None,
                    "new_state": new_state.name,
                },
                timestamp=datetime.now().timestamp(),
            )
        )
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
