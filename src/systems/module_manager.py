"""
Module Management System: Handles module registration, lifecycle, and operations.

This module provides functionality for managing game modules, including
module registration, state management, and inter-module communication.
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto


# Module States
class ModuleState(Enum):
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()
    TERMINATED = auto()


# Module Categories
class ModuleCategory(Enum):
    CORE = auto()  # Essential system modules
    RESOURCE = auto()  # Resource handling modules
    PRODUCTION = auto()  # Production and crafting modules
    RESEARCH = auto()  # Research and development modules
    EXPLORATION = auto()  # Exploration and discovery modules
    UI = auto()  # User interface modules
    UTILITY = auto()  # Helper and utility modules


# Module Dependencies
@dataclass
class ModuleDependency:
    """Represents a module's dependency on another module."""

    module_id: str
    required: bool = True
    min_version: Optional[str] = None
    max_version: Optional[str] = None


# Module Configuration
@dataclass
class ModuleConfig:
    """Configuration settings for a module."""

    update_interval: float = 1.0
    auto_start: bool = True
    priority: int = 1
    category: ModuleCategory = ModuleCategory.UTILITY


# Module Interface Definition
class ModuleInterface:
    """Base interface that all modules must implement."""

    def initialize(self) -> bool:
        """Initialize the module."""
        raise NotImplementedError

    def update(self, dt: float) -> None:
        """Update module state."""
        raise NotImplementedError

    def pause(self) -> None:
        """Pause module operations."""
        raise NotImplementedError

    def resume(self) -> None:
        """Resume module operations."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the module."""
        raise NotImplementedError


class ModuleManager:
    """
    Central manager for handling all module-related operations.
    """

    def __init__(self) -> None:
        """Initialize the module manager."""
        # Module tracking
        self.modules: Dict[str, ModuleInterface] = {}
        self.states: Dict[str, ModuleState] = {}
        self.configs: Dict[str, ModuleConfig] = {}
        self.dependencies: Dict[str, List[ModuleDependency]] = {}

        # Update tracking
        self.update_times: Dict[str, float] = {}
        self.error_counts: Dict[str, int] = {}
        self.max_errors = 3  # Maximum errors before disabling a module

        # System state
        self.active = True
        self.paused = False

        # Event handlers
        self.on_module_state_change: Dict[str, List[Callable[[ModuleState], None]]] = {}

        logging.info("ModuleManager initialized")

    def register_module(
        self,
        module_id: str,
        module: ModuleInterface,
        config: Optional[ModuleConfig] = None,
        dependencies: Optional[List[ModuleDependency]] = None,
    ) -> bool:
        """
        Register a new module with the system.

        Args:
            module_id: Unique identifier for the module
            module: Module instance implementing ModuleInterface
            config: Optional module configuration
            dependencies: Optional list of module dependencies

        Returns:
            bool: True if registration successful
        """
        if module_id in self.modules:
            logging.warning(f"Module {module_id} already registered")
            return False

        self.modules[module_id] = module
        self.states[module_id] = ModuleState.UNINITIALIZED
        self.configs[module_id] = config or ModuleConfig()
        self.dependencies[module_id] = dependencies or []
        self.update_times[module_id] = 0.0
        self.error_counts[module_id] = 0
        self.on_module_state_change[module_id] = []

        logging.info(f"Registered module {module_id}")
        return True

    def unregister_module(self, module_id: str) -> bool:
        """
        Unregister a module from the system.

        Args:
            module_id: ID of module to unregister

        Returns:
            bool: True if unregistration successful
        """
        if module_id not in self.modules:
            logging.warning(f"Module {module_id} not found")
            return False

        # Check if other modules depend on this one
        for dep_id, deps in self.dependencies.items():
            if any(dep.module_id == module_id and dep.required for dep in deps):
                logging.error(f"Cannot unregister {module_id}: required by {dep_id}")
                return False

        # Shutdown module if active
        if self.states[module_id] in {ModuleState.ACTIVE, ModuleState.PAUSED}:
            try:
                self.modules[module_id].shutdown()
            except Exception as e:
                logging.error(f"Error shutting down {module_id}: {e}")

        # Clean up all references
        del self.modules[module_id]
        del self.states[module_id]
        del self.configs[module_id]
        del self.dependencies[module_id]
        del self.update_times[module_id]
        del self.error_counts[module_id]
        del self.on_module_state_change[module_id]

        logging.info(f"Unregistered module {module_id}")
        return True

    def start_module(self, module_id: str) -> bool:
        """
        Start a module.

        Args:
            module_id: ID of module to start

        Returns:
            bool: True if start successful
        """
        if module_id not in self.modules:
            logging.error(f"Module {module_id} not found")
            return False

        if self.states[module_id] != ModuleState.UNINITIALIZED:
            logging.warning(f"Module {module_id} already started")
            return False

        # Check dependencies
        for dep in self.dependencies[module_id]:
            if dep.required and (
                dep.module_id not in self.modules
                or self.states[dep.module_id] != ModuleState.ACTIVE
            ):
                logging.error(f"Module {module_id} missing dependency: {dep.module_id}")
                return False

        # Initialize module
        try:
            self._set_module_state(module_id, ModuleState.INITIALIZING)
            if self.modules[module_id].initialize():
                self._set_module_state(module_id, ModuleState.ACTIVE)
                logging.info(f"Started module {module_id}")
                return True
            else:
                self._set_module_state(module_id, ModuleState.ERROR)
                logging.error(f"Failed to initialize module {module_id}")
                return False
        except Exception as e:
            self._set_module_state(module_id, ModuleState.ERROR)
            logging.error(f"Error starting module {module_id}: {e}")
            return False

    def stop_module(self, module_id: str) -> bool:
        """
        Stop a module.

        Args:
            module_id: ID of module to stop

        Returns:
            bool: True if stop successful
        """
        if module_id not in self.modules:
            logging.error(f"Module {module_id} not found")
            return False

        if self.states[module_id] not in {ModuleState.ACTIVE, ModuleState.PAUSED}:
            logging.warning(f"Module {module_id} not running")
            return False

        try:
            self._set_module_state(module_id, ModuleState.SHUTTING_DOWN)
            self.modules[module_id].shutdown()
            self._set_module_state(module_id, ModuleState.TERMINATED)
            logging.info(f"Stopped module {module_id}")
            return True
        except Exception as e:
            self._set_module_state(module_id, ModuleState.ERROR)
            logging.error(f"Error stopping module {module_id}: {e}")
            return False

    def update(self, dt: float) -> None:
        """
        Update all active modules.

        Args:
            dt: Time delta since last update
        """
        if not self.active or self.paused:
            return

        for module_id, module in self.modules.items():
            if self.states[module_id] != ModuleState.ACTIVE:
                continue

            self.update_times[module_id] += dt
            if self.update_times[module_id] < self.configs[module_id].update_interval:
                continue

            try:
                module.update(self.update_times[module_id])
                self.update_times[module_id] = 0
                self.error_counts[module_id] = (
                    0  # Reset error count on successful update
                )
            except Exception as e:
                self.error_counts[module_id] += 1
                logging.error(f"Error updating module {module_id}: {e}")

                if self.error_counts[module_id] >= self.max_errors:
                    logging.error(f"Module {module_id} exceeded error limit, disabling")
                    self._set_module_state(module_id, ModuleState.ERROR)

    def pause_module(self, module_id: str) -> bool:
        """
        Pause a module.

        Args:
            module_id: ID of module to pause

        Returns:
            bool: True if pause successful
        """
        if module_id not in self.modules:
            logging.error(f"Module {module_id} not found")
            return False

        if self.states[module_id] != ModuleState.ACTIVE:
            logging.warning(f"Module {module_id} not active")
            return False

        try:
            self.modules[module_id].pause()
            self._set_module_state(module_id, ModuleState.PAUSED)
            logging.info(f"Paused module {module_id}")
            return True
        except Exception as e:
            logging.error(f"Error pausing module {module_id}: {e}")
            return False

    def resume_module(self, module_id: str) -> bool:
        """
        Resume a paused module.

        Args:
            module_id: ID of module to resume

        Returns:
            bool: True if resume successful
        """
        if module_id not in self.modules:
            logging.error(f"Module {module_id} not found")
            return False

        if self.states[module_id] != ModuleState.PAUSED:
            logging.warning(f"Module {module_id} not paused")
            return False

        try:
            self.modules[module_id].resume()
            self._set_module_state(module_id, ModuleState.ACTIVE)
            logging.info(f"Resumed module {module_id}")
            return True
        except Exception as e:
            logging.error(f"Error resuming module {module_id}: {e}")
            return False

    def get_module_state(self, module_id: str) -> Optional[ModuleState]:
        """
        Get the current state of a module.

        Args:
            module_id: Module to check

        Returns:
            ModuleState: Current state or None if not found
        """
        return self.states.get(module_id)

    def add_state_change_handler(
        self, module_id: str, handler: Callable[[ModuleState], None]
    ) -> bool:
        """
        Add a handler for module state changes.

        Args:
            module_id: Module to watch
            handler: Callback function for state changes

        Returns:
            bool: True if handler added successfully
        """
        if module_id not in self.modules:
            logging.error(f"Module {module_id} not found")
            return False

        self.on_module_state_change[module_id].append(handler)
        return True

    def _set_module_state(self, module_id: str, state: ModuleState) -> None:
        """
        Update a module's state and notify handlers.

        Args:
            module_id: Module to update
            state: New state
        """
        self.states[module_id] = state
        for handler in self.on_module_state_change[module_id]:
            try:
                handler(state)
            except Exception as e:
                logging.error(f"Error in state change handler for {module_id}: {e}")

    def pause(self) -> None:
        """Pause all module processing."""
        self.paused = True
        logging.info("ModuleManager paused")

    def resume(self) -> None:
        """Resume all module processing."""
        self.paused = False
        logging.info("ModuleManager resumed")

    def shutdown(self) -> None:
        """Shutdown the module manager and all modules."""
        self.active = False
        for module_id in list(self.modules.keys()):
            self.stop_module(module_id)
        logging.info("ModuleManager shut down")
