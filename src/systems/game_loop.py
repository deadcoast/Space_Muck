#!/usr/bin/env python3
# /src/systems/game_loop.py
"""
Game Loop System

Provides a standardized game loop implementation that follows the layered architecture
with event-driven communication. Separates update logic, rendering, and event handling
into distinct phases for better maintainability and performance monitoring.
"""

import time
import logging
from typing import Dict, List, Callable, Any, Optional, Set, TypeVar, Generic

# Local application imports
from utils.logging_setup import (
    log_performance_start,
    log_performance_end,
    LogContext,
    log_memory_usage,
)

# Type definitions
T = TypeVar("T")
ManagerType = TypeVar("ManagerType")
UpdateFunction = Callable[[float], None]
EventHandlerFunction = Callable[[], bool]
RenderFunction = Callable[[], None]


class GameLoop:
    """
    Central game loop implementation that orchestrates updates, event handling, and rendering.

    Follows the layered architecture pattern with standardized update cycles and
    performance monitoring. Serves as the central orchestration point for all game systems.
    """

    def __init__(self) -> None:
        """Initialize the game loop system with empty registries for managers and functions."""
        # Performance tracking
        self.delta_time: float = 0.0
        self.frame_counter: int = 0
        self.game_time: float = 0.0
        self.fps_history: List[float] = []
        self.performance_metrics: Dict[str, float] = {}

        # Manager registries
        self.managers: Dict[str, Any] = {}
        self.update_functions: List[UpdateFunction] = []
        self.render_functions: List[RenderFunction] = []
        self.event_handlers: List[EventHandlerFunction] = []

        # Update intervals for different systems
        self.update_intervals: Dict[str, float] = {}
        self.last_update_times: Dict[str, float] = {}

        # System state
        self.is_running: bool = False
        self.is_paused: bool = False
        self.requested_exit: bool = False

        # Logging
        logging.info("GameLoop system initialized")

    def register_manager(self, name: str, manager: Any) -> None:
        """
        Register a manager with the game loop.

        Args:
            name: Unique identifier for the manager
            manager: Manager instance to register
        """
        if name in self.managers:
            logging.warning(f"Overwriting existing manager: {name}")

        self.managers[name] = manager
        logging.info(f"Registered manager: {name}")

    def get_manager(self, name: str) -> Optional[Any]:
        """
        Get a registered manager by name.

        Args:
            name: Name of the manager to retrieve

        Returns:
            The manager instance or None if not found
        """
        return self.managers.get(name)

    def register_update_function(self, update_func: UpdateFunction) -> None:
        """
        Register a function to be called during the update phase.

        Args:
            update_func: Function that takes delta_time as parameter
        """
        self.update_functions.append(update_func)

    def register_render_function(self, render_func: RenderFunction) -> None:
        """
        Register a function to be called during the render phase.

        Args:
            render_func: Function with no parameters that handles rendering
        """
        self.render_functions.append(render_func)

    def register_event_handler(self, handler_func: EventHandlerFunction) -> None:
        """
        Register a function to be called during the event handling phase.

        Args:
            handler_func: Function that returns True if the game should exit
        """
        self.event_handlers.append(handler_func)

    def register_interval_update(self, name: str, interval: float) -> None:
        """
        Register a system that should update at a specific interval.

        Args:
            name: Name of the system
            interval: Update interval in seconds
        """
        self.update_intervals[name] = interval
        self.last_update_times[name] = 0.0

    def should_update(self, name: str) -> bool:
        """
        Check if a system should update based on its interval.

        Args:
            name: Name of the system to check

        Returns:
            True if the system should update, False otherwise
        """
        if name not in self.update_intervals:
            return True  # If no interval is set, update every frame

        interval = self.update_intervals[name]
        last_update = self.last_update_times[name]

        if self.game_time - last_update >= interval:
            self.last_update_times[name] = self.game_time
            return True

        return False

    def update(self, delta_time: float) -> None:
        """
        Update all registered systems.

        Args:
            delta_time: Time since last frame in seconds
        """
        with LogContext("GameLoop.update"):
            self.delta_time = delta_time
            self.game_time += delta_time
            self.frame_counter += 1

            # Call all update functions
            for update_func in self.update_functions:
                try:
                    log_performance_start(f"update_{update_func.__name__}")
                    update_func(delta_time)
                    log_performance_end(f"update_{update_func.__name__}")
                except Exception as e:
                    logging.error(
                        f"Error in update function {update_func.__name__}: {e}"
                    )

    def handle_events(self) -> bool:
        """
        Process all events using registered event handlers.

        Returns:
            True if the game should exit, False otherwise
        """
        with LogContext("GameLoop.handle_events"):
            for handler in self.event_handlers:
                try:
                    log_performance_start(f"event_{handler.__name__}")
                    should_exit = handler()
                    log_performance_end(f"event_{handler.__name__}")

                    if should_exit:
                        return True
                except Exception as e:
                    logging.error(f"Error in event handler {handler.__name__}: {e}")

            return self.requested_exit

    def render(self) -> None:
        """Render the game state using all registered render functions."""
        with LogContext("GameLoop.render"):
            for render_func in self.render_functions:
                try:
                    log_performance_start(f"render_{render_func.__name__}")
                    render_func()
                    log_performance_end(f"render_{render_func.__name__}")
                except Exception as e:
                    logging.error(
                        f"Error in render function {render_func.__name__}: {e}"
                    )

    def run(
        self, clock_func: Callable[[], float], frame_limit: Optional[int] = None
    ) -> None:
        """
        Run the game loop until exit is requested.

        Args:
            clock_func: Function that returns the current time in seconds
            frame_limit: Optional limit on the number of frames to run (mainly for testing)
        """
        self.is_running = True
        frame_count = 0
        previous_time = clock_func()

        logging.info("Starting game loop")

        try:
            while self.is_running:
                # Calculate delta time
                current_time = clock_func()
                delta_time = current_time - previous_time
                previous_time = current_time

                # Check for exit condition
                if self.handle_events():
                    self.is_running = False
                    break

                # Skip updates if paused but still render
                if not self.is_paused:
                    self.update(delta_time)

                # Always render
                self.render()

                # Memory usage logging (every 100 frames)
                if self.frame_counter % 100 == 0:
                    log_memory_usage()

                # FPS calculation
                if delta_time > 0:
                    current_fps = 1.0 / delta_time
                    self.fps_history.append(current_fps)
                    if len(self.fps_history) > 100:
                        self.fps_history.pop(0)

                frame_count += 1
                if frame_limit and frame_count >= frame_limit:
                    self.is_running = False

        except Exception as e:
            logging.error(f"Fatal error in game loop: {e}")
            # Force exit the loop
            self.is_running = False

        finally:
            # Cleanup resources
            logging.info("Game loop terminated")
            self.cleanup()

    def request_exit(self) -> None:
        """Request the game loop to exit on the next iteration."""
        self.requested_exit = True

    def pause(self) -> None:
        """Pause the game loop (updates will be skipped, but rendering continues)."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume the game loop if it was paused."""
        self.is_paused = False

    def get_fps(self) -> float:
        """
        Get the current frames per second.

        Returns:
            Average FPS over recent frames
        """
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

    def cleanup(self) -> None:
        """Perform cleanup operations when the game loop exits."""
        # Clear all registries
        self.managers.clear()
        self.update_functions.clear()
        self.render_functions.clear()
        self.event_handlers.clear()

        logging.info("GameLoop cleanup completed")


# Singleton instance for global access
game_loop = GameLoop()


def get_game_loop() -> GameLoop:
    """
    Get the singleton game loop instance.

    Returns:
        The global GameLoop instance
    """
    return game_loop
