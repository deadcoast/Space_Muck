"""
ASCII Game Screen for Space Muck.

This module provides the main game screen component that serves as a central hub
for integrating all UI components with proper layout management.
"""

# Standard library imports
import time
from typing import Any, Dict, Optional

import pygame

from config import COLOR_TEXT

# Local application imports
from ui.ui_base.ascii_base import UIStyle
from ui.ui_base.ascii_ui import ASCIIBox, ASCIIButton

# Third-party library imports


class ASCIIGameScreen:
    """
    A central hub that integrates all UI components with proper layout management.

    This component serves as the main game screen, organizing various panels and
    UI elements into a cohesive interface following the Space Muck ASCII aesthetic.
    """

    def __init__(
        self,
        rect: pygame.Rect,
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """
        Initialize the main game screen.

        Args:
            rect: Rectangle defining position and size
            style: Visual style for the UI
        """
        self.rect = rect
        self.style = style
        self.components: Dict[str, Any] = {}
        self.active_modal: Optional[str] = None

        # Track animation and update state
        self.last_update_time = time.time()
        self.animation_state = {
            "active": False,
            "progress": 0.0,
            "start_time": 0,
            "duration": 0.8,
        }

        # Initialize UI layout
        self._initialize_layout()

        # Subscribe to relevant events
        self._subscribe_to_events()

    def _initialize_layout(self) -> None:
        """Initialize the layout of all UI components."""
        screen_width = self.rect.width
        screen_height = self.rect.height

        # Calculate layout dimensions
        sidebar_width = int(screen_width * 0.25)
        main_area_width = screen_width - sidebar_width
        header_height = int(screen_height * 0.08)
        footer_height = int(screen_height * 0.05)
        main_content_height = screen_height - header_height - footer_height

        # Create header panel (title and quick modules)
        header_rect = pygame.Rect(0, 0, screen_width, header_height)
        self.components["header"] = ASCIIBox(
            header_rect, title="SPACE MUCK", style=self.style
        )

        # Create sidebar panel
        sidebar_rect = pygame.Rect(0, header_height, sidebar_width, main_content_height)
        self.components["sidebar"] = ASCIIBox(
            sidebar_rect, title="QUICK MODULES", style=self.style
        )

        # Create main content area
        main_content_rect = pygame.Rect(
            sidebar_width, header_height, main_area_width, main_content_height
        )
        self.components["main_content"] = ASCIIBox(
            main_content_rect, title="MAIN DISPLAY", style=self.style
        )

        # Create footer panel (status bar)
        footer_rect = pygame.Rect(
            0, header_height + main_content_height, screen_width, footer_height
        )
        self.components["footer"] = ASCIIBox(footer_rect, style=self.style)

        # Add sidebar navigation buttons
        self._initialize_sidebar_buttons()

    def _initialize_sidebar_buttons(self) -> None:
        """Initialize the navigation buttons in the sidebar."""
        sidebar = self.components["sidebar"]
        sidebar_width = sidebar.rect.width
        button_x = sidebar.rect.x + int(sidebar_width * 0.1)
        button_y = sidebar.rect.y + 30
        button_spacing = 25

        # Create navigation buttons
        nav_buttons = [
            ("NAVIGATION", self._on_navigation_click),
            ("CARGO BAY", self._on_cargo_click),
            ("STATION DOCK", self._on_station_click),
            ("AUTO-CHART", self._on_auto_chart_click),
            ("MINING LOG", self._on_mining_log_click),
            ("CREW ASSIGN", self._on_crew_assign_click),
            ("COMMS", self._on_comms_click),
            ("DRONE OPS", self._on_drone_ops_click),
            ("TASK LIST", self._on_task_list_click),
        ]

        # Add buttons to sidebar
        for i, (label, callback) in enumerate(nav_buttons):
            button = ASCIIButton(
                button_x, button_y + (i * button_spacing), label, callback, self.style
            )
            sidebar.add_component(button)
            self.components[f"btn_{label.lower().replace(' ', '_')}"] = button

    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant game events."""
        # This would connect to the event system
        # EventSystem().subscribe("resource_update", self._on_resource_update)
        # EventSystem().subscribe("location_change", self._on_location_change)
        # EventSystem().subscribe("alert_status", self._on_alert_status)
        pass

    def set_active_content(self, content_key: str) -> None:
        """
        Set the active content to display in the main content area.

        Args:
            content_key: Key identifying the content to display
        """
        # This would swap out the content in the main display area
        # based on which sidebar button was clicked
        pass

    def show_modal(self, modal_key: str) -> None:
        """
        Show a modal dialog over the main interface.

        Args:
            modal_key: Key identifying the modal to display
        """
        self.active_modal = modal_key
        self.animation_state["active"] = True
        self.animation_state["progress"] = 0.0
        self.animation_state["start_time"] = time.time()

    def hide_modal(self) -> None:
        """Hide the currently active modal dialog."""
        self.active_modal = None

    def update(
        self, player_position=None, resources=None, health=None, energy=None
    ) -> None:
        """Update animation and component states.

        Args:
            player_position: Optional tuple of (x, y) coordinates
            resources: Optional dictionary of player resources
            health: Optional player health value
            energy: Optional player energy value
        """
        current_time = time.time()
        self.last_update_time = current_time

        # Update animation state
        if self.animation_state["active"]:
            elapsed = current_time - self.animation_state["start_time"]
            self.animation_state["progress"] = min(
                1.0, elapsed / self.animation_state["duration"]
            )
            if self.animation_state["progress"] >= 1.0:
                self.animation_state["active"] = False

        # Update component data if provided
        if player_position is not None:
            self.player_position = player_position

        if resources is not None:
            self.resources = resources

        if health is not None:
            self.health = health

        if energy is not None:
            self.energy = energy

    def update_animations(self, delta_time: float) -> None:
        """Update animations based on elapsed time.

        Args:
            delta_time: Time elapsed since last frame in seconds
        """
        # Update animation state
        if self.animation_state["active"]:
            self.animation_state["progress"] = min(
                1.0,
                self.animation_state["progress"]
                + delta_time / self.animation_state["duration"],
            )
            if self.animation_state["progress"] >= 1.0:
                self.animation_state["active"] = False

        # Update animations for all components
        for component in self.components.values():
            if hasattr(component, "update_animation"):
                component.update_animation()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame events for this screen.

        Args:
            event: Pygame event to handle

        Returns:
            bool: True if the event was consumed
        """
        # Handle modal events first if a modal is active
        if (
            self.active_modal
            and self.active_modal in self.components
            and self.components[self.active_modal].handle_event(event)
        ):
            return True

        # Then try the main components
        return any(
            hasattr(component, "handle_event") and component.handle_event(event)
            for _, component in self.components.items()
        )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """
        Draw the game screen and all its components.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        # Draw the base screen components
        for key, component in self.components.items():
            if key not in ["header", "sidebar", "main_content", "footer"]:
                continue
            component.draw(surface, font)

        # Draw the status information in the footer
        self._draw_status_bar(surface, font)

        # Draw modal on top if active
        if self.active_modal and self.active_modal in self.components:
            self.components[self.active_modal].draw(surface, font)

        return self.rect

    def _draw_status_bar(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the status bar in the footer panel."""
        footer = self.components["footer"]
        footer_rect = footer.rect

        # Status information
        status_items = [
            "TIME: 14:47",
            "SECTOR: BETA-12",
            "ALERT: GREEN",
            "CREDITS: 7420",
            "FPS: 60",
        ]

        # Calculate positions
        total_width = footer_rect.width - 20
        item_width = total_width // len(status_items)

        for i, item in enumerate(status_items):
            x = footer_rect.x + 10 + (i * item_width)
            y = footer_rect.y + (footer_rect.height // 2) - (font.get_height() // 2)
            text_surface = font.render(f"[{item}]", True, COLOR_TEXT)
            surface.blit(text_surface, (x, y))

    # Button callback methods
    def _on_navigation_click(self) -> None:
        """Handle navigation button click."""
        self.set_active_content("navigation")

    def _on_cargo_click(self) -> None:
        """Handle cargo bay button click."""
        self.set_active_content("cargo")

    def _on_station_click(self) -> None:
        """Handle station dock button click."""
        self.set_active_content("station")

    def _on_auto_chart_click(self) -> None:
        """Handle auto-chart button click."""
        self.set_active_content("auto_chart")

    def _on_mining_log_click(self) -> None:
        """Handle mining log button click."""
        self.set_active_content("mining_log")

    def _on_crew_assign_click(self) -> None:
        """Handle crew assignment button click."""
        self.set_active_content("crew_assign")

    def _on_comms_click(self) -> None:
        """Handle communications button click."""
        self.set_active_content("comms")

    def _on_drone_ops_click(self) -> None:
        """Handle drone operations button click."""
        self.set_active_content("drone_ops")

    def _on_task_list_click(self) -> None:
        """Handle task list button click."""
        self.set_active_content("task_list")
