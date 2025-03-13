

# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height

class ASCIIPanel(UIElement):
    """A panel with ASCII styling for complex UI layouts.

    This class provides a container for multiple UI components with a consistent
    visual style. It supports titles, borders, and can contain any number of
    child components that implement a draw method.

    Inherits from UIElement to leverage standardized animation framework and styling.
    """

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from ..draw_utils import draw_panel
from src.ui.ui_base.ascii_base import UIStyle
from .ascii_box import ASCIIBox
from config import COLOR_TEXT
from typing import Tuple, List, Optional, Any, TypeVar
import pygame

    def __init__(
        self,
        rect: pygame.Rect,
        title: Optional[str] = None,
        style: UIStyle = UIStyle.MECHANICAL,
        converter_type: Optional[str] = None,
    ):
        """
        Initialize an ASCII panel.

        Args:
            rect: Rectangle defining position and size
            title: Optional title to display at the top
            style: Visual style for the panel
            converter_type: Optional converter type to determine style
        """
        # Determine the style based on converter type if provided
        actual_style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )

        # Initialize the parent UIElement class
        super().__init__(rect.x, rect.y, rect.width, rect.height, actual_style, title)

        # ASCIIPanel specific attributes
        self.rect = rect  # Keep rect for compatibility with existing code
        self.components: List[Any] = []

        self._initialize_colors()

    def add_component(self, component: Any) -> None:
        """
        Add a component to the panel.

        Args:
            component: Component to add
        """
        self.components.append(component)

    def _initialize_colors(self) -> None:
        """Initialize style-based background colors."""
        style_colors = {
            UIStyle.QUANTUM: (20, 20, 30, 200),
            UIStyle.SYMBIOTIC: (20, 30, 20, 200),
            UIStyle.MECHANICAL: (30, 30, 40, 200),
        }
        self.background_color = style_colors.get(self.style, (25, 25, 35, 200))

    def _create_ascii_box(self, font: pygame.font.Font) -> ASCIIBox:
        """Create an ASCII box for the panel border."""
        char_width, char_height = font.size("X")
        width_chars = self.rect.width // char_width
        height_chars = self.rect.height // char_height

        return ASCIIBox(
            self.rect.x,
            self.rect.y,
            width_chars,
            height_chars,
            title=None,  # Already handled by panel
            style=self.style,
        )

    # Override start_animation to also animate the border box if it exists
    def start_animation(self, duration: float = 0.3) -> None:
        """Start panel animation with error handling

        Args:
            duration: Animation duration in seconds
        """
        # Call the parent class's start_animation method
        super().start_animation(duration)

        # Also animate the border box if we create one
        try:
            if hasattr(self, "_box"):
                self._box.start_animation(duration)
        except Exception as e:
            logging.error(f"Error starting border box animation: {e}")

    # Override update_animation to match the parent class's signature
    def update_animation(self, dt: Optional[float] = None) -> None:
        """Update animation state with error handling

        Args:
            dt: Optional time delta in seconds. If None, will calculate based on current time.
        """
        # Call the parent class's update_animation method
        super().update_animation(dt)

        # Update the border box animation if it exists
        try:
            if hasattr(self, "_box") and self._box.animation["active"]:
                self._box.update_animation(dt)
        except Exception as e:
            logging.error(f"Error updating border box animation: {e}")

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
    ) -> pygame.Rect:
        """
        Draw the ASCII panel and its components.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        try:
            # Update animation if active
            if self.animation["active"]:
                self.update_animation()

            # Create and draw background panel
            panel_rect = draw_panel(
                surface,
                self.rect,
                color=self.background_color,
                border_color=(100, 100, 140),
                header=self.title,
                header_color=(40, 40, 60, 220),
            )

            # Draw ASCII border
            self._box = self._create_ascii_box(font)
            self._box.draw(surface, font, COLOR_TEXT, bg_color=(0, 0, 0, 0))

            # Draw components with error handling
            for component in self.components:
                try:
                    component.draw(surface, font)
                except Exception as e:
                    logging.error(f"Error drawing component in panel: {e}")

            return panel_rect
        except Exception as e:
            logging.error(f"Error drawing panel: {e}")
            return self.rect
