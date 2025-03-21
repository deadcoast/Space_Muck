"""
ascii_template_manager.py

Provides a specialized template manager for managing converter chain templates.
"""

# Standard library imports
from typing import Any, Dict, List, Optional, Tuple, TypeVar

# Local application imports
from config import COLOR_TEXT
from src.ui.draw_utils import draw_text
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_base.ascii_ui import ASCIIButton, ASCIIPanel

# Third-party library imports
import pygame

# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height


class ASCIIChainTemplateManager:
    """Interface for managing converter chain templates."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Chain Templates",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a chain template manager interface.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style

        # Template data
        self.templates: List[Dict[str, Any]] = []
        self.selected_idx: Optional[int] = None
        self.scroll_offset = 0
        self.max_visible_items = 8

        # Create control buttons
        self.save_button = ASCIIButton(
            rect.x + rect.width // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Save Template",
            self._save_template,
            style,
        )
        self.load_button = ASCIIButton(
            rect.x + rect.width * 3 // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Load Template",
            self._load_template,
            style,
        )

    def set_templates(self, templates: List[Dict[str, Any]]) -> None:
        """Set the list of available templates.

        Args:
            templates: List of template info dictionaries
        """
        self.templates = templates
        self.selected_idx = 0 if templates else None
        self.scroll_offset = 0

    def _save_template(self) -> None:
        """Save the current chain as a template."""
        # This would be handled by the main game logic
        pass

    def _load_template(self) -> None:
        """Load the selected template."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(
            self.templates
        ):
            template = self.templates[self.selected_idx]
            print(f"Loading template: {template['name']}")
            # This would be handled by the main game logic

    def _draw_template_entry(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        template: Dict[str, Any],
        is_selected: bool,
    ) -> None:
        """Draw a single template entry."""
        # Draw selection indicator
        if is_selected:
            draw_text(
                surface, ">", x - 2, y, size=font.get_height(), color=(255, 255, 100)
            )

        # Draw template name and details
        draw_text(
            surface, template["name"], x, y, size=font.get_height(), color=COLOR_TEXT
        )

        # Draw template info
        y += font.get_height()
        info = f"  {len(template['converters'])} converters, {len(template['connections'])} connections"
        draw_text(surface, info, x, y, size=font.get_height(), color=(160, 160, 160))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events.

        Returns:
            bool: True if the event was handled
        """
        if event.type == pygame.KEYDOWN and self.selected_idx is not None:
            if event.key == pygame.K_UP:
                self.selected_idx = max(0, self.selected_idx - 1)
                if self.selected_idx < self.scroll_offset:
                    self.scroll_offset = self.selected_idx
                return True
            if event.key == pygame.K_DOWN:
                max_idx = len(self.templates) - 1
                self.selected_idx = min(max_idx, self.selected_idx + 1)
                if self.selected_idx >= self.scroll_offset + self.max_visible_items:
                    self.scroll_offset = self.selected_idx - self.max_visible_items + 1
                return True
            if event.key == pygame.K_RETURN:
                self._load_template()
                return True

        # Handle button events
        return self.save_button.handle_event(event) or self.load_button.handle_event(
            event
        )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the template manager interface.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)

        # Calculate layout
        margin = font.get_height()
        content_x = self.rect.x + margin
        content_y = self.rect.y + margin * 2  # Extra margin for title

        # Draw template list
        list_y = content_y + margin

        if self.templates:
            # Draw visible templates
            template_height = font.get_height() * 2 + 1  # 2 lines + spacing
            end_idx = min(
                self.scroll_offset + self.max_visible_items, len(self.templates)
            )

            for i in range(self.scroll_offset, end_idx):
                template = self.templates[i]
                y = list_y + (i - self.scroll_offset) * template_height
                self._draw_template_entry(
                    surface,
                    font,
                    content_x + margin,
                    y,
                    template,
                    i == self.selected_idx,
                )
        else:
            draw_text(
                surface,
                "No templates available",
                content_x + margin,
                list_y,
                size=font.get_height(),
                color=COLOR_TEXT,
            )

        # Draw scroll indicators if needed
        if self.scroll_offset > 0:
            draw_text(
                surface,
                "^ More ^",
                self.rect.centerx - 4,
                self.rect.y + margin,
                size=font.get_height(),
                color=COLOR_TEXT,
            )
        if end_idx < len(self.templates):
            draw_text(
                surface,
                "v More v",
                self.rect.centerx - 4,
                self.rect.bottom - margin * 4,
                size=font.get_height(),
                color=COLOR_TEXT,
            )

        # Update and draw control buttons
        button_height = font.get_height() * 3
        self.save_button.y = self.rect.bottom - button_height
        self.load_button.y = self.rect.bottom - button_height
        self.save_button.draw(surface, font)
        self.load_button.draw(surface, font)

        return panel_rect
