import logging
from typing import Tuple, List, Dict, Optional, Any, Callable, TypeVar, cast
import pygame
import time
import math
import random
import itertools

from ui.draw_utils import draw_text, draw_panel
from ui.ascii_base import UIStyle, UIElement
from ui.ui_element.ascii_button import ASCIIButton
from ui.ui_element.ascii_panel import ASCIIPanel
from ui.ui_element.ascii_table import ASCIITable
from config import COLOR_TEXT, COLOR_BG

# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height


class ASCIITemplateListView:
    """Interface for displaying and managing a list of chain templates."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Template List",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a template list view interface.

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
        self.max_visible_items = 10

        # Create control buttons
        self.delete_button = ASCIIButton(
            rect.x + rect.width // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Delete Template",
            self._delete_template,
            style,
        )
        self.duplicate_button = ASCIIButton(
            rect.x + rect.width * 3 // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Duplicate Template",
            self._duplicate_template,
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

    def _delete_template(self) -> None:
        """Delete the selected template."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(
            self.templates
        ):
            template = self.templates[self.selected_idx]
            print(f"Deleting template: {template['name']}")
            # This would be handled by the main game logic

    def _duplicate_template(self) -> None:
        """Create a copy of the selected template."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(
            self.templates
        ):
            template = self.templates[self.selected_idx]
            print(f"Duplicating template: {template['name']}")
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

        # Draw template name
        draw_text(
            surface, template["name"], x, y, size=font.get_height(), color=COLOR_TEXT
        )

        # Draw template details
        y += font.get_height()
        details = f"  Created: {template.get('created_at', 'Unknown')}"
        draw_text(surface, details, x, y, size=font.get_height(), color=(160, 160, 160))

        # Draw template stats
        y += font.get_height()
        stats = f"  Converters: {len(template['converters'])}, Efficiency: {template.get('efficiency', 0.0):.1%}"
        draw_text(surface, stats, x, y, size=font.get_height(), color=(160, 160, 160))

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

        # Handle button events
        return self.delete_button.handle_event(
            event
        ) or self.duplicate_button.handle_event(event)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the template list interface.

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
            template_height = font.get_height() * 3 + 2  # 3 lines + spacing
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
        self.delete_button.y = self.rect.bottom - button_height
        self.duplicate_button.y = self.rect.bottom - button_height
        self.delete_button.draw(surface, font)
        self.duplicate_button.draw(surface, font)

        return panel_rect
