"""
ASCIIInventoryPanel: Display and manage player inventory with sorting and filtering.

This component provides a user interface for displaying the player's inventory items,
with options for sorting, filtering, and basic inventory management.
"""

# Standard library imports
import logging

# Local application imports
from typing import Dict, List, Tuple

import pygame

from config import COLOR_TEXT, COLOR_HIGHLIGHT
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_base.ascii_ui import ASCIIPanel
from src.ui.ui_element.ascii_button import ASCIIButton

# Third-party library imports


# Type definitions for better type checking
Color = Tuple[int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height


class ASCIIInventoryPanel:
    """Display and manage player inventory with sorting and filtering."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Inventory",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize an inventory panel.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style

        # Inventory data
        self.inventory: Dict[str, int] = {}  # item_id -> quantity
        self.cargo_capacity: int = 100  # Default cargo capacity
        self.used_capacity: int = 0  # Used cargo space

        # Display settings
        self.items_per_page = 10
        self.current_page = 0
        self.selected_item_index = -1
        self.sort_by = "name"  # Options: "name", "quantity", "type"
        self.sort_ascending = True
        self.filter_text = ""
        self.filter_type = "all"  # Options: "all", "resource", "equipment", "misc"

        # Item details
        self.item_details: Dict[str, Dict] = {}  # Cache for item details

        # Colors for different item types
        self.type_colors = {
            "resource": (100, 255, 100),  # Green
            "equipment": (100, 100, 255),  # Blue
            "rare": (255, 100, 255),  # Purple
            "quest": (255, 255, 100),  # Yellow
            "misc": (200, 200, 200),  # Light gray
        }

        # Buttons
        self.buttons = []
        self._create_buttons()

    def _create_buttons(self) -> None:
        """Create control buttons for the inventory panel."""
        try:
            button_width = 12
            button_height = 3
            margin = 2

            # Calculate button positions
            button_y = self.rect.bottom - button_height - margin

            # Sort button
            sort_button = ASCIIButton(
                self.rect.x + margin,
                button_y,
                button_width,
                button_height,
                self.style,
                "Sort",
                self._toggle_sort,
            )

            # Filter button
            filter_button = ASCIIButton(
                self.rect.x + margin + button_width + margin,
                button_y,
                button_width,
                button_height,
                self.style,
                "Filter",
                self._toggle_filter,
            )

            # Previous page button
            prev_button = ASCIIButton(
                self.rect.right - button_width * 2 - margin * 3,
                button_y,
                button_width,
                button_height,
                self.style,
                "Previous",
                self._prev_page,
            )

            # Next page button
            next_button = ASCIIButton(
                self.rect.right - button_width - margin,
                button_y,
                button_width,
                button_height,
                self.style,
                "Next",
                self._next_page,
            )

            self.buttons = [sort_button, filter_button, prev_button, next_button]
        except Exception as e:
            logging.error(f"Error creating inventory buttons: {e}")

    def update_inventory(
        self, inventory: Dict[str, int], cargo_capacity: int = None
    ) -> None:
        """Update the inventory data.

        Args:
            inventory: Dictionary of item IDs and quantities
            cargo_capacity: Optional cargo capacity value
        """
        try:
            self.inventory = inventory.copy()

            if cargo_capacity is not None:
                self.cargo_capacity = cargo_capacity

            # Calculate used capacity (assuming 1 unit per item for now)
            self.used_capacity = sum(inventory.values())

            # Reset page if inventory is now smaller
            max_pages = max(
                1,
                (len(self._get_filtered_items()) + self.items_per_page - 1)
                // self.items_per_page,
            )
            if self.current_page >= max_pages:
                self.current_page = max(0, max_pages - 1)
        except Exception as e:
            logging.error(f"Error updating inventory: {e}")

    def update_item_details(self, item_details: Dict[str, Dict]) -> None:
        """Update item details cache.

        Args:
            item_details: Dictionary of item details by item ID
        """
        try:
            self.item_details = item_details.copy()
        except Exception as e:
            logging.error(f"Error updating item details: {e}")

    def _get_filtered_items(self) -> List[Tuple[str, int]]:
        """Get filtered and sorted inventory items.

        Returns:
            List of (item_id, quantity) tuples
        """
        try:
            # Start with all items
            items = list(self.inventory.items())

            # Apply type filter
            if self.filter_type != "all":
                items = [
                    (item_id, qty)
                    for item_id, qty in items
                    if self._get_item_type(item_id) == self.filter_type
                ]

            # Apply text filter
            if self.filter_text:
                items = [
                    (item_id, qty)
                    for item_id, qty in items
                    if self.filter_text.lower() in item_id.lower()
                ]

            # Apply sorting
            if self.sort_by == "name":
                items.sort(key=lambda x: x[0], reverse=not self.sort_ascending)
            elif self.sort_by == "quantity":
                items.sort(key=lambda x: x[1], reverse=not self.sort_ascending)
            elif self.sort_by == "type":
                items.sort(
                    key=lambda x: self._get_item_type(x[0]),
                    reverse=not self.sort_ascending,
                )

            return items
        except Exception as e:
            logging.error(f"Error filtering inventory items: {e}")
            return []

    def _get_item_type(self, item_id: str) -> str:
        """Get the type of an item based on its ID or details.

        Args:
            item_id: Item identifier

        Returns:
            Item type string
        """
        # Check item details first
        if item_id in self.item_details and "type" in self.item_details[item_id]:
            return self.item_details[item_id]["type"]

        # Fallback to guessing based on item_id
        if "ore" in item_id or "material" in item_id or "gas" in item_id:
            return "resource"
        elif "weapon" in item_id or "shield" in item_id or "engine" in item_id:
            return "equipment"
        elif "rare" in item_id or "artifact" in item_id:
            return "rare"
        elif "quest" in item_id:
            return "quest"
        else:
            return "misc"

    def _get_item_color(self, item_id: str) -> Color:
        """Get the display color for an item.

        Args:
            item_id: Item identifier

        Returns:
            RGB color tuple
        """
        item_type = self._get_item_type(item_id)
        return self.type_colors.get(item_type, COLOR_TEXT)

    def _toggle_sort(self) -> None:
        """Toggle sorting method."""
        sort_options = ["name", "quantity", "type"]
        current_index = sort_options.index(self.sort_by)
        next_index = (current_index + 1) % len(sort_options)

        # If we're cycling back to the first option, toggle direction
        if next_index == 0:
            self.sort_ascending = not self.sort_ascending

        self.sort_by = sort_options[next_index]

    def _toggle_filter(self) -> None:
        """Toggle filter type."""
        filter_options = ["all", "resource", "equipment", "rare", "quest", "misc"]
        current_index = filter_options.index(self.filter_type)
        next_index = (current_index + 1) % len(filter_options)
        self.filter_type = filter_options[next_index]

        # Reset to first page when filter changes
        self.current_page = 0

    def _prev_page(self) -> None:
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1

    def _next_page(self) -> None:
        """Go to next page."""
        filtered_items = self._get_filtered_items()
        max_pages = max(
            1, (len(filtered_items) + self.items_per_page - 1) // self.items_per_page
        )

        if self.current_page < max_pages - 1:
            self.current_page += 1

    def handle_input(self, key: int) -> None:
        """Handle keyboard input.

        Args:
            key: Key code of the pressed key
        """
        try:
            # Handle navigation keys
            if key == pygame.K_UP:
                self.selected_item_index = max(0, self.selected_item_index - 1)
            elif key == pygame.K_DOWN:
                items_on_page = self._get_items_on_current_page()
                self.selected_item_index = min(
                    len(items_on_page) - 1, self.selected_item_index + 1
                )
            elif key == pygame.K_PAGEUP:
                self._prev_page()
                self.selected_item_index = 0
            elif key == pygame.K_PAGEDOWN:
                self._next_page()
                self.selected_item_index = 0

            # Pass input to buttons
            for button in self.buttons:
                button.handle_input(key)
        except Exception as e:
            logging.error(f"Error handling input: {e}")

    def handle_mouse_event(self, event_type: int, pos: Tuple[int, int]) -> None:
        """Handle mouse events.

        Args:
            event_type: Type of mouse event
            pos: Mouse position
        """
        try:
            # Check if any button was clicked
            for button in self.buttons:
                button.handle_mouse_event(event_type, pos)

            # Check if an inventory item was clicked
            if event_type == pygame.MOUSEBUTTONDOWN:
                # Calculate item area
                item_area_start_y = self.rect.y + 40  # Approximate start of items list
                item_height = 20  # Approximate height per item

                # Check if click is in item area
                if (
                    self.rect.x <= pos[0] <= self.rect.right
                    and item_area_start_y
                    <= pos[1]
                    <= item_area_start_y + self.items_per_page * item_height
                ):
                    # Calculate which item was clicked
                    item_index = (pos[1] - item_area_start_y) // item_height
                    items_on_page = self._get_items_on_current_page()

                    if 0 <= item_index < len(items_on_page):
                        self.selected_item_index = item_index
        except Exception as e:
            logging.error(f"Error handling mouse event: {e}")

    def _get_items_on_current_page(self) -> List[Tuple[str, int]]:
        """Get items on the current page.

        Returns:
            List of (item_id, quantity) tuples for the current page
        """
        filtered_items = self._get_filtered_items()
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page

        return filtered_items[start_idx:end_idx]

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the inventory panel.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        try:
            # Draw panel background and border
            panel_rect = self._draw_panel_background(surface, font)
            
            # Calculate layout
            margin = font.get_height() // 2
            content_x = self.rect.x + margin
            content_y = self.rect.y + margin * 3  # Extra margin for title
            content_width = self.rect.width - margin * 2

            # Draw components
            content_y = self._draw_panel_header(surface, font, content_x, content_y, content_width, margin)
            content_y = self._draw_inventory_content(surface, font, content_x, content_y, content_width, margin)
            self._draw_page_indicator(surface, font, margin)
            self._draw_buttons(surface, font)

            return panel_rect
        except Exception as e:
            logging.error(f"Error drawing inventory panel: {e}")
            return self.rect
            
    def _draw_panel_background(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the panel background and border.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            
        Returns:
            pygame.Rect: The panel rect
        """
        panel = ASCIIPanel(
            self.rect.x,
            self.rect.y,
            self.rect.width,
            self.rect.height,
            self.style,
            self.title,
        )
        return panel.draw(surface, font)
    
    def _draw_panel_header(self, surface: pygame.Surface, font: pygame.font.Font, 
                           x: int, y: int, width: int, margin: int) -> int:
        """Draw the capacity bar and sort/filter info.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: Starting x position
            y: Starting y position
            width: Content width
            margin: Margin size
            
        Returns:
            int: Updated y position after drawing
        """
        # Draw capacity indicator
        self._draw_capacity_bar(surface, font, x, y, width)
        y += font.get_height() + margin

        # Draw sort and filter info
        sort_text = f"Sort: {self.sort_by.capitalize()} ({'↑' if self.sort_ascending else '↓'})"
        filter_text = f"Filter: {self.filter_type.capitalize()}"
        info_text = f"{sort_text}  |  {filter_text}"

        surface.blit(
            font.render(info_text, True, COLOR_TEXT), (x, y)
        )
        y += font.get_height() + margin
        
        return y
    
    def _draw_inventory_content(self, surface: pygame.Surface, font: pygame.font.Font,
                              x: int, y: int, width: int, margin: int) -> int:
        """Draw the inventory items or empty message.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: Starting x position
            y: Starting y position
            width: Content width
            margin: Margin size
            
        Returns:
            int: Updated y position after drawing
        """
        if items := self._get_items_on_current_page():
            y = self._draw_column_headers(surface, font, x, y, width, margin)
            y = self._draw_item_list(surface, font, x, y, width, margin, items)
        else:
            y = self._draw_empty_message(surface, font, x, y, width, margin)

        return y
    
    def _draw_column_headers(self, surface: pygame.Surface, font: pygame.font.Font,
                           x: int, y: int, width: int, margin: int) -> int:
        """Draw the column headers for the inventory list.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: Starting x position
            y: Starting y position
            width: Content width
            margin: Margin size
            
        Returns:
            int: Updated y position after drawing
        """
        header_item = "Item"
        header_type = "Type"

        surface.blit(
            font.render(header_item, True, COLOR_HIGHLIGHT),
            (x, y),
        )

        type_x = x + width - font.size(header_type)[0] - margin * 4
        surface.blit(
            font.render(header_type, True, COLOR_HIGHLIGHT), (type_x, y)
        )

        header_qty = "Qty"
        qty_x = type_x - font.size(header_qty)[0] - margin * 4
        surface.blit(
            font.render(header_qty, True, COLOR_HIGHLIGHT), (qty_x, y)
        )

        y += font.get_height() + margin
        return y
    
    def _draw_item_list(self, surface: pygame.Surface, font: pygame.font.Font,
                      x: int, y: int, width: int, margin: int, items: list) -> int:
        """Draw the list of inventory items.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: Starting x position
            y: Starting y position
            width: Content width
            margin: Margin size
            items: List of inventory items to draw
            
        Returns:
            int: Updated y position after drawing
        """
        # Calculate column positions
        header_type = "Type"
        header_qty = "Qty"
        type_x = x + width - font.size(header_type)[0] - margin * 4
        qty_x = type_x - font.size(header_qty)[0] - margin * 4
        
        for i, (item_id, quantity) in enumerate(items):
            item_color = self._get_item_color(item_id)
            item_type = self._get_item_type(item_id)

            # Highlight selected item
            if i == self.selected_item_index:
                highlight_rect = pygame.Rect(
                    x - margin // 2,
                    y - margin // 2,
                    width + margin,
                    font.get_height() + margin,
                )
                pygame.draw.rect(surface, (50, 50, 70), highlight_rect)

            # Draw item name (truncate if too long)
            item_name = self._get_truncated_item_name(item_id, font, x, qty_x, margin)

            surface.blit(
                font.render(item_name, True, item_color), (x, y)
            )

            # Draw quantity
            qty_text = str(quantity)
            surface.blit(
                font.render(qty_text, True, item_color), (qty_x, y)
            )

            # Draw type
            type_text = item_type.capitalize()
            surface.blit(
                font.render(type_text, True, item_color), (type_x, y)
            )

            y += font.get_height() + margin
            
        return y
    
    def _get_truncated_item_name(self, item_id: str, font: pygame.font.Font, 
                               content_x: int, qty_x: int, margin: int) -> str:
        """Get the truncated item name that fits within the available width.
        
        Args:
            item_id: Original item name/ID
            font: Font used for rendering
            content_x: Starting x position
            qty_x: X position of quantity column
            margin: Margin size
            
        Returns:
            str: Truncated item name with ellipsis if needed
        """
        max_name_width = qty_x - content_x - margin
        item_name = item_id

        if font.size(item_name)[0] > max_name_width:
            # Truncate and add ellipsis
            while (
                font.size(f"{item_name}...")[0] > max_name_width
                and item_name != ""
            ):
                item_name = item_name[:-1]
            item_name += "..."

        return item_name
    
    def _draw_empty_message(self, surface: pygame.Surface, font: pygame.font.Font,
                          x: int, y: int, width: int, margin: int) -> int:
        """Draw message when inventory is empty.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: Starting x position
            y: Starting y position
            width: Content width
            margin: Margin size
            
        Returns:
            int: Updated y position after drawing
        """
        empty_text = "No items found"
        if self.filter_type != "all" or self.filter_text:
            empty_text += " with current filters"

        surface.blit(
            font.render(empty_text, True, COLOR_TEXT),
            (
                x + width // 2 - font.size(empty_text)[0] // 2,
                y + margin * 3,
            ),
        )
        
        return y + margin * 3 + font.get_height()
    
    def _draw_page_indicator(self, surface: pygame.Surface, font: pygame.font.Font, margin: int) -> None:
        """Draw the page indicator at the bottom of the panel.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            margin: Margin size
        """
        items = self._get_filtered_items()
        max_pages = max(
            1, (len(items) + self.items_per_page - 1) // self.items_per_page
        )
        page_text = f"Page {self.current_page + 1}/{max_pages}"

        page_text_width = font.size(page_text)[0]
        surface.blit(
            font.render(page_text, True, COLOR_TEXT),
            (
                self.rect.x + self.rect.width // 2 - page_text_width // 2,
                self.rect.bottom - font.get_height() * 2 - margin,
            ),
        )
    
    def _draw_buttons(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the buttons at the bottom of the panel.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        for button in self.buttons:
            button.draw(surface, font)

    def _draw_capacity_bar(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
    ) -> None:
        """Draw cargo capacity bar.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            width: Width of the bar
        """
        try:
            # Calculate percentage and bar width
            percentage = (
                min(100.0, (self.used_capacity / self.cargo_capacity) * 100.0)
                if self.cargo_capacity > 0
                else 0
            )
            bar_width = int((width - 20) * (percentage / 100.0))

            # Determine color based on capacity usage
            if percentage < 70:
                color = (100, 255, 100)  # Green
            elif percentage < 90:
                color = (255, 255, 100)  # Yellow
            else:
                color = (255, 100, 100)  # Red

            # Draw label
            label = "Cargo: "
            label_width = font.size(label)[0]
            surface.blit(font.render(label, True, COLOR_TEXT), (x, y))

            # Draw bar background
            bar_bg_rect = pygame.Rect(
                x + label_width, y, width - label_width - 80, font.get_height()
            )
            pygame.draw.rect(surface, (50, 50, 50), bar_bg_rect)

            # Draw bar fill
            if bar_width > 0:
                bar_fill_rect = pygame.Rect(
                    x + label_width, y, bar_width, font.get_height()
                )
                pygame.draw.rect(surface, color, bar_fill_rect)

            # Draw capacity text
            capacity_text = f"{self.used_capacity}/{self.cargo_capacity} units"
            capacity_pos = (x + width - font.size(capacity_text)[0], y)
            surface.blit(font.render(capacity_text, True, color), capacity_pos)
        except Exception as e:
            logging.error(f"Error drawing capacity bar: {e}")
