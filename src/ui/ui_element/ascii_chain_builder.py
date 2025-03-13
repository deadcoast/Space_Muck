

# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height

class ASCIIChainBuilder:
    """Interface for step-by-step creation of converter chains."""

# Standard library imports

# Third-party library imports

# Local application imports
from config import COLOR_TEXT
from typing import Tuple, List, Dict, Optional, Any, TypeVar
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.draw_utils import draw_text
from src.ui.ui_base.ascii_ui import ASCIIButton
from src.ui.ui_base.ascii_ui import ASCIIPanel
import pygame

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Chain Builder",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a chain builder interface.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style

        # Chain building state
        self.available_converters: List[Dict[str, Any]] = []
        self.selected_converters: List[Dict[str, Any]] = []
        self.connections: List[Tuple[int, int]] = []  # (from_idx, to_idx)
        self.selected_idx: Optional[int] = None
        self.connecting_from: Optional[int] = None

        # UI state
        self.scroll_offset = 0
        self.max_visible_items = 8
        self.mode = "select"  # 'select' or 'connect'

        # Create control buttons with initial positions
        self.mode_button = ASCIIButton(
            rect.x + rect.width // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Switch Mode",
            self._toggle_mode,
            style,
        )
        self.save_button = ASCIIButton(
            rect.x + rect.width * 3 // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Save Chain",
            self._save_chain,
            style,
        )

    def set_available_converters(self, converters: List[Dict[str, Any]]) -> None:
        """Set the list of available converters.

        Args:
            converters: List of converter info dictionaries
        """
        self.available_converters = converters
        self.selected_idx = 0 if converters else None
        self.scroll_offset = 0

    def _toggle_mode(self) -> None:
        """Toggle between selection and connection modes."""
        self.mode = "connect" if self.mode == "select" else "select"
        self.selected_idx = None
        self.connecting_from = None
        self.mode_button.text = (
            "Connect Mode" if self.mode == "select" else "Select Mode"
        )

    def _save_chain(self) -> None:
        """Save the current chain configuration."""
        if self.selected_converters and self.connections:
            print(f"Saving chain with {len(self.selected_converters)} converters")
            # This would be handled by the main game logic

    def _check_compatibility(
        self, from_conv: Dict[str, Any], to_conv: Dict[str, Any]
    ) -> bool:
        """Check if two converters can be connected.

        Args:
            from_conv: Source converter info
            to_conv: Target converter info

        Returns:
            bool: True if converters are compatible
        """
        # This is a simplified check - the actual implementation would need to
        # verify resource types, throughput compatibility, etc.
        outputs = from_conv.get("outputs", {})
        inputs = to_conv.get("inputs", {})
        return bool(set(outputs) & set(inputs))

    def _draw_converter_entry(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        converter: Dict[str, Any],
        is_selected: bool,
        is_connecting: bool,
    ) -> None:
        """Draw a single converter entry."""
        # Draw selection/connection indicator
        if is_selected:
            draw_text(
                surface, ">", x - 2, y, size=font.get_height(), color=(255, 255, 100)
            )
        elif is_connecting:
            draw_text(
                surface, "*", x - 2, y, size=font.get_height(), color=(100, 255, 100)
            )

        # Draw converter name and type
        draw_text(
            surface, converter["name"], x, y, size=font.get_height(), color=COLOR_TEXT
        )

        # Draw input/output summary
        y += font.get_height()
        inputs = " + ".join(
            f"{amt}{res}" for res, amt in converter.get("inputs", {}).items()
        )
        outputs = " + ".join(
            f"{amt}{res}" for res, amt in converter.get("outputs", {}).items()
        )
        draw_text(
            surface,
            f"  {inputs} -> {outputs}",
            x,
            y,
            size=font.get_height(),
            color=(160, 160, 160),
        )

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events.

        Returns:
            bool: True if the event was handled
        """
        # Handle keyboard events
        if (
            event.type == pygame.KEYDOWN
            and self.selected_idx is not None
            and self._handle_keydown_event(event)
        ):
            return True

        # Handle button events
        if self.mode_button.handle_event(event):
            return True

        # Check save button
        return bool(self.save_button.handle_event(event))

    def _handle_keydown_event(self, event: pygame.event.Event) -> bool:
        """Handle keyboard events.

        Args:
            event: The pygame event to handle

        Returns:
            bool: True if the event was handled
        """
        # Handle navigation keys
        if event.key in (pygame.K_UP, pygame.K_DOWN):
            return self._handle_navigation_key(event.key)

        # Handle selection key
        elif event.key == pygame.K_RETURN:
            return self._handle_selection_key()

        return False

    def _handle_navigation_key(self, key: int) -> bool:
        """Handle up/down navigation keys.

        Args:
            key: The key code (pygame.K_UP or pygame.K_DOWN)

        Returns:
            bool: True if the event was handled
        """
        if key == pygame.K_UP:
            self.selected_idx = max(0, self.selected_idx - 1)

            # Adjust scroll if needed
            if self.mode == "select" and self.selected_idx < self.scroll_offset:
                self.scroll_offset = self.selected_idx

            return True

        elif key == pygame.K_DOWN:
            # Get the appropriate list based on mode
            items = (
                self.available_converters
                if self.mode == "select"
                else self.selected_converters
            )
            max_idx = len(items) - 1

            self.selected_idx = min(max_idx, self.selected_idx + 1)

            # Adjust scroll if needed
            if (
                self.mode == "select"
                and self.selected_idx >= self.scroll_offset + self.max_visible_items
            ):
                self.scroll_offset = self.selected_idx - self.max_visible_items + 1

            return True

        return False

    def _handle_selection_key(self) -> bool:
        """Handle the selection key (Enter/Return).

        Returns:
            bool: True if an action was taken, False otherwise
        """
        if self.mode == "select":
            # Add selected converter to chain
            converter = self.available_converters[self.selected_idx]
            if converter not in self.selected_converters:
                self.selected_converters.append(converter)
                return True  # Action taken: added a converter
            return False  # No action taken: converter already in list

        # Connection mode
        if self.connecting_from is None:
            self.connecting_from = self.selected_idx
            return True  # Action taken: started connection
        else:
            connection_made = self._try_connect_converters()
            self.connecting_from = None
            return connection_made  # Return whether a connection was made

    def _try_connect_converters(self) -> None:
        """Try to connect the selected converters."""
        from_conv = self.selected_converters[self.connecting_from]
        to_conv = self.selected_converters[self.selected_idx]

        if self._check_compatibility(from_conv, to_conv):
            self.connections.append((self.connecting_from, self.selected_idx))

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the chain builder interface.

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

        # Draw mode indicator
        mode_text = "Mode: " + ("Connect" if self.mode == "connect" else "Select")
        draw_text(surface, font, mode_text, (content_x, content_y), COLOR_TEXT)

        # Draw converter list
        list_y = content_y + margin * 2

        converters = (
            self.selected_converters
            if self.mode == "connect"
            else self.available_converters
        )

        if not converters:
            draw_text(
                surface,
                font,
                (
                    "No converters available"
                    if self.mode == "select"
                    else "No converters in chain"
                ),
                (content_x + margin, list_y),
                COLOR_TEXT,
            )
        else:
            # Draw visible converters
            converter_height = font.get_height() * 2 + 1  # 2 lines + spacing
            end_idx = min(self.scroll_offset + self.max_visible_items, len(converters))

            for i in range(self.scroll_offset, end_idx):
                converter = converters[i]
                y = list_y + (i - self.scroll_offset) * converter_height
                self._draw_converter_entry(
                    surface,
                    font,
                    content_x + margin,
                    y,
                    converter,
                    i == self.selected_idx,
                    i == self.connecting_from,
                )

        # Draw connection preview in connect mode
        if self.mode == "connect" and self.connecting_from is not None:
            preview_y = list_y + converter_height * len(converters) + margin
            from_conv = self.selected_converters[self.connecting_from]
            draw_text(
                surface,
                font,
                f"Connecting from: {from_conv['name']}",
                (content_x + margin, preview_y),
                (100, 255, 100),
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
        if end_idx < len(converters):
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
        self.mode_button.y = self.rect.bottom - button_height
        self.save_button.y = self.rect.bottom - button_height
        self.mode_button.draw(surface, font)
        self.save_button.draw(surface, font)

        return panel_rect
