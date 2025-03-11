import logging
import pygame
from typing import List, Dict, Optional, Tuple, Callable

# Import base classes from ascii_base.py
from ui.ui_base.ascii_base import UIElement, UIStyle, COLOR_TEXT, COLOR_HIGHLIGHT


class ASCIIBox(UIElement):
    """A simple box UI element with customizable borders and content."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        style: UIStyle,
        title: Optional[str] = None,
        content: Optional[str] = None,
    ):
        """Initialize an ASCII box.

        Args:
            x: X coordinate of the top-left corner
            y: Y coordinate of the top-left corner
            width: Width of the box in characters
            height: Height of the box in characters
            style: Visual style for the box
            title: Optional title for the box
            content: Optional content text for the box
        """
        super().__init__(x, y, width, height, style, title)
        self.content = content
        self.content_lines = []
        self.scroll_offset = 0
        self.max_scroll = 0

        # Parse content into lines if provided
        if content:
            self.set_content(content)

    def set_content(self, content: str) -> None:
        """Set the content text for the box.

        Args:
            content: Content text to display in the box
        """
        try:
            self.content = content

            # Split content into lines and handle word wrapping
            self.content_lines = []
            for line in content.split("\n"):
                # Account for box borders (width - 2)
                max_line_width = self.width - 2

                if len(line) <= max_line_width:
                    self.content_lines.append(line)
                else:
                    # Simple word wrapping
                    words = line.split()
                    current_line = ""

                    for word in words:
                        if len(current_line) + len(word) + 1 <= max_line_width:
                            if current_line:
                                current_line += f" {word}"
                            else:
                                current_line = word
                        else:
                            self.content_lines.append(current_line)
                            current_line = word

                    if current_line:
                        self.content_lines.append(current_line)

            # Calculate max scroll offset
            max_visible_lines = self.height - 2  # Account for box borders
            self.max_scroll = max(0, len(self.content_lines) - max_visible_lines)

            # Reset scroll offset if needed
            self.scroll_offset = min(self.scroll_offset, self.max_scroll)
        except Exception as e:
            logging.error(f"Error setting content: {e}")

    def scroll_up(self, lines: int = 1) -> None:
        """Scroll content up by the specified number of lines.

        Args:
            lines: Number of lines to scroll up
        """
        self.scroll_offset = max(0, self.scroll_offset - lines)

    def scroll_down(self, lines: int = 1) -> None:
        """Scroll content down by the specified number of lines.

        Args:
            lines: Number of lines to scroll down
        """
        self.scroll_offset = min(self.max_scroll, self.scroll_offset + lines)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the ASCII box.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        # Draw base UI element (border and background)
        super().draw(surface, font)

        if not self.visible:
            return

        try:
            # Draw content if available
            if self.content_lines:
                self._draw_content(surface, font)

            # Draw scroll indicators if needed
            if self.max_scroll > 0:
                self._draw_scroll_indicators(surface, font)

        except Exception as e:
            logging.error(f"Error drawing ASCII box: {e}")

    def _draw_content(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the content text.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        try:
            # Calculate visible content area
            content_start_x = self.x + 1
            content_start_y = self.y + 1
            visible_lines = min(self.height - 2, len(self.content_lines))

            # Draw visible lines with scroll offset
            for i in range(visible_lines):
                line_index = i + self.scroll_offset
                if 0 <= line_index < len(self.content_lines):
                    line = self.content_lines[line_index]
                    self._draw_text(
                        surface, font, content_start_x, content_start_y + i, line
                    )

        except Exception as e:
            logging.error(f"Error drawing content: {e}")

    def _draw_scroll_indicators(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Draw scroll indicators if content is scrollable.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        try:
            # Draw up indicator if scrolled down
            if self.scroll_offset > 0:
                self._draw_char(surface, font, self.x + self.width - 1, self.y + 1, "▲")

            # Draw down indicator if can scroll further
            if self.scroll_offset < self.max_scroll:
                self._draw_char(
                    surface,
                    font,
                    self.x + self.width - 1,
                    self.y + self.height - 2,
                    "▼",
                )

        except Exception as e:
            logging.error(f"Error drawing scroll indicators: {e}")

    def handle_input(self, key: int) -> Optional[str]:
        """Handle input for the ASCII box.

        Args:
            key: Key code of the pressed key

        Returns:
            Optional result string based on the input
        """
        try:
            # Handle scrolling
            if key == pygame.K_UP:
                self.scroll_up()
                return "scroll_up"
            elif key == pygame.K_DOWN:
                self.scroll_down()
                return "scroll_down"
            elif key == pygame.K_PAGEUP:
                self.scroll_up(self.height - 2)
                return "page_up"
            elif key == pygame.K_PAGEDOWN:
                self.scroll_down(self.height - 2)
                return "page_down"
            elif key == pygame.K_HOME:
                self.scroll_offset = 0
                return "scroll_home"
            elif key == pygame.K_END:
                self.scroll_offset = self.max_scroll
                return "scroll_end"

        except Exception as e:
            logging.error(f"Error handling input: {e}")

        return None


class ASCIIPanel(UIElement):
    """A panel UI element that can contain other UI elements."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        style: UIStyle,
        title: Optional[str] = None,
    ):
        """Initialize an ASCII panel.

        Args:
            x: X coordinate of the top-left corner
            y: Y coordinate of the top-left corner
            width: Width of the panel in characters
            height: Height of the panel in characters
            style: Visual style for the panel
            title: Optional title for the panel
        """
        super().__init__(x, y, width, height, style, title)
        self.children: List[UIElement] = []
        self.active_child_index = -1

    def add_child(self, child: UIElement) -> None:
        """Add a child UI element to the panel.

        Args:
            child: UI element to add as a child
        """
        try:
            # Adjust child coordinates to be relative to panel
            child.x += self.x
            child.y += self.y

            # Ensure child fits within panel bounds
            if (
                child.x + child.width > self.x + self.width
                or child.y + child.height > self.y + self.height
            ):
                logging.warning("Child element exceeds panel bounds")

            self.children.append(child)

        except Exception as e:
            logging.error(f"Error adding child: {e}")

    def remove_child(self, child: UIElement) -> bool:
        """Remove a child UI element from the panel.

        Args:
            child: UI element to remove

        Returns:
            True if the child was removed, False otherwise
        """
        try:
            if child in self.children:
                self.children.remove(child)

                # Update active child index if needed
                if self.active_child_index >= len(self.children):
                    self.active_child_index = len(self.children) - 1

                return True

        except Exception as e:
            logging.error(f"Error removing child: {e}")

        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the panel and its children.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        # Draw base UI element (border and background)
        super().draw(surface, font)

        if not self.visible:
            return

        try:
            # Draw children
            for child in self.children:
                child.draw(surface, font)

        except Exception as e:
            logging.error(f"Error drawing ASCII panel: {e}")

    def handle_input(self, key: int) -> Optional[str]:
        """Handle input for the panel and its children.

        Args:
            key: Key code of the pressed key

        Returns:
            Optional result string based on the input
        """
        try:
            # If there's an active child, pass input to it first
            if 0 <= self.active_child_index < len(self.children):
                if result := self.children[self.active_child_index].handle_input(key):
                    return result

            if key == pygame.K_TAB:
                if self.children:
                    self.active_child_index = (self.active_child_index + 1) % len(
                        self.children
                    )
                    return "next_child"

                if pygame.key.get_mods() & pygame.KMOD_SHIFT and self.children:
                    self.active_child_index = (self.active_child_index - 1) % len(
                        self.children
                    )
                    return "prev_child"

        except Exception as e:
            logging.error(f"Error handling input: {e}")

        return None


class ASCIIButton(UIElement):
    """A clickable button UI element."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        style: UIStyle,
        text: str,
        callback: Optional[Callable[[], object]] = None,
    ):
        """Initialize an ASCII button.

        Args:
            x: X coordinate of the top-left corner
            y: Y coordinate of the top-left corner
            width: Width of the button in characters
            height: Height of the button in characters
            style: Visual style for the button
            text: Text to display on the button
            callback: Optional function to call when the button is clicked
        """
        super().__init__(x, y, width, height, style)
        self.text = text
        self.callback = callback
        self.pressed = False
        self.focused = False

        # Center text if it's shorter than the button width
        self.centered_text = text
        if len(text) < width - 2:
            padding = (width - 2 - len(text)) // 2
            self.centered_text = " " * padding + text + " " * padding

            # Ensure text doesn't exceed button width
            if len(self.centered_text) > width - 2:
                self.centered_text = self.centered_text[: width - 2]

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the button.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        # Draw base UI element (border and background)
        super().draw(surface, font)

        if not self.visible:
            return

        try:
            self._render_button_text(surface, font)
        except Exception as e:
            logging.error(f"Error drawing ASCII button: {e}")

    def _render_button_text(self, surface, font):
        # Determine text color based on button state
        text_color = COLOR_TEXT
        if self.pressed:
            text_color = COLOR_HIGHLIGHT
        elif self.focused or self.hover:
            # Slightly brighter than normal
            r, g, b = COLOR_TEXT
            text_color = (
                min(255, int(r * 1.2)),
                min(255, int(g * 1.2)),
                min(255, int(b * 1.2)),
            )

        # Draw button text
        text_x = self.x + 1
        text_y = self.y + (self.height // 2)
        self._draw_text(surface, font, text_x, text_y, self.centered_text, text_color)

        # Draw focus indicator if focused
        if self.focused:
            self._draw_focus_indicator(surface, font)

    def _draw_focus_indicator(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Draw a focus indicator around the button.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        try:
            # Draw small indicators at corners
            self._draw_char(surface, font, self.x, self.y, "◆", COLOR_HIGHLIGHT)
            self._draw_char(
                surface, font, self.x + self.width - 1, self.y, "◆", COLOR_HIGHLIGHT
            )
            self._draw_char(
                surface, font, self.x, self.y + self.height - 1, "◆", COLOR_HIGHLIGHT
            )
            self._draw_char(
                surface,
                font,
                self.x + self.width - 1,
                self.y + self.height - 1,
                "◆",
                COLOR_HIGHLIGHT,
            )

        except Exception as e:
            logging.error(f"Error drawing focus indicator: {e}")

    def handle_input(self, key: int) -> Optional[str]:
        """Handle input for the button.

        Args:
            key: Key code of the pressed key

        Returns:
            Optional result string based on the input
        """
        try:
            # Handle enter/space to activate button
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.pressed = True

                # Call callback if provided
                if self.callback:
                    self.callback()

                # Schedule button to be released after a short delay
                self.pressed = False

                return self.text

        except Exception as e:
            logging.error(f"Error handling input: {e}")

        return None

    def handle_mouse_event(
        self, event_type: int, pos: Tuple[int, int]
    ) -> Optional[str]:
        """Handle mouse events for the button.

        Args:
            event_type: Type of mouse event (e.g., pygame.MOUSEBUTTONDOWN)
            pos: Mouse position in character coordinates

        Returns:
            Optional result string based on the input
        """
        try:
            x, y = pos

            # Check if mouse is over button
            if self.contains_point(x, y):
                if event_type == pygame.MOUSEMOTION:
                    self.hover = True
                elif event_type == pygame.MOUSEBUTTONDOWN:
                    self.pressed = True
                elif event_type == pygame.MOUSEBUTTONUP and self.pressed:
                    self.pressed = False

                    # Call callback if provided
                    if self.callback:
                        self.callback()

                    return self.text
            else:
                self.hover = False
                if event_type == pygame.MOUSEBUTTONUP:
                    self.pressed = False

        except Exception as e:
            logging.error(f"Error handling mouse event: {e}")

        return None


class ASCIIProgressBar(UIElement):
    """A progress bar UI element."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        style: UIStyle,
        progress: float = 0.0,
        title: Optional[str] = None,
        show_percentage: bool = True,
    ):
        """Initialize an ASCII progress bar.

        Args:
            x: X coordinate of the top-left corner
            y: Y coordinate of the top-left corner
            width: Width of the progress bar in characters
            height: Height of the progress bar in characters
            style: Visual style for the progress bar
            progress: Initial progress value (0.0 to 1.0)
            title: Optional title for the progress bar
            show_percentage: Whether to show percentage text
        """
        super().__init__(x, y, width, height, style, title)
        self.progress = max(0.0, min(1.0, progress))  # Clamp to valid range
        self.target_progress = self.progress
        self.show_percentage = show_percentage
        self.animation_speed = 0.05  # Progress units per frame
        self.fill_chars = self._get_fill_chars()

    def _get_fill_chars(self) -> Dict[str, object]:
        """Get fill characters based on the UI style.

        Returns:
            Dictionary of fill characters for the progress bar
        """
        try:
            if self.style == UIStyle.SYMBIOTIC:
                return {
                    "empty": "·",
                    "partial": ["░", "▒", "▓"],
                    "full": "█",
                }
            elif self.style == UIStyle.ASTEROID:
                return {
                    "empty": "·",
                    "partial": ["▪", "◆", "◈"],
                    "full": "■",
                }
            elif self.style == UIStyle.MECHANICAL:
                return {
                    "empty": "·",
                    "partial": ["▫", "▪", "◻"],
                    "full": "◼",
                }
            elif self.style == UIStyle.QUANTUM:
                return {
                    "empty": "·",
                    "partial": ["∴", "∷", "⋮"],
                    "full": "⋯",
                }
            elif self.style == UIStyle.FLEET:
                return {
                    "empty": "·",
                    "partial": ["◌", "◍", "◉"],
                    "full": "●",
                }
            else:
                return {
                    "empty": "·",
                    "partial": ["░", "▒", "▓"],
                    "full": "█",
                }

        except Exception as e:
            logging.error(f"Error getting fill chars: {e}")
            return {
                "empty": "·",
                "partial": ["░", "▒", "▓"],
                "full": "█",
            }

    def set_progress(self, progress: float, animate: bool = True) -> None:
        """Set the progress value.

        Args:
            progress: New progress value (0.0 to 1.0)
            animate: Whether to animate the progress change
        """
        try:
            # Clamp progress to valid range
            progress = max(0.0, min(1.0, progress))

            if animate:
                # Set target for animation
                self.target_progress = progress

                # Start animation if not already active
                if not self.animation["active"]:
                    self.start_animation(0.5)
            else:
                # Set progress immediately
                self.progress = progress
                self.target_progress = progress

        except Exception as e:
            logging.error(f"Error setting progress: {e}")

    def update_animation(self, delta_time: float = 0.0) -> bool:
        """Update animation state based on elapsed time.

        Args:
            delta_time: Time elapsed since last update (if 0, will calculate internally)

        Returns:
            True if animation is still active, False if complete
        """
        # Update base animation
        still_active = super().update_animation(delta_time)

        try:
            # Update progress animation
            if self.progress != self.target_progress:
                # Calculate step based on animation speed
                step = self.animation_speed

                # Move progress toward target
                if self.progress < self.target_progress:
                    self.progress = min(self.target_progress, self.progress + step)
                else:
                    self.progress = max(self.target_progress, self.progress - step)

            # Keep animation active if still animating progress
            if self.progress != self.target_progress:
                return True

        except Exception as e:
            logging.error(f"Error updating progress animation: {e}")

        return still_active

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the progress bar.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        # Draw base UI element (border and background)
        super().draw(surface, font)

        if not self.visible:
            return

        try:
            # Draw progress bar
            self._draw_progress_bar(surface, font)

            # Draw percentage text if enabled
            if self.show_percentage:
                self._draw_percentage(surface, font)

        except Exception as e:
            logging.error(f"Error drawing ASCII progress bar: {e}")

    def _draw_progress_bar(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Draw the progress bar fill.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        try:
            # Calculate bar dimensions
            bar_width = self.width - 2  # Account for borders
            bar_x = self.x + 1
            bar_y = self.y + (self.height // 2)

            # Calculate filled width
            filled_width = int(bar_width * self.progress)

            # Get fill characters
            empty_char = self.fill_chars["empty"]
            full_char = self.fill_chars["full"]
            partial_chars = self.fill_chars["partial"]

            # Draw filled portion
            for i in range(filled_width):
                self._draw_char(
                    surface,
                    font,
                    bar_x + i,
                    bar_y,
                    full_char,
                    self._get_progress_color(i / bar_width),
                )

            # Draw partial character at the edge if needed
            if filled_width < bar_width:
                # Calculate partial fill (0.0 to 1.0)
                partial = (self.progress * bar_width) - filled_width

                # Choose partial character based on fill amount
                partial_index = min(
                    len(partial_chars) - 1, int(partial * len(partial_chars))
                )
                partial_char = partial_chars[partial_index]

                self._draw_char(
                    surface,
                    font,
                    bar_x + filled_width,
                    bar_y,
                    partial_char,
                    self._get_progress_color(filled_width / bar_width),
                )

                # Draw empty portion
                for i in range(filled_width + 1, bar_width):
                    self._draw_char(surface, font, bar_x + i, bar_y, empty_char)

        except Exception as e:
            logging.error(f"Error drawing progress bar fill: {e}")

    def _draw_percentage(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the percentage text.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        try:
            # Format percentage text
            percentage = int(self.progress * 100)
            text = f"{percentage}%"

            # Calculate text position (centered)
            text_x = self.x + (self.width // 2) - (len(text) // 2)
            text_y = self.y + self.height - 1

            # Draw text
            self._draw_text(surface, font, text_x, text_y, text)

        except Exception as e:
            logging.error(f"Error drawing percentage text: {e}")

    def _get_progress_color(self, position: float) -> Tuple[int, int, int]:
        """Get color for progress bar based on position and style.

        Args:
            position: Position along the progress bar (0.0 to 1.0)

        Returns:
            RGB color tuple for the progress bar at the specified position
        """
        try:
            base_color = COLOR_TEXT
            r, g, b = base_color

            # Adjust color based on style
            if self.style == UIStyle.SYMBIOTIC:
                # Symbiotic style: green gradient
                g = min(255, int(g * (0.8 + 0.4 * position)))
                return (int(r * 0.8), g, int(b * 0.8))
            elif self.style == UIStyle.ASTEROID:
                # Asteroid style: orange/red gradient
                r = min(255, int(r * (0.8 + 0.4 * position)))
                return (r, int(g * 0.9), int(b * 0.7))
            elif self.style == UIStyle.MECHANICAL:
                # Mechanical style: blue gradient
                b = min(255, int(b * (0.8 + 0.4 * position)))
                return (int(r * 0.9), int(g * 0.9), b)
            elif self.style == UIStyle.QUANTUM:
                # Quantum style: purple gradient
                r = min(255, int(r * (0.8 + 0.3 * position)))
                b = min(255, int(b * (0.8 + 0.3 * position)))
                return (r, int(g * 0.8), b)
            elif self.style == UIStyle.FLEET:
                # Fleet style: cyan gradient
                g = min(255, int(g * (0.8 + 0.3 * position)))
                b = min(255, int(b * (0.8 + 0.3 * position)))
                return (int(r * 0.8), g, b)
            else:
                # Default: white gradient
                factor = 0.8 + 0.4 * position
                return (
                    min(255, int(r * factor)),
                    min(255, int(g * factor)),
                    min(255, int(b * factor)),
                )

        except Exception as e:
            logging.error(f"Error getting progress color: {e}")
            return COLOR_TEXT
