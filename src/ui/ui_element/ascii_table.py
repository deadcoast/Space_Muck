"""
ascii_table.py

Provides a specialized table for displaying ASCII-style tables.
"""

# Standard library imports
import logging
import time
from typing import Any, List, Optional, Tuple, TypeVar

import pygame

# Local application imports
from config import COLOR_TEXT
from src.ui.draw_utils import draw_text
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_base.ascii_ui import ASCIIPanel

# Third-party library imports


# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height


class ASCIITable:
    def __init__(
        self,
        rect: pygame.Rect,
        headers: List[str],
        data: List[List[Any]],
        title: Optional[str] = None,
        style: UIStyle = UIStyle.MECHANICAL,
        converter_type: Optional[str] = None,
        selectable: bool = True,
        sortable: bool = True,
    ):
        """Initialize an ASCII table.

        Args:
            rect: Rectangle defining position and size
            headers: List of column headers
            data: 2D list of data rows
            title: Optional title to display at the top
            style: Visual style for the table
            converter_type: Optional converter type to determine style
            selectable: Whether rows can be selected
            sortable: Whether columns can be sorted
        """

        self.rect = rect
        self.headers = headers
        self.data = data
        self.title = title
        self.style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )
        self.selectable = selectable
        self.sortable = sortable

        # Table state
        self.selected_row = -1  # -1 means no selection
        self.sort_column = -1  # -1 means no sorting
        self.sort_ascending = True
        self.scroll_offset = 0
        self.column_widths: List[int] = []
        self.visible_rows = 0  # Will be calculated during drawing

        # Style-based characters
        style_chars = {
            UIStyle.QUANTUM: {
                "header_sep": "╌",
                "row_sep": "·",
                "col_sep": "┆",
                "scroll_up": "▲",
                "scroll_down": "▼",
                "sort_asc": "△",
                "sort_desc": "▽",
                "selected_marker": "►",
            },
            UIStyle.SYMBIOTIC: {
                "header_sep": "─",
                "row_sep": "·",
                "col_sep": "│",
                "scroll_up": "↑",
                "scroll_down": "↓",
                "sort_asc": "↑",
                "sort_desc": "↓",
                "selected_marker": ">",
            },
            UIStyle.MECHANICAL: {
                "header_sep": "═",
                "row_sep": "─",
                "col_sep": "│",
                "scroll_up": "▲",
                "scroll_down": "▼",
                "sort_asc": "▲",
                "sort_desc": "▼",
                "selected_marker": ">",
            },
            UIStyle.ASTEROID: {
                "header_sep": "═",
                "row_sep": "─",
                "col_sep": "┃",
                "scroll_up": "▲",
                "scroll_down": "▼",
                "sort_asc": "▲",
                "sort_desc": "▼",
                "selected_marker": ">",
            },
            UIStyle.FLEET: {
                "header_sep": "━",
                "row_sep": "─",
                "col_sep": "┃",
                "scroll_up": "▲",
                "scroll_down": "▼",
                "sort_asc": "▲",
                "sort_desc": "▼",
                "selected_marker": "►",
            },
        }
        self.chars = style_chars.get(self.style, style_chars[UIStyle.MECHANICAL])

        # Animation state
        self.animation = {
            "active": False,
            "progress": 0.0,
            "start_time": 0,
            "duration": 0.3,
            "type": "none",  # none, sort, select, scroll
        }

        # Calculate initial column widths
        self._calculate_column_widths()

    def _calculate_column_widths(self) -> None:
        """Calculate the width of each column based on content."""
        if not self.headers or not self.data:
            self.column_widths = []
            return

        # Initialize with header widths
        self.column_widths = [len(str(h)) for h in self.headers]

        # Update with data widths
        for row in self.data:
            for i, cell in enumerate(row[: len(self.column_widths)]):
                self.column_widths[i] = max(self.column_widths[i], len(str(cell)))

        # Add padding
        self.column_widths = [w + 2 for w in self.column_widths]

    def set_data(self, data: List[List[Any]]) -> None:
        """Set the table data.

        Args:
            data: 2D list of data rows
        """
        self.data = data
        self.selected_row = -1
        self.scroll_offset = 0
        self._calculate_column_widths()
        self._start_animation("data")

    def set_headers(self, headers: List[str]) -> None:
        """Set the table headers.

        Args:
            headers: List of column headers
        """
        self.headers = headers
        self._calculate_column_widths()

    def sort_by_column(self, column_index: int) -> None:
        """Sort the table data by the specified column.

        Args:
            column_index: Index of the column to sort by
        """
        if not self.sortable or column_index < 0 or column_index >= len(self.headers):
            return

        # Toggle sort direction if already sorting by this column
        if self.sort_column == column_index:
            self.sort_ascending = not self.sort_ascending
        else:
            self.sort_column = column_index
            self.sort_ascending = True

        # Start sort animation
        self._start_animation("sort")

        # Sort the data - this will be handled by the animation completion
        # in _update_animation to provide visual feedback during sorting

    def select_row(self, row_index: int) -> None:
        """Select a row in the table.

        Args:
            row_index: Index of the row to select
        """
        if not self.selectable:
            return

        if 0 <= row_index < len(self.data):
            self.selected_row = row_index
            # Ensure the selected row is visible
            if self.selected_row < self.scroll_offset:
                self.scroll_offset = self.selected_row
            elif (
                self.visible_rows > 0
                and self.selected_row >= self.scroll_offset + self.visible_rows
            ):
                self.scroll_offset = max(0, self.selected_row - self.visible_rows + 1)
            self._start_animation("select")

    def get_selected_row_data(self) -> Optional[List[Any]]:
        """Get the data for the currently selected row.

        Returns:
            The selected row data or None if no row is selected
        """
        if self.selected_row >= 0 and self.selected_row < len(self.data):
            return self.data[self.selected_row]
        return None

    def scroll(self, amount: int) -> None:
        """Scroll the table by the specified amount.

        Args:
            amount: Number of rows to scroll (positive for down, negative for up)
        """
        max_scroll = max(0, len(self.data) - self.visible_rows)
        new_offset = max(0, min(max_scroll, self.scroll_offset + amount))

        if new_offset != self.scroll_offset:
            self.scroll_offset = new_offset
            self._start_animation("scroll")

    def _start_animation(self, anim_type: str) -> None:
        """Start an animation sequence.

        Args:
            anim_type: Type of animation ('sort', 'select', 'scroll', 'data')
        """
        self.animation["active"] = True
        self.animation["progress"] = 0.0
        self.animation["start_time"] = time.time()
        self.animation["type"] = anim_type

        # Adjust duration based on animation type
        durations = {
            "sort": 0.3,
            "select": 0.2,
            "scroll": 0.15,
            "data": 0.4,
            "none": 0.3,
        }
        self.animation["duration"] = durations.get(anim_type, 0.3)

    @staticmethod
    def _safe_sort_key(row: List[Any], column_index: int) -> Any:
        """Get a safe sort key for a row at the specified column index.

        Args:
            row: The data row
            column_index: The column index to get the sort key for

        Returns:
            A sort key that can be safely compared
        """
        if column_index < 0 or column_index >= len(row):
            return ""

        value = row[column_index]

        # Handle different types for sorting
        if value is None:
            return ""
        elif isinstance(value, (int, float)):
            return value
        elif isinstance(value, bool):
            return int(value)
        else:
            # Convert to string for safe comparison
            try:
                return str(value).lower()
            except Exception:
                return ""

    def _update_animation(self, dt: Optional[float] = None) -> None:
        """Update animation state based on elapsed time.

        Args:
            dt: Optional time delta in seconds. If None, will calculate based on current time.
        """
        if not self.animation["active"]:
            return

        if dt is None:
            current_time = time.time()
            dt = current_time - self.animation["start_time"]
            self.animation["start_time"] = current_time

        self.animation["progress"] = min(
            1.0, self.animation["progress"] + dt / self.animation["duration"]
        )

        if self.animation["progress"] >= 1.0:
            self.animation["active"] = False

            # Apply final sort if this was a sort animation
            if self.animation["type"] == "sort" and self.sort_column >= 0:
                try:
                    # Sort the data using a stable sort
                    self.data.sort(
                        key=lambda row: self._safe_sort_key(row, self.sort_column),
                        reverse=not self.sort_ascending,
                    )
                except (TypeError, IndexError) as e:
                    # Handle sorting errors gracefully
                    logging.warning(f"Error sorting table data: {e}")
                    self.sort_column = -1

    def _handle_header_click(self, event: pygame.event.Event) -> bool:
        """Handle clicks on column headers for sorting.

        Args:
            event: The pygame event

        Returns:
            bool: True if the event was handled
        """
        if not (self.sortable and event.button == 1):
            return False

        header_height = 30  # Approximate height of header
        if not (self.rect.y <= event.pos[1] <= self.rect.y + header_height):
            return False

        # Determine which column was clicked
        x = self.rect.x + 10  # Starting x position with margin
        for i, width in enumerate(self.column_widths):
            if x <= event.pos[0] <= x + width:
                self.sort_by_column(i)
                return True
            x += width + 5  # Add spacing between columns

        return False

    def _handle_row_selection(self, event: pygame.event.Event) -> bool:
        """Handle row selection clicks.

        Args:
            event: The pygame event

        Returns:
            bool: True if the event was handled
        """
        if not (self.selectable and event.button == 1):
            return False

        row_height = 20  # Approximate height of each row
        header_offset = 40  # Space for headers
        y = self.rect.y + header_offset

        # Calculate which row was clicked
        row_idx = self.scroll_offset + (event.pos[1] - y) // row_height
        if y <= event.pos[
            1
        ] < y + row_height * self.visible_rows and 0 <= row_idx < len(self.data):
            self.select_row(row_idx)
            return True

        return False

    def _handle_scrolling(self, event: pygame.event.Event) -> bool:
        """Handle scrolling events.

        Args:
            event: The pygame event

        Returns:
            bool: True if the event was handled
        """
        if event.button == 4:  # Scroll up
            self.scroll(-1)
            return True
        elif event.button == 5:  # Scroll down
            self.scroll(1)
            return True

        return False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events for this table.

        Returns:
            bool: True if the event was consumed
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Try to handle the event with each handler in order
            if self._handle_header_click(event):
                return True

            if self._handle_row_selection(event):
                return True

            if self._handle_scrolling(event):
                return True

        return False

    def _draw_headers(
        self, surface: pygame.Surface, font: pygame.font.Font, x: int, y: int
    ) -> int:
        """Draw the table headers.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x, y: Position to start drawing

        Returns:
            int: Y position after drawing headers
        """
        header_height = font.get_height() + 10
        current_x = x

        for i, header in enumerate(self.headers):
            if i >= len(self.column_widths):
                break

            width = self.column_widths[i]
            header_text = str(header)

            # Add sort indicator if this column is sorted
            if self.sort_column == i:
                indicator = self.chars[
                    "sort_asc" if self.sort_ascending else "sort_desc"
                ]
                header_text = f"{header_text} {indicator}"

            # Draw header text
            draw_text(
                surface,
                header_text,
                current_x,
                y + 5,  # Center vertically
                size=font.get_height(),
                color=(255, 255, 255),
            )

            current_x += width + 5  # Add spacing between columns

        # Draw header separator
        separator_y = y + header_height - 2
        separator_char = self.chars["header_sep"]
        separator_text = separator_char * (self.rect.width - 20)  # Leave margin

        draw_text(
            surface,
            separator_text,
            x,
            separator_y,
            size=font.get_height(),
            color=(200, 200, 200),
        )

        return y + header_height

    def _draw_rows(
        self, surface: pygame.Surface, font: pygame.font.Font, x: int, y: int
    ) -> None:
        """Draw the table rows.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x, y: Position to start drawing
        """
        row_height = font.get_height() + 6
        max_y = self.rect.y + self.rect.height - 40  # Leave space at bottom

        # Calculate how many rows can be displayed
        self.visible_rows = (max_y - y) // row_height

        # Determine visible row range
        visible_count = min(self.visible_rows, len(self.data) - self.scroll_offset)

        # Draw each visible row
        for i in range(visible_count):
            row_idx = i + self.scroll_offset
            row = self.data[row_idx]
            row_y = y + i * row_height

            # Skip if row would be drawn outside the visible area
            if row_y > max_y:
                break

            # Draw row components
            self._draw_row_highlight(surface, font, x, row_y, row_height, row_idx)
            self._draw_row_cells(surface, font, x, row_y, row)
            self._draw_row_separator(
                surface, font, x, row_y, row_height, i, row_idx, visible_count
            )

    def _draw_row_highlight(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        row_y: int,
        row_height: int,
        row_idx: int,
    ) -> None:
        """Draw highlight for selected row.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X position of the row
            row_y: Y position of the row
            row_height: Height of the row
            row_idx: Index of the row in data
        """
        if row_idx != self.selected_row:
            return

        # Create highlight rectangle
        highlight_rect = pygame.Rect(
            x - 5,
            row_y - 2,
            self.rect.width - 10,
            row_height,
        )

        # Determine highlight color based on style
        highlight_color = self._get_highlight_color()

        # Create and draw highlight surface
        highlight_surface = pygame.Surface(
            (highlight_rect.width, highlight_rect.height),
            pygame.SRCALPHA,
        )
        highlight_surface.fill(highlight_color)
        surface.blit(highlight_surface, highlight_rect)

        # Draw selection marker
        draw_text(
            surface,
            self.chars["selected_marker"],
            x - 15,
            row_y + 2,
            size=font.get_height(),
            color=(255, 255, 100),
        )

    def _get_highlight_color(self) -> Tuple[int, int, int, int]:
        """Get the highlight color based on current style.

        Returns:
            Tuple[int, int, int, int]: RGBA color value
        """
        if self.style == UIStyle.QUANTUM:
            return (30, 60, 100, 150)  # Semi-transparent blue
        elif self.style == UIStyle.SYMBIOTIC:
            return (30, 100, 30, 150)  # Semi-transparent green
        else:
            return (60, 60, 60, 150)  # Semi-transparent gray

    def _draw_row_cells(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        row_y: int,
        row: List[Any],
    ) -> None:
        """Draw cells for a row.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: Starting X position
            row_y: Y position of the row
            row: Row data to display
        """
        current_x = x

        for j, cell in enumerate(row):
            if j >= len(self.column_widths):
                break

            width = self.column_widths[j]
            cell_text = self._format_cell_text(str(cell), width)

            # Draw cell text
            draw_text(
                surface,
                cell_text,
                current_x,
                row_y + 3,  # Center vertically
                size=font.get_height(),
                color=(200, 200, 200),
            )

            current_x += width + 5  # Add spacing between columns

    @staticmethod
    def _format_cell_text(cell_text: str, width: int) -> str:
        """Format cell text, truncating if necessary.

        Args:
            cell_text: Text to format
            width: Available width for the text

        Returns:
            str: Formatted text
        """
        if len(cell_text) > width - 2:
            return f"{cell_text[: width - 5]}..."
        return cell_text

    def _draw_row_separator(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        row_y: int,
        row_height: int,
        row_index: int,
        data_index: int,
        visible_count: int,
    ) -> None:
        """Draw separator between rows.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X position
            row_y: Y position of the row
            row_height: Height of the row
            row_index: Index in the visible rows
            data_index: Index in the data array
            visible_count: Number of visible rows
        """
        # Only draw separator if not the last visible row and not the last data row
        if row_index >= visible_count - 1 or data_index >= len(self.data) - 1:
            return

        separator_y = row_y + row_height - 1
        separator_char = self.chars["row_sep"]
        separator_text = separator_char * (self.rect.width - 20)  # Leave margin

        draw_text(
            surface,
            separator_text,
            x,
            separator_y,
            size=font.get_height(),
            color=(100, 100, 100),
        )

    def _draw_scrollbar(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw a scrollbar if needed.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
        """
        if len(self.data) <= self.visible_rows:
            return  # No need for scrollbar

        scrollbar_x = self.rect.x + self.rect.width - 15
        scrollbar_y = self.rect.y + 40  # Start below headers
        scrollbar_height = self.rect.height - 80  # Leave space at top and bottom

        # Draw scrollbar track
        track_rect = pygame.Rect(scrollbar_x, scrollbar_y, 10, scrollbar_height)
        track_surface = pygame.Surface(
            (track_rect.width, track_rect.height), pygame.SRCALPHA
        )
        track_surface.fill((50, 50, 50, 100))  # Semi-transparent dark gray
        surface.blit(track_surface, track_rect)

        # Calculate thumb position and size
        total_rows = len(self.data)
        thumb_height = max(20, scrollbar_height * self.visible_rows / total_rows)
        thumb_pos = scrollbar_y + (
            scrollbar_height - thumb_height
        ) * self.scroll_offset / max(1, total_rows - self.visible_rows)

        # Draw scrollbar thumb
        thumb_rect = pygame.Rect(scrollbar_x, thumb_pos, 10, thumb_height)
        thumb_surface = pygame.Surface(
            (thumb_rect.width, thumb_rect.height), pygame.SRCALPHA
        )

        # Use different thumb colors based on style
        if self.style == UIStyle.QUANTUM:
            thumb_color = (100, 200, 255, 200)  # Semi-transparent blue
        elif self.style == UIStyle.SYMBIOTIC:
            thumb_color = (150, 255, 150, 200)  # Semi-transparent green
        else:
            thumb_color = (150, 150, 150, 200)  # Semi-transparent light gray

        thumb_surface.fill(thumb_color)
        surface.blit(thumb_surface, thumb_rect)

        # Draw scroll indicators
        if self.scroll_offset > 0:
            draw_text(
                surface,
                self.chars["scroll_up"],
                scrollbar_x,
                scrollbar_y - 15,
                size=font.get_height(),
                color=(200, 200, 200),
            )

        if self.scroll_offset < total_rows - self.visible_rows:
            draw_text(
                surface,
                self.chars["scroll_down"],
                scrollbar_x,
                scrollbar_y + scrollbar_height + 5,
                size=font.get_height(),
                color=(200, 200, 200),
            )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the table.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        # Update animation state
        self._update_animation()

        # Create the panel for the table
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)

        # Calculate starting position for table content
        x = self.rect.x + 10  # Left margin
        y = self.rect.y + 10  # Top margin

        # Adjust y if there's a title
        if self.title:
            y += font.get_height() + 5

        # Draw headers
        y = self._draw_headers(surface, font, x, y)

        # Draw rows
        self._draw_rows(surface, font, x, y)

        # Draw scrollbar if needed
        self._draw_scrollbar(surface, font)

        return panel_rect

    def draw_ascii_table(
        self,
        x: int,
        y: int,
        headers: List[str],
        rows: List[List[str]],
        col_widths: Optional[List[int]] = None,
        font: Optional[pygame.font.Font] = None,
        color: Tuple[int, int, int] = COLOR_TEXT,
        border_style: str = "single",
    ) -> pygame.Rect:
        """
        Draw an ASCII-styled table.

        Args:
            surface: Surface to draw on
            x, y: Position coordinates
            headers: Column headers
            rows: Table data rows
            col_widths: Optional fixed column widths
            font: Font to use (or None to use default)
            color: Text color
            border_style: Border style ('single', 'double', 'heavy')

        Returns:
            pygame.Rect: Bounding rectangle of the table
        """
        # Initialize font and border characters
        font = self._initialize_ascii_table_font(font)
        borders = self._get_border_chars_for_style(border_style)

        # Calculate column widths if not provided
        if not col_widths:
            col_widths = self._calculate_ascii_table_col_widths(headers, rows)

        # Generate table strings
        table_strings = self._generate_ascii_table_strings(
            headers, rows, col_widths, borders
        )

        # Render the table
        return self._render_ascii_table_strings(table_strings, x, y, font, color)

    @staticmethod
    def _initialize_ascii_table_font(
        font: Optional[pygame.font.Font]
    ) -> pygame.font.Font:
        """Initialize font for ASCII table drawing.

        Args:
            font: Optional font to use

        Returns:
            pygame.font.Font: Initialized font
        """
        if font:
            return font

        try:
            return pygame.font.SysFont(
                "Courier New", 16
            )  # Monospace font works best for ASCII art
        except Exception:
            return pygame.font.Font(None, 16)

    @staticmethod
    def _get_border_chars_for_style(border_style: str) -> dict:
        """Get border characters based on style.

        Args:
            border_style: Style name ('single', 'double', 'heavy')

        Returns:
            dict: Dictionary of border characters
        """
        borders = {
            "single": {
                "tl": "+",
                "tr": "+",
                "bl": "+",
                "br": "+",
                "h": "-",
                "v": "|",
                "lc": "+",
                "rc": "+",
                "tc": "+",
                "bc": "+",
            },
            "double": {
                "tl": "╔",
                "tr": "╗",
                "bl": "╚",
                "br": "╝",
                "h": "═",
                "v": "║",
                "lc": "╠",
                "rc": "╣",
                "tc": "╦",
                "bc": "╩",
            },
            "heavy": {
                "tl": "┏",
                "tr": "┓",
                "bl": "┗",
                "br": "┛",
                "h": "━",
                "v": "┃",
                "lc": "┣",
                "rc": "┫",
                "tc": "┳",
                "bc": "┻",
            },
        }

        return borders.get(border_style, borders["single"])

    @staticmethod
    def _calculate_ascii_table_col_widths(
        headers: List[str], rows: List[List[str]]
    ) -> List[int]:
        """Calculate optimal column widths based on content.

        Args:
            headers: Column headers
            rows: Table data rows

        Returns:
            List[int]: List of column widths
        """
        col_widths = []

        for i, item in enumerate(headers):
            # Start with header width
            col_width = len(item)

            # Check all rows for maximum content width
            for row in rows:
                if i < len(row):
                    col_width = max(col_width, len(row[i]))

            # Add padding
            col_widths.append(col_width + 2)

        return col_widths

    @staticmethod
    def _generate_top_border(col_widths: List[int], b: dict) -> str:
        """Generate the top border of the table.

        Args:
            col_widths: List of column widths
            b: Border characters dictionary

        Returns:
            str: Top border string
        """
        top_border = b["tl"]

        for i, width in enumerate(col_widths):
            top_border += b["h"] * width
            top_border += b["tc"] if i < len(col_widths) - 1 else b["tr"]

        return top_border

    @staticmethod
    def _generate_header_row(
        headers: List[str], col_widths: List[int], b: dict
    ) -> str:
        """Generate the header row of the table.

        Args:
            headers: Column headers
            col_widths: List of column widths
            b: Border characters dictionary

        Returns:
            str: Header row string
        """
        header_row = b["v"]

        for i, header in enumerate(headers):
            header_row += header.ljust(col_widths[i]) + b["v"]

        return header_row

    @staticmethod
    def _generate_separator(col_widths: List[int], b: dict) -> str:
        """Generate the separator between header and data rows.

        Args:
            col_widths: List of column widths
            b: Border characters dictionary

        Returns:
            str: Separator string
        """
        separator = b["lc"]

        for width in col_widths:
            separator += b["h"] * width
            # Always add the right connector character
            separator += b["rc"]

        return separator

    @staticmethod
    def _generate_data_row(row: List[str], col_widths: List[int], b: dict) -> str:
        """Generate a data row string.

        Args:
            row: Row data
            col_widths: List of column widths
            b: Border characters dictionary

        Returns:
            str: Data row string
        """
        data_row = b["v"]

        for i, item in enumerate(col_widths):
            cell = row[i] if i < len(row) else ""
            data_row += cell.ljust(item) + b["v"]

        return data_row

    @staticmethod
    def _generate_bottom_border(col_widths: List[int], b: dict) -> str:
        """Generate the bottom border of the table.

        Args:
            col_widths: List of column widths
            b: Border characters dictionary

        Returns:
            str: Bottom border string
        """
        bottom_border = b["bl"]

        for i, width in enumerate(col_widths):
            bottom_border += b["h"] * width
            bottom_border += b["bc"] if i < len(col_widths) - 1 else b["br"]

        return bottom_border

    def _generate_ascii_table_strings(
        self, headers: List[str], rows: List[List[str]], col_widths: List[int], b: dict
    ) -> List[str]:
        """Generate all strings needed to render the ASCII table.

        Args:
            headers: Column headers
            rows: Table data rows
            col_widths: List of column widths
            b: Border characters dictionary

        Returns:
            List[str]: List of table strings to render
        """
        # Initialize with top border, headers, and separator
        table_strings = [
            self._generate_top_border(col_widths, b),
            self._generate_header_row(headers, col_widths, b),
            self._generate_separator(col_widths, b),
        ]

        # Add data rows
        table_strings.extend(
            [self._generate_data_row(row, col_widths, b) for row in rows]
        )

        # Add bottom border
        table_strings.append(self._generate_bottom_border(col_widths, b))

        return table_strings

    def _render_ascii_table_strings(
        self,
        table_strings: List[str],
        x: int,
        y: int,
        font: pygame.font.Font,
        color: Tuple[int, int, int],
    ) -> pygame.Rect:
        """Render all table strings to the surface.

        Args:
            table_strings: List of strings to render
            x, y: Position coordinates
            font: Font to use for rendering
            color: Text color

        Returns:
            pygame.Rect: Bounding rectangle of the table
        """
        char_height = font.size("X")[1]
        table_rect = None

        for i, line in enumerate(table_strings):
            line_rect = draw_text(
                self,
                line,
                x,
                y + i * char_height,
                size=font.get_height(),
                color=color,
            )

            table_rect = table_rect.union(line_rect) if table_rect else line_rect

        return table_rect or pygame.Rect(x, y, 0, 0)
