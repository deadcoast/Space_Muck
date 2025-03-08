"""
ASCII UI components for Space Muck.

This module provides components and utilities for creating ASCII-style UI interfaces
for the converter management system and other game elements.
"""

from typing import Tuple, List, Dict, Optional, Any, Callable
import pygame

from ui.draw_utils import draw_text, draw_panel
from config import COLOR_TEXT, COLOR_BG


class ASCIIBox:
    """A box drawn with ASCII characters for borders."""
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: Optional[str] = None,
        border_style: str = "single",
    ):
        """
        Initialize an ASCII box.
        
        Args:
            x: X coordinate of the top-left corner
            y: Y coordinate of the top-left corner
            width: Width of the box in characters
            height: Height of the box in characters
            title: Optional title to display at the top of the box
            border_style: Style of border ('single', 'double', 'heavy')
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title = title
        self.content: List[Tuple[int, int, str, Optional[Dict[str, Any]]]] = []
        self.set_border_style(border_style)
        
    def set_border_style(self, style: str) -> None:
        """
        Set the border style.
        
        Args:
            style: Border style ('single', 'double', 'heavy')
        """
        if style == "double":
            self.borders = {
                "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝",
                "h": "═", "v": "║"
            }
        elif style == "heavy":
            self.borders = {
                "tl": "┏", "tr": "┓", "bl": "┗", "br": "┛",
                "h": "━", "v": "┃"
            }
        else:
            # Default to single
            self.borders = {
                "tl": "+", "tr": "+", "bl": "+", "br": "+",
                "h": "-", "v": "|"
            }
    
    def add_text(
        self,
        x: int,
        y: int,
        text: str,
        props: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add text to the box content.
        
        Args:
            x: Relative X position within the box
            y: Relative Y position within the box
            text: Text to display
            props: Optional properties for styling
        """
        self.content.append((x, y, text, props))
    
    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = COLOR_TEXT,
        bg_color: Tuple[int, int, int] = COLOR_BG,
        alpha: int = 255
    ) -> pygame.Rect:
        """
        Draw the ASCII box on the surface.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            base_color: Base color for drawing
            bg_color: Background color
            alpha: Transparency (0-255)
            
        Returns:
            pygame.Rect: The drawn area
        """
        # Calculate dimensions based on font size
        char_width, char_height = font.size("X")
        box_width = self.width * char_width
        box_height = self.height * char_height

        # Create box surface with alpha
        box_surf = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        box_surf.fill((bg_color[0], bg_color[1], bg_color[2], alpha))

        # Draw borders
        # Top border
        top_border = self.borders["tl"] + self.borders["h"] * (self.width - 2) + self.borders["tr"]
        draw_text(box_surf, top_border, 0, 0, font=font, color=base_color)

        # Bottom border
        bottom_border = self.borders["bl"] + self.borders["h"] * (self.width - 2) + self.borders["br"]
        draw_text(box_surf, bottom_border, 0, (self.height - 1) * char_height, font=font, color=base_color)

        # Side borders
        for i in range(1, self.height - 1):
            draw_text(box_surf, self.borders["v"], 0, i * char_height, font=font, color=base_color)
            draw_text(box_surf, self.borders["v"], (self.width - 1) * char_width, i * char_height, font=font, color=base_color)

        # Draw title if provided
        if self.title:
            # Calculate centered position
            title_x = (self.width - len(self.title)) // 2
            title_x = max(title_x, 1)
            draw_text(box_surf, self.title, title_x * char_width, 0, font=font, color=base_color)

        # Draw content
        for x, y, text, props in self.content:
            color = props.get("color", base_color) if props else base_color
            draw_text(box_surf, text, x * char_width, y * char_height, font=font, color=color)

        return surface.blit(box_surf, (self.x, self.y))


class ASCIIPanel:
    """A panel with ASCII styling for complex UI layouts."""
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: Optional[str] = None,
        border_style: str = "single",
    ):
        """
        Initialize an ASCII panel.
        
        Args:
            rect: Rectangle defining position and size
            title: Optional title to display at the top
            border_style: Style of border ('single', 'double', 'heavy')
        """
        self.rect = rect
        self.title = title
        self.border_style = border_style
        self.components: List[Any] = []
        self.background_color = (30, 30, 40, 200)  # Dark with alpha
        
    def add_component(self, component: Any) -> None:
        """
        Add a component to the panel.
        
        Args:
            component: Component to add
        """
        self.components.append(component)
        
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
        # Create background panel
        panel_rect = draw_panel(
            surface, 
            self.rect,
            color=self.background_color,
            border_color=(100, 100, 140),
            header=self.title,
            header_color=(40, 40, 60, 220)
        )
        
        # Draw ASCII border based on style
        char_width, char_height = font.size("X")
        width_chars = self.rect.width // char_width
        height_chars = self.rect.height // char_height
        
        box = ASCIIBox(
            self.rect.x, 
            self.rect.y, 
            width_chars,
            height_chars,
            title=None,  # Already handled by panel
            border_style=self.border_style
        )
        
        box.draw(surface, font, COLOR_TEXT, bg_color=(0, 0, 0, 0))  # Transparent background
        
        # Draw components
        for component in self.components:
            component.draw(surface, font)
            
        return panel_rect


class ASCIIProgressBar:
    """ASCII-style progress bar."""
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        progress: float = 0.0,
        style: str = "block",
    ):
        """
        Initialize an ASCII progress bar.
        
        Args:
            x: X position
            y: Y position
            width: Width in characters
            progress: Initial progress (0.0 to 1.0)
            style: Visual style ('block', 'line', 'equal')
        """
        self.x = x
        self.y = y
        self.width = width
        self.progress = max(0.0, min(1.0, progress))
        self.set_style(style)
        
    def set_style(self, style: str) -> None:
        """
        Set the visual style of the progress bar.
        
        Args:
            style: Style name ('block', 'line', 'equal')
        """
        if style == "block":
            self.fill_char = "█"
            self.empty_char = "▒"
        elif style == "line":
            self.fill_char = "|"
            self.empty_char = " "
        elif style == "equal":
            self.fill_char = "="
            self.empty_char = "-"
        else:
            # Default
            self.fill_char = "#"
            self.empty_char = " "
    
    def set_progress(self, progress: float) -> None:
        """
        Set the current progress.
        
        Args:
            progress: Progress value (0.0 to 1.0)
        """
        self.progress = max(0.0, min(1.0, progress))
        
    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        fill_color: Tuple[int, int, int] = (100, 255, 100),
        empty_color: Tuple[int, int, int] = (100, 100, 100),
    ) -> pygame.Rect:
        """
        Draw the progress bar.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            fill_color: Color for the filled portion
            empty_color: Color for the empty portion
            
        Returns:
            pygame.Rect: The drawn area
        """
        filled_width = int(self.width * self.progress)

        # Render filled portion
        if filled_width > 0:
            filled_text = self.fill_char * filled_width
            filled_rect = draw_text(
                surface, 
                filled_text,
                self.x, 
                self.y,
                font=font,
                color=fill_color
            )
        else:
            filled_rect = pygame.Rect(self.x, self.y, 0, 0)

        # Render empty portion
        if filled_width < self.width:
            empty_text = self.empty_char * (self.width - filled_width)
            empty_rect = draw_text(
                surface,
                empty_text,
                self.x + filled_rect.width,
                self.y,
                font=font,
                color=empty_color
            )

            return filled_rect.union(empty_rect)
        return filled_rect


class ASCIIButton:
    """Interactive button with ASCII styling."""
    
    def __init__(
        self,
        x: int,
        y: int,
        text: str,
        callback: Optional[Callable[[], None]] = None,
        style: str = "bracket",
    ):
        """
        Initialize an ASCII button.
        
        Args:
            x: X position
            y: Y position
            text: Button text
            callback: Function to call when clicked
            style: Button style ('bracket', 'block', 'underline')
        """
        self.x = x
        self.y = y
        self.text = text
        self.callback = callback
        self.hover = False
        self.set_style(style)
        self.rect = pygame.Rect(0, 0, 0, 0)  # Will be set properly when drawn
        
    def set_style(self, style: str) -> None:
        """
        Set the button style.
        
        Args:
            style: Style name ('bracket', 'block', 'underline')
        """
        if style == "bracket":
            self.prefix = "["
            self.suffix = "]"
        elif style == "block":
            self.prefix = "█ "
            self.suffix = " █"
        elif style == "underline":
            self.prefix = "_"
            self.suffix = "_"
        else:
            # Default
            self.prefix = "<"
            self.suffix = ">"
            
    def is_hover(self, mouse_pos: Tuple[int, int]) -> bool:
        """Check if the mouse is hovering over this button."""
        return self.rect.collidepoint(mouse_pos)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame events for this button.
        
        Returns:
            bool: True if the event was consumed
        """
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.is_hover(event.pos)
            return self.hover
            
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and (self.is_hover(event.pos) and self.callback):
            self.callback()
            return True
                
        return False
        
    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        color: Tuple[int, int, int] = COLOR_TEXT,
        hover_color: Tuple[int, int, int] = (200, 200, 255),
    ) -> pygame.Rect:
        """
        Draw the button.
        
        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            color: Normal text color
            hover_color: Color when hovering
            
        Returns:
            pygame.Rect: The drawn area
        """
        current_color = hover_color if self.hover else color
        
        button_text = f"{self.prefix}{self.text}{self.suffix}"
        self.rect = draw_text(
            surface,
            button_text,
            self.x,
            self.y,
            font=font,
            color=current_color
        )
        
        return self.rect


def draw_ascii_table(
    surface: pygame.Surface,
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
    if not font:
        try:
            font = pygame.font.SysFont("Courier New", 16)  # Monospace font works best for ASCII art
        except Exception:
            font = pygame.font.Font(None, 16)

    # Determine column widths if not provided
    if not col_widths:
        col_widths = []
        for i in range(len(headers)):
            # Calculate max width needed for this column
            col_width = len(headers[i])
            for row in rows:
                if i < len(row):
                    col_width = max(col_width, len(row[i]))
            col_widths.append(col_width + 2)  # Add padding

    # Border characters based on style
    borders = {
        "single": {
            "tl": "+", "tr": "+", "bl": "+", "br": "+",
            "h": "-", "v": "|", "lc": "+", "rc": "+", "tc": "+", "bc": "+"
        },
        "double": {
            "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝",
            "h": "═", "v": "║", "lc": "╠", "rc": "╣", "tc": "╦", "bc": "╩"
        },
        "heavy": {
            "tl": "┏", "tr": "┓", "bl": "┗", "br": "┛",
            "h": "━", "v": "┃", "lc": "┣", "rc": "┫", "tc": "┳", "bc": "┻"
        }
    }

    b = borders.get(border_style, borders["single"])

    # Calculate total width
    total_width = sum(col_widths) + len(col_widths) + 1

    # Top border
    top_border = b["tl"]
    for i, width in enumerate(col_widths):
        top_border += b["h"] * width
        top_border += b["tc"] if i < len(col_widths) - 1 else b["tr"]
    table_strings = [top_border]
    # Headers
    header_row = b["v"]
    for i, header in enumerate(headers):
        header_row += header.ljust(col_widths[i]) + b["v"]
    table_strings.append(header_row)

    # Separator
    separator = b["lc"]
    for i, width in enumerate(col_widths):
        separator += b["h"] * width
        separator += b["rc"] if i < len(col_widths) - 1 else b["rc"]
    table_strings.append(separator)

    # Data rows
    for row in rows:
        data_row = b["v"]
        for i in range(len(col_widths)):
            cell = row[i] if i < len(row) else ""
            data_row += cell.ljust(col_widths[i]) + b["v"]
        table_strings.append(data_row)

    # Bottom border
    bottom_border = b["bl"]
    for i, width in enumerate(col_widths):
        bottom_border += b["h"] * width
        bottom_border += b["bc"] if i < len(col_widths) - 1 else b["br"]
    table_strings.append(bottom_border)

    # Render all table strings
    char_height = font.size("X")[1]
    table_rect = None

    for i, line in enumerate(table_strings):
        line_rect = draw_text(
            surface,
            line,
            x,
            y + i * char_height,
            font=font,
            color=color
        )

        table_rect = table_rect.union(line_rect) if table_rect else line_rect
    return table_rect or pygame.Rect(x, y, 0, 0)
