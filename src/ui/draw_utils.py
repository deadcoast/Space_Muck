"""
Drawing utility functions for Space Muck.

This module provides a collection of helper functions for rendering text,
shapes, buttons, and other UI elements consistently across the game.
"""

from typing import Tuple, List, Optional, Dict

import pygame
import numpy as np

from config import COLOR_TEXT, COLOR_BG


def draw_text(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    size: int = 18,
    color: Tuple[int, int, int] = COLOR_TEXT,
    align: str = "left",
    shadow: bool = False,
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: int = 255,
    max_width: Optional[int] = None,
) -> pygame.Rect:
    """
    Draw text on the given surface with specified properties.

    Args:
        surface: Surface to draw on
        text: Text content to render
        x: X position
        y: Y position
        size: Font size
        color: Text color (RGB tuple)
        align: Text alignment ('left', 'center', 'right')
        shadow: Whether to draw text shadow
        shadow_color: Color of the shadow
        alpha: Text transparency (0-255)
        max_width: Maximum width before text wrapping

    Returns:
        pygame.Rect: Bounding rectangle of the rendered text
    """
    try:
        font = pygame.font.SysFont("Arial", size)
    except Exception:
        font = pygame.font.Font(None, size)

    # Handle wrapping if max_width is specified
    if max_width is not None and font.size(text)[0] > max_width:
        try:
            lines = _wrap_text(text, font, max_width)
            height = 0
            max_rect = None

            for i, line in enumerate(lines):
                rect = draw_text(
                    surface,
                    line,
                    x,
                    y + height,
                    size,
                    color,
                    align,
                    shadow,
                    shadow_color,
                    alpha,
                )
                if max_rect is None:
                    max_rect = rect
                else:
                    max_rect.height += rect.height
                    max_rect.width = max(max_rect.width, rect.width)
                height += rect.height + 2  # Add small spacing between lines

            return max_rect or pygame.Rect(x, y, 0, 0)
        except Exception as e:
            # Fallback if text wrapping fails
            import logging

            logging.warning(f"Text wrapping failed: {e}")
            # Continue with unwrapped text

    # Create text surface with transparency support
    text_surface = font.render(text, True, color)

    # Apply transparency if needed
    if alpha < 255:
        text_surface.set_alpha(alpha)

    # Calculate position based on alignment
    text_width, _ = text_surface.get_size()
    if align == "center":
        x -= text_width // 2
    elif align == "right":
        x -= text_width

    # Draw shadow first if requested
    if shadow:
        shadow_surf = font.render(text, True, shadow_color)
        surface.blit(shadow_surf, (x + 1, y + 1))

    return surface.blit(text_surface, (x, y))


def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> List[str]:
    """
    Wrap text to fit within max_width.

    Args:
        text: Text to wrap
        font: Font to use for size calculations
        max_width: Maximum width in pixels

    Returns:
        list: List of wrapped text lines
    """
    words = text.split(" ")
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        # Check if adding this word exceeds max width
        if font.size(test_line)[0] <= max_width:
            current_line.append(word)
        elif current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            # Word is too long, force split it
            lines.append(word)
            current_line = []

    # Add the last line
    if current_line:
        lines.append(" ".join(current_line))

    return lines


def draw_progress_bar(
    surface: pygame.Surface,
    x: int,
    y: int,
    width: int,
    height: int,
    progress: float,
    color: Tuple[int, int, int],
    background_color: Tuple[int, int, int] = COLOR_BG,
    border_color: Optional[Tuple[int, int, int]] = None,
    border_width: int = 1,
    label: Optional[str] = None,
    label_color: Tuple[int, int, int] = COLOR_TEXT,
    label_size: int = 14,
    show_percentage: bool = False,
) -> pygame.Rect:
    """
    Draw a progress bar on the given surface.

    Args:
        surface: Surface to draw on
        x, y: Position coordinates
        width, height: Dimensions of the progress bar
        progress: Progress value (0.0 to 1.0)
        color: Fill color of the progress bar
        background_color: Background color
        border_color: Color of the border (None for no border)
        border_width: Width of the border
        label: Optional text label
        label_color: Color for the label
        label_size: Size of the label text
        show_percentage: Whether to show percentage text

    Returns:
        pygame.Rect: Bounding rectangle of the progress bar
    """
    # Clamp progress to valid range
    progress = max(0.0, min(1.0, progress))

    # Draw background
    background_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(surface, background_color, background_rect)

    # Draw progress fill
    fill_width = int(width * progress)
    fill_rect = pygame.Rect(x, y, fill_width, height)
    pygame.draw.rect(surface, color, fill_rect)

    # Draw border if specified
    if border_color:
        pygame.draw.rect(surface, border_color, background_rect, border_width)

    # Draw label if provided
    if label:
        label_x = x + width // 2
        label_y = y + (height - label_size) // 2
        draw_text(
            surface, label, label_x, label_y, label_size, label_color, align="center"
        )

    # Draw percentage if requested
    if show_percentage:
        percentage_text = f"{int(progress * 100)}%"
        percentage_x = x + width // 2
        percentage_y = y + (height - label_size) // 2
        draw_text(
            surface,
            percentage_text,
            percentage_x,
            percentage_y,
            label_size,
            label_color,
            align="center",
            shadow=True,
        )

    return background_rect


def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font_size: int = 16,
    text_color: Tuple[int, int, int] = COLOR_TEXT,
    button_color: Tuple[int, int, int] = (80, 80, 100),
    hover_color: Tuple[int, int, int] = (100, 100, 120),
    disabled_color: Tuple[int, int, int] = (60, 60, 70),
    border_color: Optional[Tuple[int, int, int]] = None,
    border_width: int = 1,
    hover: bool = False,
    disabled: bool = False,
    shadow_size: int = 2,
    icon: Optional[pygame.Surface] = None,
) -> pygame.Rect:
    """
    Draw an interactive button with hover effects.

    Args:
        surface: Surface to draw on
        rect: Button rectangle
        text: Button text
        font_size: Size of the text
        text_color: Color of the text
        button_color: Base button color
        hover_color: Color when hovered
        disabled_color: Color when disabled
        border_color: Color of the border (None for no border)
        border_width: Width of the border
        hover: Whether the button is being hovered
        disabled: Whether the button is disabled
        shadow_size: Size of the button shadow
        icon: Optional icon to display next to text

    Returns:
        pygame.Rect: Bounding rectangle of the button
    """
    color = disabled_color if disabled else (hover_color if hover else button_color)

    # Draw shadow
    if shadow_size > 0 and not disabled:
        shadow_rect = pygame.Rect(
            rect.x + shadow_size, rect.y + shadow_size, rect.width, rect.height
        )
        pygame.draw.rect(surface, (30, 30, 35), shadow_rect, border_radius=3)

    # Draw button
    pygame.draw.rect(surface, color, rect, border_radius=3)

    # Draw border
    if border_color:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=3)

    # Position for the text
    text_x = rect.x + rect.width // 2
    text_y = rect.y + (rect.height - font_size) // 2

    # Adjust if we have an icon
    if icon:
        icon_padding = 5
        icon_x = rect.x + icon_padding
        icon_y = rect.y + (rect.height - icon.get_height()) // 2
        surface.blit(icon, (icon_x, icon_y))
        text_x += icon.get_width() // 2

    # Draw text
    draw_text(surface, text, text_x, text_y, font_size, text_color, align="center")

    return rect


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: Tuple[int, int, int] = (50, 50, 70, 230),
    border_color: Optional[Tuple[int, int, int]] = (100, 100, 140, 255),
    border_width: int = 2,
    alpha: int = 230,
    header: Optional[str] = None,
    header_height: int = 30,
    header_color: Tuple[int, int, int] = (40, 40, 60, 230),
) -> pygame.Rect:
    """
    Draw a semi-transparent panel with optional header.

    Args:
        surface: Surface to draw on
        rect: Panel rectangle
        color: Panel background color (can include alpha)
        border_color: Border color (can include alpha)
        border_width: Width of the border
        alpha: Overall panel transparency
        header: Optional header text
        header_height: Height of the header section
        header_color: Background color for the header

    Returns:
        pygame.Rect: Bounding rectangle of the panel
    """
    # Create a surface for the panel with alpha support
    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)

    # Draw main panel background
    background_color = color if len(color) == 4 else (*color, alpha)
    pygame.draw.rect(panel, background_color, (0, 0, rect.width, rect.height))

    # Draw header if specified
    if header:
        header_color = (
            header_color if len(header_color) == 4 else (*header_color, alpha)
        )
        header_rect = pygame.Rect(0, 0, rect.width, header_height)
        pygame.draw.rect(panel, header_color, header_rect)

        # Draw header text
        font_size = header_height // 2
        draw_text(
            panel,
            header,
            rect.width // 2,
            (header_height - font_size) // 2,
            font_size,
            COLOR_TEXT,
            align="center",
        )

        # Draw separator line
        pygame.draw.line(
            panel,
            border_color if len(border_color) == 4 else (*border_color, alpha),
            (0, header_height),
            (rect.width, header_height),
            2,
        )

    # Draw border if specified
    if border_color and border_width > 0:
        border_color = (
            border_color if len(border_color) == 4 else (*border_color, alpha)
        )
        pygame.draw.rect(
            panel, border_color, (0, 0, rect.width, rect.height), border_width
        )

    # Blit the panel onto the main surface
    surface.blit(panel, (rect.x, rect.y))

    return rect


def draw_tooltip(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    font_size: int = 14,
    padding: int = 8,
    color: Tuple[int, int, int] = (50, 50, 70, 230),
    text_color: Tuple[int, int, int] = COLOR_TEXT,
    border_color: Optional[Tuple[int, int, int]] = (100, 100, 140, 255),
    max_width: int = 250,
) -> pygame.Rect:
    """
    Draw a tooltip with automatic sizing based on content.

    Args:
        surface: Surface to draw on
        text: Tooltip text content
        x, y: Position coordinates (tooltip tip will point here)
        font_size: Size of the text
        padding: Padding around the text
        color: Background color
        text_color: Text color
        border_color: Border color
        max_width: Maximum width before text wrapping

    Returns:
        pygame.Rect: Bounding rectangle of the tooltip
    """
    # Create temporary font to calculate text size
    try:
        font = pygame.font.SysFont("Arial", font_size)
    except Exception:
        font = pygame.font.Font(None, font_size)

    # Wrap text if needed
    lines = _wrap_text(text, font, max_width)
    line_surfaces = [font.render(line, True, text_color) for line in lines]

    # Calculate dimensions
    width = max(surface.get_width() for surface in line_surfaces) + (padding * 2)
    height = (
        sum(surface.get_height() for surface in line_surfaces)
        + (padding * 2)
        + (len(lines) - 1) * 2
    )

    # Ensure tooltip stays within screen bounds
    screen_width = surface.get_width()

    # Adjust position to keep tooltip on screen
    tooltip_x = min(x, screen_width - width - 10)
    tooltip_y = y - height - 10  # Position above cursor by default

    # If tooltip would go off the top of the screen, put it below the cursor
    if tooltip_y < 10:
        tooltip_y = y + 20

    # Create tooltip rect
    tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, width, height)

    # Draw tooltip
    draw_panel(surface, tooltip_rect, color, border_color, 1)

    # Draw text lines
    y_offset = padding
    for line_surface in line_surfaces:
        surface.blit(line_surface, (tooltip_x + padding, tooltip_y + y_offset))
        y_offset += line_surface.get_height() + 2

    return tooltip_rect


def draw_minimap(
    surface: pygame.Surface,
    rect: pygame.Rect,
    grid: np.ndarray,
    entity_grid: Optional[np.ndarray] = None,
    player_pos: Optional[Tuple[int, int]] = None,
    view_rect: Optional[Tuple[int, int, int, int]] = None,
    border_color: Tuple[int, int, int] = (100, 100, 140),
    background_color: Tuple[int, int, int] = (20, 20, 30),
    entity_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> pygame.Rect:
    """
    Draw a minimap showing the game world.

    Args:
        surface: Surface to draw on
        rect: Rectangle defining the minimap area
        grid: 2D array representing the asteroid field
        entity_grid: 2D array representing entities
        player_pos: (x, y) coordinates of player
        view_rect: (x1, y1, x2, y2) defining current viewport
        border_color: Border color
        background_color: Background color

    Returns:
        pygame.Rect: Bounding rectangle of the minimap
    """
    # Create minimap surface
    minimap = pygame.Surface((rect.width, rect.height))
    minimap.fill(background_color)

    # Calculate scale factors and sampling steps
    scale_x, scale_y = rect.width / grid.shape[1], rect.height / grid.shape[0]
    step_x, step_y = max(1, grid.shape[1] // 100), max(1, grid.shape[0] // 100)

    # Draw the grid elements
    _draw_grid_on_minimap(minimap, grid, scale_x, scale_y, step_x, step_y)

    # Draw entities if provided
    if entity_grid is not None:
        _draw_entities_on_minimap(
            minimap,
            entity_grid,
            scale_x,
            scale_y,
            step_x,
            step_y,
            entity_colors=entity_colors,
        )

    # Draw player position if provided
    if player_pos:
        _draw_player_on_minimap(minimap, player_pos, scale_x, scale_y)

    # Draw current view rectangle if provided
    if view_rect:
        _draw_view_rect_on_minimap(minimap, view_rect, scale_x, scale_y)

    # Draw border and blit to surface
    pygame.draw.rect(minimap, border_color, (0, 0, rect.width, rect.height), 2)
    surface.blit(minimap, (rect.x, rect.y))

    return rect


def _draw_grid_on_minimap(
    minimap: pygame.Surface,
    grid: np.ndarray,
    scale_x: float,
    scale_y: float,
    step_x: int,
    step_y: int,
) -> None:
    """Draw the asteroid grid on the minimap."""
    for y in range(0, grid.shape[0], step_y):
        for x in range(0, grid.shape[1], step_x):
            if grid[y, x] > 0:
                color_value = min(255, 80 + grid[y, x] // 2)
                pygame.draw.rect(
                    minimap,
                    (color_value, color_value, color_value),
                    (
                        x * scale_x,
                        y * scale_y,
                        max(1, scale_x * step_x),
                        max(1, scale_y * step_y),
                    ),
                )


def _draw_entities_on_minimap(
    minimap: pygame.Surface,
    entity_grid: np.ndarray,
    scale_x: float,
    scale_y: float,
    step_x: int,
    step_y: int,
    entity_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> None:
    """Draw entities on the minimap.

    Args:
        minimap: Surface to draw on
        entity_grid: 2D array representing entities
        scale_x: X scale factor
        scale_y: Y scale factor
        step_x: X sampling step
        step_y: Y sampling step
        entity_colors: Optional custom color mapping for entities
    """
    # Default entity color mapping
    default_colors = {
        1: (50, 100, 255),  # Blue
        2: (255, 50, 150),  # Pink
        3: (255, 165, 0),  # Orange
    }

    # Use provided colors or defaults
    colors = entity_colors or default_colors

    # Find all unique entity IDs in the grid
    try:
        unique_entities = np.unique(entity_grid)
        unique_entities = unique_entities[unique_entities > 0]  # Exclude 0 (empty)
    except Exception:
        # Fallback if numpy operations fail
        unique_entities = [1, 2, 3]  # Default entity IDs

    for entity_id in unique_entities:
        # Get color for this entity (default to white if not found)
        color = colors.get(int(entity_id), (255, 255, 255))

        for y in range(0, entity_grid.shape[0], step_y):
            for x in range(0, entity_grid.shape[1], step_x):
                try:
                    if entity_grid[y, x] == entity_id:
                        pygame.draw.rect(
                            minimap,
                            color,
                            (
                                x * scale_x,
                                y * scale_y,
                                max(1, scale_x * step_x),
                                max(1, scale_y * step_y),
                            ),
                        )
                except IndexError:
                    # Skip out-of-bounds coordinates
                    continue


def _draw_player_on_minimap(
    minimap: pygame.Surface, player_pos: Tuple[int, int], scale_x: float, scale_y: float
) -> None:
    """Draw the player position on the minimap."""
    pygame.draw.circle(
        minimap,
        (0, 255, 0),  # Green
        (int(player_pos[0] * scale_x), int(player_pos[1] * scale_y)),
        max(3, int(min(scale_x, scale_y) * 2)),
    )


def _draw_view_rect_on_minimap(
    minimap: pygame.Surface,
    view_rect: Tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
) -> None:
    """Draw the current view rectangle on the minimap."""
    x1, y1, x2, y2 = view_rect
    pygame.draw.rect(
        minimap,
        (200, 200, 255),  # Light blue
        (x1 * scale_x, y1 * scale_y, (x2 - x1) * scale_x, (y2 - y1) * scale_y),
        1,  # Line width
    )


def draw_histogram(
    surface: pygame.Surface,
    rect: pygame.Rect,
    data: List[int],
    color: Tuple[int, int, int] = (100, 100, 255),
    background_color: Tuple[int, int, int] = (30, 30, 40),
    border_color: Tuple[int, int, int] = (100, 100, 140),
    max_bars: int = 50,
    grid_lines: bool = True,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
) -> pygame.Rect:
    """
    Draw a histogram showing data trends.

    Args:
        surface: Surface to draw on
        rect: Rectangle defining the histogram area
        data: List of data points to visualize
        color: Bar color
        background_color: Background color
        border_color: Border color
        max_bars: Maximum number of bars to display
        grid_lines: Whether to show grid lines
        title: Optional title for the histogram
        y_label: Optional y-axis label

    Returns:
        pygame.Rect: Bounding rectangle of the histogram
    """
    # Create histogram surface
    histogram = pygame.Surface((rect.width, rect.height))
    histogram.fill(background_color)

    # Draw title if provided
    title_height = 0
    if title:
        title_height = 20
        draw_text(histogram, title, rect.width // 2, 5, 16, COLOR_TEXT, align="center")

    # Draw y-label if provided
    y_label_width = 0
    if y_label:
        y_label_width = 20
        # Draw rotated text for y-label
        font = pygame.font.SysFont("Arial", 14)
        label_surf = font.render(y_label, True, COLOR_TEXT)
        label_surf = pygame.transform.rotate(label_surf, 90)
        histogram.blit(label_surf, (3, rect.height // 2 - label_surf.get_width() // 2))

    # Calculate chart dimensions
    chart_x = y_label_width + 5
    chart_y = title_height + 5
    chart_width = rect.width - chart_x - 5
    chart_height = rect.height - chart_y - 20  # Leave room for x-axis

    # Draw chart background
    chart_rect = pygame.Rect(chart_x, chart_y, chart_width, chart_height)
    pygame.draw.rect(histogram, (20, 20, 30), chart_rect)

    # Draw grid lines if requested
    if grid_lines:
        for i in range(1, 5):
            y_pos = chart_y + chart_height * i // 4
            pygame.draw.line(
                histogram,
                (50, 50, 60),
                (chart_x, y_pos),
                (chart_x + chart_width, y_pos),
                1,
            )

        for i in range(1, 5):
            x_pos = chart_x + chart_width * i // 4
            pygame.draw.line(
                histogram,
                (50, 50, 60),
                (x_pos, chart_y),
                (x_pos, chart_y + chart_height),
                1,
            )

    # Draw data
    if data:
        # If too much data, sample it
        display_data = data
        if len(data) > max_bars:
            step = len(data) / max_bars
            display_data = [
                data[min(len(data) - 1, int(i * step))] for i in range(max_bars)
            ]

        # Find max value for scaling
        max_val = max(display_data, default=1)
        if max_val == 0:
            max_val = 1

        # Draw bars
        bar_width = chart_width / len(display_data)
        for i, value in enumerate(display_data):
            bar_height = (value / max_val) * chart_height
            bar_x = chart_x + i * bar_width
            bar_y = chart_y + chart_height - bar_height

            pygame.draw.rect(
                histogram, color, (bar_x, bar_y, bar_width - 1, bar_height)
            )

    # Draw border
    pygame.draw.rect(histogram, border_color, (0, 0, rect.width, rect.height), 2)

    # Blit histogram to surface
    surface.blit(histogram, (rect.x, rect.y))

    return rect


def draw_circle_button(
    surface: pygame.Surface,
    center_x: int,
    center_y: int,
    radius: int,
    icon: Optional[pygame.Surface] = None,
    text: Optional[str] = None,
    color: Tuple[int, int, int] = (80, 80, 100),
    hover_color: Tuple[int, int, int] = (100, 100, 120),
    border_color: Optional[Tuple[int, int, int]] = None,
    text_color: Tuple[int, int, int] = COLOR_TEXT,
    hover: bool = False,
    shadow: bool = True,
) -> pygame.Rect:
    """
    Draw a circular button with icon or text.

    Args:
        surface: Surface to draw on
        center_x, center_y: Center position of the button
        radius: Radius of the button
        icon: Optional icon to display
        text: Optional text (used if no icon provided)
        color: Button color
        hover_color: Color when hovered
        border_color: Border color (None for no border)
        text_color: Text color
        hover: Whether the button is being hovered
        shadow: Whether to draw a shadow

    Returns:
        pygame.Rect: Bounding rectangle of the button
    """
    # Create button rect for hit-testing
    button_rect = pygame.Rect(
        center_x - radius, center_y - radius, radius * 2, radius * 2
    )

    # Draw shadow
    if shadow:
        pygame.draw.circle(surface, (30, 30, 35), (center_x + 2, center_y + 2), radius)

    # Draw button
    pygame.draw.circle(
        surface, hover_color if hover else color, (center_x, center_y), radius
    )

    # Draw border
    if border_color:
        pygame.draw.circle(surface, border_color, (center_x, center_y), radius, 2)

    # Draw icon or text
    if icon:
        icon_rect = icon.get_rect(center=(center_x, center_y))
        surface.blit(icon, icon_rect)
    elif text:
        draw_text(
            surface,
            text,
            center_x,
            center_y - 7,
            min(16, radius),
            text_color,
            align="center",
        )

    return button_rect
