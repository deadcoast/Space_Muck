

# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height

class ASCIIEfficiencyMonitor:
    """Panel for monitoring and displaying efficiency metrics over time."""

# Standard library imports

# Third-party library imports

# Local application imports
from typing import Tuple, List, TypeVar
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.draw_utils import draw_text
from src.ui.ui_base.ascii_ui import ASCIIPanel
import pygame

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Efficiency Monitor",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize an efficiency monitor panel.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        self.history: List[float] = []
        self.max_history = 24  # Keep 24 hours of hourly data
        self.current_efficiency = 0.0
        self.daily_average = 0.0
        self.trend_direction = 0  # -1: down, 0: stable, 1: up

        # Style-based characters
        style_chars = {
            UIStyle.QUANTUM: {
                "graph_bg": "·",
                "graph_point": "◆",
                "trend_up": "↗",
                "trend_down": "↘",
                "trend_stable": "→",
            },
            UIStyle.SYMBIOTIC: {
                "graph_bg": ".",
                "graph_point": "*",
                "trend_up": "↑",
                "trend_down": "↓",
                "trend_stable": "-",
            },
            UIStyle.MECHANICAL: {
                "graph_bg": ".",
                "graph_point": "#",
                "trend_up": "^",
                "trend_down": "v",
                "trend_stable": "-",
            },
        }
        self.chars = style_chars.get(self.style, style_chars[UIStyle.MECHANICAL])

    def update_efficiency(self, current: float) -> None:
        """Update efficiency metrics with a new value.

        Args:
            current: Current efficiency value (0.0 to 1.0)
        """
        self.current_efficiency = max(0.0, min(1.0, current))

        # Update history
        self.history.append(current)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Calculate daily average
        self.daily_average = sum(self.history) / len(self.history)

        # Calculate trend
        if len(self.history) >= 2:
            recent_change = self.history[-1] - self.history[-2]
            # Determine trend direction based on recent change
            if recent_change > 0.05:
                self.trend_direction = 1  # Upward trend
            elif recent_change < -0.05:
                self.trend_direction = -1  # Downward trend
            else:
                self.trend_direction = 0  # Stable

    def _draw_graph(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Draw efficiency history graph.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x, y: Position to draw at
            width, height: Size of the graph area
        """
        # Draw background grid
        for row in range(height):
            line = self.chars["graph_bg"] * width
            draw_text(
                surface,
                line,
                x,
                y + row * font.get_height(),
                size=font.get_height(),
                color=(60, 60, 60),
            )

        # Draw data points
        if self.history:
            max_val = max(self.history)
            min_val = min(self.history)
            val_range = max(0.001, max_val - min_val)  # Avoid division by zero

            for i, val in enumerate(self.history):
                # Calculate position
                px = x + int(i * width / len(self.history))
                py = (
                    y
                    + int((1 - (val - min_val) / val_range) * (height - 1))
                    * font.get_height()
                )

                # Draw point
                # Determine color based on efficiency value
                if val > 0.8:
                    color = (100, 255, 100)  # Green for high efficiency
                elif val > 0.5:
                    color = (255, 255, 100)  # Yellow for medium efficiency
                else:
                    color = (255, 100, 100)  # Red for low efficiency
                draw_text(
                    surface,
                    self.chars["graph_point"],
                    px,
                    py,
                    size=font.get_height(),
                    color=color,
                )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the efficiency monitor panel.

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
        content_width = self.rect.width - margin * 2

        # Draw current efficiency
        # Determine color based on current efficiency
        if self.current_efficiency > 0.8:
            eff_color = (100, 255, 100)  # Green for high efficiency
        elif self.current_efficiency > 0.5:
            eff_color = (255, 255, 100)  # Yellow for medium efficiency
        else:
            eff_color = (255, 100, 100)  # Red for low efficiency
        current_text = f"Current: {self.current_efficiency*100:.1f}%"
        draw_text(
            surface,
            current_text,
            content_x,
            content_y,
            size=font.get_height(),
            color=eff_color,
        )

        # Draw trend indicator
        trend_char = {
            1: self.chars["trend_up"],
            0: self.chars["trend_stable"],
            -1: self.chars["trend_down"],
        }[self.trend_direction]
        trend_color = {1: (100, 255, 100), 0: (200, 200, 200), -1: (255, 100, 100)}[
            self.trend_direction
        ]
        trend_x = content_x + font.size(current_text)[0] + margin
        draw_text(
            surface,
            trend_char,
            trend_x,
            content_y,
            size=font.get_height(),
            color=trend_color,
        )

        # Draw daily average
        avg_y = content_y + margin
        draw_text(
            surface,
            f"24h Average: {self.daily_average*100:.1f}%",
            content_x,
            avg_y,
            size=font.get_height(),
            color=(200, 200, 200),
        )

        # Draw efficiency history graph
        graph_y = avg_y + margin * 2
        graph_height = 8  # lines
        self._draw_graph(
            surface, font, content_x, graph_y, content_width - margin, graph_height
        )

        return panel_rect
