from typing import Tuple, List, Dict, Optional, Any, TypeVar
import pygame

from ui.draw_utils import draw_text
from ui.ascii_base import UIStyle
from ui.ui_element.ascii_panel import ASCIIPanel

# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height


class ASCIIMetricsPanel:
    """A specialized panel for displaying production metrics and performance indicators."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Production Metrics",
        style: UIStyle = UIStyle.MECHANICAL,
        converter_type: Optional[str] = None,
    ):
        """Initialize a metrics panel.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
            converter_type: Optional converter type to determine style
        """
        self.rect = rect
        self.title = title
        self.style = (
            UIStyle.get_style_for_converter(converter_type) if converter_type else style
        )
        self.metrics: Dict[str, Any] = {
            "throughput": 0.0,  # units/min
            "energy_usage": 0.0,  # kW
            "queue_size": 0,  # items in process queue
            "utilization": 0.0,  # percentage
            "uptime": 0.0,  # hours
            "efficiency": 0.0,  # percentage
        }
        self.history: Dict[str, List[float]] = {
            "throughput": [],
            "energy_usage": [],
            "efficiency": [],
        }
        self.max_history = 60  # Keep 1 hour of minute-by-minute data

        # Base colors for different styles
        style_colors = {
            UIStyle.QUANTUM: {
                "high": (100, 200, 255),  # Cyan
                "medium": (255, 200, 100),  # Orange
                "low": (255, 100, 100),  # Red
                "primary": (100, 200, 255),  # Cyan
                "accent1": (255, 100, 100),  # Red
                "accent2": (100, 255, 100),  # Green
                "accent3": (200, 100, 255),  # Purple
            },
            UIStyle.SYMBIOTIC: {
                "high": (150, 255, 150),  # Light green
                "medium": (255, 255, 150),  # Light yellow
                "low": (255, 150, 150),  # Light red
                "primary": (150, 255, 150),  # Light green
                "accent1": (255, 150, 150),  # Light red
                "accent2": (150, 150, 255),  # Light blue
                "accent3": (255, 150, 255),  # Light purple
            },
            UIStyle.MECHANICAL: {
                "high": (100, 255, 100),  # Green
                "medium": (255, 255, 100),  # Yellow
                "low": (255, 100, 100),  # Red
                "primary": (100, 255, 100),  # Green
                "accent1": (255, 100, 100),  # Red
                "accent2": (100, 100, 255),  # Blue
                "accent3": (255, 100, 255),  # Magenta
            },
        }

        # Get base colors for current style
        base_colors = style_colors.get(self.style, style_colors[UIStyle.MECHANICAL])

        # Map colors to metrics
        self.colors = {
            **base_colors,  # Include high/medium/low and base colors
            "throughput": base_colors["primary"],
            "energy": base_colors["accent1"],
            "queue": base_colors["accent2"],
            "efficiency": base_colors["medium"],
            "uptime": base_colors["accent3"],
        }

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update current metrics and historical data.

        Args:
            metrics: Dictionary of current metric values
        """
        # Update current metrics
        self.metrics.update(metrics)

        # Update historical data
        for key in ["throughput", "energy_usage", "efficiency"]:
            if key in metrics:
                self.history[key].append(metrics[key])
                if len(self.history[key]) > self.max_history:
                    self.history[key].pop(0)

    def _get_trend_indicator(self, values: List[float]) -> str:
        """Get trend indicator based on recent values."""
        if len(values) < 2:
            return "-"

        recent = values[-5:] if len(values) >= 5 else values
        old_values = values[:-5] if len(values) > 5 else [values[0]]

        avg_old = sum(old_values) / len(old_values)
        avg_new = sum(recent) / len(recent)

        trend_indicators = {
            lambda x: x > 1.05: "↗",  # Significant increase
            lambda x: x < 0.95: "↘",  # Significant decrease
            lambda x: True: "→",  # Stable
        }

        ratio = avg_new / avg_old
        return next(
            indicator
            for condition, indicator in trend_indicators.items()
            if condition(ratio)
        )

    def _draw_metric_label(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        label: str,
        color: Tuple[int, int, int],
    ) -> int:
        """Draw the metric label and return its width."""
        text = f"{label}: "
        draw_text(surface, text, x, y, size=font.get_height(), color=color)
        return font.size(text)[0]

    def _get_bar_chars(self, fill_width: int, bar_width: int) -> str:
        """Generate the bar string with appropriate characters."""
        chars = ["=", "#", "%", "@"]
        return "".join(
            chars[min(3, int(4 * i / fill_width))] if i < fill_width else "="
            for i in range(bar_width)
        )

    def _draw_metric_bar(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
        value: float,
        max_value: float,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a metric bar with label and value."""
        # Draw label and get its width
        label_width = self._draw_metric_label(surface, font, x, y, label, color)

        # Calculate bar dimensions
        bar_width = int((width - label_width) * 0.6)
        fill_width = int(bar_width * (value / max_value))

        # Draw bar
        bar_str = self._get_bar_chars(fill_width, bar_width)
        draw_text(
            surface, bar_str, x + label_width, y, size=font.get_height(), color=color
        )

        # Draw value
        value_str = f"{value:.1f}"
        draw_text(
            surface,
            value_str,
            x + label_width + bar_width + font.size(" ")[0],
            y,
            size=font.get_height(),
            color=color,
        )

    def _get_metric_color(self, metric_name: str) -> Tuple[int, int, int]:
        """Get the appropriate color for a metric based on its value."""
        if metric_name == "utilization":
            value = self.metrics["utilization"]
            thresholds = [(80, "high"), (50, "medium"), (0, "low")]
        elif metric_name == "efficiency":
            value = self.metrics["efficiency"]
            thresholds = [(90, "high"), (70, "medium"), (0, "low")]
        else:
            return self.colors["medium"]

        return next(
            self.colors[level] for threshold, level in thresholds if value > threshold
        )

    def _draw_throughput(
        self, surface: pygame.Surface, font: pygame.font.Font, x: int, y: int
    ) -> None:
        """Draw throughput metrics."""
        throughput = self.metrics["throughput"]
        trend = self._get_trend_indicator(self.history["throughput"])
        draw_text(
            surface,
            f"Throughput: {throughput:.1f} units/min {trend}",
            x,
            y,
            size=font.get_height(),
            color=self.colors["high"],
        )

    def _draw_uptime(
        self, surface: pygame.Surface, font: pygame.font.Font, x: int, y: int
    ) -> None:
        """Draw uptime information."""
        hours = int(self.metrics["uptime"])
        minutes = int((self.metrics["uptime"] - hours) * 60)
        draw_text(
            surface,
            f"Uptime: {hours}h {minutes}m",
            x,
            y,
            size=font.get_height(),
            color=self.colors["low"],
        )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the metrics panel.

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
        x = self.rect.x + margin
        y = self.rect.y + margin * 2  # Extra margin for title
        width = self.rect.width - margin * 2

        # Draw metrics
        self._draw_throughput(surface, font, x, y)

        # Draw energy usage bar
        y += margin
        self._draw_metric_bar(
            surface,
            font,
            x,
            y,
            width,
            self.metrics["energy_usage"],
            100.0,
            "Energy",
            self.colors["medium"],
        )

        # Draw queue size
        y += margin
        draw_text(
            surface,
            f"Queue: {self.metrics['queue_size']} items",
            x,
            y,
            size=font.get_height(),
            color=self.colors["medium"],
        )

        # Draw utilization
        y += margin
        self._draw_metric_bar(
            surface,
            font,
            x,
            y,
            width,
            self.metrics["utilization"],
            100.0,
            "Util%",
            self._get_metric_color("utilization"),
        )

        # Draw uptime
        y += margin
        self._draw_uptime(surface, font, x, y)

        # Draw efficiency with trend
        y += margin
        eff_trend = self._get_trend_indicator(self.history["efficiency"])
        self._draw_metric_bar(
            surface,
            font,
            x,
            y,
            width,
            self.metrics["efficiency"],
            100.0,
            f"Eff%{eff_trend}",
            self._get_metric_color("efficiency"),
        )

        return panel_rect
