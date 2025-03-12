from typing import Tuple, List, Dict, Optional, Any, TypeVar
import pygame
import time

from ui.draw_utils import draw_text
from ui.ascii_base import UIStyle
from ui.ui_element.ascii_panel import ASCIIPanel
from config import COLOR_TEXT

# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height


class ASCIIChainVisualizer:
    """Visualizer for displaying converter chains and resource flows."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Chain Flow",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a chain visualizer.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        self.converters: List[Dict[str, Any]] = []
        self.connections: List[Tuple[int, int]] = []  # (from_idx, to_idx)
        self.flow_rates: Dict[Tuple[int, int], float] = {}  # (from, to) -> rate

        # Enhanced tracking of converter states
        self.converter_states: Dict[int, Dict[str, Any]] = {}
        self.efficiency_history: Dict[int, List[float]] = {}
        self.max_history_points = 60  # Keep one hour of minute-by-minute data

        # Animation and visual settings
        self.animation_settings = {
            "base_speed": 8.0,
            "speed_scale": 0.1,  # For flow rate scaling
            "max_speed": 24.0,  # Maximum animation speed
            "pulse_frequency": 0.5,  # Pulses per second
            "glow_intensity": 0.8,  # For status indicators
        }

        # Performance optimization
        self.cache_timeout = 1.0  # Seconds before refreshing cached calculations
        self._cached_layouts: Dict[str, Any] = {}
        self._last_cache_time = 0.0

        # Style-based characters
        style_chars = {
            UIStyle.QUANTUM: {
                "node": "@",
                "flow": "~",
                "arrow": ">",
                "corner": "+",
                "anim_chars": ["⋅", "∙", "•", "∙"],  # Quantum dots
                "resource_chars": {
                    "energy": "⚡",
                    "matter": "⬢",
                    "fluid": "⬡",
                    "data": "⬟",
                    "default": "⬡",
                },
            },
            UIStyle.SYMBIOTIC: {
                "node": "*",
                "flow": ".",
                "arrow": ">",
                "corner": "+",
                "anim_chars": ["·", "•", "○", "•"],  # Organic shapes
                "resource_chars": {
                    "energy": "✧",
                    "matter": "✦",
                    "fluid": "❋",
                    "data": "✺",
                    "default": "✧",
                },
            },
            UIStyle.MECHANICAL: {
                "node": "#",
                "flow": "-",
                "arrow": ">",
                "corner": "+",
                "anim_chars": [".", "o", "O", "o"],  # Mechanical progression
                "resource_chars": {
                    "energy": "⚊",
                    "matter": "⚌",
                    "fluid": "⚏",
                    "data": "⚎",
                    "default": "⚊",
                },
            },
        }
        self.chars = style_chars.get(self.style, style_chars[UIStyle.MECHANICAL])
        self.animation_offset = 0.0
        self.last_animation_time = time.time()

    def set_chain(
        self,
        converters: List[Dict[str, Any]],
        connections: List[Tuple[int, int]],
        flow_rates: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> None:
        """Set the current chain configuration.

        Args:
            converters: List of converter info dictionaries
            connections: List of (from_idx, to_idx) connections
            flow_rates: Optional dict of flow rates between converters
        """
        self.converters = converters
        self.connections = connections
        self.flow_rates = flow_rates or {}

        # Update converter states and initialize history if needed
        current_time = time.time()
        for idx, converter in enumerate(converters):
            if idx not in self.converter_states:
                self.converter_states[idx] = {
                    "last_update": current_time,
                    "status_duration": 0.0,
                    "total_uptime": 0.0,
                    "status_changes": [],
                    "peak_efficiency": 0.0,
                    "min_efficiency": 100.0,
                }
            if idx not in self.efficiency_history:
                self.efficiency_history[idx] = []

            # Update state tracking
            state = self.converter_states[idx]
            efficiency = converter.get("efficiency", 0.0)
            state["peak_efficiency"] = max(state["peak_efficiency"], efficiency)
            state["min_efficiency"] = min(state["min_efficiency"], efficiency)

            # Track status changes
            current_status = converter.get("status", "idle")
            if (
                state["status_changes"]
                and state["status_changes"][-1][0] != current_status
            ):
                state["status_changes"].append((current_status, current_time))
                if len(state["status_changes"]) > 10:  # Keep last 10 status changes
                    state["status_changes"] = state["status_changes"][-10:]

            # Update efficiency history
            self.efficiency_history[idx].append(efficiency)
            if len(self.efficiency_history[idx]) > self.max_history_points:
                self.efficiency_history[idx] = self.efficiency_history[idx][
                    -self.max_history_points :
                ]

        # Clear cache to force layout recalculation
        self._cached_layouts.clear()
        self._last_cache_time = current_time

        # Status colors based on style
        status_colors = {
            UIStyle.QUANTUM: {
                "active": (100, 200, 255),  # Cyan
                "error": (255, 100, 100),  # Red
                "idle": (255, 200, 100),  # Orange
                "paused": (200, 100, 255),  # Purple
            },
            UIStyle.SYMBIOTIC: {
                "active": (150, 255, 150),  # Light green
                "error": (255, 150, 150),  # Light red
                "idle": (255, 255, 150),  # Light yellow
                "paused": (150, 150, 255),  # Light blue
            },
            UIStyle.MECHANICAL: {
                "active": (100, 255, 100),  # Green
                "error": (255, 100, 100),  # Red
                "idle": (200, 200, 100),  # Yellow
                "paused": (100, 100, 255),  # Blue
            },
        }
        self.status_colors = status_colors.get(
            self.style, status_colors[UIStyle.MECHANICAL]
        )

    def _get_node_color(
        self, converter: Dict[str, Any], idx: int
    ) -> Tuple[int, int, int]:
        """Get the appropriate color for a converter node based on its status and efficiency."""
        base_color = self.status_colors.get(converter.get("status", ""), COLOR_TEXT)

        # Get state information
        state = self.converter_states.get(idx, {})
        efficiency = converter.get("efficiency", 0.0)
        peak_efficiency = state.get("peak_efficiency", 0.0)

        # Calculate color intensity based on current vs peak efficiency
        if peak_efficiency > 0:
            intensity = min(
                1.0, (efficiency / peak_efficiency) * 1.2
            )  # Allow 20% boost for visual pop
            return tuple(int(c * intensity) for c in base_color)

        return base_color

    def _calculate_animation_speed(self, flow_rate: float) -> float:
        """Calculate animation speed based on flow rate and settings."""
        base_speed = self.animation_settings["base_speed"]
        speed_scale = self.animation_settings["speed_scale"]
        max_speed = self.animation_settings["max_speed"]

        # Scale speed with flow rate, but cap it
        speed = base_speed + (flow_rate * speed_scale)
        return min(speed, max_speed)

    def _get_efficiency_trend(self, idx: int) -> float:
        """Calculate efficiency trend for a converter."""
        history = self.efficiency_history.get(idx, [])
        if len(history) < 2:
            return 0.0

        # Calculate trend over last 5 points or all points if fewer
        window = min(5, len(history))
        recent = history[-window:]
        if not recent:
            return 0.0

        # Simple linear regression for trend
        x = list(range(len(recent)))
        y = recent
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        return numerator / denominator if denominator != 0 else 0.0

    def _draw_node_stats(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        idx: int,
        converter: Dict[str, Any],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw converter statistics below the node."""
        # Get state information
        state = self.converter_states.get(idx, {})
        efficiency_trend = self._get_efficiency_trend(idx)

        # Get tier indicator character based on tier level
        tier_chars = {1: "⚍", 2: "⚎", 3: "⚏", 4: "⚌", 5: "⚋"}
        tier = converter.get("tier", 1)
        tier_char = tier_chars.get(tier, "⚍")

        # Get status indicator
        status_chars = {"active": "▶", "error": "⚠", "idle": "⏸", "paused": "⏹"}
        status = converter.get("status", "idle")
        status_char = status_chars.get(status, "?")

        # Format efficiency with trend indicator
        efficiency = converter.get("efficiency", 0)
        
        # Determine trend character based on efficiency trend
        if efficiency_trend > 0:
            trend_char = "↗"
        elif efficiency_trend < 0:
            trend_char = "↘"
        else:
            trend_char = "→"
        peak_efficiency = state.get("peak_efficiency", 0.0)

        # Calculate utilization and uptime
        utilization = converter.get("utilization", 0)
        uptime = state.get("total_uptime", 0.0)
        uptime_hours = uptime / 3600  # Convert seconds to hours

        # Get color based on utilization
        if utilization > 80:
            util_color = self.status_colors["active"]
        elif utilization > 40:
            util_color = self.status_colors["idle"]
        else:
            util_color = self.status_colors["error"]

        # Format stats with enhanced information
        stats = [
            f"{tier_char} T{tier} {status_char}",
            f"Eff: {efficiency:.1f}% {trend_char} (Peak: {peak_efficiency:.1f}%)",
            f"Util: {utilization:.1f}% (Up: {uptime_hours:.1f}h)",
            f"Rate: {converter.get('rate', 0):.1f}/s",
            f"Queue: {converter.get('queue_size', 0)}",
        ]

        # Draw each stat with appropriate color
        draw_text(
            surface, stats[0], x, y, size=font.get_height(), color=color
        )  # Tier and status
        draw_text(
            surface, stats[1], x, y + 1, size=font.get_height(), color=color
        )  # Efficiency
        draw_text(
            surface, stats[2], x, y + 2, size=font.get_height(), color=util_color
        )  # Utilization
        draw_text(
            surface, stats[3], x, y + 3, size=font.get_height(), color=color
        )  # Rate
        draw_text(
            surface, stats[4], x, y + 4, size=font.get_height(), color=color
        )  # Queue

    def _draw_node(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        idx: int,
        converter: Dict[str, Any],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a converter node with detailed information."""
        # Calculate dimensions
        name_width = len(converter["name"]) + 4
        stats_width = max(
            len("Eff: 100.0% (Peak: 100.0%)"),
            len("Rate: 100.0/s (Max: 100.0/s)"),
            len("Queue: 100 Uptime: 100%"),
        )
        width = max(name_width, stats_width) + 2
        height = 8  # Added a line for extended stats

        # Draw borders
        top_border = f"{self.chars['corner']}{self.chars['node'] * (width - 2)}{self.chars['corner']}"
        draw_text(surface, top_border, x, y, size=font.get_height(), color=color)

        # Draw name
        name_padding = (width - 2 - len(converter["name"])) // 2
        name_line = f"{self.chars['node']}{' ' * name_padding}{converter['name']}{' ' * name_padding}{self.chars['node']}"
        if (width - 2 - len(converter["name"])) % 2 == 1:
            name_line = f"{name_line[:-1]} {name_line[-1]}"
        draw_text(surface, name_line, x, y + 1, size=font.get_height(), color=color)

        # Draw separator
        separator = f"{self.chars['corner']}{self.chars['node'] * (width - 2)}{self.chars['corner']}"
        draw_text(surface, separator, x, y + 2, size=font.get_height(), color=color)

        # Draw stats
        self._draw_node_stats(surface, font, x + 1, y + 3, idx, converter, color)

        # Draw bottom border
        draw_text(
            surface, top_border, x, y + height, size=font.get_height(), color=color
        )

    def _draw_connection(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        flow_rate: Optional[float],
        color: Tuple[int, int, int],
        source_converter: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Draw a connection between nodes with optional flow rate and resource type.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            flow_rate: Optional flow rate between nodes
            color: Color to draw with
            source_converter: Optional source converter info for resource type
        """
        # Calculate connection parameters
        dx = end_x - start_x
        dy = end_y - start_y
        needs_vertical = dy != 0

        # Draw the connection lines
        self._draw_connection_lines(
            surface, font, start_x, start_y, end_x, end_y, dx, dy, needs_vertical, color
        )

        # Draw flow information and animation if provided
        if flow_rate is not None:
            self._draw_flow_information(
                surface,
                font,
                start_x,
                start_y,
                end_x,
                end_y,
                dx,
                dy,
                needs_vertical,
                flow_rate,
                color,
                source_converter,
            )

    def _draw_connection_lines(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        dx: int,
        dy: int,
        needs_vertical: bool,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw the connection lines between nodes.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            dx, dy: Coordinate differences
            needs_vertical: Whether vertical lines are needed
            color: Color to draw with
        """
        if needs_vertical:
            mid_y = start_y + dy // 2
            self._draw_vertical_segments(
                surface, font, start_x, start_y, end_x, end_y, mid_y, color
            )
            self._draw_horizontal_segment(
                surface, font, start_x, end_x, mid_y, dx, needs_vertical, color
            )
        else:
            # Simple horizontal line for same-row connections
            self._draw_horizontal_segment(
                surface, font, start_x, end_x, start_y, dx, needs_vertical, color
            )

    def _draw_vertical_segments(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        mid_y: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw vertical line segments.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            mid_y: Middle Y coordinate for horizontal segment
            color: Color to draw with
        """
        v_line = "|"

        # Draw first vertical segment
        for y in range(start_y + 1, mid_y):
            draw_text(surface, v_line, start_x, y, size=font.get_height(), color=color)

        # Draw second vertical segment
        for y in range(mid_y, end_y):
            draw_text(surface, v_line, end_x, y, size=font.get_height(), color=color)

    def _draw_horizontal_segment(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        start_x: int,
        end_x: int,
        y_pos: int,
        dx: int,
        needs_vertical: bool,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw horizontal line segment.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            start_x, end_x: X coordinates
            y_pos: Y position for the horizontal line
            dx: X-axis difference
            needs_vertical: Whether vertical segments are present
            color: Color to draw with
        """
        h_line = self.chars["flow"] * (abs(dx) - 1)

        # Determine line decorations and position based on direction
        if dx > 0:  # Left to right
            prefix = self.chars["corner"] if needs_vertical else ""
            suffix = self.chars["arrow"]
            x_pos = start_x + (0 if needs_vertical else 1)
        else:  # Right to left
            prefix = self.chars["arrow"]
            suffix = self.chars["corner"] if needs_vertical else ""
            x_pos = end_x + (0 if needs_vertical else 1)

        # Draw the complete horizontal line
        line_text = prefix + h_line + suffix
        draw_text(surface, line_text, x_pos, y_pos, size=font.get_height(), color=color)

    def _draw_flow_information(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        dx: int,
        dy: int,
        needs_vertical: bool,
        flow_rate: float,
        color: Tuple[int, int, int],
        source_converter: Optional[Dict[str, Any]],
    ) -> None:
        """Draw flow rate information and animated flow.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            dx, dy: Coordinate differences
            needs_vertical: Whether vertical lines are needed
            flow_rate: Flow rate between nodes
            color: Color to draw with
            source_converter: Source converter info for resource type
        """
        # Get resource information
        resource_type, resource_char = self._get_resource_info(source_converter)

        # Draw flow rate text
        self._draw_flow_rate_text(
            surface,
            font,
            start_x,
            start_y,
            dx,
            dy,
            needs_vertical,
            flow_rate,
            resource_type,
            resource_char,
            color,
        )

        # Update animation state
        self._update_animation_state(flow_rate)

        # Draw animated flow
        self._draw_animated_flow(
            surface,
            font,
            start_x,
            start_y,
            end_x,
            end_y,
            dx,
            needs_vertical,
            source_converter,
            color,
        )

    def _get_resource_info(
        self, source_converter: Optional[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Get resource type and character.

        Args:
            source_converter: Source converter info

        Returns:
            Tuple[str, str]: Resource type and its visual character
        """
        resource_type = (
            source_converter.get("output_type", "") if source_converter else ""
        )
        resource_char = self.chars["resource_chars"].get(
            resource_type.lower(), self.chars["resource_chars"]["default"]
        )
        return resource_type, resource_char

    def _draw_flow_rate_text(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        start_x: int,
        start_y: int,
        dx: int,
        dy: int,
        needs_vertical: bool,
        flow_rate: float,
        resource_type: str,
        resource_char: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw flow rate text with resource information.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            start_x, start_y: Starting coordinates
            dx, dy: Coordinate differences
            needs_vertical: Whether vertical lines are needed
            flow_rate: Flow rate between nodes
            resource_type: Type of resource
            resource_char: Visual character for the resource
            color: Color to draw with
        """
        # Format flow rate with units and resource indicator
        display_str = self._format_flow_rate(flow_rate, resource_char)

        # Add resource type name if space allows
        if len(display_str) + len(resource_type) + 2 <= abs(dx):
            display_str = f"{resource_type}: {display_str}"

        # Calculate display position
        mid_y = start_y + dy // 2 if needs_vertical else start_y
        rate_x = start_x + (dx - len(display_str)) // 2
        rate_y = mid_y - 1 if needs_vertical else start_y - 1

        # Draw with slightly dimmed color for better readability
        dimmed_color = tuple(max(0, c - 40) for c in color)
        draw_text(
            surface,
            display_str,
            rate_x,
            rate_y,
            size=font.get_height(),
            color=dimmed_color,
        )

    def _format_flow_rate(self, flow_rate: float, resource_char: str) -> str:
        """Format flow rate with appropriate units.

        Args:
            flow_rate: Flow rate value
            resource_char: Visual character for the resource

        Returns:
            str: Formatted flow rate string
        """
        if flow_rate >= 1000:
            return f"{resource_char} {flow_rate/1000:.1f}k/s"
        return f"{resource_char} {flow_rate:.1f}/s"

    def _update_animation_state(self, flow_rate: float) -> None:
        """Update animation timing state.

        Args:
            flow_rate: Flow rate value
        """
        current_time = time.time()
        animation_speed = (
            min(flow_rate / 10.0, 3.0) * 8
        )  # Scale speed with flow rate, capped for readability

        self.animation_offset = (
            self.animation_offset
            + (current_time - self.last_animation_time) * animation_speed
        ) % 1000
        self.last_animation_time = current_time

    def _draw_animated_flow(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        dx: int,
        needs_vertical: bool,
        source_converter: Optional[Dict[str, Any]],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw animated flow between nodes.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            dx: X-axis difference for direction
            needs_vertical: Whether vertical lines are needed
            source_converter: Source converter info containing resource type
            color: Color to draw with
        """
        # Get animation pattern based on source converter info
        pattern, pattern_pos = self._get_animation_pattern(
            source_converter, dx
        )

        # Calculate animation parameters
        h_spacing = len(pattern[0]) if isinstance(pattern[0], str) else 3
        offset = int(self.animation_offset * 3)
        anim_char = pattern[pattern_pos]

        # Draw vertical animation points if needed
        points = []
        if needs_vertical:
            points = self._calculate_vertical_animation_points(
                start_x, start_y, end_x, end_y, offset
            )

        # Draw horizontal animation points
        self._draw_horizontal_animation(
            surface,
            font,
            start_x,
            start_y,
            end_x,
            end_y,
            dx,
            needs_vertical,
            pattern,
            pattern_pos,
            h_spacing,
            offset,
            color,
        )

        # Draw vertical animation points
        for point in points:
            draw_text(
                surface,
                anim_char,
                point[0],
                point[1],
                size=font.get_height(),
                color=color,
            )

    def _get_animation_pattern(
        self, source_converter: Optional[Dict[str, Any]], dx: int
    ) -> Tuple[List[str], int]:
        """Get animation pattern for the resource type.

        Args:
            source_converter: Source converter info containing output_type
            dx: X-axis difference for direction

        Returns:
            Tuple[List[str], int]: Selected pattern and current position
        """
        # Resource-specific animation patterns
        anim_patterns = {
            "energy": [
                "⚊⚋⚌⚍⚎⚏",  # Energy pulse
                "⚏⚎⚍⚌⚋⚊",  # Reverse pulse
            ],
            "matter": [
                "⬡⬢⬣",  # Matter flow
                "⬣⬢⬡",  # Reverse flow
            ],
            "fluid": [
                "⎾⏋⏌⎿",  # Fluid wave
                "⎿⏌⏋⎾",  # Reverse wave
            ],
            "data": [
                "⬒⬓⬔⬕",  # Data stream
                "⬕⬔⬓⬒",  # Reverse stream
            ],
        }

        # Get normalized resource type
        norm_resource_type = (
            source_converter.get("output_type", "").lower() if source_converter else ""
        )

        # Get pattern set for this resource type
        pattern_set = anim_patterns.get(norm_resource_type, [self.chars["anim_chars"]])

        # Select pattern based on flow direction
        # Use second pattern only if flow is leftward and we have multiple patterns
        pattern_idx = 1 if dx <= 0 and len(pattern_set) > 1 else 0
        pattern = pattern_set[pattern_idx]

        # Get current animation position
        pattern_pos = int(self.animation_offset * 4) % len(pattern)

        return pattern, pattern_pos

    def _calculate_vertical_animation_points(
        self, start_x: int, start_y: int, end_x: int, end_y: int, offset: int
    ) -> List[Tuple[int, int]]:
        """Calculate points for vertical animation.

        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            offset: Animation offset

        Returns:
            List[Tuple[int, int]]: List of animation points
        """
        mid_y = start_y + (end_y - start_y) // 2
        v_spacing = 2  # Tighter vertical spacing for better visuals

        # Collect points for both vertical segments
        return [
            (x, y)
            for x, y_range in [
                (start_x, range(start_y + 1, mid_y)),
                (end_x, range(mid_y, end_y)),
            ]
            for y in y_range
            if (y + offset) % v_spacing == 0
        ]

    def _draw_horizontal_animation(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        dx: int,
        needs_vertical: bool,
        pattern: List[str],
        pattern_pos: int,
        h_spacing: int,
        offset: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw horizontal animation points.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            dx: X-axis difference
            needs_vertical: Whether vertical lines are needed
            pattern: Animation pattern to use
            pattern_pos: Current position in pattern
            h_spacing: Horizontal spacing between animation points
            offset: Animation offset
            color: Color to draw with
        """
        # Determine Y position for horizontal animation
        y_pos = start_y + (end_y - start_y) // 2 if needs_vertical else start_y
        h_offset = offset % h_spacing

        # Create flowing pattern effect
        h_points = [
            (start_x + (i * h_spacing) + h_offset, y_pos)
            for i in range(abs(dx) // h_spacing)
            if start_x + (i * h_spacing) + h_offset < end_x - 1
        ]

        # Draw each horizontal animation point with cycling pattern
        for idx, point in enumerate(h_points):
            pattern_char = pattern[(pattern_pos + idx) % len(pattern)]
            draw_text(
                surface,
                pattern_char,
                point[0],
                point[1],
                size=font.get_height(),
                color=color,
            )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the chain visualization.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)

        if not self.converters:
            return self._draw_empty_state(surface, font, panel_rect)

        # Calculate layout parameters
        layout_params = self._calculate_layout_parameters(font)

        # Update animation state
        dt = self._update_animation_time()

        # Position nodes and draw them
        node_positions = self._position_and_draw_nodes(surface, font, layout_params, dt)

        # Draw connections between nodes
        self._draw_node_connections(surface, font, node_positions)

        return panel_rect

    def _draw_empty_state(
        self, surface: pygame.Surface, font: pygame.font.Font, panel_rect: pygame.Rect
    ) -> pygame.Rect:
        """Draw empty state message when no converters are present.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            panel_rect: The panel rectangle

        Returns:
            pygame.Rect: The panel rectangle
        """
        x = self.rect.x + self.rect.width // 4
        y = self.rect.y + self.rect.height // 2
        draw_text(
            surface,
            "No converters in chain",
            x,
            y,
            size=font.get_height(),
            color=COLOR_TEXT,
        )
        return panel_rect

    def _calculate_layout_parameters(self, font: pygame.font.Font) -> dict:
        """Calculate layout parameters for node positioning.

        Args:
            font: Font to use for rendering

        Returns:
            dict: Dictionary of layout parameters
        """
        margin = font.get_height() * 2
        node_spacing = font.get_height() * 4
        start_x = self.rect.x + margin
        start_y = self.rect.y + margin * 2  # Extra margin for title

        # Calculate optimal layout
        total_width = self.rect.width - (margin * 2)
        node_width = max(len(conv["name"]) for conv in self.converters) + 8
        max_nodes_per_row = max(1, total_width // (node_width + 4))

        return {
            "margin": margin,
            "node_spacing": node_spacing,
            "start_x": start_x,
            "start_y": start_y,
            "total_width": total_width,
            "node_width": node_width,
            "max_nodes_per_row": max_nodes_per_row,
        }

    def _update_animation_time(self) -> float:
        """Update animation time and return time delta.

        Returns:
            float: Time delta since last update
        """
        current_time = time.time()
        dt = current_time - self.last_animation_time
        self.last_animation_time = current_time
        return dt

    def _update_converter_state(self, conv: dict, idx: int, dt: float) -> None:
        """Update state tracking for a converter.

        Args:
            conv: Converter data
            idx: Converter index
            dt: Time delta
        """
        state = self.converter_states.setdefault(idx, {})

        # Update uptime if active
        if conv.get("status") == "active":
            state["total_uptime"] = state.get("total_uptime", 0.0) + dt

        # Update efficiency history
        if "efficiency" in conv:
            self._update_efficiency_history(conv, idx, state)

    def _update_efficiency_history(self, conv: dict, idx: int, state: dict) -> None:
        """Update efficiency history for a converter.

        Args:
            conv: Converter data
            idx: Converter index
            state: Converter state dictionary
        """
        history = self.efficiency_history.setdefault(idx, [])
        history.append(conv["efficiency"])

        # Maintain history size limit
        if len(history) > self.max_history_points:
            history.pop(0)

        # Update peak efficiency
        state["peak_efficiency"] = max(
            state.get("peak_efficiency", 0.0), conv["efficiency"]
        )

    def _position_and_draw_nodes(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        layout_params: dict,
        dt: float,
    ) -> dict:
        """Position nodes in a grid layout, update states, and draw them.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            layout_params: Layout parameters
            dt: Time delta

        Returns:
            dict: Dictionary mapping node indices to positions
        """
        node_positions = {}  # idx -> (x, y)

        for i, conv in enumerate(self.converters):
            # Calculate position
            row = i // layout_params["max_nodes_per_row"]
            col = i % layout_params["max_nodes_per_row"]

            x = layout_params["start_x"] + (col * (layout_params["node_width"] + 4))
            y = layout_params["start_y"] + (row * layout_params["node_spacing"])
            node_positions[i] = (x, y)

            # Update state tracking
            self._update_converter_state(conv, i, dt)

            # Draw node with appropriate color
            color = self._get_node_color(conv, i)
            self._draw_node(surface, font, x, y, i, conv, color)

        return node_positions

    def _get_connection_color(
        self, flow_rate: Optional[float], base_color: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Get color for connection based on flow rate.

        Args:
            flow_rate: Flow rate between nodes
            base_color: Base color to use

        Returns:
            Tuple[int, int, int]: Color tuple
        """
        if flow_rate is not None:
            animation_speed = self._calculate_animation_speed(flow_rate)
            alpha = min(255, int(128 + (animation_speed * 127)))
            return tuple(min(255, c * alpha // 255) for c in base_color)

        return base_color

    def _draw_node_connections(
        self, surface: pygame.Surface, font: pygame.font.Font, node_positions: dict
    ) -> None:
        """Draw connections between nodes.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            node_positions: Dictionary mapping node indices to positions
        """
        for from_idx, to_idx in self.connections:
            if from_idx in node_positions and to_idx in node_positions:
                start = node_positions[from_idx]
                end = node_positions[to_idx]
                flow_rate = self.flow_rates.get((from_idx, to_idx))

                # Get style-based color with alpha for active flows
                base_color = {
                    UIStyle.QUANTUM: (100, 200, 255),
                    UIStyle.SYMBIOTIC: (100, 255, 100),
                }.get(self.style, (200, 200, 200))

                # Get color based on flow rate
                color = self._get_connection_color(flow_rate, base_color)

                # Draw connection with animated flow
                self._draw_connection(
                    surface,
                    font,
                    start[0],
                    start[1] + 6,  # Adjusted for node height
                    end[0],
                    end[1] + 6,
                    flow_rate,
                    color,
                    self.converters[from_idx],
                )
