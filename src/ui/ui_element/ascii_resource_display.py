"""
ASCIIResourceDisplay: A panel showing current resources, energy levels, and other critical stats.

This component displays the current state of resources, energy levels, and other
critical game statistics in an ASCII-based UI panel.
"""

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from systems.resource_manager import ResourceType, ResourceState
from typing import Dict, Tuple, Any
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_base.ascii_ui import ASCIIPanel
import pygame

# Type definitions for better type checking
Color = Tuple[int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height

class ASCIIResourceDisplay:
    """A panel showing current resources, energy levels, and other critical stats."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Resources",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a resource display panel.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        
        # Resource tracking
        self.resources: Dict[ResourceType, Dict[str, Any]] = {
            ResourceType.ENERGY: {
                "amount": 0.0,
                "capacity": 100.0,
                "state": ResourceState.STABLE,
                "history": [],
            },
            ResourceType.MATTER: {
                "amount": 0.0,
                "capacity": 100.0,
                "state": ResourceState.STABLE,
                "history": [],
            },
            ResourceType.FLUID: {
                "amount": 0.0,
                "capacity": 100.0,
                "state": ResourceState.STABLE,
                "history": [],
            },
            ResourceType.DATA: {
                "amount": 0.0,
                "capacity": 100.0,
                "state": ResourceState.STABLE,
                "history": [],
            },
        }
        
        # Critical stats
        self.critical_stats: Dict[str, Any] = {
            "shield": 0.0,  # Shield strength percentage
            "hull": 100.0,  # Hull integrity percentage
            "power_output": 0.0,  # Current power output
            "power_usage": 0.0,  # Current power usage
            "efficiency": 0.0,  # System efficiency
        }
        
        self.max_history = 30  # Keep 30 data points of history
        
        # Define colors for different resource types and states
        self.colors = {
            ResourceType.ENERGY: (255, 255, 100),  # Yellow
            ResourceType.MATTER: (100, 255, 100),  # Green
            ResourceType.FLUID: (100, 100, 255),  # Blue
            ResourceType.DATA: (255, 100, 255),  # Purple
            "shield": (100, 200, 255),  # Cyan
            "hull": (200, 100, 100),  # Red
            "power": (255, 200, 100),  # Orange
            "efficiency": (150, 255, 150),  # Light green
            ResourceState.STABLE: (100, 255, 100),  # Green
            ResourceState.DEPLETING: (255, 255, 100),  # Yellow
            ResourceState.GROWING: (100, 255, 200),  # Teal
            ResourceState.CRITICAL: (255, 100, 100),  # Red
        }

    def update_resources(self, resources: Dict[ResourceType, Dict[str, Any]]) -> None:
        """Update resource information.

        Args:
            resources: Dictionary of resource data by type
        """
        try:
            for resource_type, data in resources.items():
                if resource_type in self.resources:
                    # Update current values
                    self.resources[resource_type].update(data)
                    
                    # Update history if amount is provided
                    if "amount" in data:
                        self.resources[resource_type]["history"].append(data["amount"])
                        if len(self.resources[resource_type]["history"]) > self.max_history:
                            self.resources[resource_type]["history"].pop(0)
        except Exception as e:
            logging.error(f"Error updating resources: {e}")

    def update_critical_stats(self, stats: Dict[str, Any]) -> None:
        """Update critical stats information.

        Args:
            stats: Dictionary of critical stats data
        """
        try:
            self.critical_stats.update(stats)
        except Exception as e:
            logging.error(f"Error updating critical stats: {e}")

    def _get_state_indicator(self, state: ResourceState) -> str:
        """Get a text indicator for resource state.

        Args:
            state: Resource state

        Returns:
            str: Text indicator for the state
        """
        indicators = {
            ResourceState.STABLE: "◆",  # Stable
            ResourceState.DEPLETING: "▼",  # Depleting
            ResourceState.GROWING: "▲",  # Growing
            ResourceState.CRITICAL: "!",  # Critical
        }
        return indicators.get(state, "?")

    def _draw_resource_bar(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
        resource_type: ResourceType,
    ) -> int:
        """Draw a resource bar with label, value, and state indicator.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            width: Width of the bar
            resource_type: Type of resource to display

        Returns:
            int: Height of the drawn element
        """
        try:
            resource = self.resources[resource_type]
            amount = resource["amount"]
            capacity = resource["capacity"]
            state = resource["state"]
            
            # Calculate percentage and bar width
            percentage = min(100.0, (amount / capacity) * 100.0) if capacity > 0 else 0
            bar_width = int((width - 20) * (percentage / 100.0))
            
            # Get resource name and color
            resource_name = resource_type.name.capitalize()
            color = self.colors[resource_type]
            state_color = self.colors[state]
            
            # Draw resource name
            label = f"{resource_name}: "
            label_width = font.size(label)[0]
            pygame.draw.rect(surface, (30, 30, 30), (x, y, width, font.get_height() + 4))
            surface.blit(font.render(label, True, color), (x, y))
            
            # Draw bar background
            bar_bg_rect = pygame.Rect(x + label_width, y, width - label_width - 40, font.get_height())
            pygame.draw.rect(surface, (50, 50, 50), bar_bg_rect)
            
            # Draw bar fill
            if bar_width > 0:
                bar_fill_rect = pygame.Rect(x + label_width, y, bar_width, font.get_height())
                pygame.draw.rect(surface, color, bar_fill_rect)
            
            # Draw percentage and state indicator
            value_text = f"{percentage:.1f}% {self._get_state_indicator(state)}"
            value_pos = (x + width - font.size(value_text)[0] - 5, y)
            surface.blit(font.render(value_text, True, state_color), value_pos)
            
            return font.get_height() + 4
        except Exception as e:
            logging.error(f"Error drawing resource bar: {e}")
            return font.get_height() + 4

    def _draw_stat_value(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
        label: str,
        value: float,
        unit: str = "",
        color_key: str = "efficiency",
    ) -> int:
        """Draw a stat value with label.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            width: Width of the element
            label: Label for the stat
            value: Value to display
            unit: Unit string (e.g., "%", "kW")
            color_key: Key for color lookup

        Returns:
            int: Height of the drawn element
        """
        try:
            # Format text
            text = f"{label}: {value:.1f}{unit}"
            color = self.colors.get(color_key, COLOR_TEXT)
            
            # Draw background
            pygame.draw.rect(surface, (30, 30, 30), (x, y, width, font.get_height() + 2))
            
            # Draw text
            surface.blit(font.render(text, True, color), (x, y))
            
            return font.get_height() + 2
        except Exception as e:
            logging.error(f"Error drawing stat value: {e}")
            return font.get_height() + 2

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the resource display panel.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        try:
            # Draw panel background and border
            panel = ASCIIPanel(
                self.rect.x, self.rect.y, self.rect.width, self.rect.height, 
                self.style, self.title
            )
            panel_rect = panel.draw(surface, font)
            
            # Calculate layout
            margin = font.get_height() // 2
            x = self.rect.x + margin
            y = self.rect.y + margin * 3  # Extra margin for title
            width = self.rect.width - margin * 2
            
            # Draw resource bars
            y += self._draw_resource_bar(surface, font, x, y, width, ResourceType.ENERGY)
            y += margin
            y += self._draw_resource_bar(surface, font, x, y, width, ResourceType.MATTER)
            y += margin
            y += self._draw_resource_bar(surface, font, x, y, width, ResourceType.FLUID)
            y += margin
            y += self._draw_resource_bar(surface, font, x, y, width, ResourceType.DATA)
            
            # Draw separator
            y += margin
            pygame.draw.line(surface, (100, 100, 100), (x, y), (x + width, y))
            y += margin
            
            # Draw critical stats
            y += self._draw_stat_value(
                surface, font, x, y, width // 2 - margin,
                "Shield", self.critical_stats["shield"], "%", "shield"
            )
            y += margin
            y += self._draw_stat_value(
                surface, font, x, y, width // 2 - margin,
                "Hull", self.critical_stats["hull"], "%", "hull"
            )
            
            # Draw power stats on the right side
            power_x = x + width // 2 + margin
            power_y = y - (margin + font.get_height() + 2) * 2
            self._draw_stat_value(
                surface, font, power_x, power_y, width // 2 - margin,
                "Output", self.critical_stats["power_output"], "kW", "power"
            )
            power_y += margin + font.get_height() + 2
            self._draw_stat_value(
                surface, font, power_x, power_y, width // 2 - margin,
                "Usage", self.critical_stats["power_usage"], "kW", "power"
            )
            
            # Draw efficiency at the bottom
            y += margin * 2
            self._draw_stat_value(
                surface, font, x, y, width,
                "System Efficiency", self.critical_stats["efficiency"], "%", "efficiency"
            )
            
            return panel_rect
        except Exception as e:
            logging.error(f"Error drawing resource display: {e}")
            return self.rect
