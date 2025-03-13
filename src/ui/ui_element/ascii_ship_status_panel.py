"""
ASCIIShipStatusPanel: Show ship health, energy, and system statuses.

This component provides a user interface for displaying the ship's health, energy levels,
and the status of various ship systems.
"""

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from typing import Dict, Tuple, Any
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_base.ascii_ui import ASCIIPanel
import pygame

# Type definitions for better type checking
Color = Tuple[int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height

class ASCIIShipStatusPanel:
    """Display ship health, energy, and system statuses."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Ship Status",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a ship status panel.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        
        # Ship status data
        self.ship_level = 1
        self.ship_name = "Unknown"
        
        # Hull and shield data
        self.max_hull = 100
        self.current_hull = 100
        self.hull_integrity = 100  # Percentage
        
        self.max_shield = 100
        self.current_shield = 100
        self.shield_integrity = 100  # Percentage
        self.shield_recharge_rate = 1.0
        
        # Combat stats
        self.attack_power = 10
        self.attack_speed = 1.0
        self.weapon_range = 100
        self.crit_chance = 5.0
        self.evasion = 10.0
        self.armor = 5.0
        
        # System statuses
        self.systems = {
            "engines": {
                "status": "online",  # online, damaged, offline
                "efficiency": 100,   # percentage
                "power_usage": 10,   # units
            },
            "weapons": {
                "status": "online",
                "efficiency": 100,
                "power_usage": 15,
            },
            "shields": {
                "status": "online",
                "efficiency": 100,
                "power_usage": 20,
            },
            "sensors": {
                "status": "online",
                "efficiency": 100,
                "power_usage": 5,
            },
            "life_support": {
                "status": "online",
                "efficiency": 100,
                "power_usage": 10,
            }
        }
        
        # Power data
        self.max_power = 100
        self.current_power = 60
        self.power_generation = 70
        self.power_usage = 60
        
        # Colors for different status types
        self.status_colors = {
            "online": (100, 255, 100),       # Green
            "damaged": (255, 255, 100),      # Yellow
            "offline": (255, 100, 100),      # Red
            "critical": (255, 50, 50),       # Bright red
            "warning": (255, 200, 50),       # Orange
            "good": (50, 255, 100),          # Green
            "neutral": (200, 200, 200),      # Light gray
        }

    def update_ship_info(self, ship_level: int, ship_name: str) -> None:
        """Update basic ship information.

        Args:
            ship_level: Current ship level
            ship_name: Ship name
        """
        try:
            self.ship_level = ship_level
            self.ship_name = ship_name
        except Exception as e:
            logging.error(f"Error updating ship info: {e}")

    def update_hull_shield(
        self, 
        current_hull: float, 
        max_hull: float, 
        current_shield: float, 
        max_shield: float,
        shield_recharge_rate: float = None
    ) -> None:
        """Update hull and shield status.

        Args:
            current_hull: Current hull points
            max_hull: Maximum hull points
            current_shield: Current shield points
            max_shield: Maximum shield points
            shield_recharge_rate: Optional shield recharge rate
        """
        try:
            self.current_hull = current_hull
            self.max_hull = max_hull
            self.hull_integrity = int((current_hull / max_hull) * 100) if max_hull > 0 else 0
            
            self.current_shield = current_shield
            self.max_shield = max_shield
            self.shield_integrity = int((current_shield / max_shield) * 100) if max_shield > 0 else 0
            
            if shield_recharge_rate is not None:
                self.shield_recharge_rate = shield_recharge_rate
        except Exception as e:
            logging.error(f"Error updating hull and shield: {e}")

    def update_combat_stats(
        self,
        attack_power: float = None,
        attack_speed: float = None,
        weapon_range: float = None,
        crit_chance: float = None,
        evasion: float = None,
        armor: float = None
    ) -> None:
        """Update combat statistics.

        Args:
            attack_power: Attack power value
            attack_speed: Attacks per time unit
            weapon_range: Maximum attack distance
            crit_chance: Critical hit chance percentage
            evasion: Evasion chance percentage
            armor: Damage reduction percentage
        """
        try:
            if attack_power is not None:
                self.attack_power = attack_power
            if attack_speed is not None:
                self.attack_speed = attack_speed
            if weapon_range is not None:
                self.weapon_range = weapon_range
            if crit_chance is not None:
                self.crit_chance = crit_chance
            if evasion is not None:
                self.evasion = evasion
            if armor is not None:
                self.armor = armor
        except Exception as e:
            logging.error(f"Error updating combat stats: {e}")

    def update_system_status(self, system_name: str, status: Dict[str, Any]) -> None:
        """Update the status of a specific ship system.

        Args:
            system_name: Name of the system to update
            status: Dictionary with status information
        """
        try:
            if system_name in self.systems:
                self.systems[system_name].update(status)
        except Exception as e:
            logging.error(f"Error updating system status: {e}")

    def update_power(
        self, 
        current_power: float, 
        max_power: float, 
        power_generation: float, 
        power_usage: float
    ) -> None:
        """Update power information.

        Args:
            current_power: Current power level
            max_power: Maximum power capacity
            power_generation: Power generation rate
            power_usage: Power consumption rate
        """
        try:
            self.current_power = current_power
            self.max_power = max_power
            self.power_generation = power_generation
            self.power_usage = power_usage
        except Exception as e:
            logging.error(f"Error updating power info: {e}")

    def _get_status_color(self, status: str) -> Color:
        """Get the display color for a status.

        Args:
            status: Status string

        Returns:
            RGB color tuple
        """
        return self.status_colors.get(status, COLOR_TEXT)

    def _get_percentage_color(self, percentage: float) -> Color:
        """Get color based on percentage value.

        Args:
            percentage: Percentage value (0-100)

        Returns:
            RGB color tuple
        """
        if percentage < 25:
            return self.status_colors["critical"]
        elif percentage < 50:
            return self.status_colors["warning"]
        elif percentage < 75:
            return self.status_colors["neutral"]
        else:
            return self.status_colors["good"]

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the ship status panel.

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
            content_x = self.rect.x + margin
            content_y = self.rect.y + margin * 3  # Extra margin for title
            content_width = self.rect.width - margin * 2
            
            # Draw ship info
            ship_info = f"{self.ship_name} (Level {self.ship_level})"
            surface.blit(
                font.render(ship_info, True, COLOR_HIGHLIGHT),
                (content_x, content_y)
            )
            content_y += font.get_height() + margin
            
            # Draw hull and shield bars
            self._draw_status_bar(
                surface, font, content_x, content_y, content_width,
                "Hull", self.hull_integrity, self._get_percentage_color(self.hull_integrity)
            )
            content_y += font.get_height() + margin // 2
            
            self._draw_status_bar(
                surface, font, content_x, content_y, content_width,
                "Shield", self.shield_integrity, self._get_percentage_color(self.shield_integrity)
            )
            content_y += font.get_height() + margin // 2
            
            # Add shield recharge info
            recharge_text = f"Shield Recharge: {self.shield_recharge_rate:.1f}/s"
            surface.blit(
                font.render(recharge_text, True, COLOR_TEXT),
                (content_x + content_width - font.size(recharge_text)[0], content_y)
            )
            content_y += font.get_height() + margin
            
            # Draw power section
            power_header = "Power Systems"
            surface.blit(
                font.render(power_header, True, COLOR_HIGHLIGHT),
                (content_x, content_y)
            )
            content_y += font.get_height() + margin // 2
            
            # Power bar
            power_percentage = int((self.current_power / self.max_power) * 100) if self.max_power > 0 else 0
            self._draw_status_bar(
                surface, font, content_x, content_y, content_width,
                "Power", power_percentage, self._get_percentage_color(power_percentage)
            )
            content_y += font.get_height() + margin // 2
            
            # Power generation/usage
            power_balance = self.power_generation - self.power_usage
            balance_color = (
                self.status_colors["good"] if power_balance >= 0 
                else self.status_colors["warning"]
            )
            
            power_text = f"Generation: {self.power_generation}  Usage: {self.power_usage}  "
            power_text += f"Balance: {power_balance:+.1f}"
            
            surface.blit(
                font.render(power_text, True, balance_color),
                (content_x, content_y)
            )
            content_y += font.get_height() + margin
            
            # Draw systems section
            systems_header = "Ship Systems"
            surface.blit(
                font.render(systems_header, True, COLOR_HIGHLIGHT),
                (content_x, content_y)
            )
            content_y += font.get_height() + margin // 2
            
            # Calculate column widths
            col1_width = content_width * 0.3  # System name
            col2_width = content_width * 0.2  # Status
            col3_width = content_width * 0.25  # Efficiency
            # Last column (Power usage) takes remaining width
            
            # Draw system headers
            surface.blit(
                font.render("System", True, COLOR_TEXT),
                (content_x, content_y)
            )
            
            surface.blit(
                font.render("Status", True, COLOR_TEXT),
                (content_x + col1_width, content_y)
            )
            
            surface.blit(
                font.render("Efficiency", True, COLOR_TEXT),
                (content_x + col1_width + col2_width, content_y)
            )
            
            surface.blit(
                font.render("Power", True, COLOR_TEXT),
                (content_x + col1_width + col2_width + col3_width, content_y)
            )
            
            content_y += font.get_height() + margin // 2
            
            # Draw each system
            for system_name, system_data in self.systems.items():
                # System name (capitalized)
                display_name = system_name.replace('_', ' ').title()
                surface.blit(
                    font.render(display_name, True, COLOR_TEXT),
                    (content_x, content_y)
                )
                
                # Status
                status = system_data.get("status", "unknown")
                status_color = self._get_status_color(status)
                surface.blit(
                    font.render(status.title(), True, status_color),
                    (content_x + col1_width, content_y)
                )
                
                # Efficiency
                efficiency = system_data.get("efficiency", 0)
                efficiency_text = f"{efficiency}%"
                efficiency_color = self._get_percentage_color(efficiency)
                surface.blit(
                    font.render(efficiency_text, True, efficiency_color),
                    (content_x + col1_width + col2_width, content_y)
                )
                
                # Power usage
                power_usage = system_data.get("power_usage", 0)
                power_text = f"{power_usage} units"
                surface.blit(
                    font.render(power_text, True, COLOR_TEXT),
                    (content_x + col1_width + col2_width + col3_width, content_y)
                )
                
                content_y += font.get_height() + margin // 2
            
            # Draw combat stats section if there's room
            if content_y + font.get_height() * 4 < self.rect.bottom - margin:
                content_y += margin
                combat_header = "Combat Statistics"
                surface.blit(
                    font.render(combat_header, True, COLOR_HIGHLIGHT),
                    (content_x, content_y)
                )
                content_y += font.get_height() + margin // 2
                
                # First row of combat stats
                attack_text = f"Attack: {self.attack_power}"
                speed_text = f"Speed: {self.attack_speed:.1f}"
                range_text = f"Range: {self.weapon_range}"
                
                surface.blit(
                    font.render(attack_text, True, COLOR_TEXT),
                    (content_x, content_y)
                )
                
                surface.blit(
                    font.render(speed_text, True, COLOR_TEXT),
                    (content_x + content_width // 3, content_y)
                )
                
                surface.blit(
                    font.render(range_text, True, COLOR_TEXT),
                    (content_x + 2 * content_width // 3, content_y)
                )
                
                content_y += font.get_height() + margin // 2
                
                # Second row of combat stats
                crit_text = f"Crit: {self.crit_chance:.1f}%"
                evasion_text = f"Evasion: {self.evasion:.1f}%"
                armor_text = f"Armor: {self.armor:.1f}%"
                
                surface.blit(
                    font.render(crit_text, True, COLOR_TEXT),
                    (content_x, content_y)
                )
                
                surface.blit(
                    font.render(evasion_text, True, COLOR_TEXT),
                    (content_x + content_width // 3, content_y)
                )
                
                surface.blit(
                    font.render(armor_text, True, COLOR_TEXT),
                    (content_x + 2 * content_width // 3, content_y)
                )
            
            return panel_rect
        except Exception as e:
            logging.error(f"Error drawing ship status panel: {e}")
            return self.rect

    def _draw_status_bar(
        self, 
        surface: pygame.Surface, 
        font: pygame.font.Font, 
        x: int, 
        y: int, 
        width: int,
        label: str,
        percentage: float,
        color: Color
    ) -> None:
        """Draw a status bar with label and percentage.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            width: Width of the bar
            label: Bar label
            percentage: Percentage value (0-100)
            color: Bar color
        """
        try:
            # Draw label
            label_width = font.size(label)[0]
            surface.blit(font.render(label, True, COLOR_TEXT), (x, y))
            
            # Draw percentage
            percentage_text = f"{percentage}%"
            percentage_width = font.size(percentage_text)[0]
            
            # Calculate bar dimensions
            bar_x = x + label_width + 10
            bar_width = width - label_width - percentage_width - 20
            bar_height = font.get_height()
            
            # Draw bar background
            bar_bg_rect = pygame.Rect(bar_x, y, bar_width, bar_height)
            pygame.draw.rect(surface, (50, 50, 50), bar_bg_rect)
            
            # Draw bar fill
            if percentage > 0:
                fill_width = int(bar_width * (percentage / 100.0))
                bar_fill_rect = pygame.Rect(bar_x, y, fill_width, bar_height)
                pygame.draw.rect(surface, color, bar_fill_rect)
            
            # Draw percentage text
            surface.blit(
                font.render(percentage_text, True, color),
                (x + width - percentage_width, y)
            )
        except Exception as e:
            logging.error(f"Error drawing status bar: {e}")
