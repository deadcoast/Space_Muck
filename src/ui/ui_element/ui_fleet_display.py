"""
ui_fleet_display.py
"""

# Standard library imports
import contextlib
import curses
from typing import Any, Dict, List

# Local application imports
from src.ui.ui_base.ascii_base import UIElement, UIStyle

# Third-party library imports

# Import config constants if needed in the future

# Ship pattern constants
CAPITAL_SHIP_PATTERN = "<[≡≡≡≡≡]>"


class FleetDisplay(UIElement):
    """Displays fleet information with cellular automaton-based animations"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        fleet_data: Dict[str, Any],
        style: UIStyle = UIStyle.FLEET,
    ):
        super().__init__(x, y, width, height, style)
        self.fleet_data = fleet_data
        self.animation_frame = 0
        self.ship_patterns = self._generate_ship_patterns()

    def _generate_ship_patterns(self) -> Dict[str, List[str]]:
        """Generate ASCII art patterns for different ship types"""
        return {
            "miner": [
                "⤧≡≡≡◊",
                "⤧≡≡◊≡",
                "⤧≡◊≡≡",
                "⤧◊≡≡≡",
            ],
            "fighter": [
                "/=|=\\",
                "/≡|≡\\",
                "/≡∫≡\\",
                "/≡|≡\\",
            ],
            "capital": [
                self.CAPITAL_SHIP_PATTERN,
                self.CAPITAL_SHIP_PATTERN,
                self.CAPITAL_SHIP_PATTERN,
                self.CAPITAL_SHIP_PATTERN,
            ],
        }

    def draw(self, surface, font=None):
        """Draw the fleet display with animated ships

        Args:
            surface: The surface to draw on
            font: Optional font to use for rendering (not used in curses mode)
        """
        super().draw(surface, font)

        # Draw title
        with contextlib.suppress(curses.error):
            surface.addstr(self.y + 1, self.x + 2, "FLEET STATUS", curses.A_BOLD)
        # Draw ships with animation
        y_offset = 3
        for ship_type, count in self.fleet_data.items():
            if ship_type in self.ship_patterns:
                pattern = self.ship_patterns[ship_type][
                    self.animation_frame % len(self.ship_patterns[ship_type])
                ]

                with contextlib.suppress(curses.error):
                    # Ship type and count
                    surface.addstr(
                        self.y + y_offset,
                        self.x + 2,
                        f"{ship_type.capitalize()}: {count}",
                    )

                    # Animated ships (show up to 5)
                    for i in range(min(count, 5)):
                        ship_x = self.x + 18 + i * (len(pattern) + 1)
                        if ship_x + len(pattern) < self.x + self.width - 1:
                            surface.addstr(self.y + y_offset, ship_x, pattern)
                y_offset += 2

        # Update animation frame
        self.animation_frame += 1

    def update_fleet(self, new_fleet_data: Dict[str, Any]):
        """Update the fleet information"""
        self.fleet_data = new_fleet_data
