# Define standard colors for UI components
COLOR_TEXT = (220, 220, 220)  # Standard text color
COLOR_BG = (20, 20, 30)  # Standard background color
COLOR_HIGHLIGHT = (180, 180, 255)  # Standard highlight color


class FleetDisplay(UIElement):
    """Displays fleet information with cellular automaton-based animations"""


# Standard library imports

# Third-party library imports

# Local application imports
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_base.ascii_base import UIElement
from typing import List, Dict, Any
import contextlib
import curses


# Ship pattern constants
CAPITAL_SHIP_PATTERN = "<[≡≡≡≡≡]>"


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


def draw(self, stdscr, font=None):
    """Draw the fleet display with animated ships

    Args:
        stdscr: The curses screen to draw on
        font: Optional font to use for rendering (not used in curses mode)
    """
    super().draw(stdscr, font)

    # Draw title
    with contextlib.suppress(curses.error):
        stdscr.addstr(self.y + 1, self.x + 2, "FLEET STATUS", curses.A_BOLD)
    # Draw ships with animation
    y_offset = 3
    for ship_type, count in self.fleet_data.items():
        if ship_type in self.ship_patterns:
            pattern = self.ship_patterns[ship_type][
                self.animation_frame % len(self.ship_patterns[ship_type])
            ]

            with contextlib.suppress(curses.error):
                # Ship type and count
                stdscr.addstr(
                    self.y + y_offset,
                    self.x + 2,
                    f"{ship_type.capitalize()}: {count}",
                )

                # Animated ships (show up to 5)
                for i in range(min(count, 5)):
                    ship_x = self.x + 18 + i * (len(pattern) + 1)
                    if ship_x + len(pattern) < self.x + self.width - 1:
                        stdscr.addstr(self.y + y_offset, ship_x, pattern)
            y_offset += 2

    # Update animation frame
    self.animation_frame += 1


def update_fleet(self, new_fleet_data: Dict[str, Any]):
    """Update the fleet information"""
    self.fleet_data = new_fleet_data
