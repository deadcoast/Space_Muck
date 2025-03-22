"""
ui_mining_status.py

Provides a mining status UI element with cellular automaton-inspired behaviors.
"""

# Third-party library imports
import contextlib
import curses

# Standard library imports
from typing import Dict

# Local application imports
from ui.ui_base.ascii_base import UIElement, UIStyle


class MiningStatus(UIElement):
    """Display for resource mining operations and statistics"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        resources: Dict[str, float],
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        super().__init__(x, y, width, height, style)
        self.resources = resources
        self.extraction_rate = {k: 0.0 for k in resources}
        self.efficiency = 0.85
        self.symbols = {
            "Iron": "Fe",
            "Copper": "Cu",
            "Gold": "Au",
            "Titanium": "Ti",
            "Quantum": "Qm",
        }

    def draw(self, stdscr):
        """Draw the mining status display"""
        super().draw(stdscr)

        # Draw title
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 1, self.x + 2, "MINING OPERATIONS", curses.A_BOLD)
        # Draw efficiency meter
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 3, self.x + 2, f"Efficiency: {self.efficiency:.0%}")
            meter_width = self.width - 6
            filled_width = int(meter_width * self.efficiency)

            meter = "▓" * filled_width + "░" * (meter_width - filled_width)
            stdscr.addstr(self.y + 4, self.x + 2, meter)
        # Draw resources and extraction rates
        y_offset = 6
        for resource, amount in self.resources.items():
            symbol = self.symbols.get(resource, resource[:2])

            with contextlib.suppress(curses.error):
                # Resource name and amount
                stdscr.addstr(self.y + y_offset, self.x + 2, f"{resource} ({symbol}):")
                stdscr.addstr(self.y + y_offset, self.x + 16, f"{amount:.1f}")

                # Extraction rate with arrow indicator
                rate = self.extraction_rate[resource]
                rate_str = f"{rate:+.2f}/s"

                # Determine direction indicator based on rate
                if rate > 0:
                    direction = "↑"
                elif rate < 0:
                    direction = "↓"
                else:
                    direction = "→"

                if rate > 0:
                    stdscr.addstr(
                        self.y + y_offset,
                        self.x + 24,
                        f"{direction} {rate_str}",
                        curses.A_BOLD,
                    )
                else:
                    stdscr.addstr(
                        self.y + y_offset, self.x + 24, f"{direction} {rate_str}"
                    )
            y_offset += 2

    def update_resources(
        self,
        resources: Dict[str, float],
        extraction_rate: Dict[str, float],
        efficiency: float,
    ):
        """Update resource information"""
        self.resources = resources
        self.extraction_rate = extraction_rate
        self.efficiency = efficiency
