import contextlib
import time
import random
import curses
import math
import logging
import pygame
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Tuple
from ui.ui_element.ui_style import UIStyle
from ui.ui_element.ui_element import UIElement


# Define standard colors for UI components
COLOR_TEXT = (220, 220, 220)  # Standard text color
COLOR_BG = (20, 20, 30)  # Standard background color
COLOR_HIGHLIGHT = (180, 180, 255)  # Standard highlight color


class Menu(UIElement):
    """Interactive menu with dynamically evolving options"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str,
        options: List[str],
        style: UIStyle = UIStyle.SYMBIOTIC,
    ):
        super().__init__(x, y, width, height, style)
        self.title = title
        self.options = options
        self.selected_index = 0
        self.active = False
        # Evolutionary metrics for each option (how often used)
        self.option_metrics = [0] * len(options)
        # Mutation rate for menu evolution
        self.mutation_rate = 0.05

    def draw(self, stdscr):
        """Draw menu with options"""
        super().draw(stdscr)

        # Draw title
        title_x = self.x + (self.width - len(self.title)) // 2
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 1, title_x, self.title, curses.A_BOLD)
        # Draw options with evolutionary prominence
        for i, option in enumerate(self.options):
            # Option appearance changes based on usage patterns
            prefix = "  "
            if i == self.selected_index and self.active:
                prefix = "▶ "  # Selection indicator
                with contextlib.suppress(curses.error):
                    # Highlighted option
                    stdscr.addstr(
                        self.y + 3 + i, self.x + 2, prefix + option, curses.A_REVERSE
                    )
            else:
                # Normal option with prominence-based rendering
                attr = curses.A_NORMAL
                # Calculate prominence factor based on usage
                prominence = min(1.0, 0.5 + self.option_metrics[i] * 0.1)

                if prominence > 0.8:
                    attr = curses.A_BOLD

                # Apply adaptive decoration based on usage
                evolving_chars = ["·", "∘", "○", "◌", "◍", "◎", "●"]
                decoration = evolving_chars[
                    min(len(evolving_chars) - 1, int(prominence * len(evolving_chars)))
                ]

                prefix = f"{decoration} "

                with contextlib.suppress(curses.error):
                    stdscr.addstr(self.y + 3 + i, self.x + 2, prefix + option, attr)

    def handle_input(self, key: int) -> Optional[str]:
        """Handle navigation and selection input"""
        if not self.active:
            return None

        if key == curses.KEY_UP:
            self.selected_index = (self.selected_index - 1) % len(self.options)
        elif key == curses.KEY_DOWN:
            self.selected_index = (self.selected_index + 1) % len(self.options)
        elif key in [curses.KEY_ENTER, ord("\n"), ord(" ")]:
            # Record the selection in metrics
            self.option_metrics[self.selected_index] += 1
            self._evolve_menu()
            return self.options[self.selected_index]

        return None

    def _evolve_menu(self):
        """Simulate evolution of the menu based on usage patterns"""
        # Reorder options based on usage, but with randomness
        if random.random() < self.mutation_rate:
            # Sometimes shuffle the least used options
            least_used = sorted(
                range(len(self.option_metrics)), key=lambda i: self.option_metrics[i]
            )[:2]
            if len(least_used) >= 2:
                i, j = least_used[0], least_used[1]
                self.options[i], self.options[j] = self.options[j], self.options[i]
                self.option_metrics[i], self.option_metrics[j] = (
                    self.option_metrics[j],
                    self.option_metrics[i],
                )
