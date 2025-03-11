import contextlib
import random
import curses
from typing import List
from ui.ui_base.ui_element import UIElement
from ui.ui_element.ui_style import UIStyle

# Define standard colors for UI components
COLOR_TEXT = (220, 220, 220)  # Standard text color
COLOR_BG = (20, 20, 30)  # Standard background color
COLOR_HIGHLIGHT = (180, 180, 255)  # Standard highlight color


class AsteroidFieldVisualizer(UIElement):
    """Cellular automaton-based asteroid field visualizer"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        density: float = 0.3,
        style: UIStyle = UIStyle.ASTEROID,
    ):
        super().__init__(x, y, width, height, style)
        self.density = density
        self.automaton_grid = self._initialize_grid()
        self.generation = 0
        self.asteroid_chars = ["·", "∙", "•", "◦", "○", "◌", "◍", "◎", "●", "@"]

    def _initialize_grid(self) -> List[List[float]]:
        """Initialize the asteroid field grid with random values"""
        grid = []
        for _ in range(self.height - 2):
            row = []
            for _ in range(self.width - 2):
                # Random value representing asteroid density/size
                value = random.random() if random.random() < self.density else 0
                row.append(value)
            grid.append(row)
        return grid

    def _evolve_grid(self):  # sourcery skip: low-code-quality
        """Evolve the asteroid field using cellular automaton rules"""
        new_grid = [[0 for _ in range(self.width - 2)] for _ in range(self.height - 2)]

        for y in range(self.height - 2):
            for x in range(self.width - 2):
                # Calculate the average of neighboring cells
                total = 0
                count = 0

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width - 2 and 0 <= ny < self.height - 2:
                            total += self.automaton_grid[ny][nx]
                            count += 1

                avg = total / count if count > 0 else 0

                # Evolution rules for asteroids (forming clusters)
                current = self.automaton_grid[y][x]
                if current > 0:
                    # Existing asteroids evolve based on neighbors
                    if avg < 0.2:  # Isolated asteroids shrink
                        new_grid[y][x] = max(0, current - 0.1)
                    elif avg > 0.6:  # Dense clusters grow
                        new_grid[y][x] = min(1.0, current + 0.05)
                    else:  # Stable clusters stay similar
                        new_grid[y][x] = current + (random.random() - 0.5) * 0.1
                        new_grid[y][x] = max(0, min(1.0, new_grid[y][x]))
                elif avg > 0.5 and random.random() < 0.1:
                    new_grid[y][x] = random.random() * 0.3
                else:
                    new_grid[y][x] = 0

        self.automaton_grid = new_grid
        self.generation += 1

    def draw(self, stdscr):
        """Draw the asteroid field visualization"""
        super().draw(stdscr)

        # Draw title
        with contextlib.suppress(curses.error):
            stdscr.addstr(
                self.y + 1,
                self.x + 2,
                f"ASTEROID FIELD - GEN {self.generation}",
                curses.A_BOLD,
            )
        # Draw the asteroid grid
        for y in range(self.height - 2):
            for x in range(self.width - 2):
                value = self.automaton_grid[y][x]
                if value > 0:
                    # Convert value to asteroid character
                    char_index = min(
                        len(self.asteroid_chars) - 1,
                        int(value * len(self.asteroid_chars)),
                    )
                    char = self.asteroid_chars[char_index]

                    with contextlib.suppress(curses.error):
                        stdscr.addstr(self.y + y + 2, self.x + x + 1, char)
        # Occasionally evolve the grid
        if random.random() < 0.1:
            self._evolve_grid()
