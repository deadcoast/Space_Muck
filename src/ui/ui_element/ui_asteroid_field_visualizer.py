# Standard library imports
import random

# Third-party library imports

# Local application imports
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.ui_base.ascii_base import UIElement
from typing import List
import contextlib
import curses

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

    def _calculate_cell_average(self, x: int, y: int) -> float:
        """Calculate the average value of neighboring cells.

        Args:
            x: X coordinate of the cell
            y: Y coordinate of the cell

        Returns:
            float: Average value of neighboring cells
        """
        total = 0
        count = 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width - 2 and 0 <= ny < self.height - 2:
                    total += self.automaton_grid[ny][nx]
                    count += 1

        return total / count if count > 0 else 0

    def _evolve_existing_asteroid(
        self, x: int, y: int, avg: float, new_grid: list
    ) -> None:
        """Evolve an existing asteroid cell based on neighborhood average.

        Args:
            x: X coordinate of the cell
            y: Y coordinate of the cell
            avg: Average value of neighboring cells
            new_grid: Grid to update with new values
        """
        current = self.automaton_grid[y][x]

        if avg < 0.2:  # Isolated asteroids shrink
            new_grid[y][x] = max(0, current - 0.1)
        elif avg > 0.6:  # Dense clusters grow
            new_grid[y][x] = min(1.0, current + 0.05)
        else:  # Stable clusters stay similar
            variation = (random.random() - 0.5) * 0.1
            new_value = current + variation
            new_grid[y][x] = max(0, min(1.0, new_value))

    def _evolve_empty_cell(self, x: int, y: int, avg: float, new_grid: list) -> None:
        """Evolve an empty cell based on neighborhood average.

        Args:
            x: X coordinate of the cell
            y: Y coordinate of the cell
            avg: Average value of neighboring cells
            new_grid: Grid to update with new values
        """
        # Chance to spawn new asteroid if surrounded by others
        if avg > 0.5 and random.random() < 0.1:
            new_grid[y][x] = random.random() * 0.3
        else:
            new_grid[y][x] = 0

    def _evolve_grid(self):
        """Evolve the asteroid field using cellular automaton rules"""
        new_grid = [[0 for _ in range(self.width - 2)] for _ in range(self.height - 2)]

        for y in range(self.height - 2):
            for x in range(self.width - 2):
                # Calculate the average of neighboring cells
                avg = self._calculate_cell_average(x, y)

                # Evolution rules for asteroids (forming clusters)
                current = self.automaton_grid[y][x]
                if current > 0:
                    # Evolve existing asteroid
                    self._evolve_existing_asteroid(x, y, avg, new_grid)
                else:
                    # Evolve empty cell
                    self._evolve_empty_cell(x, y, avg, new_grid)

        self.automaton_grid = new_grid
        self.generation += 1

    def draw(self, stdscr, font=None):
        """Draw the asteroid field visualization

        Args:
            stdscr: The curses screen to draw on
            font: Optional font to use for rendering (not used in curses mode)
        """
        super().draw(stdscr, font)

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

        stdscr.refresh()
        time.sleep(0.1)
        """Quantum-inspired probability wave collapse animation"""
        self._quantum_animation(stdscr)

    def _quantum_animation(self, stdscr):
        """Quantum-inspired probability wave collapse animation

        Creates a visual effect that resembles quantum probability wave collapse,
        where multiple potential states exist until 'observed', at which point
        they collapse into a definite state. This is visualized through characters
        that flicker between different states before settling.

        Args:
            stdscr: The curses screen to draw on
        """
        with contextlib.suppress(Exception):
            # Only trigger quantum effects occasionally (10% chance)
            if random.random() > 0.1:
                return

            # Select cells for quantum effects
            quantum_cells = self._select_quantum_cells()
            if not quantum_cells:
                return

            # Quantum probability characters (from uncertain to certain)
            quantum_chars = [
                "?",
                "∿",
                "⌇",
                "⌁",
                "⌀",
                "∴",
                "∵",
                "⊛",
                "⊚",
                "⊙",
                "⊗",
                "⊕",
                "⊖",
            ]

            # Perform the quantum animation sequence
            self._animate_quantum_collapse(stdscr, quantum_cells, quantum_chars)

    def _select_quantum_cells(self):
        """Select cells for quantum effects based on probability.

        Returns:
            List of (x, y) coordinates for cells to apply quantum effects
        """
        quantum_cells = []
        for y in range(self.height - 2):
            for x in range(self.width - 2):
                # Higher probability for cells that already have asteroids
                prob = 0.05  # Base probability for empty cells
                if self.automaton_grid[y][x] > 0:
                    prob = 0.3  # Higher probability for asteroid cells

                if random.random() < prob:
                    quantum_cells.append((x, y))
        return quantum_cells

    def _animate_quantum_collapse(self, stdscr, quantum_cells, quantum_chars):
        """Animate the quantum collapse effect across multiple phases.

        Args:
            stdscr: The curses screen to draw on
            quantum_cells: List of (x, y) coordinates for cells to animate
            quantum_chars: List of characters to use for quantum states
        """
        for phase in range(5):  # 5 phases of collapse
            for x, y in quantum_cells:
                char, attr = self._get_quantum_char_for_phase(
                    x, y, phase, quantum_chars
                )

                # Draw the character
                with contextlib.suppress(curses.error):
                    stdscr.addstr(self.y + y + 2, self.x + x + 1, char, attr)

            # Refresh and short delay between phases
            stdscr.refresh()
            time.sleep(0.05)

    def _get_quantum_char_for_phase(self, x, y, phase, quantum_chars):
        """Get the appropriate character and attributes for a cell at a given phase.

        Args:
            x: X coordinate of the cell
            y: Y coordinate of the cell
            phase: Current phase of the quantum collapse (0-4)
            quantum_chars: List of characters to use for quantum states

        Returns:
            Tuple of (character, attributes) to display
        """
        if phase < 4:
            return self._get_uncertain_quantum_state(x, y, phase, quantum_chars)
        else:
            return self._get_collapsed_quantum_state(y, x)

    def _get_uncertain_quantum_state(self, x, y, phase, quantum_chars):
        """Get character and attributes for a cell in an uncertain quantum state.

        Args:
            x: X coordinate of the cell
            y: Y coordinate of the cell
            phase: Current phase of the quantum collapse
            quantum_chars: List of characters to use for quantum states

        Returns:
            Tuple of (character, attributes) to display
        """
        # Calculate a position and time-dependent index
        wave_factor = math.sin(x / 3 + y / 2 + phase / 2) + 1  # Range 0-2
        char_idx = min(
            int(wave_factor * len(quantum_chars) / 2), len(quantum_chars) - 1
        )

        # Uncertain state - use quantum characters
        char = quantum_chars[char_idx]
        # Use a different color for quantum states
        attr = curses.A_BOLD

        return char, attr

    def _get_collapsed_quantum_state(self, y, x):
        """Get character and attributes for a cell in its final collapsed state.

        Args:
            y: Y coordinate of the cell
            x: X coordinate of the cell

        Returns:
            Tuple of (character, attributes) to display
        """
        current_value = self.automaton_grid[y][x]
        collapse_prob = 0.7  # Probability of state change in final phase

        # Determine the final state
        char, attr = self._determine_final_state(y, x, current_value, collapse_prob)
        return char, attr

    def _determine_final_state(self, y, x, current_value, collapse_prob):
        """Determine the final state of a cell after quantum collapse.

        Args:
            y: Y coordinate of the cell
            x: X coordinate of the cell
            current_value: Current value of the cell in the automaton grid
            collapse_prob: Probability of state change

        Returns:
            Tuple of (character, attributes) to display
        """
        if random.random() < collapse_prob:
            # Collapse to a new state
            if current_value > 0 and random.random() < 0.4:
                # Asteroid disappears
                self.automaton_grid[y][x] = 0
                char = " "
            else:
                # Empty space becomes asteroid or asteroid changes
                new_value = random.uniform(0.3, 1.0)
                self.automaton_grid[y][x] = new_value
                char = self._get_asteroid_char(new_value)
        elif current_value > 0:
            char = self._get_asteroid_char(current_value)
        else:
            char = " "

        return char, curses.A_NORMAL

    def _get_asteroid_char(self, value):
        """Convert a value to an asteroid character.

        Args:
            value: Asteroid density value (0.0-1.0)

        Returns:
            Character to represent the asteroid
        """
        char_index = min(
            len(self.asteroid_chars) - 1,
            int(value * len(self.asteroid_chars)),
        )
        return self.asteroid_chars[char_index]
