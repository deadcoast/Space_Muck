import contextlib
import curses
from typing import List
from ui.ui_element.ui_style import UIStyle
from ui.ui_element.ui_element import UIElement

# Define standard colors for UI components
COLOR_TEXT = (220, 220, 220)  # Standard text color
COLOR_BG = (20, 20, 30)  # Standard background color
COLOR_HIGHLIGHT = (180, 180, 255)  # Standard highlight color


class SymbioteEvolutionMonitor(UIElement):
    """Monitor for the symbiote evolution process"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        style: UIStyle = UIStyle.SYMBIOTIC,
    ):
        super().__init__(x, y, width, height, style)
        self.evolution_stage = 0
        self.fitness_values = [0.0] * 20  # Track last 20 fitness values
        self.mutation_rate = 0.2
        self.generation = 0
        self.symbiote_patterns = self._generate_patterns()

    def _generate_patterns(self) -> List[List[str]]:
        """Generate evolving patterns for symbiotes at different stages"""
        return [
            # Stage 0: Simple forms
            ["∙", "·", "∙", "·"],
            # Stage 1: Growing
            ["∘", "○", "∘", "○"],
            # Stage 2: Basic structures
            ["◌", "◍", "◎", "●"],
            # Stage 3: Complex structures
            ["◐", "◑", "◒", "◓"],
            # Stage 4: Advanced forms
            ["◧", "◨", "◩", "◪"],
            # Stage 5: Final forms
            ["⬡", "⬢", "⬣", "⏣"],
        ]

    def draw(self, stdscr):
        """Draw the symbiote evolution monitor"""
        super().draw(stdscr)

        # Draw title
        with contextlib.suppress(curses.error):
            stdscr.addstr(
                self.y + 1,
                self.x + 2,
                f"SYMBIOTE EVOLUTION - GEN {self.generation}",
                curses.A_BOLD,
            )
        # Draw evolution stage and pattern
        stage_text = f"Stage: {self.evolution_stage}/5"
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 3, self.x + 2, stage_text)

            pattern = self.symbiote_patterns[min(5, self.evolution_stage)]
            pattern_index = self.generation % len(pattern)

            # Draw current pattern (larger)
            current_pattern = pattern[pattern_index] * 3
            stdscr.addstr(self.y + 3, self.x + 2 + len(stage_text) + 2, current_pattern)
        # Draw fitness trend
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 5, self.x + 2, "Fitness trend:")

            # Calculate trend line using simple bars
            for i, val in enumerate(self.fitness_values):
                bar_height = int(val * 5)  # Scale to 0-5
                for j in range(5):
                    char = "▓" if j < bar_height else "░"
                    if self.x + 2 + i < self.x + self.width - 1:
                        stdscr.addstr(self.y + 10 - j, self.x + 2 + i, char)
        # Draw mutation rate
        with contextlib.suppress(curses.error):
            stdscr.addstr(
                self.y + 12, self.x + 2, f"Mutation rate: {self.mutation_rate:.2f}"
            )
        # Draw generation count
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 14, self.x + 2, f"Generation: {self.generation}")

    def update_evolution(self, fitness: float):
        """Update evolution state with new fitness value"""
        # Update fitness history
        self.fitness_values.pop(0)
        self.fitness_values.append(fitness)

        # Update generation
        self.generation += 1

        # Potentially evolve to next stage
        if (
            self.generation > 10
            and sum(self.fitness_values[-5:]) > 0.8 * 5
            and self.evolution_stage < 5
        ):
            self.evolution_stage += 1

        # Adapt mutation rate based on fitness trend
        if len(self.fitness_values) >= 5:
            recent_avg = sum(self.fitness_values[-5:]) / 5
            older_avg = sum(self.fitness_values[-10:-5]) / 5

            if recent_avg > older_avg:
                # Reducing mutation as we're improving
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
            else:
                # Increasing mutation as we're plateauing
                self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
