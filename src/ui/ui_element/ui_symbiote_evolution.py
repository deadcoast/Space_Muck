"""
ui_symbiote_evolution.py

Provides a symbiote evolution monitor UI element with cellular automaton-inspired behaviors.
"""

# Standard library imports
from typing import List

# Third-party library imports
import pygame

# Local application imports
from config import COLOR_BG, COLOR_TEXT, COLOR_ACCENT
from src.ui.ui_base.ascii_base import UIElement, UIStyle


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
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 28)
        self.pattern_font = pygame.font.Font(None, 36)

    @staticmethod
    def _generate_patterns() -> List[List[str]]:
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

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the symbiote evolution monitor

        Args:
            surface: The pygame surface to draw on
        """
        # Draw panel background
        pygame.draw.rect(
            surface,
            COLOR_BG,
            pygame.Rect(self.x, self.y, self.width, self.height),
        )

        # Draw border
        pygame.draw.rect(
            surface,
            COLOR_ACCENT,
            pygame.Rect(self.x, self.y, self.width, self.height),
            2,
        )

        # Draw title
        title_text = f"SYMBIOTE EVOLUTION - GEN {self.generation}"
        title_surface = self.title_font.render(title_text, True, COLOR_TEXT)
        surface.blit(title_surface, (self.x + 10, self.y + 10))

        # Draw evolution stage and pattern
        stage_text = f"Stage: {self.evolution_stage}/5"
        stage_surface = self.font.render(stage_text, True, COLOR_TEXT)
        surface.blit(stage_surface, (self.x + 10, self.y + 40))

        # Draw current pattern
        pattern = self.symbiote_patterns[min(5, self.evolution_stage)]
        pattern_index = self.generation % len(pattern)
        current_pattern = pattern[pattern_index] * 3
        pattern_surface = self.pattern_font.render(current_pattern, True, COLOR_TEXT)
        surface.blit(
            pattern_surface, (self.x + 10 + stage_surface.get_width() + 10, self.y + 35)
        )

        # Draw "Fitness trend:" label
        trend_surface = self.font.render("Fitness trend:", True, COLOR_TEXT)
        surface.blit(trend_surface, (self.x + 10, self.y + 70))

        # Draw fitness trend as bars
        bar_width = 8
        bar_spacing = 2
        bar_max_height = 50

        for i, val in enumerate(self.fitness_values):
            bar_height = int(val * bar_max_height)
            bar_color = pygame.Color(
                int(255 * (1.0 - val)),  # R (high when fitness is low)
                int(255 * val),  # G (high when fitness is high)
                100,  # B (constant)
            )
            pygame.draw.rect(
                surface,
                bar_color,
                pygame.Rect(
                    self.x + 10 + i * (bar_width + bar_spacing),
                    self.y + 100 + (bar_max_height - bar_height),
                    bar_width,
                    bar_height,
                ),
            )

        # Draw mutation rate
        mutation_text = f"Mutation rate: {self.mutation_rate:.2f}"
        mutation_surface = self.font.render(mutation_text, True, COLOR_TEXT)
        surface.blit(mutation_surface, (self.x + 10, self.y + 160))

        # Draw generation count
        gen_text = f"Generation: {self.generation}"
        gen_surface = self.font.render(gen_text, True, COLOR_TEXT)
        surface.blit(gen_surface, (self.x + 10, self.y + 190))

    def update_evolution(self, fitness: float) -> None:
        """Update evolution state with new fitness value

        Args:
            fitness: New fitness value between 0.0 and 1.0
        """
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
        if len(self.fitness_values) >= 10:
            recent_avg = sum(self.fitness_values[-5:]) / 5
            older_avg = sum(self.fitness_values[-10:-5]) / 5

            if recent_avg > older_avg:
                # Reducing mutation as we're improving
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
            else:
                # Increasing mutation as we're plateauing
                self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
