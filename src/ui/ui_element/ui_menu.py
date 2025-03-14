# Use absolute imports for consistency

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from typing import Tuple

# Standard library imports
import random

# Third-party library imports

# Local application imports
from typing import List, Optional, Tuple, TYPE_CHECKING, Any
import contextlib
import pygame

# For backward compatibility, keep Menu as an alias of UIMenu
Menu = UIMenu

# Define standard colors for UI components
COLOR_TEXT = (220, 220, 220)  # Standard text color
COLOR_BG = (20, 20, 30)  # Standard background color
COLOR_HIGHLIGHT = (180, 180, 255)  # Standard highlight color

# Import at runtime to avoid circular imports


class UIMenu(ASCIIBox):
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

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = COLOR_TEXT,
        bg_color: Tuple[int, int, int] = COLOR_BG,
        alpha: int = 255,
    ) -> pygame.Rect:
        """Draw menu with options

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            base_color: Base color for drawing
            bg_color: Background color
            alpha: Transparency (0-255)

        Returns:
            The rectangle area that was drawn
        """
        super().draw(surface, font, base_color, bg_color, alpha)

        # Draw title
        title_x = self.x + (self.width - len(self.title)) // 2
        draw_utils.draw_text(
            surface, self.title, (title_x, self.y + 1), font, base_color, bold=True
        )

        # Draw options with evolutionary prominence
        for i, option in enumerate(self.options):
            option_y = self.y + 3 + i

            if i == self.selected_index and self.active:
                # Selection indicator for highlighted option
                prefix = "▶ "
                # Highlighted option - use inverse colors
                draw_utils.draw_text(
                    surface,
                    prefix + option,
                    (self.x + 2, option_y),
                    font,
                    bg_color,  # Swap colors for highlight effect
                    bg_color=base_color,
                )
            else:
                # Normal option with prominence-based rendering
                bold = False
                # Calculate prominence factor based on usage
                prominence = min(1.0, 0.5 + self.option_metrics[i] * 0.1)

                if prominence > 0.8:
                    bold = True

                # Apply adaptive decoration based on usage
                evolving_chars = ["·", "∘", "○", "◌", "◍", "◎", "●"]
                decoration = evolving_chars[
                    min(len(evolving_chars) - 1, int(prominence * len(evolving_chars)))
                ]

                prefix = f"{decoration} "

                draw_utils.draw_text(
                    surface,
                    prefix + option,
                    (self.x + 2, option_y),
                    font,
                    base_color,
                    bold=bold,
                )

    def handle_input(self, key: int) -> Optional[str]:
        """Handle navigation and selection input

        Args:
            key: Pygame key constant

        Returns:
            Selected option text or None if no selection was made
        """
        if not self.active:
            return None

        if key == pygame.K_UP:
            self.selected_index = (self.selected_index - 1) % len(self.options)
        elif key == pygame.K_DOWN:
            self.selected_index = (self.selected_index + 1) % len(self.options)
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
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
