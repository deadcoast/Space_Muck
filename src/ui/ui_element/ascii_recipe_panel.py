# Standard library imports
import logging

# Third-party library imports

# Local application imports
from config import COLOR_TEXT
from typing import Tuple, List, Dict, Optional, Any, TypeVar
from src.ui.ui_base.ascii_base import UIStyle
from src.ui.draw_utils import draw_text
from src.ui.ui_base.ascii_ui import ASCIIButton
from src.ui.ui_base.ascii_ui import ASCIIPanel
import pygame


# Type definitions for better type checking
T = TypeVar("T")
Color = Tuple[int, int, int]
ColorWithAlpha = Tuple[int, int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height


class ASCIIRecipePanel:
    """Panel for displaying and managing converter recipes."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Available Recipes",
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        """Initialize a recipe panel.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
        """
        self.rect = rect
        self.title = title
        self.style = style
        self.recipes: List[Dict[str, Any]] = []
        self.selected_idx: Optional[int] = None
        self.scroll_offset = 0
        self.max_visible_recipes = 8

        # Button height will be set when drawn
        self.button_height = 0
        self.start_button = ASCIIButton(
            rect.x + rect.width // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Start Recipe",
            self._on_start_recipe,
            style,
        )
        self.stop_button = ASCIIButton(
            rect.x + rect.width * 3 // 4,
            rect.bottom,  # Y position will be adjusted when drawn
            "Stop Recipe",
            self._on_stop_recipe,
            style,
        )

    def set_recipes(self, recipes: List[Dict[str, Any]]) -> None:
        """Set the list of available recipes.

        Args:
            recipes: List of recipe dictionaries containing name, inputs, outputs,
                    and efficiency metrics
        """
        self.recipes = recipes
        self.selected_idx = 0 if recipes else None
        self.scroll_offset = 0

    def _on_start_recipe(self) -> None:
        """Callback for starting the selected recipe."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.recipes):
            self._handle_recipe_action("Starting recipe: ", "start")

    def _on_stop_recipe(self) -> None:
        """Callback for stopping the current recipe."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.recipes):
            self._handle_recipe_action("Stopping recipe: ", "stop")

    def _handle_recipe_action(self, action_message, action_type):
        recipe = self.recipes[self.selected_idx]
        print(f"{action_message}{recipe['name']}")
        self.on_recipe_action(recipe, action_type)

    def on_recipe_action(self, recipe: Dict[str, Any], action: str) -> None:
        """Override this method to handle recipe actions.

        Args:
            recipe: The recipe being acted upon
            action: The action being performed ("start" or "stop")
        """
        # This is a placeholder method that should be overridden by the game code
        pass

    def _draw_recipe_entry(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        recipe: Dict[str, Any],
        is_selected: bool,
    ) -> None:
        """Draw a single recipe entry.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate to draw at
            y: Y coordinate to draw at
            recipe: Recipe data dictionary
            is_selected: Whether this recipe is currently selected
        """
        try:
            # Draw selection indicator
            if is_selected:
                draw_text(
                    surface,
                    ">",
                    x - 2,
                    y,
                    size=font.get_height(),
                    color=(255, 255, 100),
                )

            # Ensure recipe has a name
            recipe_name = recipe.get("name", "Unknown Recipe")

            # Draw recipe name and base efficiency
            name_color = (100, 255, 100) if recipe.get("active", False) else COLOR_TEXT
            draw_text(
                surface, recipe_name, x, y, size=font.get_height(), color=name_color
            )

            # Draw efficiency bar
            eff = min(
                max(recipe.get("efficiency", 0.0), 0.0), 1.0
            )  # Clamp between 0 and 1
            bar_x = x + font.size(recipe_name)[0] + 4
            bar_width = 20
            bar = "=" * int(bar_width * eff)
            bar = bar.ljust(bar_width, ".")

            # Determine color based on efficiency
            if eff > 0.8:
                eff_color = (100, 255, 100)  # Green for high efficiency
            elif eff > 0.5:
                eff_color = (255, 255, 100)  # Yellow for medium efficiency
            else:
                eff_color = (255, 100, 100)  # Red for low efficiency
            draw_text(
                surface,
                f"[{bar}] {eff * 100:.1f}%",
                bar_x,
                y,
                size=font.get_height(),
                color=eff_color,
            )

            # Draw input/output summary
            y += font.get_height()
            inputs = " + ".join(str(item) for item in recipe.get("inputs", []))
            outputs = " + ".join(str(item) for item in recipe.get("outputs", []))
            draw_text(
                surface,
                f"  {inputs} -> {outputs}",
                x,
                y,
                size=font.get_height(),
                color=(160, 160, 160),
            )
        except Exception as e:
            # Handle any errors gracefully
            logging.warning(f"Error drawing recipe entry: {e}")
            draw_text(
                surface,
                f"Error: {str(e)[:20]}...",
                x,
                y,
                size=font.get_height(),
                color=(255, 100, 100),
            )

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events.

        Returns:
            bool: True if the event was handled
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.selected_idx is not None:
                self.selected_idx = max(0, self.selected_idx - 1)
                if self.selected_idx < self.scroll_offset:
                    self.scroll_offset = self.selected_idx
                return True
            elif event.key == pygame.K_DOWN and self.selected_idx is not None:
                self.selected_idx = min(len(self.recipes) - 1, self.selected_idx + 1)
                if self.selected_idx >= self.scroll_offset + self.max_visible_recipes:
                    self.scroll_offset = (
                        self.selected_idx - self.max_visible_recipes + 1
                    )
                return True

        # Handle button events
        if self.start_button.handle_event(event):
            return True  # Start button was handled
        return bool(self.stop_button.handle_event(event))

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the recipe panel.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        # Draw panel background and border
        panel = ASCIIPanel(self.rect, self.title, self.style)
        panel_rect = panel.draw(surface, font)

        if not self.recipes:
            # Draw empty state
            x = self.rect.x + self.rect.width // 4
            y = self.rect.y + self.rect.height // 2
            draw_text(
                surface,
                "No recipes available",
                x,
                y,
                size=font.get_height(),
                color=COLOR_TEXT,
            )
            return panel_rect

        # Calculate layout
        margin = font.get_height()
        x = self.rect.x + margin
        y = self.rect.y + margin * 2  # Extra margin for title
        recipe_height = font.get_height() * 2 + 1  # 2 lines + spacing

        # Draw visible recipes
        end_idx = min(self.scroll_offset + self.max_visible_recipes, len(self.recipes))
        for i in range(self.scroll_offset, end_idx):
            recipe = self.recipes[i]
            self._draw_recipe_entry(surface, font, x, y, recipe, i == self.selected_idx)
            y += recipe_height

        # Draw scroll indicators if needed
        if self.scroll_offset > 0:
            draw_text(
                surface,
                "^ More ^",
                self.rect.centerx - 4,
                self.rect.y + margin,
                size=font.get_height(),
                color=COLOR_TEXT,
            )
        if end_idx < len(self.recipes):
            draw_text(
                surface,
                "v More v",
                self.rect.centerx - 4,
                self.rect.bottom - margin * 4,
                size=font.get_height(),
                color=COLOR_TEXT,
            )

        # Update and draw control buttons
        self.button_height = font.get_height() * 3
        self.start_button.y = self.rect.bottom - self.button_height
        self.stop_button.y = self.rect.bottom - self.button_height

        # Enable/disable buttons based on selection state
        has_selection = self.selected_idx is not None and 0 <= self.selected_idx < len(
            self.recipes
        )
        self.start_button.enabled = has_selection
        self.stop_button.enabled = has_selection and self.recipes[
            self.selected_idx
        ].get("active", False)

        start_rect = self.start_button.draw(surface, font)
        stop_rect = self.stop_button.draw(surface, font)

        # Return the appropriate rect based on selection state
        if has_selection:
            # When we have a selection, include button areas in the returned rect
            return panel_rect.union(start_rect).union(stop_rect)
        else:
            # When no selection, just return the panel rect
            return panel_rect
