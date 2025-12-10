"""
ASCIICrewManagementPanel: Assign and manage crew members to different ship functions.

This component provides a user interface for displaying, assigning, and managing
crew members across different ship stations.
"""

# Standard library imports
import logging

# Local application imports
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party library imports
import pygame

# Use absolute imports for consistency
from src.ui.ui_base.ascii_base import COLOR_HIGHLIGHT, UIStyle
from src.ui.ui_element.ascii_button import ASCIIButton
from src.ui.ui_element.ascii_panel import ASCIIPanel

# Define additional colors
COLOR_TEXT = (200, 200, 200)  # Light gray text color
COLOR_WARNING = (255, 100, 100)  # Red warning color

# Type definitions for better type checking
Color = Tuple[int, int, int]
Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height


class ASCIICrewManagementPanel:
    """Display and manage crew members and their assignments to ship stations."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str = "Crew Management",
        style: UIStyle = UIStyle.MECHANICAL,
        on_assign_callback: Optional[Callable[[str, str], None]] = None,
        on_unassign_callback: Optional[Callable[[str], None]] = None,
        on_train_callback: Optional[Callable[[str, str], None]] = None,
        on_rest_callback: Optional[Callable[[str], None]] = None,
        on_recruit_callback: Optional[Callable[[], None]] = None,
    ):
        """Initialize a crew management panel.

        Args:
            rect: Rectangle defining position and size
            title: Panel title
            style: Visual style
            on_assign_callback: Callback for crew assignment (crew_id, station)
            on_unassign_callback: Callback for crew unassignment (crew_id)
            on_train_callback: Callback for crew training (crew_id, skill)
            on_rest_callback: Callback for crew rest (crew_id)
            on_recruit_callback: Callback for recruiting new crew
        """
        self.rect = rect
        self.title = title
        self.style = style

        # Callbacks
        self.on_assign_callback = on_assign_callback
        self.on_unassign_callback = on_unassign_callback
        self.on_train_callback = on_train_callback
        self.on_rest_callback = on_rest_callback
        self.on_recruit_callback = on_recruit_callback

        # Panel state
        self.crew_members = []  # List of crew member data
        self.station_assignments = {}  # Dict of station: [crew_ids]
        self.station_efficiencies = {}  # Dict of station: efficiency

        # Custom font that supports box drawing characters
        self.custom_font = None

        # View state
        self.selected_crew_id = None
        self.selected_station = None
        self.view_mode = "crew"  # "crew" or "stations"
        self.page = 0
        self.items_per_page = 5

        # Colors for different status types
        self.status_colors = {
            "available": (100, 255, 100),  # Green
            "working": (100, 200, 255),  # Blue
            "resting": (200, 200, 100),  # Yellow
            "exhausted": (255, 100, 100),  # Red
            "injured": (255, 50, 50),  # Bright red
        }

        # Skill level colors
        self.skill_colors = {
            "Untrained": (150, 150, 150),  # Gray
            "Novice": (200, 200, 200),  # Light gray
            "Trained": (100, 255, 100),  # Green
            "Skilled": (100, 200, 255),  # Blue
            "Expert": (255, 200, 100),  # Orange
            "Master": (255, 100, 255),  # Purple
        }

        # Create buttons
        self.buttons = []
        self._create_buttons()

        logging.info("ASCIICrewManagementPanel initialized")

    def _create_buttons(self) -> None:
        """Create the panel's buttons."""
        button_height = 20
        button_margin = 5
        button_width = 120

        # Calculate button positions
        button_x = self.rect.x + self.rect.width - button_width - button_margin
        button_y = self.rect.y + self.rect.height - button_height - button_margin

        # Create buttons
        self.buttons = [
            ASCIIButton(
                button_x,
                button_y,
                button_width,
                button_height,
                self.style,
                "Recruit Crew",
                self._on_recruit_click,
            ),
            ASCIIButton(
                button_x - button_width - button_margin,
                button_y,
                button_width,
                button_height,
                self.style,
                "Toggle View",
                self._on_toggle_view_click,
            ),
            ASCIIButton(
                button_x - 2 * (button_width + button_margin),
                button_y,
                button_width,
                button_height,
                self.style,
                "Next Page",
                self._on_next_page_click,
            ),
            ASCIIButton(
                button_x - 3 * (button_width + button_margin),
                button_y,
                button_width,
                button_height,
                self.style,
                "Prev Page",
                self._on_prev_page_click,
            ),
        ]

    def update_crew_data(
        self,
        crew_members: List[Dict[str, Any]],
        station_assignments: Dict[str, List[str]],
        station_efficiencies: Dict[str, float],
    ) -> None:
        """Update the crew and station data.

        Args:
            crew_members: List of crew member data dictionaries
            station_assignments: Dict of station: [crew_ids]
            station_efficiencies: Dict of station: efficiency
        """
        try:
            self.crew_members = crew_members
            self.station_assignments = station_assignments
            self.station_efficiencies = station_efficiencies

            # Reset selection if needed
            if self.selected_crew_id and all(
                cm["id"] != self.selected_crew_id for cm in crew_members
            ):
                self.selected_crew_id = None

            if (
                self.selected_station
                and self.selected_station not in station_assignments
            ):
                self.selected_station = None
        except Exception as e:
            logging.error(f"Error updating crew data: {e}")

    def handle_input(self, event: pygame.event.Event) -> bool:
        """Handle input events.

        Args:
            event: Pygame event

        Returns:
            True if the event was handled, False otherwise
        """
        try:
            # Check if any button was clicked
            if self._handle_button_input(event):
                return True

            # Try handling with mouse input first, then keyboard input
            return self._handle_mouse_input(event) or self._handle_keyboard_input(event)
        except Exception as e:
            logging.error(f"Error handling input: {e}")
            return False

    def _handle_button_input(self, event: pygame.event.Event) -> bool:
        """Handle input for buttons.

        Args:
            event: Pygame event

        Returns:
            True if the event was handled, False otherwise
        """
        return any(button.handle_input(event) for button in self.buttons)

    def _handle_mouse_input(self, event: pygame.event.Event) -> bool:
        """Handle mouse input for crew/station selection.

        Args:
            event: Pygame event

        Returns:
            True if the event was handled, False otherwise
        """
        # Check if this is a left mouse click within the content area
        is_valid_click = (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self._is_point_in_content_area(event.pos)
        )

        if not is_valid_click:
            return False

        # Handle the click based on current view mode
        if self.view_mode == "crew":
            self._handle_crew_click(event.pos)
        else:
            self._handle_station_click(event.pos)

        return True

    def _handle_keyboard_input(self, event: pygame.event.Event) -> bool:
        """Handle keyboard input for crew actions.

        Args:
            event: Pygame event

        Returns:
            True if the event was handled, False otherwise
        """
        if event.type != pygame.KEYDOWN:
            return False

        if event.key == pygame.K_a and self.selected_crew_id and self.selected_station:
            self._assign_selected_crew()
            return True

        if event.key == pygame.K_u and self.selected_crew_id:
            self._unassign_selected_crew()
            return True

        if event.key == pygame.K_r and self.selected_crew_id:
            self._rest_selected_crew()
            return True

        if event.key == pygame.K_t and self.selected_crew_id:
            self._handle_train_crew()
            return True

        return False

    def _handle_train_crew(self) -> None:
        """Handle training the selected crew member."""
        crew_member = next(
            (cm for cm in self.crew_members if cm["id"] == self.selected_crew_id), None
        )
        if not crew_member or not self.on_train_callback:
            return

        skills = crew_member.get("skills", {})
        if not skills:
            return

        best_skill = max(skills.items(), key=lambda x: x[1])[0]
        self.on_train_callback(self.selected_crew_id, best_skill)

    def _is_point_in_content_area(self, point: Point) -> bool:
        """Check if a point is within the content area.

        Args:
            point: Point to check (x, y)

        Returns:
            True if the point is in the content area, False otherwise
        """
        # Define content area (excluding title and buttons)
        content_x = self.rect.x + 10
        content_y = self.rect.y + 30
        content_width = self.rect.width - 20
        content_height = self.rect.height - 60

        return (
            content_x <= point[0] <= content_x + content_width
            and content_y <= point[1] <= content_y + content_height
        )

    def _handle_crew_click(self, point: Point) -> None:
        """Handle a click in crew view mode.

        Args:
            point: Click position (x, y)
        """
        # Calculate item positions
        font_height = 20  # Approximate
        item_height = font_height * 3  # Each crew entry takes 3 lines
        content_y = self.rect.y + 30

        # Determine which crew member was clicked
        start_idx = self.page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.crew_members))

        for i in range(start_idx, end_idx):
            item_y = content_y + (i - start_idx) * item_height

            if item_y <= point[1] <= item_y + item_height:
                self.selected_crew_id = self.crew_members[i]["id"]
                return

    def _handle_station_click(self, point: Point) -> None:
        """Handle a click in station view mode.

        Args:
            point: Click position (x, y)
        """
        # Calculate item positions
        font_height = 20  # Approximate
        item_height = font_height * 2  # Each station entry takes 2 lines
        content_y = self.rect.y + 30

        # Get sorted station list
        stations = sorted(self.station_assignments.keys())

        # Determine which station was clicked
        for i, station in enumerate(stations):
            item_y = content_y + i * item_height

            if item_y <= point[1] <= item_y + item_height:
                self.selected_station = station
                return

    def _assign_selected_crew(self) -> None:
        """Assign the selected crew member to the selected station."""
        if not self.selected_crew_id or not self.selected_station:
            return

        if self.on_assign_callback:
            self.on_assign_callback(self.selected_crew_id, self.selected_station)

    def _unassign_selected_crew(self) -> None:
        """Unassign the selected crew member from their current station."""
        if not self.selected_crew_id:
            return

        # Find the crew member
        crew_member = next(
            (cm for cm in self.crew_members if cm["id"] == self.selected_crew_id), None
        )
        if not crew_member or not crew_member.get("current_station"):
            return

        if self.on_unassign_callback:
            self.on_unassign_callback(self.selected_crew_id)

    def _rest_selected_crew(self) -> None:
        """Rest the selected crew member."""
        if not self.selected_crew_id:
            return

        if self.on_rest_callback:
            self.on_rest_callback(self.selected_crew_id)

    def _on_recruit_click(self) -> None:
        """Handle recruit button click."""
        if self.on_recruit_callback:
            self.on_recruit_callback()

    def _on_toggle_view_click(self) -> None:
        """Handle toggle view button click."""
        self.view_mode = "stations" if self.view_mode == "crew" else "crew"
        self.page = 0  # Reset page when changing view

    def _on_next_page_click(self) -> None:
        """Handle next page button click."""
        if self.view_mode == "crew":
            max_page = (len(self.crew_members) - 1) // self.items_per_page
            self.page = min(self.page + 1, max_page)
        # Stations view doesn't need pagination

    def _on_prev_page_click(self) -> None:
        """Handle previous page button click."""
        if self.view_mode == "crew":
            self.page = max(0, self.page - 1)
        # Stations view doesn't need pagination

    def _get_status_color(self, status: str) -> Color:
        """Get the display color for a status.

        Args:
            status: Status string

        Returns:
            RGB color tuple
        """
        return self.status_colors.get(status, COLOR_TEXT)

    def _get_skill_color(self, skill_level: str) -> Color:
        """Get the display color for a skill level.

        Args:
            skill_level: Skill level string

        Returns:
            RGB color tuple
        """
        return self.skill_colors.get(skill_level, COLOR_TEXT)

    def _get_efficiency_color(self, efficiency: float) -> Color:
        """Get color based on efficiency value.

        Args:
            efficiency: Efficiency value (0.0-5.0)

        Returns:
            RGB color tuple
        """
        if efficiency < 1.0:
            return self.status_colors["exhausted"]  # Red
        elif efficiency < 2.0:
            return self.status_colors["resting"]  # Yellow
        elif efficiency < 3.0:
            return COLOR_TEXT  # White
        elif efficiency < 4.0:
            return self.status_colors["working"]  # Blue
        else:
            return self.status_colors["available"]  # Green

    def set_font(self, font: pygame.font.Font) -> None:
        """Set a custom font that better supports box drawing characters.

        Args:
            font: Font to use for rendering box drawing characters
        """
        self.custom_font = font

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the crew management panel.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering

        Returns:
            pygame.Rect: The drawn area
        """
        try:
            # Draw panel background and border
            panel = ASCIIPanel(
                self.rect.x,
                self.rect.y,
                self.rect.width,
                self.rect.height,
                self.style,
                self.title,
            )

            # Use custom font if available, otherwise use the provided font
            panel_rect = panel.draw(surface, self.custom_font or font)

            # Calculate layout
            margin = font.get_height() // 2
            content_x = self.rect.x + margin
            content_y = self.rect.y + margin * 3  # Extra margin for title
            content_width = self.rect.width - margin * 2

            # Draw view mode indicator
            view_text = f"View: {'Crew List' if self.view_mode == 'crew' else 'Station Assignments'}"
            surface.blit(
                font.render(view_text, True, COLOR_HIGHLIGHT), (content_x, content_y)
            )
            content_y += font.get_height() + margin

            # Draw content based on view mode
            if self.view_mode == "crew":
                self._draw_crew_view(surface, font, content_x, content_y, content_width)
            else:
                self._draw_station_view(
                    surface, font, content_x, content_y, content_width
                )

            # Draw buttons
            for button in self.buttons:
                button.draw(surface, font)

            # Draw key commands
            commands_y = self.rect.y + self.rect.height - font.get_height() - margin * 3
            commands_text = "Commands: [A]ssign  [U]nassign  [R]est  [T]rain"
            surface.blit(
                font.render(commands_text, True, COLOR_TEXT), (content_x, commands_y)
            )

            return panel_rect
        except Exception as e:
            logging.error(f"Error drawing crew management panel: {e}")
            return self.rect

    def _draw_crew_view(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
    ) -> None:
        """Draw the crew list view.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            width: Width of the content area
        """
        if not self.crew_members:
            self._draw_empty_crew_message(surface, font, x, y)
            return

        # Calculate pagination
        start_idx, end_idx, total_pages = self._calculate_pagination()

        # Draw page indicator
        y = self._draw_page_indicator(surface, font, x, y, width, total_pages)

        # Draw crew list
        for i in range(start_idx, end_idx):
            crew = self.crew_members[i]
            y = self._draw_crew_member(surface, font, x, y, width, crew)

            # Add spacing between crew members
            y += 5

    @staticmethod
    def _draw_empty_crew_message(
        surface: pygame.Surface, font: pygame.font.Font, x: int, y: int
    ) -> None:
        """Draw message when no crew members are available.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
        """
        surface.blit(
            font.render(
                "No crew members available. Click 'Recruit Crew' to add crew.",
                True,
                COLOR_WARNING,
            ),
            (x, y),
        )

    def _calculate_pagination(self) -> Tuple[int, int, int]:
        """Calculate pagination values for crew list.

        Returns:
            Tuple containing start index, end index, and total pages
        """
        start_idx = self.page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.crew_members))
        total_pages = (len(self.crew_members) - 1) // self.items_per_page + 1
        return start_idx, end_idx, total_pages

    def _draw_page_indicator(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
        total_pages: int,
    ) -> int:
        """Draw the page indicator.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            width: Width of the content area
            total_pages: Total number of pages

        Returns:
            Updated Y coordinate after drawing
        """
        page_text = f"Page {self.page + 1}/{total_pages}"
        surface.blit(
            font.render(page_text, True, COLOR_TEXT),
            (x + width - font.size(page_text)[0], y),
        )
        return y + font.get_height() + 5

    def _draw_crew_member(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
        crew: Dict,
    ) -> int:
        """Draw a single crew member's information.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            width: Width of the content area
            crew: Crew member data dictionary

        Returns:
            Updated Y coordinate after drawing
        """
        # Highlight selected crew
        if crew["id"] == self.selected_crew_id:
            self._draw_crew_highlight(surface, x, y, width, font)

        # Draw crew name and level
        name_text = f"{crew['name']} (Level {crew['level']})"
        surface.blit(font.render(name_text, True, COLOR_HIGHLIGHT), (x, y))

        # Draw status and station
        y = self._draw_crew_status(surface, font, x, y, width, crew)

        # Draw skills
        return self._draw_crew_skills(surface, font, x, y, crew)

    @staticmethod
    def _draw_crew_highlight(
        surface: pygame.Surface,
        x: int,
        y: int,
        width: int,
        font: pygame.font.Font,
    ) -> None:
        """Draw highlight rectangle for selected crew member.

        Args:
            surface: Surface to draw on
            x: X coordinate
            y: Y coordinate
            width: Width of the highlight area
            font: Font to use for calculating height
        """
        highlight_rect = pygame.Rect(
            x - 5, y - 2, width + 10, font.get_height() * 3 + 4
        )
        pygame.draw.rect(surface, (50, 50, 80), highlight_rect)

    def _draw_crew_status(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
        crew: Dict,
    ) -> int:
        """Draw crew member status and station assignment.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            width: Width of the content area
            crew: Crew member data dictionary

        Returns:
            Updated Y coordinate after drawing
        """
        status = crew.get("status", "available")
        status_color = self._get_status_color(status)
        station_text = f"Status: {status.title()}"

        if crew.get("current_station"):
            station_text += f" - Assigned to: {crew['current_station'].title()}"

        surface.blit(font.render(station_text, True, status_color), (x + width // 2, y))
        return y + font.get_height()

    def _draw_crew_skills(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        crew: Dict,
    ) -> int:
        """Draw crew member skills.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            crew: Crew member data dictionary

        Returns:
            Updated Y coordinate after drawing
        """
        skills = crew.get("skills", {})

        # Draw first row of skills (up to 3)
        skills_text = "Skills: "
        for j, (skill, level) in enumerate(skills.items()):
            if j < 3:  # First row shows 3 skills
                level_name = self._get_skill_level_name(level)
                skills_text += f"{skill.title()}: {level_name}  "

        surface.blit(font.render(skills_text, True, COLOR_TEXT), (x, y))
        y += font.get_height()

        # Draw second row of skills (remaining skills)
        if len(skills) > 3:
            skills_text = "        "  # Align with "Skills: " above
            for j, (skill, level) in enumerate(skills.items()):
                if j >= 3:  # Second row shows remaining skills
                    level_name = self._get_skill_level_name(level)
                    skills_text += f"{skill.title()}: {level_name}  "

            surface.blit(font.render(skills_text, True, COLOR_TEXT), (x, y))
            y += font.get_height()

        return y

    def _draw_station_view(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        width: int,
    ) -> None:
        """Draw the station assignments view.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X coordinate
            y: Y coordinate
            width: Width of the content area
        """
        if not self.station_assignments:
            # Draw empty message
            surface.blit(
                font.render("No stations available.", True, COLOR_WARNING), (x, y)
            )
            return

        # Get sorted station list
        stations = sorted(self.station_assignments.keys())

        # Draw each station
        for station in stations:
            # Highlight selected station
            if station == self.selected_station:
                highlight_rect = pygame.Rect(
                    x - 5, y - 2, width + 10, font.get_height() * 2 + 4
                )
                pygame.draw.rect(surface, (50, 50, 80), highlight_rect)

            # Draw station name and efficiency
            efficiency = self.station_efficiencies.get(station, 0.0)
            efficiency_text = f"Efficiency: {efficiency:.1f}"
            efficiency_color = self._get_efficiency_color(efficiency)

            station_text = f"{station.title()} - {efficiency_text}"
            surface.blit(font.render(station_text, True, efficiency_color), (x, y))
            y += font.get_height()

            if crew_ids := self.station_assignments.get(station, []):
                crew_names = []
                for crew_id in crew_ids:
                    if crew := next(
                        (c for c in self.crew_members if c["id"] == crew_id), None
                    ):
                        crew_names.append(crew["name"])

                crew_text = f"Crew: {', '.join(crew_names)}"
                surface.blit(font.render(crew_text, True, COLOR_TEXT), (x, y))
            else:
                surface.blit(
                    font.render("Crew: None assigned", True, COLOR_WARNING), (x, y)
                )

            y += font.get_height() + 5

    @staticmethod
    def _get_skill_level_name(level: int) -> str:
        """Get the name of a skill level.

        Args:
            level: Skill level (0-5)

        Returns:
            Skill level name
        """
        skill_levels = {
            0: "Untrained",
            1: "Novice",
            2: "Trained",
            3: "Skilled",
            4: "Expert",
            5: "Master",
        }
        return skill_levels.get(level, "Unknown")
