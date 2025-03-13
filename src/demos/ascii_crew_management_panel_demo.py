"""
Demo application for the ASCIICrewManagementPanel component.

This demo showcases the crew management panel's functionality with a simulated
crew management system.
"""

# Standard library imports
import logging
import os
import random
import sys

# Third-party library imports

# Local application imports
from systems.crew_management_system import CrewManagementSystem
from ui.ui_base.ascii_base import UIStyle
from ui.ui_element.ascii_crew_management_panel import ASCIICrewManagementPanel
import contextlib
import pygame

# No typing imports needed

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the modules directly

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ASCIICrewManagementPanelDemo:
    """Demo application for the ASCIICrewManagementPanel component."""

    def __init__(self, width: int = 1024, height: int = 768):
        """Initialize the demo application.

        Args:
            width: Window width
            height: Window height
        """
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ASCII Crew Management Panel Demo")
        self.clock = pygame.time.Clock()

        # Try to load fonts in order of preference for box drawing character support
        fonts_to_try = [
            # Monospace fonts with good Unicode support
            "DejaVuSansMono",
            "LiberationMono",
            "UbuntuMono",
            "CourierNew",
            "Courier",
            "Menlo",
            "Monaco",
            "Consolas",
            # Fallbacks
            "monospace",
            None  # pygame's default font
        ]

        self.font = None
        test_chars = "┌─┐│└─┘"  # Box drawing characters to test

        for font_name in fonts_to_try:
            try:
                if font_name is None:
                    # Try pygame's default font
                    test_font = pygame.font.Font(None, 16)
                    logging.info("Trying pygame's default font")
                else:
                    # Try the named font
                    test_font = pygame.font.SysFont(font_name, 16, bold=True)
                    logging.info(f"Trying font: {font_name}")

                # Test if the font can render box drawing characters
                with contextlib.suppress(Exception):
                    test_surface = test_font.render(test_chars, True, (255, 255, 255))
                    if test_surface.get_width() >= len(test_chars) * 7:  # Reasonable width check
                        self.font = test_font
                        logging.info(f"Successfully loaded font: {font_name}")
                        break
            except Exception as e:
                logging.warning(f"Error loading font {font_name}: {e}")

        # If no font worked for box characters, fall back to a basic font that at least works
        if self.font is None:
            logging.warning("No font with box drawing support found, using fallback")
            try:
                self.font = pygame.font.SysFont("Arial", 16)
            except Exception:
                self.font = pygame.font.Font(None, 16)

        # Create crew management system
        self.crew_system = CrewManagementSystem()

        # Create some initial crew members
        self._recruit_initial_crew()

        # Create the crew management panel
        panel_rect = pygame.Rect(50, 50, width - 100, height - 100)
        self.crew_panel = ASCIICrewManagementPanel(
            panel_rect,
            "Crew Management",
            UIStyle.MECHANICAL,
            on_assign_callback=self._on_assign_crew,
            on_unassign_callback=self._on_unassign_crew,
            on_train_callback=self._on_train_crew,
            on_rest_callback=self._on_rest_crew,
            on_recruit_callback=self._on_recruit_crew
        )

        # Update panel with initial data
        self._update_panel_data()

        # Demo state
        self.running = True
        self.time_counter = 0
        self.message_log = []
        self.max_messages = 5

        logging.info("Demo initialized")

    def _recruit_initial_crew(self) -> None:
        """Recruit initial crew members."""
        # Recruit 5 random crew members
        result = self.crew_system.recruit_random_crew(5)
        
        if result["success"]:
            # Assign some crew to stations
            crew_members = self.crew_system.get_all_crew_members()
            
            if len(crew_members) >= 3:
                self.crew_system.assign_to_station(crew_members[0].id, "navigation")
                self.crew_system.assign_to_station(crew_members[1].id, "engineering")
                self.crew_system.assign_to_station(crew_members[2].id, "weapons")
        
        logging.info(f"Recruited initial crew: {len(self.crew_system.get_all_crew_members())} members")

    def _update_panel_data(self) -> None:
        """Update the panel with current crew system data."""
        # Convert crew members to dictionaries
        crew_data = [
            {
                "id": crew.id,
                "name": crew.name,
                "level": crew.level,
                "status": crew.status,
                "current_station": crew.current_station,
                "skills": crew.skills,
                "traits": crew.traits,
                "experience": crew.experience
            }
            for crew in self.crew_system.get_all_crew_members()
        ]
        
        # Update the panel
        self.crew_panel.update_crew_data(
            crew_data,
            self.crew_system.station_assignments,
            self.crew_system.get_all_station_efficiencies()
        )
        
        # Pass the custom font to the panel
        self.crew_panel.set_font(self.font)

    def _on_assign_crew(self, crew_id: str, station: str) -> None:
        """Handle crew assignment.

        Args:
            crew_id: ID of the crew member to assign
            station: Station to assign to
        """
        result = self.crew_system.assign_to_station(crew_id, station)
        
        if result["success"]:
            self._add_message(f"Assigned {result['crew_member'].name} to {station}")
        else:
            self._add_message(f"Failed to assign crew: {result['message']}")
        
        self._update_panel_data()

    def _on_unassign_crew(self, crew_id: str) -> None:
        """Handle crew unassignment.

        Args:
            crew_id: ID of the crew member to unassign
        """
        result = self.crew_system.unassign_from_station(crew_id)
        
        if result["success"]:
            self._add_message(f"Unassigned {result['crew_member'].name} from {result['station']}")
        else:
            self._add_message(f"Failed to unassign crew: {result['message']}")
        
        self._update_panel_data()

    def _on_train_crew(self, crew_id: str, skill_to_train: str) -> None:
        """Handle crew training.

        Args:
            crew_id: ID of the crew member to train
            skill_to_train: Skill to train
        """
        # For demo, just train a random skill
        crew_member = self.crew_system.get_crew_member(crew_id)
        if not crew_member:
            return
        
        # Pick a random skill to train instead of using the provided one (for demo purposes)
        selected_skill = random.choice(list(crew_member.skills.keys()))
        
        result = self.crew_system.train_crew(crew_id, selected_skill, 4)
        
        if result["success"]:
            self._add_message(f"{crew_member.name} trained {selected_skill} (+{result['skill_increase']})")
        else:
            self._add_message(f"Failed to train crew: {result['message']}")
        
        self._update_panel_data()

    def _on_rest_crew(self, crew_id: str) -> None:
        """Handle crew rest.

        Args:
            crew_id: ID of the crew member to rest
        """
        result = self.crew_system.rest_crew(crew_id, 8)
        
        if result["rested_crew"]:
            crew_info = result["rested_crew"][0]
            self._add_message(f"{crew_info['name']} is resting for 8 hours")
        else:
            self._add_message("Failed to rest crew")
        
        self._update_panel_data()

    def _on_recruit_crew(self) -> None:
        """Handle crew recruitment."""
        result = self.crew_system.recruit_random_crew(1)
        
        if result["success"]:
            crew = result["recruited"][0]
            self._add_message(f"Recruited new crew member: {crew.name}")
        else:
            self._add_message(f"Failed to recruit: {result['message']}")
        
        self._update_panel_data()

    def _add_message(self, message: str) -> None:
        """Add a message to the message log.

        Args:
            message: Message to add
        """
        self.message_log.append(message)
        
        # Trim log if needed
        if len(self.message_log) > self.max_messages:
            self.message_log = self.message_log[-self.max_messages:]

    def _draw_message_log(self) -> None:
        """Draw the message log."""
        # Draw background
        log_rect = pygame.Rect(50, 10, self.screen.get_width() - 100, 30)
        pygame.draw.rect(self.screen, (20, 20, 30), log_rect)
        pygame.draw.rect(self.screen, (100, 100, 120), log_rect, 1)
        
        # Draw message
        if self.message_log:
            message = self.message_log[-1]
            text = self.font.render(message, True, (200, 200, 200))
            self.screen.blit(text, (log_rect.x + 10, log_rect.y + 5))

    def _update_crew_status(self) -> None:
        """Update crew status based on time passing."""
        # Update every 5 seconds in demo
        self.time_counter += 1
        if self.time_counter < 5 * 60:  # 5 seconds at 60 FPS
            return
            
        self.time_counter = 0
        
        # Update crew status (simulate 1 hour passing)
        results = self.crew_system.update_crew_status(1.0)
        
        # Process and log events
        self._process_level_ups(results["level_ups"])
        self._process_fatigue_increases(results["fatigue_increased"])
        
        # Update panel data
        self._update_panel_data()
        
    def _process_level_ups(self, level_ups):
        """Process level up events and add messages."""
        if not level_ups:
            return
            
        for level_up in level_ups:
            self._add_message(f"{level_up['name']} leveled up to level {level_up['new_level']}!")
    
    def _process_fatigue_increases(self, fatigue_increases):
        """Process fatigue increase events and add messages for exhausted crew."""
        if not fatigue_increases:
            return
            
        for fatigue in fatigue_increases:
            crew = self.crew_system.get_crew_member(fatigue["crew_id"])
            if crew and crew.fatigue > 80:
                self._add_message(f"{fatigue['name']} is becoming exhausted!")

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            # Check for exit conditions
            if self._should_exit(event):
                self.running = False
                logging.info("Exit requested")
                return

            # Pass event to panel with error handling
            try:
                self.crew_panel.handle_input(event)
            except Exception as event_error:
                logging.error(f"Error handling input: {event_error}", exc_info=True)
    
    def _should_exit(self, event):
        """Determine if the application should exit based on the event."""
        return (event.type == pygame.QUIT or 
                (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE))
    
    def _draw_frame(self):
        """Draw a single frame of the application."""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw panel with error handling
        try:
            self.crew_panel.draw(self.screen, self.font)
        except Exception as draw_error:
            logging.error(f"Error drawing crew panel: {draw_error}", exc_info=True)
            # Draw a fallback message instead
            error_text = self.font.render("Error drawing panel", True, (255, 0, 0))
            self.screen.blit(error_text, (100, 100))
        
        # Draw message log and update display
        self._draw_message_log()
        pygame.display.flip()
        self.clock.tick(60)

    def run(self) -> None:
        """Run the demo application."""
        logging.info("Starting demo application")
        try:
            while self.running:
                try:
                    # Process events, update game state, and render
                    self._handle_events()
                    self._update_crew_status()
                    self._draw_frame()
                except Exception as loop_error:
                    logging.error(f"Error in main loop: {loop_error}", exc_info=True)
                    # Don't crash on a single frame error
                    continue

        except Exception as e:
            logging.error(f"Critical error in demo: {e}", exc_info=True)
        finally:
            pygame.quit()
            logging.info("Demo closed")

if __name__ == "__main__":
    demo = ASCIICrewManagementPanelDemo()
    demo.run()
