import contextlib
import time
import random
import curses
from ui.ui_element.ui_style import UIStyle
from ui.ui_element.ui_menu import Menu
from ui.ui_element.ui_fleet_display import FleetDisplay
from ui.ui_element.ui_asteroid_field_visualizer import AsteroidFieldVisualizer
from ui.ui_element.ui_symbiote_evolution_monitor import SymbioteEvolutionMonitor
from ui.ui_element.ui_mining_status import MiningStatus
from ui.ui_helpers.animation_helper import AnimationStyle


# Define standard colors for UI components
COLOR_TEXT = (220, 220, 220)  # Standard text color
COLOR_BG = (20, 20, 30)  # Standard background color
COLOR_HIGHLIGHT = (180, 180, 255)  # Standard highlight color


class SpaceMuckMainUI:
    """Main UI manager for Space Muck game"""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()
        self.ui_elements = {}
        self.active_element = None
        self.running = True

        # Initialize colors if terminal supports it
        self._init_colors()

        # Create UI elements
        self._init_ui_elements()

    def _init_colors(self):
        """Initialize color pairs for the UI"""
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()

            # Define color pairs for different UI elements
            curses.init_pair(1, curses.COLOR_GREEN, -1)  # Symbiotic elements
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Asteroid elements
            curses.init_pair(3, curses.COLOR_CYAN, -1)  # Mechanical elements
            curses.init_pair(4, curses.COLOR_MAGENTA, -1)  # Quantum elements
            curses.init_pair(5, curses.COLOR_BLUE, -1)  # Fleet elements

    def _init_ui_elements(self):
        """Create and initialize all UI elements"""
        # Create main menu
        main_menu = Menu(
            2,
            2,
            30,
            20,
            "SPACE MUCK",
            [
                "Start Expedition",
                "Fleet Management",
                "Research",
                "Trading",
                "Settings",
                "Exit",
            ],
            UIStyle.MECHANICAL,
        )
        self.ui_elements["main_menu"] = main_menu
        main_menu.active = True
        self.active_element = "main_menu"

        # Fleet display
        fleet_data = {"miner": 3, "fighter": 2, "capital": 1}
        fleet_display = FleetDisplay(34, 2, 44, 12, fleet_data, UIStyle.FLEET)
        self.ui_elements["fleet_display"] = fleet_display

        # Asteroid field visualizer
        asteroid_field = AsteroidFieldVisualizer(34, 15, 44, 20, 0.3, UIStyle.ASTEROID)
        self.ui_elements["asteroid_field"] = asteroid_field

        # Symbiote evolution monitor
        symbiote_monitor = SymbioteEvolutionMonitor(2, 23, 30, 25, UIStyle.SYMBIOTIC)
        self.ui_elements["symbiote_monitor"] = symbiote_monitor

        # Mining status
        resources = {
            "Iron": 256.0,
            "Copper": 128.5,
            "Gold": 32.1,
            "Titanium": 64.7,
            "Quantum": 15.3,
        }
        mining_status = MiningStatus(80, 2, 38, 20, resources, UIStyle.MECHANICAL)
        self.ui_elements["mining_status"] = mining_status

        # Set some initial extraction rates for demo
        mining_status.extraction_rate = {
            "Iron": 1.2,
            "Copper": 0.8,
            "Gold": 0.15,
            "Titanium": -0.3,
            "Quantum": 0.05,
        }

    def draw(self):
        """Draw all UI elements"""
        self.stdscr.clear()

        # Draw fancy header
        self._draw_header()

        # Draw all UI elements
        for name, element in self.ui_elements.items():
            element.draw(self.stdscr)

        self.stdscr.refresh()

    def _draw_header(self):
        """Draw the game header with ASCII art"""
        header_text = [
            "╭─────────────────────────────────────────────────────╮",
            "│                                                     │",
            "│ ░░      ░░░       ░░░░      ░░░░      ░░░        ░░ │",
            "│ ▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒ │",
            "│ ▓▓      ▓▓▓       ▓▓▓  ▓▓▓▓  ▓▓  ▓▓▓▓▓▓▓▓       ▓▓▓`│",
            "│ ███████  ██  ████████        ██  ████  ██  ████████ │",
            "│ ██      ███  ████████  ████  ███      ███        ██ │",
            "│ ███████████████████████████████████████████████████ |",
            "|                                                     |",
            "|     ░░░  ░░░░  ░░  ░░░░  ░░░      ░░░  ░░░░  ░      |",
            "|     ▒▒▒   ▒▒   ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒  ▒▒      |",
            "|     ▓▓▓        ▓▓  ▓▓▓▓  ▓▓  ▓▓▓▓▓▓▓▓     ▓▓▓▓      |",
            "|     ███  █  █  ██  ████  ██  ████  ██  ███  ██      |",
            "|     ███  ████  ███      ████      ███  ████  █      |",
            "|     ██████████████████████████████████████████      |",
            "╰─────────────────────────────────────────────────────╯",
        ]

        for i, line in enumerate(header_text):
            with contextlib.suppress(curses.error):
                self.stdscr.addstr(i, 0, line)

    def handle_input(self):
        """Handle user input"""
        key = self.stdscr.getch()

        # ESC key to exit
        if key == 27:
            self.running = False
            return

        # Pass input to active element
        if self.active_element == "main_menu":
            result = self.ui_elements["main_menu"].handle_input(key)
            if result == "Exit":
                self.running = False
            elif result == "Fleet Management":
                # Demo: Update fleet when selecting this option
                new_fleet = {"miner": 4, "fighter": 3, "capital": 1}
                self.ui_elements["fleet_display"].update_fleet(new_fleet)
            elif result == "Research":
                # Demo: Update symbiote evolution when selecting this option
                self.ui_elements["symbiote_monitor"].update_evolution(random.random())
            elif result == "Trading":
                # Demo: Update mining status when selecting this option
                current_resources = self.ui_elements["mining_status"].resources
                new_resources = {
                    k: v + random.uniform(-10, 20) for k, v in current_resources.items()
                }
                new_rates = {
                    k: random.uniform(-0.5, 1.5) for k in current_resources.keys()
                }
                self.ui_elements["mining_status"].update_resources(
                    new_resources, new_rates, random.uniform(0.5, 0.95)
                )

    def main_loop(self):
        """Main game loop"""
        # Perform initial animations
        for name, element in self.ui_elements.items():
            # Use different animation styles for different elements
            if name == "main_menu":
                element.animate(self.stdscr, AnimationStyle.WARP)
            elif name == "fleet_display":
                element.animate(self.stdscr, AnimationStyle.FRACTAL)
            elif name == "asteroid_field":
                element.animate(self.stdscr, AnimationStyle.CELLULAR)
            elif name == "symbiote_monitor":
                element.animate(self.stdscr, AnimationStyle.QUANTUM_FLUX)
            elif name == "mining_status":
                element.animate(self.stdscr, AnimationStyle.MINERAL_GROWTH)

        # Main game loop
        while self.running:
            self.draw()
            self.handle_input()

            # Update asteroid field occasionally
            if random.random() < 0.05:
                self.ui_elements["asteroid_field"]._evolve_grid()

            # Animate fleet display on each frame
            self.ui_elements["fleet_display"].animation_frame += 1

            # Sleep briefly to control frame rate
            time.sleep(0.05)


def run_space_muck_ui():
    """Initialize and run the Space Muck UI"""

    def main(stdscr):
        # Set up terminal
        curses.curs_set(0)  # Hide cursor
        stdscr.timeout(100)  # Non-blocking input with 100ms timeout

        # Create and run UI
        ui = SpaceMuckMainUI(stdscr)
        ui.main_loop()

    curses.wrapper(main)


if __name__ == "__main__":
    run_space_muck_ui()
