"""
ui_main.py

Main module for the Space Muck UI.
"""

# Standard library imports
import contextlib
import curses
import random
import time
from enum import Enum, auto

# Third-party library imports

# Local application imports
# No imports from config needed here


class AnimationStyle(Enum):
    WARP = auto()
    FRACTAL = auto()
    CELLULAR = auto()
    QUANTUM_FLUX = auto()
    MINERAL_GROWTH = auto()


class UIStyle(Enum):
    MECHANICAL = auto()
    SYMBIOTIC = auto()
    QUANTUM = auto()
    FLEET = auto()
    ASTEROID = auto()


class Menu:
    def __init__(self, x, y, width, height, title, options, style):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title = title
        self.options = options
        self.style = style
        self.selected_index = 0
        self.active = False

    def handle_input(self, key):
        if key == curses.KEY_UP:
            self.selected_index = max(0, self.selected_index - 1)
        elif key == curses.KEY_DOWN:
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
        elif key in (curses.KEY_ENTER, 10):
            return self.options[self.selected_index]
        return None

    def render(self, stdscr):
        # Drawing the menu with the title and options
        # Title
        stdscr.addstr(self.y, self.x, self.title, curses.A_BOLD)
        # Options
        for i, option in enumerate(self.options):
            attr = curses.A_REVERSE if i == self.selected_index else 0
            stdscr.addstr(self.y + 2 + i, self.x + 2, option, attr)


class SymbioteEvolutionMonitor:
    def __init__(self, x, y, width, height, style):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.style = style

    def animate(self, stdscr, style):
        # Animation implementation would go here
        pass


class FleetDisplay:
    def __init__(self, x, y, width, height, fleet_data, style):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.fleet_data = fleet_data
        self.style = style

    def animate(self, stdscr, frame):
        # Animation implementation would go here
        pass


class AsteroidFieldVisualizer:
    def __init__(self, x, y, width, height, density, style):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.density = density
        self.style = style

    def animate(self, stdscr, frame):
        # Animation implementation would go here
        pass


class MiningStatus:
    def __init__(self, x, y, width, height, resources, style):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.resources = resources
        self.style = style
        self.extraction_rate = {}

    def animate(self, stdscr, style):
        # Animation implementation would go here
        pass


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
