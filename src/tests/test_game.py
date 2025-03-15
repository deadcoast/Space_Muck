"""
Test module for the Game class state management functionality.
"""

# Standard library imports
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Local application imports
from events.event_bus import EventBus, clear_event_buses
from main import Game, GameStateError, InvalidStateTransitionError  # noqa: E402

# Third-party library imports


# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import event bus first to ensure it's available for mocking

# Import game classes

# Mock dependencies
sys.modules["pygame"] = MagicMock()
sys.modules["pygame.display"] = MagicMock()
sys.modules["pygame.font"] = MagicMock()
sys.modules["pygame.mixer"] = MagicMock()
sys.modules["pygame.macosx"] = MagicMock()
sys.modules["perlin_noise"] = MagicMock()
sys.modules["sklearn.cluster"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["scipy.ndimage"] = MagicMock()

# Mock config module
config_mock = MagicMock()
config_mock.COLOR_UI_BUTTON = (100, 100, 100)
config_mock.COLOR_UI_BUTTON_HOVER = (150, 150, 150)
config_mock.COLOR_UI_BUTTON_ACTIVE = (200, 200, 200)
config_mock.COLOR_UI_TEXT = (255, 255, 255)
config_mock.COLOR_UI_BORDER = (80, 80, 80)
config_mock.COLOR_UI_BACKGROUND = (50, 50, 50)
config_mock.COLOR_UI_HEADER = (120, 120, 120)
config_mock.WINDOW_WIDTH = 1280
config_mock.WINDOW_HEIGHT = 720
config_mock.GRID_WIDTH = 100
config_mock.GRID_HEIGHT = 100
config_mock.GAME_CONFIG = {
    "version": "1.0.0",
    "states": ["menu", "play", "pause", "shop"],
    "valid_transitions": {
        "menu": ["play", "shop"],
        "play": ["pause", "shop", "menu"],
        "pause": ["play", "menu"],
        "shop": ["play", "menu"],
    },
}
sys.modules["config"] = config_mock

# Create test event bus
test_bus = EventBus("test")

# Mock event bus module
sys.modules["events.event_bus"] = MagicMock(
    get_event_bus=MagicMock(return_value=test_bus),
    EventBus=EventBus,
    clear_event_buses=clear_event_buses,
)

# Mock UI components
shop_mock = MagicMock()
notifier_mock = MagicMock()
renderer_mock = MagicMock()

sys.modules["ui.shop"] = MagicMock(Shop=MagicMock(return_value=shop_mock))
sys.modules["ui.notification_manager"] = MagicMock(
    NotificationManager=MagicMock(return_value=notifier_mock)
)
sys.modules["ui.asteroid_field_renderer"] = MagicMock(
    AsteroidFieldRenderer=MagicMock(return_value=renderer_mock)
)

# Mock game components
field_mock = MagicMock()
field_mock.width = 100
field_mock.height = 100
field_mock.entities = []
field_mock.races = []

player_mock = MagicMock()
player_mock.x = 50
player_mock.y = 50
player_mock.currency = 1000
player_mock.mining_power = 1
player_mock.health = 100

sys.modules["generators.asteroid_field"] = MagicMock(
    AsteroidField=MagicMock(return_value=field_mock)
)
sys.modules["entities.player"] = MagicMock(Player=MagicMock(return_value=player_mock))


class TestGameStateMachine(unittest.TestCase):
    """Test cases for the Game class state management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset test bus state
        test_bus._subscribers.clear()
        test_bus._event_history.clear()

        # Create a game instance for testing
        with patch("pygame.display.set_mode"):
            self.game = Game()

        # Verify event bus subscription
        self.assertIn("state_change", test_bus._subscribers)

    def _verify_state_tracking_data(self, state_name, expected_count=0):
        """Helper method to verify state tracking data."""
        self.assertIn(state_name, self.game._state_timestamps)
        self.assertIn(state_name, self.game._state_transition_counts)
        self.assertEqual(self.game._state_transition_counts[state_name], expected_count)

    def test_initial_state(self):
        """Test that Game has the correct initial state configuration."""
        # Test initial state
        self.assertEqual(self.game.state, "menu")
        self.assertIsNone(self.game.previous_state)
        self.assertTrue(self.game._state_valid)

        # Test state tracking initialization
        self.assertIsInstance(self.game._state_timestamps, dict)
        self.assertIsInstance(self.game._state_transition_counts, dict)
        self.assertIsInstance(self.game._state_history, list)

        # Test that all valid states have timestamps and counts
        for state in ["menu", "play", "pause", "shop"]:
            self._verify_state_tracking_data(state)

    def test_valid_state_transitions(self):
        """Test valid state transitions."""
        # Test menu -> play transition
        self.game.change_state("play")
        self.assertEqual(self.game.state, "play")
        self.assertEqual(self.game.previous_state, "menu")
        self.assertTrue(self.game._state_valid)

        # Verify event bus published state change
        self.assertEqual(len(test_bus._event_history), 1)
        event = test_bus._event_history[0]
        self.assertEqual(event["type"], "state_change")
        self.assertEqual(event["data"]["from_state"], "menu")
        self.assertEqual(event["data"]["to_state"], "play")

        # Test play -> pause transition
        self.game.change_state("pause")
        self.assertEqual(self.game.state, "pause")
        self.assertEqual(self.game.previous_state, "play")
        self.assertTrue(self.game._state_valid)

        # Verify second state change event
        self.assertEqual(len(test_bus._event_history), 2)
        event = test_bus._event_history[1]
        self.assertEqual(event["type"], "state_change")
        self.assertEqual(event["data"]["from_state"], "play")
        self.assertEqual(event["data"]["to_state"], "pause")

        # Test pause -> play transition
        self.game.change_state("play")
        self.assertEqual(self.game.state, "play")
        self.assertEqual(self.game.previous_state, "pause")
        self.assertTrue(self.game._state_valid)

        # Verify third state change event
        self.assertEqual(len(test_bus._event_history), 3)
        event = test_bus._event_history[2]
        self.assertEqual(event["type"], "state_change")
        self.assertEqual(event["data"]["from_state"], "pause")
        self.assertEqual(event["data"]["to_state"], "play")

    def test_invalid_state_transitions(self):
        """Test that invalid state transitions raise appropriate errors."""
        # Test invalid menu -> pause transition
        with self.assertRaises(InvalidStateTransitionError):
            self.game.change_state("pause")

        # Verify no event was published for invalid transition
        self.assertEqual(len(test_bus._event_history), 0)

        # Test transition to non-existent state
        with self.assertRaises(GameStateError):
            self.game.change_state("nonexistent_state")

        # Verify still no events published
        self.assertEqual(len(test_bus._event_history), 0)

        # Test that state remains unchanged after failed transition
        self.assertEqual(self.game.state, "menu")
        self.assertIsNone(self.game.previous_state)

    def _verify_state_history_entry(self, latest_entry, from_state, to_state):
        """Helper method to verify a state history entry."""
        self.assertEqual(latest_entry["from_state"], from_state)
        self.assertEqual(latest_entry["to_state"], to_state)
        self.assertIsInstance(latest_entry["timestamp"], float)
        self.assertIsInstance(latest_entry["frame"], int)
        self.assertIsInstance(latest_entry["game_time"], float)

    def test_state_history_tracking(self):
        """Test that state transitions are properly recorded in history."""
        # Make a series of valid transitions
        transitions = [
            ("play", "menu"),  # menu -> play
            ("pause", "play"),  # play -> pause
            ("play", "pause"),  # pause -> play
            ("menu", "play"),  # play -> menu
        ]

        for new_state, expected_from_state in transitions:
            self.game.change_state(new_state)

            # Check history entry
            latest = self.game._state_history[-1]
            self._verify_state_history_entry(latest, expected_from_state, new_state)

            # Check transition count
            self._verify_state_tracking_data(new_state, expected_count=1)

    def test_state_timing(self):
        """Test that state timing information is accurate."""
        start_time = time.time()

        # Change to play state
        self.game.change_state("play")
        time.sleep(0.1)  # Small delay

        # Get debug info
        debug_info = self.game.get_state_debug_info()

        # Check timing
        self.assertGreater(debug_info["state_times"]["play"], 0.0)
        self.assertLess(debug_info["state_times"]["play"], time.time() - start_time)

        # Check that other states have 0 time
        for state in ["menu", "pause", "shop"]:
            self.assertEqual(debug_info["state_times"][state], 0.0)

    def test_state_validation(self):
        """Test state validation functionality."""
        # Test validation during valid transition
        self.assertTrue(self.game._validate_state_transition("play"))

        # Test validation with invalid state
        with self.assertRaises(GameStateError):
            self.game._validate_state_transition("invalid_state")

        # Test validation with invalid transition
        self.game.state = "menu"
        with self.assertRaises(InvalidStateTransitionError):
            self.game._validate_state_transition("pause")

    def test_debug_info(self):
        """Test that debug info contains all required information."""
        # Make some state transitions
        self.game.change_state("play")
        self.game.change_state("pause")

        # Get debug info
        debug_info = self.game.get_state_debug_info()

        # Check structure
        self.assertIn("current_state", debug_info)
        self.assertIn("previous_state", debug_info)
        self.assertIn("state_valid", debug_info)
        self.assertIn("state_times", debug_info)
        self.assertIn("transition_counts", debug_info)
        self.assertIn("state_history", debug_info)
        self.assertIn("performance", debug_info)

        # Check performance metrics
        perf = debug_info["performance"]
        self.assertIn("current_fps", perf)
        self.assertIn("avg_fps", perf)
        self.assertIn("frame_counter", perf)
        self.assertIn("game_time", perf)


if __name__ == "__main__":
    unittest.main()
