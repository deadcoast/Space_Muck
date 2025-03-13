"""
Test module for ASCIIBox event integration.

This module provides tests to verify that ASCIIBox components
properly integrate with the UI event system.
"""

# Standard library imports
import time

# Third-party library imports

# Local application imports
from unittest.mock import MagicMock, patch
import unittest

# No typing imports needed

from ui.ui_helpers.ascii_box_event_helper import (
    register_ascii_box,
    unregister_ascii_box,
    add_click_handler,
    add_hover_handlers,
    create_interactive_box,
    handle_mouse_events,
    get_box_by_id,
    is_registered_with_events,
)

class TestASCIIBoxEventIntegration(unittest.TestCase):
    """Test case for ASCIIBox event integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test box
        self.box = ASCIIBox(5, 5, 20, 10, "Test Box")

        # Mock event handlers
        self.click_handler = MagicMock()
        self.hover_enter_handler = MagicMock()
        self.hover_leave_handler = MagicMock()

    def test_register_ascii_box(self):
        """Test registering an ASCIIBox with the event system."""
        # Register box
        box_id = register_ascii_box(self.box)

        # Check that box was registered
        self.assertIsNotNone(box_id)
        self.assertGreater(len(box_id), 0)
        self.assertEqual(self.box.component_id, box_id)

    def test_register_with_custom_id(self):
        """Test registering an ASCIIBox with a custom ID."""
        # Register box with custom ID
        custom_id = "test_box_123"
        box_id = register_ascii_box(self.box, custom_id)

        # Check that box was registered with custom ID
        self.assertEqual(box_id, custom_id)
        self.assertEqual(self.box.component_id, custom_id)

    @patch("ui.ui_helpers.ascii_box_event_helper.unregister_from_events")
    def test_unregister_ascii_box(self, mock_unregister):
        """Test unregistering an ASCIIBox from the event system."""
        # Set up mock
        mock_unregister.return_value = True

        # Register box with a known ID for testing
        test_id = "test_box_id_123"
        self.box.component_id = test_id
        
        # Unregister box
        result = unregister_ascii_box(self.box)

        # Check that box was unregistered
        self.assertTrue(result)
        mock_unregister.assert_called_once_with(test_id)

    def test_add_click_handler(self):
        """Test adding a click handler to an ASCIIBox."""
        # Add click handler
        result = add_click_handler(self.box, self.click_handler)

        # Check that handler was added
        self.assertTrue(result)
        self.assertTrue(self.box.is_clickable)

        # Simulate click event
        event_data = UIEventData(
            event_type=UIEventType.MOUSE_CLICK,
            source_id=self.box.component_id,
            data={"x": 10, "y": 10, "timestamp": time.time()},
        )

        # Trigger handler directly
        self.box._event_handlers[UIEventType.MOUSE_CLICK][0](event_data)

        # Check that handler was called
        self.click_handler.assert_called_once_with(event_data)

    def test_add_hover_handlers(self):
        """Test adding hover handlers to an ASCIIBox."""
        # Add hover handlers
        result = add_hover_handlers(
            self.box, self.hover_enter_handler, self.hover_leave_handler
        )

        # Check that handlers were added
        self.assertTrue(result)

        # Simulate hover enter event
        enter_event_data = UIEventData(
            event_type=UIEventType.MOUSE_ENTER,
            source_id=self.box.component_id,
            data={"x": 10, "y": 10, "timestamp": time.time()},
        )

        # Trigger enter handler directly
        self.box._event_handlers[UIEventType.MOUSE_ENTER][0](enter_event_data)

        # Check that enter handler was called
        self.hover_enter_handler.assert_called_once_with(enter_event_data)

        # Simulate hover leave event
        leave_event_data = UIEventData(
            event_type=UIEventType.MOUSE_LEAVE,
            source_id=self.box.component_id,
            data={"x": 30, "y": 30, "timestamp": time.time()},
        )

        # Trigger leave handler directly
        self.box._event_handlers[UIEventType.MOUSE_LEAVE][0](leave_event_data)

        # Check that leave handler was called
        self.hover_leave_handler.assert_called_once_with(leave_event_data)

    def test_create_interactive_box(self):
        """Test creating an interactive box with event handlers."""
        # Create interactive box
        box, box_id = create_interactive_box(
            x=10,
            y=10,
            width=30,
            height=15,
            title="Interactive Box",
            on_click=self.click_handler,
            on_hover_enter=self.hover_enter_handler,
            on_hover_leave=self.hover_leave_handler,
        )

        # Check that box was created with correct properties
        self.assertIsInstance(box, ASCIIBox)
        self.assertEqual(box.x, 10)
        self.assertEqual(box.y, 10)
        self.assertEqual(box.width, 30)
        self.assertEqual(box.height, 15)
        self.assertEqual(box.title, "Interactive Box")
        self.assertEqual(box.component_id, box_id)

        # Check that handlers were registered
        self.assertTrue(box.is_clickable)
        self.assertIn(UIEventType.MOUSE_CLICK, box._event_handlers)
        self.assertIn(UIEventType.MOUSE_ENTER, box._event_handlers)
        self.assertIn(UIEventType.MOUSE_LEAVE, box._event_handlers)

    def test_handle_mouse_events(self):
        """Test handling mouse events for multiple boxes."""
        # Create test boxes
        boxes = [
            ASCIIBox(5, 5, 10, 5, "Box 1"),
            ASCIIBox(20, 5, 10, 5, "Box 2"),
            ASCIIBox(35, 5, 10, 5, "Box 3"),
        ]

        # Register boxes
        for i, box in enumerate(boxes):
            register_ascii_box(box, f"box_{i}")

            # Make box handle mouse events
            box.handle_mouse_event = MagicMock(
                return_value=i == 1
            )  # Only box 1 handles

        # Test mouse event handling
        mouse_pos = (25, 7)  # Position over Box 2
        char_size = (1, 1)

        # Handle mouse events
        handled_by = handle_mouse_events(boxes, mouse_pos, "click", char_size)

        # Check that correct box handled event
        self.assertEqual(len(handled_by), 1)
        self.assertEqual(handled_by[0], "box_1")

        # Check that handle_mouse_event was called for each box
        for box in boxes:
            box.handle_mouse_event.assert_called_once_with(
                "click", mouse_pos, char_size
            )

    @patch("ui.ui_helpers.ascii_box_event_helper.ComponentRegistry")
    def test_get_box_by_id(self, mock_registry_class):
        """Test getting an ASCIIBox by its component ID."""
        # Set up mock
        mock_registry = MagicMock()
        mock_registry_class.get_instance.return_value = mock_registry
        mock_registry.get_component.return_value = self.box

        # Get box by ID
        test_id = "test_box_id"
        result = get_box_by_id(test_id)

        # Check that correct box was returned
        self.assertEqual(result, self.box)
        mock_registry.get_component.assert_called_once_with(test_id)
        
    @patch("ui.ui_helpers.ascii_box_event_helper.ComponentRegistry")
    def test_is_registered_with_events(self, mock_registry_class):
        """Test checking if an ASCIIBox is registered with the event system."""
        # Set up mock
        mock_registry = MagicMock()
        mock_registry_class.get_instance.return_value = mock_registry
        mock_registry.is_registered.return_value = True
        
        # Set a known component ID for testing
        test_id = "test_box_id_456"
        self.box.component_id = test_id
        
        # Check if box is registered
        result = is_registered_with_events(self.box)
        
        # Check result
        self.assertTrue(result)
        mock_registry.is_registered.assert_called_once_with(test_id)
        
    def test_is_registered_with_events_no_id(self):
        """Test checking if an ASCIIBox without ID is registered with events."""
        # Create box without ID
        box = ASCIIBox(5, 5, 20, 10, "Test Box")
        box.component_id = None
        
        # Check if box is registered
        result = is_registered_with_events(box)
        
        # Should return False for box without ID
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
