"""
Test for ASCII Box drawing functionality.
Tests that the box drawing characters render correctly.
"""

# Standard library imports
import os
import sys
import unittest

# Local application imports
import pygame

# Third-party library imports


# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Use absolute imports for tests

# Import the draw_text function directly from the module
from src.ui.ui_element.ascii_box import ASCIIBox, draw_text
from src.ui.ui_style import UIStyle


class TestASCIIBoxDrawing(unittest.TestCase):
    """Test the ASCII Box drawing functionality."""

    def setUp(self):
        """Set up the test environment."""
        pygame.init()
        self.surface = pygame.Surface((400, 300))
        self.box = ASCIIBox(
            x=50, y=50, width=300, height=200, style=UIStyle.SINGLE, title="Test Box"
        )

    def tearDown(self):
        """Clean up after the test."""
        pygame.quit()

    def test_box_initialization(self):
        """Test that the box initializes with correct properties."""
        self.assertEqual(self.box.x, 50)
        self.assertEqual(self.box.y, 50)
        self.assertEqual(self.box.width, 300)
        self.assertEqual(self.box.height, 200)
        self.assertEqual(self.box.style, UIStyle.SINGLE)
        self.assertEqual(self.box.title, "Test Box")

    def test_box_drawing(self):
        """Test that the box draws without errors."""
        try:
            self.box.draw(self.surface)
            # Test passes if no exception is raised
        except Exception as e:
            self.fail(f"Box drawing raised exception: {e}")

    def test_draw_text_import(self):
        """Test that the draw_text function is properly imported and works."""
        try:
            draw_text(surface=self.surface, text="Test Text", x=100, y=100, size=18)
            # Test passes if no exception is raised
        except Exception as e:
            self.fail(f"draw_text raised exception: {e}")


if __name__ == "__main__":
    unittest.main()
