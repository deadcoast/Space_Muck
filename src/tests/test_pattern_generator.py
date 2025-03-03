"""
Unit tests for pattern_generator module.

This module contains comprehensive tests for the pattern generation utilities.
"""

import unittest
import numpy as np
import math
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pattern_generator import (
    generate_spiral_pattern,
    generate_ring_pattern,
    generate_gradient_pattern,
    generate_void_pattern,
    apply_weighted_patterns,
)


class TestPatternGenerator(unittest.TestCase):
    """Test cases for pattern_generator module."""

    def setUp(self):
        """Set up test fixtures."""
        self.width = 50
        self.height = 40
        self.center = (25, 20)
        self.seed = 42
        np.random.seed(self.seed)

    def test_spiral_pattern_shape(self):
        """Test that spiral pattern has correct shape."""
        pattern = generate_spiral_pattern(self.width, self.height)
        self.assertEqual(pattern.shape, (self.height, self.width))

    def test_spiral_pattern_range(self):
        """Test that spiral pattern values are in [0, 1] range."""
        pattern = generate_spiral_pattern(self.width, self.height)
        self.assertTrue(np.all(pattern >= 0))
        self.assertTrue(np.all(pattern <= 1))

    def test_spiral_pattern_center(self):
        """Test that spiral pattern respects custom center."""
        custom_center = (10, 10)
        pattern = generate_spiral_pattern(self.width, self.height, center=custom_center)

        # Find maximum value which should be near the center
        max_y, max_x = np.unravel_index(np.argmax(pattern), pattern.shape)

        # Should be close to the center (within a few pixels)
        self.assertTrue(abs(max_x - custom_center[0]) < 5)
        self.assertTrue(abs(max_y - custom_center[1]) < 5)

    def test_spiral_pattern_density(self):
        """Test that density parameter affects spiral pattern."""
        low_density = generate_spiral_pattern(self.width, self.height, density=0.2)
        high_density = generate_spiral_pattern(self.width, self.height, density=1.0)

        # Higher density should have more variation
        self.assertTrue(np.std(high_density) > np.std(low_density))

    def test_ring_pattern_shape(self):
        """Test that ring pattern has correct shape."""
        pattern = generate_ring_pattern(self.width, self.height)
        self.assertEqual(pattern.shape, (self.height, self.width))

    def test_ring_pattern_range(self):
        """Test that ring pattern values are in [0, 1] range."""
        pattern = generate_ring_pattern(self.width, self.height)
        self.assertTrue(np.all(pattern >= 0))
        self.assertTrue(np.all(pattern <= 1))

    def test_ring_pattern_num_rings(self):
        """Test that number of rings parameter affects pattern."""
        few_rings = generate_ring_pattern(self.width, self.height, num_rings=3)
        many_rings = generate_ring_pattern(self.width, self.height, num_rings=10)

        # Count local maxima along a radial line from center
        center_x, center_y = self.width // 2, self.height // 2
        few_rings_count = 0
        many_rings_count = 0

        for r in range(1, min(self.width // 2, self.height // 2)):
            x = center_x + r
            y = center_y

            if r > 1 and r < min(self.width // 2, self.height // 2) - 1:
                if (
                    few_rings[y, x] > few_rings[y, x - 1]
                    and few_rings[y, x] > few_rings[y, x + 1]
                ):
                    few_rings_count += 1

                if (
                    many_rings[y, x] > many_rings[y, x - 1]
                    and many_rings[y, x] > many_rings[y, x + 1]
                ):
                    many_rings_count += 1

        # Many rings should have more local maxima than few rings
        self.assertTrue(many_rings_count >= few_rings_count)

    def test_gradient_pattern_shape(self):
        """Test that gradient pattern has correct shape."""
        pattern = generate_gradient_pattern(self.width, self.height)
        self.assertEqual(pattern.shape, (self.height, self.width))

    def test_gradient_pattern_range(self):
        """Test that gradient pattern values are in [0, 1] range."""
        pattern = generate_gradient_pattern(self.width, self.height)
        self.assertTrue(np.all(pattern >= 0))
        self.assertTrue(np.all(pattern <= 1))

    def test_gradient_pattern_direction(self):
        """Test that direction parameter affects gradient pattern."""
        # Horizontal gradient (left to right)
        horizontal = generate_gradient_pattern(self.width, self.height, direction=0)

        # Vertical gradient (top to bottom)
        vertical = generate_gradient_pattern(
            self.width, self.height, direction=math.pi / 2
        )

        # Check that horizontal gradient varies more along x-axis
        h_x_variation = np.mean(np.abs(np.diff(horizontal, axis=1)))
        h_y_variation = np.mean(np.abs(np.diff(horizontal, axis=0)))

        # Check that vertical gradient varies more along y-axis
        v_x_variation = np.mean(np.abs(np.diff(vertical, axis=1)))
        v_y_variation = np.mean(np.abs(np.diff(vertical, axis=0)))

        self.assertTrue(h_x_variation > h_y_variation)
        self.assertTrue(v_y_variation > v_x_variation)

    def test_void_pattern_shape(self):
        """Test that void pattern has correct shape."""
        pattern = generate_void_pattern(self.width, self.height)
        self.assertEqual(pattern.shape, (self.height, self.width))

    def test_void_pattern_range(self):
        """Test that void pattern values are in [0, 1] range."""
        pattern = generate_void_pattern(self.width, self.height)
        self.assertTrue(np.all(pattern >= 0))
        self.assertTrue(np.all(pattern <= 1))

    def test_void_pattern_num_voids(self):
        """Test that number of voids parameter affects pattern."""
        few_voids = generate_void_pattern(self.width, self.height, num_voids=2)
        many_voids = generate_void_pattern(self.width, self.height, num_voids=10)

        # More voids should result in more areas with low values
        few_voids_low = np.sum(few_voids < 0.2)
        many_voids_low = np.sum(many_voids < 0.2)

        self.assertTrue(many_voids_low >= few_voids_low)

    def test_apply_weighted_patterns(self):
        """Test applying weighted patterns."""
        # Define pattern functions and weights
        pattern_functions = [
            generate_spiral_pattern,
            generate_ring_pattern,
            generate_gradient_pattern,
        ]

        pattern_weights = [0.5, 0.3, 0.2]

        # Apply weighted patterns
        combined = apply_weighted_patterns(
            self.width, self.height, pattern_functions, pattern_weights
        )

        self._pattern_handler(combined)

    def test_apply_weighted_patterns_with_args(self):
        """Test applying weighted patterns with custom arguments."""
        # Define pattern functions and weights
        pattern_functions = [generate_spiral_pattern, generate_ring_pattern]

        pattern_weights = [0.6, 0.4]

        pattern_args = [
            {"density": 0.8, "rotation": 2.0},
            {"num_rings": 8, "falloff": 0.3},
        ]

        # Apply weighted patterns with custom args
        combined = apply_weighted_patterns(
            self.width, self.height, pattern_functions, pattern_weights, pattern_args
        )

        self._pattern_handler(combined)

    def _pattern_handler(self, combined):
        self.assertEqual(combined.shape, (self.height, self.width))
        self.assertTrue(np.all(combined >= 0))
        self.assertTrue(np.all(combined <= 1))


if __name__ == "__main__":
    unittest.main()
