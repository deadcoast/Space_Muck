"""
Unit tests for value_generator module.

This module contains comprehensive tests for the value generation utilities.
"""

# Standard library imports
import os
import sys
import unittest

# Third-party library imports
import numpy as np

# Local application imports
from utils.value_generator import (  # Add parent directory to path to import modules
    add_value_clusters,
    generate_rare_resource_distribution,
    generate_value_distribution,
)

# Add the current directory to path to ensure proper importing
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class TestValueGenerator(unittest.TestCase):
    """Test cases for value_generator module."""

    def setUp(self):
        """Set up test fixtures."""
        self.width = 50
        self.height = 40
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)

        # Create test grids
        self.binary_grid = np.zeros((self.height, self.width), dtype=int)
        self.binary_grid[10:30, 15:35] = 1  # Create a rectangular region of 1s

        # Create noise grids using modern numpy.random.Generator API
        rng = np.random.default_rng(42)  # Use a fixed seed for reproducibility
        self.base_noise = rng.random((self.height, self.width))
        self.rare_noise = rng.random((self.height, self.width))
        self.precious_noise = rng.random((self.height, self.width))
        self.anomaly_noise = rng.random((self.height, self.width))

    def test_generate_value_distribution_shape(self):
        """Test that value distribution has correct shape."""
        values = generate_value_distribution(self.binary_grid, self.base_noise)
        self.assertEqual(values.shape, (self.height, self.width))

    def test_generate_value_distribution_zeros(self):
        """Test that value distribution preserves zeros."""
        values = generate_value_distribution(self.binary_grid, self.base_noise)
        # Where binary grid is 0, values should be 0
        zero_mask = self.binary_grid == 0
        np.testing.assert_array_equal(
            values[zero_mask],
            np.zeros_like(values[zero_mask]),
            "Zero areas in binary grid should result in zero values",
        )

    def test_generate_value_distribution_min_value(self):
        """Test that value distribution respects minimum value."""
        min_value = 5
        values = generate_value_distribution(
            self.binary_grid, self.base_noise, min_value=min_value
        )
        # Where binary grid is 1, values should be at least min_value
        nonzero_mask = self.binary_grid > 0
        self.assertTrue(
            (values[nonzero_mask] >= min_value).all(),
            f"All non-zero values should be at least {min_value}",
        )

    def test_generate_value_distribution_mean(self):
        """Test that value distribution approximates requested mean."""
        value_mean = 10.0
        values = generate_value_distribution(
            self.binary_grid, self.base_noise, value_mean=value_mean
        )
        # Calculate mean of non-zero values
        nonzero_mask = self.binary_grid > 0
        actual_mean = np.mean(values[nonzero_mask])
        # Mean should be approximately as requested (within reasonable bounds)
        self.assertLess(abs(actual_mean - value_mean), 3.0)

    def test_add_value_clusters(self):
        """Test adding value clusters to a grid."""
        # Generate initial values
        values = generate_value_distribution(self.binary_grid, self.base_noise)

        # Add clusters
        clustered_values = add_value_clusters(
            values, self.binary_grid, num_clusters=3, cluster_value_multiplier=2.0
        )

        # Check shape
        self.assertEqual(clustered_values.shape, (self.height, self.width))

        # Clustered values should have higher maximum than original
        self.assertGreater(np.max(clustered_values), np.max(values))

        # Zero areas should remain zero
        zero_mask = self.binary_grid == 0
        np.testing.assert_array_equal(
            clustered_values[zero_mask],
            np.zeros_like(clustered_values[zero_mask]),
            "Zero areas should remain zero after adding clusters",
        )

    def test_add_value_clusters_zero_clusters(self):
        """Test adding zero clusters."""
        # Generate initial values
        values = generate_value_distribution(self.binary_grid, self.base_noise)

        # Add zero clusters
        clustered_values = add_value_clusters(values, self.binary_grid, num_clusters=0)

        # Should be identical to input
        np.testing.assert_array_equal(clustered_values, values)

    def test_generate_rare_resource_distribution_shape(self):
        """Test that rare resource distribution has correct shape."""
        rare_resources = generate_rare_resource_distribution(
            self.binary_grid, self.rare_noise, self.precious_noise, self.anomaly_noise
        )
        self.assertEqual(rare_resources.shape, (self.height, self.width))

    def test_generate_rare_resource_distribution_range(self):
        """Test that rare resource values are in valid range."""
        rare_resources = generate_rare_resource_distribution(
            self.binary_grid, self.rare_noise, self.precious_noise, self.anomaly_noise
        )
        # Values should be 0, 1, 2, or 3
        unique_values = np.unique(rare_resources)
        expected_values = np.array([0, 1, 2, 3])
        self.assertTrue(
            np.all(np.isin(unique_values, expected_values)),
            f"Rare resource values should only be in {expected_values}, but found {unique_values}",
        )

    def test_generate_rare_resource_distribution_zeros(self):
        """Test that rare resource distribution preserves zeros."""
        rare_resources = generate_rare_resource_distribution(
            self.binary_grid, self.rare_noise, self.precious_noise, self.anomaly_noise
        )
        # Where binary grid is 0, rare resources should be 0
        zero_mask = self.binary_grid == 0
        np.testing.assert_array_equal(
            rare_resources[zero_mask],
            np.zeros_like(rare_resources[zero_mask]),
            "Zero areas in binary grid should have zero rare resources",
        )

    def test_generate_rare_resource_distribution_frequency(self):
        """Test that rare resource frequency respects parameters."""
        # Set high chance for testing
        rare_chance = 0.5
        precious_factor = 0.5  # 50% of rare chance
        anomaly_factor = 0.2  # 20% of rare chance

        rare_resources = generate_rare_resource_distribution(
            self.binary_grid,
            self.rare_noise,
            self.precious_noise,
            self.anomaly_noise,
            rare_chance=rare_chance,
            precious_factor=precious_factor,
            anomaly_factor=anomaly_factor,
        )

        # Count occurrences
        nonzero_mask = self.binary_grid > 0
        total_cells = np.sum(nonzero_mask)

        rare_count = np.sum(rare_resources[nonzero_mask] == 1)
        precious_count = np.sum(rare_resources[nonzero_mask] == 2)
        anomaly_count = np.sum(rare_resources[nonzero_mask] == 3)

        # Calculate observed frequencies
        rare_freq = rare_count / total_cells
        precious_freq = precious_count / total_cells
        anomaly_freq = anomaly_count / total_cells

        # Check that frequencies are within reasonable bounds
        # (using wide bounds due to randomness)
        self.assertLess(abs(rare_freq - rare_chance), 0.2)
        self.assertLess(abs(precious_freq - (rare_chance * precious_factor)), 0.2)
        self.assertLess(abs(anomaly_freq - (rare_chance * anomaly_factor)), 0.2)


if __name__ == "__main__":
    unittest.main()
