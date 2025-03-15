"""
Test the mock data generation functions.
"""

import unittest
import numpy as np

from tests.tools.mock_data_test import (
    create_mock_field_data,
    create_mock_race,
    create_mock_shop_upgrade,
    _create_asteroid_clusters,
    _create_rare_grid,
    _create_energy_grid,
    _create_entity_grid,
    _add_circular_cluster,
)


class TestMockData(unittest.TestCase):
    """Test the mock data generation functions."""

    def test_create_mock_field_data(self):
        """Test the create_mock_field_data function."""
        # Test with default parameters
        data = create_mock_field_data()
        self.assertIsInstance(data, dict)
        self.assertIn("grid", data)
        self.assertIn("rare_grid", data)
        self.assertIn("energy_grid", data)
        self.assertIn("entity_grid", data)

        # Check shapes
        self.assertEqual(data["grid"].shape, (80, 100))
        self.assertEqual(data["rare_grid"].shape, (80, 100))
        self.assertEqual(data["energy_grid"].shape, (80, 100))
        self.assertEqual(data["entity_grid"].shape, (80, 100))

        # Test with custom parameters
        data = create_mock_field_data(width=50, height=40)
        self.assertEqual(data["grid"].shape, (40, 50))

    def test_create_asteroid_clusters(self):
        """Test the _create_asteroid_clusters function."""
        grid = _create_asteroid_clusters(100, 80)
        self.assertEqual(grid.shape, (80, 100))
        self.assertGreater(np.sum(grid > 0), 0)  # Should have some non-zero values

    def test_create_rare_grid(self):
        """Test the _create_rare_grid function."""
        rare_grid, rare_positions = _create_rare_grid(100, 80)
        self.assertEqual(rare_grid.shape, (80, 100))
        self.assertEqual(rare_positions.shape, (80, 100))
        self.assertTrue(
            np.array_equal(rare_grid[rare_positions], np.ones(np.sum(rare_positions)))
        )

    def test_create_energy_grid(self):
        """Test the _create_energy_grid function."""
        _, rare_positions = _create_rare_grid(100, 80)
        energy_grid = _create_energy_grid(80, 100, rare_positions)
        self.assertEqual(energy_grid.shape, (80, 100))
        self.assertGreater(
            np.sum(energy_grid > 0), 0
        )  # Should have some non-zero values

    def test_create_entity_grid(self):
        """Test the _create_entity_grid function."""
        entity_grid = _create_entity_grid(100, 80)
        self.assertEqual(entity_grid.shape, (80, 100))
        # Should have values between 0 and 3 (inclusive)
        self.assertTrue(np.all((entity_grid >= 0) & (entity_grid <= 3)))
        # Should have at least one non-zero value
        self.assertGreater(np.sum(entity_grid > 0), 0)

    def test_add_circular_cluster(self):
        """Test the _add_circular_cluster function."""
        grid = np.zeros((80, 100), dtype=np.int16)
        _add_circular_cluster(grid, 50, 40, 10, lambda: 1)
        # Should have some non-zero values
        self.assertGreater(np.sum(grid > 0), 0)
        # Values should be exactly 1 where set
        self.assertTrue(np.all((grid == 0) | (grid == 1)))

    def test_create_mock_race(self):
        """Test the create_mock_race function."""
        race = create_mock_race()
        self.assertIsInstance(race, dict)
        self.assertIn("race_id", race)
        self.assertIn("color", race)
        self.assertIn("birth_set", race)
        self.assertIn("survival_set", race)
        self.assertIn("trait", race)

        # Check value ranges
        self.assertTrue(1 <= race["race_id"] <= 3)
        self.assertTrue(all(100 <= c <= 255 for c in race["color"]))
        self.assertTrue(0 <= race["hunger"] < 0.7)

    def test_create_mock_shop_upgrade(self):
        """Test the create_mock_shop_upgrade function."""
        upgrade = create_mock_shop_upgrade()
        self.assertIsInstance(upgrade, dict)
        self.assertIn("name", upgrade)
        self.assertIn("cost", upgrade)
        self.assertIn("description", upgrade)
        self.assertIn("action", upgrade)
        self.assertIn("category", upgrade)

        # Check value ranges
        self.assertTrue(50 <= upgrade["cost"] <= 500)
        self.assertIn(upgrade["category"], ["ship", "field", "race", "special"])


if __name__ == "__main__":
    unittest.main()
