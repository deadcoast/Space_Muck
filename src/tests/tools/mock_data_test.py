"""
Mock data generators for Space Muck tests.

This module provides functions to generate consistent test data.
"""

# Standard library imports

# Third-party library imports
import numpy as np

# Local application imports
from typing import Dict


def create_mock_field_data(width: int = 100, height: int = 80) -> Dict[str, np.ndarray]:
    """
    Create mock field data for testing.

    Args:
        width: Field width
        height: Field height

    Returns:
        dict: Dictionary with grid data
    """
    # Create grid with some asteroid clusters
    grid = np.zeros((height, width), dtype=np.int16)

    # Add clusters of asteroids
    for _ in range(5):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        radius = np.random.randint(5, 15)

        for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    grid[y, x] = np.random.randint(10, 100)

    # Create rare grid with some rare asteroids
    rare_grid = np.zeros((height, width), dtype=np.int8)
    rare_positions = np.random.random((height, width)) > 0.95
    rare_grid[rare_positions] = 1

    # Energy grid
    energy_grid = np.zeros((height, width), dtype=np.float32)
    energy_grid[rare_positions] = np.random.random(size=np.sum(rare_positions))

    # Entity grid with some race entities
    entity_grid = np.zeros((height, width), dtype=np.int8)

    # Add a few race colonies
    for i in range(1, 4):  # 3 races
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        radius = np.random.randint(3, 8)

        for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
                if (
                    np.random.random() < 0.7
                    and (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
                ):
                    entity_grid[y, x] = i

    return {
        "grid": grid,
        "rare_grid": rare_grid,
        "energy_grid": energy_grid,
        "entity_grid": entity_grid,
    }


def create_mock_race() -> dict:
    """
    Create a mock race entity with sensible defaults.

    Returns:
        dict: Race data
    """
    return {
        "race_id": np.random.randint(1, 4),
        "color": (
            np.random.randint(100, 255),
            np.random.randint(100, 255),
            np.random.randint(100, 255),
        ),
        "birth_set": {3},
        "survival_set": {2, 3},
        "initial_density": 0.1,
        "trait": np.random.choice(["adaptive", "expansive", "selective"]),
        "hunger": np.random.random() * 0.7,
        "population": np.random.randint(1000, 5000),
        "evolution_stage": np.random.randint(0, 3),
        "evolution_points": np.random.randint(0, 100),
        "mining_efficiency": 0.5 + np.random.random() * 0.3,
    }


def create_mock_shop_upgrade() -> dict:
    """
    Create a mock shop upgrade option.

    Returns:
        dict: Upgrade data
    """
    categories = ["ship", "field", "race", "special"]
    return {
        "name": f"Test Upgrade {np.random.randint(1, 100)}",
        "cost": np.random.randint(50, 500),
        "description": "This is a test upgrade for unit testing.",
        "action": lambda player, field: None,  # Mock action
        "category": np.random.choice(categories),
        "max_level": np.random.randint(1, 10),
        "current_level": 0,
        "locked": np.random.random() < 0.3,
        "unlock_condition": lambda player, field: True,
        "scaling_factor": 1.5,
    }
