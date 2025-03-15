"""
Mock data generators for Space Muck tests.

This module provides functions to generate consistent test data.
"""

# Standard library imports
import itertools

# Local application imports
from typing import Dict, Callable

# Third-party library imports
import numpy as np

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(42)


def _create_asteroid_clusters(width: int, height: int) -> np.ndarray:
    """
    Create a grid with asteroid clusters.

    Args:
        width: Grid width
        height: Grid height

    Returns:
        np.ndarray: Grid with asteroid clusters
    """
    grid = np.zeros((height, width), dtype=np.int16)

    for _ in range(5):
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        radius = rng.integers(5, 15)

        _add_circular_cluster(grid, cx, cy, radius, lambda: rng.integers(10, 100))

    return grid


def _create_rare_grid(width: int, height: int) -> tuple:
    """
    Create a grid with rare resource positions.

    Args:
        width: Grid width
        height: Grid height

    Returns:
        tuple: (rare_grid, rare_positions)
    """
    rare_grid = np.zeros((height, width), dtype=np.int8)
    rare_positions = rng.random((height, width)) > 0.95
    rare_grid[rare_positions] = 1

    return rare_grid, rare_positions


def _create_energy_grid(
    height: int, width: int, rare_positions: np.ndarray
) -> np.ndarray:
    """
    Create an energy grid based on rare positions.

    Args:
        height: Grid height
        width: Grid width
        rare_positions: Boolean array indicating rare positions

    Returns:
        np.ndarray: Energy grid
    """
    energy_grid = np.zeros((height, width), dtype=np.float32)
    energy_grid[rare_positions] = rng.random(size=np.sum(rare_positions))

    return energy_grid


def _create_entity_grid(width: int, height: int) -> np.ndarray:
    """
    Create a grid with race entities.

    Args:
        width: Grid width
        height: Grid height

    Returns:
        np.ndarray: Entity grid
    """
    entity_grid = np.zeros((height, width), dtype=np.int8)

    # Add a few race colonies
    for i in range(1, 4):  # 3 races
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        radius = rng.integers(3, 8)

        # Capture the current value of i to avoid late binding issues
        race_id = i
        _add_circular_cluster(
            entity_grid,
            cx,
            cy,
            radius,
            lambda race_id=race_id: race_id if rng.random() < 0.7 else 0,
        )

    return entity_grid


def _add_circular_cluster(
    grid: np.ndarray, cx: int, cy: int, radius: int, value_func: Callable
):
    """
    Add a circular cluster to a grid.

    Args:
        grid: Target grid
        cx: Center x coordinate
        cy: Center y coordinate
        radius: Radius of the cluster
        value_func: Function that returns the value to set
    """
    height, width = grid.shape

    y_range = range(max(0, cy - radius), min(height, cy + radius + 1))
    x_range = range(max(0, cx - radius), min(width, cx + radius + 1))

    for y, x in itertools.product(y_range, x_range):
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
            value = value_func()
            if value != 0:  # Only set non-zero values
                grid[y, x] = value


def create_mock_field_data(width: int = 100, height: int = 80) -> Dict[str, np.ndarray]:
    """
    Create mock field data for testing.

    Args:
        width: Field width
        height: Field height

    Returns:
        dict: Dictionary with grid data
    """
    # Create grid with asteroid clusters
    grid = _create_asteroid_clusters(width, height)

    # Create rare grid with rare resources
    rare_grid, rare_positions = _create_rare_grid(width, height)

    # Create energy grid
    energy_grid = _create_energy_grid(height, width, rare_positions)

    # Create entity grid with race entities
    entity_grid = _create_entity_grid(width, height)

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
        "race_id": rng.integers(1, 4),
        "color": (
            rng.integers(100, 255),
            rng.integers(100, 255),
            rng.integers(100, 255),
        ),
        "birth_set": {3},
        "survival_set": {2, 3},
        "initial_density": 0.1,
        "trait": rng.choice(["adaptive", "expansive", "selective"]),
        "hunger": rng.random() * 0.7,
        "population": rng.integers(1000, 5000),
        "evolution_stage": rng.integers(0, 3),
        "evolution_points": rng.integers(0, 100),
        "mining_efficiency": 0.5 + rng.random() * 0.3,
    }


def create_mock_shop_upgrade() -> dict:
    """
    Create a mock shop upgrade option.

    Returns:
        dict: Upgrade data
    """
    categories = ["ship", "field", "race", "special"]
    return {
        "name": f"Test Upgrade {rng.integers(1, 100)}",
        "cost": rng.integers(50, 500),
        "description": "This is a test upgrade for unit testing.",
        "action": lambda player, field: None,  # Mock action
        "category": rng.choice(categories),
        "max_level": rng.integers(1, 10),
        "current_level": 0,
        "locked": rng.random() < 0.3,
        "unlock_condition": lambda player, field: True,
        "scaling_factor": 1.5,
    }
