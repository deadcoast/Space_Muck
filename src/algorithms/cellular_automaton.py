"""
Cellular automaton algorithms for procedural generation.

This module provides utility functions for applying cellular automaton rules
to 2D grids, primarily used for procedural generation of asteroid fields and
other game elements.
"""

from itertools import product

# Standard library imports
from typing import Optional

# Third-party library imports
import numpy as np
import scipy.signal as signal

# Local application imports

# Standard library imports

# Third-party imports


def count_neighbors(grid: np.ndarray) -> np.ndarray:
    """
    Count neighbors for each cell in a grid using convolution.

    Args:
        grid: 2D numpy array, where non-zero values are considered "alive"

    Returns:
        2D numpy array with neighbor counts for each cell
    """
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    return signal.convolve2d(
        (grid > 0).astype(int), kernel, mode="same", boundary="wrap"
    )


def _process_cell_with_energy(
    grid: np.ndarray,
    new_grid: np.ndarray,
    neighbor_counts: np.ndarray,
    birth_set: set,
    survival_set: set,
    y: int,
    x: int,
    energy_level: float,
) -> None:
    """
    Process a single cell with energy influence.

    Args:
        grid: Current state grid
        new_grid: Grid being constructed
        neighbor_counts: Neighbor count for each cell
        birth_set: Set of neighbor counts that create new cells
        survival_set: Set of neighbor counts that allow cells to survive
        y: Y coordinate of cell
        x: X coordinate of cell
        energy_level: Energy level at this cell
    """
    # Energy can boost survival by adding to neighbor count
    energy_boost = min(2, int(energy_level * 3))
    adjusted_survival = survival_set.union({n + energy_boost for n in survival_set})

    if grid[y, x] > 0:  # Cell is alive
        if neighbor_counts[y, x] in adjusted_survival:
            new_grid[y, x] = grid[y, x]  # Cell survives
    elif neighbor_counts[y, x] in birth_set:
        new_grid[y, x] = 1  # Cell is born


def _process_cell_simple(
    grid: np.ndarray,
    new_grid: np.ndarray,
    neighbor_counts: np.ndarray,
    birth_set: set,
    survival_set: set,
    y: int,
    x: int,
) -> None:
    """
    Process a single cell without energy influence.

    Args:
        grid: Current state grid
        new_grid: Grid being constructed
        neighbor_counts: Neighbor count for each cell
        birth_set: Set of neighbor counts that create new cells
        survival_set: Set of neighbor counts that allow cells to survive
        y: Y coordinate of cell
        x: X coordinate of cell
    """
    if grid[y, x] > 0:  # Cell is alive
        if neighbor_counts[y, x] in survival_set:
            new_grid[y, x] = grid[y, x]  # Cell survives
    elif neighbor_counts[y, x] in birth_set:
        new_grid[y, x] = 1  # Cell is born


def _apply_rules_with_energy(
    grid: np.ndarray,
    new_grid: np.ndarray,
    neighbor_counts: np.ndarray,
    birth_set: set,
    survival_set: set,
    energy_grid: np.ndarray,
) -> None:
    """
    Apply cellular automaton rules with energy influence.

    Args:
        grid: Current state grid
        new_grid: Grid being constructed
        neighbor_counts: Neighbor count for each cell
        birth_set: Set of neighbor counts that create new cells
        survival_set: Set of neighbor counts that allow cells to survive
        energy_grid: Energy grid that modifies rules
    """
    height, width = grid.shape
    for y, x in product(range(height), range(width)):
        energy_level = energy_grid[y, x]
        _process_cell_with_energy(
            grid, new_grid, neighbor_counts, birth_set, survival_set, y, x, energy_level
        )


def _apply_rules_simple(
    grid: np.ndarray,
    new_grid: np.ndarray,
    neighbor_counts: np.ndarray,
    birth_set: set,
    survival_set: set,
) -> None:
    """
    Apply cellular automaton rules without energy influence.

    Args:
        grid: Current state grid
        new_grid: Grid being constructed
        neighbor_counts: Neighbor count for each cell
        birth_set: Set of neighbor counts that create new cells
        survival_set: Set of neighbor counts that allow cells to survive
    """
    height, width = grid.shape
    for y, x in product(range(height), range(width)):
        _process_cell_simple(
            grid, new_grid, neighbor_counts, birth_set, survival_set, y, x
        )


def apply_life_rules(
    grid: np.ndarray,
    birth_set: set,
    survival_set: set,
    energy_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply Conway's Game of Life rules with custom birth and survival sets.
    Optionally use energy grid as a modifier.

    Args:
        grid: Current state grid (0 = dead, >0 = alive)
        birth_set: Set of neighbor counts that create new cells
        survival_set: Set of neighbor counts that allow cells to survive
        energy_grid: Optional energy grid that modifies rules

    Returns:
        New grid after applying rules
    """
    new_grid = np.zeros_like(grid)
    neighbor_counts = count_neighbors(grid)

    if energy_grid is not None:
        _apply_rules_with_energy(
            grid, new_grid, neighbor_counts, birth_set, survival_set, energy_grid
        )
    else:
        _apply_rules_simple(grid, new_grid, neighbor_counts, birth_set, survival_set)

    return new_grid


def diffuse_energy(
    energy_grid: np.ndarray, decay_rate: float = 0.02, spread_rate: float = 0.1
) -> np.ndarray:
    """
    Diffuse energy throughout a grid with decay.

    Args:
        energy_grid: Current energy grid
        decay_rate: Rate at which energy decays
        spread_rate: Rate at which energy spreads to neighbors

    Returns:
        New energy grid after diffusion
    """
    new_energy = energy_grid.copy() * (1.0 - decay_rate)

    # Calculate neighborhood energy using convolution
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighborhood = signal.convolve2d(energy_grid, kernel, mode="same", boundary="wrap")

    # Add diffused energy from neighbors
    new_energy += neighborhood * spread_rate / 8.0

    return new_energy
