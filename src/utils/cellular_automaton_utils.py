#!/usr/bin/env python3
"""
Cellular Automaton utilities for Space Muck.

This module provides common cellular automaton functions that can be used
by different generator classes to avoid code duplication.
"""

# Standard library imports
import itertools

# Local application imports
from typing import Any, Dict, Set, Tuple

import numpy as np

# Third-party library imports
from numpy.random import PCG64, Generator

# Initialize random number generator with a fixed seed for reproducibility
rng = Generator(PCG64(42))

# Optional dependencies
try:
    import scipy.ndimage as ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print(
        "Warning: scipy not available, using fallback implementation for optimized cellular automaton"
    )


def _count_neighbors(
    grid: np.ndarray, x: int, y: int, width: int, height: int, wrap: bool
) -> int:
    """
    Count the number of live neighbors for a cell at position (x, y).

    Args:
        grid: The grid containing cell states
        x: X-coordinate of the cell
        y: Y-coordinate of the cell
        width: Width of the grid
        height: Height of the grid
        wrap: Whether to wrap around grid edges

    Returns:
        int: Number of live neighbors
    """
    neighbors = 0

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            # Skip the cell itself
            if dx == 0 and dy == 0:
                continue

            nx, ny = x + dx, y + dy

            # Handle edge conditions
            if wrap:
                nx = nx % width
                ny = ny % height
            elif nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            neighbors += grid[ny, nx]

    return neighbors


def _apply_rules(
    current_state: int, neighbors: int, birth_set: Set[int], survival_set: Set[int]
) -> int:
    """
    Apply cellular automaton rules to determine the next state of a cell.

    Args:
        current_state: Current state of the cell (0 or 1)
        neighbors: Number of live neighbors
        birth_set: Set of neighbor counts that cause cell birth
        survival_set: Set of neighbor counts that allow cell survival

    Returns:
        int: Next state of the cell (0 or 1)
    """
    if current_state == 1:
        # Cell is alive - check survival rule
        return 1 if neighbors in survival_set else 0
    else:
        # Cell is dead - check birth rule
        return 1 if neighbors in birth_set else 0


def _evolve_grid_once(
    grid: np.ndarray,
    birth_set: Set[int],
    survival_set: Set[int],
    wrap: bool,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Evolve the grid through one iteration of the cellular automaton rules.

    Args:
        grid: Current state of the grid
        birth_set: Set of neighbor counts that cause cell birth
        survival_set: Set of neighbor counts that allow cell survival
        wrap: Whether to wrap around grid edges
        width: Width of the grid
        height: Height of the grid

    Returns:
        np.ndarray: Evolved grid after one iteration
    """
    new_grid = grid.copy()

    # Process all cells using itertools.product for better performance
    for y, x in itertools.product(range(height), range(width)):
        # Count neighbors and apply rules
        neighbors = _count_neighbors(grid, x, y, width, height, wrap)
        new_grid[y, x] = _apply_rules(grid[y, x], neighbors, birth_set, survival_set)

    return new_grid


def apply_cellular_automaton(
    grid: np.ndarray,
    birth_set: Set[int] = None,
    survival_set: Set[int] = None,
    iterations: int = 3,
    wrap: bool = True,
    width: int = None,
    height: int = None,
) -> np.ndarray:
    """
    Apply cellular automaton rules to a grid.

    Args:
        grid: Input grid to evolve
        birth_set: Set of neighbor counts that cause cell birth
        survival_set: Set of neighbor counts that allow cell survival
        iterations: Number of iterations to perform
        wrap: Whether to wrap around grid edges
        width: Width of the grid (if None, inferred from grid)
        height: Height of the grid (if None, inferred from grid)

    Returns:
        np.ndarray: Evolved grid
    """
    # Handle default parameters
    if birth_set is None:
        birth_set = {3}
    if survival_set is None:
        survival_set = {2, 3}
    if width is None:
        height, width = grid.shape

    # Convert to binary grid for processing
    binary_grid = (grid > 0).astype(np.int8)
    result_grid = binary_grid.copy()

    # Evolve the grid for the specified number of iterations
    for _ in range(iterations):
        result_grid = _evolve_grid_once(
            result_grid, birth_set, survival_set, wrap, width, height
        )

    # Preserve original values where cells are alive
    return grid * result_grid


def apply_cellular_automaton_optimized(
    grid: np.ndarray, birth_set: Set[int], survival_set: Set[int]
) -> np.ndarray:
    """
    Apply cellular automaton rules using optimized convolution method.

    Args:
        grid: Input grid to evolve
        birth_set: Set of neighbor counts that cause cell birth
        survival_set: Set of neighbor counts that allow cell survival

    Returns:
        np.ndarray: Evolved grid
    """
    # Convert to binary grid for consistent processing
    binary_grid = (grid > 0).astype(np.int8)

    if SCIPY_AVAILABLE:
        return _apply_convolution_based_cellular_automaton(
            binary_grid, survival_set, birth_set, grid
        )
    # Fallback to standard implementation if scipy is not available
    height, width = grid.shape
    return apply_cellular_automaton(
        grid,
        birth_set,
        survival_set,
        iterations=1,
        wrap=True,
        width=width,
        height=height,
    )


def _apply_convolution_based_cellular_automaton(
    binary_grid, survival_set, birth_set, grid
):
    # Count neighbors using scipy's optimized convolution
    neighbors_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndimage.convolve(
        binary_grid, neighbors_kernel, mode="wrap", cval=0
    )

    # Create new grid
    new_grid = np.zeros_like(binary_grid)

    # Apply survival rules
    for n in survival_set:
        new_grid |= (neighbor_count == n) & (binary_grid > 0)

    # Apply birth rules
    for n in birth_set:
        new_grid |= (neighbor_count == n) & (binary_grid == 0)

    # Preserve original values where cells are alive
    return grid * new_grid


def generate_cellular_automaton_rules(
    hunger: float, genome: Dict[str, Any]
) -> Tuple[Set[int], Set[int]]:
    """
    Generate cellular automaton rules based on genome and hunger.

    Args:
        hunger: Hunger level (0-1)
        genome: Genome dictionary with traits

    Returns:
        Tuple[Set[int], Set[int]]: Birth and survival rule sets
    """
    # Default rule sets
    birth_set = {3}
    survival_set = {2, 3}

    # Adjust based on hunger and genome
    if hunger > 0.7:
        # Hungry symbiotes are more aggressive in growth
        birth_set.add(2)

    if genome.get("expansion_drive", 1.0) > 1.2:
        # Expansive races grow more easily
        birth_set.add(2)

    if genome.get("intelligence", 0.5) > 0.8:
        # Intelligent races are more strategic about survival
        survival_set.add(4)

    return birth_set, survival_set


def apply_environmental_effects(
    grid: np.ndarray, mineral_map: np.ndarray, hostility: float
) -> np.ndarray:
    """
    Apply environmental effects to the grid based on mineral distribution.

    Args:
        grid: Input grid with entities
        mineral_map: Map of mineral distribution
        hostility: Environmental hostility factor (0-1)

    Returns:
        np.ndarray: Grid after environmental effects
    """
    return grid & (rng.random(grid.shape) < (1 - hostility + mineral_map * 0.5))
