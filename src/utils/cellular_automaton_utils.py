#!/usr/bin/env python3
"""
Cellular Automaton utilities for Space Muck.

This module provides common cellular automaton functions that can be used
by different generator classes to avoid code duplication.
"""

import itertools

# Standard library imports
from typing import Any, Dict, Set, Tuple

# Third-party library imports
import numpy as np

# Optional dependencies
try:
    import scipy.ndimage as ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print(
        "Warning: scipy not available, using fallback implementation for optimized cellular automaton"
    )


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
    if birth_set is None:
        birth_set = {3}
    if survival_set is None:
        survival_set = {2, 3}
    if width is None:
        height, width = grid.shape

    binary_grid = (grid > 0).astype(np.int8)
    result_grid = binary_grid.copy()

    for _ in range(iterations):
        new_grid = result_grid.copy()

        for y, x in itertools.product(range(height), range(width)):
            # Count live neighbors
            neighbors = 0

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy

                    if wrap:
                        nx = nx % width
                        ny = ny % height
                    elif nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue

                    neighbors += result_grid[ny, nx]

            # Apply rules
            if result_grid[y, x] == 1:
                # Cell is alive
                if neighbors not in survival_set:
                    new_grid[y, x] = 0  # Cell dies
            elif neighbors in birth_set:
                new_grid[y, x] = 1  # Cell is born

        result_grid = new_grid

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
    if SCIPY_AVAILABLE:
        # Count neighbors using scipy's optimized convolution
        neighbors_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(
            grid.astype(np.int8), neighbors_kernel, mode="constant", cval=0
        )

        # Create new grid
        new_grid = np.zeros_like(grid)

        # Apply survival rules
        for n in survival_set:
            new_grid |= (neighbor_count == n) & grid

        # Apply birth rules
        for n in birth_set:
            new_grid |= (neighbor_count == n) & (~grid)

        return new_grid
    else:
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


def generate_cellular_automaton_rules(
    hunger: float, genome: Dict[str, Any], race_id: str = None
) -> Tuple[Set[int], Set[int]]:
    """
    Generate cellular automaton rules based on genome and hunger.

    Args:
        hunger: Hunger level (0-1)
        genome: Genome dictionary with traits
        race_id: Optional race identifier

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
    return grid & (np.random.random(grid.shape) < (1 - hostility + mineral_map * 0.5))
