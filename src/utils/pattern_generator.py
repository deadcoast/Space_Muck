#!/usr/bin/env python3
"""
Pattern Generator utilities for Space Muck.

This module provides common pattern generation functions that can be used
by different generator classes to avoid code duplication.
"""

# Standard library imports
import itertools
import math

# Third-party library imports
from numpy.random import Generator, PCG64
import numpy as np

# Local application imports
from typing import Callable, List, Optional, Tuple

# Standard library imports

# Third-party library imports

# Initialize random number generator with a fixed seed for reproducibility
rng = Generator(PCG64(42))

def generate_spiral_pattern(
    width: int,
    height: int,
    center: Optional[Tuple[int, int]] = None,
    density: float = 0.5,
    rotation: float = 1.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a spiral pattern.

    Args:
        width: Width of the pattern
        height: Height of the pattern
        center: Optional center point (x, y), defaults to center of grid
        density: Density of the spiral (higher = tighter spiral)
        rotation: Rotation factor
        scale: Scale factor

    Returns:
        np.ndarray: 2D grid with spiral pattern values (0-1)
    """
    if center is None:
        center = (width // 2, height // 2)

    center_x, center_y = center
    grid = np.zeros((height, width), dtype=float)

    for y, x in itertools.product(range(height), range(width)):
        # Skip the center point as we've already set it
        if x == center_x and y == center_y:
            continue

        # Calculate distance and angle from center
        dx = (x - center_x) / scale
        dy = (y - center_y) / scale
        distance = math.sqrt(dx**2 + dy**2)

        # For points very close to center, set high values
        if distance < 1.0:
            grid[y, x] = 0.9
            continue

        angle = math.atan2(dy, dx)

        # Spiral function - invert so that values decrease as distance increases
        spiral = 1.0 - ((angle / (2 * math.pi) + distance * density * rotation) % 1.0)
        # Ensure values decrease as we move away from center
        spiral = spiral / (1.0 + distance * 0.1)
        grid[y, x] = spiral

    # Ensure the center point has the maximum value of 1.0
    grid[center_y, center_x] = 1.0
    return grid

def generate_ring_pattern(
    width: int,
    height: int,
    center: Optional[Tuple[int, int]] = None,
    num_rings: int = 5,
    falloff: float = 0.5,
) -> np.ndarray:
    """
    Generate a concentric ring pattern.

    Args:
        width: Width of the pattern
        height: Height of the pattern
        center: Optional center point (x, y), defaults to center of grid
        num_rings: Number of rings
        falloff: How quickly ring intensity falls off with distance

    Returns:
        np.ndarray: 2D grid with ring pattern values (0-1)
    """
    if center is None:
        center = (width // 2, height // 2)

    center_x, center_y = center
    grid = np.zeros((height, width), dtype=float)

    max_distance = math.sqrt(width**2 + height**2) / 2

    for y, x in itertools.product(range(height), range(width)):
        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        distance = math.sqrt(dx**2 + dy**2)

        # Normalize distance
        normalized_distance = distance / max_distance

        # Ring function
        ring_value = math.sin(normalized_distance * num_rings * math.pi)
        ring_value = abs(ring_value) * (1 - normalized_distance * falloff)

        grid[y, x] = max(0, ring_value)

    return grid

def generate_gradient_pattern(
    width: int, height: int, direction: float = 0.0, steepness: float = 1.0
) -> np.ndarray:
    """
    Generate a gradient pattern.

    Args:
        width: Width of the pattern
        height: Height of the pattern
        direction: Direction of the gradient in radians (0 = left to right)
        steepness: Steepness of the gradient

    Returns:
        np.ndarray: 2D grid with gradient pattern values (0-1)
    """
    grid = np.zeros((height, width), dtype=float)

    # Calculate direction vector
    dx = math.cos(direction)
    dy = math.sin(direction)

    # Normalize for different aspect ratios
    aspect_ratio = width / height
    if aspect_ratio > 1:
        dy *= aspect_ratio
    else:
        dx /= aspect_ratio

    # Calculate max projection
    max_proj = abs(dx * width) + abs(dy * height)

    for y, x in itertools.product(range(height), range(width)):
        # Project point onto direction vector
        projection = (x * dx + y * dy) / max_proj

        # Apply steepness
        gradient_value = 0.5 + (projection - 0.5) * steepness
        grid[y, x] = max(0, min(1, gradient_value))

    return grid

def generate_void_pattern(
    width: int,
    height: int,
    num_voids: int = 3,
    void_size: float = 0.2,
    sharpness: float = 3.0,
) -> np.ndarray:
    """
    Generate a pattern with void areas.

    Args:
        width: Width of the pattern
        height: Height of the pattern
        num_voids: Number of void areas
        void_size: Size of void areas relative to grid size
        sharpness: Sharpness of void edges

    Returns:
        np.ndarray: 2D grid with void pattern values (0-1)
    """
    grid = np.ones((height, width), dtype=float)

    # Generate random void centers
    void_centers = []
    for _ in range(num_voids):
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        void_centers.append((x, y))

    # Calculate void size in pixels
    void_radius = min(width, height) * void_size

    for y, x, (center_x, center_y) in itertools.product(
        range(height), range(width), void_centers
    ):
        # Calculate distance from void center
        dx = x - center_x
        dy = y - center_y
        distance = math.sqrt(dx**2 + dy**2)

        # Calculate void effect
        void_effect = 1.0 - max(0, 1.0 - (distance / void_radius) ** sharpness)

        # Apply void effect
        grid[y, x] = min(grid[y, x], void_effect)

    return grid

def apply_weighted_patterns(
    width: int,
    height: int,
    pattern_functions: List[Callable],
    pattern_weights: List[float],
    pattern_args: List[dict] = None,
) -> np.ndarray:
    """
    Apply multiple patterns with weights.

    Args:
        width: Width of the pattern
        height: Height of the pattern
        pattern_functions: List of pattern generation functions
        pattern_weights: List of weights for each pattern
        pattern_args: List of argument dictionaries for each pattern function

    Returns:
        np.ndarray: 2D grid with combined pattern values (0-1)
    """
    if pattern_args is None:
        pattern_args = [{} for _ in pattern_functions]

    # Normalize weights
    total_weight = sum(pattern_weights)
    if total_weight == 0:
        return np.zeros((height, width), dtype=float)

    normalized_weights = [w / total_weight for w in pattern_weights]

    # Initialize result grid
    result_grid = np.zeros((height, width), dtype=float)

    # Apply each pattern
    for pattern_func, weight, args in zip(
        pattern_functions, normalized_weights, pattern_args
    ):
        if weight > 0:
            pattern_grid = pattern_func(width, height, **args)
            result_grid += pattern_grid * weight

    # Ensure values are in [0, 1]
    return np.clip(result_grid, 0, 1)
