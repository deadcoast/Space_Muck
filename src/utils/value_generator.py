#!/usr/bin/env python3
"""
Value Generator utilities for Space Muck.

This module provides common value generation functions that can be used
by different generator classes to avoid code duplication.
"""

# Standard library imports
from typing import Any, Dict, Optional, Tuple

# Third-party library imports
import numpy as np


def generate_value_distribution(
    grid: np.ndarray,
    base_grid: np.ndarray,
    value_mean: float = 5.0,
    value_stddev: float = 2.0,
    min_value: int = 1,
) -> np.ndarray:
    """
    Generate resource values for a grid based on a noise pattern.

    Args:
        grid: Binary grid indicating entity presence
        base_grid: Noise grid for value distribution
        value_mean: Mean value for resources
        value_stddev: Standard deviation for resource values
        min_value: Minimum value for non-zero cells

    Returns:
        np.ndarray: Grid with resource values
    """
    # Create a value grid with the same shape
    value_grid = np.zeros_like(grid, dtype=float)

    # Scale noise to desired mean and standard deviation
    value_noise = base_grid * value_stddev + value_mean

    # Apply values only to non-zero cells
    value_grid = grid * value_noise.astype(int)

    # Ensure minimum value for non-zero cells
    value_grid[value_grid > 0] = np.maximum(value_grid[value_grid > 0], min_value)

    return value_grid


def add_value_clusters(
    value_grid: np.ndarray,
    num_clusters: int = 5,
    cluster_radius: int = 10,
    value_multiplier: float = 2.0,
) -> np.ndarray:
    """
    Add value clusters to a grid - some areas have higher value entities.

    Args:
        value_grid: Grid with entity values
        num_clusters: Number of high-value clusters to create
        cluster_radius: Radius of each cluster
        value_multiplier: Multiplier for values in clusters

    Returns:
        np.ndarray: Grid with value clusters added
    """
    height, width = value_grid.shape
    result_grid = value_grid.copy()

    # Find cells with entities
    entity_cells = np.argwhere(value_grid > 0)

    if len(entity_cells) == 0:
        return result_grid

    # Create clusters
    for _ in range(num_clusters):
        # Pick a random entity cell as cluster center
        idx = np.random.randint(0, len(entity_cells))
        center_y, center_x = entity_cells[idx]

        # Apply multiplier to entities in radius
        for y in range(
            max(0, center_y - cluster_radius),
            min(height, center_y + cluster_radius + 1),
        ):
            for x in range(
                max(0, center_x - cluster_radius),
                min(width, center_x + cluster_radius + 1),
            ):
                if value_grid[y, x] > 0:
                    # Calculate distance from center
                    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                    # Apply multiplier with falloff based on distance
                    if distance <= cluster_radius:
                        falloff = 1.0 - (distance / cluster_radius)
                        multiplier = 1.0 + (value_multiplier - 1.0) * falloff
                        result_grid[y, x] = int(value_grid[y, x] * multiplier)

    return result_grid


def generate_rare_resource_distribution(
    grid: np.ndarray,
    rare_noise: np.ndarray,
    precious_noise: np.ndarray,
    anomaly_noise: np.ndarray,
    rare_chance: float = 0.1,
    precious_factor: float = 0.3,
    anomaly_factor: float = 0.1,
) -> np.ndarray:
    """
    Generate rare resource distribution across a grid.

    Args:
        grid: Binary grid indicating entity presence
        rare_noise: Noise grid for rare resource distribution
        precious_noise: Noise grid for precious resource distribution
        anomaly_noise: Noise grid for anomaly distribution
        rare_chance: Base chance of rare resources
        precious_factor: Factor for precious resource chance (relative to rare_chance)
        anomaly_factor: Factor for anomaly chance (relative to rare_chance)

    Returns:
        np.ndarray: Grid with rare resource indicators (0 = common, 1 = rare, 2 = precious, 3 = anomaly)
    """
    # Create a rare resource grid with the same shape
    rare_grid = np.zeros_like(grid, dtype=np.int8)

    # No rare resources where there are no entities
    entity_mask = grid > 0

    # Apply thresholds for different resource types
    # Only apply to cells with entities
    rare_threshold = 1 - rare_chance
    precious_threshold = 1 - rare_chance * precious_factor
    anomaly_threshold = 1 - rare_chance * anomaly_factor

    # Apply rare resources (1)
    rare_mask = (rare_noise > rare_threshold) & entity_mask
    rare_grid[rare_mask] = 1

    # Apply precious resources (2) - overrides rare
    precious_mask = (precious_noise > precious_threshold) & entity_mask
    rare_grid[precious_mask] = 2

    # Apply anomalies (3) - overrides everything
    anomaly_mask = (anomaly_noise > anomaly_threshold) & entity_mask
    rare_grid[anomaly_mask] = 3

    return rare_grid
