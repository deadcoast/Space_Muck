"""
ProceduralGenerator class: Handles procedural generation for asteroid fields.

This module contains the ProceduralGenerator class which inherits from BaseGenerator
and provides specialized functionality for generating asteroid fields using multiple
noise algorithms and cellular automaton rules.
"""

import math
import random
import logging
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable

import numpy as np
import scipy.ndimage as ndimage
import scipy.stats as stats
from skimage import measure

from src.config import *
from src.entities.base_generator import BaseGenerator
from src.world.asteroid_field import AsteroidField
from src.utils.noise_generator import NoiseGenerator, get_noise_generator
from src.utils.dependency_injection import inject
from src.utils.logging_setup import (
    log_performance_start,
    log_performance_end,
    log_exception,
    LogContext,
)


@inject
class ProceduralGenerator(BaseGenerator):
    """
    Handles procedural generation for asteroid fields using multiple algorithms.
    Inherits from BaseGenerator to leverage common generation functionality.
    """

    def __init__(
        self,
        entity_id: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 100,
        height: int = 100,
        color: Tuple[int, int, int] = (100, 200, 100),
        position: Optional[Tuple[int, int]] = None,
        noise_generator: Optional[NoiseGenerator] = None,
    ) -> None:
        """
        Initialize the procedural generator with an optional seed.

        Args:
            entity_id: Unique identifier for the entity (defaults to a UUID)
            seed: Random seed for reproducibility
            width: Width of the generated area
            height: Height of the generated area
            color: RGB color tuple for visualization
            position: Initial position as (x, y) tuple
            noise_generator: Injected noise generator (defaults to auto-selected implementation)
        """
        # Call the parent class constructor
        super().__init__(
            entity_id=entity_id,
            entity_type="procedural",
            seed=seed,
            width=width,
            height=height,
            color=color,
            position=position,
            noise_generator=noise_generator,
        )

        # Statistical parameters
        self.value_distribution = stats.lognorm(s=0.6, scale=50)  # For asteroid values

        # Configure cellular automaton parameters
        self.set_parameter(
            "birth_set", {3}
        )  # Default: cell is born if it has exactly 3 neighbors
        self.set_parameter(
            "survival_set", {2, 3}
        )  # Default: cell survives with 2 or 3 neighbors
        self.set_parameter("iterations", 3)  # Default iterations for CA evolution
        self.set_parameter("wrap", True)  # Whether the grid wraps around edges

        # Fractal parameters
        self.set_parameter("fractal_depth", 3)  # Depth for fractal generation
        self.set_parameter("fractal_roughness", 0.6)  # Roughness factor for fractals

        # Anomaly generation parameters
        self.set_parameter("anomaly_chance", 0.01)  # Base chance of anomaly per cell
        self.set_parameter(
            "anomaly_cluster_chance", 0.2
        )  # Chance of anomaly clustering

        logging.info(
            f"ProceduralGenerator initialized: ID: {self.entity_id}, Seed: {self.seed}"
        )

    def generate_asteroid_field(self, density: float = 0.3) -> np.ndarray:
        """
        Generate a procedural asteroid field using multiple noise layers.

        Args:
            density: Overall density of asteroids (0-1)

        Returns:
            np.ndarray: 2D grid with asteroid values
        """
        start_time = log_performance_start("generate_asteroid_field")

        try:
            return self.generate_multi_layer_asteroid_field(density, start_time)
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.random.binomial(1, density * 0.5, (self.height, self.width)) * 50

    def generate_multi_layer_asteroid_field(self, density, start_time):
        # Initialize the grid
        grid = np.zeros((self.height, self.width), dtype=np.int32)

        # Generate primary noise layer
        primary = self.generate_noise_layer("medium", scale=0.02)
        primary = (primary - primary.min()) / (primary.max() - primary.min())

        # Generate secondary noise for details
        secondary = self.generate_noise_layer("high", scale=0.05)
        secondary = (secondary - secondary.min()) / (secondary.max() - secondary.min())

        # Combine noise layers
        combined = (primary * 0.7 + secondary * 0.3) * density * 1.5

        # Apply threshold to create asteroid locations
        asteroid_mask = combined > (1 - density)

        # Generate values for asteroids using lognormal distribution
        values = self.value_distribution.rvs(size=np.sum(asteroid_mask))
        values = np.clip(values, 10, 200).astype(
            np.int32
        )  # Clip values to reasonable range

        # Assign values to asteroid locations
        grid[asteroid_mask] = values

        # Apply cellular automaton to refine the pattern
        birth_set = self.get_parameter("birth_set", {3})
        survival_set = self.get_parameter("survival_set", {2, 3})
        iterations = self.get_parameter("iterations", 3)
        wrap = self.get_parameter("wrap", True)

        grid = self.apply_cellular_automaton(
            grid, birth_set, survival_set, iterations, wrap
        )

        # Create clusters of higher-value asteroids
        grid = self.create_value_clusters(grid)

        # Generate void spaces for visual interest
        grid = self.create_void_areas(grid)

        # Add edge degradation for natural appearance
        grid = self.apply_edge_degradation(grid)

        log_performance_end("generate_asteroid_field", start_time)
        return grid

    def generate_rare_minerals(
        self, grid: np.ndarray, rare_chance: float = 0.1
    ) -> np.ndarray:
        """
        Generate rare mineral distribution based on the asteroid field.

        Args:
            grid: Asteroid field grid
            rare_chance: Base chance of rare minerals

        Returns:
            np.ndarray: Grid with rare mineral indicators (0 = common, 1 = rare, 2 = precious, 3 = anomaly)
        """
        start_time = log_performance_start("generate_rare_minerals")

        try:
            return self.generate_tiered_mineral_distribution(
                grid, rare_chance, start_time
            )
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.zeros_like(grid, dtype=np.int8)

    def generate_tiered_mineral_distribution(self, grid, rare_chance, start_time):
        rare_grid = np.zeros_like(grid, dtype=np.int8)

        # No rare minerals where there are no asteroids
        asteroid_mask = grid > 0

        # Generate noise for rare mineral distribution
        rare_noise = self.generate_noise_layer("high", scale=0.03)
        rare_noise = (rare_noise - rare_noise.min()) / (
            rare_noise.max() - rare_noise.min()
        )

        # Generate precious noise (even rarer)
        precious_noise = self.generate_noise_layer("high", scale=0.015)
        precious_noise = (precious_noise - precious_noise.min()) / (
            precious_noise.max() - precious_noise.min()
        )

        # Generate anomaly noise (rarest)
        anomaly_noise = self.generate_noise_layer("detail", scale=0.01)
        anomaly_noise = (anomaly_noise - anomaly_noise.min()) / (
            anomaly_noise.max() - anomaly_noise.min()
        )

        # Apply thresholds for different mineral types
        # Only apply to cells with asteroids
        rare_threshold = 1 - rare_chance
        precious_threshold = 1 - rare_chance * 0.3
        anomaly_threshold = 1 - rare_chance * 0.1
        anomaly_chance = self.get_parameter("anomaly_chance", 0.01)

        # Apply masks for different mineral types
        rare_mask = (rare_noise > rare_threshold) & asteroid_mask
        precious_mask = (precious_noise > precious_threshold) & asteroid_mask
        anomaly_mask = (
            (anomaly_noise > anomaly_threshold)
            & asteroid_mask
            & (np.random.random(grid.shape) < anomaly_chance)
        )

        # Assign values to the rare grid
        # Priority: anomaly > precious > rare
        rare_grid[rare_mask] = 1  # Rare
        rare_grid[precious_mask] = 2  # Precious
        rare_grid[anomaly_mask] = 3  # Anomaly

        # Apply cellular automaton to create more natural clusters
        rare_grid = self.apply_cellular_automaton(
            rare_grid,
            birth_set={3, 4, 5},
            survival_set={2, 3, 4, 5},
            iterations=2,
            wrap=True,
        )

        log_performance_end("generate_rare_minerals", start_time)
        return rare_grid

    def create_value_clusters(
        self, grid: np.ndarray, num_clusters: int = 5
    ) -> np.ndarray:
        """
        Create clusters of higher-value asteroids in the grid.

        Args:
            grid: Input asteroid grid
            num_clusters: Number of clusters to create

        Returns:
            np.ndarray: Grid with value clusters
        """
        # Use the BaseGenerator's create_clusters method with asteroid-specific parameters
        return self.create_clusters(grid, num_clusters, cluster_value_multiplier=2.5)

    def create_void_areas(
        self, grid: np.ndarray, count: int = 3, max_size: int = 50
    ) -> np.ndarray:
        """
        Create void areas in the asteroid field for visual interest.

        Args:
            grid: Input asteroid grid
            count: Number of void areas to create
            max_size: Maximum size of void areas

        Returns:
            np.ndarray: Grid with void areas
        """
        result_grid = grid.copy()

        for _ in range(count):
            # Random center point
            cx = np.random.randint(0, self.width)
            cy = np.random.randint(0, self.height)

            # Random size
            size = np.random.randint(10, max_size)

            # Create void
            for y in range(max(0, cy - size), min(self.height, cy + size + 1)):
                for x in range(max(0, cx - size), min(self.width, cx + size + 1)):
                    # Calculate distance from center
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

                    # Clear cells within the void radius with some noise at the edges
                    if dist < size * (0.7 + np.random.random() * 0.3):
                        result_grid[y, x] = 0

        return result_grid

    def apply_edge_degradation(
        self, grid: np.ndarray, decay_factor: float = 0.1
    ) -> np.ndarray:
        """
        Apply edge degradation to the asteroid field for a more natural appearance.

        Args:
            grid: Input asteroid grid
            decay_factor: Factor controlling edge decay rate

        Returns:
            np.ndarray: Grid with edge degradation
        """
        result_grid = grid.copy()

        # Find edges using a simple edge detection method
        edges = np.zeros_like(grid, dtype=bool)

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if grid[y, x] > 0:
                    # Check if this cell has an empty neighbor
                    has_empty_neighbor = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            if grid[y + dy, x + dx] == 0:
                                has_empty_neighbor = True
                                break
                        if has_empty_neighbor:
                            break

                    if has_empty_neighbor:
                        edges[y, x] = True

        # Apply decay to edges
        for y in range(self.height):
            for x in range(self.width):
                if edges[y, x] and np.random.random() < decay_factor:
                    result_grid[y, x] = int(
                        result_grid[y, x] * (0.5 + np.random.random() * 0.3)
                    )

        return result_grid

    def generate(self) -> Dict[str, np.ndarray]:
        """
        Generate a complete asteroid field with all components.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing grid, rare_grid, and energy_grid
        """
        start_time = log_performance_start("generate")

        try:
            # Generate the main asteroid grid
            density = self.get_parameter("field_density", 0.3)
            grid = self.generate_asteroid_field(density=density)

            # Generate rare minerals distribution
            rare_chance = self.get_parameter("rare_chance", 0.1)
            rare_grid = self.generate_rare_minerals(grid, rare_chance=rare_chance)

            # Generate energy grid based on asteroid values
            energy_grid = grid.astype(np.float32) / 100.0

            # Apply rare mineral bonuses to energy grid
            rare_bonus = self.get_parameter("rare_bonus_multiplier", 2.0)
            for rare_level in range(1, 4):
                rare_mask = rare_grid == rare_level
                energy_grid[rare_mask] *= rare_bonus * rare_level

            log_performance_end("generate", start_time)

            # Return all grids in a dictionary
            return {"grid": grid, "rare_grid": rare_grid, "energy_grid": energy_grid}

        except Exception as e:
            log_exception(e)
            # Return simple fallback grids if generation fails
            logging.warning("Using fallback generation due to error")
            fallback_grid = np.random.binomial(1, 0.2, (self.height, self.width)) * 50
            return {
                "grid": fallback_grid,
                "rare_grid": np.zeros_like(fallback_grid, dtype=np.int8),
                "energy_grid": fallback_grid.astype(np.float32) / 100.0,
            }


def create_field_with_multiple_algorithms(
    width: int = 100,
    height: int = 100,
    seed: Optional[int] = None,
    rare_chance: float = 0.1,
    rare_bonus: float = 2.0,
) -> AsteroidField:
    """
    Create an asteroid field using multiple procedural generation algorithms.

    This function combines various generation techniques to create a rich,
    detailed asteroid field with rare minerals and interesting features.

    Args:
        width: Width of the field
        height: Height of the field
        seed: Random seed for reproducibility
        rare_chance: Chance of rare minerals appearing
        rare_bonus: Value multiplier for rare minerals

    Returns:
        AsteroidField: Fully initialized asteroid field
    """
    # Create the asteroid field
    field = AsteroidField(width=width, height=height)

    # Initialize the procedural generator
    generator = ProceduralGenerator(seed=seed, width=width, height=height)

    # Generate the asteroid grid
    field.grid = generator.generate_asteroid_field(density=0.3)

    # Generate rare minerals
    field.rare_grid = generator.generate_rare_minerals(
        field.grid, rare_chance=rare_chance
    )

    # Initialize energy grid based on asteroid values
    field.energy_grid = field.grid.astype(np.float32) / 100.0

    # Apply rare mineral bonuses to energy grid
    for rare_level in range(1, 4):
        rare_mask = field.rare_grid == rare_level
        field.energy_grid[rare_mask] *= rare_bonus * rare_level

    logging.info(f"Created asteroid field with seed {seed}, size {width}x{height}")
    logging.info(f"Field contains {np.sum(field.grid > 0)} asteroids")
    logging.info(
        f"Rare minerals: {np.sum(field.rare_grid == 1)} rare, {np.sum(field.rare_grid == 2)} precious, {np.sum(field.rare_grid == 3)} anomaly"
    )

    return field
