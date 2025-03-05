"""
AsteroidGenerator class: Specialized generator for asteroid fields.

This module contains the AsteroidGenerator class which inherits from BaseGenerator
and provides specialized functionality for generating asteroid fields with various
patterns and distributions.
"""

# Standard library imports
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

# Third-party library imports
import numpy as np

# Local application imports
# Note: Config constants are referenced via parameters instead of direct imports
from src.generators.base_generator import BaseGenerator
from src.utils.dependency_injection import inject
from src.utils.noise_generator import NoiseGenerator
from src.utils.pattern_generator import (
    generate_spiral_pattern,
    generate_ring_pattern,
    generate_gradient_pattern,
    generate_void_pattern,
    apply_weighted_patterns,
)
from src.utils.value_generator import (
    generate_value_distribution,
    add_value_clusters,
    generate_rare_resource_distribution,
)
from src.utils.logging_setup import (
    log_performance_start,
    log_performance_end,
    log_exception,
)


@inject
class AsteroidGenerator(BaseGenerator):
    """
    Generator for procedural asteroid fields with multiple layers and patterns.
    Inherits from BaseGenerator to leverage common generation functionality.
    """

    def __init__(
        self,
        entity_id: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 100,
        height: int = 100,
        color: Tuple[int, int, int] = (150, 150, 100),
        position: Optional[Tuple[int, int]] = None,
        noise_generator: Optional[NoiseGenerator] = None,
    ) -> None:
        """
        Initialize the asteroid generator.

        Args:
            entity_id: Unique identifier for the entity (defaults to a UUID)
            seed: Random seed for reproducibility
            width: Width of the field to generate
            height: Height of the field to generate
            color: RGB color tuple for visualization
            position: Initial position as (x, y) tuple
            noise_generator: Injected noise generator (defaults to auto-selected implementation)
        """
        # Call the parent class constructor
        super().__init__(
            entity_id=entity_id,
            entity_type="asteroid",
            seed=seed,
            width=width,
            height=height,
            color=color,
            position=position,
            noise_generator=noise_generator,
        )

        # Generation parameters
        self.set_parameter("density", 0.2)  # Base density of asteroids
        self.set_parameter("value_mean", 5.0)  # Mean value for asteroid resources
        self.set_parameter(
            "value_stddev", 2.0
        )  # Standard deviation for resource values
        self.set_parameter("rare_chance", 0.05)  # Chance of rare resource types
        self.set_parameter(
            "cluster_tendency", 0.6
        )  # How much asteroids tend to cluster (0-1)
        self.set_parameter(
            "pattern_strength", 0.4
        )  # Strength of pattern influence (0-1)

        # Pattern types
        self.pattern_types = ["spiral", "ring", "gradient", "void"]

        logging.info(
            f"AsteroidGenerator initialized: ID: {self.entity_id}, Seed: {self.seed}"
        )

    def generate_field(
        self, pattern_weights: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a complete asteroid field with optional pattern weighting.

        Args:
            pattern_weights: Optional weights for different patterns

        Returns:
            tuple: (asteroid_grid, metadata)
        """
        start_time = log_performance_start("generate_field")

        try:
            return self._field_handler(pattern_weights, start_time)
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.random.binomial(1, 0.1, (self.height, self.width)), {
                "seed": self.seed
            }

    def _field_handler(self, pattern_weights, start_time):
        # Create empty grid
        grid = np.zeros((self.height, self.width), dtype=float)

        # Apply base noise layer
        noise_grid = self._generate_base_noise()

        # Apply patterns if weights provided
        if pattern_weights and len(pattern_weights) == len(self.pattern_types):
            # Generate patterns based on pattern types
            patterns = []
            for pattern_type in self.pattern_types:
                if pattern_type == "spiral":
                    patterns.append(self._spiral_pattern())
                elif pattern_type == "ring":
                    patterns.append(self._ring_pattern())
                elif pattern_type == "gradient":
                    patterns.append(self._gradient_pattern())
                elif pattern_type == "void":
                    patterns.append(self._void_pattern())

            # Apply weighted patterns
            pattern_grid = apply_weighted_patterns(
                patterns, pattern_weights, self.height, self.width
            )

            # Blend noise and patterns based on pattern_strength
            pattern_strength = self.get_parameter("pattern_strength", 0.4)
            grid = noise_grid * (1 - pattern_strength) + pattern_grid * pattern_strength
        else:
            grid = noise_grid

        # Normalize grid to 0-1 range
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-10)

        # Apply threshold to create binary asteroid field
        density = self.get_parameter("density", 0.2)
        asteroid_grid = (grid > (1 - density)).astype(int)

        # Apply cellular automaton to smooth field
        asteroid_grid = self._apply_cellular_automaton(asteroid_grid)

        # Generate metadata
        metadata = {
            "seed": self.seed,
            "density": density,
            "pattern_strength": self.get_parameter("pattern_strength", 0.4),
            "asteroid_count": np.sum(asteroid_grid),
        }

        log_performance_end("generate_field", start_time)
        return asteroid_grid, metadata

    def generate_values(self, asteroid_grid: np.ndarray) -> np.ndarray:
        """
        Generate resource values for each asteroid in the field.

        Args:
            asteroid_grid: Binary grid indicating asteroid presence

        Returns:
            np.ndarray: Grid with resource values
        """
        start_time = log_performance_start("generate_values")

        try:
            return self._value_handler(asteroid_grid, start_time)
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return asteroid_grid * 3

    def _value_handler(self, asteroid_grid, start_time):
        # Generate base values using perlin noise for spatial coherence
        value_noise = self._generate_value_noise()

        # Get value parameters
        value_mean = self.get_parameter("value_mean", 5.0)
        value_stddev = self.get_parameter("value_stddev", 2.0)

        # Generate value distribution
        value_grid = generate_value_distribution(
            grid=asteroid_grid,
            base_grid=value_noise,
            value_mean=value_mean,
            value_stddev=value_stddev,
            min_value=1,
        )

        # Add value clusters - some areas have higher value asteroids
        value_grid = self._add_value_clusters(value_grid)

        log_performance_end("generate_values", start_time)
        return value_grid

    def generate_rare_resources(self, asteroid_grid: np.ndarray) -> np.ndarray:
        """
        Generate rare resource distribution across the asteroid field.

        Args:
            asteroid_grid: Binary grid indicating asteroid presence

        Returns:
            np.ndarray: Grid with rare resource indicators
        """
        start_time = log_performance_start("generate_rare_resources")

        try:
            return self._rare_resource_handler(asteroid_grid, start_time)
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.zeros_like(asteroid_grid)

    def _rare_resource_handler(self, asteroid_grid, start_time):
        # Generate coherent noise pattern for rare distribution
        rare_noise = self.generate_noise_layer("medium", scale=0.05)

        # Get rare resource parameters
        rare_chance = self.get_parameter("rare_chance", 0.05)

        # Generate rare resource distribution using the utility function
        rare_grid = generate_rare_resource_distribution(
            asteroid_grid=asteroid_grid,
            noise_grid=rare_noise,
            rare_chance=rare_chance,
            apply_cellular_automaton=True,
            birth_set={3},
            survival_set={2, 3, 4},
            iterations=1,
            wrap=True,
        )

        log_performance_end("generate_rare_resources", start_time)
        return rare_grid

    def _generate_base_noise(self) -> np.ndarray:
        """
        Generate base noise layer for asteroid field.

        Returns:
            np.ndarray: Base noise grid
        """
        # Use the BaseGenerator's noise generation capabilities
        large_scale = self.generate_noise_layer("low", scale=0.03)
        medium_scale = self.generate_noise_layer("medium", scale=0.06)
        small_scale = self.generate_noise_layer("high", scale=0.1)

        return large_scale * 0.5 + medium_scale * 0.3 + small_scale * 0.2

    def _generate_value_noise(self) -> np.ndarray:
        """
        Generate noise for asteroid values.

        Returns:
            np.ndarray: Value noise grid
        """
        # Use medium detail noise for values
        return self.generate_noise_layer("medium", scale=0.08)

    def _apply_cellular_automaton(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply cellular automaton rules to smooth the asteroid field.

        Args:
            grid: Input binary grid

        Returns:
            np.ndarray: Smoothed grid
        """
        # Use the BaseGenerator's cellular automaton implementation
        return self.apply_cellular_automaton(
            grid,
            birth_set={3, 4, 5},
            survival_set={2, 3, 4, 5},
            iterations=2,
            wrap=True,
        )

    def _add_value_clusters(self, value_grid: np.ndarray) -> np.ndarray:
        """
        Add clusters of higher-value asteroids.

        Args:
            value_grid: Input value grid

        Returns:
            np.ndarray: Grid with value clusters
        """
        # Get cluster parameters
        cluster_tendency = self.get_parameter("cluster_tendency", 0.6)
        num_clusters = int(np.sqrt(self.width * self.height) * 0.05)

        # Use the utility function to add value clusters
        return add_value_clusters(
            value_grid=value_grid,
            num_clusters=num_clusters,
            cluster_radius=int(min(self.width, self.height) * 0.05),  # 5% of the smaller dimension
            value_multiplier=1.0 + cluster_tendency,
        )

    def _spiral_pattern(self) -> np.ndarray:
        """
        Generate a spiral pattern for asteroid distribution.

        Returns:
            np.ndarray: Spiral pattern grid
        """
        return generate_spiral_pattern(
            width=self.width,
            height=self.height,
            spiral_density=3,  # Number of spiral arms
        )

    def _ring_pattern(self) -> np.ndarray:
        """
        Generate a ring pattern for asteroid distribution.

        Returns:
            np.ndarray: Ring pattern grid
        """
        # Number of rings
        num_rings = random.randint(2, 5)
        ring_width = 0.2  # Width of each ring as a fraction of max_radius

        return generate_ring_pattern(
            width=self.width,
            height=self.height,
            num_rings=num_rings,
            ring_width=ring_width,
            random_generator=self.random,
        )

    def _gradient_pattern(self) -> np.ndarray:
        """
        Generate a gradient pattern for asteroid distribution.

        Returns:
            np.ndarray: Gradient pattern grid
        """
        # Random gradient direction
        angle = random.uniform(0, 2 * math.pi)

        return generate_gradient_pattern(
            width=self.width,
            height=self.height,
            angle=angle,
            noise_amount=0.1,
            random_generator=self.random,
        )

    def _void_pattern(self) -> np.ndarray:
        """
        Generate a void pattern with empty areas.

        Returns:
            np.ndarray: Void pattern grid
        """
        # Create several void areas
        num_voids = random.randint(1, 3)
        min_radius = min(self.width, self.height) // 8
        max_radius = min(self.width, self.height) // 3

        return generate_void_pattern(
            width=self.width,
            height=self.height,
            num_voids=num_voids,
            min_radius=min_radius,
            max_radius=max_radius,
            random_generator=self.random,
        )
