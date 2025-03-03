"""
AsteroidGenerator class: Specialized generator for asteroid fields.

This module contains the AsteroidGenerator class which inherits from BaseGenerator
and provides specialized functionality for generating asteroid fields with various
patterns and distributions.
"""

# Standard library imports
import logging
from typing import Any, Dict, Optional, Set, Tuple

# Third-party library imports
import numpy as np

# Local application imports
from src.entities.base_generator import BaseGenerator
from src.utils.dependency_injection import inject
from src.utils.noise_generator import NoiseGenerator
from src.utils.logging_setup import (
    log_performance_start,
    log_performance_end,
    log_exception,
)

# Try to import optional dependencies
try:
    import scipy.signal as signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Define signal as None to avoid unbound variable errors
    signal = None
    logging.warning(
        "SciPy not available. Using manual implementation for cellular automaton."
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

        # Initialize cache
        self._cache = {}

        logging.info(
            f"AsteroidGenerator initialized: ID: {self.entity_id}, Seed: {self.seed}"
        )

    def _get_cache_key(self, method_name: str, **kwargs) -> str:
        """Generate a cache key for a method call with specific parameters."""
        # Convert kwargs to a sorted string representation
        params_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{method_name}_{params_str}_{self.width}_{self.height}_{self.seed}"

    def _cache_result(self, key: str, result: Any) -> None:
        """Cache a result with the given key."""
        self._cache[key] = result

    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get a cached result if it exists."""
        return self._cache.get(key)

    def generate_field(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a complete asteroid field.

        Returns:
            tuple: (asteroid_grid, metadata)
        """
        start_time = log_performance_start("generate_field")

        try:
            return self._cache_key_handler(start_time)
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.random.binomial(1, 0.1, (self.height, self.width)), {
                "seed": self.seed
            }

    def _cache_key_handler(self, start_time):
        # Create cache key
        cache_key = self._get_cache_key(
            "generate_field",
            density=self.get_parameter("density"),
            pattern_strength=self.get_parameter("pattern_strength"),
        )

        # Check cache
        cached_result = self._get_cached_result(key=cache_key)
        if cached_result is not None:
            log_performance_end("generate_field", start_time, "cached")
            return cached_result

        # Create empty grid
        grid = np.zeros((self.height, self.width), dtype=float)

        # Apply base noise layer
        noise_grid = self._generate_base_noise()

        # Normalize grid to 0-1 range
        grid = (noise_grid - noise_grid.min()) / (
            noise_grid.max() - noise_grid.min() + 1e-10
        )

        # Apply threshold to create binary asteroid field
        density = self.get_parameter("density", 0.2)
        asteroid_grid = (grid > (1 - density)).astype(int)

        # Apply cellular automaton to smooth field
        asteroid_grid = self.apply_cellular_automaton(
            asteroid_grid,
            birth_set={3, 4, 5},
            survival_set={2, 3, 4, 5},
            iterations=2,
            wrap=True,
        )

        # Generate metadata
        metadata = {
            "seed": self.seed,
            "density": density,
            "pattern_strength": self.get_parameter("pattern_strength", 0.4),
            "asteroid_count": np.sum(asteroid_grid),
        }

        # Cache result
        result = (asteroid_grid, metadata)
        return self._cache_result_handler(
            cache_key, result, "generate_field", start_time
        )

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
            return self._create_cache_key_handler(asteroid_grid, start_time)
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return asteroid_grid * 3

    def _create_cache_key_handler(self, asteroid_grid, start_time):
        # Create cache key
        cache_key = self._get_cache_key(
            "generate_values",
            grid_hash=hash(asteroid_grid.tobytes()),
            value_mean=self.get_parameter("value_mean"),
            value_stddev=self.get_parameter("value_stddev"),
        )

        # Check cache
        cached_result = self._get_cached_result(key=cache_key)
        if cached_result is not None:
            log_performance_end("generate_values", start_time, "cached")
            return cached_result

        # Generate base values using perlin noise for spatial coherence
        value_noise = self.generate_noise_layer("medium", scale=0.08)

        # Get value parameters
        value_mean = self.get_parameter("value_mean", 5.0)
        value_stddev = self.get_parameter("value_stddev", 2.0)

        # Scale noise to desired mean and standard deviation
        value_noise = value_noise * value_stddev + value_mean

        # Apply values only to non-zero cells
        value_grid = asteroid_grid * value_noise.astype(int)

        # Ensure minimum value for non-zero cells
        value_grid[value_grid > 0] = np.maximum(value_grid[value_grid > 0], 1)

        # Add value clusters - some areas have higher value asteroids
        cluster_tendency = self.get_parameter("cluster_tendency", 0.6)
        num_clusters = int(np.sqrt(self.width * self.height) * 0.05)

        # Find cells with asteroids
        asteroid_cells = np.argwhere(value_grid > 0)

        if len(asteroid_cells) > 0:
            for _ in range(num_clusters):
                # Pick a random asteroid cell as cluster center
                idx = np.random.randint(0, len(asteroid_cells))
                center_y, center_x = asteroid_cells[idx]

                # Apply multiplier to asteroids in radius
                cluster_radius = int(min(self.width, self.height) * 0.1)
                for y in range(
                    max(0, center_y - cluster_radius),
                    min(self.height, center_y + cluster_radius + 1),
                ):
                    for x in range(
                        max(0, center_x - cluster_radius),
                        min(self.width, center_x + cluster_radius + 1),
                    ):
                        if value_grid[y, x] > 0:
                            # Calculate distance from center
                            distance = np.sqrt(
                                (x - center_x) ** 2 + (y - center_y) ** 2
                            )

                            # Apply multiplier with falloff based on distance
                            if distance <= cluster_radius:
                                falloff = 1.0 - (distance / cluster_radius)
                                multiplier = 1.0 + (cluster_tendency * falloff)
                                value_grid[y, x] = int(value_grid[y, x] * multiplier)

        return self._cache_result_handler(
            cache_key, value_grid, "generate_values", start_time
        )

    def generate_rare_resources(self, value_grid: np.ndarray) -> np.ndarray:
        """
        Generate rare resource distribution across the asteroid field.

        Args:
            value_grid: Grid with asteroid values

        Returns:
            np.ndarray: Grid with rare resource indicators (0 = common, 1 = rare, 2 = anomaly)
        """
        start_time = log_performance_start("generate_rare_resources")

        try:
            return self._generate_rare_resource_handler(value_grid, start_time)
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.zeros_like(value_grid)

    def _generate_rare_resource_handler(self, value_grid, start_time):
        # Create cache key
        cache_key = self._get_cache_key(
            "generate_rare_resources",
            grid_hash=hash(value_grid.tobytes()),
            rare_chance=self.get_parameter("rare_chance"),
        )

        # Check cache
        cached_result = self._get_cached_result(key=cache_key)
        if cached_result is not None:
            log_performance_end("generate_rare_resources", start_time, "cached")
            return cached_result

        # Create a rare resource grid with the same shape
        rare_grid = np.zeros_like(value_grid, dtype=np.int8)

        # Generate coherent noise patterns for rare distribution
        rare_noise = self.generate_noise_layer("medium", scale=0.05)
        anomaly_noise = self.generate_noise_layer("high", scale=0.03)

        # Get rare resource parameters
        rare_chance = self.get_parameter("rare_chance", 0.05)

        # Normalize noise to 0-1 range
        rare_noise = (rare_noise - rare_noise.min()) / (
            rare_noise.max() - rare_noise.min() + 1e-10
        )
        anomaly_noise = (anomaly_noise - anomaly_noise.min()) / (
            anomaly_noise.max() - anomaly_noise.min() + 1e-10
        )

        # No rare resources where there are no asteroids
        asteroid_mask = value_grid > 0

        # Apply thresholds for different resource types
        rare_threshold = 1 - rare_chance
        anomaly_threshold = 1 - rare_chance * 0.2  # Anomalies are rarer

        # Apply rare resources (1)
        rare_mask = (rare_noise > rare_threshold) & asteroid_mask
        rare_grid[rare_mask] = 1

        # Apply anomalies (2) - overrides rare
        anomaly_mask = (anomaly_noise > anomaly_threshold) & asteroid_mask
        rare_grid[anomaly_mask] = 2

        if SCIPY_AVAILABLE:
            # Use scipy for efficient implementation
            # Create a binary grid for each resource type
            rare_binary = (rare_grid == 1).astype(np.int8)
            anomaly_binary = (rare_grid == 2).astype(np.int8)

            # Apply cellular automaton to each type
            rare_binary = self._apply_cellular_automaton_scipy(
                rare_binary, birth_set={3}, survival_set={2, 3, 4}, iterations=1
            )
            anomaly_binary = self._apply_cellular_automaton_scipy(
                anomaly_binary, birth_set={2, 3}, survival_set={2, 3}, iterations=1
            )

        else:
            # Manual implementation
            # Apply to rare resources
            rare_binary = (rare_grid == 1).astype(np.int8)
            rare_binary = self.apply_cellular_automaton(
                rare_binary, birth_set={3}, survival_set={2, 3, 4}, iterations=1
            )

            # Apply to anomalies
            anomaly_binary = (rare_grid == 2).astype(np.int8)
            anomaly_binary = self.apply_cellular_automaton(
                anomaly_binary, birth_set={2, 3}, survival_set={2, 3}, iterations=1
            )

        # Combine results back into rare_grid
        rare_grid = np.zeros_like(rare_grid)
        rare_grid[rare_binary > 0] = 1
        rare_grid[anomaly_binary > 0] = 2
        # Ensure rare resources only exist where there are asteroids
        rare_grid = rare_grid * asteroid_mask

        return self._cache_result_handler(
            cache_key, rare_grid, "generate_rare_resources", start_time
        )

    def _cache_result_handler(self, cache_key, arg1, arg2, start_time):
        self._cache_result(cache_key, arg1)
        log_performance_end(arg2, start_time)
        return arg1

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

    def _apply_cellular_automaton_scipy(
        self,
        grid: np.ndarray,
        birth_set: Set[int] = {3},
        survival_set: Set[int] = {2, 3},
        iterations: int = 1,
    ) -> np.ndarray:
        """
        Apply cellular automaton rules using scipy for efficiency.

        Args:
            grid: Input binary grid
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            iterations: Number of iterations to perform

        Returns:
            np.ndarray: Evolved grid
        """
        if birth_set is None:
            birth_set = {3}
        if survival_set is None:
            survival_set = {2, 3}
        # Convert to binary grid
        binary_grid = (grid > 0).astype(np.int8)
        result_grid = binary_grid.copy()

        # Create kernel for neighbor counting
        kernel = np.ones((3, 3), dtype=np.int8)
        kernel[1, 1] = 0  # Don't count the cell itself

        # Apply cellular automaton for specified iterations
        for _ in range(iterations):
            # Count neighbors using convolution
            neighbors = signal.convolve2d(
                result_grid, kernel, mode="same", boundary="wrap"
            )

            # Create masks for rule application (vectorized operations)
            alive_mask = result_grid == 1
            dead_mask = result_grid == 0

            # Create masks for birth and survival based on neighbor counts
            birth_mask = np.zeros_like(neighbors, dtype=bool)
            survival_mask = np.zeros_like(neighbors, dtype=bool)

            for n in birth_set:
                birth_mask |= neighbors == n

            for n in survival_set:
                survival_mask |= neighbors == n

            # Apply rules using vectorized operations
            new_grid = np.zeros_like(result_grid)
            new_grid[alive_mask & survival_mask] = 1  # Cells that survive
            new_grid[dead_mask & birth_mask] = 1  # Cells that are born

            result_grid = new_grid

        return result_grid
