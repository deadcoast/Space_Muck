"""
src/generators/procedural_generator.py

ProceduralGenerator class: Handles procedural generation for asteroid fields.

This module contains the ProceduralGenerator class which inherits from BaseGenerator
and provides specialized functionality for generating asteroid fields using multiple
noise algorithms and cellular automaton rules.
"""

# Standard library imports
import logging

import numpy as np
from numpy.random import default_rng

# Third-party library imports
from scipy import stats

# Optional dependencies
try:
    # Check if scipy.ndimage is available without importing it
    import importlib.util

    SCIPY_AVAILABLE = importlib.util.find_spec("scipy.ndimage") is not None
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available, using fallback implementation.")

from typing import TYPE_CHECKING, Dict, Optional, Tuple

# Local application imports
from generators.base_generator import BaseGenerator
from utils.dependency_injection import inject
from utils.noise_generator import NoiseGenerator

# Type checking imports
if TYPE_CHECKING:
    from generators.asteroid_field import AsteroidField

# Local imports with correct paths
from utils.logging_setup import (
    log_exception,
    log_performance_end,
    log_performance_start,
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

        # Initialize random number generator with the provided seed
        self.rng = default_rng(seed)
        
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

    def generate_field(self, density: float = 0.3) -> np.ndarray:
        """Generate an asteroid field with the given density.
        This is the main interface method expected by AsteroidField.

        Args:
            density (float): The desired density of asteroids in the field

        Returns:
            np.ndarray: The generated asteroid field grid
        """
        return self.generate_asteroid_field(density=density)

    def generate_asteroid_field(
        self, density: float = 0.3, noise_scale: float = 0.1, threshold: float = 0.4
    ) -> np.ndarray:
        """
        Generate a procedural asteroid field using multiple noise layers.

        Args:
            density: Overall density of asteroids (0-1)
            noise_scale: Scale factor for noise generation
            threshold: Threshold value for asteroid formation

        Returns:
            np.ndarray: 2D grid with asteroid values
        """
        # Track performance
        start = log_performance_start("generate_asteroid_field")

        try:
            result = self.generate_multi_layer_asteroid_field(
                density, noise_scale, threshold
            )
            log_performance_end("generate_asteroid_field", start)
            return result
            # Already returned above
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return self.rng.random((self.height, self.width)) * density * 50

    def generate_multi_layer_asteroid_field(
        self,
        density: float = 0.3,
        noise_scale: float = 0.1,
        threshold: float = 0.4,
        octaves=None,
    ) -> np.ndarray:
        """
        Generate a multi-layered asteroid field with detailed features.

        Args:
            density: Overall density of asteroids (0-1)
            noise_scale: Scale factor for noise generation
            threshold: Threshold value for asteroid formation
            octaves: Optional list of octave values for noise generation

        Returns:
            np.ndarray: 2D grid with asteroid values
        """
        start_time = log_performance_start("generate_multi_layer_asteroid_field")

        # Initialize the grid
        grid = np.zeros((self.height, self.width), dtype=np.int32)

        # Use provided octaves or default
        # Note: octaves parameter is handled by the noise generator internally

        # Generate primary noise layer
        primary = self.generate_noise_layer("medium", scale=noise_scale * 0.2)
        # Prevent division by zero when normalizing
        min_val = primary.min()
        max_val = primary.max()
        if max_val > min_val:  # Only normalize if there's a range
            primary = (primary - min_val) / (max_val - min_val)
        else:
            primary = np.zeros_like(primary)

        # Generate secondary noise for details
        secondary = self.generate_noise_layer("high", scale=noise_scale * 0.5)
        # Prevent division by zero when normalizing
        min_val = secondary.min()
        max_val = secondary.max()
        if max_val > min_val:  # Only normalize if there's a range
            secondary = (secondary - min_val) / (max_val - min_val)
        else:
            secondary = np.zeros_like(secondary)

        # Combine noise layers
        combined = (primary * 0.7 + secondary * 0.3) * density * 1.5

        # Apply threshold to create asteroid locations
        asteroid_mask = combined > threshold

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

        log_performance_end("generate_multi_layer_asteroid_field", start_time)
        return grid

    def generate_rare_minerals(
        self,
        asteroid_grid: np.ndarray,
        rare_chance: float = 0.1,
        rare_bonus: float = 2.5,
        anomaly_chance: float = 0.05,
    ) -> np.ndarray:
        """
        Generate rare mineral distribution based on the asteroid field.

        Args:
            asteroid_grid: Base asteroid grid
            rare_chance: Chance of rare minerals (0-1)
            rare_bonus: Multiplier for rare mineral values
            anomaly_chance: Chance of anomalous rare mineral deposits

        Returns:
            np.ndarray: Grid with rare mineral values
        """
        start_time = log_performance_start("generate_rare_minerals")

        try:
            return self.generate_tiered_mineral_distribution(
                asteroid_grid, rare_chance, rare_bonus, start_time
            )
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.zeros_like(asteroid_grid, dtype=np.int8)

    def generate_tiered_mineral_distribution(
        self,
        asteroid_grid: np.ndarray,
        rare_chance: float = 0.2,
        rare_bonus: float = 2.5,
        tiers: int = 3,
    ) -> np.ndarray:
        """
        Generate tiered mineral distribution within the asteroid field.

        Args:
            asteroid_grid: Base asteroid grid
            rare_chance: Chance of rare minerals (0-1)
            rare_bonus: Multiplier for rare mineral values
            tiers: Number of rarity tiers to generate

        Returns:
            np.ndarray: Grid with tiered mineral values
        """
        start_time = log_performance_start("generate_tiered_mineral_distribution")
        rare_grid = np.zeros_like(asteroid_grid, dtype=np.int32)

        # No rare minerals where there are no asteroids
        asteroid_mask = asteroid_grid > 0

        # Generate noise for rare mineral distribution
        rare_noise = self.generate_noise_layer("high", scale=0.03)
        # Prevent division by zero when normalizing rare_noise
        min_val = rare_noise.min()
        max_val = rare_noise.max()
        if max_val > min_val:  # Only normalize if there's a range
            rare_noise = (rare_noise - min_val) / (max_val - min_val)
        else:
            rare_noise = np.zeros_like(rare_noise)

        # Generate precious noise (even rarer)
        precious_noise = self.generate_noise_layer("high", scale=0.015)
        # Prevent division by zero when normalizing precious_noise
        min_val = precious_noise.min()
        max_val = precious_noise.max()
        if max_val > min_val:  # Only normalize if there's a range
            precious_noise = (precious_noise - min_val) / (max_val - min_val)
        else:
            precious_noise = np.zeros_like(precious_noise)

        # Generate anomaly noise (rarest)
        anomaly_noise = self.generate_noise_layer("detail", scale=0.01)
        # Prevent division by zero when normalizing anomaly_noise
        min_val = anomaly_noise.min()
        max_val = anomaly_noise.max()
        if max_val > min_val:  # Only normalize if there's a range
            anomaly_noise = (anomaly_noise - min_val) / (max_val - min_val)
        else:
            anomaly_noise = np.zeros_like(anomaly_noise)

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
            & (np.random.Generator(asteroid_grid.shape) < anomaly_chance)
        )

        # Set rare mineral values based on asteroid values and rare_bonus
        if np.any(rare_mask):
            rare_grid[rare_mask] = (
                asteroid_grid[rare_mask] * (rare_bonus * 0.5)
            ).astype(np.int32)
        if np.any(precious_mask):
            rare_grid[precious_mask] = (
                asteroid_grid[precious_mask] * (rare_bonus * 0.8)
            ).astype(np.int32)
        if np.any(anomaly_mask):
            rare_grid[anomaly_mask] = (asteroid_grid[anomaly_mask] * rare_bonus).astype(
                np.int32
            )

        # Apply cellular automaton to create more natural clusters
        rare_grid = self.apply_cellular_automaton(
            rare_grid,
            birth_set={3, 4, 5},
            survival_set={2, 3, 4, 5},
            iterations=2,
            wrap=True,
        )

        log_performance_end("generate_tiered_mineral_distribution", start_time)
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
            cx = self.rng.integers(0, self.width)
            cy = self.rng.integers(0, self.height)

            # Random size
            size = self.rng.integers(10, max_size)

            # Create void
            for y in range(max(0, cy - size), min(self.height, cy + size + 1)):
                for x in range(max(0, cx - size), min(self.width, cx + size + 1)):
                    # Calculate distance from center
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

                    # Clear cells within the void radius with some noise at the edges
                    if dist < size * (0.7 + self.rng.random() * 0.3):
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
        # Create a copy of the grid to store the result
        result_grid = grid.copy()

        # Find edges in the grid
        edges = self._find_grid_edges(grid)

        # Apply decay to the identified edges
        result_grid = self._apply_decay_to_edges(result_grid, edges, decay_factor)

        return result_grid

    def _find_grid_edges(self, grid: np.ndarray) -> np.ndarray:
        """
        Find edges in the grid by detecting cells with empty neighbors.

        Args:
            grid: Input asteroid grid

        Returns:
            np.ndarray: Boolean mask of edge cells
        """
        edges = np.zeros_like(grid, dtype=bool)

        # Iterate through the grid (excluding the outer border)
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if grid[y, x] > 0 and self._has_empty_neighbor(grid, y, x):
                    edges[y, x] = True

        return edges

    def _has_empty_neighbor(self, grid: np.ndarray, y: int, x: int) -> bool:
        """
        Check if a cell has at least one empty neighboring cell.

        Args:
            grid: Input asteroid grid
            y: Y-coordinate of the cell
            x: X-coordinate of the cell

        Returns:
            bool: True if the cell has at least one empty neighbor
        """
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                # Skip the cell itself
                if dx == 0 and dy == 0:
                    continue

                # Check if neighbor is empty
                if grid[y + dy, x + dx] == 0:
                    return True

        return False

    def _apply_decay_to_edges(
        self, grid: np.ndarray, edges: np.ndarray, decay_factor: float
    ) -> np.ndarray:
        """
        Apply random decay to edge cells to create a more natural appearance.

        Args:
            grid: Input asteroid grid
            edges: Boolean mask of edge cells
            decay_factor: Factor controlling edge decay rate

        Returns:
            np.ndarray: Grid with decay applied to edges
        """
        result_grid = grid.copy()

        # Create a random number generator
        rng = np.random.default_rng(self.seed)

        # Apply decay to edges
        for y in range(self.height):
            for x in range(self.width):
                if edges[y, x] and rng.random() < decay_factor:
                    # Apply random decay factor between 0.5 and 0.8
                    result_grid[y, x] = int(
                        result_grid[y, x] * (0.5 + rng.random() * 0.3)
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
            # Generate the asteroid field and rare minerals
            grid, rare_grid = self._generate_asteroid_and_rare_grids()

            # Generate energy grid with rare mineral bonuses
            energy_grid = self._generate_energy_grid(grid, rare_grid)

            log_performance_end("generate", start_time)

            # Return all grids in a dictionary
            return {"grid": grid, "rare_grid": rare_grid, "energy_grid": energy_grid}

        except Exception as e:
            log_exception(e)
            # Return simple fallback grids if generation fails
            logging.warning("Using fallback generation due to error")
            fallback_grid = np.random.Generator(1, 0.2, (self.height, self.width)) * 50
            return {
                "grid": fallback_grid,
                "rare_grid": np.zeros_like(fallback_grid, dtype=np.int8),
                "energy_grid": fallback_grid.astype(np.float32) / 100.0,
            }

    def generate_energy_sources(
        self,
        asteroid_grid: np.ndarray,
        rare_grid: np.ndarray,
        energy_chance: float = 0.1,
        energy_value: float = 3.0,
    ) -> np.ndarray:
        """
        Generate energy sources within the asteroid field.

        Args:
            asteroid_grid: Base asteroid grid
            rare_grid: Rare minerals grid
            energy_chance: Chance of energy sources (0-1)
            energy_value: Base value for energy sources

        Returns:
            np.ndarray: Grid with energy source values
        """
        start_time = log_performance_start("generate_energy_sources")

        try:
            # Initialize energy grid
            energy_grid = np.zeros_like(asteroid_grid, dtype=np.float32)

            # Only place energy sources where asteroids exist
            asteroid_mask = asteroid_grid > 0

            # Generate noise for energy distribution
            energy_noise = self.generate_noise_layer("medium", scale=0.04)
            # Prevent division by zero when normalizing energy_noise
            min_val = energy_noise.min()
            max_val = energy_noise.max()
            if max_val > min_val:  # Only normalize if there's a range
                energy_noise = (energy_noise - min_val) / (max_val - min_val)
            else:
                energy_noise = np.zeros_like(energy_noise)

            # Random chance for each asteroid to contain energy sources
            energy_mask = np.logical_and(
                asteroid_mask,
                np.logical_or(
                    energy_noise > (1 - energy_chance),
                    np.random.Generator(asteroid_grid.shape) < energy_chance * 0.5,
                ),
            )

            # Set energy values
            energy_grid[energy_mask] = (
                asteroid_grid[energy_mask].astype(np.float32)
                * np.random.Generator(0.5, energy_value, np.sum(energy_mask))
                / 100.0
            )

            # Apply bonuses from rare minerals
            rare_mask = rare_grid > 0
            if np.any(rare_mask & energy_mask):
                energy_grid[
                    rare_mask & energy_mask
                ] *= 2.0  # Double energy in rare mineral locations

            log_performance_end("generate_energy_sources", start_time)
            return energy_grid

        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.zeros_like(asteroid_grid, dtype=np.float32)

    def _generate_energy_grid(
        self, grid: np.ndarray, rare_grid: np.ndarray
    ) -> np.ndarray:
        """
        Generate energy grid based on asteroid values and apply rare mineral bonuses.

        Args:
            grid: The asteroid grid
            rare_grid: The rare minerals grid

        Returns:
            np.ndarray: The energy grid with rare mineral bonuses applied
        """
        # Generate energy grid based on asteroid values
        energy_grid = grid.astype(np.float32) / 100.0

        # Apply rare mineral bonuses to energy grid
        rare_bonus = self.get_parameter("rare_bonus_multiplier", 2.0)
        for rare_level in range(1, 4):
            rare_mask = rare_grid == rare_level
            energy_grid[rare_mask] *= rare_bonus * rare_level

        return energy_grid

    def _generate_asteroid_and_rare_grids(self):
        """Generate the asteroid field grid and rare minerals grid.

        Returns:
            tuple: (asteroid_grid, rare_minerals_grid)
        """
        # Generate the main asteroid grid
        density = self.get_parameter("field_density", 0.3)
        grid = self.generate_asteroid_field(density=density)

        # Generate rare minerals distribution
        rare_chance = self.get_parameter("rare_chance", 0.1)
        rare_grid = self.generate_rare_minerals(grid, rare_chance=rare_chance)

        return grid, rare_grid


def create_field_with_multiple_algorithms(
    width: int = 100,
    height: int = 100,
    seed: Optional[int] = None,
    rare_chance: float = 0.1,
    energy_chance: float = 0.1,
) -> "AsteroidField":
    """
    Create an asteroid field using multiple procedural generation algorithms.

    This function combines various generation techniques to create a rich,
    detailed asteroid field with rare minerals and interesting features.

    Args:
        width: Width of the field
        height: Height of the field
        seed: Random seed for reproducibility
        rare_chance: Chance of rare minerals appearing
        energy_chance: Chance of energy sources appearing

    Returns:
        AsteroidField: Fully initialized asteroid field
    """
    # Create the asteroid field
    # Import here to avoid circular imports
    from generators.asteroid_field import AsteroidField as AsteroidFieldImpl

    # Also make AsteroidField available at the module level for mocking in tests
    globals()["AsteroidField"] = AsteroidFieldImpl
    field = AsteroidFieldImpl(width=width, height=height)

    # Initialize the procedural generator
    generator = ProceduralGenerator(seed=seed, width=width, height=height)

    # Generate the asteroid grid
    field.grid = generator.generate_asteroid_field(density=0.3)

    # Generate rare minerals
    field.rare_grid = generator.generate_rare_minerals(
        field.grid, rare_chance=rare_chance
    )

    # Generate energy sources using the energy_chance parameter
    field.energy_grid = generator.generate_energy_sources(
        field.grid, field.rare_grid, energy_chance=energy_chance
    )

    logging.info("Created asteroid field with seed %s, size %sx%s", seed, width, height)
    logging.info("Field contains %s asteroids", np.sum(field.grid > 0))
    logging.info(
        "Rare minerals: %s rare, %s precious, %s anomaly",
        np.sum(field.rare_grid == 1),
        np.sum(field.rare_grid == 2),
        np.sum(field.rare_grid == 3),
    )

    return field
