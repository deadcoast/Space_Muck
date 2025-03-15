"""
AsteroidField class: Manages the asteroid field grid and symbiote entities.

This module contains the AsteroidField class which handles:
- Procedural asteroid generation using multiple noise algorithms
- Cellular automaton simulation for asteroid evolution
- Energy flow modeling for dynamic resource distribution
- Symbiote race ecosystems that evolve within the field
- Optimized rendering and interaction systems
"""

# Standard library imports
import itertools
import logging
import math
import random
import time
from typing import Dict, List, Optional, Set, Tuple

# Third-party library imports
import numpy as np
import pygame
import scipy

# Local application imports
from config import GRID_HEIGHT, GRID_WIDTH, VIEW_HEIGHT, VIEW_WIDTH
from entities.miner_entity import MinerEntity
from generators.procedural_generator import ProceduralGenerator
from generators.symbiote_evolution_generator import SymbioteEvolutionGenerator

# Handle optional dependencies gracefully
try:
    import scipy.ndimage as ndimage
    import scipy.signal as signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available. Falling back to manual implementations.")

try:
    from perlin_noise import PerlinNoise

    PERLIN_AVAILABLE = True
except ImportError:
    PERLIN_AVAILABLE = False
    logging.warning("perlin_noise package not available. Some features may be limited.")

# Local application imports

from utils.logging_setup import (
    LogContext,
    log_exception,
    log_performance_end,
    log_performance_start,
)

# Try to import the optimized generators if available
try:
    from generators.asteroid_generator import AsteroidGenerator

    ASTEROID_GENERATOR_AVAILABLE = True
except ImportError:
    ASTEROID_GENERATOR_AVAILABLE = False
    logging.warning(
        "AsteroidGenerator not available. Using ProceduralGenerator as fallback."
    )

# Check if scipy is available for optimized cellular automaton

SCIPY_AVAILABLE = hasattr(scipy, "signal")
if not SCIPY_AVAILABLE:
    logging.warning(
        "SciPy signal module not available. Using manual implementation for cellular automaton."
    )


# Forward references for type hints
class NotificationManager:
    """Type hint for NotificationManager class."""

    pass


class Player:
    """Type hint for Player class."""

    pass


class AsteroidField:
    """
    Represents a field of asteroids on a grid.
    Uses NumPy for performance and supports complex cellular automaton rules.
    """

    def __init__(
        self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT, seed: int = None
    ) -> None:
        """
        Initialize a new asteroid field with the specified dimensions.

        Args:
            width: Width of the field in cells
            height: Height of the field in cells
            seed: Optional seed for random number generation
        """
        self.width = width
        self.height = height

        # NumPy arrays for better performance
        # 0 = empty, 1-100 = asteroid value
        self.grid = np.zeros((height, width), dtype=np.int16)

        # Rare status: 0 = normal, 1 = rare
        self.rare_grid = np.zeros((height, width), dtype=np.int8)

        # Energy levels for each cell (affects growth and stability)
        self.energy_grid = np.zeros((height, width), dtype=np.float32)

        # Entity grid (0 = empty, 1,2,3 = race ID)
        self.entity_grid = np.zeros((height, width), dtype=np.int8)

        # Specialized grids for each race's influence
        self.influence_grids = {}

        # Camera position (center of view)
        self.camera_x = width // 2
        self.camera_y = height // 2
        self.zoom = 1.0

        # Display settings
        self.show_energy = False
        self.show_grid_lines = True
        self.show_influence = False
        self.use_advanced_rendering = True
        self.draw_mode = "standard"  # Options: standard, heatmap, influence, entities

        # Cellular automata parameters (default: similar to Conway's Game of Life)
        self.birth_set: Set[int] = {3}
        self.survival_set: Set[int] = {2, 3}
        self.energy_decay = 0.02  # Energy decay rate
        self.energy_spread = 0.1  # How much energy spreads to neighbors
        self.apply_edge_wrapping = True  # Whether to wrap around the grid edges

        # Asteroid parameters
        self.regen_rate: float = 0.01
        self.rare_chance: float = 0.1
        self.rare_bonus_multiplier: float = 3.0
        self.anomaly_chance: float = 0.01  # Chance for anomalous minerals

        # Advanced generation parameters
        self.pattern_complexity = 0.5  # 0-1: Higher = more complex patterns
        self.field_density = 0.3  # 0-1: Higher = more asteroids
        self.turbulence = 0.5  # 0-1: Higher = more chaotic distribution

        # Track mining races
        self.races: List[MinerEntity] = []
        self.race_counter = 0  # Used for generating unique race IDs
        self.max_races = 10  # Maximum number of races the field can support

        # Stats tracking
        self.total_asteroids = 0
        self.total_rare = 0
        self.total_energy = 0.0
        self.asteroid_history = []  # Track asteroid count over time
        self.rare_history = []  # Track rare asteroid count over time
        self.energy_history = []  # Track energy levels over time
        self.race_population_history = {}  # Track race populations over time

        # Cache for rendering
        self.cell_colors = {}  # Cache for asteroid colors
        self.entity_colors = {}  # Cache for entity colors
        self.rare_overlay = None  # Surface for rare asteroid overlay
        self.anomaly_overlay = None  # Surface for anomaly mineral overlay
        self.render_cache = {}  # Cache for pre-rendered cells
        self.last_view_bounds = None  # Track last drawn view bounds
        self.redraw_needed = True  # Flag to indicate redraw is needed

        # References to other objects
        self.player = None
        self.notifier = None

        # Initialize color gradients for rendering
        self.asteroid_gradient = [
            (80, 80, 80),  # Low value
            (120, 120, 120),  # Medium value
            (180, 180, 180),  # High value
        ]

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize with random pattern
        self.initialize_patterns()

        # Create render surface
        self.render_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        self.minimap_surface = pygame.Surface((150, 150))

    def _initialize_field_with_generator(self) -> None:
        """
        Initialize the field using the ProceduralGenerator.
        Handles grid generation, rare asteroids, energy grid, and life patterns.
        """
        generator = ProceduralGenerator(self.width, self.height)
        self.grid = generator.generate_field(
            density=self.field_density,
            complexity=self.pattern_complexity,
            turbulence=self.turbulence,
        )

        # Use numpy random Generator API instead of legacy functions
        # Use current time as seed for reproducibility
        rng = np.random.default_rng(int(time.time()))
        self.rare_grid = rng.random((self.height, self.width)) < self.rare_chance
        self.energy_grid = rng.random((self.height, self.width))

        self.total_asteroids = np.sum(self.grid > 0)
        self.total_rare = np.sum(self.rare_grid)
        self.total_energy = np.sum(self.energy_grid)

        self._life_patterns_handler()

        logging.info(
            f"Field initialized with {self.total_asteroids} asteroids, "
            + f"{self.total_rare} rare minerals."
        )

    def initialize_patterns(self) -> None:
        """
        Initialize the grid with procedurally generated asteroid formations
        using the ProceduralGenerator class.
        """
        with LogContext("Asteroid field initialization"):
            self._initialize_handler()

            try:
                self._initialize_field_with_generator()
            except Exception as e:
                log_exception("Field initialization failed", e)
                logging.error(f"Failed to initialize patterns: {str(e)}")
                self._legacy_initialize_patterns()

    def generate_field_with_procedural_generator(self) -> None:
        """
        Generate asteroid field using either the optimized AsteroidGenerator class
        or the ProceduralGenerator class as a fallback.
        """
        try:
            self._asteroid_handler()
        except Exception as e:
            self._handle_asteroid_generation_error(
                e, "Failed to generate field with generator: "
            )

    def _asteroid_handler(self) -> None:
        """Generate asteroid field using available generators.

        This method selects the appropriate generator based on availability:
        - Uses optimized AsteroidGenerator if available
        - Falls back to ProceduralGenerator if optimized generator is not available
        - Handles grid initialization and ensures all required grids are populated

        Raises:
            Exception: Passes through exceptions from generator methods for centralized handling
        """
        # Use the optimized AsteroidGenerator if available, otherwise fall back to ProceduralGenerator
        seed = random.randint(1, 10000)

        # Common parameters for both generator types
        common_params = {
            "width": self.width,
            "height": self.height,
            "seed": seed,
        }

        # Performance tracking
        start_time = log_performance_start("generate_field_with_generator")

        try:
            if ASTEROID_GENERATOR_AVAILABLE:
                self._generate_asteroid_field_with_optimized_generator(
                    common_params, seed
                )
                logging.info(
                    f"Asteroid field generated using optimized AsteroidGenerator (seed: {seed})"
                )
            else:
                self._generate_asteroid_field_with_procedural_generator(
                    common_params, seed
                )
                logging.info(
                    f"Asteroid field generated using ProceduralGenerator (seed: {seed})"
                )

            # Initialize energy grid based on asteroid values if not included in result
            if np.all(self.energy_grid == 0):
                self._initialize_energy_grid()

        except Exception as e:
            self._handle_asteroid_generation_error(
                e, "Error in asteroid field generation: "
            )
        finally:
            # Log performance regardless of success or failure
            log_performance_end("generate_field_with_generator", start_time)

    def _handle_asteroid_generation_error(self, e, error_prefix):
        log_exception(e)
        logging.error(f"{error_prefix}{str(e)}")
        raise

    def _generate_asteroid_field_with_optimized_generator(
        self, common_params: dict, seed: int
    ) -> None:
        """Generate asteroid field using the optimized AsteroidGenerator.

        Args:
            common_params: Dictionary containing common parameters for the generator
            seed: Random seed for reproducibility

        Raises:
            Exception: If there's an error during generation with the optimized generator
        """
        try:
            # Create an AsteroidGenerator with optimized caching
            generator = AsteroidGenerator(**common_params)

            # Set parameters
            generator.set_parameter("density", self.field_density)
            generator.set_parameter("pattern_strength", self.pattern_complexity)
            generator.set_parameter("cluster_tendency", self.turbulence)
            generator.set_parameter("rare_chance", self.rare_chance)

            # Generate the asteroid field
            asteroid_grid, _ = generator.generate_field()

            # Generate values for asteroids
            value_grid = generator.generate_values(asteroid_grid)

            # Generate rare resources
            rare_grid = generator.generate_rare_resources(value_grid)

            # Set the grids
            self.grid = value_grid
            self.rare_grid = rare_grid

            # Generate energy grid based on asteroid values
            self.energy_grid = value_grid.astype(np.float32) / 100.0

        except Exception as e:
            logging.error(f"Error in optimized asteroid generator: {str(e)}")
            raise

    def _generate_asteroid_field_with_procedural_generator(
        self, common_params: dict, seed: int
    ) -> None:
        """Generate asteroid field using the ProceduralGenerator.

        Args:
            common_params: Dictionary containing common parameters for the generator
            seed: Random seed for reproducibility

        Raises:
            Exception: If there's an error during generation with the procedural generator
        """
        try:
            # Fall back to the original ProceduralGenerator
            generator = ProceduralGenerator(**common_params)

            # Set parameters individually instead of using a parameters dict
            generator.set_parameter("pattern_complexity", self.pattern_complexity)
            generator.set_parameter("field_density", self.field_density)
            generator.set_parameter("turbulence", self.turbulence)
            generator.set_parameter("birth_set", self.birth_set)
            generator.set_parameter("survival_set", self.survival_set)
            generator.set_parameter("anomaly_chance", self.anomaly_chance)
            generator.set_parameter("rare_chance", self.rare_chance)
            generator.set_parameter("rare_bonus_multiplier", self.rare_bonus_multiplier)

            # Generate the asteroid field grid
            result = generator.generate()

            if isinstance(result, dict):
                # If the generator returns a dictionary with multiple grids
                if "grid" in result:
                    self.grid = result["grid"]
                if "rare_grid" in result:
                    self.rare_grid = result["rare_grid"]
                if "energy_grid" in result:
                    self.energy_grid = result["energy_grid"]
            else:
                # If the generator returns just the main grid
                self.grid = result

                # Generate rare minerals distribution if not included in result
                if np.all(self.rare_grid == 0):
                    self._generate_rare_minerals()

        except Exception as e:
            logging.error(f"Error in procedural asteroid generator: {str(e)}")
            raise

    def _generate_rare_minerals(self) -> None:
        """
        Generate rare minerals distribution based on the asteroid grid.
        """
        # Create thresholds for rare minerals
        asteroid_threshold = 0.7 - self.field_density * 0.3
        rare_threshold = asteroid_threshold + ((1.0 - asteroid_threshold) * 0.6)
        anomaly_threshold = rare_threshold + ((1.0 - rare_threshold) * 0.7)

        # Normalize grid values to 0-1 range for threshold comparison
        max_val = np.max(self.grid) if np.max(self.grid) > 0 else 1
        normalized_grid = self.grid / max_val

        # Apply rare mineral distribution
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] > 0:
                    value = normalized_grid[y, x]

                    # Chance for rare minerals in high-value areas
                    if value > rare_threshold and random.random() < self.rare_chance:
                        self.rare_grid[y, x] = 1
                        # Rare asteroids worth more
                        self.grid[y, x] = int(
                            self.grid[y, x] * self.rare_bonus_multiplier
                        )

                    # Chance for anomalous minerals in extremely high-value areas
                    elif (
                        value > anomaly_threshold
                        and random.random() < self.anomaly_chance
                    ):
                        self.rare_grid[y, x] = 2  # Mark as anomaly
                        # Anomalies are worth even more
                        self.grid[y, x] = int(
                            self.grid[y, x] * self.rare_bonus_multiplier * 2
                        )

    def _initialize_energy_grid(self) -> None:
        """
        Initialize energy grid based on asteroid values.
        """
        for y, x in itertools.product(range(self.height), range(self.width)):
            if self.grid[y, x] > 0:
                # Energy level based on asteroid value (normalized to 0-0.5 range)
                max_val = 200  # Expected maximum asteroid value
                self.energy_grid[y, x] = min(self.grid[y, x] / max_val, 1.0) * 0.5

    def generate_symbiote_evolution(
        self, num_colonies: int = None, iterations: int = None
    ) -> None:
        """
        Generate symbiote evolution using the SymbioteEvolutionGenerator class.

        Args:
            num_colonies: Number of initial colonies to generate
            iterations: Number of evolution iterations to simulate
        """
        try:
            # Create a SymbioteEvolutionGenerator with the same dimensions as the field
            generator = SymbioteEvolutionGenerator(
                width=self.width, height=self.height, seed=random.randint(1, 10000)
            )

            # Set parameters
            generator.set_parameter("initial_colonies", num_colonies or 3)
            generator.set_parameter("evolution_iterations", iterations or 10)
            generator.set_parameter("colony_size", 5)
            generator.set_parameter("mutation_chance", 0.1)
            generator.set_parameter("environmental_hostility", 0.3)

            # Generate initial colonies
            colony_grid, colony_metadata = generator.generate_initial_colonies(
                num_colonies
            )

            # Generate mineral distribution
            mineral_grid = generator.generate_mineral_distribution()

            # Simulate evolution
            evolved_grid, evolution_history = generator.simulate_evolution(
                colony_grid, mineral_grid, iterations
            )

            # Update entity grid with evolved colonies
            self.entity_grid = evolved_grid

            # Generate mutation map
            mutation_grid = generator.generate_mutation_map(
                evolved_grid, evolution_history
            )

            # Store evolution history and metadata
            self._symbiote_evolution_history = evolution_history
            self._symbiote_colony_metadata = colony_metadata

            # Create influence grids based on mutation map
            for race_id in range(1, 4):  # Assuming up to 3 races
                race_mask = self.entity_grid == race_id
                if np.any(race_mask):
                    # Create influence grid for this race
                    self.influence_grids[race_id] = mutation_grid * race_mask

            logging.info(
                f"Symbiote evolution generated with {len(evolution_history)} iterations"
            )
            logging.info(f"Colony metadata: {colony_metadata}")

        except Exception as e:
            log_exception(e)
            logging.error(f"Failed to generate symbiote evolution: {str(e)}")

    def _legacy_initialize_patterns(self) -> None:
        """
        Legacy method to initialize the grid with procedurally generated asteroid formations
        using Perlin noise for natural-looking distribution. Used as a fallback if the
        generator classes fail.
        """
        logging.warning("Using legacy pattern initialization method")

        self._initialize_handler()

        # Generate noise map using Perlin noise
        noise_map = self._generate_perlin_noise_map()

        # Normalize the noise map to 0-1 range
        noise_map = self._normalize_noise_map(noise_map)

        # Calculate thresholds for different asteroid types
        asteroid_threshold, rare_threshold, anomaly_threshold = (
            self._calculate_asteroid_thresholds()
        )

        # Create asteroid formations based on the noise map and thresholds
        self._create_asteroid_formations(
            noise_map, asteroid_threshold, rare_threshold, anomaly_threshold
        )

    def _generate_perlin_noise_map(self) -> np.ndarray:
        """
        Generate a noise map using multiple Perlin noise generators at different scales.

        Returns:
            np.ndarray: The generated noise map
        """
        # Create Perlin noise generators for different scales
        seed1 = random.randint(1, 1000)
        seed2 = random.randint(1, 1000)
        seed3 = random.randint(1, 1000)

        # Adjust octaves based on complexity
        octave1 = int(3 + self.pattern_complexity * 2)
        octave2 = int(6 + self.pattern_complexity * 3)
        octave3 = int(10 + self.pattern_complexity * 5)

        noise1 = PerlinNoise(octaves=octave1, seed=seed1)
        noise2 = PerlinNoise(octaves=octave2, seed=seed2)
        noise3 = PerlinNoise(octaves=octave3, seed=seed3)

        # Generate noise map
        noise_map = np.zeros((self.height, self.width))

        # Use vectorized operations where possible for better performance
        for y in range(self.height):
            for x in range(self.width):
                # Combine noise at different scales
                nx, ny = x / self.width, y / self.height

                # Add turbulence to coordinates
                if self.turbulence > 0:
                    nx += math.sin(ny * 10) * self.turbulence * 0.05
                    ny += math.cos(nx * 10) * self.turbulence * 0.05

                # Combine noise layers with different weights
                noise_val = (
                    noise1([nx, ny]) * 0.5
                    + noise2([nx, ny]) * 0.3
                    + noise3([nx, ny]) * 0.2
                )
                noise_map[y, x] = noise_val

        return noise_map

    def _normalize_noise_map(self, noise_map: np.ndarray) -> np.ndarray:
        """
        Normalize the noise map to a 0-1 range.

        Args:
            noise_map: The noise map to normalize

        Returns:
            np.ndarray: The normalized noise map
        """
        # Normalize noise map to 0-1 range
        return (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))

    def _calculate_asteroid_thresholds(self) -> tuple:
        """
        Calculate thresholds for different asteroid types based on field density.

        Returns:
            tuple: (asteroid_threshold, rare_threshold, anomaly_threshold)
        """
        # Adjust threshold based on desired density
        asteroid_threshold = 0.7 - self.field_density * 0.3
        rare_threshold = asteroid_threshold + ((1.0 - asteroid_threshold) * 0.6)
        anomaly_threshold = rare_threshold + ((1.0 - rare_threshold) * 0.7)

        return asteroid_threshold, rare_threshold, anomaly_threshold

    def _create_asteroid_formations(
        self,
        noise_map: np.ndarray,
        asteroid_threshold: float,
        rare_threshold: float,
        anomaly_threshold: float,
    ) -> None:
        """
        Create asteroid formations based on the noise map and thresholds.

        Args:
            noise_map: The normalized noise map
            asteroid_threshold: Threshold for asteroid creation
            rare_threshold: Threshold for rare resources
            anomaly_threshold: Threshold for anomalies
        """
        # Create asteroid clusters
        self._populate_asteroid_grid(
            noise_map, asteroid_threshold, rare_threshold, anomaly_threshold
        )

    def _populate_asteroid_grid(
        self,
        noise_map: np.ndarray,
        asteroid_threshold: float,
        rare_threshold: float,
        anomaly_threshold: float,
    ) -> None:
        """
        Populate the grid with asteroids based on the noise map and thresholds.

        Args:
            noise_map: The normalized noise map
            asteroid_threshold: Threshold for asteroid creation
            rare_threshold: Threshold for rare resources
            anomaly_threshold: Threshold for anomalies
        """
        # Process each cell in the grid
        for y in range(self.height):
            for x in range(self.width):
                self._process_grid_cell(
                    noise_map[y, x],
                    x,
                    y,
                    asteroid_threshold,
                    rare_threshold,
                    anomaly_threshold,
                )

    def _process_grid_cell(
        self,
        noise_val: float,
        x: int,
        y: int,
        asteroid_threshold: float,
        rare_threshold: float,
        anomaly_threshold: float,
    ) -> None:
        """
        Process a single cell in the grid based on its noise value and thresholds.

        Args:
            noise_val: The noise value at this cell
            x: X coordinate
            y: Y coordinate
            asteroid_threshold: Threshold for asteroid creation
            rare_threshold: Threshold for rare resources
            anomaly_threshold: Threshold for anomalies
        """
        # Skip cells below the asteroid threshold
        if noise_val <= asteroid_threshold:
            return

        # Calculate value factor for mineral richness
        value_factor = self._calculate_value_factor(noise_val, asteroid_threshold)

        # Set the base asteroid value
        self.grid[y, x] = int(50 + 150 * value_factor)

        # Set energy level based on proximity to threshold
        self.energy_grid[y, x] = value_factor * 0.5

        # Process rare and anomalous minerals
        self._process_special_minerals(
            noise_val, x, y, rare_threshold, anomaly_threshold
        )

    def _calculate_value_factor(
        self, noise_val: float, asteroid_threshold: float
    ) -> float:
        """
        Calculate the value factor for mineral richness based on noise value.

        Args:
            noise_val: The noise value at this cell
            asteroid_threshold: Threshold for asteroid creation

        Returns:
            float: The calculated value factor
        """
        # Higher noise values = higher mineral value
        return (noise_val - asteroid_threshold) / (1 - asteroid_threshold)

    def _process_special_minerals(
        self,
        noise_val: float,
        x: int,
        y: int,
        rare_threshold: float,
        anomaly_threshold: float,
    ) -> None:
        """
        Process rare and anomalous minerals based on noise value and thresholds.

        Args:
            noise_val: The noise value at this cell
            x: X coordinate
            y: Y coordinate
            rare_threshold: Threshold for rare resources
            anomaly_threshold: Threshold for anomalies
        """
        # Chance for rare minerals in high-value areas
        if noise_val > rare_threshold and random.random() < self.rare_chance:
            self._set_rare_mineral(x, y)
        # Chance for anomalous minerals in extremely high-value areas
        elif noise_val > anomaly_threshold and random.random() < self.anomaly_chance:
            self._set_anomalous_mineral(x, y)

    def _set_rare_mineral(self, x: int, y: int) -> None:
        """
        Set a cell as containing rare minerals.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.rare_grid[y, x] = 1  # Mark as rare
        # Rare asteroids worth more
        self.grid[y, x] = int(self.grid[y, x] * self.rare_bonus_multiplier)

    def _set_anomalous_mineral(self, x: int, y: int) -> None:
        """
        Set a cell as containing anomalous minerals.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.rare_grid[y, x] = 2  # Mark as anomaly
        # Anomalies are worth even more
        self.grid[y, x] = int(self.grid[y, x] * self.rare_bonus_multiplier * 2)

        self._life_patterns_handler()

    def _initialize_handler(self):
        self.grid.fill(0)
        self.rare_grid.fill(0)
        self.energy_grid.fill(0)

    def _life_patterns_handler(self):
        self.add_life_patterns()
        self.update_statistics()
        self.redraw_needed = True

    def add_life_patterns(self) -> None:
        """
        Add Game of Life patterns at random locations to create interesting formations.
        """
        # Get the predefined life patterns
        life_patterns = self._get_life_patterns()

        # Determine number of patterns to place based on field size
        num_patterns = self._calculate_pattern_count()

        # Place patterns in the field
        self._place_life_patterns(life_patterns, num_patterns)

    def _get_life_patterns(self) -> list:
        """
        Define and return a list of Game of Life patterns.

        Returns:
            list: List of patterns, where each pattern is a list of (x,y) coordinates
        """
        return [
            # R-pentomino (chaotic growth)
            [(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)],
            # Glider
            [(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)],
            # Lightweight spaceship
            [
                (0, 1),
                (0, 4),
                (1, 0),
                (2, 0),
                (3, 0),
                (3, 4),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3),
            ],
            # Gosper glider gun (creates ongoing gliders)
            [
                (0, 24),
                (1, 22),
                (1, 24),
                (2, 12),
                (2, 13),
                (2, 20),
                (2, 21),
                (2, 34),
                (2, 35),
                (3, 11),
                (3, 15),
                (3, 20),
                (3, 21),
                (3, 34),
                (3, 35),
                (4, 0),
                (4, 1),
                (4, 10),
                (4, 16),
                (4, 20),
                (4, 21),
                (5, 0),
                (5, 1),
                (5, 10),
                (5, 14),
                (5, 16),
                (5, 17),
                (5, 22),
                (5, 24),
                (6, 10),
                (6, 16),
                (6, 24),
                (7, 11),
                (7, 15),
                (8, 12),
                (8, 13),
            ],
            # Pulsar (high-period oscillator)
            [
                (2, 4),
                (2, 5),
                (2, 6),
                (2, 10),
                (2, 11),
                (2, 12),
                (4, 2),
                (4, 7),
                (4, 9),
                (4, 14),
                (5, 2),
                (5, 7),
                (5, 9),
                (5, 14),
                (6, 2),
                (6, 7),
                (6, 9),
                (6, 14),
                (7, 4),
                (7, 5),
                (7, 6),
                (7, 10),
                (7, 11),
                (7, 12),
                (9, 4),
                (9, 5),
                (9, 6),
                (9, 10),
                (9, 11),
                (9, 12),
                (10, 2),
                (10, 7),
                (10, 9),
                (10, 14),
                (11, 2),
                (11, 7),
                (11, 9),
                (11, 14),
                (12, 2),
                (12, 7),
                (12, 9),
                (12, 14),
                (14, 4),
                (14, 5),
                (14, 6),
                (14, 10),
                (14, 11),
                (14, 12),
            ],
            # Pufferfish (moves and leaves debris)
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 6),
                (0, 7),
                (0, 8),
                (1, 0),
                (1, 4),
                (1, 8),
                (2, 4),
                (3, 0),
                (3, 4),
                (3, 8),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 6),
                (4, 7),
                (4, 8),
            ],
        ]

    def _calculate_pattern_count(self) -> int:
        """
        Calculate the number of patterns to place based on field size.

        Returns:
            int: Number of patterns to place
        """
        # Number of patterns scales with field size
        num_patterns = int(math.sqrt(self.width * self.height) / 20)
        return max(3, min(10, num_patterns))

    def _place_life_patterns(self, life_patterns: list, num_patterns: int) -> None:
        """
        Place life patterns in the field.

        Args:
            life_patterns: List of patterns to choose from
            num_patterns: Number of patterns to place
        """
        for _ in range(num_patterns):
            try:
                # Select a pattern
                pattern = self._select_pattern(life_patterns)

                # Find a suitable location for the pattern
                placement_info = self._find_pattern_placement(pattern)
                if not placement_info:
                    continue  # Skip if no suitable placement found

                offset_x, offset_y = placement_info

                # Add the pattern to the grid
                self._add_pattern_to_grid(pattern, offset_x, offset_y)

            except Exception as e:
                # Log the error but continue with other patterns
                print(f"Error placing life pattern: {str(e)}")

    def _select_pattern(self, life_patterns: list) -> list:
        """
        Select a pattern based on pattern complexity.

        Args:
            life_patterns: List of patterns to choose from

        Returns:
            list: The selected pattern
        """
        # Choose a pattern with complexity proportional to pattern_complexity
        pattern_index = min(
            len(life_patterns) - 1,
            int(
                random.random() ** (1.0 - self.pattern_complexity) * len(life_patterns)
            ),
        )
        return life_patterns[pattern_index]

    def _find_pattern_placement(self, pattern: list) -> Optional[tuple]:
        """
        Find a suitable location for placing a pattern.

        Args:
            pattern: The pattern to place

        Returns:
            Optional[tuple]: (offset_x, offset_y) if a suitable location is found, None otherwise
        """
        margin = 20

        # Calculate pattern dimensions
        pattern_dimensions = self._calculate_pattern_dimensions(pattern)
        if not pattern_dimensions:
            return None

        # Unpack dimensions, only using max_pattern_size for placement calculations
        _, _, max_pattern_size = pattern_dimensions

        # Check if field is large enough for the pattern
        if not self._is_field_large_enough(max_pattern_size, margin):
            return None

        # Calculate valid placement ranges
        placement_ranges = self._calculate_placement_ranges(max_pattern_size, margin)
        if not placement_ranges:
            return None

        min_x, max_x, min_y, max_y = placement_ranges

        # Generate random position within valid ranges
        return self._generate_random_position(min_x, max_x, min_y, max_y)

    def _calculate_pattern_dimensions(self, pattern: list) -> Optional[tuple]:
        """
        Calculate the dimensions of a pattern.

        Args:
            pattern: The pattern to calculate dimensions for

        Returns:
            Optional[tuple]: (max_width, max_height, max_size) or None if pattern is empty
        """
        if not pattern:
            return None

        # Calculate pattern dimensions with padding
        max_width = max(dx for dx, dy in pattern) + 5
        max_height = max(dy for dx, dy in pattern) + 5
        max_size = max(max_width, max_height)

        return max_width, max_height, max_size

    def _is_field_large_enough(self, max_pattern_size: int, margin: int) -> bool:
        """
        Check if the field is large enough to place the pattern with margins.

        Args:
            max_pattern_size: Maximum dimension of the pattern
            margin: Margin to keep from field edges

        Returns:
            bool: True if field is large enough, False otherwise
        """
        return self.width > (margin * 2 + max_pattern_size) and self.height > (
            margin * 2 + max_pattern_size
        )

    def _calculate_placement_ranges(
        self, max_pattern_size: int, margin: int
    ) -> Optional[tuple]:
        """
        Calculate valid ranges for pattern placement.

        Args:
            max_pattern_size: Maximum dimension of the pattern
            margin: Margin to keep from field edges

        Returns:
            Optional[tuple]: (min_x, max_x, min_y, max_y) or None if ranges are invalid
        """
        min_x = margin
        max_x = self.width - max_pattern_size - margin
        min_y = margin
        max_y = self.height - max_pattern_size - margin

        # Ensure valid ranges (prevent ValueError in randint)
        if min_x >= max_x or min_y >= max_y:
            return None

        return min_x, max_x, min_y, max_y

    def _generate_random_position(
        self, min_x: int, max_x: int, min_y: int, max_y: int
    ) -> tuple:
        """
        Generate a random position within the given ranges.

        Args:
            min_x: Minimum x coordinate
            max_x: Maximum x coordinate
            min_y: Minimum y coordinate
            max_y: Maximum y coordinate

        Returns:
            tuple: (offset_x, offset_y) random position
        """
        offset_x = random.randint(min_x, max_x)
        offset_y = random.randint(min_y, max_y)

        return offset_x, offset_y

    def _add_pattern_to_grid(self, pattern: list, offset_x: int, offset_y: int) -> None:
        """
        Add a pattern to the grid at the specified offset.

        Args:
            pattern: The pattern to add
            offset_x: X coordinate offset
            offset_y: Y coordinate offset
        """
        for dx, dy in pattern:
            x, y = offset_x + dx, offset_y + dy
            if 0 <= x < self.width and 0 <= y < self.height:
                # Set asteroid value
                self.grid[y, x] = random.randint(80, 120)  # Good value range

                # Set energy level
                self.energy_grid[y, x] = random.uniform(0.6, 0.9)  # High energy

                # Process rare minerals
                self._process_pattern_rare_minerals(x, y)

    def _process_pattern_rare_minerals(self, x: int, y: int) -> None:
        """
        Process rare minerals for a cell in a pattern.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        # Small chance for a rare mineral (double chance in patterns)
        if random.random() < self.rare_chance * 2:
            self.rare_grid[y, x] = 1
            self.grid[y, x] = int(self.grid[y, x] * self.rare_bonus_multiplier)

    def update_statistics(self) -> None:
        """Update field statistics and history."""
        # Basic counts
        self.total_asteroids = np.sum(self.grid > 0)
        self.total_rare = np.sum(self.rare_grid == 1)
        self.total_anomaly = np.sum(self.rare_grid == 2)
        self.total_energy = np.sum(self.energy_grid)

        # Update history (keep a reasonable history size)
        max_history = 100

        self.asteroid_history.append(self.total_asteroids)
        if len(self.asteroid_history) > max_history:
            self.asteroid_history.pop(0)

        self.rare_history.append(self.total_rare)
        if len(self.rare_history) > max_history:
            self.rare_history.pop(0)

        self.energy_history.append(self.total_energy)
        if len(self.energy_history) > max_history:
            self.energy_history.pop(0)

        # Update race population history
        for race in self.races:
            race_id = race.race_id
            if race_id not in self.race_population_history:
                self.race_population_history[race_id] = []

            self.race_population_history[race_id].append(race.population)

            if len(self.race_population_history[race_id]) > max_history:
                self.race_population_history[race_id].pop(0)

    def get_view_bounds(self) -> Tuple[int, int, int, int]:
        """
        Get the bounds of the current viewport.

        Returns:
            tuple: (x1, y1, x2, y2) coordinates of the viewport
        """
        half_width = int(VIEW_WIDTH / (2 * self.zoom))
        half_height = int(VIEW_HEIGHT / (2 * self.zoom))

        view_x1 = max(0, self.camera_x - half_width)
        view_y1 = max(0, self.camera_y - half_height)
        view_x2 = min(self.width, self.camera_x + half_width)
        view_y2 = min(self.height, self.camera_y + half_height)

        # Ensure valid bounds
        view_x2 = max(view_x1 + 1, view_x2)
        view_y2 = max(view_y1 + 1, view_y2)

        return view_x1, view_y1, view_x2, view_y2

    def update(self) -> None:
        """
        Update the asteroid field state including:
        - Asteroid growth and decay using cellular automaton rules
        - Energy flow throughout the system
        - Symbiote race behaviors and interactions
        """
        perf_start = log_performance_start("field_update")

        # Update asteroid grid using cellular automaton
        self.update_asteroids()

        # Update race entities with advanced behavior
        self.update_entities()

        # Update statistics
        self.update_statistics()

        # Balance symbiotes and mining
        self.balance_ecosystem()

        # Mark for redraw
        self.redraw_needed = True

        log_performance_end("field_update", perf_start)

    def apply_cellular_automaton(
        self, grid, energy_grid=None, energy_boost=None, iterations=1
    ):
        """
        Apply cellular automaton rules to a binary grid with energy influence.

        This method serves as the main entry point for cellular automaton operations,
        delegating to specialized helper methods based on available dependencies and
        optimization opportunities.

        Args:
            grid (numpy.ndarray): Binary grid where 1 represents an asteroid and 0 represents empty space
            energy_grid (numpy.ndarray, optional): Grid of energy values between 0 and 1
            energy_boost (numpy.ndarray, optional): Grid of integer boost values for survival rules
            iterations (int, optional): Number of iterations to apply the cellular automaton rules

        Returns:
            numpy.ndarray: Updated binary grid after applying cellular automaton rules
        """
        # Performance tracking
        start_time = log_performance_start("apply_cellular_automaton")

        # Create a copy to avoid modifying the original
        result_grid = grid.copy()

        # Create cache key for this operation if caching is enabled
        if hasattr(self, "_ca_cache"):
            # Create a hash of the grid and parameters for caching
            grid_hash = hash(grid.tobytes())
            birth_set_hash = hash(frozenset(self.birth_set))
            survival_set_hash = hash(frozenset(self.survival_set))

            # Include energy parameters in cache key if provided
            energy_key = ""
            if energy_grid is not None and energy_boost is not None:
                energy_key = f"_energy_{hash(energy_grid.tobytes())}_{hash(energy_boost.tobytes())}"

            cache_key = f"ca_{grid_hash}_{birth_set_hash}_{survival_set_hash}_{iterations}_{self.apply_edge_wrapping}{energy_key}"

            # Check if we have this result cached
            if cache_key in self._ca_cache:
                log_performance_end("apply_cellular_automaton", start_time, "cached")
                return self._ca_cache[cache_key]
        else:
            # Initialize cache if not exists
            self._ca_cache = {}
            cache_key = None

        # Process each iteration
        for _ in range(iterations):
            result_grid = self._process_cellular_automaton_iteration(
                result_grid, energy_grid, energy_boost
            )

        # Cache the result if caching is enabled
        if cache_key is not None:
            self._ca_cache[cache_key] = result_grid

        log_performance_end("apply_cellular_automaton", start_time)
        return result_grid

    def _process_cellular_automaton_iteration(
        self, grid, energy_grid=None, energy_boost=None
    ):
        """
        Process a single iteration of the cellular automaton rules.

        Args:
            grid (numpy.ndarray): Binary grid where 1 represents an asteroid and 0 represents empty space
            energy_grid (numpy.ndarray, optional): Grid of energy values between 0 and 1
            energy_boost (numpy.ndarray, optional): Grid of integer boost values for survival rules

        Returns:
            numpy.ndarray: Updated binary grid after applying cellular automaton rules
        """
        # Use the optimized scipy implementation if available
        if SCIPY_AVAILABLE:
            return self._apply_cellular_automaton_scipy(grid, energy_grid, energy_boost)
        else:
            # Fall back to manual implementation
            return self._apply_cellular_automaton_manual(
                grid, energy_grid, energy_boost
            )

    def _apply_cellular_automaton_scipy(
        self, grid, energy_grid=None, energy_boost=None
    ):
        """
        Apply cellular automaton rules using scipy for efficiency.

        Args:
            grid (numpy.ndarray): Binary grid where 1 represents an asteroid and 0 represents empty space
            energy_grid (numpy.ndarray, optional): Grid of energy values between 0 and 1
            energy_boost (numpy.ndarray, optional): Grid of integer boost values for survival rules

        Returns:
            numpy.ndarray: Updated binary grid after applying cellular automaton rules
        """
        # Check if scipy is available
        if not SCIPY_AVAILABLE:
            return self._apply_cellular_automaton_manual(
                grid, energy_grid, energy_boost
            )

        # Create a copy to avoid modifying the original
        new_grid = np.zeros_like(grid)

        # Calculate neighbor counts using convolution
        neighbor_counts = self._calculate_neighbor_counts_scipy(grid)

        # Apply the appropriate rules based on whether energy is used
        if energy_grid is not None and energy_boost is not None:
            self._apply_energy_based_rules(
                grid, neighbor_counts, new_grid, energy_boost
            )
        else:
            self._rule_handler(grid, neighbor_counts, new_grid)

        return new_grid

    def _calculate_neighbor_counts_scipy(self, grid):
        """
        Calculate neighbor counts using scipy's convolution function.

        Args:
            grid (numpy.ndarray): Binary grid where 1 represents an asteroid and 0 represents empty space

        Returns:
            numpy.ndarray: Grid of neighbor counts for each cell
        """
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        return signal.convolve2d(
            grid,
            kernel,
            mode="same",
            boundary="wrap" if self.apply_edge_wrapping else "fill",
        )

    def _apply_energy_based_rules(self, grid, neighbor_counts, new_grid, energy_boost):
        """
        Apply cellular automaton rules with energy boost considerations.

        Args:
            grid (numpy.ndarray): Binary grid where 1 represents an asteroid and 0 represents empty space
            neighbor_counts (numpy.ndarray): Grid of neighbor counts for each cell
            new_grid (numpy.ndarray): Grid to store the results
            energy_boost (numpy.ndarray): Grid of integer boost values for survival rules
        """
        for y in range(self.height):
            for x in range(self.width):
                # Get adjusted survival set for this cell based on energy
                cell_survival_set = self._get_energy_adjusted_survival_set(
                    energy_boost[y, x]
                )

                # Apply rules to determine if cell lives or dies
                if self._should_cell_live(
                    grid[y, x], neighbor_counts[y, x], cell_survival_set
                ):
                    new_grid[y, x] = 1

    def _get_energy_adjusted_survival_set(self, boost):
        """
        Get a survival set adjusted by energy boost.

        Args:
            boost (int): Energy boost value

        Returns:
            set: Adjusted survival set
        """
        cell_survival_set = self.survival_set.copy()

        # Add energy-boosted values to the survival set
        if boost > 0:
            for n in list(self.survival_set):
                cell_survival_set.add(n + boost)

        return cell_survival_set

    def _should_cell_live(self, cell_state, neighbor_count, survival_set):
        """
        Determine if a cell should be alive in the next generation.

        Args:
            cell_state (int): Current state of the cell (0 or 1)
            neighbor_count (int): Number of live neighbors
            survival_set (set): Set of neighbor counts that allow survival

        Returns:
            bool: True if the cell should be alive, False otherwise
        """
        # Cell is alive and has the right number of neighbors to survive
        if cell_state > 0 and neighbor_count in survival_set:
            return True

        # Cell is dead but has the right number of neighbors to be born
        return cell_state <= 0 and neighbor_count in self.birth_set

    def _rule_handler(self, grid, neighbor_counts, new_grid):
        # Simple rules without energy influence - fully vectorized approach
        alive_mask = grid > 0
        dead_mask = ~alive_mask

        # Create masks for birth and survival based on neighbor counts
        birth_mask = np.zeros_like(neighbor_counts, dtype=bool)
        survival_mask = np.zeros_like(neighbor_counts, dtype=bool)

        for n in self.birth_set:
            birth_mask |= neighbor_counts == n

        for n in self.survival_set:
            survival_mask |= neighbor_counts == n

        # Apply rules using vectorized operations
        new_grid[alive_mask & survival_mask] = 1  # Cells that survive
        new_grid[dead_mask & birth_mask] = 1  # Cells that are born

    def _apply_cellular_automaton_manual(
        self, grid, energy_grid=None, energy_boost=None
    ):
        """
        Apply cellular automaton rules using manual implementation (fallback when scipy is unavailable).

        Args:
            grid (numpy.ndarray): Binary grid where 1 represents an asteroid and 0 represents empty space
            energy_grid (numpy.ndarray, optional): Grid of energy values between 0 and 1
            energy_boost (numpy.ndarray, optional): Grid of integer boost values for survival rules

        Returns:
            numpy.ndarray: Updated binary grid after applying cellular automaton rules
        """
        # Create a copy to avoid modifying the original
        new_grid = np.zeros_like(grid)

        # Process each cell in the grid
        for y in range(self.height):
            for x in range(self.width):
                # Count neighbors and apply rules
                neighbors = self._count_neighbors_manual(grid, x, y)

                # Get adjusted survival set if energy is used
                if energy_grid is not None and energy_boost is not None:
                    cell_survival_set = self._get_energy_adjusted_survival_set(
                        energy_boost[y, x]
                    )
                else:
                    cell_survival_set = self.survival_set

                # Determine if cell should live
                if self._should_cell_live(grid[y, x], neighbors, cell_survival_set):
                    new_grid[y, x] = 1

        return new_grid

    def _count_neighbors_manual(self, grid, x, y):
        """
        Count the number of live neighbors for a cell using manual iteration.

        Args:
            grid (numpy.ndarray): Binary grid where 1 represents an asteroid and 0 represents empty space
            x (int): X coordinate of the cell
            y (int): Y coordinate of the cell

        Returns:
            int: Number of live neighbors
        """
        neighbors = 0

        # Check all 8 neighboring cells
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                # Skip the center cell
                if dx == 0 and dy == 0:
                    continue

                # Calculate neighbor coordinates
                nx, ny = x + dx, y + dy

                # Apply coordinate adjustments based on edge wrapping setting
                nx, ny = self._adjust_coordinates(nx, ny)

                # Skip if coordinates are out of bounds
                if nx is None or ny is None:
                    continue

                # Count live neighbor
                if grid[ny, nx] > 0:
                    neighbors += 1

        return neighbors

    def _adjust_coordinates(self, x, y):
        """
        Adjust coordinates based on edge wrapping settings.

        Args:
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            tuple: Adjusted (x, y) coordinates, or (None, None) if out of bounds
        """
        # Apply edge wrapping if enabled
        if self.apply_edge_wrapping:
            return x % self.width, y % self.height

        # Check if coordinates are within bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None, None

        return x, y

    def update_asteroids(self) -> None:
        """Update asteroid grid using cellular automaton rules and energy flow."""
        # Performance tracking
        start_time = log_performance_start("update_asteroids")

        # Check cache for existing results
        cache_key = self._get_update_cache_key()
        if self._check_update_cache(cache_key, start_time):
            return

        # Create new grids and prepare for update
        new_grid, new_rare_grid, new_energy_grid = self._initialize_new_grids()
        binary_grid, energy_grid_normalized, energy_boost = self._prepare_grid_data()

        # Apply cellular automaton and update grid values
        new_binary_grid = self._apply_automaton_with_energy(
            binary_grid, energy_grid_normalized, energy_boost
        )
        energy_neighborhood = self._calculate_energy_neighborhood(self.energy_grid)
        self._update_grid_values(
            binary_grid,
            new_binary_grid,
            new_grid,
            new_rare_grid,
            new_energy_grid,
            energy_neighborhood,
        )

        # Update main grids with new values
        self._update_main_grids(new_grid, new_rare_grid, new_energy_grid)
        self._add_energy_to_low_density_areas(binary_grid)

        # Cache results for future use
        self._cache_update_results(cache_key)

        log_performance_end("update_asteroids", start_time)

    def _get_update_cache_key(self) -> Optional[str]:
        """Generate a cache key for the current grid state.

        Returns:
            Optional[str]: Cache key string or None if caching is not initialized
        """
        if hasattr(self, "_update_cache"):
            grid_hash = hash(self.grid.tobytes())
            energy_hash = hash(self.energy_grid.tobytes())
            rare_hash = hash(self.rare_grid.tobytes())
            return f"update_{grid_hash}_{energy_hash}_{rare_hash}_{self.energy_decay}_{self.energy_spread}"
        else:
            # Initialize cache if not exists
            self._update_cache = {}
            return None

    def _check_update_cache(self, cache_key: Optional[str], start_time) -> bool:
        """Check if the current update is already cached and apply if found.

        Args:
            cache_key: The cache key to check
            start_time: Start time for performance logging

        Returns:
            bool: True if cache was used, False otherwise
        """
        if cache_key is not None and cache_key in self._update_cache:
            self.grid, self.rare_grid, self.energy_grid = self._update_cache[cache_key]
            log_performance_end("update_asteroids", start_time, "cached")
            return True
        return False

    def _initialize_new_grids(self):
        """Initialize new grid arrays for the update process.

        Returns:
            tuple: Tuple of (new_grid, new_rare_grid, new_energy_grid)
        """
        new_grid = np.zeros_like(self.grid)
        new_rare_grid = np.zeros_like(self.rare_grid)
        new_energy_grid = np.zeros_like(self.energy_grid)
        return new_grid, new_rare_grid, new_energy_grid

    def _prepare_grid_data(self):
        """Prepare grid data for cellular automaton processing.

        Returns:
            tuple: Tuple of (binary_grid, energy_grid_normalized, energy_boost)
        """
        binary_grid = (self.grid > 0).astype(np.int8)
        energy_grid_normalized = np.clip(self.energy_grid, 0, 1)
        energy_boost = np.minimum(2, (energy_grid_normalized * 3).astype(np.int8))
        return binary_grid, energy_grid_normalized, energy_boost

    def _apply_automaton_with_energy(
        self, binary_grid, energy_grid_normalized, energy_boost
    ):
        """Apply cellular automaton with energy-adjusted rules.

        Args:
            binary_grid: Binary representation of the asteroid grid
            energy_grid_normalized: Normalized energy grid
            energy_boost: Energy boost values for survival rules

        Returns:
            numpy.ndarray: Updated binary grid after applying cellular automaton
        """
        return self.apply_cellular_automaton(
            binary_grid, energy_grid=energy_grid_normalized, energy_boost=energy_boost
        )

    def _update_main_grids(self, new_grid, new_rare_grid, new_energy_grid):
        """Update the main grid arrays with new values.

        Args:
            new_grid: New asteroid grid
            new_rare_grid: New rare minerals grid
            new_energy_grid: New energy grid
        """
        self.grid = new_grid
        self.rare_grid = new_rare_grid
        self.energy_grid = new_energy_grid

    def _cache_update_results(self, cache_key: Optional[str]):
        """Cache the update results for future use.

        Args:
            cache_key: Cache key to use for storing results
        """
        if cache_key is not None:
            self._update_cache[cache_key] = (
                self.grid.copy(),
                self.rare_grid.copy(),
                self.energy_grid.copy(),
            )

            # Limit cache size to prevent memory issues
            if len(self._update_cache) > 10:  # Keep only the 10 most recent updates
                oldest_key = next(iter(self._update_cache))
                del self._update_cache[oldest_key]

    def _calculate_energy_neighborhood(self, energy_grid):
        """Calculate the energy neighborhood for each cell using optimized methods.

        Args:
            energy_grid: Grid of energy values

        Returns:
            numpy.ndarray: Grid of summed energy values in the neighborhood of each cell
        """
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Use optimized convolution if available
        if SCIPY_AVAILABLE:
            return self._calculate_energy_neighborhood_scipy(energy_grid, kernel)
        else:
            # Manual fallback for energy neighborhood calculation
            return self._calculate_energy_neighborhood_manual(energy_grid)

    def _calculate_energy_neighborhood_scipy(self, energy_grid, kernel):
        """Calculate energy neighborhood using scipy's convolution function.

        Args:
            energy_grid: Grid of energy values
            kernel: Convolution kernel for neighborhood calculation

        Returns:
            numpy.ndarray: Grid of summed energy values in the neighborhood of each cell
        """
        boundary_mode = "wrap" if self.apply_edge_wrapping else "fill"
        return signal.convolve2d(
            energy_grid,
            kernel,
            mode="same",
            boundary=boundary_mode,
        )

    def _calculate_energy_neighborhood_manual(self, energy_grid):
        """Calculate energy neighborhood using manual iteration (fallback method).

        Args:
            energy_grid: Grid of energy values

        Returns:
            numpy.ndarray: Grid of summed energy values in the neighborhood of each cell
        """
        energy_neighborhood = np.zeros_like(energy_grid)

        for y in range(self.height):
            for x in range(self.width):
                energy_sum = self._calculate_cell_energy_sum(energy_grid, x, y)
                energy_neighborhood[y, x] = energy_sum

        return energy_neighborhood

    def _calculate_cell_energy_sum(self, energy_grid, x, y):
        """Calculate the sum of energy in the neighborhood of a single cell.

        Args:
            energy_grid: Grid of energy values
            x: X coordinate of the cell
            y: Y coordinate of the cell

        Returns:
            float: Sum of energy in the neighborhood
        """
        energy_sum = 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                # Skip the center cell
                if dx == 0 and dy == 0:
                    continue

                # Get neighbor coordinates with appropriate boundary handling
                nx, ny = x + dx, y + dy
                nx, ny = self._get_neighbor_coordinates(nx, ny)

                # Add energy if coordinates are valid
                if nx is not None and ny is not None:
                    energy_sum += energy_grid[ny, nx]

        return energy_sum

    def _get_neighbor_coordinates(self, nx, ny):
        """Get valid neighbor coordinates based on edge wrapping settings.

        Args:
            nx: Neighbor X coordinate
            ny: Neighbor Y coordinate

        Returns:
            tuple: Valid (nx, ny) coordinates or (None, None) if invalid
        """
        if self.apply_edge_wrapping:
            # Wrap around edges
            return nx % self.width, ny % self.height
        elif 0 <= nx < self.width and 0 <= ny < self.height:
            # Within bounds
            return nx, ny
        else:
            # Out of bounds
            return None, None

    def _update_grid_values(
        self,
        binary_grid,
        new_binary_grid,
        new_grid,
        new_rare_grid,
        new_energy_grid,
        energy_neighborhood,
    ):
        """Update grid values based on cellular automaton results and energy distribution."""
        # Try to use vectorized operations where possible
        try:
            self._update_grid_vectorized(
                binary_grid,
                new_binary_grid,
                new_grid,
                new_rare_grid,
                new_energy_grid,
                energy_neighborhood,
            )
        except Exception as e:
            # Fall back to non-vectorized approach if vectorization fails
            log_exception(e)
            logging.warning(f"Falling back to non-vectorized grid update: {str(e)}")
            self._update_grid_non_vectorized(
                binary_grid,
                new_binary_grid,
                new_grid,
                new_rare_grid,
                new_energy_grid,
                energy_neighborhood,
            )

    def _update_grid_vectorized(
        self,
        binary_grid,
        new_binary_grid,
        new_grid,
        new_rare_grid,
        new_energy_grid,
        energy_neighborhood,
    ):
        """Update grid values using vectorized operations for better performance.

        Args:
            binary_grid: Binary representation of the current grid
            new_binary_grid: Binary representation of the next generation grid
            new_grid: Grid to store the updated asteroid values
            new_rare_grid: Grid to store the updated rare mineral status
            new_energy_grid: Grid to store the updated energy values
            energy_neighborhood: Grid of energy values in the neighborhood of each cell
        """
        # Calculate new energy levels (vectorized)
        new_energy = self._calculate_new_energy_levels(energy_neighborhood)

        # Create masks for different cell states
        old_alive, new_alive = self._create_cell_state_masks(
            binary_grid, new_binary_grid
        )
        survival_mask, death_mask, birth_mask = self._create_transition_masks(
            old_alive, new_alive
        )

        # Handle surviving cells (keep values and rare status)
        self._process_surviving_cells(survival_mask, new_grid, new_rare_grid)

        # Handle dying cells (add energy)
        new_energy[death_mask] += 0.2

        # Process birth cells (requires random generation)
        self._process_birth_cells(
            birth_mask, new_grid, new_rare_grid, energy_neighborhood
        )

        # Process regeneration in dead cells
        self._process_regeneration_cells(old_alive, new_alive, new_grid, new_rare_grid)

        # Cap energy at 1.0 (vectorized)
        new_energy_grid[:] = np.minimum(1.0, new_energy)

    def _calculate_new_energy_levels(self, energy_neighborhood):
        """Calculate new energy levels based on current energy and neighborhood.

        Args:
            energy_neighborhood: Grid of energy values in the neighborhood of each cell

        Returns:
            numpy.ndarray: New energy levels before capping
        """
        new_energy = self.energy_grid * (1.0 - self.energy_decay)
        new_energy += energy_neighborhood * self.energy_spread / 8.0
        return new_energy

    def _create_cell_state_masks(self, binary_grid, new_binary_grid):
        """Create masks for cells that are alive in current and next generation.

        Args:
            binary_grid: Binary representation of the current grid
            new_binary_grid: Binary representation of the next generation grid

        Returns:
            tuple: (old_alive, new_alive) boolean masks
        """
        old_alive = binary_grid > 0
        new_alive = new_binary_grid > 0
        return old_alive, new_alive

    def _create_transition_masks(self, old_alive, new_alive):
        """Create masks for different cell state transitions.

        Args:
            old_alive: Boolean mask of cells that are alive in current generation
            new_alive: Boolean mask of cells that are alive in next generation

        Returns:
            tuple: (survival_mask, death_mask, birth_mask) boolean masks
        """
        # Survival mask: cells that were alive and remain alive
        survival_mask = old_alive & new_alive

        # Death mask: cells that were alive but died
        death_mask = old_alive & ~new_alive

        # Birth mask: cells that were dead but became alive
        birth_mask = ~old_alive & new_alive

        return survival_mask, death_mask, birth_mask

    def _process_surviving_cells(self, survival_mask, new_grid, new_rare_grid):
        """Process cells that survive from one generation to the next.

        Args:
            survival_mask: Boolean mask of cells that survive
            new_grid: Grid to store the updated asteroid values
            new_rare_grid: Grid to store the updated rare mineral status
        """
        new_grid[survival_mask] = self.grid[survival_mask]
        new_rare_grid[survival_mask] = self.rare_grid[survival_mask]

    def _process_birth_cells(
        self, birth_mask, new_grid, new_rare_grid, energy_neighborhood
    ):
        """Process cells where new asteroids are born.

        Args:
            birth_mask: Boolean mask of cells where new asteroids are born
            new_grid: Grid to store the updated asteroid values
            new_rare_grid: Grid to store the updated rare mineral status
            energy_neighborhood: Grid of energy values in the neighborhood of each cell
        """
        birth_cells = np.argwhere(birth_mask)

        for y, x in birth_cells:
            # New asteroid born - calculate value based on energy
            new_grid[y, x] = int(50 + energy_neighborhood[y, x] * 100)

            # Small chance for rare asteroid in births
            if random.random() < self.rare_chance:
                new_rare_grid[y, x] = 1
                new_grid[y, x] = int(new_grid[y, x] * self.rare_bonus_multiplier)

    def _process_regeneration_cells(
        self, old_alive, new_alive, new_grid, new_rare_grid
    ):
        """Process cells that may regenerate (dead cells that stay dead).

        Args:
            old_alive: Boolean mask of cells that are alive in current generation
            new_alive: Boolean mask of cells that are alive in next generation
            new_grid: Grid to store the updated asteroid values
            new_rare_grid: Grid to store the updated rare mineral status
        """
        # Create mask for cells that are dead and stay dead
        regen_mask = ~old_alive & ~new_alive
        regen_cells = np.argwhere(regen_mask)

        for y, x in regen_cells:
            local_energy = self.energy_grid[y, x]
            if random.random() < self.regen_rate * local_energy:
                new_grid[y, x] = int(30 + random.random() * 70)
                if random.random() < self.rare_chance:
                    new_rare_grid[y, x] = 1
                    new_grid[y, x] = int(new_grid[y, x] * self.rare_bonus_multiplier)

    def _update_grid_non_vectorized(
        self,
        binary_grid,
        new_binary_grid,
        new_grid,
        new_rare_grid,
        new_energy_grid,
        energy_neighborhood,
    ):
        """Update grid values using non-vectorized operations (fallback method).

        Args:
            binary_grid: Binary representation of the current grid
            new_binary_grid: Binary representation of the next generation grid
            new_grid: Grid to store the updated asteroid values
            new_rare_grid: Grid to store the updated rare mineral status
            new_energy_grid: Grid to store the updated energy values
            energy_neighborhood: Grid of energy values in the neighborhood of each cell
        """
        # Process each cell to update values and energy
        for y in range(self.height):
            for x in range(self.width):
                # Calculate new energy for this cell
                new_energy = self._calculate_new_cell_energy(y, x, energy_neighborhood)

                # Process cell based on its state transition
                self._process_cell_state_transition(
                    y,
                    x,
                    binary_grid,
                    new_binary_grid,
                    new_grid,
                    new_rare_grid,
                    new_energy,
                )

                # Store new energy level (capped at 1.0)
                new_energy_grid[y, x] = min(1.0, new_energy)

    def _calculate_new_cell_energy(self, y, x, energy_neighborhood):
        """Calculate the new energy level for a cell.

        Args:
            y, x: Cell coordinates
            energy_neighborhood: Grid of energy values in the neighborhood of each cell

        Returns:
            float: New energy level for the cell (before capping)
        """
        local_energy = self.energy_grid[y, x]

        # New energy starts with decayed version of current energy
        new_energy = local_energy * (1.0 - self.energy_decay)
        # Add energy from neighbors
        new_energy += energy_neighborhood[y, x] * self.energy_spread / 8.0

        return new_energy

    def _process_cell_state_transition(
        self, y, x, binary_grid, new_binary_grid, new_grid, new_rare_grid, new_energy
    ):
        """Process a cell based on its state transition between generations.

        Args:
            y, x: Cell coordinates
            binary_grid: Binary representation of the current grid
            new_binary_grid: Binary representation of the next generation grid
            new_grid: Grid to store the updated asteroid values
            new_rare_grid: Grid to store the updated rare mineral status
            new_energy: Reference to the new energy value for this cell

        Returns:
            float: Updated new energy value
        """
        old_has_asteroid = binary_grid[y, x] > 0
        new_has_asteroid = new_binary_grid[y, x] > 0

        # Handle different state transitions
        if old_has_asteroid and new_has_asteroid:
            self._handle_surviving_cell(y, x, new_grid, new_rare_grid)

        elif old_has_asteroid:
            # Asteroid dies - add energy
            new_energy += 0.2

        elif new_has_asteroid:
            self._handle_birth_cell(y, x, new_grid, new_rare_grid, new_energy)

        else:
            self._handle_regeneration_cell(y, x, new_grid, new_rare_grid)

        return new_energy

    def _handle_surviving_cell(self, y, x, new_grid, new_rare_grid):
        """Handle a cell where an asteroid survives from one generation to the next.

        Args:
            y, x: Cell coordinates
            new_grid: Grid to store the updated asteroid values
            new_rare_grid: Grid to store the updated rare mineral status
        """
        # Asteroid survives - keep its value and rare status
        new_grid[y, x] = self.grid[y, x]
        new_rare_grid[y, x] = self.rare_grid[y, x]

    def _handle_birth_cell(self, y, x, new_grid, new_rare_grid, energy_value):
        """Handle a cell where a new asteroid is born.

        Args:
            y, x: Cell coordinates
            new_grid: Grid to store the updated asteroid values
            new_rare_grid: Grid to store the updated rare mineral status
            energy_value: Energy value to use for calculating asteroid value
        """
        # New asteroid born - calculate value based on energy
        new_grid[y, x] = int(50 + energy_value * 100)

        # Small chance for rare asteroid in births
        self._apply_rare_chance(y, x, new_grid, new_rare_grid)

    def _handle_regeneration_cell(self, y, x, new_grid, new_rare_grid):
        """Handle potential regeneration in a cell that is dead and stays dead.

        Args:
            y, x: Cell coordinates
            new_grid: Grid to store the updated asteroid values
            new_rare_grid: Grid to store the updated rare mineral status
        """
        local_energy = self.energy_grid[y, x]

        # Check if regeneration occurs based on energy level
        if random.random() < self.regen_rate * local_energy:
            # Random regeneration - new asteroid value
            new_grid[y, x] = int(30 + random.random() * 70)

            # Check for rare asteroid
            self._apply_rare_chance(y, x, new_grid, new_rare_grid)

    def _apply_rare_chance(self, y, x, grid, rare_grid):
        """Apply chance for rare mineral and adjust value accordingly.

        Args:
            y, x: Cell coordinates
            grid: Grid to store the asteroid values
            rare_grid: Grid to store the rare mineral status
        """
        if random.random() < self.rare_chance:
            rare_grid[y, x] = 1
            grid[y, x] = int(grid[y, x] * self.rare_bonus_multiplier)

    def _add_energy_to_low_density_areas(self, binary_grid):
        """Add energy to low density areas to encourage new growth.

        Args:
            binary_grid: Binary representation of the asteroid grid
        """
        if SCIPY_AVAILABLE:
            self._add_energy_to_low_density_areas_vectorized(binary_grid)
        else:
            self._add_energy_to_low_density_areas_manual(binary_grid)

    def _add_energy_to_low_density_areas_vectorized(self, binary_grid):
        """Add energy to low density areas using vectorized operations.

        Args:
            binary_grid: Binary representation of the asteroid grid
        """
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Use vectorized operations with scipy
        boundary_mode = "wrap" if self.apply_edge_wrapping else "fill"
        neighbor_counts = signal.convolve2d(
            binary_grid,
            kernel,
            mode="same",
            boundary=boundary_mode,
        )

        # Identify and boost low density areas
        low_density_mask = neighbor_counts < 2
        self.energy_grid[low_density_mask] += 0.05

    def _add_energy_to_low_density_areas_manual(self, binary_grid):
        """Add energy to low density areas using manual iteration.

        Args:
            binary_grid: Binary representation of the asteroid grid
        """
        # Manual calculation of low density areas
        for y, x in itertools.product(range(self.height), range(self.width)):
            # Count neighbors with asteroids
            neighbor_count = self._count_asteroid_neighbors(binary_grid, x, y)

            # Add energy to cells with few asteroid neighbors
            if neighbor_count < 2:
                self.energy_grid[y, x] += 0.05

    def _count_asteroid_neighbors(self, binary_grid, x, y):
        """Count the number of neighboring cells that contain asteroids.

        Args:
            binary_grid: Binary representation of the asteroid grid
            x: X coordinate of the cell
            y: Y coordinate of the cell

        Returns:
            int: Number of neighboring cells with asteroids
        """
        neighbor_count = 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                # Skip the center cell
                if dx == 0 and dy == 0:
                    continue

                # Get valid neighbor coordinates
                nx, ny = self._get_valid_neighbor_coordinates(x + dx, y + dy)

                # Count if neighbor has asteroid
                if nx is not None and ny is not None and binary_grid[ny, nx] > 0:
                    neighbor_count += 1

        return neighbor_count

    def _get_valid_neighbor_coordinates(self, nx, ny):
        """Get valid neighbor coordinates based on edge wrapping settings.

        Args:
            nx: Raw neighbor X coordinate
            ny: Raw neighbor Y coordinate

        Returns:
            tuple: Valid (nx, ny) coordinates or (None, None) if invalid
        """
        if self.apply_edge_wrapping:
            return nx % self.width, ny % self.height

        if 0 <= nx < self.width and 0 <= ny < self.height:
            return nx, ny

        return None, None

    def update_entities(self) -> Dict[int, int]:
        """
        Update symbiote races with enhanced mathematical modeling.

        Returns:
            dict: Dictionary of race incomes
        """
        try:
            with LogContext("entity_update"):
                return self._update_entities_implementation()
        except Exception as e:
            logging.critical(f"Error in update_entities: {str(e)}")
            log_exception(e, critical=False)
            return {}

    def _update_entities_implementation(self) -> Dict[int, int]:
        """Implementation of entity update logic."""
        if not self.races:
            return {}

        # Initialize race income dictionary
        race_income = {race.race_id: 0 for race in self.races}

        # Initialize other income dictionaries
        symbiote_income = {}
        fleet_income = {}

        # Initialize entity lists for return value
        symbiote_entities = (
            self.symbiote_entities.copy() if hasattr(self, "symbiote_entities") else []
        )
        miner_entities = (
            self.miner_entities.copy() if hasattr(self, "miner_entities") else []
        )
        fleet_entities = (
            self.fleet_entities.copy() if hasattr(self, "fleet_entities") else []
        )

        # Reset fed status for all races
        for race in self.races:
            race.fed_this_turn = False

        # Create new entity grid
        new_entity_grid = np.zeros_like(self.entity_grid)

        # Process race colonies to update colony metrics
        self._process_race_colonies()

        # Process cellular automaton rules for each race
        self._process_race_automaton_rules(new_entity_grid)

        # Process mining and interactions with asteroids
        self._process_mining_interactions(race_income)

        # Process population dynamics and evolution
        self._process_population_dynamics(race_income, new_entity_grid)

        # Update entity algorithms and entities
        self._update_entity_algorithms_and_entities()

        # Store additional data for later use
        self._last_update_data = {
            "symbiote_income": symbiote_income,
            "fleet_income": fleet_income,
            "symbiote_entities": symbiote_entities,
            "miner_entities": miner_entities,
            "fleet_entities": fleet_entities,
        }

        # Return race income dictionary
        return race_income

    def _process_race_colonies(self):
        """
        Process race colonies using scipy.ndimage for spatial analysis.
        Identifies connected regions (colonies) and updates race colony metrics.
        """
        # Process each race's colonies separately
        for race in self.races:
            self._analyze_race_colonies(race)

    def _analyze_race_colonies(self, race):
        """
        Analyze colonies for a specific race using spatial analysis.

        Args:
            race: The race to analyze colonies for
        """
        # Create a binary mask for this race's entities
        race_mask = self._create_race_mask(race)

        # Find connected regions (colonies)
        labeled_regions, num_regions = self._identify_colonies(race_mask)

        # Update race colony data if colonies exist
        if num_regions > 0:
            self._update_colony_metrics(race, race_mask, labeled_regions, num_regions)

    def _create_race_mask(self, race):
        """
        Create a binary mask for a race's entities.

        Args:
            race: The race to create a mask for

        Returns:
            numpy.ndarray: Boolean mask where race entities are located
        """
        return self.entity_grid == race.race_id

    def _identify_colonies(self, race_mask):
        """
        Identify connected regions (colonies) in the race mask.

        Args:
            race_mask: Boolean mask where race entities are located

        Returns:
            tuple: (labeled_regions, num_regions) - labeled array and count of regions
        """
        return ndimage.label(race_mask)

    def _update_colony_metrics(self, race, race_mask, labeled_regions, num_regions):
        """
        Update colony metrics for a race based on spatial analysis results.

        Args:
            race: The race to update metrics for
            race_mask: Boolean mask where race entities are located
            labeled_regions: Array with labeled regions
            num_regions: Number of distinct regions (colonies)
        """
        # Calculate colony sizes
        sizes = self._calculate_colony_sizes(race_mask, labeled_regions, num_regions)

        # Store colony metrics in race data
        race.colony_data = {
            "count": num_regions,
            "sizes": sizes,
            "mean_size": np.mean(sizes),
            "max_size": np.max(sizes),
            "total_population": np.sum(sizes),
        }

        # Update race population based on colony data
        race.population = int(race.colony_data["total_population"])

    def _calculate_colony_sizes(self, race_mask, labeled_regions, num_regions):
        """
        Calculate the size of each colony.

        Args:
            race_mask: Boolean mask where race entities are located
            labeled_regions: Array with labeled regions
            num_regions: Number of distinct regions (colonies)

        Returns:
            numpy.ndarray: Array of colony sizes
        """
        return ndimage.sum(race_mask, labeled_regions, range(1, num_regions + 1))

    def _process_race_automaton_rules(self, new_entity_grid):
        """
        Process cellular automaton rules for each race.

        This method applies Game of Life-like rules to determine which cells survive
        and which new cells are born based on race-specific parameters and hunger levels.

        Args:
            new_entity_grid: The new entity grid to populate with updated race positions
        """
        # Process symbiote cell-by-cell interactions using ndimage filters
        # These operations can calculate neighbor counts very efficiently
        for race in self.races:
            # Create a binary mask for this race's entities
            race_mask = self.entity_grid == race.race_id

            # Apply survival and birth rules
            self._apply_survival_rules(race, race_mask, new_entity_grid)
            self._apply_birth_rules(race, race_mask, new_entity_grid)

    def _apply_survival_rules(self, race, race_mask, new_entity_grid):
        """
        Apply survival rules for a race's existing cells.

        Args:
            race: The race to process
            race_mask: Binary mask of the race's current cells
            new_entity_grid: The new entity grid to update
        """
        # Calculate neighbor counts using convolution
        own_neighbors = self._calculate_neighbor_counts(race_mask)

        # Get adjusted survival rules based on hunger
        adjusted_survival_set = self._get_adjusted_survival_set(race)

        # Create survival mask based on rules
        survival_mask = self._create_survival_mask(
            own_neighbors, race_mask, adjusted_survival_set
        )

        # Update the entity grid with surviving cells
        self._update_grid_with_survivors(survival_mask, new_entity_grid, race.race_id)

    def _get_adjusted_survival_set(self, race):
        """
        Get the adjusted survival set based on race hunger.

        Args:
            race: The race to process

        Returns:
            set: Adjusted set of neighbor counts that allow survival
        """
        # Calculate hunger modifier
        hunger_modifier = int(race.hunger * 2)

        return race.survival_set.union({n + hunger_modifier for n in race.survival_set})

    def _create_survival_mask(self, own_neighbors, race_mask, survival_set):
        """
        Create a mask of cells that survive based on neighbor counts and rules.

        Args:
            own_neighbors: Array with neighbor counts for each cell
            race_mask: Binary mask of the race's current cells
            survival_set: Set of neighbor counts that allow survival

        Returns:
            numpy.ndarray: Boolean mask of cells that survive
        """
        survival_mask = np.zeros_like(race_mask, dtype=bool)

        for n in survival_set:
            survival_mask |= (own_neighbors == n) & race_mask

        return survival_mask

    def _update_grid_with_survivors(self, survival_mask, new_entity_grid, race_id):
        """
        Update the entity grid with cells that survive.

        Args:
            survival_mask: Boolean mask of cells that survive
            new_entity_grid: The new entity grid to update
            race_id: ID of the race to assign to surviving cells
        """
        new_entity_grid[survival_mask] = race_id

    def _apply_birth_rules(self, race, race_mask, new_entity_grid):
        """
        Apply birth rules for a race's new cells.

        Args:
            race: The race to process
            race_mask: Binary mask of the race's current cells
            new_entity_grid: The new entity grid to update
        """
        # Calculate neighbor counts using convolution
        own_neighbors = self._calculate_neighbor_counts(race_mask)

        # Get empty cells where new entities can be born
        empty_mask = self.entity_grid == 0

        # Get adjusted birth rules based on race state
        adjusted_birth_set = self._get_adjusted_birth_set(race)

        # Create birth mask based on neighbor counts and birth rules
        birth_mask = self._create_birth_mask(
            own_neighbors, empty_mask, adjusted_birth_set
        )

        # Apply influence-based probability modifiers if available
        birth_mask = self._apply_influence_probability(race, birth_mask)

        # Apply final birth probability based on hunger
        self._apply_final_birth_probability(race, birth_mask, new_entity_grid)

    def _calculate_neighbor_counts(self, race_mask):
        """
        Calculate the number of neighbors for each cell using convolution.

        Args:
            race_mask: Binary mask of the race's current cells

        Returns:
            numpy.ndarray: Array with neighbor counts for each cell
        """
        neighbors_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        return ndimage.convolve(
            race_mask.astype(np.int8), neighbors_kernel, mode="constant", cval=0
        )

    def _get_adjusted_birth_set(self, race):
        """
        Get the adjusted birth set based on race behavior and hunger.

        Args:
            race: The race to process

        Returns:
            set: Adjusted set of neighbor counts that trigger birth
        """
        # Start with the race's base birth set
        adjusted_birth_set = race.birth_set

        # Expand birth conditions when hungry or in expansion mode
        if race.current_behavior == "expanding" or race.hunger > 0.7:
            adjusted_birth_set = adjusted_birth_set.union(
                {birth - 1 for birth in race.birth_set}
            )

        return adjusted_birth_set

    def _create_birth_mask(self, own_neighbors, empty_mask, birth_set):
        """
        Create a mask of cells where new entities should be born.

        Args:
            own_neighbors: Array with neighbor counts for each cell
            empty_mask: Mask of empty cells
            birth_set: Set of neighbor counts that trigger birth

        Returns:
            numpy.ndarray: Boolean mask of cells where birth should occur
        """
        birth_mask = np.zeros_like(empty_mask, dtype=bool)

        for n in birth_set:
            birth_mask |= (own_neighbors == n) & empty_mask

        return birth_mask

    def _apply_influence_probability(self, race, birth_mask):
        """
        Apply probability modifiers based on influence grids if available.

        Args:
            race: The race to process
            birth_mask: Initial birth mask

        Returns:
            numpy.ndarray: Modified birth mask after applying influence probabilities
        """
        if race.race_id in self.influence_grids:
            # Get influence grid for this race
            influence = self.influence_grids[race.race_id]

            # Calculate birth probability based on influence and expansion drive
            birth_proba = influence * race.genome["expansion_drive"]

            # Apply probabilistic filter
            rng = np.random.default_rng(int(time.time()))
            random_mask = rng.random(birth_mask.shape) < birth_proba

            # Update birth mask
            birth_mask &= random_mask

        return birth_mask

    def _apply_final_birth_probability(self, race, birth_mask, new_entity_grid):
        """
        Apply final birth probability based on race hunger and update the grid.

        Args:
            race: The race to process
            birth_mask: Birth mask after influence processing
            new_entity_grid: The new entity grid to update
        """
        # Calculate birth probability based on hunger
        birth_probability = min(1.0, 0.8 + race.hunger * 0.4)

        # Apply random filter based on probability
        rng = np.random.default_rng(int(time.time()))
        birth_random = rng.random(birth_mask.shape) < birth_probability

        # Update the grid with new entities
        new_entity_grid[birth_mask & birth_random] = race.race_id

    def _process_mining_interactions(self, race_income):
        """
        Process mining interactions between races and asteroids.

        This method handles how races mine asteroids, calculates income,
        and updates the grid accordingly.

        Args:
            race_income: Dictionary mapping race IDs to their income values
        """
        # Process each race's mining and interactions with asteroids
        for y in range(self.height):
            for x in range(self.width):
                entity = self.entity_grid[y, x]
                if entity <= 0:  # No race here
                    continue

                # Find the race at this location
                race = next((r for r in self.races if r.race_id == entity), None)
                if not race or self.grid[y, x] <= 0:
                    continue

                # Process mining at this location
                self._process_single_mining_interaction(race, y, x, race_income)

    def _process_single_mining_interaction(self, race, y, x, race_income):
        """
        Process a single mining interaction at a specific location.

        Args:
            race: The race that is mining
            y: Y-coordinate of the mining location
            x: X-coordinate of the mining location
            race_income: Dictionary mapping race IDs to their income values
        """
        # Entity mines asteroid
        value = self.grid[y, x]
        rare_type = self.rare_grid[y, x]

        # Race's mining efficiency affects value
        income = int(value * race.mining_efficiency)

        # Process different mineral types
        if rare_type == 1:  # Rare
            income = int(income * self.rare_bonus_multiplier)
        elif rare_type == 2:  # Anomalous
            income = int(income * self.rare_bonus_multiplier * 2)

        # Add income to race
        race_income[race.race_id] += income

        # Mark race as fed this turn
        race.fed_this_turn = True

        # Remove asteroid after mining
        self.grid[y, x] = 0
        self.rare_grid[y, x] = 0

    def _process_population_dynamics(self, race_income, new_entity_grid):
        """
        Process population dynamics and evolution for each race.

        This method handles population tracking, income history, evolution points,
        and potential race evolution based on accumulated resources.

        Args:
            race_income: Dictionary mapping race IDs to their income values
            new_entity_grid: The updated entity grid with new race positions
        """
        # Apply complex dynamics based on statistics
        for race in self.races:
            # Get the population from the new entity grid
            new_mask = new_entity_grid == race.race_id
            population = np.sum(new_mask)

            # Skip processing if race is extinct
            if population <= 0:
                continue

            # Update race population and history
            self._update_race_population(race, population)

            # Update race income history and evolution
            self._update_race_income_and_evolution(race, race_income)

        # Process mineral availability for each race
        self._process_mineral_availability()

    def _update_race_population(self, race, population):
        """
        Update a race's population and population history.

        Args:
            race: The race to update
            population: The current population count
        """
        # Store current population
        race.population = population

        # Update population history with a fixed length
        race.population_history.append(population)
        if len(race.population_history) > 100:
            race.population_history.pop(0)

    def _update_race_income_and_evolution(self, race, race_income):
        """
        Update a race's income history and handle evolution if applicable.

        Args:
            race: The race to update
            race_income: Dictionary mapping race IDs to their income values
        """
        # Calculate income and update history
        income = race_income[race.race_id]
        race.last_income = income
        race.income_history.append(income)
        if len(race.income_history) > 100:
            race.income_history.pop(0)

        # Update evolution points based on income
        race.evolution_points += income // 10

        # Check if race should evolve
        if race.evolution_points >= race.evolution_threshold:
            self._handle_race_evolution(race)

    def _handle_race_evolution(self, race):
        """
        Handle the evolution of a race when it reaches the evolution threshold.

        Args:
            race: The race that is evolving
        """
        # Evolution attempt
        metrics = race.evolve()
        race.evolution_points -= race.evolution_threshold
        race.evolution_threshold = int(race.evolution_threshold * 1.5)

        # Log evolution
        logging.info(f"Race {race.race_id} evolved to stage {race.evolution_stage}")
        logging.info(
            f"  - Territory: {metrics['radius']} radius with density {metrics['density']:.2f}"
        )

        # Notify player
        if self.notifier:
            self.notifier.notify_event(
                "race",
                f"Race {race.race_id} has evolved to stage {race.evolution_stage}!",
                importance=2,
            )

    def _process_mineral_availability(self):
        """
        Process mineral availability for each race's territory.

        This method analyzes the distribution of minerals around each race's colonies
        and updates the race's mineral availability metrics.
        """
        for race in self.races:
            # Analyze mineral distribution around race colonies
            race_mask = self.entity_grid == race.race_id
            if np.sum(race_mask) == 0:
                continue  # Race is extinct

            # Calculate mineral availability in race territory
            minerals_available = self._calculate_race_mineral_availability(race_mask)

            # Update race's mineral availability
            race.mineral_availability = minerals_available

    def _calculate_race_mineral_availability(self, race_mask):
        """
        Calculate the mineral availability in a race's territory.

        Args:
            race_mask: Binary mask indicating the race's territory

        Returns:
            Dictionary with counts of different mineral types
        """
        # Initialize mineral counts
        minerals_available = {
            "common": 0,
            "rare": 0,
            "precious": 0,
            "anomaly": 0,
        }

        # Find all race entity locations
        entity_locations = np.nonzero(race_mask)
        # Check surrounding area for minerals
        search_radius = 3  # Look in a 3-cell radius

        for i in range(len(entity_locations[0])):
            y, x = entity_locations[0][i], entity_locations[1][i]
            self._check_minerals_in_radius(x, y, search_radius, minerals_available)

        return minerals_available

    def _check_minerals_in_radius(self, center_x, center_y, radius, minerals_available):
        """
        Check for minerals within a radius of a given center point.

        Args:
            center_x: X-coordinate of the center point
            center_y: Y-coordinate of the center point
            radius: Radius to check around the center point
            minerals_available: Dictionary to update with mineral counts
        """
        # Check surrounding area for minerals
        for dy, dx in itertools.product(
            range(-radius, radius + 1),
            range(-radius, radius + 1),
        ):
            nx, ny = center_x + dx, center_y + dy
            if (
                0 <= nx < self.width
                and 0 <= ny < self.height
                and self.grid[ny, nx] == 0
            ):
                rare_type = self.rare_grid[ny, nx]
                self._update_mineral_counts(rare_type, minerals_available)

    def _update_mineral_counts(self, rare_type, minerals_available):
        """
        Update mineral counts based on the rare type.

        Args:
            rare_type: The type of mineral (0=common, 1=rare, 2=anomalous, 3=precious)
            minerals_available: Dictionary to update with mineral counts
        """
        if rare_type == 1:  # Rare
            minerals_available["rare"] += 1
        elif rare_type == 2:  # Anomalous
            minerals_available["anomaly"] += 1
        elif rare_type == 3:  # Precious
            minerals_available["precious"] += 1
        else:  # Common
            minerals_available["common"] += 1

    def _update_entity_algorithms_and_entities(self):
        """
        Update all entity algorithms and entities.

        This method updates the evolution algorithm, symbiote algorithm,
        and all entity types (symbiotes, miners, fleets).
        """
        # Update evolution algorithm
        if self.evolution_algorithm:
            self.evolution_algorithm.update(self.races)

        # Update symbiote evolution algorithm
        if hasattr(self, "symbiote_algorithm") and self.symbiote_algorithm:
            self.symbiote_algorithm.update(self)

        # Update symbiote entities
        symbiote_entities = getattr(self, "symbiote_entities", [])
        for symbiote in symbiote_entities:
            symbiote.update()

        # Update miner entities
        miner_entities = getattr(self, "miner_entities", [])
        for miner in miner_entities:
            miner.update()

        # Update fleet entities
        fleet_entities = getattr(self, "fleet_entities", [])
        for fleet in fleet_entities:
            fleet.update()
