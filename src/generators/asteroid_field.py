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
from typing import Dict, List, Tuple, Set

# Third-party imports
import numpy as np
import pygame

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
from config import GRID_WIDTH, GRID_HEIGHT, VIEW_WIDTH, VIEW_HEIGHT
from utils.logging_setup import (
    log_exception,
    LogContext,
    log_performance_start,
    log_performance_end,
)
from entities.miner_entity import MinerEntity
from generators.procedural_generator import ProceduralGenerator
from generators.symbiote_evolution_generator import SymbioteEvolutionGenerator

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
import scipy

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

        self.rare_grid = np.random.random((self.height, self.width)) < self.rare_chance
        self.energy_grid = np.random.random((self.height, self.width))

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
            asteroid_grid, metadata = generator.generate_field()

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

        # Normalize noise map to 0-1 range
        noise_map = (noise_map - np.min(noise_map)) / (
            np.max(noise_map) - np.min(noise_map)
        )

        # Create asteroid formations based on noise threshold
        # Adjust threshold based on desired density
        asteroid_threshold = 0.7 - self.field_density * 0.3
        rare_threshold = asteroid_threshold + ((1.0 - asteroid_threshold) * 0.6)
        anomaly_threshold = rare_threshold + ((1.0 - rare_threshold) * 0.7)

        # Create asteroid clusters
        for y in range(self.height):
            for x in range(self.width):
                noise_val = noise_map[y, x]
                if noise_val > asteroid_threshold:
                    # Higher noise values = higher mineral value
                    value_factor = (noise_val - asteroid_threshold) / (
                        1 - asteroid_threshold
                    )
                    self.grid[y, x] = int(50 + 150 * value_factor)

                    # Energy level based on proximity to threshold
                    self.energy_grid[y, x] = value_factor * 0.5

                    # Chance for rare minerals in high-value areas
                    if (
                        noise_val > rare_threshold
                        and random.random() < self.rare_chance
                    ):
                        self.rare_grid[y, x] = 1
                        # Rare asteroids worth more
                        self.grid[y, x] = int(
                            self.grid[y, x] * self.rare_bonus_multiplier
                        )

                    # Chance for anomalous minerals in extremely high-value areas
                    elif (
                        noise_val > anomaly_threshold
                        and random.random() < self.anomaly_chance
                    ):
                        self.rare_grid[y, x] = 2  # Mark as anomaly
                        # Anomalies are worth even more
                        self.grid[y, x] = int(
                            self.grid[y, x] * self.rare_bonus_multiplier * 2
                        )

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
        # Advanced Game of Life patterns that create interesting growths
        life_patterns = [
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

        # Number of patterns scales with field size
        num_patterns = int(math.sqrt(self.width * self.height) / 20)
        num_patterns = max(3, min(10, num_patterns))

        for _ in range(num_patterns):
            try:
                # Choose a pattern with complexity proportional to pattern_complexity
                pattern_index = min(
                    len(life_patterns) - 1,
                    int(
                        random.random() ** (1.0 - self.pattern_complexity)
                        * len(life_patterns)
                    ),
                )
                pattern = life_patterns[pattern_index]

                # Find a suitable location (not too close to edges)
                margin = 20

                # Calculate maximum pattern dimensions
                max_pattern_width = max(dx for dx, dy in pattern) + 5 if pattern else 5
                max_pattern_height = max(dy for dx, dy in pattern) + 5 if pattern else 5
                max_pattern_size = max(max_pattern_width, max_pattern_height)

                # Ensure we have enough space to place the pattern
                if self.width <= (margin * 2 + max_pattern_size) or self.height <= (
                    margin * 2 + max_pattern_size
                ):
                    # Field is too small for this pattern, skip it
                    continue

                # Calculate valid ranges for pattern placement
                min_x = margin
                max_x = self.width - max_pattern_size - margin
                min_y = margin
                max_y = self.height - max_pattern_size - margin

                # Ensure valid ranges (prevent ValueError in randint)
                if min_x >= max_x or min_y >= max_y:
                    # Not enough space for this pattern, skip it
                    continue

                # Place pattern at random valid position
                offset_x = random.randint(min_x, max_x)
                offset_y = random.randint(min_y, max_y)

                # Add the pattern to the grid
                for dx, dy in pattern:
                    x, y = offset_x + dx, offset_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        self.grid[y, x] = random.randint(80, 120)  # Good value range
                        self.energy_grid[y, x] = random.uniform(0.6, 0.9)  # High energy

                        # Small chance for a rare mineral
                        if (
                            random.random() < self.rare_chance * 2
                        ):  # Double chance in patterns
                            self.rare_grid[y, x] = 1
                            self.grid[y, x] = int(
                                self.grid[y, x] * self.rare_bonus_multiplier
                            )
            except Exception as e:
                # Log the error but continue with other patterns
                print(f"Error placing life pattern: {str(e)}")

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
        # Check if scipy is available
        if not SCIPY_AVAILABLE:
            return self._apply_cellular_automaton_manual(
                grid, energy_grid, energy_boost
            )
        """
        Apply cellular automaton rules using scipy for efficiency.

        Args:
            grid (numpy.ndarray): Binary grid where 1 represents an asteroid and 0 represents empty space
            energy_grid (numpy.ndarray, optional): Grid of energy values between 0 and 1
            energy_boost (numpy.ndarray, optional): Grid of integer boost values for survival rules

        Returns:
            numpy.ndarray: Updated binary grid after applying cellular automaton rules
        """
        # Create a copy to avoid modifying the original
        new_grid = np.zeros_like(grid)

        # Count neighbors using convolution
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_counts = signal.convolve2d(
            grid,
            kernel,
            mode="same",
            boundary="wrap" if self.apply_edge_wrapping else "fill",
        )

        # Apply rules vectorized
        if energy_grid is not None and energy_boost is not None:
            # For each cell, calculate its adjusted survival set based on energy
            for y in range(self.height):
                for x in range(self.width):
                    # Get base survival set
                    cell_survival_set = self.survival_set.copy()

                    # Add energy-boosted values to the survival set
                    boost = energy_boost[y, x]
                    if boost > 0:
                        for n in list(self.survival_set):
                            cell_survival_set.add(n + boost)

                    # Apply rules
                    neighbors = neighbor_counts[y, x]
                    if (
                        grid[y, x] > 0
                        and neighbors in cell_survival_set
                        or grid[y, x] <= 0
                        and neighbors in self.birth_set
                    ):
                        new_grid[y, x] = 1
        else:
            self._rule_handler(grid, neighbor_counts, new_grid)
        return new_grid

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

        # Manual implementation without scipy
        for y in range(self.height):
            for x in range(self.width):
                # Count live neighbors
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue

                        nx, ny = x + dx, y + dy

                        # Handle edge wrapping
                        if self.apply_edge_wrapping:
                            nx = nx % self.width
                            ny = ny % self.height
                        elif nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                            continue

                        if grid[ny, nx] > 0:
                            neighbors += 1

                # Get cell survival set with energy boost if applicable
                cell_survival_set = self.survival_set.copy()
                if energy_grid is not None and energy_boost is not None:
                    boost = energy_boost[y, x]
                    if boost > 0:
                        for n in list(self.survival_set):
                            cell_survival_set.add(n + boost)

                # Apply rules
                if (
                    grid[y, x] > 0
                    and neighbors in cell_survival_set
                    or grid[y, x] <= 0
                    and neighbors in self.birth_set
                ):
                    new_grid[y, x] = 1
        return new_grid

    def update_asteroids(self) -> None:
        """Update asteroid grid using cellular automaton rules and energy flow."""
        # Performance tracking
        start_time = log_performance_start("update_asteroids")

        # Create cache key for this operation
        if hasattr(self, "_update_cache"):
            grid_hash = hash(self.grid.tobytes())
            energy_hash = hash(self.energy_grid.tobytes())
            rare_hash = hash(self.rare_grid.tobytes())
            cache_key = f"update_{grid_hash}_{energy_hash}_{rare_hash}_{self.energy_decay}_{self.energy_spread}"

            # Check if we have this result cached
            if cache_key in self._update_cache:
                self.grid, self.rare_grid, self.energy_grid = self._update_cache[
                    cache_key
                ]
                log_performance_end("update_asteroids", start_time, "cached")
                return
        else:
            # Initialize cache if not exists
            self._update_cache = {}
            cache_key = None

        # Create new grids to avoid affecting the update in progress
        new_grid = np.zeros_like(self.grid)
        new_rare_grid = np.zeros_like(self.rare_grid)
        new_energy_grid = np.zeros_like(self.energy_grid)

        # Apply cellular automaton to the asteroid grid
        binary_grid = (self.grid > 0).astype(np.int8)

        # Adjust survival set based on energy levels
        energy_grid_normalized = np.clip(self.energy_grid, 0, 1)
        energy_boost = np.minimum(2, (energy_grid_normalized * 3).astype(np.int8))

        # Apply cellular automaton with energy-adjusted rules
        new_binary_grid = self.apply_cellular_automaton(
            binary_grid, energy_grid=energy_grid_normalized, energy_boost=energy_boost
        )

        # Calculate energy neighborhood using optimized methods
        energy_neighborhood = self._calculate_energy_neighborhood(self.energy_grid)

        # Process grid updates using vectorized operations where possible
        self._update_grid_values(
            binary_grid,
            new_binary_grid,
            new_grid,
            new_rare_grid,
            new_energy_grid,
            energy_neighborhood,
        )

        # Update grids
        self.grid = new_grid
        self.rare_grid = new_rare_grid
        self.energy_grid = new_energy_grid

        # Add energy to low density areas (encouraging new growth)
        self._add_energy_to_low_density_areas(binary_grid)

        # Cache the result if caching is enabled
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

        log_performance_end("update_asteroids", start_time)

    def _calculate_energy_neighborhood(self, energy_grid):
        """Calculate the energy neighborhood for each cell using optimized methods."""
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Use optimized convolution if available
        if SCIPY_AVAILABLE:
            energy_neighborhood = signal.convolve2d(
                energy_grid,
                kernel,
                mode="same",
                boundary="wrap" if self.apply_edge_wrapping else "fill",
            )
        else:
            # Manual fallback for energy neighborhood calculation
            energy_neighborhood = np.zeros_like(energy_grid)
            for y in range(self.height):
                for x in range(self.width):
                    energy_sum = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if self.apply_edge_wrapping:
                                nx = nx % self.width
                                ny = ny % self.height
                            elif 0 <= nx < self.width and 0 <= ny < self.height:
                                energy_sum += energy_grid[ny, nx]
                    energy_neighborhood[y, x] = energy_sum

        return energy_neighborhood

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
            # Calculate new energy levels (vectorized)
            new_energy = self.energy_grid * (1.0 - self.energy_decay)
            new_energy += energy_neighborhood * self.energy_spread / 8.0

            # Create masks for different cell states
            old_alive = binary_grid > 0
            new_alive = new_binary_grid > 0

            # Survival mask: cells that were alive and remain alive
            survival_mask = old_alive & new_alive

            # Death mask: cells that were alive but died
            death_mask = old_alive & ~new_alive

            # Birth mask: cells that were dead but became alive
            birth_mask = ~old_alive & new_alive

            # Handle surviving cells (keep values and rare status)
            new_grid[survival_mask] = self.grid[survival_mask]
            new_rare_grid[survival_mask] = self.rare_grid[survival_mask]

            # Handle dying cells (add energy)
            new_energy[death_mask] += 0.2

            # Use birth_mask to pre-identify cells that need processing in the loop
            # This helps optimize the loop by reducing iterations
            birth_cells = np.argwhere(birth_mask)

            # For births and random regeneration, we need to use loops
            # as they involve random number generation
            # Use the pre-calculated birth_mask to optimize loop iterations
            for y, x in birth_cells:
                # New asteroid born - calculate value based on energy
                new_grid[y, x] = int(50 + energy_neighborhood[y, x] * 100)

                # Small chance for rare asteroid in births
                if random.random() < self.rare_chance:
                    new_rare_grid[y, x] = 1
                    new_grid[y, x] = int(new_grid[y, x] * self.rare_bonus_multiplier)

            # Handle random regeneration in dead cells that stay dead
            # Create mask for cells that are dead and stay dead
            regen_mask = ~old_alive & ~new_alive
            regen_cells = np.argwhere(regen_mask)

            for y, x in regen_cells:
                local_energy = self.energy_grid[y, x]
                if random.random() < self.regen_rate * local_energy:
                    new_grid[y, x] = int(30 + random.random() * 70)
                    if random.random() < self.rare_chance:
                        new_rare_grid[y, x] = 1
                        new_grid[y, x] = int(
                            new_grid[y, x] * self.rare_bonus_multiplier
                        )

            # Cap energy at 1.0 (vectorized)
            new_energy_grid[:] = np.minimum(1.0, new_energy)

        except Exception as e:
            # Fall back to non-vectorized approach if vectorization fails
            log_exception(e)
            logging.warning(f"Falling back to non-vectorized grid update: {str(e)}")

            # Process each cell to update values and energy
            for y in range(self.height):
                for x in range(self.width):
                    old_has_asteroid = binary_grid[y, x] > 0
                    new_has_asteroid = new_binary_grid[y, x] > 0
                    local_energy = self.energy_grid[y, x]

                    # New energy starts with decayed version of current energy
                    new_energy = local_energy * (1.0 - self.energy_decay)
                    new_energy += energy_neighborhood[y, x] * self.energy_spread / 8.0

                    if old_has_asteroid and new_has_asteroid:
                        # Asteroid survives - keep its value and rare status
                        new_grid[y, x] = self.grid[y, x]
                        new_rare_grid[y, x] = self.rare_grid[y, x]

                    elif old_has_asteroid:
                        # Asteroid dies - add energy
                        new_energy += 0.2

                    elif new_has_asteroid:
                        # New asteroid born - calculate value based on energy
                        new_grid[y, x] = int(50 + energy_neighborhood[y, x] * 100)

                        # Small chance for rare asteroid in births
                        if random.random() < self.rare_chance:
                            new_rare_grid[y, x] = 1
                            new_grid[y, x] = int(
                                new_grid[y, x] * self.rare_bonus_multiplier
                            )

                    # Handle cells that are dead and stay dead
                    elif (
                        not old_has_asteroid
                        and not new_has_asteroid
                        and random.random() < self.regen_rate * local_energy
                    ):
                        # Random regeneration based on energy level
                        new_grid[y, x] = int(30 + random.random() * 70)
                        if random.random() < self.rare_chance:
                            new_rare_grid[y, x] = 1
                            new_grid[y, x] = int(
                                new_grid[y, x] * self.rare_bonus_multiplier
                            )

                    # Store new energy level
                    new_energy_grid[y, x] = min(1.0, new_energy)  # Cap energy at 1.0

    def _add_energy_to_low_density_areas(self, binary_grid):
        """Add energy to low density areas to encourage new growth."""
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        if SCIPY_AVAILABLE:
            # Use vectorized operations with scipy
            neighbor_counts = signal.convolve2d(
                binary_grid,
                kernel,
                mode="same",
                boundary="wrap" if self.apply_edge_wrapping else "fill",
            )
            low_density_mask = neighbor_counts < 2
            self.energy_grid[low_density_mask] += 0.05
        else:
            # Manual calculation of low density areas
            for y, x in itertools.product(range(self.height), range(self.width)):
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if self.apply_edge_wrapping:
                            nx = nx % self.width
                            ny = ny % self.height
                        elif 0 <= nx < self.width and 0 <= ny < self.height:
                            if binary_grid[ny, nx] > 0:
                                neighbors += 1
                if neighbors < 2:
                    self.energy_grid[y, x] += 0.05

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

        race_income = {race.race_id: 0 for race in self.races}
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

        # Reset fed status
        for race in self.races:
            race.fed_this_turn = False

        # Create new entity grid
        new_entity_grid = np.zeros_like(self.entity_grid)

        # Using scipy.ndimage for spatial analysis of entity distributions
        # This creates labeled regions of connected components for each race
        for race in self.races:
            # Create a binary mask for this race's entities
            race_mask = self.entity_grid == race.race_id

            # Find connected regions (colonies) using ndimage
            labeled_regions, num_regions = ndimage.label(race_mask)

            if num_regions > 0:
                # Process colonies and track their metrics
                sizes = ndimage.sum(
                    race_mask, labeled_regions, range(1, num_regions + 1)
                )
                race.colony_data = {
                    "count": num_regions,
                    "sizes": sizes,
                    "mean_size": np.mean(sizes),
                    "max_size": np.max(sizes),
                    "total_population": np.sum(sizes),
                }

                # Set population based on colony data
                race.population = int(race.colony_data["total_population"])

        # Process symbiote cell-by-cell interactions using ndimage filters
        # These operations can calculate neighbor counts very efficiently
        for race in self.races:
            race_mask = self.entity_grid == race.race_id

            # Calculate neighbor counts using convolution
            neighbors_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            own_neighbors = ndimage.convolve(
                race_mask.astype(np.int8), neighbors_kernel, mode="constant", cval=0
            )

            # Apply Game of Life rules based on hunger level
            # Calculate survival mask - cells that survive
            hunger_modifier = int(race.hunger * 2)
            adjusted_survival_set = race.survival_set.union(
                {n + hunger_modifier for n in race.survival_set}
            )

            # Create a mask of cells that survive
            survival_mask = np.zeros_like(race_mask, dtype=bool)
            for n in adjusted_survival_set:
                survival_mask |= (own_neighbors == n) & race_mask

            # Add these cells to the new grid
            new_entity_grid[survival_mask] = race.race_id

            # Handle birth using similar approach
            # Calculate birth mask - empty cells that should become new entities
            empty_mask = self.entity_grid == 0

            # Adjust birth rules based on hunger and behavior
            adjusted_birth_set = race.birth_set
            if race.current_behavior == "expanding" or race.hunger > 0.7:
                # Add more birth conditions when hungry or in expansion mode
                adjusted_birth_set = adjusted_birth_set.union(
                    {birth - 1 for birth in race.birth_set}
                )

            # Create birth mask
            birth_mask = np.zeros_like(empty_mask, dtype=bool)
            for n in adjusted_birth_set:
                birth_mask |= (own_neighbors == n) & empty_mask

            # Apply probabilistic birth based on influence and expansion drive
            if race.race_id in self.influence_grids:
                influence = self.influence_grids[race.race_id]
                # Higher influence = higher chance of birth
                birth_proba = influence * race.genome["expansion_drive"]
                random_mask = np.random.random(birth_mask.shape) < birth_proba
                birth_mask &= random_mask

            # Add these cells to the new grid with probability based on hunger
            birth_probability = min(1.0, 0.8 + race.hunger * 0.4)
            birth_random = np.random.random(birth_mask.shape) < birth_probability
            new_entity_grid[birth_mask & birth_random] = race.race_id

        # Process each race's mining and interactions with asteroids
        for y in range(self.height):
            for x in range(self.width):
                entity = self.entity_grid[y, x]
                if entity > 0:  # There's a race here
                    race = next((r for r in self.races if r.race_id == entity), None)
                    if race and self.grid[y, x] > 0:
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

        # Apply complex dynamics based on statstics
        for race in self.races:
            new_mask = new_entity_grid == race.race_id
            population = np.sum(new_mask)

            # Use statistical models to simulate natural phenomena
            if population > 0:
                # Calculate population change
                # Store current population for future use if needed
                # race.population is updated below
                race.population = population

                # Update population history
                race.population_history.append(population)
                if len(race.population_history) > 100:
                    race.population_history.pop(0)

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
                    # Evolution attempt
                    metrics = race.evolve()
                    race.evolution_points -= race.evolution_threshold
                    race.evolution_threshold = int(race.evolution_threshold * 1.5)

                    # Log evolution
                    logging.info(
                        f"Race {race.race_id} evolved to stage {race.evolution_stage}"
                    )
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

        # Process mineral availability for the evolution algorithm
        for race in self.races:
            # Analyze mineral distribution around race colonies
            race_mask = self.entity_grid == race.race_id
            if np.sum(race_mask) == 0:
                continue  # Race is extinct

            # Calculate mineral availability in race territory
            minerals_available = {
                "common": 0,
                "rare": 0,
                "precious": 0,
                "anomaly": 0,
            }

            # Find all race entity locations
            entity_locations = np.where(race_mask)
            # Check surrounding area for minerals
            search_radius = 3  # Look in a 3-cell radius
            for i in range(len(entity_locations[0])):
                y, x = entity_locations[0][i], entity_locations[1][i]

                # Check surrounding area for minerals
                for dy, dx in itertools.product(
                    range(-search_radius, search_radius + 1),
                    range(-search_radius, search_radius + 1),
                ):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[ny, nx] == 0:
                            rare_type = self.rare_grid[ny, nx]
                        if rare_type == 1:  # Rare
                            minerals_available["rare"] += 1
                        elif rare_type == 2:  # Anomalous
                            minerals_available["anomaly"] += 1
                        elif rare_type == 3:  # Precious
                            minerals_available["precious"] += 1
                        else:  # Common
                            minerals_available["common"] += 1

            # Update race's mineral availability
            race.mineral_availability = minerals_available

        # Update evolution algorithm
        if self.evolution_algorithm:
            self.evolution_algorithm.update(self)

        # Update symbiote evolution algorithm
        if self.symbiote_algorithm:
            self.symbiote_algorithm.update(self)

        # Update symbiote entities
        for symbiote in self.symbiote_entities:
            symbiote.update()

        # Update miner entities
        for miner in self.miner_entities:
            miner.update()

        # Update fleet entities
        for fleet in self.fleet_entities:
            fleet.update()

        return (
            race_income,
            symbiote_income,
            fleet_income,
            symbiote_entities,
            miner_entities,
            fleet_entities,
        )
