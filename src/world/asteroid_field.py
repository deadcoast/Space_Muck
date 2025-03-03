"""
AsteroidField class: Manages the asteroid field grid and symbiote entities.

This module contains the AsteroidField class which handles:
- Procedural asteroid generation using multiple noise algorithms
- Cellular automaton simulation for asteroid evolution
- Energy flow modeling for dynamic resource distribution
- Symbiote race ecosystems that evolve within the field
- Optimized rendering and interaction systems
"""

import itertools
import logging
import math
import random
from typing import Dict, List, Tuple, Set

import numpy as np
import pygame
import scipy.ndimage as ndimage
import scipy.signal as signal
from perlin_noise import PerlinNoise

from src.config import (
    GRID_WIDTH, GRID_HEIGHT, VIEW_WIDTH, VIEW_HEIGHT
)
from src.utils.logging_setup import (
    log_exception,
    LogContext,
    log_performance_start,
    log_performance_end,
)
from src.entities.miner_entity import MinerEntity
from src.generators.procedural_generator import ProceduralGenerator
from src.generators.symbiote_evolution_generator import SymbioteEvolutionGenerator


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

    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT) -> None:
        """
        Initialize a new asteroid field with the specified dimensions.

        Args:
            width: Width of the field in cells
            height: Height of the field in cells
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

        # Initialize with random pattern
        self.initialize_patterns()

        # Create render surface
        self.render_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        self.minimap_surface = pygame.Surface((150, 150))

    def initialize_patterns(self) -> None:
        """
        Initialize the grid with procedurally generated asteroid formations
        using the ProceduralGenerator class.
        """
        with LogContext("Asteroid field initialization"):
            # Clear grid
            self.grid.fill(0)
            self.rare_grid.fill(0)
            self.energy_grid.fill(0)

            try:
                # Use the ProceduralGenerator to generate the asteroid field
                self.generate_field_with_procedural_generator()

                # Add some Game of Life patterns at random locations
                self.add_life_patterns()

                # Update statistics
                self.update_statistics()

                # Mark for redraw
                self.redraw_needed = True

                logging.info(
                    f"Field initialized with {self.total_asteroids} asteroids, "
                    + f"{self.total_rare} rare minerals."
                )
            except Exception as e:
                log_exception(e)
                logging.error(f"Failed to initialize patterns: {str(e)}")
                # Fall back to legacy initialization method
                self._legacy_initialize_patterns()

    def generate_field_with_procedural_generator(self) -> None:
        """
        Generate asteroid field using the ProceduralGenerator class.
        """
        try:
            # Create a ProceduralGenerator with the same dimensions as the field
            generator = ProceduralGenerator(
                width=self.width,
                height=self.height,
                seed=random.randint(1, 10000),
                parameters={
                    "pattern_complexity": self.pattern_complexity,
                    "field_density": self.field_density,
                    "turbulence": self.turbulence,
                    "birth_set": self.birth_set,
                    "survival_set": self.survival_set,
                    "anomaly_chance": self.anomaly_chance,
                    "rare_chance": self.rare_chance,
                    "rare_bonus_multiplier": self.rare_bonus_multiplier,
                },
            )

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

            # Initialize energy grid based on asteroid values if not included in result
            if np.all(self.energy_grid == 0):
                self._initialize_energy_grid()

            logging.info("Asteroid field generated using ProceduralGenerator")

        except Exception as e:
            log_exception(e)
            logging.error(
                f"Failed to generate field with ProceduralGenerator: {str(e)}"
            )
            # Re-raise to trigger fallback in initialize_patterns
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

        # Clear grid
        self.grid.fill(0)
        self.rare_grid.fill(0)
        self.energy_grid.fill(0)

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

        # Add some Game of Life patterns at random locations
        self.add_life_patterns()

        # Update statistics
        self.update_statistics()

        # Mark for redraw
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
            max_pattern_size = (
                max(max(dx for dx, dy in pattern), max(dy for dx, dy in pattern)) + 5
            )

            offset_x = random.randint(margin, self.width - max_pattern_size - margin)
            offset_y = random.randint(margin, self.height - max_pattern_size - margin)

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

    def update_asteroids(self) -> None:
        """Update asteroid grid using cellular automaton rules and energy flow."""
        # Create new grids to avoid affecting the update in progress
        new_grid = np.zeros_like(self.grid)
        new_rare_grid = np.zeros_like(self.rare_grid)
        new_energy_grid = np.zeros_like(self.energy_grid)

        # Use convolution for neighbor counting (more efficient)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Count asteroid neighbors using convolution
        neighbor_counts = signal.convolve2d(
            self.grid > 0,
            kernel,
            mode="same",
            boundary="wrap" if self.apply_edge_wrapping else "fill",
        )

        # Count energy in neighborhood using convolution
        energy_neighborhood = signal.convolve2d(
            self.energy_grid,
            kernel,
            mode="same",
            boundary="wrap" if self.apply_edge_wrapping else "fill",
        )

        # Apply Game of Life rules with energy dynamics
        for y in range(self.height):
            for x in range(self.width):
                has_asteroid = self.grid[y, x] > 0
                local_energy = self.energy_grid[y, x]
                neighbors = neighbor_counts[y, x]

                # New energy starts with decayed version of current energy
                new_energy = local_energy * (1.0 - self.energy_decay)
                new_energy += energy_neighborhood[y, x] * self.energy_spread / 8.0

                # Survival depends on neighbors and energy
                energy_survival_boost = min(2, int(local_energy * 3))
                survival_set = self.survival_set.union(
                    {n + energy_survival_boost for n in self.survival_set}
                )

                if has_asteroid:
                    # Check if asteroid survives
                    if neighbors in survival_set:
                        # If it survives, keep its value and rare status
                        new_grid[y, x] = self.grid[y, x]
                        new_rare_grid[y, x] = self.rare_grid[y, x]
                    else:
                        # Dying asteroid adds energy
                        new_energy += 0.2
                elif neighbors in self.birth_set:
                    # New asteroid born from energy and neighbors
                    new_grid[y, x] = int(50 + energy_neighborhood[y, x] * 100)

                    # Small chance for rare asteroid in births
                    if random.random() < self.rare_chance:
                        new_rare_grid[y, x] = 1
                        new_grid[y, x] = int(
                            new_grid[y, x] * self.rare_bonus_multiplier
                        )

                elif random.random() < self.regen_rate * local_energy:
                    new_grid[y, x] = int(30 + random.random() * 70)
                    if random.random() < self.rare_chance:
                        new_rare_grid[y, x] = 1
                        new_grid[y, x] = int(
                            new_grid[y, x] * self.rare_bonus_multiplier
                        )

                # Store new energy level
                new_energy_grid[y, x] = min(1.0, new_energy)  # Cap energy at 1.0

        # Update grids
        self.grid = new_grid
        self.rare_grid = new_rare_grid
        self.energy_grid = new_energy_grid

        # Add energy to low density areas (encouraging new growth)
        low_density_mask = neighbor_counts < 2
        self.energy_grid[low_density_mask] += 0.05

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
        symbiote_entities = self.symbiote_entities.copy() if hasattr(self, 'symbiote_entities') else []
        miner_entities = self.miner_entities.copy() if hasattr(self, 'miner_entities') else []
        fleet_entities = self.fleet_entities.copy() if hasattr(self, 'fleet_entities') else []

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
