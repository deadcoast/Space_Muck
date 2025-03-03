#!/usr/bin/env python3
"""
Mining Asteroids – Ultimate Procedural Generation Edition

Theme:
  Mine evolving asteroids in a field that follows enhanced Game of Life–style rules.
  Earn credits by mining asteroids and use those credits to purchase upgrades
  that not only improve your ship but also let you influence the procedural generation.
"""

import itertools
import math
import numpy as np
import pygame
import random
import sys
import time
import logging
from typing import List, Tuple, Dict, Any

# Additional imports for advanced mathematical modeling
import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.signal as signal
from sklearn.cluster import KMeans
import networkx as nx
from perlin_noise import PerlinNoise

from src.symbiote_algorithm import SymbioteEvolutionAlgorithm

# -------------------------------------
# Logging Configuration
# -------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# -------------------------------------
# Global Constants & Configuration
# -------------------------------------
# Make grid much larger with smaller cells for better emergent patterns
CELL_SIZE: int = 8  # Smaller cells for larger world
GRID_WIDTH: int = 200  # Much larger grid
GRID_HEIGHT: int = 150
WINDOW_WIDTH: int = 1600  # Fixed window size
WINDOW_HEIGHT: int = 1200
VIEW_WIDTH: int = WINDOW_WIDTH // CELL_SIZE  # Visible grid cells
VIEW_HEIGHT: int = WINDOW_HEIGHT // CELL_SIZE
FPS: int = 60  # Higher FPS for smoother gameplay
MINIMAP_SIZE: int = 150  # Size of the minimap

# Colors (RGB)
COLOR_BG: Tuple[int, int, int] = (10, 10, 15)
COLOR_GRID: Tuple[int, int, int] = (20, 20, 30)
COLOR_ASTEROID: Tuple[int, int, int] = (80, 80, 80)
COLOR_ASTEROID_RARE: Tuple[int, int, int] = (255, 215, 0)
COLOR_PLAYER: Tuple[int, int, int] = (0, 255, 0)
COLOR_TEXT: Tuple[int, int, int] = (220, 220, 220)
COLOR_HIGHLIGHT: Tuple[int, int, int] = (255, 0, 0)
COLOR_SHOP_BG: Tuple[int, int, int] = (20, 20, 30)
COLOR_EVENT: Tuple[int, int, int] = (0, 255, 255)

# Race colors with better distinction
COLOR_RACE_1: Tuple[int, int, int] = (50, 100, 255)  # Blue race
COLOR_RACE_2: Tuple[int, int, int] = (255, 50, 150)  # Magenta race
COLOR_RACE_3: Tuple[int, int, int] = (255, 165, 0)  # Orange race

# Game States
STATE_PLAY: str = "PLAY"
STATE_SHOP: str = "SHOP"
STATE_MAP: str = "MAP"
STATE_SHOP: str = "SHOP"
STATE_MAP: str = "MAP"


# -------------------------------------
# AsteroidField Class
# -------------------------------------
class AsteroidField:
    """
    Represents a field of asteroids on a grid.
    Uses NumPy for performance and supports complex cellular automaton rules.
    """

    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT) -> None:
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

        # Cellular automata parameters (default: similar to Conway's Game of Life)
        self.birth_set: set = {3}
        self.survival_set: set = {2, 3}
        self.energy_decay = 0.02  # Energy decay rate
        self.energy_spread = 0.1  # How much energy spreads to neighbors

        # Asteroid parameters
        self.regen_rate: float = 0.01
        self.rare_chance: float = 0.1
        self.rare_bonus_multiplier: float = 3.0

        # Track mining races
        self.races: List[MinerEntity] = []

        # Stats tracking
        self.total_asteroids = 0
        self.total_rare = 0
        self.total_energy = 0.0

        # Initialize with random pattern
        self.initialize_patterns()

    def initialize_patterns(self) -> None:
        """
        Initialize the grid with procedurally generated asteroid formations
        using Perlin noise for natural-looking distribution
        """
        # Clear grid
        self.grid.fill(0)
        self.rare_grid.fill(0)
        self.energy_grid.fill(0)

        # Create Perlin noise generators for different scales
        noise1 = PerlinNoise(octaves=3, seed=random.randint(1, 1000))
        noise2 = PerlinNoise(octaves=6, seed=random.randint(1, 1000))
        noise3 = PerlinNoise(octaves=12, seed=random.randint(1, 1000))

        # Generate noise map
        noise_map = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                # Normalize coordinates
                nx, ny = x / self.width, y / self.height
                # Combine different noise scales
                noise_val = (
                    0.5 * noise1([nx, ny])
                    + 0.3 * noise2([nx * 2, ny * 2])
                    + 0.2 * noise3([nx * 4, ny * 4])
                )
                noise_map[y, x] = noise_val

        # Normalize noise map to 0-1 range
        noise_map = (noise_map - np.min(noise_map)) / (
            np.max(noise_map) - np.min(noise_map)
        )

        # Create asteroid formations based on noise threshold
        asteroid_threshold = 0.65  # Higher threshold = fewer asteroids
        rare_threshold = 0.85
        # Create asteroid clusters
        for y in range(self.height):
            for x in range(self.width):
                if noise_map[y, x] > asteroid_threshold:
                    # Create asteroid with value based on noise
                    value = int((noise_map[y, x] - asteroid_threshold) * 500) + 100
                    self.grid[y, x] = value

                    # Add energy to this cell
                    self.energy_grid[y, x] = noise_map[y, x] * 0.8

                # Check for rare asteroid
                if noise_map[y, x] > rare_threshold:
                    self.rare_grid[y, x] = 1
                    if noise_map[y, x] > asteroid_threshold:
                        # Higher noise = more valuable asteroids
                        value_factor = (noise_map[y, x] - asteroid_threshold) / (
                            1 - asteroid_threshold
                        )
                        self.grid[y, x] = int(100 + 400 * value_factor)

                        # Energy based on noise value
                        self.energy_grid[y, x] = noise_map[y, x] * 0.8

                        # Rare asteroids
                        if noise_map[y, x] > rare_threshold:
                            self.rare_grid[y, x] = 1
                            self.grid[y, x] = int(
                                self.grid[y, x] * self.rare_bonus_multiplier
                            )

        # Add some Game of Life patterns at random locations
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
        ]

        for _ in range(3):  # Place several patterns
            pattern = random.choice(life_patterns)
            offset_x = random.randint(10, self.width - 20)
            offset_y = random.randint(10, self.height - 20)

            for dx, dy in pattern:
                nx, ny = offset_x + dx, offset_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # High value for pattern cells
                    self.grid[ny, nx] = random.randint(300, 600)
                    self.energy_grid[ny, nx] = random.uniform(0.7, 1.0)
                    if random.random() < 0.3:
                        self.rare_grid[ny, nx] = 1
            pattern = random.choice(life_patterns)
            offset_x = random.randint(10, self.width - 20)
            offset_y = random.randint(10, self.height - 20)

            for dx, dy in pattern:
                nx, ny = offset_x + dx, offset_y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # High value for pattern cells
                    self.grid[ny, nx] = random.randint(300, 600)
                    self.energy_grid[ny, nx] = random.uniform(0.7, 1.0)
                    if random.random() < 0.3:
                        self.rare_grid[ny, nx] = 1
                        self.grid[ny, nx] *= self.rare_bonus_multiplier

    def get_view_bounds(self) -> Tuple[int, int, int, int]:
        """Get the bounds of the current viewport"""
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
        """Update the game state"""
        new_grid = np.zeros_like(self.grid)
        new_rare_grid = np.zeros_like(self.rare_grid)
        new_energy_grid = np.zeros_like(self.energy_grid)

        # Use convolution for neighbor counting (more efficient)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Count asteroid neighbors using convolution
        neighbor_counts = signal.convolve2d(
            self.grid > 0, kernel, mode="same", boundary="wrap"
        )

        # Count energy in neighborhood using convolution
        energy_neighborhood = signal.convolve2d(
            self.energy_grid, kernel, mode="same", boundary="wrap"
        )

        # Apply Game of Life rules with energy dynamics
        for y in range(self.height):
            for x in range(self.width):
                # Get current cell state
                has_asteroid = self.grid[y, x] > 0
                neighbors = neighbor_counts[y, x]
                local_energy = self.energy_grid[y, x]

            # New energy starts with decayed version of current energy
            new_energy = local_energy * (1.0 - self.energy_decay)
            new_energy += energy_neighborhood[y, x] * self.energy_spread / 8.0

            # Survival depends on neighbors and energy
            energy_survival_boost = min(2, int(local_energy * 3))
            survival_set = self.survival_set.union(
                {n + energy_survival_boost for n in self.survival_set}
            )

            if has_asteroid:
                if neighbors in survival_set:
                    # Keep asteroid with possible value change
                    new_grid[y, x] = self.grid[y, x]
                    new_rare_grid[y, x] = self.rare_grid[y, x]
                else:
                    # Asteroid dies (turns to energy)
                    new_energy_grid[y, x] += 0.3

                # Survival depends on neighbors and energy
                energy_survival_boost = min(2, int(local_energy * 3))
                survival_set = self.survival_set.union(
                    {n + energy_survival_boost for n in self.survival_set}
                )
                # Birth chance modified by energy
                birth_chance = self.regen_rate + (local_energy * 0.2)

                # Need to define birth_set for this condition
                birth_set = self.birth_set.union(
                    {n + energy_survival_boost for n in self.birth_set}
                )

                if neighbors in birth_set or random.random() < birth_chance:
                    # Create new asteroid
                    new_grid[y, x] = random.randint(50, 200)
                    # Rare asteroid chance
                    if random.random() < self.rare_chance:
                        new_rare_grid[y, x] = 1
                    new_rare_grid[y, x] = self.rare_grid[y, x]
                    # Survival grants energy
                    new_energy_grid[y, x] += 0.1
                else:
                    # Cell dies - energy release
                    new_energy_grid[y, x] += local_energy * 0.3
                # Store energy in grid with bounds check
                new_energy_grid[y, x] += new_energy
                new_energy_grid[y, x] = max(0.0, min(1.0, new_energy_grid[y, x]))

        # Update grids
        self.grid = new_grid
        self.rare_grid = new_rare_grid
        self.energy_grid = new_energy_grid

        # Add energy to low density areas (encouraging new growth)
        low_density_mask = neighbor_counts < 2
        self.energy_grid[low_density_mask] += 0.05

        # Update race entities with advanced behavior
        self.update_entities()

        # Update statistics
        self.total_asteroids = np.sum(self.grid > 0)
        self.total_rare = np.sum(self.rare_grid)
        self.total_energy = np.sum(self.energy_grid)

        # Balance symbiotes and mining
        self.balance_symbiotes_and_mining()

    def update_entities(self) -> Dict[int, int]:
        """
        Update symbiote races with enhanced mathematical modeling
        """
        try:
            return self._extracted_from_update_entities_4()
        except Exception as e:
            logging.critical(f"Error in update_entities: {str(e)}")
            import traceback

            logging.critical(traceback.format_exc())
            return {}

    # TODO Rename this here and in `update_entities`
    def _extracted_from_update_entities_4(self):
        if not self.races:
            return {}

        race_income = {race.race_id: 0 for race in self.races}

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
                # Calculate size of each colony
                region_sizes = ndimage.sum(
                    race_mask, labeled_regions, range(1, num_regions + 1)
                )

                # Find center of mass for each colony
                centers = ndimage.center_of_mass(
                    race_mask, labeled_regions, range(1, num_regions + 1)
                )

                # Find distance transform (distance to nearest zero)
                # This helps identify the core vs. edge of colonies
                distance_map = ndimage.distance_transform_edt(race_mask)

                # Get all entity locations for this race
                entity_locations = list(zip(*np.where(race_mask)))

                # Calculate colony density based on distance map
                total_distance = np.sum(distance_map)
                self.territory_density = total_distance / max(1, len(entity_locations))

                # Use scipy.stats to model colony growth based on size distribution
                # Larger colonies have better survival chances (normal distribution)
                size_mean = np.mean(region_sizes)
                size_std = max(1, np.std(region_sizes))
                growth_factors = stats.norm.cdf(
                    region_sizes, loc=size_mean, scale=size_std
                )

                # Store this information for behavior decisions
                race.colony_data = {
                    "num_regions": num_regions,
                    "region_sizes": region_sizes,
                    "centers": centers,
                    "growth_factors": growth_factors,
                }

                # Update race's territory metrics
                if centers and len(centers) > 0:
                    # Find the largest colony
                    largest_idx = np.argmax(region_sizes)
                    race.territory_center = (
                        int(centers[largest_idx][1]),
                        int(centers[largest_idx][0]),
                    )

                    # The radius is approximated by sqrt(size/π)
                    race.territory_radius = int(
                        np.sqrt(region_sizes[largest_idx] / np.pi)
                    )

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
                adjusted_birth_set = adjusted_birth_set.union(
                    {n - 1 for n in adjusted_birth_set if n > 1}
                )

            # Create birth mask
            birth_mask = np.zeros_like(empty_mask, dtype=bool)
            for n in adjusted_birth_set:
                birth_mask |= (own_neighbors == n) & empty_mask

            # Apply probabilistic birth based on influence and expansion drive
            if race.race_id in self.influence_grids:
                influence = self.influence_grids[race.race_id]
                # Calculate birth probability based on influence and hunger
                birth_probs = np.zeros_like(birth_mask, dtype=np.float32)
                birth_probs[birth_mask] = (
                    0.6 + influence[birth_mask] * race.genome["expansion_drive"]
                )

                # Apply random sampling using scipy.stats
                random_field = stats.uniform.rvs(0, 1, size=birth_mask.shape)
                successful_births = (random_field < birth_probs) & birth_mask

                # Add new births to the grid
                new_entity_grid[successful_births] = race.race_id

        # Process each race's mining and interactions with asteroids
        for y in range(self.height):
            for x in range(self.width):
                current_race = self.entity_grid[y, x]
                if current_race == 0:  # Skip empty cells
                    continue

                current_race_obj = next(
                    (r for r in self.races if r.race_id == current_race), None
                )

                if not current_race_obj:
                    continue

                # Process mining
                if self.grid[y, x] > 0:
                    # Mine asteroid with efficiency
                    mined_value = int(
                        self.grid[y, x] * current_race_obj.mining_efficiency
                    )
                    race_income[current_race] += mined_value

                    # Consume asteroid
                    self.grid[y, x] = 0
                    if self.rare_grid[y, x] == 1:
                        self.rare_grid[y, x] = 0
                        mined_value *= 2  # Bonus for rare asteroids

                    # Update evolution points
                    current_race_obj.evolution_points += mined_value * 0.05

                    # Check for evolution
                    if (
                        current_race_obj.evolution_points
                        >= current_race_obj.evolution_threshold
                    ):
                        current_race_obj.evolve()
                        logging.info(
                            f"Race {current_race} evolved to stage {current_race_obj.evolution_stage}!"
                        )

        # Apply complex dynamics based on statstics
        for race in self.races:
            new_mask = new_entity_grid == race.race_id
            population = np.sum(new_mask)

            # Use statistical models to simulate natural phenomena
            if population > 0:
                # Apply carrying capacity using logistic growth model
                # dN/dt = rN(1-N/K) - simplified discrete version
                K = 1000  # Carrying capacity
                r = 0.05  # Growth rate

                growth_pressure = r * (1 - population / K)

                if growth_pressure < 0:  # Over carrying capacity
                    # Use distance transform to find perimeter cells (lowest distance values)
                    distance = ndimage.distance_transform_edt(new_mask)

                    # Find cells on the edge (lowest distance)
                    perimeter_mask = (distance == 1) & new_mask

                    # Remove some perimeter cells based on overcrowding pressure
                    removal_count = int(abs(growth_pressure) * population * 0.1)
                    perimeter_indices = np.where(perimeter_mask)

                    if len(perimeter_indices[0]) > 0:
                        # Remove a random sample of perimeter cells
                        indices_to_remove = np.random.choice(
                            len(perimeter_indices[0]),
                            size=min(removal_count, len(perimeter_indices[0])),
                            replace=False,
                        )
                        for idx in indices_to_remove:
                            y, x = (
                                perimeter_indices[0][idx],
                                perimeter_indices[1][idx],
                            )
                            new_entity_grid[y, x] = 0

            # Process mineral availability for the evolution algorithm
        for race in self.races:
            # Analyze mineral distribution around race colonies
            race_mask = self.entity_grid == race.race_id
            if np.sum(race_mask) == 0:
                continue

            # Calculate mineral availability in race territory
            minerals_available = {
                "common": 0,
                "rare": 0,
                # For the algorithm's precious and anomaly types,
                # we'll consider high-value asteroids as precious
                "precious": 0,
                "anomaly": 0,
            }

            # Find all race entity locations
            entity_locations = np.where(race_mask)
            # Check surrounding area for minerals
            search_radius = 3  # Look in a 3-cell radius
            for i in range(len(entity_locations[0])):
                y, x = entity_locations[0][i], entity_locations[1][i]

                for dy, dx in itertools.product(
                    range(-search_radius, search_radius + 1),
                    range(-search_radius, search_radius + 1),
                ):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and self.grid[ny, nx] > 0
                    ):
                        value = self.grid[ny, nx]
                        if self.rare_grid[ny, nx] == 1:
                            if value > 500:
                                minerals_available["anomaly"] += 1
                            else:
                                minerals_available["rare"] += 1
                        elif value > 300:
                            minerals_available["precious"] += 1
                        else:
                            minerals_available["common"] += 1

            # Process minerals through evolution algorithm
            # Only process a portion of available minerals
            minerals_to_process = {
                k: min(v, 10)
                for k, v in minerals_available.items()  # Cap to avoid overwhelming
            }

            if any(minerals_to_process.values()):
                population_change, mutations = race.process_minerals(
                    minerals_to_process
                )

                if mutations:
                    for mutation in mutations:
                        if mutation.get("type") in ["significant", "beneficial"]:
                            # Log significant mutations for player awareness
                            attr = mutation.get("attribute", "unknown")
                            mag = mutation.get("magnitude", 1.0)
                            direction = "increased" if mag > 1 else "decreased"
                            pct = abs((mag - 1) * 100)

                            logging.info(
                                f"Race {race.race_id} mutation: {attr} {direction} by {pct:.1f}%"
                            )

        # Update entity grid
        self.entity_grid = new_entity_grid

        # Update population stats and behavior
        for race in self.races:
            race.population = np.sum(self.entity_grid == race.race_id)
            race.last_income = race_income.get(race.race_id, 0)

            # Record history
            race.income_history.append(race.last_income)
            race.population_history.append(race.population)

            # Apply statistical analysis to historical data
            if len(race.population_history) > 20:
                # Use linear regression to analyze population trend
                x = np.arange(len(race.population_history[-20:]))
                y = np.array(race.population_history[-20:])

                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # Adjust behavior based on population trend
                if slope < -0.5:  # Rapidly declining
                    # Increase expansion drive to recover
                    race.genome["expansion_drive"] *= 1.05
                    race.genome["aggression_base"] *= 1.05
                elif slope > 1.0:  # Rapidly growing
                    # Can afford to be more selective
                    race.genome["intelligence"] *= 1.05

            # Trim history if too long
            if len(race.income_history) > 100:
                race.income_history.pop(0)
            if len(race.population_history) > 100:
                race.population_history.pop(0)

        return race_income

    def balance_symbiotes_and_mining(self):
        """Balance symbiote growth with resource availability"""
        # Calculate resource availability across the grid
        total_minerals = np.sum(self.grid)
        grid_area = self.width * self.height
        resource_density = total_minerals / (grid_area * 100)  # Normalize

        total_symbiotes = sum(race.population for race in self.races)
        # Calculate symbiote density
        symbiote_density = total_symbiotes / grid_area

        # Determine if we need to adjust resource regeneration
        if symbiote_density > resource_density * 2:
            # Too many symbiotes, increase resources
            self.regen_rate = min(0.05, self.regen_rate * 1.05)
        elif resource_density > symbiote_density * 5:
            # Too many resources, decrease regeneration
            self.regen_rate = max(0.005, self.regen_rate * 0.95)

        # For each race, adjust visibility and activity based on population and feeding
        for race in self.races:
            # Ensure minimum visibility
            if (
                race.population > 0
                and np.sum(self.entity_grid == race.race_id) < race.population * 0.2
            ):
                # Force manifest the race if it's not visible enough
                empty_cells = np.where((self.entity_grid == 0) & (self.grid > 0))
                if len(empty_cells[0]) > 0:
                    # Select random positions near minerals
                    indices = np.random.choice(
                        len(empty_cells[0]),
                        min(len(empty_cells[0]), max(1, race.population // 20)),
                    )
                    for i in indices:
                        self.entity_grid[empty_cells[0][i], empty_cells[1][i]] = (
                            race.race_id
                        )

                    logging.info(
                        f"Race {race.race_id} forcibly manifested with {len(indices)} visible symbiotes"
                    )

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the visible portion of the field"""
        try:
            # Get view bounds
            view_x1, view_y1, view_x2, view_y2 = self.get_view_bounds()

            # Add safety checks to ensure view bounds are valid
            view_x1 = max(0, min(view_x1, self.width - 1))
            view_y1 = max(0, min(view_y1, self.height - 1))
            view_x2 = max(view_x1 + 1, min(view_x2, self.width))
            view_y2 = max(view_y1 + 1, min(view_y2, self.height))

            # Calculate screen position and cell size based on zoom
            screen_cell_size = int(CELL_SIZE * self.zoom)

            # Clear entire surface
            surface.fill(COLOR_BG)

            # Draw visible grid
            for y, x in itertools.product(
                range(view_y1, view_y2), range(view_x1, view_x2)
            ):
                # Calculate screen position
                screen_x = int((x - view_x1) * screen_cell_size)
                screen_y = int((y - view_y1) * screen_cell_size)

                # Create cell rectangle
                rect = pygame.Rect(
                    screen_x, screen_y, screen_cell_size, screen_cell_size
                )

                # Draw cell content
                if 0 <= y < self.height and 0 <= x < self.width:  # Extra bounds check
                    if self.grid[y, x] > 0:
                        # Asteroid
                        value = self.grid[y, x]
                        if self.rare_grid[y, x] == 1:
                            # Rare asteroid with value-based color shade
                            intensity = min(255, 180 + value // 3)
                            color = (intensity, intensity * 0.8, 0)
                        else:
                            brightness = min(200, 80 + value // 3)
                            color = (brightness, brightness, brightness)
                        pygame.draw.rect(surface, color, rect)
                    else:
                        # Empty cell - visualize energy levels
                        energy = self.energy_grid[y, x]
                        if energy > 0.1:
                            # Only show significant energy
                            alpha = int(energy * 100)
                            color = (0, 0, int(energy * 150))
                            s = pygame.Surface((rect.width, rect.height))
                            s.fill(color)
                            s.set_alpha(alpha)
                            surface.blit(s, rect)

                        # Draw entity on top if present
                if 0 <= y < self.height and 0 <= x < self.width:  # Extra bounds check
                    entity = self.entity_grid[y, x]
                    if entity > 0:
                        if race := next(
                            (r for r in self.races if r.race_id == entity), None
                        ):
                            # Draw entity based on race
                            pygame.draw.rect(surface, race.color, rect)

        except Exception as e:
            logging.error(f"Error in AsteroidField.draw: {str(e)}")
            import traceback

            logging.error(traceback.format_exc())

    def draw_entities(self, surface, viewport_x, viewport_y, view_width, view_height):
        """Draw symbiote entities with enhanced visual clarity"""
        # Create a surface for glow effects
        glow_surface = pygame.Surface((CELL_SIZE * 3, CELL_SIZE * 3), pygame.SRCALPHA)

        # For each visible cell with an entity
        for y in range(viewport_y, viewport_y + view_height):
            for x in range(viewport_x, viewport_x + view_width):
                if 0 <= y < self.height and 0 <= x < self.width:
                    entity_id = self.entity_grid[y, x]
                    if entity_id > 0:
                        if race := next(
                            (r for r in self.races if r.race_id == entity_id), None
                        ):
                            # Calculate screen position
                            screen_x = (x - viewport_x) * CELL_SIZE
                            screen_y = (y - viewport_y) * CELL_SIZE

                            # Draw base entity with enhanced visibility
                            pygame.draw.rect(
                                surface,
                                race.color,
                                (screen_x, screen_y, CELL_SIZE, CELL_SIZE),
                            )

                            # Add pulsing glow effect for better visibility
                            pulse = abs(math.sin(time.time() * 3)) * 0.7 + 0.3
                            glow_color = tuple(
                                min(255, int(c * pulse)) for c in race.color
                            )

                            # Draw the glow
                            pygame.draw.circle(
                                glow_surface,
                                (*glow_color, 100),
                                (CELL_SIZE * 1.5, CELL_SIZE * 1.5),
                                CELL_SIZE * 1.2,
                            )

                            # Blit the glow centered on the entity
                            surface.blit(
                                glow_surface,
                                (screen_x - CELL_SIZE, screen_y - CELL_SIZE),
                                special_flags=pygame.BLEND_ADD,
                            )

                            # Add colony identification
                            if (
                                hasattr(race, "colony_map")
                                and race.colony_map is not None
                            ):
                                if colony_id := race.colony_map.get((y, x), 0):
                                    # Mark colony centers with a symbol
                                    is_center = race.colony_centers.get(
                                        colony_id, (0, 0)
                                    ) == (y, x)
                                    if is_center:
                                        pygame.draw.circle(
                                            surface,
                                            (255, 255, 255),
                                            (
                                                screen_x + CELL_SIZE // 2,
                                                screen_y + CELL_SIZE // 2,
                                            ),
                                            CELL_SIZE // 3,
                                        )

    def manual_seed(self, center_x: int, center_y: int, radius: int = 3) -> None:
        """
        Manually seed a cluster of asteroids at the specified location

        Args:
            center_x: Center X coordinate
            center_y: Center Y coordinate
            radius: Radius of the asteroid cluster
        """
        for y, x in itertools.product(
            range(center_y - radius, center_y + radius + 1),
            range(center_x - radius, center_x + radius + 1),
        ):
            # Check if coordinates are within bounds
            if 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] == 0:
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                # Only place asteroids within the radius
                if dist <= radius:
                    # Higher value near the center
                    value = int(300 - (dist / radius) * 200)
                    self.grid[y, x] = value

                    # Add energy to seeded asteroids
                    self.energy_grid[y, x] = 0.6 - (dist / radius) * 0.4

                    # Small chance for rare asteroid
                    if random.random() < self.rare_chance:
                        self.rare_grid[y, x] = 1
                        self.grid[y, x] = int(
                            self.grid[y, x] * self.rare_bonus_multiplier
                        )


# -------------------------------------
# Enhanced Symbiote Evolution System
# -------------------------------------
class MinerEntity:
    def __init__(
        self, race_id, color, birth_set=None, survival_set=None, initial_density=0.001
    ):
        # Initialize race parameters
        self.race_id = race_id
        self.color = color
        self.birth_set = birth_set or {3}
        self.survival_set = survival_set or {2, 3}
        self.initial_density = initial_density

        # Add missing attributes
        self.aggression = 0.2  # Default aggression level
        self.hunger = 0.0  # Hunger level (0-1)
        self.hunger_rate = 0.01  # Rate at which hunger increases
        self.trait = random.choice(
            ["adaptive", "expansive", "selective"]
        )  # Random trait
        self.population = 0
        self.fed_this_turn = False
        self.last_income = 0
        self.income_history = []
        self.population_history = []
        self.evolution_points = 0
        self.evolution_threshold = 100
        self.evolution_stage = 1
        self.current_behavior = "feeding"  # Default behavior
        self.mining_efficiency = 0.5  # Base mining efficiency

        # Initialize genome
        self.genome = self._initialize_genome_by_trait()

        # Spatial analysis data
        self.territory_center = None
        self.territory_radius = 0
        self.territory_density = 0
        self.colony_data = {}

        # Initialize the advanced algorithm
        self.evolution_algorithm = SymbioteEvolutionAlgorithm(
            initial_aggression=self.aggression,
            growth_rate=0.05 * self.genome["metabolism_rate"],
            base_mutation_rate=self.genome["mutation_rate"],
            learning_enabled=True,
        )

        # Track mineral consumption for the algorithm
        self.mineral_consumption = {"common": 0, "rare": 0, "precious": 0, "anomaly": 0}

    # Add a new method to process minerals through the evolution algorithm
    def process_minerals(self, minerals_data):
        """
        Process minerals through the evolution algorithm and apply the results

        Args:
            minerals_data: Dictionary with mineral types and counts
        """
        # Convert from game's mineral representation to algorithm's format
        algorithm_minerals = {
            "common": minerals_data.get("common", 0),
            "rare": minerals_data.get("rare", 0),
            "precious": minerals_data.get("precious", 0),
            "anomaly": minerals_data.get("anomaly", 0),
        }

        # Track consumed minerals
        for mineral_type, amount in algorithm_minerals.items():
            self.mineral_consumption[mineral_type] += amount

        # Process minerals through the evolution algorithm
        new_population, new_aggression, mutations = (
            self.evolution_algorithm.process_mineral_feeding(
                self.race_id, algorithm_minerals, self.population, self.aggression
            )
        )

        # Apply the results
        population_change = new_population - self.population
        self.population = new_population
        self.aggression = new_aggression

        # Process mutations
        self.apply_mutations(mutations)

        return population_change, mutations

    def apply_mutations(self, mutations):
        """
        Apply mutations from the evolution algorithm to the race

        Args:
            mutations: List of mutation dictionaries from the algorithm
        """
        if not mutations:
            return

        for mutation in mutations:
            attribute = mutation.get("attribute")
            magnitude = mutation.get("magnitude", 1.0)

            # Apply mutation to genome if attribute exists
            if attribute in self.genome:
                old_value = self.genome[attribute]
                self.genome[attribute] *= magnitude

                # Log significant mutations
                if abs(1 - magnitude) > 0.1:
                    mutation_type = "significant" if magnitude > 1.0 else "negative"
                    logging.info(
                        f"Race {self.race_id} experienced {mutation_type} mutation: "
                        f"{attribute} changed from {old_value:.2f} to {self.genome[attribute]:.2f}"
                    )

            # Special mutations might grant evolution points
            if mutation.get("type") == "beneficial":
                self.evolution_points += 50

    def _initialize_genome_by_trait(self) -> Dict[str, float]:
        """Initialize genome based on race trait"""
        # Base genome values
        genome = {
            "metabolism_rate": random.uniform(0.8, 1.2),  # How fast they consume energy
            "expansion_drive": random.uniform(0.5, 1.5),  # How aggressively they expand
            "mutation_rate": random.uniform(0.01, 0.1),  # How quickly they evolve
            "intelligence": random.uniform(
                0.1, 0.5
            ),  # How strategically they target resources
            "aggression_base": random.uniform(0.1, 0.3),  # Base aggression level
        }

        # Adjust genome based on trait
        if self.trait == "adaptive":
            genome["metabolism_rate"] *= 1.2
            genome["mutation_rate"] *= 1.3
        elif self.trait == "expansive":
            genome["expansion_drive"] *= 1.5
            genome["aggression_base"] *= 1.2
        elif self.trait == "selective":
            genome["intelligence"] *= 1.5
            genome["metabolism_rate"] *= 0.8  # More efficient

        return genome

    def update_hunger(self, minerals_consumed: int) -> float:
        """
        Update hunger based on minerals consumed and genome properties
        Returns the current aggression level
        """
        # Hunger increases over time, affected by metabolism rate
        self.hunger += self.hunger_rate * self.genome["metabolism_rate"]

        # Feeding reduces hunger
        if minerals_consumed > 0:
            self.fed_this_turn = True
            hunger_reduction = (minerals_consumed / 1000.0) * (
                1.0 / self.genome["metabolism_rate"]
            )
            self.hunger -= hunger_reduction

            # Feeding can contribute to evolution
            self.evolution_points += minerals_consumed * 0.1

        # Cap hunger between 0 and 1
        self.hunger = max(0.0, min(1.0, self.hunger))

        # Calculate current aggression based on hunger and genome
        current_aggression = (
            self.genome["aggression_base"]
            + (self.hunger * 0.8) * self.genome["expansion_drive"]
        )

        # Update behavior state based on hunger and population
        self._update_behavior_state()

        return current_aggression

    def _update_behavior_state(self) -> None:
        """Update the current behavior state using statistical models"""
        try:
            # Use a Markov model to determine behavior transitions
            # This creates more realistic and consistent behavior changes
            transition_matrix = {
                "feeding": {
                    "feeding": 0.6,
                    "expanding": 0.2,
                    "defensive": 0.1,
                    "aggressive": 0.1,
                },
                "expanding": {
                    "feeding": 0.2,
                    "expanding": 0.6,
                    "defensive": 0.1,
                    "aggressive": 0.1,
                },
                "defensive": {
                    "feeding": 0.1,
                    "expanding": 0.1,
                    "defensive": 0.7,
                    "aggressive": 0.1,
                },
                "aggressive": {
                    "feeding": 0.1,
                    "expanding": 0.1,
                    "defensive": 0.2,
                    "aggressive": 0.6,
                },
            }

            # Modify transition probabilities based on current state
            current_transitions = transition_matrix[self.current_behavior]

            # Hunger greatly increases chance of feeding behavior
            if self.hunger > 0.7:
                for state in current_transitions:
                    if state == "feeding":
                        current_transitions[state] *= 2.0
                    else:
                        current_transitions[state] *= 0.5

            # Population size affects expansion drive
            if self.population > 200:
                # Large populations tend to expand more
                for state in current_transitions:
                    if state == "expanding":
                        current_transitions[state] *= 1.5

            # Being fed reduces aggression
            if self.fed_this_turn:
                for state in current_transitions:
                    if state == "aggressive":
                        current_transitions[state] *= 0.3

            # Normalize probabilities
            total = sum(current_transitions.values())
            normalized_transitions = {
                state: prob / total for state, prob in current_transitions.items()
            }

            # Select new state using weighted choice
            states = list(normalized_transitions.keys())
            probs = list(normalized_transitions.values())

            # Use scipy.stats for proper statistical sampling
            # This gives more realistic probability distributions
            cum_probs = np.cumsum(probs)
            sample = stats.uniform.rvs(0, 1)

            for i, threshold in enumerate(cum_probs):
                if sample <= threshold:
                    self.current_behavior = states[i]
                    break

        except Exception as e:
            logging.error(f"Error in behavior update: {str(e)}")
            # Default to feeding as fallback
            self.current_behavior = "feeding"

    def evolve(self) -> None:
        """Evolve the race to the next stage, improving its abilities"""
        self.evolution_stage += 1
        entity_locations = [
            (x, y)
            for y, x in itertools.product(
                range(self.field.height), range(self.field.width)
            )
            if self.field.entity_grid[y, x] == self.race_id
        ]
        if not entity_locations:
            return {
                "center": None,
                "radius": 0,
                "density": 0,
                "resource_access": 0,
                "fragmentation": 0,
            }

        # Convert to numpy array for clustering
        points = np.array(entity_locations)

        # K-means clustering to find centers of population
        k = min(3, len(points))  # Use up to 3 clusters
        if k > 0:
            return self._extracted_from_evolve_26(k, points, entity_locations)
        return {
            "center": None,
            "radius": 0,
            "density": 0,
            "resource_access": 0,
            "fragmentation": 0,
        }

    # TODO Rename this here and in `evolve`
    def _extracted_from_evolve_26(self, k, points, entity_locations):
        kmeans = KMeans(n_clusters=k).fit(points)
        clusters = kmeans.labels()
        centers = kmeans.cluster_centers()

        # Find main cluster (largest population)
        cluster_sizes = [np.sum(clusters == i) for i in range(k)]
        main_cluster_idx = np.argmax(cluster_sizes)
        main_center = centers[main_cluster_idx]

        # Calculate territory metrics
        self.territory_center = (int(main_center[0]), int(main_center[1]))

        # Calculate radius as distance to furthest entity in main cluster
        main_cluster_points = points[clusters == main_cluster_idx]
        distances = np.sqrt(((main_cluster_points - main_center) ** 2).sum(axis=1))
        self.territory_radius = int(np.max(distances))

        # Calculate density as entities per unit area
        area = max(1, np.pi * (self.territory_radius**2))
        self.territory_density = len(main_cluster_points) / area

        # Measure resource access (asteroids within territory)
        resource_access = 0
        for y in range(
            max(0, int(main_center[1] - self.territory_radius)),
            min(self.field.height, int(main_center[1] + self.territory_radius + 1)),
        ):
            for x in range(
                max(0, int(main_center[0] - self.territory_radius)),
                min(
                    self.field.width,
                    int(main_center[0] + self.territory_radius + 1),
                ),
            ):
                if self.field.grid[y, x] > 0:
                    distance = np.sqrt(
                        (x - main_center[0]) ** 2 + (y - main_center[1]) ** 2
                    )
                    if distance <= self.territory_radius:
                        resource_access += self.field.grid[y, x]

        # Measure fragmentation using network analysis
        # Create a graph where close entities are connected
        G = nx.Graph()
        for i, (x, y) in enumerate(entity_locations):
            G.add_node(i, pos=(x, y))

        # Connect entities that are close to each other
        for i in range(len(entity_locations)):
            for j in range(i + 1, len(entity_locations)):
                x1, y1 = entity_locations[i]
                x2, y2 = entity_locations[j]
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if distance < 10:  # Connection threshold
                    G.add_edge(i, j)

            # Calculate fragmentation as number of connected components
        fragmentation = (
            nx.number_connected_components(G) / len(G.nodes) if len(G.nodes) > 0 else 0
        )
        return {
            "center": self.territory_center,
            "radius": self.territory_radius,
            "density": self.territory_density,
            "resource_access": resource_access,
            "fragmentation": fragmentation,
        }

    def populate(self, field: AsteroidField) -> None:
        """Populate the field with this race's symbiotes using unique patterns based on traits"""
        # Safe check to ensure field is valid
        if (
            not isinstance(field, AsteroidField)
            or field.width == 0
            or field.height == 0
        ):
            logging.error("Invalid field provided for population")
            return

        # Store field as instance variable
        self.field = field

        # Each race has a distinct settlement pattern
        try:
            if self.trait == "adaptive":  # Blue race - settle in clusters
                # Use Perlin noise to create organic-looking clusters
                noise = PerlinNoise(octaves=4, seed=random.randint(1, 1000))

                # Calculate a starting point based on field dimensions
                start_x = random.randint(field.width // 4, field.width * 3 // 4)
                start_y = random.randint(field.height // 4, field.height * 3 // 4)

                # Create several organic clusters
                for _ in range(3):
                    center_x = start_x + random.randint(-30, 30)
                    center_y = start_y + random.randint(-30, 30)

                    # Define cluster radius
                    radius = random.randint(10, 20)

                    # Populate cluster area using noise
                    for y in range(
                        max(0, center_y - radius), min(field.height, center_y + radius)
                    ):
                        for x in range(
                            max(0, center_x - radius),
                            min(field.width, center_x + radius),
                        ):
                            # Calculate distance from center
                            dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                            if dist <= radius:
                                # Use noise to determine placement
                                noise_val = noise([x / field.width, y / field.height])
                                # Higher chance near center, affected by noise
                                chance = (
                                    self.initial_density
                                    * (0.8 + noise_val)
                                    * (1 - dist / radius)
                                )
                                if (
                                    random.random() < chance
                                    and field.entity_grid[y, x] == 0
                                ):
                                    field.entity_grid[y, x] = self.race_id

            elif self.trait == "expansive":  # Magenta race - settle in networks/paths
                self._generate_random_nodes(field)
            else:  # Orange race (selective) - settle near resource-rich areas
                # First find high-value asteroid clusters using KMeans
                asteroid_cells = []
                asteroid_values = []

                for y, x in itertools.product(
                    range(self.field.height), range(self.field.width)
                ):
                    if self.field.grid[y, x] > 0:
                        value = field.grid[y, x]
                        if field.rare_grid[y, x] == 1:
                            value *= field.rare_bonus_multiplier
                        asteroid_cells.append((x, y))
                        asteroid_values.append(value)

                if not asteroid_cells:
                    # No asteroids found, place randomly
                    for _ in range(
                        int(field.width * field.height * self.initial_density * 0.1)
                    ):
                        x = random.randint(0, field.width - 1)
                        y = random.randint(0, field.height - 1)
                        if field.entity_grid[y, x] == 0:
                            field.entity_grid[y, x] = self.race_id
                    return

                # Convert to numpy array for clustering
                points = np.array(asteroid_cells)
                values = np.array(asteroid_values)

                # Find the best asteroid clusters
                if len(points) > 5:  # Need enough points for meaningful clustering
                    k = min(3, len(points) // 5)  # Use up to 3 clusters
                    kmeans = KMeans(n_clusters=k).fit(points, sample_weight=values)

                    # For each cluster center, settle nearby
                    for center in kmeans.cluster_centers_:
                        center_x, center_y = int(center[0]), int(center[1])

                        # Place symbiotes with higher density near the cluster center
                        radius = random.randint(10, 15)
                        for y in range(
                            max(0, center_y - radius),
                            min(field.height, center_y + radius),
                        ):
                            for x in range(
                                max(0, center_x - radius),
                                min(field.width, center_x + radius),
                            ):
                                # Calculate distance from center
                                dist = math.sqrt(
                                    (x - center_x) ** 2 + (y - center_y) ** 2
                                )
                                if dist <= radius:
                                    # Higher chance near center, drops off with distance
                                    chance = (
                                        self.initial_density
                                        * 5
                                        * (1 - dist / radius) ** 2
                                    )
                                    if (
                                        random.random() < chance
                                        and field.entity_grid[y, x] == 0
                                    ):
                                        field.entity_grid[y, x] = self.race_id
                else:
                    # Not enough points for clustering, use simple proximity
                    for cell, value in zip(asteroid_cells, asteroid_values):
                        x, y = cell

                        # Place symbiotes near valuable asteroids
                        radius = int(3 + value / 100)  # Higher value = larger cluster
                        for dy, dx in itertools.product(
                            range(-radius, radius + 1), range(-radius, radius + 1)
                        ):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < field.width and 0 <= ny < field.height:
                                dist = math.sqrt(dx * dx + dy * dy)
                                if dist <= radius:
                                    # Chance based on distance and asteroid value
                                    chance = (
                                        self.initial_density
                                        * (value / 100)
                                        * (1 - dist / radius)
                                    )
                                    if (
                                        random.random() < chance
                                        and field.entity_grid[ny, nx] == 0
                                    ):
                                        field.entity_grid[ny, nx] = self.race_id

        except Exception as e:
            logging.error(f"Error in populate method: {str(e)}")
            import traceback

            logging.error(traceback.format_exc())

    def _generate_random_nodes(self, field):
        # Create network-like structures using graph theory concepts
        import networkx as nx

        # Generate random nodes
        num_nodes = random.randint(5, 10)
        nodes = [
            (
                random.randint(10, field.width - 10),
                random.randint(10, field.height - 10),
            )
            for _ in range(num_nodes)
        ]
        # Create a minimum spanning tree to connect nodes
        G = nx.Graph()
        for i, node in enumerate(nodes):
            G.add_node(i, pos=node)

        # Connect all nodes with edges
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Calculate euclidean distance
                dist = math.sqrt(
                    (nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2
                )
                G.add_edge(i, j, weight=dist)

        # Get minimum spanning tree
        mst = nx.minimum_spanning_tree(G)

        # Draw paths along the edges
        for u, v in mst.edges():
            start_x, start_y = nodes[u]
            end_x, end_y = nodes[v]

            # Draw line between points (Bresenham's line algorithm)
            dx = abs(end_x - start_x)
            dy = abs(end_y - start_y)
            sx = 1 if start_x < end_x else -1
            sy = 1 if start_y < end_y else -1
            err = dx - dy

            x, y = start_x, start_y
            while True:
                # Place entity with some randomness
                if (
                    0 <= x < field.width
                    and 0 <= y < field.height
                    and (random.random() < 0.7 and field.entity_grid[y, x] == 0)
                ):
                    field.entity_grid[y, x] = self.race_id

                if x == end_x and y == end_y:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

        # Add some random nodes near the paths
        for y in range(field.height):
            for x in range(field.width):
                if field.entity_grid[y, x] == self.race_id:
                    # Create potential spread points nearby
                    for _ in range(2):
                        nx = x + random.randint(-3, 3)
                        ny = y + random.randint(-3, 3)
                        if (
                            0 <= nx < field.width
                            and 0 <= ny < field.height
                            and (
                                random.random() < 0.2 and field.entity_grid[ny, nx] == 0
                            )
                        ):
                            field.entity_grid[ny, nx] = self.race_id
                            field.entity_grid[ny, nx] = self.race_id

    # Enhance the MinerEntity class with better algorithm integration

    def initialize_with_algorithm(self):
        """Initialize with the advanced symbiote algorithm"""
        # Configure the algorithm based on race traits
        aggression_base = 0.2
        if self.trait == "expansive":
            aggression_base = 0.3
        elif self.trait == "aggressive":
            aggression_base = 0.4

        growth_modifier = 1.2 if self.trait == "expansive" else 1.0
        # Initialize algorithm with race-specific parameters
        self.evolution_algorithm = SymbioteEvolutionAlgorithm(
            initial_aggression=aggression_base,
            growth_rate=0.05 * growth_modifier,
            base_mutation_rate=self.genome["mutation_rate"],
            carrying_capacity=100 + int(50 * self.genome.get("territory_size", 1.0)),
            learning_enabled=True,
        )

        # Initialize colony tracking
        self.colony_map = {}
        self.colony_centers = {}
        self.previous_grid = None

    def process_evolution(self, field):
        """Process symbiote evolution with the advanced algorithm"""
        # Extract entity grid for this race
        race_grid = (field.entity_grid == self.race_id).astype(np.int8)

        if self.previous_grid is None:
            self.previous_grid = race_grid.copy()

        # Get cellular automaton rules
        birth_set, survival_set = (
            self.evolution_algorithm.generate_cellular_automaton_rules(
                self.race_id, self.hunger, self.genome
            )
        )

        # Update grid using cellular automaton rules
        new_grid = self.evolution_algorithm.update_cellular_automaton(
            race_grid, birth_set, survival_set
        )

        # Apply environmental effects from minerals
        mineral_map = field.grid.astype(np.float32) / 100.0  # Normalize
        environment_hostility = 0.05  # Low base hostility
        new_grid = self.evolution_algorithm.apply_environmental_effects(
            new_grid, mineral_map, environment_hostility
        )

        # Process colony interactions
        new_grid = self.evolution_algorithm.simulate_colony_interaction(
            new_grid, self.genome, self.aggression
        )

        # Identify colonies
        labeled_grid, num_colonies = self.evolution_algorithm.identify_colonies(
            new_grid
        )
        self.colony_map = {}
        self.colony_centers = {}

        # Get colony stats and track them
        colony_stats = self.evolution_algorithm.get_colony_stats(
            new_grid, labeled_grid, num_colonies
        )
        for colony in colony_stats:
            cy, cx = map(int, colony["centroid"])
            self.colony_centers[colony["id"]] = (cy, cx)

            # Mark cells in this colony
            for y, x in itertools.product(range(field.height), range(field.width)):
                if labeled_grid[y, x] == colony["id"]:
                    self.colony_map[(y, x)] = colony["id"]

        # Calculate population based on actual grid cells
        self.population = np.sum(new_grid)

        # Calculate expansion rate for aggression adjustment
        expansion_index = self.evolution_algorithm.calculate_expansion_index(
            new_grid, self.previous_grid
        )
        self.expansion_rate = expansion_index

        # Update the field's entity grid with our new state
        field.entity_grid[new_grid == 1] = self.race_id

        # Update previous grid for next cycle
        self.previous_grid = new_grid.copy()

        # Update hunger based on population
        self.hunger += (
            0.1 * (self.population / 100) * self.genome.get("metabolism_rate", 1.0)
        )
        self.hunger = min(1.0, self.hunger)

        return num_colonies, self.population


# -------------------------------------
# Player Class (Mining Ship)
# -------------------------------------
class Player:
    """
    Represents the mining ship controlled by the user.
    The ship can move around the field and mine asteroids to earn credits.
    """

    def __init__(self) -> None:
        self.x: int = GRID_WIDTH // 2
        self.y: int = GRID_HEIGHT // 2
        self.mining_efficiency: float = 1.0
        self.mining_range: int = 0
        self.currency: int = 100
        self.auto_miners: int = 0
        self.move_speed: int = 1
        self.total_mined: int = 0
        self.total_rare_mined: int = 0
        self.mining_history: List[int] = []
        self.auto_upgrade: bool = False  # Add this line

        # Mining fleet
        self.mining_ships = 1  # Start with one ship
        self.max_mining_ships = 10  # Maximum number of ships
        self.ship_positions = [(self.x, self.y)]  # Track ship positions
        self.ship_health = [100]  # Each ship has health
        self.ship_cost = 500  # Cost to build a new ship

    def move(self, dx: int, dy: int, field: AsteroidField) -> None:
        """
        Move the ship by (dx, dy) if within field bounds.
        Updates camera position to follow the player.
        """
        new_x: int = self.x + dx
        new_y: int = self.y + dy
        if 0 <= new_x < field.width and 0 <= new_y < field.height:
            self.x = new_x
            self.y = new_y

            # Update camera position to follow player
            field.camera_x = self.x
            field.camera_y = self.y

    def mine(self, field: AsteroidField) -> int:
        """
        Mine asteroids in the area defined by mining_range (a square centered on the ship).
        Each mined asteroid yields credits equal to its mineral value multiplied by mining_efficiency.
        The asteroid is removed from the field.
        """
        total_value: int = 0
        for dy, dx in itertools.product(
            range(-self.mining_range, self.mining_range + 1),
            range(-self.mining_range, self.mining_range + 1),
        ):
            mx: int = self.x + dx
            my: int = self.y + dy
            if (
                0 <= mx < field.width
                and 0 <= my < field.height
                and field.grid[my, mx] > 0
            ):
                total_value += field.grid[my, mx]
                if field.rare_grid[my, mx] == 1:
                    self.total_rare_mined += 1
                field.grid[my, mx] = 0
                field.rare_grid[my, mx] = 0
        reward: int = int(total_value * self.mining_efficiency)
        self.currency += reward
        self.total_mined += reward
        self.mining_history.append(reward)
        if len(self.mining_history) > 100:
            self.mining_history.pop(0)
        return reward

    def auto_mine(self, field: AsteroidField) -> int:
        """
        Auto-miners operate on the same area as manual mining at 50% efficiency.
        The yield is multiplied by the number of drones.
        """
        if self.auto_miners <= 0:
            return 0
        total_value: int = 0
        for dy, dx in itertools.product(
            range(-self.mining_range, self.mining_range + 1),
            range(-self.mining_range, self.mining_range + 1),
        ):
            mx: int = self.x + dx
            my: int = self.y + dy
            if (
                0 <= mx < field.width
                and 0 <= my < field.height
                and field.grid[my, mx] > 0
            ):
                total_value += field.grid[my, mx]
                if field.rare_grid[my, mx] == 1:
                    self.total_rare_mined += 1
                field.grid[my, mx] = 0
                field.rare_grid[my, mx] = 0
        reward: int = int(total_value * self.mining_efficiency * 0.5)
        total_reward: int = reward * self.auto_miners
        self.currency += total_reward
        self.total_mined += total_reward
        self.mining_history.append(total_reward)
        if len(self.mining_history) > 100:
            self.mining_history.pop(0)
        return total_reward

    def draw(self, surface: pygame.Surface, field: AsteroidField) -> None:
        """Draw the mining ship considering camera position and zoom."""
        # Get view bounds
        view_x1, view_y1, _, _ = field.get_view_bounds()

        # Calculate screen position based on zoom
        screen_cell_size = int(CELL_SIZE * field.zoom)
        screen_x = int((self.x - view_x1) * screen_cell_size)
        screen_y = int((self.y - view_y1) * screen_cell_size)

        # Draw player with a more distinct visual
        rect = pygame.Rect(screen_x, screen_y, screen_cell_size, screen_cell_size)

        # Draw a green triangle facing in movement direction with a highlight border
        pygame.draw.rect(surface, COLOR_BG, rect)  # Clear background first

        # Triangle shape changes size with zoom
        half_size = screen_cell_size // 2
        quarter_size = screen_cell_size // 4

        # Draw player as a diamond/ship shape
        ship_points = [
            (screen_x + half_size, screen_y),  # Top
            (screen_x + screen_cell_size, screen_y + half_size),  # Right
            (screen_x + half_size, screen_y + screen_cell_size),  # Bottom
            (screen_x, screen_y + half_size),  # Left
        ]
        pygame.draw.polygon(surface, COLOR_PLAYER, ship_points)

        # Add a highlight in the center
        center_point = (screen_x + half_size, screen_y + half_size)
        pygame.draw.circle(surface, (220, 255, 220), center_point, quarter_size)

        # Draw border
        pygame.draw.polygon(surface, (150, 255, 150), ship_points, 2)

    def zoom_camera(self, field: AsteroidField, zoom_factor: float) -> None:
        """Change camera zoom level by the specified factor."""
        new_zoom = field.zoom * zoom_factor
        # Limit zoom range
        if 0.25 <= new_zoom <= 4.0:
            field.zoom = new_zoom

    def feed_symbiotes(self, field: AsteroidField, minerals: int) -> int:
        """Feed minerals to symbiotes to reduce their aggression"""
        if self.currency < minerals:
            minerals = self.currency  # Can't feed more than you have

        minerals_per_race = minerals // max(1, len(field.races))
        total_fed = 0

        for race in field.races:
            race.update_hunger(minerals_per_race)
            race.fed_this_turn = True
            total_fed += minerals_per_race

        self.currency -= total_fed
        return total_fed

    def update_fleet(self, field: AsteroidField) -> Dict[str, int]:
        """
        Update the mining fleet status and check for symbiote attacks
        Returns dictionary with damage and lost ships information
        """
        results = {"damage_taken": 0, "ships_lost": 0, "minerals_mined": 0}

        # Process each ship
        for i in range(len(self.ship_positions) - 1, -1, -1):
            ship_x, ship_y = self.ship_positions[i]

            # Check for nearby symbiotes that could attack
            attack_chance = 0.0
            for race in field.races:
                # Only hungry symbiotes attack
                if race.hunger > 0.6 and not race.fed_this_turn:
                    # Check cells around ship
                    for dy, dx in itertools.product(range(-3, 4), range(-3, 4)):
                        nx, ny = ship_x + dx, ship_y + dy
                        if (
                            0 <= nx < field.width
                            and 0 <= ny < field.height
                            and field.entity_grid[ny, nx] == race.race_id
                        ):
                            dist = max(1, abs(dx) + abs(dy))
                            attack_chance += race.hunger * (4 - dist) * 0.05

            # Check if ship is attacked
            if random.random() < attack_chance:
                # Ship takes damage
                damage = random.randint(5, 20)
                self.ship_health[i] -= damage
                results["damage_taken"] += damage

                # Check if ship is destroyed
                if self.ship_health[i] <= 0:
                    self.ship_positions.pop(i)
                    self.ship_health.pop(i)
                    self.mining_ships -= 1
                    results["ships_lost"] += 1

                    # Show attack notification
                    logging.info(
                        f"Ship at ({ship_x}, {ship_y}) destroyed by symbiotes!"
                    )

        # Update main ship position
        if self.mining_ships > 0:
            self.ship_positions[0] = (self.x, self.y)

        # Each ship mines nearby asteroids
        total_mined = 0
        for ship_x, ship_y in self.ship_positions:
            for dy, dx in itertools.product(
                range(-self.mining_range, self.mining_range + 1),
                range(-self.mining_range, self.mining_range + 1),
            ):
                nx, ny = ship_x + dx, ship_y + dy
                if (
                    0 <= nx < field.width
                    and 0 <= ny < field.height
                    and (field.grid[ny, nx] > 0)
                ):
                    value = field.grid[ny, nx]
                    if field.rare_grid[ny, nx] == 1:
                        self.total_rare_mined += 1

                    total_mined += value
                    field.grid[ny, nx] = 0
                    field.rare_grid[ny, nx] = 0

        # Calculate mining reward
        reward = int(total_mined * self.mining_efficiency)
        self.currency += reward
        self.total_mined += reward
        results["minerals_mined"] = reward

        return results


# -------------------------------------
# Shop Class (Upgrades)
# -------------------------------------
class Shop:
    """
    The shop offers upgrade options that affect both your mining ship and the asteroid field.
    """

    def __init__(self) -> None:
        self.options: List[Dict[str, Any]] = [
            {
                "name": "Upgrade Mining Efficiency (+0.1)",
                "cost": 50,
                "description": "Earn more per asteroid mined.",
                "action": lambda player, field: setattr(
                    player, "mining_efficiency", player.mining_efficiency + 0.1
                ),
                "category": "ship",
            },
            {
                "name": "Upgrade Mining Range (+1)",
                "cost": 75,
                "description": "Mine a larger area.",
                "action": lambda player, field: setattr(
                    player, "mining_range", player.mining_range + 1
                ),
                "category": "ship",
            },
            {
                "name": "Buy Auto-Mining Drone",
                "cost": 150,
                "description": "Automate mining (50% efficient).",
                "action": lambda player, field: setattr(
                    player, "auto_miners", player.auto_miners + 1
                ),
                "category": "ship",
            },
            {
                "name": "Manual Asteroid Seeding",
                "cost": 30,
                "description": "Seed a new asteroid cluster at your location.",
                "action": lambda player, field: field.manual_seed(
                    player.x, player.y, radius=3
                ),
                "category": "field",
            },
            {
                "name": "Field Control: Add Birth Option (2 Neighbors)",
                "cost": 100,
                "description": "A dead cell will spawn an asteroid if it has 2 neighbors.",
                "action": lambda player, field: field.birth_set.add(2),
                "category": "field",
            },
            {
                "name": "Field Control: Increase Regen Rate (+1%)",
                "cost": 120,
                "description": "Increase chance for empty cells to spawn asteroids.",
                "action": lambda player, field: setattr(
                    field, "regen_rate", min(field.regen_rate + 0.01, 1.0)
                ),
                "category": "field",
            },
            {
                "name": "Enhance Rare Asteroids (+5%)",
                "cost": 200,
                "description": "Increase chance for asteroids to be rare.",
                "action": lambda player, field: setattr(
                    field, "rare_chance", min(field.rare_chance + 0.05, 1.0)
                ),
                "category": "field",
            },
            {
                "name": "Discover Blue Race",
                "cost": 300,
                "description": "Discover the blue mining race.",
                "action": lambda player, field: self.discover_race(field, 0),
                "category": "race",
            },
            {
                "name": "Discover Magenta Race",
                "cost": 450,
                "description": "Discover the magenta mining race.",
                "action": lambda player, field: self.discover_race(field, 1),
                "category": "race",
            },
            {
                "name": "Discover Orange Race",
                "cost": 600,
                "description": "Discover the orange mining race.",
                "action": lambda player, field: self.discover_race(field, 2),
                "category": "race",
            },
        ]
        self.current_category = "ship"  # Start with ship upgrades
        self.categories = ["ship", "field", "race"]
        self.scroll_offset = 0
        self.max_visible_items = 7

    def discover_race(self, field: AsteroidField, race_idx: int) -> None:
        """Add a new alien race to the simulation"""
        try:
            # Access the global game instance
            import gc

            game_instance = next(
                (obj for obj in gc.get_objects() if isinstance(obj, Game)), None
            )
            logging.info(f"Discovering race {race_idx}")
            if game_instance:
                logging.info(
                    f"Found game instance, available races: {len(game_instance.available_races)}"
                )
            else:
                logging.error("No game instance found")

            if game_instance and race_idx < len(game_instance.available_races):
                race = game_instance.available_races[race_idx]
                if race not in field.races:
                    logging.info(f"Adding race {race_idx + 1} to field")
                    field.races.append(race)
                    race.populate(field)
                    logging.info(
                        f"Race {race_idx + 1} discovered and added to simulation"
                    )
                else:
                    logging.info(f"Race {race_idx + 1} already discovered")
            else:
                logging.error(f"Could not discover race at index {race_idx}")
        except Exception as e:
            logging.error(f"Error in discover_race: {str(e)}")
            import traceback

            logging.error(traceback.format_exc())

    def get_filtered_options(self) -> List[Dict[str, Any]]:
        """Return options filtered by current category"""
        return [opt for opt in self.options if opt["category"] == self.current_category]

    def draw(self, surface: pygame.Surface, player: Player) -> None:
        # Draw semi-transparent panel
        panel = pygame.Surface((WINDOW_WIDTH - 100, WINDOW_HEIGHT - 100))
        panel.set_alpha(240)
        panel.fill(COLOR_SHOP_BG)
        surface.blit(panel, (50, 50))

        # Draw header
        draw_text(surface, "SHOP", 60, 60, size=28)
        draw_text(surface, "Press ESC to Exit", WINDOW_WIDTH - 250, 60, size=20)

        # Draw category tabs
        tab_width = 150
        tab_height = 30
        tab_y = 95
        for i, category in enumerate(self.categories):
            tab_x = 60 + i * (tab_width + 10)
            tab_rect = pygame.Rect(tab_x, tab_y, tab_width, tab_height)

            if self.current_category == category:
                # Active tab
                pygame.draw.rect(surface, (80, 80, 100), tab_rect)
                color = (255, 255, 255)
            else:
                # Inactive tab
                pygame.draw.rect(surface, (50, 50, 70), tab_rect)
                color = (200, 200, 200)

            pygame.draw.rect(surface, (100, 100, 120), tab_rect, 2)  # Border
            category_name = f"{category.capitalize()} Upgrades"
            draw_text(
                surface, category_name, tab_x + 10, tab_y + 5, size=18, color=color
            )

        # Draw current credits
        draw_text(
            surface, f"Credits: {player.currency}", WINDOW_WIDTH - 250, 100, size=24
        )

        # Draw options
        filtered_options = self.get_filtered_options()
        start_y = 150
        item_height = 60

        # Draw scrollbar if needed
        visible_items = min(self.max_visible_items, len(filtered_options))
        if len(filtered_options) > self.max_visible_items:
            scrollbar_height = (
                (WINDOW_HEIGHT - 200) * visible_items / len(filtered_options)
            )
            scrollbar_y = 150 + (
                WINDOW_HEIGHT - 200 - scrollbar_height
            ) * self.scroll_offset / (len(filtered_options) - visible_items)
            scrollbar_rect = pygame.Rect(
                WINDOW_WIDTH - 70, scrollbar_y, 10, scrollbar_height
            )
            pygame.draw.rect(surface, (100, 100, 120), scrollbar_rect)

        # Draw items
        for i in range(
            self.scroll_offset,
            min(self.scroll_offset + self.max_visible_items, len(filtered_options)),
        ):
            opt = filtered_options[i]
            y_pos = start_y + (i - self.scroll_offset) * item_height

            # Item background
            item_rect = pygame.Rect(60, y_pos, WINDOW_WIDTH - 120, item_height - 5)
            bg_color = (40, 40, 60) if player.currency >= opt["cost"] else (60, 30, 30)
            pygame.draw.rect(surface, bg_color, item_rect)
            pygame.draw.rect(surface, (100, 100, 120), item_rect, 2)  # Border

            # Item content
            key_num = i + 1
            if key_num <= 9:  # Only show hotkeys for first 9 items
                key_rect = pygame.Rect(70, y_pos + 5, 30, 30)
                pygame.draw.rect(surface, (80, 80, 100), key_rect)
                draw_text(surface, str(key_num), 82, y_pos + 10, size=20)

            # Item details
            draw_text(surface, opt["name"], 110, y_pos + 5, size=20)
            draw_text(
                surface,
                opt["description"],
                110,
                y_pos + 30,
                size=16,
                color=(180, 180, 180),
            )

            # Cost button
            cost_rect = pygame.Rect(WINDOW_WIDTH - 200, y_pos + 15, 120, 30)
            if player.currency >= opt["cost"]:
                pygame.draw.rect(
                    surface, (40, 70, 40), cost_rect
                )  # Green if affordable
            else:
                pygame.draw.rect(
                    surface, (70, 40, 40), cost_rect
                )  # Red if too expensive

            pygame.draw.rect(surface, (100, 100, 120), cost_rect, 2)  # Border
            draw_text(
                surface, f"Cost: {opt['cost']}", WINDOW_WIDTH - 185, y_pos + 20, size=16
            )

        # Draw instructions
        instructions = [
            "Press 1-9 to purchase an upgrade",
            "Press Tab to switch categories",
            "Use Up/Down arrows to scroll",
        ]
        for i, instr in enumerate(instructions):
            draw_text(
                surface,
                instr,
                60,
                WINDOW_HEIGHT - 100 + i * 25,
                size=16,
                color=(180, 180, 180),
            )

    # Fix the forward reference by using a string annotation
    def handle_event(
        self,
        event: pygame.event.Event,
        player: Player,
        field: AsteroidField,
        notifier: "NotificationManager",
    ) -> bool:
        """Handle shop-related events. Returns True if the shop should close."""
        if event.type == pygame.KEYDOWN:
            filtered_options = self.get_filtered_options()

            if event.key == pygame.K_ESCAPE:
                return True  # Close shop

            elif event.key == pygame.K_TAB:
                # Switch category
                current_index = self.categories.index(self.current_category)
                next_index = (current_index + 1) % len(self.categories)
                self.current_category = self.categories[next_index]
                self.scroll_offset = 0  # Reset scroll when changing category

            elif event.key == pygame.K_UP:
                # Scroll up
                self.scroll_offset = max(0, self.scroll_offset - 1)

            elif event.key == pygame.K_DOWN:
                # Scroll down
                if len(filtered_options) > self.max_visible_items:
                    self.scroll_offset = min(
                        len(filtered_options) - self.max_visible_items,
                        self.scroll_offset + 1,
                    )

            # Handle number keys for purchase
            elif pygame.K_1 <= event.key <= pygame.K_9:
                idx = event.key - pygame.K_1 + self.scroll_offset
                if idx < len(filtered_options):
                    upg = filtered_options[idx]
                    if player.currency >= upg["cost"]:
                        player.currency -= upg["cost"]
                        try:
                            upg["action"](player, field)
                            notifier.add(f"Purchased: {upg['name']}")
                        except Exception as e:
                            logging.error(f"Upgrade error: {e}")
                            notifier.add(f"Error: {str(e)}", color=COLOR_HIGHLIGHT)
                    else:
                        notifier.add("Not enough credits!", color=COLOR_HIGHLIGHT)

        return False  # Keep shop open


# -------------------------------------
# Notification Manager Class
# -------------------------------------
class NotificationManager:
    """
    Displays on-screen notifications in a scrollable side panel.
    Each notification is a list: [text, duration, color]
    """

    def __init__(self) -> None:
        self.notifications: List[List[Any]] = []
        self.max_visible_notifications = 10  # Max number of notifications shown
        self.scroll_offset = 0
        self.panel_width = 300
        self.panel_height = WINDOW_HEIGHT - 40
        self.panel_x = 10
        self.panel_y = 20
        self.fade_timer = 0
        self.show_full_panel = False  # Toggle between compact and expanded view

        # Control tooltips
        self.tooltips = [
            {"key": "S", "description": "Open Shop"},
            {"key": "R", "description": "Seed Asteroids"},
            {"key": "SPACE", "description": "Mine"},
            {"key": "↑↓←→", "description": "Move Ship"},
            {"key": "TAB", "description": "Switch Category (in Shop)"},
            {"key": "ESC", "description": "Close Shop"},
            {"key": "N", "description": "Toggle Notification Panel"},
        ]

    def add(
        self, text: str, duration: int = 120, color: Tuple[int, int, int] = COLOR_EVENT
    ) -> None:
        self.notifications.append([text, duration, color])
        logging.info(f"Notification: {text}")

        # Auto-scroll to bottom when new notifications arrive
        if len(self.notifications) > self.max_visible_notifications:
            self.scroll_offset = (
                len(self.notifications) - self.max_visible_notifications
            )

    def update(self) -> None:
        for note in self.notifications:
            note[1] -= 1
        self.notifications = [n for n in self.notifications if n[1] > 0]

        # Update fade timer for tooltip animation
        self.fade_timer = (self.fade_timer + 1) % 240

    def draw_tooltips(self, surface: pygame.Surface) -> None:
        """Draw animated tooltips at the top of the screen"""
        tooltip_y = 10
        tooltip_spacing = 80

        # Calculate opacity for fading effect
        opacity = abs(120 - (self.fade_timer % 240)) + 135  # Values between 135-255

        for i, tip in enumerate(self.tooltips):
            # Create tooltip background
            tooltip_x = 330 + (i * tooltip_spacing)

            # Create semi-transparent surface for the tooltip
            tooltip_surface = pygame.Surface((70, 30), pygame.SRCALPHA)
            tooltip_surface.fill((40, 40, 50, opacity))

            # Draw border
            pygame.draw.rect(
                tooltip_surface, (100, 100, 140, opacity), pygame.Rect(0, 0, 70, 30), 2
            )

            # Key part
            key_rect = pygame.Rect(3, 3, 24, 24)
            pygame.draw.rect(tooltip_surface, (60, 60, 80, opacity), key_rect)
            pygame.draw.rect(tooltip_surface, (100, 100, 140, opacity), key_rect, 1)

            # Add the text to the surface
            font = pygame.font.SysFont("Arial", 14)
            key_text = font.render(tip["key"], True, (220, 220, 220, opacity))
            tooltip_surface.blit(
                key_text,
                (
                    key_rect.centerx - key_text.get_width() // 2,
                    key_rect.centery - key_text.get_height() // 2,
                ),
            )

            desc_text = font.render(tip["description"], True, (220, 220, 220, opacity))
            tooltip_surface.blit(desc_text, (30, 8))

            # Blit the tooltip to the main surface
            surface.blit(tooltip_surface, (tooltip_x, tooltip_y))

    def draw(self, surface: pygame.Surface) -> None:
        """Draw scrollable notification panel and control tooltips"""
        # Draw tooltips
        self.draw_tooltips(surface)

        # Draw notification panel
        if not self.notifications:
            return

        # Create a semi-transparent panel surface
        panel_width = 300 if self.show_full_panel else 40
        panel = pygame.Surface((panel_width, self.panel_height), pygame.SRCALPHA)
        panel.fill((30, 30, 40, 200))  # Semi-transparent dark background

        # Draw panel border
        pygame.draw.rect(
            panel,
            (80, 80, 100, 255),
            pygame.Rect(0, 0, panel_width, self.panel_height),
            2,
        )

        if self.show_full_panel:
            # Draw header
            header_rect = pygame.Rect(0, 0, panel_width, 30)
            pygame.draw.rect(panel, (50, 50, 70, 230), header_rect)
            pygame.draw.line(panel, (100, 100, 140, 255), (0, 30), (panel_width, 30), 2)

            font = pygame.font.SysFont("Arial", 16)
            header_text = font.render("NOTIFICATIONS", True, (220, 220, 220))
            panel.blit(header_text, (10, 7))

            # Draw notifications
            visible_count = min(len(self.notifications), self.max_visible_notifications)

            # Draw scrollbar if needed
            if len(self.notifications) > self.max_visible_notifications:
                scrollbar_height = (
                    (self.panel_height - 40) * visible_count / len(self.notifications)
                )
                scrollbar_y = 35 + (
                    self.panel_height - 40 - scrollbar_height
                ) * self.scroll_offset / (len(self.notifications) - visible_count)
                scrollbar_rect = pygame.Rect(
                    panel_width - 10, scrollbar_y, 6, scrollbar_height
                )
                pygame.draw.rect(panel, (100, 100, 140, 180), scrollbar_rect)

            # Draw notification entries
            start_idx = min(self.scroll_offset, len(self.notifications) - visible_count)
            end_idx = start_idx + visible_count

            for i, note in enumerate(self.notifications[start_idx:end_idx]):
                y_pos = 35 + (i * 25)

                # Calculate fade based on remaining duration
                alpha = min(255, max(100, int(note[1] * 2.5)))

                # Draw text with alpha
                text_color = (note[2][0], note[2][1], note[2][2], alpha)
                text_surface = font.render(note[0], True, text_color)

                # Wrap text if too long
                if text_surface.get_width() > panel_width - 20:
                    words = note[0].split(" ")
                    lines = []
                    current_line = words[0]

                    for word in words[1:]:
                        test_line = f"{current_line} {word}"
                        test_surface = font.render(test_line, True, text_color)

                        if test_surface.get_width() <= panel_width - 20:
                            current_line = test_line
                        else:
                            lines.append(current_line)
                            current_line = word

                    lines.append(current_line)

                    for j, line in enumerate(lines):
                        line_surface = font.render(line, True, text_color)
                        panel.blit(line_surface, (10, y_pos + j * 20))
                else:
                    panel.blit(text_surface, (10, y_pos))
        else:
            # Draw compact icon-only view
            font = pygame.font.SysFont("Arial", 20)
            text_surface = font.render("!", True, (220, 220, 220))
            panel.blit(text_surface, (15, 10))

        # Blit the panel to the main surface
        surface.blit(panel, (self.panel_x, self.panel_y))

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle notification panel events"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:  # N key toggles notifications panel
                self.show_full_panel = not self.show_full_panel

            # Scrolling when panel is open
            if self.show_full_panel:
                if event.key == pygame.K_PAGEUP:
                    self.scroll_offset = max(
                        0, self.scroll_offset - self.max_visible_notifications
                    )
                elif event.key == pygame.K_PAGEDOWN:
                    max_offset = max(
                        0, len(self.notifications) - self.max_visible_notifications
                    )
                    self.scroll_offset = min(
                        max_offset, self.scroll_offset + self.max_visible_notifications
                    )

        # Mouse wheel scrolling when hovering over the panel
        elif event.type == pygame.MOUSEWHEEL and self.show_full_panel:
            mouse_pos = pygame.mouse.get_pos()
            panel_rect = pygame.Rect(
                self.panel_x, self.panel_y, self.panel_width, self.panel_height
            )
            if panel_rect.collidepoint(mouse_pos):
                self.scroll_offset = max(
                    0,
                    min(
                        len(self.notifications) - self.max_visible_notifications,
                        self.scroll_offset - event.y,
                    ),
                )


# -------------------------------------
# Utility Function: Draw Text
# -------------------------------------
def draw_text(
    surface: pygame.Surface,
    text: str,
    x: int,
    y: int,
    size: int = 18,
    color: Tuple[int, int, int] = COLOR_TEXT,
) -> None:
    try:
        font = pygame.font.SysFont("Arial", size)
    except Exception:
        font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, (x, y))


# -------------------------------------
# Game Class (Main Engine)
# -------------------------------------
class Game:
    def __init__(self) -> None:
        pygame.init()
        self.screen: pygame.Surface = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Mining Asteroids – Procedural Generation Edition")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.field: AsteroidField = AsteroidField()
        self.player: Player = Player()
        self.shop: Shop = Shop()
        self.notifier: NotificationManager = NotificationManager()
        self.state: str = STATE_PLAY  # Either STATE_PLAY or STATE_SHOP
        self.frame_counter: int = 0
        self.update_interval: int = 10  # Update field every 10 frames

        # Game options
        self.show_minimap = True  # Toggle for minimap
        self.auto_mine = False  # Toggle for auto-mining

        # Initialize races
        self.available_races = [
            MinerEntity(
                1,
                COLOR_RACE_1,
                birth_set={2, 3},
                survival_set={3, 4},
                initial_density=0.005,
            ),
            MinerEntity(
                2,
                COLOR_RACE_2,
                birth_set={3, 4},
                survival_set={2, 3},
                initial_density=0.005,
            ),
            MinerEntity(
                3,
                COLOR_RACE_3,
                birth_set={1, 5},
                survival_set={1, 4},
                initial_density=0.005,
            ),
        ]

        for race in self.available_races:
            logging.info(f"Created race {race.race_id} with trait {race.trait}")

        # Add initial helpful notifications
        self.notifier.add("Welcome to Mining Asteroids!", duration=240)
        self.notifier.add("Press N to toggle notification panel", duration=240)
        self.notifier.add("Press S to open the shop", duration=240)
        self.notifier.add("Use + and - to zoom in/out", duration=240)
        self.notifier.add("Press M to toggle minimap", duration=240)

        # Toggle notification panel on by default for first run
        self.notifier.show_full_panel = True

        self.initialize_visual_settings()

    def initialize_visual_settings(self):
        """Initialize visual filter settings"""
        self.visual_filters = {
            "show_grid": False,
            "show_all_asteroids": True,
            "show_symbiotes": True,
            "show_mining_radius": False,
            "show_symbiote_colonies": True,
            "show_mineral_values": False,
            "show_animations": True,
            "detail_level": "medium",  # low, medium, high
        }

        # Add keyboard shortcuts for toggling filters
        self.filter_shortcuts = {
            pygame.K_g: "show_grid",
            pygame.K_a: "show_all_asteroids",
            pygame.K_c: "show_symbiote_colonies",
            pygame.K_v: "show_mineral_values",
            pygame.K_d: "cycle_detail",  # Cycles through detail levels
        }

    def handle_filter_keys(self, key):
        """Handle visual filter key presses"""
        if key in self.filter_shortcuts:
            filter_name = self.filter_shortcuts[key]
            if filter_name == "cycle_detail":
                # Cycle through detail levels
                detail_levels = ["low", "medium", "high"]
                current_idx = detail_levels.index(self.visual_filters["detail_level"])
                next_idx = (current_idx + 1) % len(detail_levels)
                self.visual_filters["detail_level"] = detail_levels[next_idx]
                self.add_notification(
                    f"Detail level: {self.visual_filters['detail_level']}"
                )
            else:
                # Toggle boolean filter
                self.visual_filters[filter_name] = not self.visual_filters[filter_name]
                self.add_notification(
                    f"{filter_name.replace('_', ' ').title()}: {'On' if self.visual_filters[filter_name] else 'Off'}"
                )

    def auto_upgrade_logic(self) -> None:
        """
        If auto-upgrade mode is enabled, automatically purchase any upgrade you can afford.
        (This simple strategy checks shop options in order.)
        """
        if self.player.auto_upgrade:
            for upg in self.shop.options:
                if self.player.currency >= upg["cost"]:
                    self.player.currency -= upg["cost"]
                    try:
                        upg["action"](self.player, self.field)
                    except Exception as e:
                        logging.error(f"Auto-upgrade error: {e}")
                    self.notifier.add(f"Auto-Purchased: {upg['name']}")

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Let the notification manager handle its events
            self.notifier.handle_event(event)

            if event.type == pygame.KEYDOWN:
                if self.state == STATE_PLAY:
                    if event.key == pygame.K_s:
                        self.state = STATE_SHOP
                    elif event.key == pygame.K_SPACE:
                        reward: int = self.player.mine(self.field)
                        self.notifier.add(f"Mined asteroids: +{reward} credits")
                    elif event.key == pygame.K_r:
                        # Manual seeding: seed asteroids at player's location.
                        self.field.manual_seed(self.player.x, self.player.y, radius=5)
                        self.notifier.add("Manual seeding activated.")
                    elif event.key == pygame.K_LEFT:
                        self.player.move(-self.player.move_speed, 0, self.field)
                    elif event.key == pygame.K_RIGHT:
                        self.player.move(self.player.move_speed, 0, self.field)
                    elif event.key == pygame.K_UP:
                        self.player.move(0, -self.player.move_speed, self.field)
                    elif event.key == pygame.K_DOWN:
                        self.player.move(0, self.player.move_speed, self.field)
                    elif event.key in [pygame.K_EQUALS, pygame.K_PLUS]:
                        self.player.zoom_camera(self.field, 1.25)
                        self.notifier.add("Zoomed in", duration=60)
                    elif event.key == pygame.K_MINUS:
                        self.player.zoom_camera(self.field, 0.8)
                        self.notifier.add("Zoomed out", duration=60)
                    elif event.key == pygame.K_w:
                        for _ in range(10):
                            self.player.move(0, -self.player.move_speed, self.field)
                        self.notifier.add("Fast move up", duration=30)
                    elif event.key == pygame.K_a:
                        for _ in range(10):
                            self.player.move(-self.player.move_speed, 0, self.field)
                        self.notifier.add("Fast move left", duration=30)
                    elif (
                        event.key == pygame.K_s
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    ):
                        for _ in range(10):
                            self.player.move(0, self.player.move_speed, self.field)
                        self.notifier.add("Fast move down", duration=30)
                    elif event.key == pygame.K_d:
                        for _ in range(10):
                            self.player.move(self.player.move_speed, 0, self.field)
                        self.notifier.add("Fast move right", duration=30)
                    elif (
                        event.key == pygame.K_m
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    ):
                        if hasattr(self, "auto_mine") and self.auto_mine:
                            self.auto_mine = False
                            self.notifier.add("Auto-mining disabled", duration=120)
                        else:
                            self.auto_mine = True
                            self.notifier.add("Auto-mining enabled", duration=120)
                    elif (
                        event.key == pygame.K_a
                        and pygame.key.get_mods() & pygame.KMOD_SHIFT
                    ):
                        if (
                            hasattr(self.player, "auto_upgrade")
                            and self.player.auto_upgrade
                        ):
                            self.player.auto_upgrade = False
                            self.notifier.add("Auto-upgrades disabled", duration=120)
                        else:
                            self.player.auto_upgrade = True
                            self.notifier.add("Auto-upgrades enabled", duration=120)
                    elif event.key == pygame.K_f:
                        # Feed a small amount
                        minerals_fed = self.player.feed_symbiotes(self.field, 50)
                        self.notifier.add(f"Fed symbiotes: {minerals_fed} minerals")

                    elif event.key == pygame.K_g:
                        # Feed a large amount
                        minerals_fed = self.player.feed_symbiotes(self.field, 200)
                        self.notifier.add(f"Fed symbiotes: {minerals_fed} minerals")

                    elif event.key == pygame.K_b:
                        # Build a new mining ship if resources allow
                        if (
                            self.player.currency >= self.player.ship_cost
                            and self.player.mining_ships < self.player.max_mining_ships
                        ):
                            self.player.currency -= self.player.ship_cost
                            self.player.mining_ships += 1
                            self.player.ship_positions.append(
                                (self.player.x, self.player.y)
                            )
                            self.player.ship_health.append(100)
                            self.notifier.add(
                                f"New mining ship built! Fleet: {self.player.mining_ships}"
                            )
                        elif self.player.mining_ships >= self.player.max_mining_ships:
                            self.notifier.add(
                                "Maximum fleet size reached!", color=COLOR_HIGHLIGHT
                            )
                        else:
                            self.notifier.add(
                                f"Not enough credits! Need {self.player.ship_cost}",
                                color=COLOR_HIGHLIGHT,
                            )
                elif self.state == STATE_SHOP:
                    # Use the shop's event handler for all shop interactions
                    if self.shop.handle_event(
                        event, self.player, self.field, self.notifier
                    ):
                        self.state = STATE_PLAY

            elif event.type == pygame.MOUSEWHEEL:
                if self.state == STATE_PLAY:
                    zoom_factor = 1.1 if event.y > 0 else 0.9
                    self.player.zoom_camera(self.field, zoom_factor)

    def update(self) -> None:
        self.frame_counter += 1
        if self.frame_counter % self.update_interval == 0:
            self.field.update()

            # Update player's mining fleet
            fleet_results = self.player.update_fleet(self.field)

            if fleet_results["minerals_mined"] > 0:
                self.notifier.add(
                    f"Mining fleet income: +{fleet_results['minerals_mined']} minerals",
                    duration=60,
                )

            if fleet_results["damage_taken"] > 0:
                self.notifier.add(
                    f"Fleet attacked! Damage taken: {fleet_results['damage_taken']}",
                    color=COLOR_HIGHLIGHT,
                    duration=60,
                )

            if fleet_results["ships_lost"] > 0:
                self.notifier.add(
                    f"ALERT: {fleet_results['ships_lost']} ships lost to symbiote attacks!",
                    color=(255, 0, 0),
                    duration=120,
                )

            # Report on symbiote races
            for race in self.field.races:
                race_count = np.sum(self.field.entity_grid == race.race_id)
                hunger_status = (
                    "Starving"
                    if race.hunger > 0.8
                    else "Hungry" if race.hunger > 0.4 else "Satiated"
                )

                self.notifier.add(
                    f"Race {race.race_id}: {race_count} symbiotes, Status: {hunger_status}",
                    duration=60,
                    color=race.color,
                )

            # Auto-upgrade check if enabled
            self.auto_upgrade_logic()

        self.notifier.update()

    def draw(self) -> None:
        self.screen.fill(COLOR_BG)

        # First draw the asteroid field
        self.field.draw(self.screen)

        # Draw player on top, passing the field for camera calculations
        self.player.draw(self.screen, self.field)

        # Draw minimap if enabled
        if self.show_minimap:
            self.draw_minimap(self.screen)

        # Status panel in the bottom right
        status_panel = pygame.Surface((350, 220), pygame.SRCALPHA)
        status_panel.fill((20, 20, 30, 180))
        pygame.draw.rect(
            status_panel, (80, 80, 100, 255), pygame.Rect(0, 0, 350, 220), 2
        )

        # Draw stats in the panel
        draw_text(status_panel, f"Credits: {self.player.currency}", 10, 10, size=20)
        draw_text(
            status_panel,
            f"Efficiency: {self.player.mining_efficiency:.2f}",
            10,
            35,
            size=18,
        )
        draw_text(status_panel, f"Range: {self.player.mining_range}", 10, 60, size=18)
        draw_text(
            status_panel, f"Auto-Miners: {self.player.auto_miners}", 10, 85, size=18
        )

        # Show field stats
        field_stats_x = 180
        draw_text(
            status_panel,
            f"Asteroids: {np.sum(self.field.grid > 0)}",
            field_stats_x,
            10,
            size=18,
        )
        draw_text(
            status_panel,
            f"Rares: {np.sum(self.field.rare_grid)}",
            field_stats_x,
            35,
            size=18,
        )
        draw_text(
            status_panel, f"Zoom: {self.field.zoom:.2f}x", field_stats_x, 60, size=18
        )
        draw_text(
            status_panel,
            f"Grid: {self.field.width}x{self.field.height}",
            field_stats_x,
            85,
            size=18,
        )

        # Draw race information
        y_offset = 110
        for race in self.field.races:
            race_count = np.sum(self.field.entity_grid == race.race_id)
            text = f"Race {race.race_id}: {race_count} miners, +{race.last_income}"
            draw_text(status_panel, text, 10, y_offset, size=16, color=race.color)
            y_offset += 25

            # Draw evolution progress bar
            if race.evolution_stage < 5:  # Max stage
                progress = race.evolution_points / race.evolution_threshold
                bar_width = 200
                progress_width = int(bar_width * progress)

                # Background bar
                pygame.draw.rect(
                    status_panel, (50, 50, 60), pygame.Rect(10, y_offset, bar_width, 5)
                )
                # Progress bar
                pygame.draw.rect(
                    status_panel,
                    race.color,
                    pygame.Rect(10, y_offset, progress_width, 5),
                )

                # Stage indicator
                draw_text(
                    status_panel,
                    f"Stage {race.evolution_stage}",
                    220,
                    y_offset - 5,
                    size=14,
                )
                y_offset += 15
        # Calculate scale factors for minimap
        scale_x = MINIMAP_SIZE / self.field.width
        scale_y = MINIMAP_SIZE / self.field.height

        # Draw viewport rectangle
        view_x1, view_y1, view_x2, view_y2 = self.field.get_view_bounds()
        view_rect = pygame.Rect(
            int(view_x1 * scale_x),
            int(view_y1 * scale_y),
            int((view_x2 - view_x1) * scale_x),
            int((view_y2 - view_y1) * scale_y),
        )
        pygame.draw.rect(status_panel, (100, 255, 100), view_rect, 1)

        # Draw player position
        player_x = int(self.player.x * scale_x)
        player_y = int(self.player.y * scale_y)
        pygame.draw.rect(status_panel, COLOR_PLAYER, (player_x - 1, player_y - 1, 3, 3))

        # Draw notification panel last so it's on top
        self.notifier.draw(status_panel)

        # Blit the status panel to the main screenview_x1, view_y1, view_x2, view_y2 = self.field.get_view_bounds()
        self.screen.blit(status_panel, (WINDOW_WIDTH - 360, WINDOW_HEIGHT - 230))

        # Show welcome message for the first 10 secondsy),
        if self.frame_counter < 300:  # 10 seconds at 30 FPS * scale_x),
            self.draw_welcome_panel()

        # Show coordinate info00), view_rect, 1)
        draw_text(
            self.screen,
            f"Position: ({self.player.x}, {self.player.y})",
            10,
            WINDOW_HEIGHT - 30,
            size=16,
        )

        # Shop overlay
        if self.state == STATE_SHOP:
            self.shop.draw(self.screen, self.player)

        # Draw notification panel last so it's on top
        self.notifier.draw(self.screen)

    def run(self) -> None:
        try:
            while True:
                self.handle_events()
                if self.state == STATE_PLAY:
                    self.update()
                self.draw()
                pygame.display.flip()
                self.clock.tick(FPS)
        except Exception as e:
            logging.critical(f"Error in game loop: {str(e)}")

            # Import traceback for detailed error info
            import traceback

            logging.critical(traceback.format_exc())

            # Clean exit
            pygame.quit()
            sys.exit(1)

    def draw_minimap(self, surface: pygame.Surface) -> None:
        """Draw a minimap in the bottom left corner of the screen."""
        minimap_size = MINIMAP_SIZE  # Use the constant defined at the top of the file
        minimap_border = 3
        minimap_x = 10
        minimap_y = WINDOW_HEIGHT - minimap_size - 10

        # Create minimap surface
        minimap = pygame.Surface((minimap_size, minimap_size))
        minimap.fill((20, 20, 30))

        # Calculate scale factor
        scale_x = minimap_size / self.field.width
        scale_y = minimap_size / self.field.height

        # Draw asteroids as dots
        for y in range(0, self.field.height, 4):  # Skip cells for performance
            for x in range(0, self.field.width, 4):
                if self.field.grid[y, x] > 0:
                    # Normal asteroid
                    dot_x = int(x * scale_x)
                    dot_y = int(y * scale_y)
                    if self.field.rare_grid[y, x] == 1:
                        # Rare asteroid - yellow
                        pygame.draw.rect(minimap, (255, 215, 0), (dot_x, dot_y, 2, 2))
                    else:
                        # Normal asteroid - white
                        pygame.draw.rect(minimap, (200, 200, 200), (dot_x, dot_y, 1, 1))

        # Draw races as colored dots
        for race in self.field.races:
            color = race.color
            # Sample entity grid for performance
            for y in range(0, self.field.height, 4):
                for x in range(0, self.field.width, 4):
                    if self.field.entity_grid[y, x] == race.race_id:
                        dot_x = int(x * scale_x)
                        dot_y = int(y * scale_y)
                        pygame.draw.rect(minimap, color, (dot_x, dot_y, 2, 2))

        # Draw viewport rectangle
        view_x1, view_y1, view_x2, view_y2 = self.field.get_view_bounds()
        view_rect = pygame.Rect(
            int(view_x1 * scale_x),
            int(view_y1 * scale_y),
            int((view_x2 - view_x1) * scale_x),
            int((view_y2 - view_y1) * scale_y),
        )
        pygame.draw.rect(minimap, (100, 255, 100), view_rect, 1)

        # Draw player position
        player_x = int(self.player.x * scale_x)
        player_y = int(self.player.y * scale_y)
        pygame.draw.rect(minimap, COLOR_PLAYER, (player_x - 1, player_y - 1, 3, 3))

        # Add border to minimap
        pygame.draw.rect(
            surface,
            (80, 80, 100),
            pygame.Rect(
                minimap_x - minimap_border,
                minimap_y - minimap_border,
                minimap_size + minimap_border * 2,
                minimap_size + minimap_border * 2,
            ),
        )

        # Blit minimap to screen
        surface.blit(minimap, (minimap_x, minimap_y))

        # Draw minimap title
        draw_text(surface, "MINIMAP", minimap_x, minimap_y - 25, size=16)
        draw_text(
            surface,
            "[M] to toggle",
            minimap_x + 80,
            minimap_y - 25,
            size=14,
            color=(180, 180, 180),
        )

    def draw_welcome_panel(self) -> None:
        """Draw the welcome panel that appears for the first few seconds of gameplay"""
        welcome_panel = pygame.Surface((500, 180), pygame.SRCALPHA)
        welcome_panel.fill((20, 20, 30, 220))
        pygame.draw.rect(
            welcome_panel, (100, 100, 140, 255), pygame.Rect(0, 0, 500, 180), 2
        )

        welcome_messages = [
            "Welcome to Mining Asteroids!",
            "",
            "Press [N] to toggle Notifications panel",
            "Press [S] to open Shop",
            "Press [SPACE] to mine asteroids",
            "Press [R] to seed new asteroids",
            "Use arrow keys to move your ship",
            "Use +/- to zoom in/out",
        ]

        for i, msg in enumerate(welcome_messages):
            if i == 0:
                draw_text(welcome_panel, msg, 20, 15 + (i * 25), size=22)
            else:
                draw_text(welcome_panel, msg, 20, 15 + (i * 25), size=18)

        self.screen.blit(
            welcome_panel, (WINDOW_WIDTH // 2 - 250, WINDOW_HEIGHT // 2 - 90)
        )

    def draw_race_stats(self, surface: pygame.Surface) -> None:
        """
        Draw detailed symbiote race statistics panel showing:
        - Population trends
        - Resource consumption
        - Territorial analysis
        - Genetic traits
        """
        if not self.field.races:
            return

        # Create panel
        panel_width = 500
        panel_height = 400
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((20, 20, 30, 220))
        pygame.draw.rect(
            panel, (80, 80, 100, 255), pygame.Rect(0, 0, panel_width, panel_height), 2
        )

        # Header
        draw_text(panel, "SYMBIOTE RACE ANALYTICS", 20, 10, size=22)

        # Calculate total space
        y_offset = 50
        available_width = panel_width - 40
        race_height = 100

        trait_x = 270
        trait_width = 100
        # Population history as mini-graph
        graph_width = 150
        graph_height = 40
        # Draw stats for each race
        for race in self.field.races:
            # Race header with colored background
            race_header = pygame.Rect(20, y_offset, available_width, 25)
            pygame.draw.rect(panel, tuple(c * 0.5 for c in race.color), race_header)
            pygame.draw.rect(panel, race.color, race_header, 2)

            # Race name and evolution stage
            draw_text(
                panel,
                f"Race {race.race_id}: {race.trait.capitalize()} - Evolution Stage {race.evolution_stage}",
                30,
                y_offset + 5,
                size=18,
            )

            # Key statistics
            stats_y = y_offset + 30

            # Population and mining stats
            draw_text(panel, f"Population: {race.population}", 30, stats_y, size=16)
            draw_text(
                panel,
                f"Mining Rate: {race.last_income}/cycle",
                30,
                stats_y + 20,
                size=16,
            )
            draw_text(panel, f"Hunger: {race.hunger:.2f}", 30, stats_y + 40, size=16)

            # Behavior state with icon
            behavior_colors = {
                "feeding": (100, 200, 100),
                "expanding": (100, 100, 200),
                "defensive": (200, 200, 100),
                "aggressive": (200, 100, 100),
            }
            pygame.draw.circle(
                panel,
                behavior_colors.get(race.current_behavior, (150, 150, 150)),
                (250, stats_y + 10),
                8,
            )
            draw_text(
                panel,
                f"Behavior: {race.current_behavior.capitalize()}",
                270,
                stats_y,
                size=16,
            )

            # Genome traits visualization as small bar charts
            genome_y = stats_y + 20
            for i, (trait, value) in enumerate(race.genome.items()):
                if i > 1:  # Only show a few traits
                    break
                # Normalized value (0.0-2.0 range to 0-100%)
                norm_value = min(1.0, value / 2.0)
                bar_width = int(trait_width * norm_value)

                # Draw bar
                pygame.draw.rect(
                    panel,
                    (50, 50, 60),
                    pygame.Rect(trait_x, genome_y + i * 20, trait_width, 15),
                )
                pygame.draw.rect(
                    panel,
                    race.color,
                    pygame.Rect(trait_x, genome_y + i * 20, bar_width, 15),
                )

                # Draw trait name
                trait_name = trait.replace("_", " ").capitalize()
                draw_text(
                    panel,
                    f"{trait_name}: {value:.2f}",
                    trait_x + trait_width + 10,
                    genome_y + i * 20,
                    size=14,
                )

            graph_x = 400 - graph_width // 2
            graph_y = stats_y

            # Draw graph background
            pygame.draw.rect(
                panel,
                (40, 40, 50),
                pygame.Rect(graph_x, graph_y, graph_width, graph_height),
            )
            pygame.draw.rect(
                panel,
                (80, 80, 90),
                pygame.Rect(graph_x, graph_y, graph_width, graph_height),
                1,
            )

            # Draw population history if available
            if race.population_history:
                # Normalize data for the graph
                max_pop = max(max(race.population_history), 1)
                points = []

                for i, pop in enumerate(race.population_history[-graph_width:]):
                    x = graph_x + i
                    y = graph_y + graph_height - int((pop / max_pop) * graph_height)
                    points.append((x, y))

                # Draw line graph if we have enough points
                if len(points) > 1:
                    pygame.draw.lines(panel, race.color, False, points, 2)

            # Draw territory info if available
            territory_info = race.analyze_territory(self.field)
            if territory_info["center"]:
                territory_text = f"Territory: radius {territory_info['radius']}, density {territory_info['density']:.2f}"
                draw_text(
                    panel,
                    territory_text,
                    30,
                    stats_y + 60,
                    size=14,
                    color=(180, 180, 180),
                )

            # Move to next race
            y_offset += race_height

        # Blit the panel to the main screen
        surface.blit(
            panel,
            (
                WINDOW_WIDTH // 2 - panel_width // 2,
                WINDOW_HEIGHT // 2 - panel_height // 2,
            ),
        )

    def update_economy(self):
        """Update the game's economy based on mining and symbiote activity."""
        # Calculate symbiote pressure effect on mining
        symbiote_count = sum(race.population for race in self.field.races)
        symbiote_pressure = min(0.5, symbiote_count / 1000)  # Cap at 50% reduction

        # Apply symbiote pressure to mining efficiency
        total_mining_capacity = self.player.mining_efficiency
        total_mining_capacity *= 1 - symbiote_pressure

        # Update player display
        self.player.mining_efficiency = total_mining_capacity

        # Add efficiency information to notification
        if symbiote_pressure > 0.1 and random.random() < 0.05:
            self.notifier.add(
                f"Mining reduced by {symbiote_pressure:.0%} due to symbiote activity"
            )


# -------------------------------------
# Main Entry Point
# -------------------------------------
if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}")
        pygame.quit()
        sys.exit(1)
# -------------------------------------
# Main Entry Point
# -------------------------------------
if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}")
        pygame.quit()
        sys.exit(1)

    def draw_control_panel(self, surface: pygame.Surface) -> None:
        """Draw an improved control panel with filtering options"""
        # Create panel
        panel_width = 300
        panel_height = 500
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((20, 20, 30, 220))
        pygame.draw.rect(
            panel, (80, 80, 100, 255), pygame.Rect(0, 0, panel_width, panel_height), 2
        )

        # Header
        draw_text(panel, "GAME CONTROLS", 20, 10, size=22)

        # Mining section
        y_offset = 50
        draw_text(panel, "MINING CONTROLS", 20, y_offset, size=18)
        draw_text(panel, "[SPACE] Mine asteroids", 30, y_offset + 25, size=16)
        draw_text(panel, "[R] Seed new asteroids", 30, y_offset + 50, size=16)
        draw_text(panel, "[F] Feed symbiotes", 30, y_offset + 75, size=16)

        # Navigation section
        y_offset += 110
        draw_text(panel, "NAVIGATION", 20, y_offset, size=18)
        draw_text(panel, "Arrow keys - Move ship", 30, y_offset + 25, size=16)
        draw_text(panel, "[+/-] Zoom in/out", 30, y_offset + 50, size=16)

        # Visual filters section
        y_offset += 110
        draw_text(panel, "VISUAL FILTERS", 20, y_offset + 25, size=18)
        draw_text(panel, "[1] Show asteroids", 30, y_offset + 50, size=16)
        draw_text(panel, "[2] Show rares", 30, y_offset + 75, size=16)
        draw_text(panel, "[3] Show races", 30, y_offset + 100, size=16)
        draw_text(panel, "[4] Show symbiotes", 30, y_offset + 125, size=16)
        draw_text(panel, "[5] Show fleet", 30, y_offset + 150, size=16)
        draw_text(panel, "[6] Show player", 30, y_offset + 175, size=16)
        draw_text(panel, "[7] Show minimap", 30, y_offset + 200, size=16)
        draw_text(panel, "[8] Show notifications", 30, y_offset + 225, size=16)

        # Draw panel
        surface.blit(panel, (self.width - panel_width, 0))

        # Draw notifications
        self.draw_notifications(surface)

        # Draw minimap
        self.draw_minimap(surface)

        # Draw race stats
        self.draw_race_stats(surface)

        # Draw symbiote stats
        self.draw_symbiote_stats(surface)

        # Draw fleet stats
        self.draw_fleet_stats(surface)

        # Draw player stats
        self.draw_player_stats(surface)
