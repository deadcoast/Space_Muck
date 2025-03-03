"""
Procedural Generation for Space Muck.

This module implements advanced procedural generation algorithms including:
- Multi-layered noise for asteroid distribution
- Cellular automaton for evolving field patterns
- Resource distribution using statistical models
- Symbiote race evolution using the symbiote evolution algorithm
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable

import numpy as np
import scipy.ndimage as ndimage
import scipy.stats as stats
from perlin_noise import PerlinNoise
from skimage import measure
from sklearn.cluster import KMeans

from src.config import *
from src.utils.logging_setup import (
    log_performance_start,
    log_performance_end,
    log_exception,
    LogContext,
)


class ProceduralGenerator:
    """
    Handles all procedural generation for the asteroid field and symbiote races.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the procedural generator with an optional seed.

        Args:
            seed: Random seed for reproducibility
        """
        # Set seed for reproducibility
        self.seed = seed if seed is not None else random.randint(1, 100000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Noise generators
        self.primary_noise = PerlinNoise(octaves=4, seed=self.seed)
        self.secondary_noise = PerlinNoise(octaves=6, seed=self.seed + 1)
        self.rare_noise = PerlinNoise(octaves=8, seed=self.seed + 2)
        self.anomaly_noise = PerlinNoise(octaves=12, seed=self.seed + 3)

        # Statistical parameters
        self.value_distribution = stats.lognorm(s=0.6, scale=50)  # For asteroid values
        self.cluster_count_range = (3, 8)  # Range for number of clusters
        self.cluster_size_range = (20, 80)  # Range for cluster size

        # Configure cellular automaton parameters
        self.ca_params = {
            "birth_set": {3},  # Default: cell is born if it has exactly 3 neighbors
            "survival_set": {2, 3},  # Default: cell survives with 2 or 3 neighbors
            "iterations": 3,  # Default iterations for CA evolution
            "wrap": True,  # Whether the grid wraps around edges
        }

        # Fractal parameters
        self.fractal_depth = 3  # Depth for fractal generation
        self.fractal_roughness = 0.6  # Roughness factor for fractals

        # Anomaly generation parameters
        self.anomaly_chance = 0.01  # Base chance of anomaly per cell
        self.anomaly_cluster_chance = 0.2  # Chance of anomaly clustering

        # Symbiote evolution parameters
        self.evolution_params = {
            "growth_rate": 0.2,  # Base growth rate for symbiotes
            "carrying_capacity": 100,  # Base carrying capacity
            "aggression_hunger_factor": 0.1,  # How aggression increases with hunger
            "aggression_fleet_factor": 0.05,  # How aggression responds to fleet size
            "mutation_chance": {
                "common": 0.03,  # 3% chance per common mineral
                "rare": 0.2,  # 20% chance per rare mineral
                "precious": 0.1,  # 10% chance per precious mineral
                "anomaly": 0.5,  # 50% chance per anomaly mineral
            },
            "mutation_magnitude": {
                "common": 1.05,  # 5% change
                "rare": 1.2,  # 20% change
                "precious": 1.15,  # 15% change
                "anomaly": 1.5,  # 50% change
            },
        }

    def generate_asteroid_field(
        self, width: int, height: int, density: float = 0.3
    ) -> np.ndarray:
        """
        Generate a procedural asteroid field using multiple noise layers.

        Args:
            width: Width of the field
            height: Height of the field
            density: Overall density of asteroids (0-1)

        Returns:
            np.ndarray: 2D grid with asteroid values
        """
        start_time = log_performance_start("generate_asteroid_field")

        try:
            # Initialize the grid
            grid = np.zeros((height, width), dtype=np.int32)

            # Generate primary noise layer
            primary = self._generate_noise_layer(
                width, height, self.primary_noise, scale=0.02
            )
            primary = (primary - primary.min()) / (primary.max() - primary.min())

            # Generate secondary noise for details
            secondary = self._generate_noise_layer(
                width, height, self.secondary_noise, scale=0.05
            )
            secondary = (secondary - secondary.min()) / (
                secondary.max() - secondary.min()
            )

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
            grid = self._apply_cellular_automaton(
                grid, iterations=self.ca_params["iterations"]
            )

            # Create clusters of higher-value asteroids
            grid = self._create_value_clusters(grid)

            # Generate void spaces for visual interest
            grid = self._create_void_areas(grid, count=3, max_size=50)

            # Add edge degradation for natural appearance
            grid = self._apply_edge_degradation(grid, decay_factor=0.1)

            log_performance_end("generate_asteroid_field", start_time)
            return grid

        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.random.binomial(1, density * 0.5, (height, width)) * 50

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
            height, width = grid.shape
            rare_grid = np.zeros_like(grid, dtype=np.int8)

            # No rare minerals where there are no asteroids
            asteroid_mask = grid > 0

            # Generate noise for rare mineral distribution
            rare_noise = self._generate_noise_layer(
                width, height, self.rare_noise, scale=0.03
            )
            rare_noise = (rare_noise - rare_noise.min()) / (
                rare_noise.max() - rare_noise.min()
            )

            # Generate precious noise (even rarer)
            precious_noise = self._generate_noise_layer(
                width, height, self.rare_noise, scale=0.015
            )
            precious_noise = (precious_noise - precious_noise.min()) / (
                precious_noise.max() - precious_noise.min()
            )

            # Generate anomaly noise (extremely rare)
            anomaly_noise = self._generate_noise_layer(
                width, height, self.anomaly_noise, scale=0.01
            )
            anomaly_noise = (anomaly_noise - anomaly_noise.min()) / (
                anomaly_noise.max() - anomaly_noise.min()
            )

            # Apply thresholds to determine rare mineral locations
            # Higher grid values slightly increase chances of rare minerals
            value_bonus = np.clip((grid / 100), 0, 0.2)

            # 1 = rare minerals (10% base chance)
            rare_mask = (rare_noise > (1 - rare_chance - value_bonus)) & asteroid_mask
            rare_grid[rare_mask] = 1

            # 2 = precious minerals (2% base chance)
            precious_mask = (precious_noise > 0.98 - value_bonus * 0.5) & asteroid_mask
            rare_grid[precious_mask] = 2

            # 3 = anomaly minerals (0.5% base chance)
            anomaly_mask = (anomaly_noise > 0.995 - value_bonus * 0.1) & asteroid_mask
            rare_grid[anomaly_mask] = 3

            # Create small clusters of rare minerals using connected component analysis
            for mineral_type in [1, 2, 3]:
                type_mask = rare_grid == mineral_type
                if np.sum(type_mask) > 0:
                    # Expand some rare minerals to create small clusters
                    expanded = ndimage.binary_dilation(type_mask, iterations=1)
                    # But only where there are asteroids and not already other rare minerals
                    growth_mask = expanded & asteroid_mask & (rare_grid == 0)
                    # Apply with some probability
                    growth_prob = (
                        0.3 if mineral_type == 1 else 0.2 if mineral_type == 2 else 0.1
                    )
                    random_mask = np.random.random(growth_mask.shape) < growth_prob
                    rare_grid[growth_mask & random_mask] = mineral_type

            log_performance_end("generate_rare_minerals", start_time)
            return rare_grid

        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.zeros_like(grid, dtype=np.int8)

    def generate_symbiote_rules(
        self, race_id: int, hunger: float, genome: Dict[str, float]
    ) -> Tuple[Set[int], Set[int]]:
        """
        Generate cellular automaton rules for symbiote races based on their traits.

        Args:
            race_id: ID of the race
            hunger: Current hunger level (0-1)
            genome: Dictionary of genome traits

        Returns:
            Tuple[set, set]: Birth rule set and survival rule set
        """
        # Base rule sets similar to Conway's Game of Life
        base_birth = {3}  # Default: cell born with exactly 3 neighbors
        base_survival = {2, 3}  # Default: cell survives with 2 or 3 neighbors

        # Adjust rules based on race and genome
        birth_set = set(base_birth)
        survival_set = set(base_survival)

        # Adaptability affects rule diversity
        adaptability = genome.get("adaptability", 0.5)

        # More adaptable races have more diverse rules
        if adaptability > 0.7 and random.random() < adaptability:
            birth_set.add(random.choice([1, 4, 5]))

        # Hunger affects survival conditions
        if hunger > 0.7:
            # Desperate hunger - more aggressive expansion
            survival_set.add(1)  # Can survive with just 1 neighbor (less die-off)
        elif hunger < 0.3:
            # Well-fed - more stable, less aggressive expansion
            if 1 in survival_set:
                survival_set.remove(1)

        # Metabolism affects energy efficiency
        metabolism = genome.get("metabolism_rate", 1.0)
        if metabolism > 1.2:
            # High metabolism - can survive with more neighbors (more energy efficient)
            survival_set.add(4)
        elif metabolism < 0.8:
            # Low metabolism - stricter survival conditions
            if 4 in survival_set:
                survival_set.remove(4)

        # Expansion drive affects birth conditions
        expansion = genome.get("expansion_drive", 1.0)
        if expansion > 1.3:
            # High expansion - more birth conditions
            birth_set.add(4)
        elif expansion < 0.7:
            # Low expansion - stricter birth conditions
            if len(birth_set) > 1 and 4 in birth_set:
                birth_set.remove(4)

        # Ensure basic rules always apply
        birth_set.add(3)  # Always keep the classic birth rule
        survival_set.add(2)  # Always keep the classic survival rules

        return birth_set, survival_set

    def place_symbiote_race(
        self,
        width: int,
        height: int,
        race_id: int,
        initial_density: float = 0.005,
        pattern: str = "random",
    ) -> np.ndarray:
        """
        Generate initial placement for a symbiote race.

        Args:
            width: Grid width
            height: Grid height
            race_id: Race identifier
            initial_density: Initial population density
            pattern: Placement pattern ('random', 'clusters', 'network')

        Returns:
            np.ndarray: Grid with symbiote placements
        """
        grid = np.zeros((height, width), dtype=np.int32)

        if pattern == "clusters":
            # Create clusters of symbiotes
            num_clusters = random.randint(2, 5)
            for _ in range(num_clusters):
                center_x = random.randint(0, width - 1)
                center_y = random.randint(0, height - 1)
                radius = random.randint(10, 30)

                # Create each cluster with a density falloff from center
                for y in range(
                    max(0, center_y - radius), min(height, center_y + radius + 1)
                ):
                    for x in range(
                        max(0, center_x - radius), min(width, center_x + radius + 1)
                    ):
                        dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if dist <= radius:
                            # Higher chance near center, drops off with distance
                            chance = initial_density * 5 * (1 - dist / radius) ** 2
                            if random.random() < chance:
                                grid[y, x] = race_id

        elif pattern == "network":
            # Create a network pattern with nodes and connections
            num_nodes = random.randint(3, 6)
            nodes = []

            # Generate nodes
            for _ in range(num_nodes):
                x = random.randint(width // 5, width * 4 // 5)
                y = random.randint(height // 5, height * 4 // 5)
                nodes.append((x, y))

                # Create symbiote cluster around node
                radius = random.randint(5, 10)
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist <= radius:
                                chance = initial_density * 3 * (1 - dist / radius)
                                if random.random() < chance:
                                    grid[ny, nx] = race_id

            # Connect nodes
            for i in range(len(nodes) - 1):
                x1, y1 = nodes[i]
                x2, y2 = nodes[i + 1]

                # Use Bresenham's line algorithm to connect nodes
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                err = dx - dy

                x, y = x1, y1
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                while True:
                    # Place symbiote at this point with some randomness
                    if random.random() < initial_density * 5:
                        grid[y, x] = race_id

                    # Also occasionally place nearby
                    if random.random() < initial_density * 2:
                        nx = x + random.randint(-2, 2)
                        ny = y + random.randint(-2, 2)
                        if 0 <= nx < width and 0 <= ny < height:
                            grid[ny, nx] = race_id

                    if x == x2 and y == y2:
                        break

                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy

        else:
            # Fallback to simple random placement
            random_mask = np.random.random((height, width)) < initial_density
            grid[random_mask] = race_id

        return grid

    def generate_resource_hotspots(
        self, grid: np.ndarray, num_hotspots: int = 3
    ) -> np.ndarray:
        """
        Generate resource hotspots - areas with increased mineral values.

        Args:
            grid: Asteroid field grid
            num_hotspots: Number of hotspots to generate

        Returns:
            np.ndarray: Modified grid with hotspots
        """
        height, width = grid.shape
        new_grid = grid.copy()

        for _ in range(num_hotspots):
            # Choose a random center for the hotspot
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)

            # Determine hotspot radius (bigger ones are rarer)
            size_roll = random.random()
            if size_roll < 0.6:  # 60% small
                radius = random.randint(10, 20)
                multiplier = random.uniform(1.2, 1.5)
            elif size_roll < 0.9:  # 30% medium
                radius = random.randint(20, 35)
                multiplier = random.uniform(1.5, 2.0)
            else:  # 10% large
                radius = random.randint(35, 50)
                multiplier = random.uniform(2.0, 3.0)

            # Apply value increase to asteroids within radius
            for y in range(
                max(0, center_y - radius), min(height, center_y + radius + 1)
            ):
                for x in range(
                    max(0, center_x - radius), min(width, center_x + radius + 1)
                ):
                    dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist <= radius and grid[y, x] > 0:
                        # Value boost decreases with distance from center
                        distance_factor = 1 - (dist / radius) ** 0.5
                        boost = multiplier * distance_factor
                        new_grid[y, x] = int(grid[y, x] * boost)

        return new_grid

    def simulate_symbiote_evolution(
        self,
        race_id: int,
        population: int,
        aggression: float,
        minerals: Dict[str, int],
        genome: Dict[str, float],
    ) -> Tuple[int, float, List[Dict[str, Any]]]:
        """
        Simulate symbiote evolution based on minerals consumed.

        Args:
            race_id: Race identifier
            population: Current population count
            aggression: Current aggression level (0-1)
            minerals: Dict with mineral counts consumed
            genome: Dictionary of genome traits

        Returns:
            Tuple[int, float, list]: New population, new aggression, and list of mutations
        """
        with LogContext("simulate_symbiote_evolution"):
            # Start with current values
            new_population = population
            new_aggression = aggression
            mutations = []

            # Apply the symbiote evolution algorithm as defined in the design doc

            # 1. Calculate base food value from minerals
            food_value = (
                minerals.get("common", 0) * 1
                + minerals.get("rare", 0) * 5
                + minerals.get("precious", 0) * 10
                + minerals.get("anomaly", 0) * 20
            )

            # 2. Calculate population growth based on food
            if food_value > 0:
                # More efficient metabolisms extract more from the same food
                metabolism_factor = genome.get("metabolism_rate", 1.0)
                efficiency = (
                    0.05 * metabolism_factor
                )  # Base of 5% population increase per food unit

                # Growth is also affected by current population (logistic growth)
                carrying_capacity = self.evolution_params["carrying_capacity"] * (
                    1 + food_value / 100
                )
                logistic_factor = max(0.1, 1 - (population / carrying_capacity))

                # Calculate growth
                growth = min(
                    int(food_value * efficiency * logistic_factor),
                    carrying_capacity - population,
                )
                new_population = min(population + growth, carrying_capacity)

                # Feeding reduces aggression
                satiation = min(
                    1.0, food_value / (population * 0.1)
                )  # How satisfied they are
                new_aggression = max(0.1, aggression - 0.2 * satiation)
            else:
                # No food increases aggression
                hunger_factor = self.evolution_params["aggression_hunger_factor"]
                new_aggression = min(1.0, aggression + hunger_factor)

            # 3. Process mutations based on mineral types
            for mineral_type, amount in minerals.items():
                if amount <= 0:
                    continue

                for _ in range(amount):
                    # Check if mutation occurs
                    if random.random() < self.evolution_params["mutation_chance"].get(
                        mineral_type, 0
                    ):
                        # Determine which attribute mutates
                        attrs = list(genome.keys())
                        attr = random.choice(attrs)

                        # Calculate mutation magnitude
                        base_magnitude = self.evolution_params[
                            "mutation_magnitude"
                        ].get(mineral_type, 1.05)

                        # Randomize slightly and apply mutation bias based on genome
                        mutation_bias = random.uniform(0.9, 1.1)
                        magnitude = base_magnitude * mutation_bias

                        # Create mutation record
                        mutation = {
                            "attribute": attr,
                            "magnitude": magnitude,
                            "mineral_type": mineral_type,
                        }

                        mutations.append(mutation)

                        # Special effects for anomaly minerals
                        if mineral_type == "anomaly":
                            # Anomalies can cause dramatic changes
                            effect = random.choice(
                                [
                                    "population_surge",
                                    "aggression_spike",
                                    "aggression_drop",
                                    "metabolic_shift",
                                ]
                            )

                            if effect == "population_surge":
                                # Sudden population increase
                                new_population = min(
                                    int(new_population * 1.3), carrying_capacity
                                )
                            elif effect == "aggression_spike":
                                # Become much more aggressive
                                new_aggression = min(1.0, new_aggression + 0.3)
                            elif effect == "aggression_drop":
                                # Become much calmer
                                new_aggression = max(0.1, new_aggression - 0.3)
                            elif effect == "metabolic_shift":
                                # Fundamental change to metabolism
                                mutations.append(
                                    {
                                        "attribute": "metabolism_rate",
                                        "magnitude": random.choice(
                                            [0.7, 1.4]
                                        ),  # Either much higher or lower
                                        "mineral_type": mineral_type,
                                    }
                                )

            return new_population, new_aggression, mutations

    def _generate_noise_layer(
        self, width: int, height: int, noise_fn: PerlinNoise, scale: float = 0.02
    ) -> np.ndarray:
        """
        Generate a noise layer using the provided noise function.

        Args:
            width: Width of the grid
            height: Height of the grid
            noise_fn: Noise function to use
            scale: Scale factor for noise (lower = more zoomed out)

        Returns:
            np.ndarray: 2D grid of noise values
        """
        noise = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                noise[y, x] = noise_fn([x * scale, y * scale])
        return noise

    def _apply_cellular_automaton(
        self, grid: np.ndarray, iterations: int = 3
    ) -> np.ndarray:
        """
        Apply cellular automaton rules to refine the asteroid field pattern.

        Args:
            grid: Initial grid
            iterations: Number of CA iterations to perform

        Returns:
            np.ndarray: Evolved grid
        """
        binary_grid = (grid > 0).astype(np.int8)
        height, width = grid.shape

        for _ in range(iterations):
            new_binary = np.zeros_like(binary_grid)

            # Count neighbors using convolution for efficiency
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            neighbors = ndimage.convolve(
                binary_grid,
                kernel,
                mode="wrap" if self.ca_params["wrap"] else "constant",
            )

            # Apply Conway's Game of Life rules
            # Birth rule: empty cell with exactly 3 neighbors becomes filled
            birth_mask = (binary_grid == 0) & np.isin(
                neighbors, list(self.ca_params["birth_set"])
            )
            # Survival rule: filled cell with 2 or 3 neighbors stays filled
            survival_mask = (binary_grid == 1) & np.isin(
                neighbors, list(self.ca_params["survival_set"])
            )

            # Update grid
            new_binary[birth_mask | survival_mask] = 1
            binary_grid = new_binary

        # Apply the final binary pattern to the original grid
        # (preserve values where the binary grid has 1s, zero out the rest)
        new_grid = grid.copy()
        new_grid[binary_grid == 0] = 0

        # Add some values to newly created cells
        birth_mask = (binary_grid == 1) & (grid == 0)
        if np.any(birth_mask):
            new_values = self.value_distribution.rvs(size=np.sum(birth_mask))
            new_values = np.clip(new_values, 10, 200).astype(np.int32)
            new_grid[birth_mask] = new_values

        return new_grid

    def _create_value_clusters(self, grid: np.ndarray) -> np.ndarray:
        """
        Create clusters of higher-value asteroids using k-means clustering.

        Args:
            grid: Asteroid field grid

        Returns:
            np.ndarray: Modified grid with value clusters
        """
        new_grid = grid.copy()
        asteroid_mask = grid > 0

        # If too few asteroids, return original grid
        if np.sum(asteroid_mask) < 10:
            return grid

        # Get coordinates of all asteroids
        asteroid_coords = np.array(np.where(asteroid_mask)).T

        # Determine number of clusters
        n_clusters = random.randint(*self.cluster_count_range)
        n_clusters = min(
            n_clusters, len(asteroid_coords) // 10
        )  # Ensure reasonable cluster size

        if n_clusters < 2:  # Need at least 2 clusters for k-means
            return grid

        try:
            # Apply k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed).fit(
                asteroid_coords
            )
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_.astype(int)

            # Assign value multipliers to clusters
            cluster_multipliers = np.random.uniform(1.0, 2.5, size=n_clusters)

            # Apply multipliers based on distance from cluster centers
            for i, (y, x) in enumerate(asteroid_coords):
                cluster_id = labels[i]
                center_y, center_x = centers[cluster_id]

                # Calculate distance to center (normalized)
                distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                max_dist = (
                    np.sqrt(grid.shape[0] ** 2 + grid.shape[1] ** 2) * 0.1
                )  # 10% of diagonal
                distance_factor = max(0, 1 - distance / max_dist)

                # Apply value boost based on distance from center
                if distance_factor > 0:
                    multiplier = (
                        1 + (cluster_multipliers[cluster_id] - 1) * distance_factor
                    )
                    new_grid[y, x] = int(grid[y, x] * multiplier)

        except Exception as e:
            log_exception(e)
            # If clustering fails, return original grid
            return grid

        return new_grid

    def _create_void_areas(
        self, grid: np.ndarray, count: int = 3, max_size: int = 50
    ) -> np.ndarray:
        """
        Create void areas (areas with no asteroids) for visual interest.

        Args:
            grid: Asteroid field grid
            count: Number of void areas to create
            max_size: Maximum radius of void areas

        Returns:
            np.ndarray: Modified grid with voids
        """
        new_grid = grid.copy()
        height, width = grid.shape

        for _ in range(count):
            # Choose a random center for the void
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)

            # Determine void radius
            radius = random.randint(10, max_size)

            # Create void by setting values to 0 within radius
            for y in range(
                max(0, center_y - radius), min(height, center_y + radius + 1)
            ):
                for x in range(
                    max(0, center_x - radius), min(width, center_x + radius + 1)
                ):
                    dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist <= radius:
                        fade_factor = dist / radius  # More likely to remove near center
                        if random.random() > fade_factor:
                            new_grid[y, x] = 0

        return new_grid

    def _apply_edge_degradation(
        self, grid: np.ndarray, decay_factor: float = 0.1
    ) -> np.ndarray:
        """
        Apply edge degradation to make borders look more natural.

        Args:
            grid: Asteroid field grid
            decay_factor: How quickly edges degrade

        Returns:
            np.ndarray: Modified grid with degraded edges
        """
        new_grid = grid.copy()
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if y == 0 or y == grid.shape[0] - 1 or x == 0 or x == grid.shape[1] - 1:
                    new_grid[y, x] = int(grid[y, x] * (1 - decay_factor))

        return new_grid

    def _apply_cellular_automaton(
        self, grid: np.ndarray, iterations: int = 3
    ) -> np.ndarray:
        """
        Apply cellular automaton to evolve field patterns.

        Args:
            grid: Asteroid field grid
            iterations: Number of iterations for cellular automaton

        Returns:
            np.ndarray: Modified grid with evolved patterns
        """
        for _ in range(iterations):
            new_grid = np.zeros_like(grid)
            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    neighbors = grid[
                        max(0, y - 1) : min(grid.shape[0], y + 2),
                        max(0, x - 1) : min(grid.shape[1], x + 2),
                    ]
                    total = np.sum(neighbors)
                    if total > 0:
                        new_grid[y, x] = 1
            grid = new_grid

        return grid

    def _apply_resource_distribution(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply resource distribution using statistical models.

        Args:
            grid: Asteroid field grid

        Returns:
            np.ndarray: Modified grid with distributed resources
        """
        new_grid = grid.copy()
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] > 0:
                    new_grid[y, x] = random.randint(1, 3)

        return new_grid

    def _apply_symbiote_evolution(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply symbiote evolution using the symbiote evolution algorithm.

        Args:
            grid: Asteroid field grid

        Returns:
            np.ndarray: Modified grid with evolved symbiotes
        """
        new_grid = grid.copy()
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] > 0:
                    new_grid[y, x] = random.randint(1, 4)

        return new_grid

    def _apply_void_distribution(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply void distribution to create empty areas.

        Args:
            grid: Asteroid field grid

        Returns:
            np.ndarray: Modified grid with voids
        """
        new_grid = grid.copy()
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == 0:
                    new_grid[y, x] = random.randint(1, 3)

        return new_grid

    def _apply_resource_hotspots(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply resource hotspots to create concentrated areas of resources.

        Args:
            grid: Asteroid field grid

        Returns:
            np.ndarray: Modified grid with hotspots
        """
        new_grid = grid.copy()
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] > 0:
                    new_grid[y, x] = random.randint(1, 3)

        return new_grid

    def _apply_symbiote_hotspots(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply symbiote hotspots to create concentrated areas of symbiotes.

        Args:
            grid: Asteroid field grid

        Returns:
            np.ndarray: Modified grid with hotspots
        """
        new_grid = grid.copy()
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] > 0:
                    new_grid[y, x] = random.randint(1, 4)

        return new_grid


"""
Procedural Generation for Space Muck.

This module implements advanced procedural generation algorithms including:
- Multi-layered noise for asteroid distribution
- Cellular automaton for evolving field patterns
- Resource distribution using statistical models
- Symbiote race evolution using the symbiote evolution algorithm
"""
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable

import numpy as np
import scipy.ndimage as ndimage
import scipy.stats as stats
from perlin_noise import PerlinNoise
from skimage import measure
from sklearn.cluster import KMeans

from src.config import *
from src.utils.logging_setup import (
    log_performance_start,
    log_performance_end,
    log_exception,
    LogContext,
)


class ProceduralGenerator:
    """
    Handles all procedural generation for the asteroid field and symbiote races.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the procedural generator with an optional seed.

        Args:
            seed: Random seed for reproducibility
        """
        # Set seed for reproducibility
        self.seed = seed if seed is not None else random.randint(1, 100000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Noise generators
        self.primary_noise = PerlinNoise(octaves=4, seed=self.seed)
        self.secondary_noise = PerlinNoise(octaves=6, seed=self.seed + 1)
        self.rare_noise = PerlinNoise(octaves=8, seed=self.seed + 2)
        self.anomaly_noise = PerlinNoise(octaves=12, seed=self.seed + 3)

        # Statistical parameters
        self.value_distribution = stats.lognorm(s=0.6, scale=50)  # For asteroid values
        self.cluster_count_range = (3, 8)  # Range for number of clusters
        self.cluster_size_range = (20, 80)  # Range for cluster size

        # Configure cellular automaton parameters
        self.ca_params = {
            "birth_set": {3},  # Default: cell is born if it has exactly 3 neighbors
            "survival_set": {2, 3},  # Default: cell survives with 2 or 3 neighbors
            "iterations": 3,  # Default iterations for CA evolution
            "wrap": True,  # Whether the grid wraps around edges
        }

        # Fractal parameters
        self.fractal_depth = 3  # Depth for fractal generation
        self.fractal_roughness = 0.6  # Roughness factor for fractals

        # Anomaly generation parameters
        self.anomaly_chance = 0.01  # Base chance of anomaly per cell
        self.anomaly_cluster_chance = 0.2  # Chance of anomaly clustering

        # Symbiote evolution parameters
        self.evolution_params = {
            "growth_rate": 0.2,  # Base growth rate for symbiotes
            "carrying_capacity": 100,  # Base carrying capacity
            "aggression_hunger_factor": 0.1,  # How aggression increases with hunger
            "aggression_fleet_factor": 0.05,  # How aggression responds to fleet size
            "mutation_chance": {
                "common": 0.03,  # 3% chance per common mineral
                "rare": 0.2,  # 20% chance per rare mineral
                "precious": 0.1,  # 10% chance per precious mineral
                "anomaly": 0.5,  # 50% chance per anomaly mineral
            },
            "mutation_magnitude": {
                "common": 1.05,  # 5% change
                "rare": 1.2,  # 20% change
                "precious": 1.15,  # 15% change
                "anomaly": 1.5,  # 50% change
            },
        }

    def generate_asteroid_field(
        self, width: int, height: int, density: float = 0.3
    ) -> np.ndarray:
        """
        Generate a procedural asteroid field using multiple noise layers.

        Args:
            width: Width of the field
            height: Height of the field
            density: Overall density of asteroids (0-1)

        Returns:
            np.ndarray: 2D grid with asteroid values
        """
        start_time = log_performance_start("generate_asteroid_field")

        try:
            # Initialize the grid
            grid = np.zeros((height, width), dtype=np.int32)

            # Generate primary noise layer
            primary = self._generate_noise_layer(
                width, height, self.primary_noise, scale=0.02
            )
            primary = (primary - primary.min()) / (primary.max() - primary.min())

            # Generate secondary noise for details
            secondary = self._generate_noise_layer(
                width, height, self.secondary_noise, scale=0.05
            )
            secondary = (secondary - secondary.min()) / (
                secondary.max() - secondary.min()
            )

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
            grid = self._apply_cellular_automaton(
                grid, iterations=self.ca_params["iterations"]
            )

            # Create clusters of higher-value asteroids
            grid = self._create_value_clusters(grid)

            # Generate void spaces for visual interest
            grid = self._create_void_areas(grid, count=3, max_size=50)

            # Add edge degradation for natural appearance
            grid = self._apply_edge_degradation(grid, decay_factor=0.1)

            log_performance_end("generate_asteroid_field", start_time)
            return grid

        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.random.binomial(1, density * 0.5, (height, width)) * 50

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
            height, width = grid.shape
            rare_grid = np.zeros_like(grid, dtype=np.int8)

            # No rare minerals where there are no asteroids
            asteroid_mask = grid > 0

            # Generate noise for rare mineral distribution
            rare_noise = self._generate_noise_layer(
                width, height, self.rare_noise, scale=0.03
            )
            rare_noise = (rare_noise - rare_noise.min()) / (
                rare_noise.max() - rare_noise.min()
            )

            # Generate precious noise (even rarer)
            precious_noise = self._generate_noise_layer(
                width, height, self.rare_noise, scale=0.015
            )
            precious_noise = (precious_noise - precious_noise.min()) / (
                precious_noise.max() - precious_noise.min()
            )

            # Generate anomaly noise (extremely rare)
            anomaly_noise = self._generate_noise_layer(
                width, height, self.anomaly_noise, scale=0.01
            )
            anomaly_noise = (anomaly_noise - anomaly_noise.min()) / (
                anomaly_noise.max() - anomaly_noise.min()
            )

            # Apply thresholds to determine rare mineral locations
            # Higher grid values slightly increase chances of rare minerals
            value_bonus = np.clip((grid / 100), 0, 0.2)

            # 1 = rare minerals (10% base chance)
            rare_mask = (rare_noise > (1 - rare_chance - value_bonus)) & asteroid_mask
            rare_grid[rare_mask] = 1

            # 2 = precious minerals (2% base chance)
            precious_mask = (precious_noise > 0.98 - value_bonus * 0.5) & asteroid_mask
            rare_grid[precious_mask] = 2

            # 3 = anomaly minerals (0.5% base chance)
            anomaly_mask = (anomaly_noise > 0.995 - value_bonus * 0.1) & asteroid_mask
            rare_grid[anomaly_mask] = 3

            # Create small clusters of rare minerals using connected component analysis
            for mineral_type in [1, 2, 3]:
                type_mask = rare_grid == mineral_type
                if np.sum(type_mask) > 0:
                    # Expand some rare minerals to create small clusters
                    expanded = ndimage.binary_dilation(type_mask, iterations=1)
                    # But only where there are asteroids and not already other rare minerals
                    growth_mask = expanded & asteroid_mask & (rare_grid == 0)
                    # Apply with some probability
                    growth_prob = (
                        0.3 if mineral_type == 1 else 0.2 if mineral_type == 2 else 0.1
                    )
                    random_mask = np.random.random(growth_mask.shape) < growth_prob
                    rare_grid[growth_mask & random_mask] = mineral_type

            log_performance_end("generate_rare_minerals", start_time)
            return rare_grid

        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.zeros_like(grid, dtype=np.int8)

    def generate_symbiote_rules(
        self, race_id: int, hunger: float, genome: Dict[str, float]
    ) -> Tuple[Set[int], Set[int]]:
        """
        Generate cellular automaton rules for symbiote races based on their traits.

        Args:
            race_id: ID of the race
            hunger: Current hunger level (0-1)
            genome: Dictionary of genome traits

        Returns:
            Tuple[set, set]: Birth rule set and survival rule set
        """
        # Base rule sets similar to Conway's Game of Life
        base_birth = {3}  # Default: cell born with exactly 3 neighbors
        base_survival = {2, 3}  # Default: cell survives with 2 or 3 neighbors

        # Adjust rules based on race and genome
        birth_set = set(base_birth)
        survival_set = set(base_survival)

        # Adaptability affects rule diversity
        adaptability = genome.get("adaptability", 0.5)

        # More adaptable races have more diverse rules
        if adaptability > 0.7 and random.random() < adaptability:
            birth_set.add(random.choice([1, 4, 5]))

        # Hunger affects survival conditions
        if hunger > 0.7:
            # Desperate hunger - more aggressive expansion
            survival_set.add(1)  # Can survive with just 1 neighbor (less die-off)
        elif hunger < 0.3:
            # Well-fed - more stable, less aggressive expansion
            if 1 in survival_set:
                survival_set.remove(1)

        # Metabolism affects energy efficiency
        metabolism = genome.get("metabolism_rate", 1.0)
        if metabolism > 1.2:
            # High metabolism - can survive with more neighbors (more energy efficient)
            survival_set.add(4)
        elif metabolism < 0.8:
            # Low metabolism - stricter survival conditions
            if 4 in survival_set:
                survival_set.remove(4)

        # Expansion drive affects birth conditions
        expansion = genome.get("expansion_drive", 1.0)
        if expansion > 1.3:
            # High expansion - more birth conditions
            birth_set.add(4)
        elif expansion < 0.7:
            # Low expansion - stricter birth conditions
            if len(birth_set) > 1 and 4 in birth_set:
                birth_set.remove(4)

        # Ensure basic rules always apply
        birth_set.add(3)  # Always keep the classic birth rule
        survival_set.add(2)  # Always keep the classic survival rules

        return birth_set, survival_set

    def place_symbiote_race(
        self,
        width: int,
        height: int,
        race_id: int,
        initial_density: float = 0.005,
        pattern: str = "random",
    ) -> np.ndarray:
        """
        Generate initial placement for a symbiote race.

        Args:
            width: Grid width
            height: Grid height
            race_id: Race identifier
            initial_density: Initial population density
            pattern: Placement pattern ('random', 'clusters', 'network')

        Returns:
            np.ndarray: Grid with symbiote placements
        """
        grid = np.zeros((height, width), dtype=np.int32)

        if pattern == "clusters":
            # Create clusters of symbiotes
            num_clusters = random.randint(2, 5)
            for _ in range(num_clusters):
                center_x = random.randint(0, width - 1)
                center_y = random.randint(0, height - 1)
                radius = random.randint(10, 30)

                # Create each cluster with a density falloff from center
                for y in range(
                    max(0, center_y - radius), min(height, center_y + radius + 1)
                ):
                    for x in range(
                        max(0, center_x - radius), min(width, center_x + radius + 1)
                    ):
                        dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if dist <= radius:
                            # Higher chance near center, drops off with distance
                            chance = initial_density * 5 * (1 - dist / radius) ** 2
                            if random.random() < chance:
                                grid[y, x] = race_id

        elif pattern == "network":
            # Create a network pattern with nodes and connections
            num_nodes = random.randint(3, 6)
            nodes = []

            # Generate nodes
            for _ in range(num_nodes):
                x = random.randint(width // 5, width * 4 // 5)
                y = random.randint(height // 5, height * 4 // 5)
                nodes.append((x, y))

                # Create symbiote cluster around node
                radius = random.randint(5, 10)
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist <= radius:
                                chance = initial_density * 3 * (1 - dist / radius)
                                if random.random() < chance:
                                    grid[ny, nx] = race_id

            # Connect nodes
            for i in range(len(nodes) - 1):
                x1, y1 = nodes[i]
                x2, y2 = nodes[i + 1]

                # Use Bresenham's line algorithm to connect nodes
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                err = dx - dy

                x, y = x1, y1
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                while True:
                    # Place symbiote at this point with some randomness
                    if random.random() < initial_density * 5:
                        grid[y, x] = race_id

                    # Also occasionally place nearby
                    if random.random() < initial_density * 2:
                        nx = x + random.randint(-2, 2)
                        ny = y + random.randint(-2, 2)
                        if 0 <= nx < width and 0 <= ny < height:
                            grid[ny, nx] = race_id

                    if x == x2 and y == y2:
                        break

                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy

        else:
            # Fallback to simple random placement
            random_mask = np.random.random((height, width)) < initial_density
            grid[random_mask] = race_id

        return grid

    def generate_resource_hotspots(
        self, grid: np.ndarray, num_hotspots: int = 3
    ) -> np.ndarray:
        """
        Generate resource hotspots - areas with increased mineral values.

        Args:
            grid: Asteroid field grid
            num_hotspots: Number of hotspots to generate

        Returns:
            np.ndarray: Modified grid with hotspots
        """
        height, width = grid.shape
        new_grid = grid.copy()

        for _ in range(num_hotspots):
            # Choose a random center for the hotspot
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)

            # Determine hotspot radius (bigger ones are rarer)
            size_roll = random.random()
            if size_roll < 0.6:  # 60% small
                radius = random.randint(10, 20)
                multiplier = random.uniform(1.2, 1.5)
            elif size_roll < 0.9:  # 30% medium
                radius = random.randint(20, 35)
                multiplier = random.uniform(1.5, 2.0)
            else:  # 10% large
                radius = random.randint(35, 50)
                multiplier = random.uniform(2.0, 3.0)

            # Apply value increase to asteroids within radius
            for y in range(
                max(0, center_y - radius), min(height, center_y + radius + 1)
            ):
                for x in range(
                    max(0, center_x - radius), min(width, center_x + radius + 1)
                ):
                    dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist <= radius and grid[y, x] > 0:
                        # Value boost decreases with distance from center
                        distance_factor = 1 - (dist / radius) ** 0.5
                        boost = multiplier * distance_factor
                        new_grid[y, x] = int(grid[y, x] * boost)

        return new_grid

    def simulate_symbiote_evolution(
        self,
        race_id: int,
        population: int,
        aggression: float,
        minerals: Dict[str, int],
        genome: Dict[str, float],
    ) -> Tuple[int, float, List[Dict[str, Any]]]:
        """
        Simulate symbiote evolution based on minerals consumed.

        Args:
            race_id: Race identifier
            population: Current population count
            aggression: Current aggression level (0-1)
            minerals: Dict with mineral counts consumed
            genome: Dictionary of genome traits

        Returns:
            Tuple[int, float, list]: New population, new aggression, and list of mutations
        """
        with LogContext("simulate_symbiote_evolution"):
            # Start with current values
            new_population = population
            new_aggression = aggression
            mutations = []

            # Apply the symbiote evolution algorithm as defined in the design doc

            # 1. Calculate base food value from minerals
            food_value = (
                minerals.get("common", 0) * 1
                + minerals.get("rare", 0) * 5
                + minerals.get("precious", 0) * 10
                + minerals.get("anomaly", 0) * 20
            )

            # 2. Calculate population growth based on food
            if food_value > 0:
                # More efficient metabolisms extract more from the same food
                metabolism_factor = genome.get("metabolism_rate", 1.0)
                efficiency = (
                    0.05 * metabolism_factor
                )  # Base of 5% population increase per food unit

                # Growth is also affected by current population (logistic growth)
                carrying_capacity = self.evolution_params["carrying_capacity"] * (
                    1 + food_value / 100
                )
                logistic_factor = max(0.1, 1 - (population / carrying_capacity))

                # Calculate growth
                growth = min(
                    int(food_value * efficiency * logistic_factor),
                    carrying_capacity - population,
                )
                new_population = min(population + growth, carrying_capacity)

                # Feeding reduces aggression
                satiation = min(
                    1.0, food_value / (population * 0.1)
                )  # How satisfied they are
                new_aggression = max(0.1, aggression - 0.2 * satiation)
            else:
                # No food increases aggression
                hunger_factor = self.evolution_params["aggression_hunger_factor"]
                new_aggression = min(1.0, aggression + hunger_factor)

            # 3. Process mutations based on mineral types
            for mineral_type, amount in minerals.items():
                if amount <= 0:
                    continue

                for _ in range(amount):
                    # Check if mutation occurs
                    if random.random() < self.evolution_params["mutation_chance"].get(
                        mineral_type, 0
                    ):
                        # Determine which attribute mutates
                        attrs = list(genome.keys())
                        attr = random.choice(attrs)

                        # Calculate mutation magnitude
                        base_magnitude = self.evolution_params[
                            "mutation_magnitude"
                        ].get(mineral_type, 1.05)

                        # Randomize slightly and apply mutation bias based on genome
                        mutation_bias = random.uniform(0.9, 1.1)
                        magnitude = base_magnitude * mutation_bias

                        # Create mutation record
                        mutation = {
                            "attribute": attr,
                            "magnitude": magnitude,
                            "mineral_type": mineral_type,
                        }

                        mutations.append(mutation)

                        # Special effects for anomaly minerals
                        if mineral_type == "anomaly":
                            # Anomalies can cause dramatic changes
                            effect = random.choice(
                                [
                                    "population_surge",
                                    "aggression_spike",
                                    "aggression_drop",
                                    "metabolic_shift",
                                ]
                            )

                            if effect == "population_surge":
                                # Sudden population increase
                                new_population = min(
                                    int(new_population * 1.3), carrying_capacity
                                )
                            elif effect == "aggression_spike":
                                # Become much more aggressive
                                new_aggression = min(1.0, new_aggression + 0.3)
                            elif effect == "aggression_drop":
                                # Become much calmer
                                new_aggression = max(0.1, new_aggression - 0.3)
                            elif effect == "metabolic_shift":
                                # Fundamental change to metabolism
                                mutations.append(
                                    {
                                        "attribute": "metabolism_rate",
                                        "magnitude": random.choice(
                                            [0.7, 1.4]
                                        ),  # Either much higher or lower
                                        "mineral_type": mineral_type,
                                    }
                                )

            return new_population, new_aggression, mutations

    def _generate_noise_layer(
        self, width: int, height: int, noise_fn: PerlinNoise, scale: float = 0.02
    ) -> np.ndarray:
        """
        Generate a noise layer using the provided noise function.

        Args:
            width: Width of the grid
            height: Height of the grid
            noise_fn: Noise function to use
            scale: Scale factor for noise (lower = more zoomed out)

        Returns:
            np.ndarray: 2D grid of noise values
        """
        noise = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                noise[y, x] = noise_fn([x * scale, y * scale])
        return noise

    def _apply_cellular_automaton(
        self, grid: np.ndarray, iterations: int = 3
    ) -> np.ndarray:
        """
        Apply cellular automaton rules to refine the asteroid field pattern.

        Args:
            grid: Initial grid
            iterations: Number of CA iterations to perform

        Returns:
            np.ndarray: Evolved grid
        """
        binary_grid = (grid > 0).astype(np.int8)
        height, width = grid.shape

        for _ in range(iterations):
            new_binary = np.zeros_like(binary_grid)

            # Count neighbors using convolution for efficiency
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            neighbors = ndimage.convolve(
                binary_grid,
                kernel,
                mode="wrap" if self.ca_params["wrap"] else "constant",
            )

            # Apply Conway's Game of Life rules
            # Birth rule: empty cell with exactly 3 neighbors becomes filled
            birth_mask = (binary_grid == 0) & np.isin(
                neighbors, list(self.ca_params["birth_set"])
            )
            # Survival rule: filled cell with 2 or 3 neighbors stays filled
            survival_mask = (binary_grid == 1) & np.isin(
                neighbors, list(self.ca_params["survival_set"])
            )

            # Update grid
            new_binary[birth_mask | survival_mask] = 1
            binary_grid = new_binary

        # Apply the final binary pattern to the original grid
        # (preserve values where the binary grid has 1s, zero out the rest)
        new_grid = grid.copy()
        new_grid[binary_grid == 0] = 0

        # Add some values to newly created cells
        birth_mask = (binary_grid == 1) & (grid == 0)
        if np.any(birth_mask):
            new_values = self.value_distribution.rvs(size=np.sum(birth_mask))
            new_values = np.clip(new_values, 10, 200).astype(np.int32)
            new_grid[birth_mask] = new_values

        return new_grid

    def _create_value_clusters(self, grid: np.ndarray) -> np.ndarray:
        """
        Create clusters of higher-value asteroids using k-means clustering.

        Args:
            grid: Asteroid field grid

        Returns:
            np.ndarray: Modified grid with value clusters
        """
        new_grid = grid.copy()
        asteroid_mask = grid > 0

        # If too few asteroids, return original grid
        if np.sum(asteroid_mask) < 10:
            return grid

        # Get coordinates of all asteroids
        asteroid_coords = np.array(np.where(asteroid_mask)).T

        # Determine number of clusters
        n_clusters = random.randint(*self.cluster_count_range)
        n_clusters = min(
            n_clusters, len(asteroid_coords) // 10
        )  # Ensure reasonable cluster size

        if n_clusters < 2:  # Need at least 2 clusters for k-means
            return grid

        try:
            # Apply k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed).fit(
                asteroid_coords
            )
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_.astype(int)

            # Assign value multipliers to clusters
            cluster_multipliers = np.random.uniform(1.0, 2.5, size=n_clusters)

            # Apply multipliers based on distance from cluster centers
            for i, (y, x) in enumerate(asteroid_coords):
                cluster_id = labels[i]
                center_y, center_x = centers[cluster_id]

                # Calculate distance to center (normalized)
                distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                max_dist = (
                    np.sqrt(grid.shape[0] ** 2 + grid.shape[1] ** 2) * 0.1
                )  # 10% of diagonal
                distance_factor = max(0, 1 - distance / max_dist)

                # Apply value boost based on distance from center
                if distance_factor > 0:
                    multiplier = (
                        1 + (cluster_multipliers[cluster_id] - 1) * distance_factor
                    )
                    new_grid[y, x] = int(grid[y, x] * multiplier)

        except Exception as e:
            log_exception(e)
            # If clustering fails, return original grid
            return grid

        return new_grid

    def _create_void_areas(
        self, grid: np.ndarray, count: int = 3, max_size: int = 50
    ) -> np.ndarray:
        """
        Create void areas (areas with no asteroids) for visual interest.

        Args:
            grid: Asteroid field grid
            count: Number of void areas to create
            max_size: Maximum radius of void areas

        Returns:
            np.ndarray: Modified grid with voids
        """
        new_grid = grid.copy()
        height, width = grid.shape

        for _ in range(count):
            # Choose a random center for the void
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)

            # Determine void radius
            radius = random.randint(10, max_size)

            # Create void by setting values to 0 within radius
            for y in range(
                max(0, center_y - radius), min(height, center_y + radius + 1)
            ):
                for x in range(
                    max(0, center_x - radius), min(width, center_x + radius + 1)
                ):
                    dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist <= radius:
                        fade_factor = dist / radius  # More likely to remove near center
                        if random.random() > fade_factor:
                            new_grid[y, x] = 0

        return new_grid

    def _apply_edge_degradation(
        self, grid: np.ndarray, decay_factor: float = 0.1
    ) -> np.ndarray:
        """
        Apply edge degradation to make borders look more natural.

        Args:
            grid: Asteroid field grid
            decay_factor: How quickly edges degrade

        Returns:
            np.ndarray: Modified grid with degraded edges
        """
        new_grid = grid.copy()

        # Apply edge degradation
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if x == 0 or x == grid.shape[1] - 1 or y == 0 or y == grid.shape[0] - 1:
                    new_grid[y, x] = int(grid[y, x] * (1 - decay_factor))

        return new_grid

    def _apply_random_noise(
        self, grid: np.ndarray, noise_factor: float = 0.1
    ) -> np.ndarray:
        """
        Apply random noise to the asteroid field.

        Args:
            grid: Asteroid field grid
            noise_factor: How much noise to apply

        Returns:
            np.ndarray: Modified grid with random noise
        """
        new_grid = grid.copy()
        noise = np.random.normal(0, noise_factor, size=grid.shape)
        new_grid += noise.astype(int)
        return new_grid

    def _apply_cellular_automaton(
        self, grid: np.ndarray, iterations: int = 3
    ) -> np.ndarray:
        """
        Apply cellular automaton rules to refine the asteroid field pattern.

        Args:
            grid: Initial grid
            iterations: Number of CA iterations to perform

        Returns:
            np.ndarray: Evolved grid
        """
        binary_grid = (grid > 0).astype(np.int8)
        height, width = grid.shape

        for _ in range(iterations):
            new_binary = np.zeros_like(binary_grid)

            # Count neighbors using convolution for efficiency
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            neighbors = ndimage.convolve(
                binary_grid,
                kernel,
                mode="wrap" if self.ca_params["wrap"] else "constant",
            )

            # Apply CA rules
            for y in range(height):
                for x in range(width):
                    neighbors_count = neighbors[y, x]
                    if (
                        binary_grid[y, x] == 1
                        and neighbors_count in self.ca_params["survival"]
                    ):
                        new_binary[y, x] = 1
                    elif (
                        binary_grid[y, x] == 0
                        and neighbors_count in self.ca_params["birth"]
                    ):
                        new_binary[y, x] = 1

            binary_grid = new_binary

        return (new_binary * grid).astype(np.int8)

    def _apply_gaussian_blur(self, grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur to smooth the asteroid field pattern.

        Args:
            grid: Initial grid
            sigma: Standard deviation of the Gaussian kernel

        Returns:
            np.ndarray: Smoothed grid
        """
        return ndimage.gaussian_filter(grid, sigma=sigma)

    def _apply_perlin_noise(self, grid: np.ndarray, scale: float = 10.0) -> np.ndarray:
        """
        Apply Perlin noise to the asteroid field.

        Args:
            grid: Initial grid
            scale: Scale factor for Perlin noise

        Returns:
            np.ndarray: Perlin-noised grid
        """
        noise = PerlinNoise(octaves=4)
        return (
            noise(
                [x / scale for x in range(grid.shape[1])] for _ in range(grid.shape[0])
            )
            * 255
        ).astype(np.int8)

    def _apply_gradient(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply gradient to the asteroid field.

        Args:
            grid: Initial grid

        Returns:
            np.ndarray: Gradient grid
        """
        return np.gradient(grid)

    def _apply_sobel(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply Sobel operator to the asteroid field.

        Args:
            grid: Initial grid

        Returns:
            np.ndarray: Sobel grid
        """
        return ndimage.sobel(grid)

    def _apply_laplacian(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply Laplacian operator to the asteroid field.

        Args:
            grid: Initial grid

        Returns:
            np.ndarray: Laplacian grid
        """
        return ndimage.laplace(grid)

    def _apply_median_filter(self, grid: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Apply median filter to the asteroid field.

        Args:
            grid: Initial grid
            size: Size of the median filter window

        Returns:
            np.ndarray: Median-filtered grid
        """
        return ndimage.median_filter(grid, size=size)

    def _apply_threshold(self, grid: np.ndarray, threshold: int = 128) -> np.ndarray:
        """
        Apply threshold to the asteroid field.

        Args:
            grid: Initial grid
            threshold: Threshold value

        Returns:
            np.ndarray: Thresholded grid
        """
        return (grid > threshold).astype(np.int8)

    def _apply_erosion(self, grid: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Apply erosion to the asteroid field.

        Args:
            grid: Initial grid
            iterations: Number of erosion iterations

        Returns:
            np.ndarray: Eroded grid
        """
        return ndimage.binary_erosion(grid, iterations=iterations)

    def _apply_dilation(self, grid: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Apply dilation to the asteroid field.

        Args:
            grid: Initial grid
            iterations: Number of dilation iterations

        Returns:
            np.ndarray: Dilated grid
        """
        return ndimage.binary_dilation(grid, iterations=iterations)

    def _apply_opening(self, grid: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Apply opening to the asteroid field.

        Args:
            grid: Initial grid
            iterations: Number of opening iterations

        Returns:
            np.ndarray: Opened grid
        """
        return ndimage.binary_opening(grid, iterations=iterations)

    def _apply_closing(self, grid: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Apply closing to the asteroid field.

        Args:
            grid: Initial grid
            iterations: Number of closing iterations

        Returns:
            np.ndarray: Closed grid
        """
        return ndimage.binary_closing(grid, iterations=iterations)

    def _apply_morphological_gradient(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply morphological gradient to the asteroid field.

        Args:
            grid: Initial grid

        Returns:
            np.ndarray: Morphological gradient grid
        """
        return ndimage.morphological_gradient(grid)

    def _apply_morphological_laplace(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply morphological Laplace operator to the asteroid field.

        Args:
            grid: Initial grid

        Returns:
            np.ndarray: Morphological Laplace grid
        """
        return ndimage.morphological_laplace(grid)

    def _apply_morphological_opening(
        self, grid: np.ndarray, iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological opening to the asteroid field.

        Args:
            grid: Initial grid
            iterations: Number of opening iterations

        Returns:
            np.ndarray: Morphological opened grid
        """
        return ndimage.morphological_opening(grid, iterations=iterations)

    def _apply_morphological_closing(
        self, grid: np.ndarray, iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological closing to the asteroid field.

        Args:
            grid: Initial grid
            iterations: Number of closing iterations

        Returns:
            np.ndarray: Morphological closed grid
        """
        return ndimage.morphological_closing(grid, iterations=iterations)

    def _apply_morphological_erosion(
        self, grid: np.ndarray, iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological erosion to the asteroid field.

        Args:
            grid: Initial grid
            iterations: Number of erosion iterations

        Returns:
            np.ndarray: Morphological eroded grid
        """
        return ndimage.morphological_erosion(grid, iterations=iterations)

    def _apply_morphological_dilation(
        self, grid: np.ndarray, iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological dilation to the asteroid field.

        Args:
            grid: Initial grid
            iterations: Number of dilation iterations

        Returns:
            np.ndarray: Morphological dilated grid
        """
        return ndimage.morphological_dilation(grid, iterations=iterations)

    def _apply_morphological_gradient(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply morphological gradient to the asteroid field.

        Args:
            grid: Initial grid

        Returns:
            np.ndarray: Morphological gradient grid
        """
        return ndimage.morphological_gradient(grid)

    def _apply_morphological_laplace(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply morphological Laplace operator to the asteroid field.

        Args:
            grid: Initial grid

        Returns:
            np.ndarray: Morphological Laplace grid
        """
        return ndimage.morphological_laplace(grid)


"""
Procedural Generation for Space Muck.

This module implements advanced procedural generation algorithms including:
- Multi-layered noise for asteroid distribution
- Cellular automaton for evolving field patterns
- Resource distribution using statistical models
- Symbiote race evolution using the symbiote evolution algorithm
"""
import logging
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable

import numpy as np
import scipy.ndimage as ndimage
from perlin_noise import PerlinNoise
from skimage import measure
from sklearn.cluster import KMeans
import scipy.stats as stats

from src.config import *
from src.utils.logging_setup import (
    log_exception,
    LogContext,
    log_performance_start,
    log_performance_end,
)


class AsteroidGenerator:
    """
    Generator for procedural asteroid fields with multiple layers and patterns.
    """

    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        """
        Initialize the asteroid generator.

        Args:
            width: Width of the field to generate
            height: Height of the field to generate
            seed: Optional random seed for reproducible generation
        """
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(1, 1000000)

        # Initialize perlin noise generators with different scales
        self.perlin_large = PerlinNoise(octaves=3, seed=self.seed)
        self.perlin_medium = PerlinNoise(octaves=5, seed=self.seed + 1)
        self.perlin_small = PerlinNoise(octaves=7, seed=self.seed + 2)
        self.perlin_detail = PerlinNoise(octaves=10, seed=self.seed + 3)

        # Generation parameters
        self.density = 0.2  # Base density of asteroids
        self.value_mean = 5.0  # Mean value for asteroid resources
        self.value_stddev = 2.0  # Standard deviation for resource values
        self.rare_chance = 0.05  # Chance of rare resource types
        self.cluster_tendency = 0.6  # How much asteroids tend to cluster (0-1)
        self.pattern_strength = 0.4  # Strength of pattern influence (0-1)

        # Pattern generators
        self.patterns = [
            self._spiral_pattern,
            self._ring_pattern,
            self._gradient_pattern,
            self._void_pattern,
        ]

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

        # Create empty grid
        grid = np.zeros((self.height, self.width), dtype=float)

        # Apply base noise layer
        noise_grid = self._generate_base_noise()

        # Apply patterns if weights provided
        if pattern_weights and len(pattern_weights) == len(self.patterns):
            pattern_grid = self._apply_weighted_patterns(pattern_weights)
            # Blend noise and patterns based on pattern_strength
            grid = (
                noise_grid * (1 - self.pattern_strength)
                + pattern_grid * self.pattern_strength
            )
        else:
            grid = noise_grid

        # Normalize grid to 0-1 range
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-10)

        # Apply threshold to create binary asteroid field
        asteroid_grid = (grid > (1 - self.density)).astype(int)

        # Apply cellular automaton to smooth field
        asteroid_grid = self._apply_cellular_automaton(asteroid_grid)

        # Generate metadata
        metadata = {
            "seed": self.seed,
            "density": self.density,
            "pattern_strength": self.pattern_strength,
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

        # Create a value grid with the same shape
        value_grid = np.zeros_like(asteroid_grid, dtype=float)

        # Generate base values using perlin noise for spatial coherence
        value_noise = self._generate_value_noise()

        # Scale noise to desired mean and standard deviation
        value_noise = value_noise * self.value_stddev + self.value_mean

        # Apply values only to asteroid cells
        value_grid = asteroid_grid * value_noise.astype(int)

        # Ensure minimum value for all asteroids
        min_value = 1
        value_grid[value_grid > 0] = np.maximum(value_grid[value_grid > 0], min_value)

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

        # Create a grid for rare resources
        rare_grid = np.zeros_like(asteroid_grid)

        # Generate coherent noise pattern for rare distribution
        rare_noise = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                # Use medium scale noise for rare resource distribution
                nx = x / self.width
                ny = y / self.height
                rare_noise[y, x] = self.perlin_medium([nx, ny])

        # Normalize to 0-1
        rare_noise = (rare_noise - rare_noise.min()) / (
            rare_noise.max() - rare_noise.min() + 1e-10
        )

        # Create mask where noise exceeds threshold and asteroid exists
        rare_mask = (rare_noise > (1 - self.rare_chance)) & (asteroid_grid > 0)
        rare_grid[rare_mask] = 1

        # Add some special rare clusters
        num_clusters = random.randint(1, 3)
        for _ in range(num_clusters):
            # Place cluster at random location
            cx = random.randint(0, self.width - 1)
            cy = random.randint(0, self.height - 1)
            radius = random.randint(5, 15)

            # Create circular mask
            y_indices, x_indices = np.ogrid[: self.height, : self.width]
            dist_from_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
            mask = (dist_from_center <= radius) & (asteroid_grid > 0)

            # Apply with decreasing probability from center
            for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                    if asteroid_grid[y, x] > 0:
                        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if (
                            dist <= radius
                            and random.random() < (1 - dist / radius) ** 2
                        ):
                            rare_grid[y, x] = 1

        log_performance_end("generate_rare_resources", start_time)
        return rare_grid

    def generate_anomalies(self, asteroid_grid: np.ndarray) -> np.ndarray:
        """
        Generate anomalies that have unique effects when mined.

        Args:
            asteroid_grid: Binary grid indicating asteroid presence

        Returns:
            np.ndarray: Grid with anomaly indicators
        """
        start_time = log_performance_start("generate_anomalies")

        # Create a grid for anomalies
        anomaly_grid = np.zeros_like(asteroid_grid)

        # Very rare - only a few anomalies in the entire grid
        num_anomalies = random.randint(1, 5)

        # Find connected components in asteroid field
        labeled_field, num_features = ndimage.label(asteroid_grid)

        if num_features > 0:
            # Choose random components to place anomalies
            for _ in range(min(num_anomalies, num_features)):
                component_id = random.randint(1, num_features)
                component_mask = labeled_field == component_id

                if points := list(zip(*np.where(component_mask))):
                    y, x = random.choice(points)
                    anomaly_grid[y, x] = random.randint(
                        1, 3
                    )  # Different types of anomalies

        log_performance_end("generate_anomalies", start_time)
        return anomaly_grid

    def _generate_base_noise(self) -> np.ndarray:
        """Generate multi-layered noise for the base asteroid distribution."""
        grid = np.zeros((self.height, self.width))

        # Apply multiple noise layers at different scales
        for y in range(self.height):
            for x in range(self.width):
                # Normalize coordinates to 0-1
                nx = x / self.width
                ny = y / self.height

                # Combine noise at different scales with decreasing weights
                noise_val = (
                    0.5 * self.perlin_large([nx, ny])
                    + 0.3 * self.perlin_medium([nx * 2, ny * 2])
                    + 0.15 * self.perlin_small([nx * 4, ny * 4])
                    + 0.05 * self.perlin_detail([nx * 8, ny * 8])
                )

                grid[y, x] = (noise_val + 1) / 2  # Convert from -1:1 to 0:1 range

        return grid

    def _generate_value_noise(self) -> np.ndarray:
        """Generate noise pattern for asteroid values."""
        grid = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                nx = x / self.width
                ny = y / self.height

                # Combine noise at different scales
                noise_val = 0.6 * self.perlin_large(
                    [nx, ny]
                ) + 0.4 * self.perlin_medium([nx * 2, ny * 2])

                grid[y, x] = (noise_val + 1) / 2  # Convert from -1:1 to 0:1 range

        return grid

    def _add_value_clusters(self, value_grid: np.ndarray) -> np.ndarray:
        """Add high-value clusters to the value grid."""
        # Number of clusters depends on grid size
        num_clusters = int(math.sqrt(self.width * self.height) / 20)

        for _ in range(num_clusters):
            # Random cluster center
            cx = random.randint(0, self.width - 1)
            cy = random.randint(0, self.height - 1)

            # Random cluster size
            radius = random.randint(5, 15)

            # Random value boost
            value_boost = random.uniform(2, 5)

            # Apply boost to asteroids in cluster with distance falloff
            for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                    if value_grid[y, x] > 0:  # If there's an asteroid
                        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if dist <= radius:
                            # Apply boost with falloff from center
                            falloff = (1 - dist / radius) ** 2
                            value_grid[y, x] += value_grid[y, x] * value_boost * falloff

        return value_grid

    def _apply_cellular_automaton(
        self, grid: np.ndarray, iterations: int = 2
    ) -> np.ndarray:
        """Apply cellular automaton rules to smooth the asteroid field."""
        result = grid.copy()

        for _ in range(iterations):
            # Count neighbors for each cell using convolution
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            neighbors = ndimage.convolve(result, kernel, mode="constant")

            # Apply rules similar to Conway's Game of Life but with bias toward structure
            # Birth rule: empty cell becomes asteroid if it has 3-6 neighbors
            birth = (result == 0) & ((neighbors >= 3) & (neighbors <= 6))

            # Survival rule: asteroid survives if it has 2-5 neighbors
            survive = (result == 1) & ((neighbors >= 2) & (neighbors <= 5))

            # Update grid
            result = np.zeros_like(result)
            result[birth | survive] = 1

        return result

    def _apply_weighted_patterns(self, weights: List[float]) -> np.ndarray:
        """Apply multiple patterns weighted by importance."""
        # Normalize weights to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(self.patterns) for _ in self.patterns]

        # Create empty grid
        grid = np.zeros((self.height, self.width))

        # Apply each pattern with its weight
        for pattern_func, weight in zip(self.patterns, weights):
            if weight > 0:
                pattern_grid = pattern_func()
                grid += pattern_grid * weight

        return grid

    def _spiral_pattern(self) -> np.ndarray:
        """Generate a spiral pattern of asteroids."""
        grid = np.zeros((self.height, self.width))
        center_x = self.width // 2
        center_y = self.height // 2

        # Spiral parameters
        max_radius = min(self.width, self.height) * 0.4
        spacing = 4 + random.random() * 4  # Random spacing between spiral arms
        arms = random.randint(1, 4)  # Random number of spiral arms

        # Generate points along spiral arms
        for radius in range(int(max_radius)):
            angle_step = 0.2 / (radius + 1) if radius > 0 else 0.2

            for angle_offset in np.linspace(0, 2 * math.pi, arms, endpoint=False):
                angle = radius / spacing + angle_offset

                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))

                if 0 <= x < self.width and 0 <= y < self.height:
                    # Make spiral arm thicker near center
                    thickness = max(1, int(2 + radius * 0.02))
                    for dx in range(-thickness, thickness + 1):
                        for dy in range(-thickness, thickness + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                dist = math.sqrt(dx * dx + dy * dy)
                                if dist <= thickness:
                                    grid[ny, nx] = 1.0 - (radius / max_radius) * 0.5

        return grid

    def _ring_pattern(self) -> np.ndarray:
        """Generate concentric rings of asteroids."""
        grid = np.zeros((self.height, self.width))
        center_x = self.width // 2
        center_y = self.height // 2

        # Ring parameters
        max_radius = min(self.width, self.height) * 0.4
        num_rings = random.randint(2, 5)

        # Generate rings
        for ring in range(num_rings):
            # Calculate ring radius
            radius = max_radius * (ring + 1) / num_rings
            thickness = max(1, int(radius * 0.08))

            # Draw the ring
            for y in range(self.height):
                for x in range(self.width):
                    dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if abs(dist - radius) <= thickness and random.random() < 0.7:
                        grid[y, x] = 1.0 - (ring / num_rings) * 0.3

        return grid

    def _gradient_pattern(self) -> np.ndarray:
        """Generate a gradient pattern with higher density on one side."""
        grid = np.zeros((self.height, self.width))

        # Choose gradient direction
        direction = random.choice(["horizontal", "vertical", "diagonal"])

        for y in range(self.height):
            for x in range(self.width):
                if direction == "horizontal":
                    # Horizontal gradient (left to right)
                    gradient = x / self.width
                elif direction == "vertical":
                    # Vertical gradient (top to bottom)
                    gradient = y / self.height
                else:
                    # Diagonal gradient
                    gradient = (x / self.width + y / self.height) / 2

                # Apply some noise to make it less uniform
                noise_val = self.perlin_medium([x / self.width, y / self.height])
                value = gradient + (noise_val * 0.2)

                # Normalize and store
                grid[y, x] = max(0, min(1, value))

        return grid

    def _void_pattern(self) -> np.ndarray:
        """Generate a pattern with void areas (negative space)."""
        grid = np.ones((self.height, self.width))  # Start with all filled

        # Create several void areas
        num_voids = random.randint(3, 7)

        for _ in range(num_voids):
            # Random void center and size
            cx = random.randint(0, self.width - 1)
            cy = random.randint(0, self.height - 1)
            radius = random.randint(10, 30)

            # Create void area with soft edges
            for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist <= radius:
                        # Smoother transition at edge
                        edge_factor = min(1.0, (radius - dist) / (radius * 0.3))
                        grid[y, x] *= 1.0 - edge_factor

        return grid


class SymbioteEvolutionGenerator:
    """
    Handles procedural generation of symbiote races and their evolution patterns.
    Based on the symbiote evolution algorithm described in algorithm_design.md.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize the symbiote evolution generator.

        Args:
            width: Width of the field
            height: Height of the field
        """
        self.width = width
        self.height = height

        # Base parameters for all races
        self.base_parameters = {
            "birth_rate": 0.05,  # Base reproduction rate
            "death_rate": 0.03,  # Base death rate
            "mutation_rate": 0.01,  # Base mutation rate
            "aggression": 0.2,  # Base aggression
            "intelligence": 0.5,  # Base intelligence
            "adaptability": 0.5,  # Base adaptability
            "migration_factor": 0.1,  # Tendency to spread
            "resource_efficiency": 0.7,  # Resource utilization efficiency
        }

        # Trait templates for different race types
        self.race_traits = {
            "adaptive": {
                "adaptability": 0.8,
                "mutation_rate": 0.02,
                "intelligence": 0.6,
                "birth_set": {2, 3},
                "survival_set": {3, 4, 5},
            },
            "expansive": {
                "birth_rate": 0.08,
                "migration_factor": 0.3,
                "aggression": 0.3,
                "birth_set": {3, 4},
                "survival_set": {2, 3},
            },
            "selective": {
                "resource_efficiency": 0.9,
                "intelligence": 0.7,
                "birth_rate": 0.04,
                "birth_set": {1, 5},
                "survival_set": {1, 4},
            },
        }

    def generate_race_parameters(
        self, race_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate parameters for a new symbiote race.

        Args:
            race_type: Optional type of race to generate, or random if None

        Returns:
            dict: Parameters for the new race
        """
        # Choose random race type if none specified
        if race_type is None:
            race_type = random.choice(list(self.race_traits.keys()))

        # Start with base parameters
        params = self.base_parameters.copy()

        # Apply race-specific traits
        race_params = self.race_traits.get(race_type, {}).copy()
        params.update(race_params)

        # Add some randomness to parameters
        for key in params:
            if isinstance(params[key], (int, float)) and key != "mutation_rate":
                variation = random.uniform(-0.1, 0.1)
                params[key] = max(0.1, min(1.0, params[key] + variation))

        # Set color based on race type
        if race_type == "adaptive":
            params["color"] = (50, 100, 255)  # Blue
        elif race_type == "expansive":
            params["color"] = (255, 50, 150)  # Magenta
        elif race_type == "selective":
            params["color"] = (255, 165, 0)  # Orange
        else:
            # Generate random color for other types
            params["color"] = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200),
            )

        # Apply the race trait for diversity
        params["trait"] = race_type

        return params

    def generate_initial_population(
        self, asteroid_grid: np.ndarray, race_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate initial population grid for a symbiote race.

        Args:
            asteroid_grid: Binary grid of asteroid locations
            race_params: Parameters for the race

        Returns:
            np.ndarray: Population grid
        """
        population_grid = np.zeros_like(asteroid_grid)

        # Get race trait
        race_trait = race_params.get("trait", "adaptive")

        if race_trait == "adaptive":
            # Adaptive races form small clusters
            num_clusters = random.randint(2, 4)

            for _ in range(num_clusters):
                # Find a random location with asteroids nearby
                attempts = 0
                while attempts < 50:
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)

                    # Check if there are asteroids in the vicinity
                    has_asteroids = False
                    for dy in range(-10, 11):
                        for dx in range(-10, 11):
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < self.width
                                and 0 <= ny < self.height
                                and asteroid_grid[ny, nx] > 0
                            ):
                                has_asteroids = True
                                break
                        if has_asteroids:
                            break

                    if has_asteroids:
                        # Place cluster around this point
                        radius = random.randint(3, 7)
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height:
                                    dist = math.sqrt(dx * dx + dy * dy)
                                    if (
                                        dist <= radius
                                        and random.random()
                                        < 0.7 - (dist / radius) * 0.5
                                    ):
                                        population_grid[ny, nx] = 1
                        break

                    attempts += 1

        elif race_trait == "expansive":
            # Expansive races spread in lines and networks
            start_points = []

            # Find a few starting points near asteroids
            for _ in range(random.randint(2, 4)):
                for _ in range(100):
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)

                    # Check if there are asteroids nearby
                    has_asteroids = False
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < self.width
                                and 0 <= ny < self.height
                                and asteroid_grid[ny, nx] > 0
                            ):
                                has_asteroids = True
                                break
                        if has_asteroids:
                            break

                    if has_asteroids:
                        start_points.append((x, y))
                        break

            # Generate paths between points
            if len(start_points) >= 2:
                for i in range(len(start_points) - 1):
                    x1, y1 = start_points[i]
                    x2, y2 = start_points[i + 1]

                    # Create a path with some randomness
                    path_points = []
                    dx = x2 - x1
                    dy = y2 - y1
                    steps = max(abs(dx), abs(dy))

                    if steps > 0:
                        x_step = dx / steps
                        y_step = dy / steps

                        # Add some waypoints with deviation
                        waypoints = [(x1, y1)]
                        for _ in range(random.randint(1, 3)):
                            t = random.random()
                            wx = x1 + dx * t + random.randint(-10, 10)
                            wy = y1 + dy * t + random.randint(-10, 10)
                            waypoints.append((int(wx), int(wy)))
                        waypoints.append((x2, y2))

                        # Connect waypoints
                        for j in range(len(waypoints) - 1):
                            wx1, wy1 = waypoints[j]
                            wx2, wy2 = waypoints[j + 1]

                            wdx = wx2 - wx1
                            wdy = wy2 - wy1
                            wsteps = max(abs(wdx), abs(wdy))

                            if wsteps > 0:
                                wx_step = wdx / wsteps
                                wy_step = wdy / wsteps

                                for k in range(int(wsteps) + 1):
                                    px = int(wx1 + wx_step * k)
                                    py = int(wy1 + wy_step * k)

                                    if 0 <= px < self.width and 0 <= py < self.height:
                                        # Add some width to the path
                                        width = random.randint(1, 2)
                                        for dy in range(-width, width + 1):
                                            for dx in range(-width, width + 1):
                                                nx, ny = px + dx, py + dy
                                                if (
                                                    0 <= nx < self.width
                                                    and 0 <= ny < self.height
                                                    and random.random() < 0.7
                                                ):
                                                    population_grid[ny, nx] = 1

            # Add some random population points
            for _ in range(int(self.width * self.height * 0.001)):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                population_grid[y, x] = 1

        else:  # selective
            # Selective races concentrate near resource-rich areas

            # Find asteroid clusters using KMeans
            asteroid_points = []
            for y in range(self.height):
                asteroid_points.extend(
                    (x, y) for x in range(self.width) if asteroid_grid[y, x] > 0
                )
            if len(asteroid_points) > 10:
                # Convert to numpy array for KMeans
                points_array = np.array(asteroid_points)

                # Find clusters
                num_clusters = min(3, len(asteroid_points) // 10)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(
                    points_array
                )

                # Place population near cluster centers
                for center in kmeans.cluster_centers_:
                    cx, cy = int(center[0]), int(center[1])

                    # Place a small colony near the center
                    radius = random.randint(3, 8)
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                dist = math.sqrt(dx * dx + dy * dy)
                                if (
                                    dist <= radius
                                    and random.random() < 0.5 - (dist / radius) * 0.3
                                ):
                                    population_grid[ny, nx] = 1
            else:
                # Fallback for small asteroid fields
                for _ in range(int(self.width * self.height * 0.002)):
                    x = random.randint(0, self.width - 1)
                    y = random
                    population_grid[y, x] = 1

        return population_grid


"""
Procedural Generation for Space Muck.

This module implements advanced procedural generation algorithms including:
- Multi-layered noise for asteroid distribution
- Cellular automaton for evolving field patterns
- Resource distribution using statistical models
- Symbiote race evolution using the symbiote evolution algorithm
"""
import logging
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable

import numpy as np
import scipy.ndimage as ndimage
from perlin_noise import PerlinNoise
from skimage import measure
from sklearn.cluster import KMeans
import scipy.stats as stats

from src.config import *
from src.utils.logging_setup import (
    log_exception,
    LogContext,
    log_performance_start,
    log_performance_end,
)


class AsteroidGenerator:
    """
    Generator for procedural asteroid fields with multiple layers and patterns.
    """

    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        """
        Initialize the asteroid generator.

        Args:
            width: Width of the field to generate
            height: Height of the field to generate
            seed: Optional random seed for reproducible generation
        """
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(1, 1000000)

        # Initialize perlin noise generators with different scales
        self.perlin_large = PerlinNoise(octaves=3, seed=self.seed)
        self.perlin_medium = PerlinNoise(octaves=5, seed=self.seed + 1)
        self.perlin_small = PerlinNoise(octaves=7, seed=self.seed + 2)
        self.perlin_detail = PerlinNoise(octaves=10, seed=self.seed + 3)

        # Generation parameters
        self.density = 0.2  # Base density of asteroids
        self.value_mean = 5.0  # Mean value for asteroid resources
        self.value_stddev = 2.0  # Standard deviation for resource values
        self.rare_chance = 0.05  # Chance of rare resource types
        self.cluster_tendency = 0.6  # How much asteroids tend to cluster (0-1)
        self.pattern_strength = 0.4  # Strength of pattern influence (0-1)

        # Pattern generators
        self.patterns = [
            self._spiral_pattern,
            self._ring_pattern,
            self._gradient_pattern,
            self._void_pattern,
        ]

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

        # Create empty grid
        grid = np.zeros((self.height, self.width), dtype=float)

        # Apply base noise layer
        noise_grid = self._generate_base_noise()

        # Apply patterns if weights provided
        if pattern_weights and len(pattern_weights) == len(self.patterns):
            pattern_grid = self._apply_weighted_patterns(pattern_weights)
            # Blend noise and patterns based on pattern_strength
            grid = (
                noise_grid * (1 - self.pattern_strength)
                + pattern_grid * self.pattern_strength
            )
        else:
            grid = noise_grid

        # Normalize grid to 0-1 range
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-10)

        # Apply threshold to create binary asteroid field
        asteroid_grid = (grid > (1 - self.density)).astype(int)

        # Apply cellular automaton to smooth field
        asteroid_grid = self._apply_cellular_automaton(asteroid_grid)

        # Generate metadata
        metadata = {
            "seed": self.seed,
            "density": self.density,
            "pattern_strength": self.pattern_strength,
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

        # Create a value grid with the same shape
        value_grid = np.zeros_like(asteroid_grid, dtype=float)

        # Generate base values using perlin noise for spatial coherence
        value_noise = self._generate_value_noise()

        # Scale noise to desired mean and standard deviation
        value_noise = value_noise * self.value_stddev + self.value_mean

        # Apply values only to asteroid cells
        value_grid = asteroid_grid * value_noise.astype(int)

        # Ensure minimum value for all asteroids
        min_value = 1
        value_grid[value_grid > 0] = np.maximum(value_grid[value_grid > 0], min_value)

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

        # Create a grid for rare resources
        rare_grid = np.zeros_like(asteroid_grid)

        # Generate coherent noise pattern for rare distribution
        rare_noise = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                # Use medium scale noise for rare resource distribution
                nx = x / self.width
                ny = y / self.height
                rare_noise[y, x] = self.perlin_medium([nx, ny])

        # Normalize to 0-1
        rare_noise = (rare_noise - rare_noise.min()) / (
            rare_noise.max() - rare_noise.min() + 1e-10
        )

        # Create mask where noise exceeds threshold and asteroid exists
        rare_mask = (rare_noise > (1 - self.rare_chance)) & (asteroid_grid > 0)
        rare_grid[rare_mask] = 1

        # Add some special rare clusters
        num_clusters = random.randint(1, 3)
        for _ in range(num_clusters):
            # Place cluster at random location
            cx = random.randint(0, self.width - 1)
            cy = random.randint(0, self.height - 1)
            radius = random.randint(5, 15)

            # Create circular mask
            y_indices, x_indices = np.ogrid[: self.height, : self.width]
            dist_from_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
            mask = (dist_from_center <= radius) & (asteroid_grid > 0)

            # Apply with decreasing probability from center
            for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                    if asteroid_grid[y, x] > 0:
                        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if (
                            dist <= radius
                            and random.random() < (1 - dist / radius) ** 2
                        ):
                            rare_grid[y, x] = 1

        log_performance_end("generate_rare_resources", start_time)
        return rare_grid

    def generate_anomalies(self, asteroid_grid: np.ndarray) -> np.ndarray:
        """
        Generate anomalies that have unique effects when mined.

        Args:
            asteroid_grid: Binary grid indicating asteroid presence

        Returns:
            np.ndarray: Grid with anomaly indicators
        """
        start_time = log_performance_start("generate_anomalies")

        # Create a grid for anomalies
        anomaly_grid = np.zeros_like(asteroid_grid)

        # Very rare - only a few anomalies in the entire grid
        num_anomalies = random.randint(1, 5)

        # Find connected components in asteroid field
        labeled_field, num_features = ndimage.label(asteroid_grid)

        if num_features > 0:
            # Choose random components to place anomalies
            for _ in range(min(num_anomalies, num_features)):
                component_id = random.randint(1, num_features)
                component_mask = labeled_field == component_id

                if points := list(zip(*np.where(component_mask))):
                    y, x = random.choice(points)
                    anomaly_grid[y, x] = random.randint(
                        1, 3
                    )  # Different types of anomalies

        log_performance_end("generate_anomalies", start_time)
        return anomaly_grid

    def _generate_base_noise(self) -> np.ndarray:
        """Generate multi-layered noise for the base asteroid distribution."""
        grid = np.zeros((self.height, self.width))

        # Apply multiple noise layers at different scales
        for y in range(self.height):
            for x in range(self.width):
                # Normalize coordinates to 0-1
                nx = x / self.width
                ny = y / self.height

                # Combine noise at different scales with decreasing weights
                noise_val = (
                    0.5 * self.perlin_large([nx, ny])
                    + 0.3 * self.perlin_medium([nx * 2, ny * 2])
                    + 0.15 * self.perlin_small([nx * 4, ny * 4])
                    + 0.05 * self.perlin_detail([nx * 8, ny * 8])
                )

                grid[y, x] = (noise_val + 1) / 2  # Convert from -1:1 to 0:1 range

        return grid

    def _generate_value_noise(self) -> np.ndarray:
        """Generate noise pattern for asteroid values."""
        grid = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                nx = x / self.width
                ny = y / self.height

                # Combine noise at different scales
                noise_val = 0.6 * self.perlin_large(
                    [nx, ny]
                ) + 0.4 * self.perlin_medium([nx * 2, ny * 2])

                grid[y, x] = (noise_val + 1) / 2  # Convert from -1:1 to 0:1 range

        return grid

    def _add_value_clusters(self, value_grid: np.ndarray) -> np.ndarray:
        """Add high-value clusters to the value grid."""
        # Number of clusters depends on grid size
        num_clusters = int(math.sqrt(self.width * self.height) / 20)

        for _ in range(num_clusters):
            # Random cluster center
            cx = random.randint(0, self.width - 1)
            cy = random.randint(0, self.height - 1)

            # Random cluster size
            radius = random.randint(5, 15)

            # Random value boost
            value_boost = random.uniform(2, 5)

            # Apply boost to asteroids in cluster with distance falloff
            for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                    if value_grid[y, x] > 0:  # If there's an asteroid
                        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if dist <= radius:
                            # Apply boost with falloff from center
                            falloff = (1 - dist / radius) ** 2
                            value_grid[y, x] += value_grid[y, x] * value_boost * falloff

        return value_grid

    def _apply_cellular_automaton(
        self, grid: np.ndarray, iterations: int = 2
    ) -> np.ndarray:
        """Apply cellular automaton rules to smooth the asteroid field."""
        result = grid.copy()

        for _ in range(iterations):
            # Count neighbors for each cell using convolution
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            neighbors = ndimage.convolve(result, kernel, mode="constant")

            # Apply rules similar to Conway's Game of Life but with bias toward structure
            # Birth rule: empty cell becomes asteroid if it has 3-6 neighbors
            birth = (result == 0) & ((neighbors >= 3) & (neighbors <= 6))

            # Survival rule: asteroid survives if it has 2-5 neighbors
            survive = (result == 1) & ((neighbors >= 2) & (neighbors <= 5))

            # Update grid
            result = np.zeros_like(result)
            result[birth | survive] = 1

        return result

    def _apply_weighted_patterns(self, weights: List[float]) -> np.ndarray:
        """Apply multiple patterns weighted by importance."""
        # Normalize weights to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(self.patterns) for _ in self.patterns]

        # Create empty grid
        grid = np.zeros((self.height, self.width))

        # Apply each pattern with its weight
        for pattern_func, weight in zip(self.patterns, weights):
            if weight > 0:
                pattern_grid = pattern_func()
                grid += pattern_grid * weight

        return grid

    def _spiral_pattern(self) -> np.ndarray:
        """Generate a spiral pattern of asteroids."""
        grid = np.zeros((self.height, self.width))
        center_x = self.width // 2
        center_y = self.height // 2

        # Spiral parameters
        max_radius = min(self.width, self.height) * 0.4
        spacing = 4 + random.random() * 4  # Random spacing between spiral arms
        arms = random.randint(1, 4)  # Random number of spiral arms

        # Generate points along spiral arms
        for radius in range(int(max_radius)):
            angle_step = 0.2 / (radius + 1) if radius > 0 else 0.2

            for angle_offset in np.linspace(0, 2 * math.pi, arms, endpoint=False):
                angle = radius / spacing + angle_offset

                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))

                if 0 <= x < self.width and 0 <= y < self.height:
                    # Make spiral arm thicker near center
                    thickness = max(1, int(2 + radius * 0.02))
                    for dx in range(-thickness, thickness + 1):
                        for dy in range(-thickness, thickness + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                dist = math.sqrt(dx * dx + dy * dy)
                                if dist <= thickness:
                                    grid[ny, nx] = 1.0 - (radius / max_radius) * 0.5

        return grid

    def _ring_pattern(self) -> np.ndarray:
        """Generate concentric rings of asteroids."""
        grid = np.zeros((self.height, self.width))
        center_x = self.width // 2
        center_y = self.height // 2

        # Ring parameters
        max_radius = min(self.width, self.height) * 0.4
        num_rings = random.randint(2, 5)

        # Generate rings
        for ring in range(num_rings):
            # Calculate ring radius
            radius = max_radius * (ring + 1) / num_rings
            thickness = max(1, int(radius * 0.08))

            # Draw the ring
            for y in range(self.height):
                for x in range(self.width):
                    dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if abs(dist - radius) <= thickness and random.random() < 0.7:
                        grid[y, x] = 1.0 - (ring / num_rings) * 0.3

        return grid

    def _gradient_pattern(self) -> np.ndarray:
        """Generate a gradient pattern with higher density on one side."""
        grid = np.zeros((self.height, self.width))

        # Choose gradient direction
        direction = random.choice(["horizontal", "vertical", "diagonal"])

        for y in range(self.height):
            for x in range(self.width):
                if direction == "horizontal":
                    # Horizontal gradient (left to right)
                    gradient = x / self.width
                elif direction == "vertical":
                    # Vertical gradient (top to bottom)
                    gradient = y / self.height
                else:
                    # Diagonal gradient
                    gradient = (x / self.width + y / self.height) / 2

                # Apply some noise to make it less uniform
                noise_val = self.perlin_medium([x / self.width, y / self.height])
                value = gradient + (noise_val * 0.2)

                # Normalize and store
                grid[y, x] = max(0, min(1, value))

        return grid

    def _void_pattern(self) -> np.ndarray:
        """Generate a pattern with void areas (negative space)."""
        grid = np.ones((self.height, self.width))  # Start with all filled

        # Create several void areas
        num_voids = random.randint(3, 7)

        for _ in range(num_voids):
            # Random void center and size
            cx = random.randint(0, self.width - 1)
            cy = random.randint(0, self.height - 1)
            radius = random.randint(10, 30)

            # Create void area with soft edges
            for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist <= radius:
                        # Smoother transition at edge
                        edge_factor = min(1.0, (radius - dist) / (radius * 0.3))
                        grid[y, x] *= 1.0 - edge_factor

        return grid


class SymbioteEvolutionGenerator:
    """
    Handles procedural generation of symbiote races and their evolution patterns.
    Based on the symbiote evolution algorithm described in algorithm_design.md.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize the symbiote evolution generator.

        Args:
            width: Width of the field
            height: Height of the field
        """
        self.width = width
        self.height = height

        # Base parameters for all races
        self.base_parameters = {
            "birth_rate": 0.05,  # Base reproduction rate
            "death_rate": 0.03,  # Base death rate
            "mutation_rate": 0.01,  # Base mutation rate
            "aggression": 0.2,  # Base aggression
            "intelligence": 0.5,  # Base intelligence
            "adaptability": 0.5,  # Base adaptability
            "migration_factor": 0.1,  # Tendency to spread
            "resource_efficiency": 0.7,  # Resource utilization efficiency
        }

        # Trait templates for different race types
        self.race_traits = {
            "adaptive": {
                "adaptability": 0.8,
                "mutation_rate": 0.02,
                "intelligence": 0.6,
                "birth_set": {2, 3},
                "survival_set": {3, 4, 5},
            },
            "expansive": {
                "birth_rate": 0.08,
                "migration_factor": 0.3,
                "aggression": 0.3,
                "birth_set": {3, 4},
                "survival_set": {2, 3},
            },
            "selective": {
                "resource_efficiency": 0.9,
                "intelligence": 0.7,
                "birth_rate": 0.04,
                "birth_set": {1, 5},
                "survival_set": {1, 4},
            },
        }

    def generate_race_parameters(
        self, race_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate parameters for a new symbiote race.

        Args:
            race_type: Optional type of race to generate, or random if None

        Returns:
            dict: Parameters for the new race
        """
        # Choose random race type if none specified
        if race_type is None:
            race_type = random.choice(list(self.race_traits.keys()))

        # Start with base parameters
        params = self.base_parameters.copy()

        # Apply race-specific traits
        race_params = self.race_traits.get(race_type, {}).copy()
        params.update(race_params)

        # Add some randomness to parameters
        for key in params:
            if isinstance(params[key], (int, float)) and key != "mutation_rate":
                variation = random.uniform(-0.1, 0.1)
                params[key] = max(0.1, min(1.0, params[key] + variation))

        # Set color based on race type
        if race_type == "adaptive":
            params["color"] = (50, 100, 255)  # Blue
        elif race_type == "expansive":
            params["color"] = (255, 50, 150)  # Magenta
        elif race_type == "selective":
            params["color"] = (255, 165, 0)  # Orange
        else:
            # Generate random color for other types
            params["color"] = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200),
            )

        # Apply the race trait for diversity
        params["trait"] = race_type

        return params

    def generate_initial_population(
        self, asteroid_grid: np.ndarray, race_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate initial population grid for a symbiote race.

        Args:
            asteroid_grid: Binary grid of asteroid locations
            race_params: Parameters for the race

        Returns:
            np.ndarray: Population grid
        """
        population_grid = np.zeros_like(asteroid_grid)

        # Get race trait
        race_trait = race_params.get("trait", "adaptive")

        if race_trait == "adaptive":
            # Adaptive races form small clusters
            num_clusters = random.randint(2, 4)

            for _ in range(num_clusters):
                # Find a random location with asteroids nearby
                attempts = 0
                while attempts < 50:
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)

                    # Check if there are asteroids in the vicinity
                    has_asteroids = False
                    for dy in range(-10, 11):
                        for dx in range(-10, 11):
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < self.width
                                and 0 <= ny < self.height
                                and asteroid_grid[ny, nx] > 0
                            ):
                                has_asteroids = True
                                break
                        if has_asteroids:
                            break

                    if has_asteroids:
                        # Place cluster around this point
                        radius = random.randint(3, 7)
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height:
                                    dist = math.sqrt(dx * dx + dy * dy)
                                    if (
                                        dist <= radius
                                        and random.random()
                                        < 0.7 - (dist / radius) * 0.5
                                    ):
                                        population_grid[ny, nx] = 1
                        break

                    attempts += 1

        elif race_trait == "expansive":
            # Expansive races spread in lines and networks
            start_points = []

            # Find a few starting points near asteroids
            for _ in range(random.randint(2, 4)):
                for _ in range(100):
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)

                    # Check if there are asteroids nearby
                    has_asteroids = False
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < self.width
                                and 0 <= ny < self.height
                                and asteroid_grid[ny, nx] > 0
                            ):
                                has_asteroids = True
                                break
                        if has_asteroids:
                            break

                    if has_asteroids:
                        start_points.append((x, y))
                        break

            # Generate paths between points
            if len(start_points) >= 2:
                for i in range(len(start_points) - 1):
                    x1, y1 = start_points[i]
                    x2, y2 = start_points[i + 1]

                    # Create a path with some randomness
                    path_points = []
                    dx = x2 - x1
                    dy = y2 - y1
                    steps = max(abs(dx), abs(dy))

                    if steps > 0:
                        x_step = dx / steps
                        y_step = dy / steps

                        # Add some waypoints with deviation
                        waypoints = [(x1, y1)]
                        for _ in range(random.randint(1, 3)):
                            t = random.random()
                            wx = x1 + dx * t + random.randint(-10, 10)
                            wy = y1 + dy * t + random.randint(-10, 10)
                            waypoints.append((int(wx), int(wy)))
                        waypoints.append((x2, y2))

                        # Connect waypoints
                        for j in range(len(waypoints) - 1):
                            wx1, wy1 = waypoints[j]
                            wx2, wy2 = waypoints[j + 1]

                            wdx = wx2 - wx1
                            wdy = wy2 - wy1
                            wsteps = max(abs(wdx), abs(wdy))

                            if wsteps > 0:
                                wx_step = wdx / wsteps
                                wy_step = wdy / wsteps

                                for k in range(int(wsteps) + 1):
                                    px = int(wx1 + wx_step * k)
                                    py = int(wy1 + wy_step * k)

                                    if 0 <= px < self.width and 0 <= py < self.height:
                                        # Add some width to the path
                                        width = random.randint(1, 2)
                                        for dy in range(-width, width + 1):
                                            for dx in range(-width, width + 1):
                                                nx, ny = px + dx, py + dy
                                                if (
                                                    0 <= nx < self.width
                                                    and 0 <= ny < self.height
                                                    and random.random() < 0.7
                                                ):
                                                    population_grid[ny, nx] = 1

            # Add some random population points
            for _ in range(int(self.width * self.height * 0.001)):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                population_grid[y, x] = 1

        else:  # selective
            # Selective races concentrate near resource-rich areas

            # Find asteroid clusters using KMeans
            asteroid_points = []
            for y in range(self.height):
                asteroid_points.extend(
                    (x, y) for x in range(self.width) if asteroid_grid[y, x] > 0
                )
            if len(asteroid_points) > 10:
                # Convert to numpy array for KMeans
                points_array = np.array(asteroid_points)

                # Find clusters
                num_clusters = min(3, len(asteroid_points) // 10)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(
                    points_array
                )

                # Place population near cluster centers
                for center in kmeans.cluster_centers_:
                    cx, cy = int(center[0]), int(center[1])

                    # Place a small colony near the center
                    radius = random.randint(3, 8)
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                dist = math.sqrt(dx * dx + dy * dy)
                                if (
                                    dist <= radius
                                    and random.random() < 0.5 - (dist / radius) * 0.3
                                ):
                                    population_grid[ny, nx] = 1
            else:
                # Fallback for small asteroid fields
                for _ in range(int(self.width * self.height * 0.002)):
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)
                    population_grid[y, x] = 1

        return population_grid

    def generate_rare_resources(self, asteroid_grid: np.ndarray) -> np.ndarray:
        """
        Generate rare resource distribution across the asteroid field.

        Args:
            asteroid_grid: Binary grid indicating asteroid presence

        Returns:
            np.ndarray: Grid with rare resource indicators
        """
        return self.rare_generator.generate(asteroid_grid)

    def generate_asteroid_field(self) -> np.ndarray:
        """
        Generate an asteroid field based on the provided asteroid generator.

        Returns:
            np.ndarray: Generated asteroid field
        """
        return self.asteroid_generator.generate()

    def generate_asteroid_clusters(self) -> np.ndarray:
        """
        Generate asteroid clusters based on the provided asteroid generator.

        Returns:
            np.ndarray: Generated asteroid clusters
        """
        return self.asteroid_clusters_generator.generate()

    def generate_population(self, asteroid_grid: np.ndarray) -> np.ndarray:
        """
        Generate population based on the provided population generator.

        Args:
            asteroid_grid: Binary grid indicating asteroid presence

        Returns:
            np.ndarray: Generated population
        """
        return self.population_generator.generate(asteroid_grid)

    def generate_rare_resources(self, asteroid_grid: np.ndarray) -> np.ndarray:
        """
        Generate rare resource distribution across the asteroid field.

        Args:
            asteroid_grid: Binary grid indicating asteroid presence

        Returns:
            np.ndarray: Grid with rare resource indicators
        """
        return self.rare_generator.generate(asteroid_grid)
