"""
MinerEntity class: Represents symbiotic mining races that evolve in the asteroid field.
"""

# Standard library imports
import itertools
import logging
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

# Third-party library imports
import networkx as nx
import numpy as np
import pygame
import scipy.stats as stats
from sklearn.cluster import KMeans

# Optional dependencies
try:
    from perlin_noise import PerlinNoise

    PERLIN_AVAILABLE = True
except ImportError:
    PERLIN_AVAILABLE = False
    print("PerlinNoise package is not available. Using fallback noise generator.")

# Local application imports
from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
from src.config import (
    COLOR_RACE_1,
    COLOR_RACE_2,
    COLOR_RACE_3,
    COLOR_ENTITY_DEFAULT,
    COLOR_ENTITY_FEEDING,
    COLOR_ENTITY_EXPANDING,
    COLOR_ENTITY_MIGRATING,
    COLOR_ENTITY_AGGRESSIVE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    CELL_SIZE as GRID_CELL_SIZE,
)
from src.entities.base_entity import BaseEntity
from src.utils.logging_setup import log_exception


# Forward reference for type hints
class AsteroidField:
    """Type hint for AsteroidField class."""

    pass


class MinerEntity(BaseEntity):
    """
    Represents a symbiotic mining race that evolves and adapts to the asteroid field.
    Each race has unique traits, behaviors, and cellular automaton rules for growth.
    Inherits from BaseEntity to leverage common entity functionality.
    """

    def __init__(
        self,
        race_id: int,
        color: Optional[Tuple[int, int, int]] = None,
        birth_set: Optional[Set[int]] = None,
        survival_set: Optional[Set[int]] = None,
        initial_density: float = 0.001,
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize a new mining race.

        Args:
            race_id: Unique identifier for the race
            color: RGB color tuple for visualization (optional, defaults to race-specific color)
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            initial_density: Initial population density (0-1)
            position: Initial position as (x, y) tuple
        """
        # Assign race-specific color if not provided
        if color is None:
            if race_id == 1:
                color = COLOR_RACE_1
            elif race_id == 2:
                color = COLOR_RACE_2
            elif race_id == 3:
                color = COLOR_RACE_3
            else:
                # Default color for any other race IDs
                color = (
                    random.randint(50, 200),
                    random.randint(50, 200),
                    random.randint(50, 200),
                )

        # Call the parent class constructor
        super().__init__(
            entity_id=str(race_id), entity_type="miner", color=color, position=position
        )

        # Initialize race parameters
        self.race_id = race_id
        self.color = color
        self.birth_set = birth_set or {3}
        self.survival_set = survival_set or {2, 3}
        self.initial_density = initial_density

        # Basic attributes
        self.aggression = 0.2  # Default aggression level (0-1)
        self.hunger = 0.0  # Hunger level (0-1)
        self.hunger_rate = 0.01  # Rate at which hunger increases per turn
        self.trait = random.choice(
            ["adaptive", "expansive", "selective"]
        )  # Random starting trait

        # Logging
        logging.info(f"Created race {race_id} with trait {self.trait}")

        # Population tracking
        self.population = 0
        self.fed_this_turn = False
        self.last_income = 0
        self.income_history = []
        self.population_history = []

        # Evolution metrics
        self.evolution_points = 0
        self.evolution_threshold = 100
        self.evolution_stage = 1
        self.current_behavior = "feeding"  # Default behavior
        self.mining_efficiency = 0.5  # Base mining efficiency

        # Initialize genome based on trait
        self.genome = self._initialize_genome_by_trait()

        # Spatial analysis data
        self.territory_center = None
        self.territory_radius = 0
        self.territory_density = 0
        self.colony_data = {}
        self.field = None  # Reference to the field will be set when added

        # Initialize the advanced algorithm
        self.evolution_algorithm = SymbioteEvolutionAlgorithm(
            initial_aggression=self.aggression,
            growth_rate=0.05 * self.genome["metabolism_rate"],
            base_mutation_rate=self.genome["mutation_rate"],
            learning_enabled=True,
        )

        # Track mineral consumption for the algorithm
        self.mineral_consumption = {"common": 0, "rare": 0, "precious": 0, "anomaly": 0}

    def _initialize_genome_by_trait(self) -> Dict[str, float]:
        """Initialize the genome based on the race's trait."""
        # Base genome values
        genome = {
            "metabolism_rate": 1.0,  # How efficiently resources are used
            "expansion_drive": 1.0,  # Tendency to expand territory
            "mutation_rate": 0.01,  # Base chance of mutation
            "intelligence": 0.5,  # Ability to optimize behavior
            "aggression_base": 0.2,  # Base aggression level
            "adaptability": 0.5,  # How quickly the race can adapt
        }

        # Trait-specific modifications
        if self.trait == "adaptive":
            genome["metabolism_rate"] = 1.2
            genome["mutation_rate"] = 0.02
            genome["adaptability"] = 0.8
            genome["aggression_base"] = 0.15

        elif self.trait == "expansive":
            genome["expansion_drive"] = 1.4
            genome["metabolism_rate"] = 0.9
            genome["aggression_base"] = 0.3

        elif self.trait == "selective":
            genome["intelligence"] = 0.7
            genome["metabolism_rate"] = 1.1
            genome["expansion_drive"] = 0.9
            genome["aggression_base"] = 0.2

        return genome

    def populate(self, field: AsteroidField) -> None:
        """Populate the field with this race's symbiotes using unique patterns based on traits"""
        # Safe check to ensure field is valid
        if not field or not hasattr(field, "width") or not hasattr(field, "height"):
            logging.error("Invalid field provided for population")
            return

        # Store field as instance variable
        self.field = field

        # Each race has a distinct settlement pattern
        try:
            if self.trait == "adaptive":  # Blue race - settle in clusters
                # Use Perlin noise to create organic-looking clusters
                if PERLIN_AVAILABLE:
                    noise = PerlinNoise(octaves=4, seed=random.randint(1, 1000))
                else:
                    # Fallback noise generator
                    def noise(x, y):
                        return (math.sin(x * 0.1) + math.cos(y * 0.1)) * 0.5

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
                # Find high-value asteroid clusters using KMeans
                asteroid_cells = []
                asteroid_values = []

                for y, x in itertools.product(range(field.height), range(field.width)):
                    if field.grid[y, x] > 0:
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
            log_exception(e)

    def _generate_random_nodes(self, field: AsteroidField) -> None:
        """Generate entities in a network pattern using NetworkX.

        Creates a connected network of colony nodes and places entities along paths
        between nodes and in clusters around node centers.

        Args:
            field: The asteroid field where entities will be placed
        """
        # Step 1: Generate colony network
        colony_graph, colony_centers = self._create_colony_network(field)

        # Step 2: Place entities along network paths
        self._place_entities_on_paths(colony_graph, field)

        # Step 3: Create clusters around colony centers
        self._create_colony_clusters(colony_graph, field)

    def _create_colony_network(
        self, field: AsteroidField
    ) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
        """Create a connected network of colony nodes.

        Returns:
            Tuple containing (NetworkX graph, list of node positions)
        """
        # Create empty graph
        colony_graph = nx.Graph()

        # Determine number of colony centers (3-8) using Poisson distribution
        num_nodes = stats.poisson.rvs(4)  # Mean of 4 nodes
        num_nodes = max(3, min(8, num_nodes))  # Ensure between 3-8 nodes

        # Generate node positions
        colony_centers = []
        for _ in range(num_nodes):
            # Calculate normalized bounds for truncated normal distribution
            x_mean, x_std = field.width // 2, field.width // 4
            x_min, x_max = field.width // 5, field.width * 4 // 5
            x_a = (x_min - x_mean) / x_std
            x_b = (x_max - x_mean) / x_std

            # Generate x position
            x = int(stats.truncnorm.rvs(x_a, x_b, loc=x_mean, scale=x_std))

            # Similar calculation for y position
            y_mean, y_std = field.height // 2, field.height // 4
            y_min, y_max = field.height // 5, field.height * 4 // 5
            y_a = (y_min - y_mean) / y_std
            y_b = (y_max - y_mean) / y_std

            # Generate y position
            y = int(stats.truncnorm.rvs(y_a, y_b, loc=y_mean, scale=y_std))

            colony_centers.append((x, y))
            colony_graph.add_node((x, y))

        # Connect nodes with 70% probability
        self._connect_nodes(colony_graph, colony_centers)

        # Ensure the graph is fully connected
        self._ensure_connected_graph(colony_graph, colony_centers)

        return colony_graph, colony_centers

    def _connect_nodes(self, graph: nx.Graph, nodes: List[Tuple[int, int]]) -> None:
        """Connect nodes with a 70% probability."""
        CONNECTION_PROBABILITY = 0.7

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < CONNECTION_PROBABILITY:
                    graph.add_edge(nodes[i], nodes[j])

    def _ensure_connected_graph(
        self, graph: nx.Graph, nodes: List[Tuple[int, int]]
    ) -> None:
        """Ensure the graph is fully connected by connecting components."""
        # Skip if already connected or empty
        if len(nodes) <= 1 or nx.is_connected(graph):
            return

        # Find connected components
        components = list(nx.connected_components(graph))

        # Connect each component to the largest one
        largest_component = max(components, key=len)

        for component in components:
            if component != largest_component:
                self._connect_components(graph, component, largest_component)

    def _connect_components(
        self, graph: nx.Graph, component1: Set, component2: Set
    ) -> None:
        """Connect two components by adding an edge between their closest nodes."""
        min_dist = float("inf")
        closest_pair = None

        # Find the closest pair of nodes between components
        for node1 in component1:
            for node2 in component2:
                # Calculate Euclidean distance
                dist = math.sqrt(
                    (node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2
                )

                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (node1, node2)

        # Connect the closest nodes
        if closest_pair:
            graph.add_edge(closest_pair[0], closest_pair[1])

    def _place_entities_on_paths(self, graph: nx.Graph, field: AsteroidField) -> None:
        """Place entities along the paths between colony nodes."""
        PATH_DENSITY_MULTIPLIER = 5.0

        for edge in graph.edges():
            node1, node2 = edge

            # Get points along the path using Bresenham's algorithm
            line_points = self._get_line_points(node1, node2, field)

            # Place entities along and around the path
            for point in line_points:
                x, y = point

                # Add entities in a small area around each path point
                for dy, dx in itertools.product(range(-1, 2), range(-1, 2)):
                    nx, ny = x + dx, y + dy

                    # Check if position is valid and empty
                    if (
                        self._is_valid_position(nx, ny, field)
                        and field.entity_grid[ny, nx] == 0
                        and random.random()
                        < self.initial_density * PATH_DENSITY_MULTIPLIER
                    ):
                        field.entity_grid[ny, nx] = self.race_id

    def _get_line_points(
        self, start: Tuple[int, int], end: Tuple[int, int], field: AsteroidField
    ) -> List[Tuple[int, int]]:
        """Get points along a line using Bresenham's line algorithm."""
        x1, y1 = start
        x2, y2 = end
        line_points = []

        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        # Current position
        x, y = x1, y1

        while True:
            # Add point if within field boundaries
            if self._is_valid_position(x, y, field):
                line_points.append((x, y))

            # Exit condition
            if x == x2 and y == y2:
                break

            # Update position
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return line_points

    def _create_colony_clusters(self, graph: nx.Graph, field: AsteroidField) -> None:
        """Create clusters of entities around colony centers."""
        CLUSTER_DENSITY_MULTIPLIER = 2.0

        for node in graph.nodes():
            x, y = node
            radius = random.randint(3, 6)

            # Create a circular cluster around the node
            for dy, dx in itertools.product(
                range(-radius, radius + 1), range(-radius, radius + 1)
            ):
                nx, ny = x + dx, y + dy

                # Check if position is valid and empty
                if (
                    self._is_valid_position(nx, ny, field)
                    and field.entity_grid[ny, nx] == 0
                ):
                    # Calculate distance from center
                    dist = math.sqrt(dx**2 + dy**2)

                    # Higher probability closer to center
                    if (
                        dist <= radius
                        and random.random()
                        < self.initial_density
                        * CLUSTER_DENSITY_MULTIPLIER
                        * (1 - dist / radius)
                    ):
                        field.entity_grid[ny, nx] = self.race_id

    def _is_valid_position(self, x: int, y: int, field: AsteroidField) -> bool:
        """Check if a position is within field boundaries."""
        return 0 <= x < field.width and 0 <= y < field.height

    def process_minerals(self, minerals: Dict[str, int]) -> None:
        """Process minerals for evolution and hunger reduction."""
        if not minerals:
            return

        # Track mineral consumption
        for mineral_type, amount in minerals.items():
            self.mineral_consumption[mineral_type] += amount

        # Use the evolution algorithm
        new_population, new_aggression, mutations = (
            self.evolution_algorithm.process_mineral_feeding(
                self.race_id, minerals, self.population, self.aggression
            )
        )

        # Apply results
        self.population = int(new_population)
        self.aggression = new_aggression

        # Apply any mutations to the genome
        self.apply_mutations(mutations)

        # Mark as fed this turn
        self.fed_this_turn = True

    def apply_mutations(self, mutations: List[Dict[str, Any]]) -> None:
        """Apply mutations to the genome."""
        if not mutations:
            return

        for mutation in mutations:
            attr = mutation["attribute"]
            magnitude = mutation["magnitude"]

            if attr in self.genome:
                old_value = self.genome[attr]
                self.genome[attr] = max(0.1, min(2.0, old_value * magnitude))

                # Log significant mutations
                if abs(old_value - self.genome[attr]) > 0.1:
                    mutation_type = "increased" if magnitude > 1 else "decreased"
                    logging.info(
                        f"Race {self.race_id} {mutation_type} {attr} "
                        f"from {old_value:.2f} to {self.genome[attr]:.2f}"
                    )

    def mutate(self) -> None:
        """Mutate the entity's genome with a small probability using scipy.stats distributions."""
        for key in self.genome:
            # Use scipy.stats for more sophisticated mutation probability
            # Beta distribution gives more control over mutation frequency
            mutation_probability = stats.beta.rvs(
                2, 18
            )  # Using scipy.stats beta distribution

            if random.random() < mutation_probability:  # Typically around 10% chance
                # Use scipy.stats normal distribution for mutation strength
                mutation_strength = stats.norm.rvs(loc=1.0, scale=0.1)
                self.genome[key] = max(
                    0.1, min(2.0, self.genome[key] * mutation_strength)
                )

    def update_hunger(self, income: int) -> float:
        """Update hunger based on income and return updated aggression level."""
        if income > 0:
            # Income reduces hunger
            hunger_reduction = min(self.hunger, 0.2 * (income / 100))
            self.hunger = max(0.0, self.hunger - hunger_reduction)
            self.fed_this_turn = True
        else:
            # Hunger increases each turn without food
            self.hunger = min(1.0, self.hunger + self.hunger_rate)
            self.fed_this_turn = False

        # Adjust aggression based on hunger
        self.aggression = max(
            0.1, min(0.9, self.genome["aggression_base"] + self.hunger * 0.5)
        )

        return self.aggression

    def process_evolution(self, field: AsteroidField) -> None:
        """Process evolution based on environmental factors."""
        if self.population < 10:
            return  # Not enough population to evolve

        # Analyze territory
        self.analyze_territory(field)

        # Get new CA rules from the algorithm
        new_birth_set, new_survival_set = (
            self.evolution_algorithm.generate_cellular_automaton_rules(
                self.race_id, self.hunger, self.genome
            )
        )

        # Apply new rules with some chance of mutation
        if random.random() < self.genome["mutation_rate"] * 2:
            possible_rules = [1, 2, 3, 4, 5, 6, 7, 8]

            # Birth rules mutation
            if random.random() < 0.3:
                rule = random.choice(possible_rules)
                if rule in self.birth_set:
                    self.birth_set.remove(rule)
                else:
                    self.birth_set.add(rule)

            # Survival rules mutation
            if random.random() < 0.3:
                rule = random.choice(possible_rules)
                if rule in self.survival_set:
                    self.survival_set.remove(rule)
                else:
                    self.survival_set.add(rule)

        # Update behavior based on conditions
        self._update_behavior(field)

    def analyze_territory(self, field: AsteroidField) -> Dict[str, Any]:
        """Analyze the race's territory."""
        # Create a mask for this race
        race_mask = field.entity_grid == self.race_id

        if np.sum(race_mask) == 0:
            # Race is extinct
            return {
                "center": None,
                "radius": 0,
                "density": 0,
                "resource_access": 0,
                "fragmentation": 0,
            }

        # Find all entity locations
        entity_locations = np.where(race_mask)
        points = np.column_stack(
            (entity_locations[1], entity_locations[0])
        )  # x, y format

        # If only a few points, return simple stats
        if len(points) < 5:
            return self._territory_handler(points)
        # For larger populations, use KMeans to identify colonies
        try:
            # Determine k based on population size
            population = len(points)
            k = min(8, max(1, population // 100))

            return self._extracted_from_evolve_26(k, points, entity_locations)

        except Exception as e:
            logging.error(f"Error in territory analysis: {e}")
            log_exception(e)
            return {
                "center": None,
                "radius": 0,
                "density": 0,
                "resource_access": 0,
                "fragmentation": 0,
            }

    # TODO Rename this here and in `analyze_territory`
    def _territory_handler(self, points):
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        distances = np.sqrt(
            np.sum((points - np.array([center_x, center_y])) ** 2, axis=1)
        )

        # Use scipy.stats to analyze the distribution of entity distances
        # Calculate statistical properties of the territory
        distance_stats = {
            "mean": np.mean(distances),
            "median": np.median(distances),
            "std": np.std(distances),
            "skewness": stats.skew(distances),  # Using scipy.stats for skewness
            "kurtosis": stats.kurtosis(distances),  # Using scipy.stats for kurtosis
        }

        # Store territory metrics based on statistical analysis
        self.territory_center = (int(center_x), int(center_y))
        self.territory_radius = int(np.max(distances))
        self.territory_density = len(points) / (math.pi * self.territory_radius**2)

        # Use skewness to determine if entities are clustered toward center or edge
        # Negative skew means more entities near the edge
        self.territory_centrality = -distance_stats["skewness"]

        return {
            "center": self.territory_center,
            "radius": self.territory_radius,
            "density": self.territory_density,
            "resource_access": 0,
            "fragmentation": 0,
            "centrality": self.territory_centrality,
            "distance_stats": distance_stats,
        }

    # Helper method for territory analysis
    def _extracted_from_evolve_26(
        self, k: int, points: np.ndarray, entity_locations: np.ndarray
    ) -> Dict[str, Any]:
        """Helper method for territory analysis using KMeans clustering."""
        kmeans = KMeans(n_clusters=k).fit(points)
        clusters = kmeans.labels_
        centers = kmeans.cluster_centers_

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

        # Calculate territory density
        area = math.pi * self.territory_radius**2
        self.territory_density = len(main_cluster_points) / area if area > 0 else 0

        # Calculate resource access (how many asteroids are within territory)
        resource_access = 0
        if self.field:
            for y, x in itertools.product(
                range(
                    max(0, int(main_center[1] - self.territory_radius)),
                    min(
                        self.field.height,
                        int(main_center[1] + self.territory_radius + 1),
                    ),
                ),
                range(
                    max(0, int(main_center[0] - self.territory_radius)),
                    min(
                        self.field.width,
                        int(main_center[0] + self.territory_radius + 1),
                    ),
                ),
            ):
                if (x - main_center[0]) ** 2 + (
                    y - main_center[1]
                ) ** 2 <= self.territory_radius**2 and self.field.grid[y, x] > 0:
                    resource_access += 1

        # Calculate fragmentation (ratio of number of clusters to population)
        fragmentation = k / len(points) if len(points) > 0 else 1.0

        return {
            "center": self.territory_center,
            "radius": self.territory_radius,
            "density": self.territory_density,
            "resource_access": resource_access,
            "fragmentation": fragmentation,
            "clusters": k,
            "cluster_sizes": cluster_sizes,
            "centers": centers.tolist(),
        }

    def _update_behavior(self, field: AsteroidField) -> None:
        """Update behavior based on conditions."""
        # Base probability for feeding increases with hunger
        # Use sigmoid function from scipy.stats.logistic.cdf
        hunger_factor = stats.logistic.cdf(self.hunger, loc=0.5, scale=0.1)
        behavior_probabilities = {
            "expanding": 0.0,
            "migrating": 0.0,
            "aggressive": 0.0,
            "feeding": hunger_factor,
        }
        # Base probability for expanding increases with low hunger and low population
        if self.hunger < 0.5 and self.population < 100:
            # Use beta distribution to model expansion probability
            # Higher when hunger is low and population is low
            expansion_factor = stats.beta.pdf(self.hunger, 2, 5) * stats.beta.pdf(
                self.population / 200, 2, 5
            )
            behavior_probabilities["expanding"] = expansion_factor

        # Calculate resource density if we have territory data
        if self.territory_center and self.territory_radius:
            self.calculate_resource_density(field, behavior_probabilities)
        # Normalize probabilities
        total_prob = sum(behavior_probabilities.values())
        if total_prob > 0:
            for value in behavior_probabilities.values():
                value /= total_prob

        # Choose behavior based on highest probability
        new_behavior = max(behavior_probabilities, key=behavior_probabilities.get)

        # Default to feeding if all probabilities are zero
        if behavior_probabilities[new_behavior] == 0:
            new_behavior = "feeding"

        # If behavior changed, log it
        if new_behavior != self.current_behavior:
            logging.info(
                f"Race {self.race_id} behavior changed: {self.current_behavior} -> {new_behavior}"
            )
            self.current_behavior = new_behavior

    def calculate_resource_density(self, field, behavior_probabilities):
        """Calculate resource density in territory and update migration probability."""
        cx, cy = self.territory_center
        resource_count = 0
        cells_checked = 0

        # Examine cells within territory radius
        for y, x in itertools.product(
            range(
                max(0, cy - self.territory_radius),
                min(field.height, cy + self.territory_radius + 1),
            ),
            range(
                max(0, cx - self.territory_radius),
                min(field.width, cx + self.territory_radius + 1),
            ),
        ):
            if (x - cx) ** 2 + (y - cy) ** 2 <= self.territory_radius**2:
                cells_checked += 1
                if field.grid[y, x] > 0:
                    resource_count += 1

        # Migration probability increases with low resource density and high population
        if self.population > 50 and cells_checked > 0:
            # Calculate resource density
            resource_density = resource_count / cells_checked

            # Use exponential distribution to model migration probability
            # Higher when resources are scarce
            migration_factor = stats.expon.pdf(resource_density * 20) * stats.norm.cdf(
                self.population, loc=150, scale=50
            )
            behavior_probabilities["migrating"] = migration_factor

    def evolve(self) -> Dict[str, Any]:
        """
        Evolve the race to the next stage.
        Returns metrics about the evolution.
        """
        original_traits = dict(self.genome)

        # Analyze territory
        metrics = self.analyze_territory(self.field)

        # Increase evolution stage
        self.evolution_stage += 1

        # Evolve genome based on conditions
        self._evolve_genome()

        # Log evolution details
        for key, value in self.genome.items():
            old_val = original_traits.get(key, 0)
            if abs(value - old_val) > 0.1:
                direction = "increased" if value > old_val else "decreased"
                logging.info(f"  - {key} {direction} from {old_val:.2f} to {value:.2f}")

        return metrics

    def _evolve_genome(self) -> None:
        """Evolve genome based on conditions and stage."""
        # Adjust mutation rate based on stage
        self.genome["mutation_rate"] = min(0.05, self.genome["mutation_rate"] * 1.1)

        # Specialize based on trait
        if self.trait == "adaptive":
            # Adaptive races improve adaptability and metabolism
            self.genome["adaptability"] = min(2.0, self.genome["adaptability"] * 1.15)
            self.genome["metabolism_rate"] = min(
                2.0, self.genome["metabolism_rate"] * 1.1
            )

        elif self.trait == "expansive":
            # Expansive races improve expansion drive and aggression
            self.genome["expansion_drive"] = min(
                2.0, self.genome["expansion_drive"] * 1.2
            )
            self.genome["aggression_base"] = min(
                0.8, self.genome["aggression_base"] * 1.1
            )

        elif self.trait == "selective":
            # Selective races improve intelligence and efficiency
            self.genome["intelligence"] = min(2.0, self.genome["intelligence"] * 1.15)
            self.mining_efficiency = min(0.9, self.mining_efficiency * 1.1)

        # Random mutations to other traits using scipy.stats distributions
        for key in self.genome:
            # Use scipy.stats beta distribution for more controlled mutation rate
            if random.random() < stats.beta.rvs(self.genome["mutation_rate"] * 20, 2):
                # Use scipy.stats normal distribution for mutation strength
                mutation_strength = stats.norm.rvs(loc=1.0, scale=0.1)
                self.genome[key] = max(
                    0.1, min(2.0, self.genome[key] * mutation_strength)
                )

    def draw(self, surface: pygame.Surface) -> None:
        """Draw race specific information on the surface."""
        # Get the appropriate color based on current behavior
        behavior_color = COLOR_ENTITY_DEFAULT  # Default gray

        if self.current_behavior == "feeding":
            behavior_color = COLOR_ENTITY_FEEDING  # Green for feeding
        elif self.current_behavior == "expanding":
            behavior_color = COLOR_ENTITY_EXPANDING  # Blue for expanding
        elif self.current_behavior == "migrating":
            behavior_color = COLOR_ENTITY_MIGRATING  # Orange for migrating
        elif self.current_behavior == "aggressive":
            behavior_color = COLOR_ENTITY_AGGRESSIVE  # Red for aggressive

        # Draw entity with behavior color
        if self.territory_center:
            cx, cy = self.territory_center
            pygame.draw.circle(
                surface, behavior_color, (cx * GRID_CELL_SIZE, cy * GRID_CELL_SIZE), 5
            )

        # Draw status bar at the top of the screen using WINDOW_WIDTH for positioning
        self.draw_status_bar(surface)

    def draw_population(self, surface: pygame.Surface, index: int) -> None:
        """Draw population stats on the surface."""
        if (
            not self.territory_center
            or not hasattr(self, "population")
            or self.population <= 0
        ):
            return

        # Calculate statistical properties for visualization
        if hasattr(self, "entity_locations") and self.entity_locations is not None:
            entity_count = np.sum(self.entity_locations > 0)
            if entity_count > 5:
                self._draw_population_handler(index, surface)

    def _draw_population_handler(self, index, surface):
        # Use scipy.stats to generate a kernel density estimate for visualization
        # This would show the distribution of entities in the territory
        points = np.column_stack(np.where(self.entity_locations > 0))

        # Draw basic stats
        # Position stats in the bottom-left corner with proper spacing
        x_offset = 20
        y_offset = WINDOW_HEIGHT - 150 + (index * 30)

        # Alternative positioning option using WINDOW_WIDTH (right side of screen)
        # Uncomment to use right-side positioning instead
        # x_offset = WINDOW_WIDTH - 300
        # y_offset = 20 + (index * 30)

        # Draw race color indicator with behavior color
        race_color = self.color

        # Add behavior color indicator
        behavior_color = COLOR_ENTITY_DEFAULT
        if self.current_behavior == "feeding":
            behavior_color = COLOR_ENTITY_FEEDING
        elif self.current_behavior == "expanding":
            behavior_color = COLOR_ENTITY_EXPANDING
        elif self.current_behavior == "migrating":
            behavior_color = COLOR_ENTITY_MIGRATING
        elif self.current_behavior == "aggressive":
            behavior_color = COLOR_ENTITY_AGGRESSIVE

        # Draw race color and behavior indicator
        pygame.draw.rect(surface, race_color, (x_offset, y_offset, 15, 15))
        pygame.draw.rect(surface, behavior_color, (x_offset + 20, y_offset, 15, 15))

        # Population statistics using scipy.stats
        if len(points) > 10:
            self._confidence_interval_handler(points, surface, x_offset, y_offset)

    def _confidence_interval_handler(self, points, surface, x_offset, y_offset):
        # Calculate confidence intervals using scipy.stats
        density_ci = stats.norm.interval(
            0.95,
            loc=self.territory_density,
            scale=self.territory_density / math.sqrt(len(points)),
        )

        # Format and display statistics with confidence interval and behavior
        ci_lower, ci_upper = density_ci
        stats_text = f"Race {self.race_id}: Pop {self.population} | Behavior: {self.current_behavior} | Density: {self.territory_density:.2f} [{ci_lower:.2f}-{ci_upper:.2f}]"
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 12)
        text_surface = font.render(stats_text, True, (255, 255, 255))
        surface.blit(text_surface, (x_offset + 20, y_offset))

    def draw_status_bar(self, surface: pygame.Surface) -> None:
        """Draw a status bar at the top of the screen showing entity metrics."""
        if self.population <= 0:
            return

        # Use WINDOW_WIDTH to position the status bar
        bar_width = 200
        bar_height = 15
        x_position = WINDOW_WIDTH - bar_width - 20 - (self.race_id * 220)
        y_position = 10

        # Draw background
        pygame.draw.rect(
            surface, (50, 50, 50), (x_position, y_position, bar_width, bar_height)
        )

        # Calculate metrics
        hunger_width = int(self.hunger * bar_width)
        aggression_width = int(self.aggression * bar_width)

        # Draw hunger bar (red)
        pygame.draw.rect(
            surface,
            (200, 0, 0),
            (x_position, y_position, hunger_width, bar_height // 2),
        )

        # Draw aggression bar (orange)
        pygame.draw.rect(
            surface,
            (255, 165, 0),
            (
                x_position,
                y_position + bar_height // 2,
                aggression_width,
                bar_height // 2,
            ),
        )

        # Draw race indicator
        pygame.draw.rect(
            surface, self.color, (x_position - 20, y_position, 15, bar_height)
        )

        # Draw behavior indicator
        behavior_color = COLOR_ENTITY_DEFAULT
        if self.current_behavior == "feeding":
            behavior_color = COLOR_ENTITY_FEEDING
        elif self.current_behavior == "expanding":
            behavior_color = COLOR_ENTITY_EXPANDING
        elif self.current_behavior == "migrating":
            behavior_color = COLOR_ENTITY_MIGRATING
        elif self.current_behavior == "aggressive":
            behavior_color = COLOR_ENTITY_AGGRESSIVE

        pygame.draw.rect(
            surface,
            behavior_color,
            (x_position - 20, y_position + bar_height + 5, 15, bar_height),
        )

        # Draw text
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 10)
        text = f"Race {self.race_id}: {self.population}"
        text_surface = font.render(text, True, (255, 255, 255))
        surface.blit(text_surface, (x_position, y_position + bar_height + 5))
