"""
MinerEntity class: Represents symbiotic mining races that evolve in the asteroid field.
"""
import itertools
import logging
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import networkx as nx
from perlin_noise import PerlinNoise
import scipy.stats as stats
from sklearn.cluster import KMeans
import pygame

from src.config import *
from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
from src.utils.logging_setup import log_exception

# Forward reference for type hints
class AsteroidField:
    """Type hint for AsteroidField class."""
    pass

class MinerEntity:
    """
    Represents a symbiotic mining race that evolves and adapts to the asteroid field.
    Each race has unique traits, behaviors, and cellular automaton rules for growth.
    """
    
    def __init__(
        self, race_id: int, color: Tuple[int, int, int], 
        birth_set: Optional[Set[int]] = None, 
        survival_set: Optional[Set[int]] = None, 
        initial_density: float = 0.001
    ) -> None:
        """
        Initialize a new mining race.
        
        Args:
            race_id: Unique identifier for the race
            color: RGB color tuple for visualization
            birth_set: Set of neighbor counts that cause cell birth
            survival_set: Set of neighbor counts that allow cell survival
            initial_density: Initial population density (0-1)
        """
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
        self.trait = random.choice(["adaptive", "expansive", "selective"])  # Random starting trait
        
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
        self.mineral_consumption = {
            "common": 0, 
            "rare": 0, 
            "precious": 0, 
            "anomaly": 0
        }

    def _initialize_genome_by_trait(self) -> Dict[str, float]:
        """Initialize the genome based on the race's trait."""
        # Base genome values
        genome = {
            "metabolism_rate": 1.0,  # How efficiently resources are used
            "expansion_drive": 1.0,  # Tendency to expand territory
            "mutation_rate": 0.01,   # Base chance of mutation
            "intelligence": 0.5,     # Ability to optimize behavior
            "aggression_base": 0.2,  # Base aggression level
            "adaptability": 0.5,     # How quickly the race can adapt
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
        if not field or not hasattr(field, 'width') or not hasattr(field, 'height'):
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
                # Find high-value asteroid clusters using KMeans
                asteroid_cells = []
                asteroid_values = []

                for y in range(field.height):
                    for x in range(field.width):
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
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
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
        """Generate entities in a network pattern using NetworkX."""
        # Create a graph
        G = nx.Graph()

        # Add random nodes
        num_nodes = random.randint(3, 6)
        nodes = []
        for _ in range(num_nodes):
            x = random.randint(field.width // 5, field.width * 4 // 5)
            y = random.randint(field.height // 5, field.height * 4 // 5)
            nodes.append((x, y))
            G.add_node((x, y))

        # Connect nodes to form a network
        # For each pair of nodes, add an edge with some probability
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < 0.7:  # 70% chance of connection
                    G.add_edge(nodes[i], nodes[j])

        # Ensure all nodes are connected
        if len(nodes) > 1 and not nx.is_connected(G):
            # Find connected components
            components = list(nx.connected_components(G))
            
            # Connect each component to the largest one
            largest = max(components, key=len)
            for comp in components:
                if comp != largest:
                    # Find the closest pair of nodes between components
                    min_dist = float('inf')
                    closest_pair = None
                    
                    for n1 in comp:
                        for n2 in largest:
                            dist = math.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_pair = (n1, n2)
                    
                    # Connect them
                    if closest_pair:
                        G.add_edge(closest_pair[0], closest_pair[1])

        # Place entities along network paths
        for edge in G.edges():
            node1, node2 = edge
            x1, y1 = node1
            x2, y2 = node2
            
            # Use Bresenham line algorithm to get all points on line
            line_points = []
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            
            while True:
                if 0 <= x1 < field.width and 0 <= y1 < field.height:
                    line_points.append((x1, y1))
                
                if x1 == x2 and y1 == y2:
                    break
                    
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
            
            # Place entities along path with higher density
            for x, y in line_points:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < field.width
                            and 0 <= ny < field.height
                            and field.entity_grid[ny, nx] == 0
                            and random.random() < self.initial_density * 5
                        ):
                            field.entity_grid[ny, nx] = self.race_id

        # Add small clusters around nodes
        for node in G.nodes():
            x, y = node
            radius = random.randint(3, 6)
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < field.width and 0 <= ny < field.height:
                        dist = math.sqrt(dx ** 2 + dy ** 2)
                        if (
                            dist <= radius
                            and field.entity_grid[ny, nx] == 0
                            and random.random() < self.initial_density * 2 * (1 - dist/radius)
                        ):
                            field.entity_grid[ny, nx] = self.race_id

    def process_minerals(self, minerals: Dict[str, int]) -> None:
        """Process minerals for evolution and hunger reduction."""
        if not minerals:
            return

        # Track mineral consumption
        for mineral_type, amount in minerals.items():
            self.mineral_consumption[mineral_type] += amount
        
        # Use the evolution algorithm
        new_population, new_aggression, mutations = self.evolution_algorithm.process_mineral_feeding(
            self.race_id, minerals, self.population, self.aggression
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
        new_birth_set, new_survival_set = self.evolution_algorithm.generate_cellular_automaton_rules(
            self.race_id, self.hunger, self.genome
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
        points = np.column_stack((entity_locations[1], entity_locations[0]))  # x, y format
        
        # If only a few points, return simple stats
        if len(points) < 5:
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            distances = np.sqrt(np.sum((points - np.array([center_x, center_y])) ** 2, axis=1))
            
            self.territory_center = (int(center_x), int(center_y))
            self.territory_radius = int(np.max(distances))
            self.territory_density = len(points) / (math.pi * self.territory_radius ** 2)
            
            return {
                "center": self.territory_center,
                "radius": self.territory_radius,
                "density": self.territory_density,
                "resource_access": 0,
                "fragmentation": 0,
            }
        
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

    # Helper method for territory analysis
    def _extracted_from_evolve_26(self, k: int, points: np.ndarray, entity_locations: np.ndarray) -> Dict[str, Any]:
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
        area = math.pi * self.territory_radius ** 2
        self.territory_density = len(main_cluster_points) / area if area > 0 else 0

        # Calculate resource access (how many asteroids are within territory)
        resource_access = 0
        if self.field:
            for y in range(
                max(0, int(main_center[1] - self.territory_radius)),
                min(self.field.height, int(main_center[1] + self.territory_radius + 1))
            ):
                for x in range(
                    max(0, int(main_center[0] - self.territory_radius)),
                    min(self.field.width, int(main_center[0] + self.territory_radius + 1))
                ):
                    if (
                        (x - main_center[0]) ** 2 + (y - main_center[1]) ** 2
                        <= self.territory_radius ** 2
                        and self.field.grid[y, x] > 0
                    ):
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
        # Default to feeding
        new_behavior = "feeding"
        
        # If hungry, prioritize seeking food
        if self.hunger > 0.7:
            new_behavior = "feeding"
        # If well-fed but low population, focus on expansion
        elif self.hunger < 0.3 and self.population < 50:
            new_behavior = "expanding"
        # If high population and low resource density, migrate
        elif self.population > 200:
            # Calculate resource density within territory
            if self.territory_center and self.territory_radius:
                cx, cy = self.territory_center
                resource_count = 0
                cells_checked = 0
                
                for y in range(
                    max(0, cy - self.territory_radius),
                    min(field.height, cy + self.territory_radius + 1)
                ):
                    for x in range(
                        max(0, cx - self.territory_radius),
                        min(field.width, cx + self.territory_radius + 1)
                    ):
                        if (x - cx) ** 2 + (y - cy) ** 2 <= self.territory_radius ** 2:
                            cells_checked += 1
                            if field.grid[y, x] > 0:
                                resource_count += 1
                
                resource_density = resource_count / cells_checked if cells_checked > 0 else 0
                
                if resource_density < 0.05:
                    new_behavior = "migrating"
        
        # If behavior changed, log it
        if new_behavior != self.current_behavior:
            logging.info(f"Race {self.race_id} behavior changed: {self.current_behavior} -> {new_behavior}")
            self.current_behavior = new_behavior

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
            self.genome["metabolism_rate"] = min(2.0, self.genome["metabolism_rate"] * 1.1)
            
        elif self.trait == "expansive":
            # Expansive races improve expansion drive and aggression
            self.genome["expansion_drive"] = min(2.0, self.genome["expansion_drive"] * 1.2)
            self.genome["aggression_base"] = min(0.8, self.genome["aggression_base"] * 1.1)
            
        elif self.trait == "selective":
            # Selective races improve intelligence and efficiency
            self.genome["intelligence"] = min(2.0, self.genome["intelligence"] * 1.15)
            self.mining_efficiency = min(0.9, self.mining_efficiency * 1.1)
            
        # Random mutations to other traits
        for key in self.genome:
            if random.random() < self.genome["mutation_rate"]:
                mutation_strength = random.normalvariate(1.0, 0.1)
                self.genome[key] = max(0.1, min(2.0, self.genome[key] * mutation_strength))

    def draw(self, surface: pygame.Surface) -> None:
        """Draw race specific information on the surface."""
        # We'll implement this later when we finalize the UI module
        pass

    def draw_population(self, surface: pygame.Surface, index: int) -> None:
        """Draw population stats on the surface."""
        # We'll implement this later when we finalize the UI module
        pass