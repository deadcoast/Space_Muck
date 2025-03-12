"""
SymbioteEvolutionAlgorithm: Advanced algorithm for symbiote race evolution.

This module provides the core algorithm for simulating symbiote race evolution
based on mineral consumption, environmental factors, and interactions between
different colonies. It uses cellular automaton principles for growth simulation.
"""

import itertools

# Standard library imports
from typing import Tuple, List, cast

# Third-party imports
import numpy as np
from numpy.random import Generator, PCG64

# Handle optional dependencies gracefully
try:
    import scipy.ndimage as ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    import logging

    logging.warning(
        "scipy not available. Some symbiote evolution features may be limited."
    )


class SymbioteEvolutionAlgorithm:
    """
    Manages symbiote evolution based on mineral consumption and environmental factors.
    """

    def __init__(
        self,
        initial_aggression=0.2,
        growth_rate=0.05,
        base_mutation_rate=0.01,
        carrying_capacity=100,
        learning_enabled=True,
    ):
        self.aggression = initial_aggression
        self.growth_rate = growth_rate
        self.base_mutation_rate = base_mutation_rate
        self.carrying_capacity = carrying_capacity
        self.learning_enabled = learning_enabled
        self.evolution_history = []
        self.mineral_consumption = {"common": 0, "rare": 0, "precious": 0, "anomaly": 0}
        # Initialize random number generator with a fixed seed for reproducibility
        self.rng = Generator(PCG64(42))

    # Add core implementation methods from your original code or create new ones
    def process_mineral_feeding(self, minerals, population, aggression):
        """
        Process minerals through the evolution algorithm and return new population and aggression values,
        along with any mutations.
        """
        # Track mineral consumption
        for mineral_type, amount in minerals.items():
            self.mineral_consumption[mineral_type] += amount

        # Calculate total mineral value
        total_value = (
            minerals.get("common", 0)
            + minerals.get("rare", 0) * 3
            + minerals.get("precious", 0) * 5
            + minerals.get("anomaly", 0) * 10
        )

        # Calculate population change based on mineral consumption
        population_boost = total_value * 0.1
        new_population = population + population_boost

        # Adjust aggression based on mineral types
        # Rare minerals reduce aggression, anomalies increase it
        aggression_change = 0
        if minerals.get("rare", 0) > 0:
            aggression_change -= 0.01 * minerals["rare"]
        if minerals.get("anomaly", 0) > 0:
            aggression_change += 0.02 * minerals["anomaly"]

        new_aggression = max(0.1, min(0.9, aggression + aggression_change))

        # Generate mutations based on mineral consumption
        mutations = self._generate_mutations(minerals)

        return new_population, new_aggression, mutations

    def _generate_mutations(self, minerals):
        """Generate mutations based on minerals consumed."""
        mutations = []

        # Mutation chance increases with rare and anomalous minerals
        mutation_chance = (
            self.base_mutation_rate
            + minerals.get("rare", 0) * 0.01
            + minerals.get("precious", 0) * 0.02
            + minerals.get("anomaly", 0) * 0.05
        )

        # Potential attributes that could mutate
        attributes = [
            "metabolism_rate",
            "expansion_drive",
            "mutation_rate",
            "intelligence",
            "aggression_base",
        ]

        # Generate random mutations
        for attr in attributes:
            if self.rng.random() < mutation_chance:
                # Determine mutation magnitude
                magnitude = self.rng.normal(1.0, 0.1)  # Mean 1.0, std dev 0.1

                # Determine mutation type based on magnitude
                mutation_type = "standard"
                if magnitude > 1.1:
                    mutation_type = "beneficial"
                elif magnitude > 1.05:
                    mutation_type = "significant"
                elif magnitude < 0.9:
                    mutation_type = "negative"

                mutations.append(
                    {"attribute": attr, "magnitude": magnitude, "type": mutation_type}
                )

        return mutations

    def generate_cellular_automaton_rules(self, hunger, genome):
        """Generate cellular automaton rules for symbiote growth."""
        # Default rule sets
        birth_set = {3}
        survival_set = {2, 3}

        # Adjust based on hunger and genome
        if hunger > 0.7:
            # Hungry symbiotes are more aggressive in growth
            birth_set.add(2)

        if genome.get("expansion_drive", 1.0) > 1.2:
            # Expansive races grow more easily
            birth_set.add(2)

        if genome.get("intelligence", 0.5) > 0.8:
            # Intelligent races are more strategic about survival
            survival_set.add(4)

        return birth_set, survival_set

    def update_cellular_automaton(self, grid, birth_set, survival_set):
        """Update grid using cellular automaton rules.

        Args:
            grid: Binary grid representing symbiote presence (1) or absence (0)
            birth_set: Set of neighbor counts that create new cells
            survival_set: Set of neighbor counts that allow cells to survive

        Returns:
            Updated grid after applying cellular automaton rules
        """
        if not SCIPY_AVAILABLE:
            # Fallback implementation when scipy is not available
            return self._update_cellular_automaton_manual(grid, birth_set, survival_set)

        # Use scipy for faster implementation when available
        # Count neighbors
        neighbors_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(
            grid.astype(np.int8), neighbors_kernel, mode="constant", cval=0
        )

        # Create new grid
        new_grid = np.zeros_like(grid)

        # Apply survival rules
        for n in survival_set:
            new_grid |= (neighbor_count == n) & grid

        # Apply birth rules
        for n in birth_set:
            new_grid |= (neighbor_count == n) & (~grid)

        return new_grid

    def _update_cellular_automaton_manual(self, grid, birth_set, survival_set):
        """Manual implementation of cellular automaton update when scipy is not available.

        This is a slower fallback implementation that doesn't require scipy.

        Args:
            grid: Binary grid representing symbiote presence (1) or absence (0)
            birth_set: Set of neighbor counts that create new cells
            survival_set: Set of neighbor counts that allow cells to survive

        Returns:
            Updated grid after applying cellular automaton rules
        """
        height, width = grid.shape
        new_grid = np.zeros_like(grid)
        
        # Process the grid using helper methods to reduce complexity
        self._process_grid_cells(grid, new_grid, birth_set, survival_set, height, width)
        return new_grid
        
    def _process_grid_cells(self, grid, new_grid, birth_set, survival_set, height, width):
        """Process each cell in the grid to apply cellular automaton rules.
        
        Args:
            grid: Original grid
            new_grid: Grid to update
            birth_set: Set of neighbor counts that create new cells
            survival_set: Set of neighbor counts that allow cells to survive
            height: Grid height
            width: Grid width
        """
        for y, x in itertools.product(range(height), range(width)):
            neighbors = self._count_neighbors(grid, x, y, width, height)
            self._apply_cell_rules(grid, new_grid, x, y, neighbors, birth_set, survival_set)
    
    def _count_neighbors(self, grid, x, y, width, height):
        """Count the number of live neighbors for a cell.
        
        Args:
            grid: Grid to analyze
            x: Cell x coordinate
            y: Cell y coordinate
            width: Grid width
            height: Grid height
            
        Returns:
            Number of live neighbors
        """
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the cell itself

                # Get neighbor coordinates with wrapping
                nx = (x + dx) % width
                ny = (y + dy) % height

                if grid[ny, nx] > 0:
                    neighbors += 1
        return neighbors
    
    def _apply_cell_rules(self, grid, new_grid, x, y, neighbors, birth_set, survival_set):
        """Apply cellular automaton rules to a single cell.
        
        Args:
            grid: Original grid
            new_grid: Grid to update
            x: Cell x coordinate
            y: Cell y coordinate
            neighbors: Number of live neighbors
            birth_set: Set of neighbor counts that create new cells
            survival_set: Set of neighbor counts that allow cells to survive
        """
        cell_alive = grid[y, x] > 0
        
        # Apply cellular automaton rules
        if (cell_alive and neighbors in survival_set) or (not cell_alive and neighbors in birth_set):
            new_grid[y, x] = 1

    def apply_environmental_effects(self, grid, mineral_map, hostility):
        """Apply environmental effects to the grid based on mineral distribution."""
        return grid & (
            self.rng.random(grid.shape) < (1 - hostility + mineral_map * 0.5)
        )

    def simulate_colony_interaction(self, grid, genome, aggression):
        """Simulate interaction between different colonies of the same race.

        Args:
            grid: Binary grid representing symbiote presence
            genome: Dictionary of genome attributes for the race
            aggression: Aggression level of the race

        Returns:
            Updated grid after colony interactions
        """
        # Identify colonies
        labeled_grid, num_colonies = self.identify_colonies(grid)

        if num_colonies <= 1:
            return grid

        # For multiple colonies, simulate interaction based on aggression
        competition_factor = min(1.0, aggression * 2)

        # Colonies compete or cooperate based on aggression
        if competition_factor > 0.7:
            return self._handle_colony_competition(grid, labeled_grid, num_colonies)
        else:
            return self._handle_colony_cooperation(grid, labeled_grid, num_colonies)
            
    def _handle_colony_competition(self, grid, labeled_grid, num_colonies):
        """Handle competition between colonies.
        
        Args:
            grid: Binary grid representing symbiote presence
            labeled_grid: Grid with labeled colonies
            num_colonies: Number of colonies
            
        Returns:
            Updated grid after competition
        """
        # Competition: smaller colonies may suffer
        colony_sizes = self._get_colony_sizes(grid, labeled_grid, num_colonies)
        max_size = np.max(colony_sizes)

        # Smaller colonies have higher death rate
        for i in range(1, num_colonies + 1):
            size_ratio = colony_sizes[i - 1] / max_size
            if size_ratio < 0.3:
                # Small colonies may die off
                death_mask = (labeled_grid == i) & (
                    self.rng.random(grid.shape) < 0.3
                )
                grid[death_mask] = 0
                
        return grid
        
    def _get_colony_sizes(self, grid, labeled_grid, num_colonies):
        """Get sizes of all colonies.
        
        Args:
            grid: Binary grid representing symbiote presence
            labeled_grid: Grid with labeled colonies
            num_colonies: Number of colonies
            
        Returns:
            Array of colony sizes
        """
        if SCIPY_AVAILABLE:
            return ndimage.sum(grid, labeled_grid, range(1, num_colonies + 1))
        else:
            return self._manual_sum_by_label(grid, labeled_grid, range(1, num_colonies + 1))
        
    def _handle_colony_cooperation(self, grid, labeled_grid, num_colonies):
        """Handle cooperation between colonies.
        
        Args:
            grid: Binary grid representing symbiote presence
            labeled_grid: Grid with labeled colonies
            num_colonies: Number of colonies
            
        Returns:
            Updated grid after cooperation
        """
        # Calculate centers of mass for each colony
        centers = self._get_colony_centers(grid, labeled_grid, num_colonies)

        # Connect nearby colonies
        for i in range(num_colonies):
            for j in range(i + 1, num_colonies):
                self._connect_colonies_if_close(grid, centers[i], centers[j])
                
        return grid
        
    def _get_colony_centers(self, grid, labeled_grid, num_colonies):
        """Get center coordinates for all colonies.
        
        Args:
            grid: Binary grid representing symbiote presence
            labeled_grid: Grid with labeled colonies
            num_colonies: Number of colonies
            
        Returns:
            List of colony center coordinates
        """
        if SCIPY_AVAILABLE:
            return cast(
                List[Tuple[float, float]],
                ndimage.center_of_mass(
                    grid, labeled_grid, range(1, num_colonies + 1)
                ),
            )
        else:
            return self._manual_center_of_mass(
                grid, labeled_grid, range(1, num_colonies + 1)
            )
        
    def _connect_colonies_if_close(self, grid, center1, center2):
        """Connect two colonies if they are close enough.
        
        Args:
            grid: Binary grid representing symbiote presence
            center1: First colony center coordinates
            center2: Second colony center coordinates
        """
        # Calculate distance between colony centers
        distance = np.sqrt(
            ((center1[0] - center2[0]) ** 2) + ((center1[1] - center2[1]) ** 2)
        )

        # If colonies are close enough, create a bridge between them
        if distance < 20:  # Threshold for cooperation
            self._create_bridge_between_colonies(grid, center1, center2, distance)
            
    def _create_bridge_between_colonies(self, grid, center1, center2, distance):
        """Create a bridge between two colony centers.
        
        Args:
            grid: Binary grid representing symbiote presence
            center1: First colony center coordinates
            center2: Second colony center coordinates
            distance: Distance between colony centers
        """
        # Create a line between centers
        steps = int(distance * 1.5)  # More points for smoother line
        if steps <= 0:
            return
            
        x_points = np.linspace(center1[0], center2[0], steps)
        y_points = np.linspace(center1[1], center2[1], steps)

        # Create bridge with some randomness
        for k in range(steps):
            x = int(round(x_points[k]))
            y = int(round(y_points[k]))

            # Ensure coordinates are within grid bounds
            if (
                0 <= x < grid.shape[0]
                and 0 <= y < grid.shape[1]
                and self.rng.random() < 0.7
            ):
                grid[x, y] = 1

    def identify_colonies(self, grid) -> Tuple[np.ndarray, int]:
        """Identify distinct colonies in the grid.

        Args:
            grid: Binary grid representing symbiote presence

        Returns:
            tuple: (labeled_grid, num_colonies) where labeled_grid is a numpy array
                  with the same shape as grid, and num_colonies is the number of colonies found.
        """
        if not SCIPY_AVAILABLE:
            # Fallback implementation when scipy is not available
            return self._manual_label_grid(grid)
        # Use cast to tell type checker that ndimage.label returns a tuple of (ndarray, int)
        label_result = cast(Tuple[np.ndarray, int], ndimage.label(grid))
        return label_result[0], label_result[1]

    def _manual_label_grid(self, grid) -> Tuple[np.ndarray, int]:
        """Manual implementation of grid labeling when scipy is not available.

        This is a slower fallback implementation that doesn't require scipy.
        It uses a flood fill algorithm to identify connected components.

        Args:
            grid: Binary grid representing symbiote presence

        Returns:
            tuple: (labeled_grid, num_colonies) where labeled_grid is a numpy array
                  with the same shape as grid, and num_colonies is the number of colonies found.
        """
        height, width = grid.shape
        labeled_grid = np.zeros_like(grid, dtype=np.int32)
        visited = np.zeros_like(grid, dtype=bool)
        current_label = 0

        # Define directions for 8-connected neighbors
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        # For each cell in the grid
        for y, x in itertools.product(range(height), range(width)):
            # Skip if cell is empty or already visited
            if grid[y, x] == 0 or visited[y, x]:
                continue

            # Found a new colony
            current_label += 1

            # Use a queue for flood fill
            queue = [(y, x)]
            visited[y, x] = True
            labeled_grid[y, x] = current_label

            # Process queue
            while queue:
                cy, cx = queue.pop(0)

                # Check all neighbors
                for dy, dx in directions:
                    ny, nx = (cy + dy) % height, (cx + dx) % width  # Wrap around

                    # If neighbor is valid, not visited, and has a cell
                    if not visited[ny, nx] and grid[ny, nx] > 0:
                        queue.append((ny, nx))
                        visited[ny, nx] = True
                        labeled_grid[ny, nx] = current_label

        return labeled_grid, current_label

    def get_colony_stats(self, grid, labeled_grid, num_colonies):
        """Get statistics for each colony.

        Args:
            grid: Binary grid representing symbiote presence
            labeled_grid: Grid with labeled colonies
            num_colonies: Number of colonies

        Returns:
            List of dictionaries containing stats for each colony
        """
        colony_stats = []

        if num_colonies == 0:
            return colony_stats

        # Calculate stats for each colony
        if SCIPY_AVAILABLE:
            sizes = ndimage.sum(grid, labeled_grid, range(1, num_colonies + 1))
            centroids = ndimage.center_of_mass(
                grid, labeled_grid, range(1, num_colonies + 1)
            )
        else:
            sizes = self._manual_sum_by_label(
                grid, labeled_grid, range(1, num_colonies + 1)
            )
            centroids = self._manual_center_of_mass(
                grid, labeled_grid, range(1, num_colonies + 1)
            )

        colony_stats.extend(
            {"id": i + 1, "size": sizes[i], "centroid": centroids[i]}
            for i in range(num_colonies)
        )
        return colony_stats

    def _manual_sum_by_label(self, grid, labeled_grid, labels):
        """Manual implementation of sum by label when scipy is not available.

        Args:
            grid: Binary grid representing symbiote presence
            labeled_grid: Grid with labeled colonies
            labels: List of labels to calculate sums for

        Returns:
            List of sums for each label
        """
        sums = []
        for label in labels:
            # Sum all cells with this label
            mask = labeled_grid == label
            label_sum = np.sum(grid[mask])
            sums.append(label_sum)
        return sums

    def _manual_center_of_mass(self, grid, labeled_grid, labels):
        """Manual implementation of center of mass calculation when scipy is not available.

        Args:
            grid: Binary grid representing symbiote presence
            labeled_grid: Grid with labeled colonies
            labels: List of labels to calculate centers for

        Returns:
            List of (y, x) center coordinates for each label
        """
        centers = []
        height, width = grid.shape

        for label in labels:
            # Get all points with this label
            y_coords, x_coords = np.nonzero(labeled_grid == label)

            if len(y_coords) == 0:
                # If no cells with this label, use center of grid
                centers.append((height // 2, width // 2))
                continue

            # Calculate center of mass
            y_center = np.mean(y_coords)
            x_center = np.mean(x_coords)
            centers.append((y_center, x_center))

        return centers

    def calculate_expansion_index(self, current_grid, previous_grid):
        """Calculate how much a race is expanding."""
        if previous_grid is None:
            return 0

        current_count = np.sum(current_grid)
        previous_count = np.sum(previous_grid)

        if previous_count == 0:
            return 1.0 if current_count > 0 else 0.0

        return (current_count - previous_count) / previous_count
