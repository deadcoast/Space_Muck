"""
SymbioteEvolutionAlgorithm: Advanced algorithm for symbiote race evolution.
"""

import numpy as np
import scipy.ndimage as ndimage


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

    # Add core implementation methods from your original code or create new ones
    def process_mineral_feeding(self, race_id, minerals, population, aggression):
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
            if np.random.random() < mutation_chance:
                # Determine mutation magnitude
                magnitude = np.random.normal(1.0, 0.1)  # Mean 1.0, std dev 0.1

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

    def generate_cellular_automaton_rules(self, race_id, hunger, genome):
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
        """Update grid using cellular automaton rules."""
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

    def apply_environmental_effects(self, grid, mineral_map, hostility):
        """Apply environmental effects to the grid based on mineral distribution."""
        return grid & (
            np.random.random(grid.shape) < (1 - hostility + mineral_map * 0.5)
        )

    def simulate_colony_interaction(self, grid, genome, aggression):
        """Simulate interaction between different colonies of the same race."""
        # Identify colonies
        labeled_grid, num_colonies = ndimage.label(grid)

        if num_colonies <= 1:
            return grid

        # For multiple colonies, simulate interaction based on aggression
        competition_factor = min(1.0, aggression * 2)

        # Colonies compete or cooperate based on aggression
        if competition_factor > 0.7:
            # Competition: smaller colonies may suffer
            colony_sizes = ndimage.sum(grid, labeled_grid, range(1, num_colonies + 1))
            max_size = np.max(colony_sizes)

            # Smaller colonies have higher death rate
            for i in range(1, num_colonies + 1):
                size_ratio = colony_sizes[i - 1] / max_size
                if size_ratio < 0.3:
                    # Small colonies may die off
                    death_mask = (labeled_grid == i) & (
                        np.random.random(grid.shape) < 0.3
                    )
                    grid[death_mask] = 0
        else:
            # Cooperation: colonies may bridge together
            centers = ndimage.center_of_mass(
                grid, labeled_grid, range(1, num_colonies + 1)
            )

            # Try to connect nearby colonies
            for i in range(num_colonies):
                for j in range(i + 1, num_colonies):
                    center1 = centers[i]
                    center2 = centers[j]
                    distance = np.sqrt(
                        ((center1[0] - center2[0]) ** 2)
                        + ((center1[1] - center2[1]) ** 2)
                    )

        return grid

    def identify_colonies(self, grid):
        """Identify distinct colonies in the grid."""
        return ndimage.label(grid)

    def get_colony_stats(self, grid, labeled_grid, num_colonies):
        """Get statistics for each colony."""
        colony_stats = []

        if num_colonies == 0:
            return colony_stats

        # Calculate stats for each colony
        sizes = ndimage.sum(grid, labeled_grid, range(1, num_colonies + 1))
        centroids = ndimage.center_of_mass(
            grid, labeled_grid, range(1, num_colonies + 1)
        )

        colony_stats.extend(
            {"id": i + 1, "size": sizes[i], "centroid": centroids[i]}
            for i in range(num_colonies)
        )
        return colony_stats

    def calculate_expansion_index(self, current_grid, previous_grid):
        """Calculate how much a race is expanding."""
        if previous_grid is None:
            return 0

        current_count = np.sum(current_grid)
        previous_count = np.sum(previous_grid)

        if previous_count == 0:
            return 1.0 if current_count > 0 else 0.0

        return (current_count - previous_count) / previous_count
