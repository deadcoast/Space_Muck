"""
SymbioteEvolutionGenerator class: Specialized generator for symbiote evolution.

This module contains the SymbioteEvolutionGenerator class which inherits from BaseGenerator
and provides specialized functionality for generating symbiote evolution patterns, colonies,
and mutations based on environmental factors.
"""

# Standard library imports
import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Third-party library imports
import numpy as np
import scipy.ndimage as ndimage

# Local application imports
try:
    from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
    from src.config import *
    from src.entities.base_generator import BaseGenerator
    from src.utils.noise_generator import NoiseGenerator, get_noise_generator
    from src.utils.dependency_injection import inject
    from src.utils.logging_setup import (
        log_performance_start,
        log_performance_end,
        log_exception,
        LogContext,
    )
    from src.utils.pattern_generator import (
        generate_spiral_pattern,
        generate_ring_pattern,
        generate_gradient_pattern,
        generate_void_pattern,
        apply_weighted_patterns,
    )
    from src.utils.value_generator import (
        generate_value_distribution,
        add_value_clusters,
        generate_rare_resource_distribution,
    )
    from src.utils.cellular_automaton_utils import (
        apply_cellular_automaton,
        apply_cellular_automaton_optimized,
        generate_cellular_automaton_rules,
        apply_environmental_effects,
    )
except ImportError:
    # Alternative import paths for when running from different directories
    from config import *
    from entities.base_generator import BaseGenerator
    from algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
    from utils.noise_generator import NoiseGenerator, get_noise_generator
    from utils.dependency_injection import inject
    from utils.logging_setup import (
        log_performance_start,
        log_performance_end,
        log_exception,
        LogContext,
    )
    from utils.pattern_generator import (
        generate_spiral_pattern,
        generate_ring_pattern,
        generate_gradient_pattern,
        generate_void_pattern,
        apply_weighted_patterns,
    )
    from utils.value_generator import (
        generate_value_distribution,
        add_value_clusters,
        generate_rare_resource_distribution,
    )
    from utils.cellular_automaton_utils import (
        apply_cellular_automaton,
        apply_cellular_automaton_optimized,
        generate_cellular_automaton_rules,
        apply_environmental_effects,
    )


@inject
class SymbioteEvolutionGenerator(BaseGenerator):
    """
    Generator for procedural symbiote evolution with multiple colonies and mutation patterns.
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
        initial_aggression: float = 0.2,
        growth_rate: float = 0.05,
        base_mutation_rate: float = 0.01,
        carrying_capacity: int = 100,
        learning_enabled: bool = True,
        noise_generator: Optional[NoiseGenerator] = None,
    ) -> None:
        """
        Initialize the symbiote evolution generator.

        Args:
            entity_id: Unique identifier for the entity (defaults to a UUID)
            seed: Random seed for reproducibility
            width: Width of the field to generate
            height: Height of the field to generate
            color: RGB color tuple for visualization
            position: Initial position as (x, y) tuple
            initial_aggression: Starting aggression level for symbiotes
            growth_rate: Base growth rate for symbiote colonies
            base_mutation_rate: Base probability of mutations
            carrying_capacity: Maximum sustainable population
            learning_enabled: Whether symbiotes can learn from environment
            noise_generator: Injected noise generator (defaults to auto-selected implementation)
        """
        # Call the parent class constructor
        super().__init__(
            entity_id=entity_id,
            entity_type="symbiote",
            seed=seed,
            width=width,
            height=height,
            color=color,
            position=position,
            noise_generator=noise_generator,
        )

        # Initialize the symbiote evolution algorithm
        self.evolution_algorithm = SymbioteEvolutionAlgorithm(
            initial_aggression=initial_aggression,
            growth_rate=growth_rate,
            base_mutation_rate=base_mutation_rate,
            carrying_capacity=carrying_capacity,
            learning_enabled=learning_enabled,
        )

        # Generation parameters
        self.set_parameter("initial_colonies", 3)  # Number of starting colonies
        self.set_parameter("colony_size", 5)  # Initial size of each colony
        self.set_parameter(
            "environmental_hostility", 0.3
        )  # Base environmental hostility
        self.set_parameter(
            "mineral_influence", 0.6
        )  # How much minerals affect evolution
        self.set_parameter(
            "mutation_chance", base_mutation_rate
        )  # Base mutation chance
        self.set_parameter(
            "evolution_iterations", 10
        )  # Number of evolution steps to simulate

        # Genome template for new symbiote races
        self.base_genome = {
            "metabolism_rate": 1.0,
            "expansion_drive": 1.0,
            "mutation_rate": base_mutation_rate,
            "intelligence": 0.5,
            "aggression_base": initial_aggression,
        }

        logging.info(
            f"SymbioteEvolutionGenerator initialized: ID: {self.entity_id}, Seed: {self.seed}"
        )

    def generate_initial_colonies(
        self, num_colonies: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate initial symbiote colonies on a grid.

        Args:
            num_colonies: Optional override for number of colonies

        Returns:
            tuple: (colony_grid, metadata)
        """
        start_time = log_performance_start("generate_initial_colonies")

        try:
            return self.generate_colony_distribution(num_colonies, start_time)
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            fallback_grid = np.zeros((self.height, self.width), dtype=int)
            fallback_grid[
                self.height // 4 : self.height // 4 + 10,
                self.width // 4 : self.width // 4 + 10,
            ] = 1
            return fallback_grid, {"seed": self.seed, "error": str(e)}

    def generate_colony_distribution(self, num_colonies, start_time):
        # Create empty grid
        grid = np.zeros((self.height, self.width), dtype=int)

        # Determine number of colonies
        if num_colonies is None:
            num_colonies = self.get_parameter("initial_colonies", 3)

        colony_size = self.get_parameter("colony_size", 5)

        # Generate noise layer for colony placement
        noise_grid = self.generate_noise_layer("medium", scale=0.1)

        # Normalize noise grid
        noise_grid = (noise_grid - noise_grid.min()) / (
            noise_grid.max() - noise_grid.min() + 1e-10
        )

        # Find potential colony locations (high noise values)
        potential_locations = np.argwhere(noise_grid > 0.7)

        if len(potential_locations) < num_colonies:
            # If not enough high-value locations, just pick random locations
            colony_centers = [
                (
                    random.randint(colony_size, self.width - colony_size),
                    random.randint(colony_size, self.height - colony_size),
                )
                for _ in range(num_colonies)
            ]
        else:
            # Pick random locations from high-value areas
            indices = np.random.choice(
                len(potential_locations), num_colonies, replace=False
            )
            colony_centers = [(x, y) for y, x in potential_locations[indices]]

        # Place colonies on the grid
        for center_x, center_y in colony_centers:
            # Create a small circular colony
            for y in range(center_y - colony_size, center_y + colony_size + 1):
                for x in range(center_x - colony_size, center_x + colony_size + 1):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        # Use distance from center to determine if cell is part of colony
                        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if distance <= colony_size and random.random() < 0.8 - (
                            distance / colony_size * 0.3
                        ):
                            grid[y, x] = 1

        # Apply cellular automaton to make colonies more natural
        birth_set, survival_set = generate_cellular_automaton_rules(
            hunger=0.5, genome=self.base_genome, race_id="initial"
        )
        grid = apply_cellular_automaton(
            grid,
            birth_set,
            survival_set,
            iterations=2,
            width=self.width,
            height=self.height,
        )

        # Generate metadata
        metadata = {
            "seed": self.seed,
            "num_colonies": num_colonies,
            "colony_size": colony_size,
            "colony_centers": colony_centers,
            "colony_population": np.sum(grid),
        }

        log_performance_end("generate_initial_colonies", start_time)
        return grid, metadata

    def generate_mineral_distribution(self) -> np.ndarray:
        """
        Generate a mineral distribution map that will influence symbiote evolution.

        Returns:
            np.ndarray: Grid with mineral values
        """
        start_time = log_performance_start("generate_mineral_distribution")

        try:
            return self.generate_mineral_concentration_map(start_time)
        except Exception as e:
            log_exception(e)
            # Return a simple fallback grid if generation fails
            return np.random.random((self.height, self.width)) * 0.3

    def generate_mineral_concentration_map(self, start_time):
        # Generate base mineral distribution using noise
        mineral_noise = self.generate_noise_layer("medium", scale=0.05)

        # Normalize noise to 0-1 range
        mineral_grid = (mineral_noise - mineral_noise.min()) / (
            mineral_noise.max() - mineral_noise.min() + 1e-10
        )

        # Create clusters of higher mineral concentration using the utility function
        mineral_grid = add_value_clusters(
            mineral_grid, num_clusters=random.randint(5, 15), value_multiplier=2.0
        )

        # Normalize again after adding clusters
        mineral_grid = (mineral_grid - mineral_grid.min()) / (
            mineral_grid.max() - mineral_grid.min() + 1e-10
        )

        log_performance_end("generate_mineral_distribution", start_time)
        return mineral_grid

    def simulate_evolution(
        self,
        colony_grid: np.ndarray,
        mineral_grid: np.ndarray,
        iterations: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Simulate symbiote evolution over time based on mineral distribution.

        Args:
            colony_grid: Initial colony distribution
            mineral_grid: Mineral distribution
            iterations: Number of evolution steps to simulate

        Returns:
            tuple: (evolved_grid, evolution_history)
        """
        start_time = log_performance_start("simulate_evolution")

        try:
            return self._evolution_handler(
                iterations, colony_grid, mineral_grid, start_time
            )
        except Exception as e:
            log_exception(e)
            # Return original grid if simulation fails
            return colony_grid, [{"error": str(e)}]

    def _evolution_handler(self, iterations, colony_grid, mineral_grid, start_time):
        if iterations is None:
            iterations = self.get_parameter("evolution_iterations", 10)

        current_grid = colony_grid.copy()
        evolution_history = []

        # Initialize genome and aggression
        genome = self.base_genome.copy()
        aggression = genome["aggression_base"]
        hostility = self.get_parameter("environmental_hostility", 0.3)

        # Simulate evolution over multiple iterations
        for i in range(iterations):
            # Calculate mineral consumption based on current colony distribution
            minerals = self._calculate_mineral_consumption(current_grid, mineral_grid)

            # Process mineral feeding through evolution algorithm
            population = np.sum(current_grid)
            new_population, new_aggression, mutations = (
                self.evolution_algorithm.process_mineral_feeding(
                    race_id="symbiote_race",
                    minerals=minerals,
                    population=population,
                    aggression=aggression,
                )
            )

            # Update genome based on mutations
            for mutation in mutations:
                attribute = mutation["attribute"]
                if attribute in genome:
                    genome[attribute] *= mutation["magnitude"]

            # Update aggression
            aggression = new_aggression

            # Generate cellular automaton rules based on current genome using utility function
            birth_set, survival_set = generate_cellular_automaton_rules(
                hunger=1.0
                - (new_population / self.evolution_algorithm.carrying_capacity),
                genome=genome,
                race_id="symbiote_race",
            )

            # Apply cellular automaton rules using optimized utility function
            current_grid = apply_cellular_automaton_optimized(
                current_grid, birth_set, survival_set
            )

            # Apply environmental effects using utility function
            current_grid = apply_environmental_effects(
                current_grid, mineral_grid, hostility
            )

            # Simulate colony interaction
            current_grid = self.evolution_algorithm.simulate_colony_interaction(
                current_grid, genome, aggression
            )

            # Limit population based on carrying capacity
            if np.sum(current_grid) > self.evolution_algorithm.carrying_capacity:
                # Randomly remove some cells to stay under capacity
                excess = (
                    np.sum(current_grid) - self.evolution_algorithm.carrying_capacity
                )
                if excess > 0:
                    active_cells = np.argwhere(current_grid == 1)
                    to_remove = np.random.choice(
                        len(active_cells), int(excess), replace=False
                    )
                    for idx in to_remove:
                        y, x = active_cells[idx]
                        current_grid[y, x] = 0

            # Record history
            evolution_history.append(
                {
                    "iteration": i,
                    "population": np.sum(current_grid),
                    "aggression": aggression,
                    "genome": genome.copy(),
                    "mutations": mutations,
                    "mineral_consumption": minerals,
                }
            )

            # Adjust hostility based on iteration (environment may become more challenging)
            hostility = min(0.8, hostility + 0.02)

        log_performance_end("simulate_evolution", start_time)
        return current_grid, evolution_history

    def generate_mutation_map(
        self, colony_grid: np.ndarray, evolution_history: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Generate a map showing mutation hotspots in the colony.

        Args:
            colony_grid: Current colony distribution
            evolution_history: History of evolution iterations

        Returns:
            np.ndarray: Grid with mutation intensity values
        """
        start_time = log_performance_start("generate_mutation_map")

        try:
            return self._mutation_handler(evolution_history, colony_grid, start_time)
        except Exception as e:
            log_exception(e)
            # Return empty grid if generation fails
            return np.zeros((self.height, self.width), dtype=float)

    def _mutation_handler(self, evolution_history, colony_grid, start_time):
        # Create a mutation intensity grid
        mutation_grid = np.zeros((self.height, self.width), dtype=float)

        # Count total mutations from history
        total_mutations = sum(len(step["mutations"]) for step in evolution_history)

        if total_mutations == 0:
            return mutation_grid

        # Generate noise layer for mutation distribution
        mutation_noise = self.generate_noise_layer("high", scale=0.1)

        # Generate a pattern for mutation distribution using utility functions
        # Create a weighted pattern with void areas and gradient
        pattern_functions = [generate_void_pattern, generate_gradient_pattern]
        pattern_weights = [0.7, 0.3]
        pattern_args = [
            {
                "width": self.width,
                "height": self.height,
                "num_voids": 5,
                "void_size": 0.15,
            },
            {
                "width": self.width,
                "height": self.height,
                "direction": random.random() * 2 * math.pi,
            },
        ]

        pattern_grid = apply_weighted_patterns(
            self.width, self.height, pattern_functions, pattern_weights, pattern_args
        )

        # Combine noise and pattern
        mutation_grid = (mutation_noise * 0.7 + pattern_grid * 0.3) * colony_grid

        # Normalize to 0-1 range
        mutation_grid = (mutation_grid - mutation_grid.min()) / (
            mutation_grid.max() - mutation_grid.min() + 1e-10
        )

        # Scale by mutation intensity
        mutation_intensity = total_mutations / (
            len(evolution_history) * 5
        )  # Normalize to expected range
        mutation_grid *= mutation_intensity

        log_performance_end("generate_mutation_map", start_time)
        return mutation_grid

    def _calculate_mineral_consumption(
        self, colony_grid: np.ndarray, mineral_grid: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate mineral consumption based on colony distribution and mineral availability.

        Args:
            colony_grid: Current colony distribution
            mineral_grid: Mineral distribution

        Returns:
            dict: Dictionary of mineral types and amounts consumed
        """
        # Calculate total mineral consumption
        total_consumption = np.sum(colony_grid * mineral_grid)

        # Distribute consumption across different mineral types
        # Higher mineral values have higher chance of rare minerals

        # Calculate thresholds for different mineral types
        common_threshold = 0.7  # Most minerals are common
        rare_threshold = 0.9  # Some are rare
        precious_threshold = 0.97  # Few are precious
        # Anything above precious_threshold is anomaly

        # Calculate mineral type distribution
        mineral_values = mineral_grid[colony_grid == 1]
        if len(mineral_values) == 0:
            return {"common": 0, "rare": 0, "precious": 0, "anomaly": 0}

        common_amount = np.sum(mineral_values[mineral_values < common_threshold])
        rare_amount = np.sum(
            mineral_values[
                (mineral_values >= common_threshold) & (mineral_values < rare_threshold)
            ]
        )
        precious_amount = np.sum(
            mineral_values[
                (mineral_values >= rare_threshold)
                & (mineral_values < precious_threshold)
            ]
        )
        anomaly_amount = np.sum(mineral_values[mineral_values >= precious_threshold])

        return {
            "common": float(common_amount),
            "rare": float(rare_amount),
            "precious": float(precious_amount),
            "anomaly": float(anomaly_amount),
        }
