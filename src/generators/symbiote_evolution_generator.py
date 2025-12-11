"""
src/generators/symbiote_evolution_generator.py

SymbioteEvolutionGenerator class: Specialized generator for symbiote evolution.

This module contains the SymbioteEvolutionGenerator class which inherits from BaseGenerator
and provides specialized functionality for generating symbiote evolution patterns, colonies,
and mutations based on environmental factors.
"""

# Standard library imports
import importlib.util
import logging
import math
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Third-party library imports
import numpy as np

# Local application imports
from algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
from generators.base_generator import BaseGenerator
from utils.cellular_automaton_utils import (
    apply_cellular_automaton,
    apply_cellular_automaton_optimized,
    apply_environmental_effects,
    generate_cellular_automaton_rules,
)
from utils.dependency_injection import inject
from utils.logging_setup import (
    log_exception,
    log_performance_end,
    log_performance_start,
)
from utils.noise_generator import NoiseGenerator
from utils.pattern_generator import (
    apply_weighted_patterns,
    generate_gradient_pattern,
    generate_void_pattern,
)
from utils.value_generator import (
    add_value_clusters,
)

# Check for optional pattern analysis dependency
PATTERN_ANALYSIS_AVAILABLE = (
    importlib.util.find_spec("algorithms.pattern_analysis") is not None
)

# For type checking only - these imports are not executed at runtime
if TYPE_CHECKING:
    from algorithms.pattern_analysis import (
        AdvancedPatternAnalyzer,
        ColonyMerger,
        VoronoiTerritoryPartitioner,
    )

# Import pattern analysis dependencies at runtime if available
if PATTERN_ANALYSIS_AVAILABLE:
    try:
        from algorithms.pattern_analysis import (
            AdvancedPatternAnalyzer,
            ColonyMerger,
            VoronoiTerritoryPartitioner,
        )
    except ImportError:
        PATTERN_ANALYSIS_AVAILABLE = False
        logging.warning(
            "Pattern analysis features not available. Using basic implementation."
        )

# Define at module level to prevent undefined variable errors if not imported
if not PATTERN_ANALYSIS_AVAILABLE:
    AdvancedPatternAnalyzer = None
    VoronoiTerritoryPartitioner = None
    ColonyMerger = None


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

        # Initialize pattern analysis components if available
        self.pattern_analyzer = None
        self.territory_partitioner = None
        self.colony_merger = None
        self.history_enabled = False
        self.territory_analysis_enabled = False
        self.colony_merging_enabled = False

        # Parameters for pattern analysis
        self.set_parameter(
            "max_pattern_history", 20
        )  # Default max history for pattern detection
        self.set_parameter(
            "max_oscillator_period", 8
        )  # Max oscillator period to detect
        self.set_parameter(
            "territory_partitioning", True
        )  # Whether to use territory partitioning
        self.set_parameter("colony_merging", True)  # Whether to merge colonies
        self.set_parameter(
            "assimilation_ratio", 0.7
        )  # Default assimilation ratio (70% winner, 30% loser)

        # Initialize pattern analysis components if available
        self._initialize_pattern_analysis()

        logging.info(
            f"SymbioteEvolutionGenerator initialized: ID: {self.entity_id}, Seed: {self.seed}"
        )

    def _initialize_pattern_analysis(self):
        """Initialize pattern analysis components if available.

        This method sets up the pattern analyzer, territory partitioner,
        and colony merger when the pattern_analysis module is available.
        """
        # If pattern analysis is not available or disabled, skip initialization
        if not self.use_advanced_patterns:
            logging.info("Advanced pattern analysis disabled or unavailable.")
            return

        try:
            self._pattern_analyzer()
        except Exception as e:
            logging.warning(
                f"Failed to initialize pattern analysis components: {str(e)}"
            )
            self.history_enabled = False
            self.territory_analysis_enabled = False
            self.colony_merging_enabled = False

    def _pattern_analyzer(self):
        # Initialize pattern analyzer
        self._initialize_pattern_analyzer()
        self.history_enabled = True

        # Set feature flags based on configuration
        self._configure_pattern_analysis_features()

        # Initialize territory partitioner if enabled
        if self.territory_analysis_enabled:
            self.territory_partitioner = VoronoiTerritoryPartitioner()

        # Create the colony merger if enabled
        if self.colony_merging_enabled:
            self.colony_merger = ColonyMerger()

        logging.info("Pattern analysis components initialized successfully.")

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
        """
        Generate the initial distribution of symbiote colonies.

        Args:
            num_colonies: Number of colonies to generate, or None to use default
            start_time: Start time for performance logging

        Returns:
            tuple: (colony_grid, metadata)
        """
        # Create empty grid
        grid = np.zeros((self.height, self.width), dtype=int)

        # Determine number of colonies and size
        if num_colonies is None:
            num_colonies = self.get_parameter("initial_colonies", 3)
        colony_size = self.get_parameter("colony_size", 5)

        # Get colony centers
        colony_centers = self._determine_colony_centers(num_colonies, colony_size)

        # Place colonies on the grid
        grid = self._place_colonies(grid, colony_centers, colony_size)

        # Apply cellular automaton to make colonies more natural
        grid = self._apply_colony_automaton(grid)

        # Generate metadata
        metadata = self._generate_colony_metadata(
            colony_centers, colony_size, grid, num_colonies
        )

        log_performance_end("generate_initial_colonies", start_time)
        return grid, metadata

    def _determine_colony_centers(self, num_colonies, colony_size):
        """
        Determine the center coordinates for each colony.

        Args:
            num_colonies: Number of colonies to generate
            colony_size: Size of each colony

        Returns:
            list: List of (x, y) coordinates for colony centers
        """
        # Generate noise layer for colony placement
        noise_grid = self.generate_noise_layer("medium", scale=0.1)

        # Normalize noise grid
        noise_grid = (noise_grid - noise_grid.min()) / (
            noise_grid.max() - noise_grid.min() + 1e-10
        )

        # Find potential colony locations (high noise values)
        potential_locations = np.argwhere(noise_grid > 0.7)

        # If not enough high-value locations, just pick random locations
        if len(potential_locations) < num_colonies:
            return self._generate_random_centers(num_colonies, colony_size)

        # Pick random locations from high-value areas
        indices = np.random.Generator.choice(
            len(potential_locations), num_colonies, replace=False
        )
        return [(x, y) for y, x in potential_locations[indices]]

    def _generate_random_centers(self, num_colonies, colony_size):
        """
        Generate random colony centers when noise-based placement isn't possible.

        Args:
            num_colonies: Number of colonies to generate
            colony_size: Size of each colony

        Returns:
            list: List of (x, y) coordinates for colony centers
        """
        return [
            (
                random.randint(colony_size, self.width - colony_size),
                random.randint(colony_size, self.height - colony_size),
            )
            for _ in range(num_colonies)
        ]

    def _place_colonies(self, grid, colony_centers, colony_size):
        """
        Place colonies on the grid at specified centers.

        Args:
            grid: Empty grid to place colonies on
            colony_centers: List of (x, y) coordinates for colony centers
            colony_size: Size of each colony

        Returns:
            np.ndarray: Grid with colonies placed
        """
        result_grid = grid.copy()

        for center_x, center_y in colony_centers:
            # Create a small circular colony at this center
            self._place_single_colony(result_grid, center_x, center_y, colony_size)

        return result_grid

    def _place_single_colony(self, grid, center_x, center_y, colony_size):
        """
        Place a single colony on the grid centered at the given coordinates.

        Args:
            grid: Grid to modify in-place
            center_x: X-coordinate of colony center
            center_y: Y-coordinate of colony center
            colony_size: Size of the colony
        """
        # Calculate bounds with boundary checking
        y_start = max(0, center_y - colony_size)
        y_end = min(self.height, center_y + colony_size + 1)
        x_start = max(0, center_x - colony_size)
        x_end = min(self.width, center_x + colony_size + 1)

        # Iterate over the bounded area
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                self._try_place_colony_cell(grid, x, y, center_x, center_y, colony_size)

    @staticmethod
    def _try_place_colony_cell(grid, x, y, center_x, center_y, colony_size):
        """
        Try to place a colony cell at the given coordinates based on distance and probability.

        Args:
            grid: Grid to modify in-place
            x: X-coordinate to check
            y: Y-coordinate to check
            center_x: X-coordinate of colony center
            center_y: Y-coordinate of colony center
            colony_size: Size of the colony
        """
        # Calculate distance from center
        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Determine probability based on distance from center
        probability = 0.8 - (distance / colony_size * 0.3)

        # Place cell if within colony radius and passes probability check
        if distance <= colony_size and random.random() < probability:
            grid[y, x] = 1

    def _apply_colony_automaton(self, grid):
        """
        Apply cellular automaton to make colonies more natural.

        Args:
            grid: Grid with initial colonies

        Returns:
            np.ndarray: Grid after cellular automaton rules applied
        """
        birth_set, survival_set = generate_cellular_automaton_rules(
            hunger=0.5, genome=self.base_genome, race_id="initial"
        )
        return apply_cellular_automaton(
            grid,
            birth_set,
            survival_set,
            iterations=2,
            width=self.width,
            height=self.height,
        )

    def _generate_colony_metadata(
        self, colony_centers, colony_size, grid, num_colonies
    ):
        """
        Generate metadata for the colony distribution.

        Args:
            colony_centers: List of colony center coordinates
            colony_size: Size of each colony
            grid: Grid with colonies
            num_colonies: Number of colonies

        Returns:
            dict: Metadata dictionary
        """
        metadata = {
            "seed": self.seed,
            "num_colonies": num_colonies,
            "colony_size": colony_size,
            "colony_centers": colony_centers,
            "colony_population": np.sum(grid),
        }

        # Add pattern analysis information if available
        if self.history_enabled and self.pattern_analyzer is not None:
            # Add initial state to pattern analyzer
            self.pattern_analyzer.add_state(grid)
            metadata["pattern_analysis_enabled"] = True

        return metadata

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
            return np.random.Generator.random((self.height, self.width)) * 0.3

    def generate_mineral_concentration_map(self, start_time):
        # Generate base mineral distribution using noise
        mineral_noise = self.generate_noise_layer("medium", scale=0.05)

        # Normalize noise to 0-1 range
        mineral_grid = (mineral_noise - mineral_noise.min()) / (
            mineral_noise.max() - mineral_noise.min() + 1e-10
        )

        # Create clusters of higher mineral concentration using the utility function
        mineral_grid = add_value_clusters(
            mineral_grid,
            num_clusters=random.randint(5, 15),
            cluster_value_multiplier=2.0,
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
        """
        Handle the evolution of symbiote colonies over multiple iterations.

        Args:
            iterations: Number of evolution iterations to simulate
            colony_grid: Initial colony distribution grid
            mineral_grid: Mineral distribution grid
            start_time: Start time for performance logging

        Returns:
            tuple: (final_grid, evolution_history)
        """
        if iterations is None:
            iterations = self.get_parameter("evolution_iterations", 10)

        current_grid = colony_grid.copy()
        evolution_history = []

        # Initialize genome and environmental factors
        genome = self.base_genome.copy()
        aggression = genome["aggression_base"]
        hostility = self.get_parameter("environmental_hostility", 0.3)

        # Initialize pattern analysis if enabled
        pattern_tracking = False
        if self.history_enabled and self.pattern_analyzer is not None:
            pattern_tracking = True
            # Reset analyzer to start fresh with this evolution sequence
            self._initialize_pattern_analyzer()
            # Add initial state
            self.pattern_analyzer.add_state(current_grid)

        # Simulate evolution over multiple iterations
        for i in range(iterations):
            # Process one evolution iteration
            current_grid, genome, aggression, minerals = (
                self._process_evolution_iteration(
                    current_grid, mineral_grid, genome, aggression, hostility, i
                )
            )

            # Update pattern analyzer with new state if enabled
            if pattern_tracking:
                self.pattern_analyzer.add_state(
                    current_grid.copy()
                )  # Use copy to prevent reference issues

            # Create evolution record for this iteration
            iteration_record = self._create_evolution_record(
                i, current_grid, aggression, genome, minerals
            )

            # Add pattern analysis data if available
            if (
                pattern_tracking and i > 0 and i % 5 == 0
            ):  # Check every 5 iterations to reduce overhead
                self._add_pattern_analysis_to_record(current_grid, iteration_record)

            # Add record to history
            evolution_history.append(iteration_record)

            # Adjust hostility based on iteration (environment may become more challenging)
            hostility = min(0.8, hostility + 0.02)

        log_performance_end("simulate_evolution", start_time)
        return current_grid, evolution_history

    def _analyze_pattern_structures(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze oscillators and stable structures in the current pattern.

        Args:
            summary: The current summary dictionary to update

        Returns:
            Updated summary dictionary with pattern analysis results
        """
        # Only analyze patterns if pattern analyzer is available
        if self.pattern_analyzer is not None:
            try:
                # Get oscillator information
                oscillator_period = self.pattern_analyzer.detect_oscillator()
                if oscillator_period > 0:
                    summary["oscillator_count"] = 1
                    summary["oscillator_period"] = oscillator_period

                # Get stable structure information
                stable_structures = self.pattern_analyzer.detect_stable_structures()
                if stable_structures is not None:
                    summary["stable_structures"] = len(stable_structures)
            except Exception as e:
                logging.warning(f"Error analyzing pattern structures: {str(e)}")
                summary["error"] = str(e)

        return summary

    def get_pattern_analysis_summary(self):
        """Get a summary of detected patterns and territories.

        This utility method is useful for testing and debugging pattern analysis functionality.

        Returns:
            dict: Summary of detected patterns and territories, or None if pattern analysis is disabled
        """
        if not self.use_advanced_patterns or self.pattern_analyzer is None:
            return None

        summary = {"enabled": True, "oscillator_count": 0, "stable_structures": 0}

        try:
            # Get pattern information
            summary = self._analyze_pattern_structures(summary)

            # Get territory information if available
            if self.territory_analysis_enabled and hasattr(
                self, "territory_partitioner"
            ):
                summary["territory_partitioning"] = True

            # Get colony merger information if available
            if self.colony_merging_enabled and self.colony_merger is not None:
                summary["colony_merging"] = True
        except Exception as e:
            logging.warning(f"Error getting pattern analysis summary: {str(e)}")
            summary["error"] = str(e)

        return summary

    def _process_evolution_iteration(
        self, current_grid, mineral_grid, genome, aggression, hostility, iteration
    ):
        """
        Process a single evolution iteration.

        Args:
            current_grid: Current colony grid
            mineral_grid: Mineral distribution grid
            genome: Current genome
            aggression: Current aggression level
            hostility: Current environmental hostility
            iteration: Current iteration number

        Returns:
            tuple: (updated_grid, updated_genome, updated_aggression, minerals_consumed)
        """
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
        updated_genome = self._apply_mutations(genome, mutations)

        # Update grid based on cellular automaton rules
        updated_grid = self._apply_automaton_rules(
            current_grid,
            updated_genome,
            new_population,
            mineral_grid,
            hostility,
            new_aggression,
        )

        # Limit population if needed
        if np.sum(updated_grid) > self.evolution_algorithm.carrying_capacity:
            updated_grid = self._limit_population(updated_grid)

        # Apply colony merging if enabled and conditions are met
        if (
            self.colony_merging_enabled
            and PATTERN_ANALYSIS_AVAILABLE
            and self.colony_merger is not None
            and iteration > 0
            and iteration % 3 == 0
        ):
            # Create basic colony data for merging based on current state
            colony_data = self._create_colony_data_for_merging(updated_grid)
            # Apply colony merging
            updated_grid = self._merge_colonies(updated_grid, colony_data)

        return updated_grid, updated_genome, new_aggression, minerals

    @staticmethod
    def _apply_mutations(genome, mutations):
        """
        Apply mutations to the genome.

        Args:
            genome: Current genome
            mutations: List of mutations to apply

        Returns:
            dict: Updated genome
        """
        updated_genome = genome.copy()
        for mutation in mutations:
            attribute = mutation["attribute"]
            if attribute in updated_genome:
                updated_genome[attribute] *= mutation["magnitude"]
        return updated_genome

    def _apply_automaton_rules(
        self, grid, genome, population, mineral_grid, hostility, aggression
    ):
        """
        Apply cellular automaton rules and environmental effects.

        Args:
            grid: Current colony grid
            genome: Current genome
            population: Current population
            mineral_grid: Mineral distribution grid
            hostility: Environmental hostility level
            aggression: Current aggression level

        Returns:
            np.ndarray: Updated grid after rules applied
        """
        # Generate cellular automaton rules based on current genome
        birth_set, survival_set = generate_cellular_automaton_rules(
            hunger=1.0 - (population / self.evolution_algorithm.carrying_capacity),
            genome=genome,
            race_id="symbiote_race",
        )

        # Apply cellular automaton rules
        updated_grid = apply_cellular_automaton_optimized(grid, birth_set, survival_set)

        # Apply environmental effects
        updated_grid = apply_environmental_effects(
            updated_grid, mineral_grid, hostility
        )

        # Simulate colony interaction
        return self.evolution_algorithm.simulate_colony_interaction(
            updated_grid, genome, aggression
        )

    def _limit_population(self, grid):
        """
        Limit population to stay under carrying capacity.

        Args:
            grid: Current colony grid

        Returns:
            np.ndarray: Grid with population limited
        """
        result_grid = grid.copy()
        excess = np.sum(result_grid) - self.evolution_algorithm.carrying_capacity

        if excess > 0:
            active_cells = np.argwhere(result_grid == 1)
            to_remove = np.random.Generator.choice(
                len(active_cells), int(excess), replace=False
            )
            for idx in to_remove:
                y, x = active_cells[idx]
                result_grid[y, x] = 0

        return result_grid

    def _initialize_pattern_analyzer(self):
        """Initialize the pattern analyzer with configured parameters."""
        if not self.use_advanced_patterns:
            self.pattern_analyzer = None
            return

        try:
            max_period = self.get_parameter("max_oscillator_period", 8)
            max_history = self.get_parameter("max_pattern_history", 20)
            self.pattern_analyzer = AdvancedPatternAnalyzer(
                max_period=max_period, max_history=max_history
            )
        except Exception as e:
            self.pattern_analyzer = None
            self.use_advanced_patterns = False
            logging.warning(f"Failed to initialize pattern analyzer: {str(e)}")

    def _add_pattern_analysis_to_record(self, grid, record):
        """Add pattern analysis data to an evolution record.

        Args:
            grid: Current colony grid
            record: Evolution record to update
        """
        if not self.history_enabled or self.pattern_analyzer is None:
            return

        # Check for oscillators and stable structures
        oscillator_period = self.pattern_analyzer.detect_oscillator()
        stable_structures = self.pattern_analyzer.detect_stable_structures()

        # Add pattern info to record
        record["pattern_analysis"] = {
            "oscillator_period": oscillator_period,
            "stable_structures": stable_structures,
        }

        # Compute territory partition if enabled
        if self.territory_analysis_enabled and PATTERN_ANALYSIS_AVAILABLE:
            territory_map = self._compute_territory_partition(grid)
            if territory_map is not None:
                record["territory_map"] = territory_map

    def _configure_pattern_analysis_features(self):
        """Configure pattern analysis features based on parameters."""
        self.territory_analysis_enabled = self.get_parameter(
            "territory_partitioning", True
        )
        self.colony_merging_enabled = self.get_parameter("colony_merging", True)

    def _merge_colonies(self, grid, colony_data=None):
        """Merge weaker colonies into stronger ones based on territory and strength.

        Args:
            grid: Colony grid with colony IDs
            colony_data: Optional dictionary with colony metadata used to determine strength

        Returns:
            np.ndarray: Updated grid with merged colonies
        """
        if (
            not self.colony_merging_enabled
            or not self.use_advanced_patterns
            or self.colony_merger is None
        ):
            return grid

        try:
            # First compute a territory map to find borders
            territory_map = self._compute_territory_partition(grid)
            if territory_map is None:
                return grid

            # Identify colonies and their strengths
            if colony_data is None:
                colony_data = self._create_colony_data_for_merging(grid)

            return self._perform_colony_merges(grid, territory_map, colony_data)
        except Exception as e:
            logging.warning(f"Error during colony merging: {str(e)}")
            return grid

    def _perform_colony_merges(self, grid, territory_map, colony_data):
        """
        Perform colony merges based on territory map and colony data.

        Args:
            grid: The colony grid to update
            territory_map: Map of territories for potential merging
            colony_data: Data about colonies and their strengths

        Returns:
            Updated grid after colony merges
        """
        # Get assimilation ratio parameter
        assimilation_ratio = self.get_parameter("assimilation_ratio", 0.7)
        updated_grid = grid.copy()

        # Find border regions where colonies might merge
        # This is a simplified approach - in a full implementation, you'd want a more
        # sophisticated algorithm to determine which colonies should merge based on
        # territory borders, relative strength, etc.
        for source_id, target_id in self._find_potential_merges(
            territory_map, colony_data
        ):
            # Merge weaker colony (source_id) into stronger colony (target_id)
            if self.colony_merger is not None:
                updated_grid = self.colony_merger.merge_colonies(
                    target_id=target_id,
                    source_id=source_id,
                    labeled_grid=updated_grid,
                    colony_data=colony_data,
                    assimilation_ratio=assimilation_ratio,
                )

        return updated_grid

    @staticmethod
    def _create_colony_data_for_merging(grid):
        """Create colony data for the merging process based on grid state.

        Args:
            grid: Colony grid with colony IDs

        Returns:
            dict: Dictionary with colony data for merging
        """
        # Find unique colony IDs, skipping empty cells (0)
        unique_colonies = np.unique(grid)
        unique_colonies = unique_colonies[unique_colonies > 0]

        # Build and return colony data dictionary with strength metrics
        return {
            int(colony_id): {
                "strength": float(np.sum(grid == colony_id)),
                # Additional metrics could be added here in the future
                # such as age, resource control, etc.
            }
            for colony_id in unique_colonies
        }

    def _find_potential_merges(self, territory_map, colony_data):
        """Find potential colony merges based on territory boundaries and strengths.

        Args:
            territory_map: Territory map with colony IDs
            colony_data: Dictionary with colony metadata

        Returns:
            list: List of (source_id, target_id) tuples for potential merges
        """
        potential_merges = []

        # This is a simplified implementation that finds colonies that are likely to merge
        # In a full implementation, you would analyze territory borders to determine
        # which colonies are actually adjacent and could potentially merge
        unique_colonies = list(colony_data.keys())

        if len(unique_colonies) <= 1:
            return []  # Need at least 2 colonies for merging

        # Sort colonies by strength (weakest first)
        sorted_colonies = sorted(
            unique_colonies, key=lambda cid: colony_data[cid].get("strength", 0)
        )

        # For each weaker colony, find a potential stronger merge target
        # This is a basic heuristic - in reality, you'd want to consider proximity and other factors
        weaker_colonies = sorted_colonies[:-1]  # All but the strongest
        strongest_colony = sorted_colonies[-1]  # The strongest colony

        for weak_colony in weaker_colonies:
            # In a real implementation, you'd be more selective about which colonies to merge,
            # consider colony adjacency, conflict history, etc.
            merge_threshold = self.get_parameter("colony_merge_threshold", 0.5)
            weak_strength = colony_data[weak_colony].get("strength", 0)
            strong_strength = colony_data[strongest_colony].get(
                "strength", 1
            )  # Avoid division by zero

            # If the weak colony is significantly weaker, consider it for merging
            if weak_strength / strong_strength < merge_threshold:
                potential_merges.append((weak_colony, strongest_colony))

        return potential_merges

    def _compute_territory_partition(self, grid):
        """Compute territory partitioning based on colony centers.

        Args:
            grid: Colony grid with colony IDs

        Returns:
            np.ndarray or None: Territory map with colony IDs, or None if partitioning failed
        """
        if (
            not self.territory_analysis_enabled
            or not self.use_advanced_patterns
            or self.territory_partitioner is None
        ):
            return None

        try:
            # Find colony centers (centroid of each labeled region)
            centers = {}
            unique_colonies = np.unique(grid)
            unique_colonies = unique_colonies[
                unique_colonies > 0
            ]  # Skip empty cells (0)

            if (
                len(unique_colonies) <= 1
            ):  # Need at least 2 colonies for territory analysis
                return None

            # Calculate centroids for each colony
            for colony_id in unique_colonies:
                colony_mask = grid == colony_id
                if np.any(colony_mask):
                    colony_coords = np.nonzero(colony_mask)
                    center_y = np.mean(colony_coords[0])
                    center_x = np.mean(colony_coords[1])
                    centers[int(colony_id)] = (center_x, center_y)

            # Use territory partitioner with these centers
            if centers and self.territory_partitioner is not None:
                # Compute and return the territory map
                return self.territory_partitioner.compute_voronoi(
                    width=self.width, height=self.height, centers=centers
                )

        except Exception as e:
            logging.warning(f"Error computing territory partition: {str(e)}")

        return None

    @staticmethod
    def _create_evolution_record(iteration, grid, aggression, genome, minerals):
        """
        Create a record of the current evolution state.

        Args:
            iteration: Current iteration number
            grid: Current colony grid
            aggression: Current aggression level
            genome: Current genome
            minerals: Minerals consumed

        Returns:
            dict: Evolution record
        """
        return {
            "iteration": iteration,
            "population": np.sum(grid),
            "aggression": aggression,
            "genome": genome.copy(),
            "mutations": [],  # Note: mutations are applied in _apply_mutations and not stored here
            "mineral_consumption": minerals,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbioteEvolutionGenerator":
        """
        Create a SymbioteEvolutionGenerator instance from a dictionary.

        This custom implementation handles the specific requirements of the SymbioteEvolutionGenerator
        class, particularly the fact that entity_type is hardcoded as "symbiote".

        Args:
            data: Dictionary containing generator data

        Returns:
            SymbioteEvolutionGenerator: New generator instance
        """
        # Extract parameters for the evolution algorithm
        parameters = data.get("parameters", {})

        # Create a new instance with the extracted parameters
        generator = cls(
            entity_id=data.get("entity_id"),
            seed=data.get("seed"),
            width=data.get("width", 100),
            height=data.get("height", 100),
            color=data.get("color", (100, 200, 100)),
            position=data.get("position"),
            initial_aggression=parameters.get("initial_aggression", 0.2),
            growth_rate=parameters.get("growth_rate", 0.05),
            base_mutation_rate=parameters.get("base_mutation_rate", 0.01),
            carrying_capacity=parameters.get("carrying_capacity", 100),
            learning_enabled=parameters.get("learning_enabled", True),
        )

        # Set any additional parameters that might be in the data
        for key, value in parameters.items():
            if key not in [
                "initial_aggression",
                "growth_rate",
                "base_mutation_rate",
                "carrying_capacity",
                "learning_enabled",
            ]:
                generator.set_parameter(key, value)

        return generator

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

    @staticmethod
    def _calculate_mineral_consumption(
        colony_grid: np.ndarray, mineral_grid: np.ndarray
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
        # Calculate but don't use total consumption - might be needed in future
        # total_consumption = np.sum(colony_grid * mineral_grid)

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
