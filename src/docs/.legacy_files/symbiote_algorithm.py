

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("symbiote_evolution.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

class SymbioteEvolutionAlgorithm:
    """
    Advanced algorithm for symbiote evolution and behavior, implementing the mathematical models
    described in algorithm_design.md.
    """

import logging
import math
import random

# Standard library imports
from collections import deque

import numpy as np

# Third-party library imports
from scipy import ndimage, stats

# Optional dependencies
try:
    import scipy.ndimage as ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available, using fallback implementation.")

try:
    import scipy.ndimage as ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available, using fallback implementation.")

# Local application imports
from typing import Dict, List, Optional, Set, Tuple

    def __init__(
        self,
        initial_aggression: float = 0.2,
        growth_rate: float = 0.05,
        base_mutation_rate: float = 0.01,
        carrying_capacity: int = 100,
        learning_enabled: bool = True,
    ):
        """
        Initialize the symbiote evolution algorithm with tunable parameters.

        Args:
            initial_aggression: Starting aggression level (0 to 1)
            growth_rate: Natural population growth rate
            base_mutation_rate: Baseline mutation probability
            carrying_capacity: Base carrying capacity (can be modified by environment)
            learning_enabled: Whether to use machine learning elements for adaptive behavior
        """
        # Core parameters
        self.growth_rate = growth_rate
        self.base_mutation_rate = base_mutation_rate
        self.carrying_capacity = carrying_capacity
        self.learning_enabled = learning_enabled

        # Aggression modifiers
        self.hunger_aggression_factor = 0.1
        self.fleet_aggression_factor = 0.05
        self.mineral_stockpile_factor = 0.02

        # Mineral influence factors (how much each mineral type affects mutations)
        self.mineral_influence = {
            "common": 0.01,  # 1% mutation chance per common mineral
            "rare": 0.2,  # 20% mutation chance per rare mineral
            "precious": 0.1,  # 10% mutation chance but guaranteed growth
            "anomaly": 0.5,  # 50% chance of wild mutation
        }

        # Population growth modifiers for each mineral type
        self.mineral_growth = {
            "common": 0.2,  # 20% chance to spawn a new symbiote
            "rare": 0.7,  # 70% chance to spawn a new symbiote
            "precious": 1.0,  # 100% chance to spawn 2 symbiotes
            "anomaly": 0.8,  # 80% chance but unpredictable outcome
        }

        # Machine learning state (if enabled)
        self.ml_memory = {
            "successful_attacks": 0,
            "failed_attacks": 0,
            "minerals_consumed": {"common": 0, "rare": 0, "precious": 0, "anomaly": 0},
            "player_feeding_pattern": deque(
                maxlen=10
            ),  # Remember last 10 feeding decisions
            "attack_outcomes": deque(maxlen=20),  # Remember outcomes of last 20 attacks
        }

        # Game balance adapters (these adjust over time based on gameplay)
        self.difficulty_adapter = 1.0  # Multiplier for symbiote effectiveness

    def process_mineral_feeding(
        self,
        race_id: int,
        minerals: Dict[str, int],
        current_population: int,
        current_aggression: float,
    ) -> Tuple[int, float, List[Dict]]:
        """
        Process the effects of feeding minerals to symbiotes.

        Args:
            race_id: ID of the symbiote race being fed
            minerals: Dictionary with keys 'common', 'rare', 'precious', 'anomaly' and count values
            current_population: Current symbiote population
            current_aggression: Current aggression level (0-1)

        Returns:
            Tuple of (new_population, new_aggression, mutations_list)
            mutations_list contains dictionaries with mutation details
        """
        new_population = current_population
        new_aggression = current_aggression
        mutations = []

        # If no minerals are fed, process hunger increase
        if sum(minerals.values()) == 0:
            # Increase aggression due to hunger
            new_aggression = min(1.0, new_aggression + self.hunger_aggression_factor)
            return new_population, new_aggression, mutations

        # Track mineral consumption for ML
        if self.learning_enabled:
            for mineral_type, amount in minerals.items():
                self.ml_memory["minerals_consumed"][mineral_type] += amount

            # Record this feeding instance
            self.ml_memory["player_feeding_pattern"].append(minerals.copy())

        # Calculate total nutritional value of minerals
        # Different mineral types have different satiation values
        total_nutrition = (
            minerals.get("common", 0) * 1
            + minerals.get("rare", 0) * 3
            + minerals.get("precious", 0) * 5
            + minerals.get("anomaly", 0) * 7
        )

        # Process each mineral type's effects

        # Common minerals - basic sustenance
        for _ in range(minerals.get("common", 0)):
            # Chance to spawn a new symbiote
            if random.random() < self.mineral_growth["common"]:
                new_population += 1

            # Small chance of minor mutation
            if random.random() < self.mineral_influence["common"]:
                mutation = {
                    "type": "minor",
                    "attribute": random.choice(["metabolism", "resilience"]),
                    "magnitude": random.uniform(0.95, 1.05),  # ±5% change
                    "source": "common",
                }
                mutations.append(mutation)

        # Rare minerals - evolutionary catalyst
        for _ in range(minerals.get("rare", 0)):
            # Higher chance to spawn new symbiotes
            if random.random() < self.mineral_growth["rare"]:
                new_population += 1

            # Moderate chance of significant mutation
            if random.random() < self.mineral_influence["rare"]:
                mutation = {
                    "type": "significant",
                    "attribute": random.choice(
                        ["aggression", "intelligence", "expansion_drive"]
                    ),
                    "magnitude": random.uniform(0.9, 1.2),  # -10% to +20% change
                    "source": "rare",
                }
                mutations.append(mutation)

        # Precious minerals - guaranteed growth with quality improvement
        for _ in range(minerals.get("precious", 0)):
            # Guaranteed to spawn 2 symbiotes
            new_population += 2

            # Chance of beneficial mutation
            if random.random() < self.mineral_influence["precious"]:
                mutation = {
                    "type": "beneficial",
                    "attribute": random.choice(["strength", "territory_size"]),
                    "magnitude": random.uniform(1.05, 1.15),  # +5% to +15% change
                    "source": "precious",
                }
                mutations.append(mutation)

            # Precious minerals have a calming effect
            new_aggression = max(0.0, new_aggression - 0.1)

        # Anomaly minerals - unpredictable outcomes
        for _ in range(minerals.get("anomaly", 0)):
            # Wild effects based on random outcome
            effect = random.random()

            if effect < 0.25:  # 25% chance: population surge
                surge = random.randint(3, 10)
                new_population += surge
                mutation = {
                    "type": "population_surge",
                    "magnitude": surge,
                    "source": "anomaly",
                }
            elif effect < 0.5:  # 25% chance: temporary docility
                calm_factor = random.uniform(0.3, 0.6)
                new_aggression = max(0.0, new_aggression - calm_factor)
                mutation = {
                    "type": "pacification",
                    "magnitude": calm_factor,
                    "source": "anomaly",
                }
            elif effect < 0.75:  # 25% chance: aggression spike
                spike_factor = random.uniform(0.2, 0.4)
                new_aggression = min(1.0, new_aggression + spike_factor)
                mutation = {
                    "type": "aggression_spike",
                    "magnitude": spike_factor,
                    "source": "anomaly",
                }
            else:  # 25% chance: boss mutation
                # A special "boss" symbiote appears
                mutation = {
                    "type": "boss_spawn",
                    "strength_multiplier": random.uniform(2.0, 5.0),
                    "source": "anomaly",
                }
                new_population += 1  # Add the boss to population

            mutations.append(mutation)

        # Apply the feeding effect on aggression (satiation reduces aggression)
        satiation_effect = min(0.5, total_nutrition / max(10, current_population))
        new_aggression = max(0.0, new_aggression - satiation_effect)

        # Cap population and aggression to valid ranges
        new_population = max(0, new_population)
        new_aggression = max(0.0, min(1.0, new_aggression))

        return new_population, new_aggression, mutations

    def natural_growth_update(
        self,
        population: int,
        mineral_availability: int,
        prev_populations: List[int] = None,
    ) -> int:
        """
        Calculate natural growth using logistic growth model.

        Args:
            population: Current symbiote population
            mineral_availability: Available mineral resources (affects carrying capacity)
            prev_populations: Recent population history for trend analysis

        Returns:
            New population after natural growth/decline
        """
        if population <= 0:
            return 0

        # Dynamic carrying capacity based on available resources
        adjusted_capacity = self.carrying_capacity + (mineral_availability * 0.5)

        # Apply logistic growth equation: dN/dt = r*N*(1-N/K)
        growth = self.growth_rate * population * (1 - population / adjusted_capacity)

        # Convert to integer growth (can be negative for population decline)
        int_growth = math.floor(growth)

        # Apply probabilistic rounding for fractional growth
        fractional = growth - int_growth
        if random.random() < fractional:
            int_growth += 1

        # Apply historical trend adjustments if history is available
        if prev_populations and len(prev_populations) > 5:
            # Calculate trend using linear regression
            x = np.arange(len(prev_populations))
            slope, _, _, _, _ = stats.linregress(x, prev_populations)

            # If population is declining rapidly, increase growth chance
            # (symbiotes adapt to prevent extinction)
            if slope < -0.5 and random.random() < 0.3:
                int_growth = max(1, int_growth)

            # If population is exploding, introduce density-dependent mortality
            if slope > 2.0 and population > adjusted_capacity * 0.8:
                int_growth = min(int_growth, 0)  # Cap growth at zero

                # Additional mortality from overcrowding
                if random.random() < 0.4:
                    int_growth -= random.randint(1, max(1, population // 20))

        return max(0, population + int_growth)

    def update_aggression(
        self,
        current_aggression: float,
        population: int,
        fleet_size: int,
        mineral_stockpile: int,
        fed_this_turn: bool,
    ) -> float:
        """
        Update symbiote aggression based on multiple factors.

        Args:
            current_aggression: Current aggression level (0-1)
            population: Current symbiote population
            fleet_size: Player's fleet size
            mineral_stockpile: Player's stockpiled minerals
            fed_this_turn: Whether symbiotes were fed this turn

        Returns:
            New aggression level (0-1)
        """
        new_aggression = current_aggression

        # 1. Hunger factor - if not fed, increase aggression
        if not fed_this_turn:
            new_aggression += self.hunger_aggression_factor

        # 2. Fleet size factor - larger player fleets make symbiotes more cautious or bold
        fleet_factor = self.fleet_aggression_factor

        # If symbiotes significantly outnumber fleet, they become bolder
        if population > fleet_size * 2:
            new_aggression += fleet_factor
        # If fleet significantly outnumbers symbiotes, they become more cautious
        elif fleet_size > population * 2:
            new_aggression -= fleet_factor

        # 3. Resource availability - player stockpiling minerals increases aggression
        # Represents symbiotes being attracted to resource concentrations
        stockpile_effect = min(
            0.3, mineral_stockpile / 1000 * self.mineral_stockpile_factor
        )
        new_aggression += stockpile_effect

        # 4. Population pressure - overcrowding leads to more aggression
        # Compare to a soft carrying capacity threshold
        soft_cap = self.carrying_capacity * 0.8
        if population > soft_cap:
            overpopulation_factor = min(0.2, (population - soft_cap) / soft_cap * 0.1)
            new_aggression += overpopulation_factor

        # 5. Apply machine learning adjustments if enabled
        if self.learning_enabled and len(self.ml_memory["attack_outcomes"]) > 5:
            # Calculate success rate of recent attacks
            success_rate = sum(
                bool(outcome) for outcome in self.ml_memory["attack_outcomes"]
            ) / len(self.ml_memory["attack_outcomes"])

            # If attacks have been successful, symbiotes become bolder
            if success_rate > 0.7:
                new_aggression += 0.05
            # If attacks have been failing, they become more cautious
            elif success_rate < 0.3:
                new_aggression -= 0.05

            # Analyze feeding patterns to make strategic decisions
            if len(self.ml_memory["player_feeding_pattern"]) > 5:
                # If player frequently feeds, symbiotes might become more demanding
                recent_feedings = [
                    sum(feed.values())
                    for feed in self.ml_memory["player_feeding_pattern"]
                ]
                if np.mean(recent_feedings) > 10:  # Player feeds a lot
                    new_aggression += 0.03  # Increase demands

        # Apply difficulty adapter
        new_aggression *= self.difficulty_adapter

        # Ensure aggression stays in valid range
        return max(0.0, min(1.0, new_aggression))

    def determine_attack(
        self, aggression: float, population: int, target_value: int
    ) -> Tuple[bool, float]:
        """
        Determine if symbiotes attack and calculate attack strength.

        Args:
            aggression: Current aggression level (0-1)
            population: Current symbiote population
            target_value: Value/strength of the target (player fleet or mining operation)

        Returns:
            Tuple of (attack_occurs, attack_strength)
        """
        # Base probability is the aggression level
        attack_probability = aggression

        # Modify probability based on relative strength
        if population < target_value * 0.5:  # Symbiotes are significantly outnumbered
            attack_probability *= 0.5  # Less likely to attack when outmatched
        elif population > target_value * 2:  # Symbiotes significantly outnumber target
            attack_probability = min(
                0.9, attack_probability * 1.5
            )  # More likely to attack when stronger

        # Introduce some randomness with sigmoid function to create more interesting patterns
        # This creates occasional surprise attacks even when aggression is moderate
        sigmoid_factor = 1 / (1 + math.exp(-10 * (aggression - 0.5)))
        attack_probability = attack_probability * 0.7 + sigmoid_factor * 0.3

        # Final decision
        attack_occurs = random.random() < attack_probability

        # Calculate attack strength if attack occurs
        attack_strength = 0.0
        if attack_occurs:
            # Base strength proportional to population and aggression
            base_strength = population * aggression * 0.1

            # Add variability - attacks aren't always at full strength
            variance = random.uniform(0.7, 1.3)
            attack_strength = base_strength * variance

            # Apply machine learning adjustments if enabled
            if self.learning_enabled and (
                self.ml_memory["successful_attacks"] > self.ml_memory["failed_attacks"]
            ):
                attack_strength *= 1.1

        return attack_occurs, attack_strength

    def process_attack_outcome(
        self,
        was_successful: bool,
        symbiote_casualties: int,
        player_casualties: int,
        population: int,
        aggression: float,
    ) -> Tuple[int, float]:
        """
        Process the aftermath of an attack, updating symbiote state.

        Args:
            was_successful: Whether the attack succeeded from symbiotes' perspective
            symbiote_casualties: Number of symbiotes lost
            player_casualties: Number of player ships/structures lost
            population: Current symbiote population
            aggression: Current aggression level

        Returns:
            Tuple of (new_population, new_aggression)
        """
        # Update ML memory
        if self.learning_enabled:
            self.ml_memory["attack_outcomes"].append(was_successful)
            if was_successful:
                self.ml_memory["successful_attacks"] += 1
            else:
                self.ml_memory["failed_attacks"] += 1

        # Update population
        new_population = max(0, population - symbiote_casualties)

        # Update aggression based on outcome
        new_aggression = aggression

        if was_successful:
            # Successful attack encourages more aggression
            new_aggression = min(1.0, aggression + 0.1)

            # If very successful (high player casualties), even more aggression
            if player_casualties > 2:
                new_aggression = min(1.0, new_aggression + 0.1)

            # Chance for population bonus (recruitment or reproduction boost from victory)
            if random.random() < 0.3:
                victory_bonus = random.randint(1, max(1, player_casualties))
                new_population += victory_bonus
        else:
            # Failed attack reduces aggression temporarily
            aggression_reduction = (
                0.1 + (symbiote_casualties / max(10, population)) * 0.3
            )
            new_aggression = max(0.0, aggression - aggression_reduction)

            # If casualties were severe, aggression drops more (retreat behavior)
            if symbiote_casualties > population * 0.3:  # Lost over 30%
                new_aggression = max(0.0, new_aggression - 0.2)

        return new_population, new_aggression

    def mutate_genome(
        self, genome: Dict[str, float], mutations: List[Dict]
    ) -> Dict[str, float]:
        """
        Apply mutations to the symbiote race's genome.

        Args:
            genome: Current genome as dictionary of attributes
            mutations: List of mutation events from feeding

        Returns:
            Updated genome dictionary
        """
        new_genome = genome.copy()

        # Process each mutation
        for mutation in mutations:
            mutation_type = mutation.get("type", "")

            # Handle specific mutation types
            if mutation_type in ["minor", "significant", "beneficial"]:
                attribute = mutation.get("attribute", "")
                magnitude = mutation.get("magnitude", 1.0)

                # Apply the mutation if the attribute exists in genome
                if attribute in new_genome:
                    new_genome[attribute] *= magnitude

            # Special mutation types
            elif mutation_type == "population_surge":
                # Population surge already handled directly, but might affect reproduction rate
                if "reproduction_rate" in new_genome:
                    new_genome["reproduction_rate"] *= 1.05

            elif mutation_type == "pacification":
                # Temporary effect already applied to aggression, but might have lasting effects
                if "aggression_base" in new_genome:
                    new_genome["aggression_base"] *= 0.95

            elif mutation_type == "aggression_spike":
                if "aggression_base" in new_genome:
                    new_genome["aggression_base"] *= 1.05

            elif mutation_type == "boss_spawn":
                # Boss creation already handled elsewhere, this might affect group mechanics
                if "leadership" in new_genome:
                    new_genome["leadership"] *= 1.1

        # Natural random mutations (separate from feeding-induced ones)
        base_mutation_probability = self.base_mutation_rate
        for key in new_genome:
            if random.random() < base_mutation_probability:
                # Most mutations are small adjustments
                if random.random() < 0.8:
                    new_genome[key] *= random.uniform(0.95, 1.05)  # ±5%
                else:
                    # But some are more significant
                    new_genome[key] *= random.uniform(0.8, 1.2)  # ±20%

        # Ensure all genome values stay within reasonable bounds
        for key in new_genome:
            # Prevent values from getting too extreme
            if new_genome[key] < 0.1:
                new_genome[key] = 0.1
            elif new_genome[key] > 10.0:
                new_genome[key] = 10.0

        return new_genome

    def generate_cellular_automaton_rules(
        self, race_id: int, hunger_level: float, genome: Dict[str, float]
    ) -> Tuple[Set[int], Set[int]]:
        """
        Generate modified Conway's Game of Life rules based on race traits.

        Args:
            race_id: The ID of the symbiote race
            hunger_level: Current hunger level (0-1)
            genome: The race's genome dictionary

        Returns:
            Tuple of (birth_set, survival_set) with cellular automaton rules
        """
        # Base rules (standard Conway's Game of Life)
        base_birth_set = {3}
        base_survival_set = {2, 3}

        # Modify rules based on race_id (different species might have different base rules)
        if race_id == 1:  # Blue race - adaptive metabolism
            birth_set = base_birth_set.union({4})  # Can birth with 3 or 4 neighbors
            survival_set = base_survival_set  # Standard survival
        elif race_id == 2:  # Magenta race - aggressive expansion
            birth_set = base_birth_set  # Standard birth
            survival_set = base_survival_set.union(
                {1}
            )  # Can survive with just 1 neighbor
        elif race_id == 3:  # Orange race - selective
            birth_set = base_birth_set.union({6})  # Can birth with 3 or 6 neighbors
            survival_set = base_survival_set  # Standard survival
        else:
            birth_set = base_birth_set
            survival_set = base_survival_set

        # Modify rules based on hunger
        if hunger_level > 0.7:  # Very hungry
            # Hungry symbiotes try to expand more aggressively
            birth_set = birth_set.union({2})  # Easier to birth new cells
            # But might die more easily too (resource scarcity)
            if 4 in survival_set and random.random() < 0.5:
                survival_set.remove(4)
        elif hunger_level < 0.3:  # Well fed
            # Well-fed symbiotes are more stable
            survival_set = survival_set.union({4})  # Can survive with more neighbors

        # Modify rules based on genome traits
        expansion_drive = genome.get("expansion_drive", 1.0)
        if expansion_drive > 1.3 and random.random() < 0.3:
            birth_set.add(2)

        intelligence = genome.get("intelligence", 1.0)
        if intelligence > 1.5:  # Highly intelligent
            # More organized colonies, can survive in more configurations
            survival_set = survival_set.union({1, 4})

        metabolism_rate = genome.get("metabolism_rate", 1.0)
        if metabolism_rate < 0.7:  # Efficient metabolism
            # Can survive with fewer resources
            survival_set.add(1)
        elif metabolism_rate > 1.3:  # Inefficient/fast metabolism
            # Needs more resources, harder to survive
            if 1 in survival_set:
                survival_set.remove(1)

        # Random evolutionary mutations in rules
        if random.random() < 0.01:  # 1% chance of rule mutation each update
            possible_rules = {1, 2, 3, 4, 5, 6}

            # 50% chance to add a birth rule, 50% to add a survival rule
            if random.random() < 0.5 and len(birth_set) < 4:
                if candidates := list(possible_rules - birth_set):
                    birth_set.add(random.choice(candidates))
            elif len(survival_set) < 4:
                if candidates := list(possible_rules - survival_set):
                    survival_set.add(random.choice(candidates))

        return birth_set, survival_set

    def update_ml_model(self, game_outcome: str) -> None:
        """
        Update machine learning model based on game outcome.

        Args:
            game_outcome: String describing game outcome ('player_win', 'symbiote_win', etc.)
        """
        if not self.learning_enabled:
            return

        # Calculate win/loss ratio
        total_attacks = (
            self.ml_memory["successful_attacks"] + self.ml_memory["failed_attacks"]
        )
        if total_attacks == 0:
            return

        success_ratio = self.ml_memory["successful_attacks"] / total_attacks

        # If symbiotes are doing too well, decrease difficulty slightly
        if game_outcome == "player_loss" or success_ratio > 0.7:
            self.difficulty_adapter = max(0.5, self.difficulty_adapter - 0.05)

        # If symbiotes are struggling, increase difficulty slightly
        elif game_outcome == "player_win" or success_ratio < 0.3:
            self.difficulty_adapter = min(1.5, self.difficulty_adapter + 0.05)

        # Reset attack counters but maintain some memory
        self.ml_memory["successful_attacks"] = int(
            self.ml_memory["successful_attacks"] * 0.5
        )
        self.ml_memory["failed_attacks"] = int(self.ml_memory["failed_attacks"] * 0.5)

        # Analyze mineral preferences to adapt behavior
        total_minerals = sum(self.ml_memory["minerals_consumed"].values())
        if total_minerals > 0:
            # If certain minerals are preferred for feeding, adjust their effectiveness
            for mineral_type, amount in self.ml_memory["minerals_consumed"].items():
                ratio = amount / total_minerals

                # If this type is fed a lot, make it slightly less effective over time
                # (adaptation/resistance), but don't go below 50% effectiveness
                if ratio > 0.5 and self.mineral_influence[mineral_type] > 0.005:
                    self.mineral_influence[mineral_type] *= 0.95

    def update_cellular_automaton(
        self, grid: np.ndarray, birth_set: Set[int], survival_set: Set[int]
    ) -> np.ndarray:
        """
        Update the cellular automaton grid based on the current rules.

        Args:
            grid: The current state of the grid (2D numpy array where 1=symbiote, 0=empty)
            birth_set: Set of neighbor counts that cause a birth
            survival_set: Set of neighbor counts that allow survival

        Returns:
            Updated grid for next generation
        """
        # Use ndimage.convolve to count neighbors efficiently
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Count the number of live neighbors for each cell
        neighbor_count = ndimage.convolve(grid, kernel, mode="wrap")

        # Create a new grid based on birth and survival rules
        new_grid = np.zeros_like(grid)

        # Apply birth rules (empty cells with the right number of neighbors become alive)
        for birth_rule in birth_set:
            new_grid[(grid == 0) & (neighbor_count == birth_rule)] = 1

        # Apply survival rules (living cells with the right number of neighbors stay alive)
        for survival_rule in survival_set:
            new_grid[(grid == 1) & (neighbor_count == survival_rule)] = 1

        return new_grid

    def detect_patterns(self, grid: np.ndarray) -> Dict[str, int]:
        """
        Detect common cellular automaton patterns in the grid.

        Args:
            grid: The current state of the grid

        Returns:
            Dictionary with counts of common patterns detected
        """
        # Detect blocks (2x2 stable patterns)
        block_kernel = np.array([[1, 1], [1, 1]])
        block_matches = ndimage.binary_hit_or_miss(grid, block_kernel)
        patterns = {
            "blinker": 0,
            "glider": 0,
            "beehive": 0,
            "loaf": 0,
            "block": np.sum(block_matches),
        }
        # Detect blinkers (horizontal line of 3)
        blinker_h_kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        blinker_v_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])

        blinker_h_matches = ndimage.binary_hit_or_miss(grid, blinker_h_kernel)
        blinker_v_matches = ndimage.binary_hit_or_miss(grid, blinker_v_kernel)
        patterns["blinker"] = np.sum(blinker_h_matches) + np.sum(blinker_v_matches)

        # Detect beehives (common still life pattern)
        beehive_kernel1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]])

        beehive_matches = ndimage.binary_hit_or_miss(grid, beehive_kernel1)
        patterns["beehive"] = np.sum(beehive_matches)

        # We can add more pattern detection here in the future

        return patterns

    def calculate_expansion_index(
        self, grid: np.ndarray, previous_grid: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate how aggressively the symbiotes are expanding.

        Args:
            grid: Current state of the grid
            previous_grid: Previous state of the grid (optional)

        Returns:
            Expansion index (higher = more aggressive expansion)
        """
        # Get edge cells (cells at the boundary of the colony)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        neighbor_count = ndimage.convolve(grid, kernel, mode="constant", cval=0.0)

        # Edge cells are alive and don't have all 8 neighbors alive
        edge_cells = (grid == 1) & (neighbor_count < 8)

        # Calculate the ratio of edge cells to total cells
        total_cells = np.sum(grid)
        if total_cells == 0:
            return 0.0

        edge_count = np.sum(edge_cells)
        edge_ratio = edge_count / total_cells

        # If we have previous grid, calculate growth rate
        growth_rate = 0.0
        if previous_grid is not None:
            previous_count = np.sum(previous_grid)
            if previous_count > 0:
                growth_rate = (total_cells - previous_count) / previous_count

        return (edge_ratio * 0.7) + (max(0, growth_rate) * 0.3)

    def apply_environmental_effects(
        self,
        grid: np.ndarray,
        mineral_map: np.ndarray,
        environment_hostility: float = 0.1,
    ) -> np.ndarray:
        """
        Apply environmental effects to the cellular automaton grid.

        Args:
            grid: Current symbiote grid
            mineral_map: Map of mineral concentrations (higher = more minerals)
            environment_hostility: How hostile the environment is (0-1)

        Returns:
            Updated grid after environmental effects
        """
        # Create a copy of the grid to modify
        new_grid = grid.copy()

        # 1. Growth boost near minerals
        # Calculate mineral influence using gaussian filter for a diffusion effect
        mineral_influence = ndimage.gaussian_filter(mineral_map, sigma=1.5)

        # Apply mineral growth boost - chance for empty cells near minerals to spawn symbiotes
        growth_mask = (grid == 0) & (mineral_influence > 0.5)
        growth_probability = mineral_influence * 0.2  # 20% max probability
        random_values = np.random.random(grid.shape)
        new_grid[(growth_mask) & (random_values < growth_probability)] = 1

        # 2. Apply environmental hazards
        if environment_hostility > 0:
            # Create a random hazard map
            hazard_map = np.random.random(grid.shape) < environment_hostility

            # Apply hazards - kill cells in hazardous areas
            new_grid[hazard_map & (grid == 1)] = 0

        # 3. Apply erosion to isolated cells (symbiotes die if too isolated)
        neighbor_count = (
            ndimage.convolve(grid, np.ones((3, 3)), mode="constant", cval=0.0) - grid
        )
        new_grid[(grid == 1) & (neighbor_count == 0)] = 0

        return new_grid

    def identify_colonies(self, grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Identify distinct colonies in the symbiote grid using connected component labeling.

        Args:
            grid: The symbiote grid

        Returns:
            Tuple of (labeled_grid, number_of_colonies)
        """
        # Use scipy.ndimage to label connected components
        labeled_grid, num_colonies = ndimage.label(grid, structure=np.ones((3, 3)))

        return labeled_grid, num_colonies

    def get_colony_stats(
        self, grid: np.ndarray, labeled_grid: np.ndarray, num_colonies: int
    ) -> List[Dict]:
        """
        Calculate statistics for each colony.

        Args:
            grid: The original symbiote grid
            labeled_grid: The grid with labeled colonies
            num_colonies: Number of distinct colonies

        Returns:
            List of dictionaries with stats for each colony
        """
        colony_stats = []

        for i in range(1, num_colonies + 1):
            # Create mask for this colony
            colony_mask = labeled_grid == i

            # Calculate basic stats
            size = np.sum(colony_mask)

            # Find centroid (center of mass)
            centroid = ndimage.center_of_mass(colony_mask)

            # Calculate compactness (ratio of area to perimeter)
            # First, get the perimeter cells
            eroded = ndimage.binary_erosion(colony_mask)
            perimeter_cells = np.sum(colony_mask & ~eroded)
            compactness = size / max(1, perimeter_cells)

            # Store stats for this colony
            colony_stats.append(
                {
                    "id": i,
                    "size": int(size),
                    "centroid": centroid,
                    "compactness": compactness,
                }
            )

        return colony_stats

    def simulate_colony_interaction(
        self, grid: np.ndarray, genome: Dict[str, float], aggression: float
    ) -> np.ndarray:
        """
        Simulate interactions between colonies based on symbiote traits.

        Args:
            grid: The symbiote grid
            genome: The symbiote genome traits
            aggression: Current aggression level

        Returns:
            Updated grid after colony interactions
        """
        # Identify colonies
        labeled_grid, num_colonies = self.identify_colonies(grid)

        if num_colonies <= 1:
            return grid  # No interactions needed

        # Get stats for each colony
        colony_stats = self.get_colony_stats(grid, labeled_grid, num_colonies)

        # Create new grid for the result
        new_grid = grid.copy()

        # Calculate interaction probabilities based on genome and aggression
        territorial_factor = genome.get("territory_size", 1.0) * aggression

        # For each pair of colonies, determine if they fight, merge, or ignore each other
        for i in range(len(colony_stats)):
            for j in range(i + 1, len(colony_stats)):
                colony_i = colony_stats[i]
                colony_j = colony_stats[j]

                # Calculate distance between colonies
                dist = np.sqrt(
                    (colony_i["centroid"][0] - colony_j["centroid"][0]) ** 2
                    + (colony_i["centroid"][1] - colony_j["centroid"][1]) ** 2
                )

                # Only interact if colonies are close enough
                interaction_range = 10 * territorial_factor
                if dist <= interaction_range:
                    # Determine interaction type based on traits and random chance
                    interaction_roll = random.random()

                    # Fight (aggressive interaction)
                    if interaction_roll < aggression * 0.7:
                        # Stronger colony damages weaker colony proportional to size difference
                        if colony_i["size"] > colony_j["size"]:
                            winner_id, loser_id = i + 1, j + 1
                            strength_ratio = colony_i["size"] / max(1, colony_j["size"])
                        else:
                            winner_id, loser_id = j + 1, i + 1
                            strength_ratio = colony_j["size"] / max(1, colony_i["size"])

                            # Recalculate strength ratio for the winner
                            strength_ratio = colony_j["size"] / max(1, colony_i["size"])
                        # Create winner mask to allow territory expansion
                        winner_mask = labeled_grid == winner_id

                        # Allow winner colony to expand into nearby empty spaces
                        expansion_kernel = np.ones((3, 3))
                        expansion_area = ndimage.binary_dilation(
                            winner_mask, expansion_kernel
                        )
                        empty_cells = grid == 0
                        new_grid[expansion_area & empty_cells] = (
                            1  # Expand into empty cells
                        )
                        # Calculate damage percentage based on strength difference
                        damage_percent = min(0.5, strength_ratio * 0.1)

                        # Apply damage - randomly remove cells from loser colony
                        loser_mask = labeled_grid == loser_id
                        damage_mask = loser_mask & (
                            np.random.random(grid.shape) < damage_percent
                        )
                        new_grid[damage_mask] = 0

                    # Merge (cooperative interaction)
                    elif interaction_roll < aggression * 0.7 + 0.2:
                        # Create a bridge between colonies
                        c1 = colony_stats[i]["centroid"]
                        c2 = colony_stats[j]["centroid"]

                        # Calculate a simple line between centroids
                        steps = max(3, int(dist / 2))
                        for step in range(steps):
                            x = int(c1[0] + (c2[0] - c1[0]) * step / steps)
                            y = int(c1[1] + (c2[1] - c1[1]) * step / steps)

                            # Stay within grid bounds
                            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                                new_grid[x, y] = 1

        return new_grid

    def compute_ecological_pressure(
        self, population: int, resource_availability: float, competition_factor: float
    ) -> float:
        """
        Calculate ecological pressure using modified Verhulst-Pearl logistic model

        Args:
            population: Current population size
            resource_availability: Available resources (0-1)
            competition_factor: Competition from other races (0-1)

        Returns:
            Pressure value (higher means more pressure)
        """
        # Base carrying capacity adjusted by resource availability
        K = self.carrying_capacity * (0.5 + resource_availability)

        # Competition adjusted growth rate (used to modify pressure calculation)
        r = self.growth_rate * (1 - competition_factor)

        # Calculate pressure using a modified logistic function
        if population > 0:
            # P = 1 / (1 + exp(-k * (N/K - threshold)))
            # Use growth rate to influence the steepness of the logistic curve
            pressure = 1.0 / (1.0 + math.exp(-5.0 * r * (population / K - 0.7)))
        else:
            pressure = 0.0

        # Apply minimum pressure to prevent extinction
        return max(0.05, pressure)

    def compute_mutation_probability(
        self,
        mineral_types: Dict[str, int],
        current_generation: int,
        population_trend: float,
    ) -> Dict[str, float]:
        """
        Calculate mutation probabilities using enhanced mathematical models

        Args:
            mineral_types: Dictionary of mineral counts by type
            current_generation: Current generation number
            population_trend: Population growth rate (-1 to 1)

        Returns:
            Dictionary of mutation probabilities by trait
        """
        # Base mutation rates for different traits
        base_rates = {
            "metabolism": self.base_mutation_rate,
            "aggression": self.base_mutation_rate * 0.8,
            "reproduction": self.base_mutation_rate * 1.2,
            "intelligence": self.base_mutation_rate * 0.5,
            "resilience": self.base_mutation_rate * 0.7,
            "adaptation": self.base_mutation_rate * 1.5,
        }

        # Calculate mineral influence factor
        # Exponential scaling of rare minerals - rarer minerals have exponentially greater impact
        mineral_power = {
            "common": 1.0,  # Linear scaling
            "rare": 1.3,  # Slightly superlinear
            "precious": 1.5,  # Moderate superlinear
            "anomaly": 2.0,  # Quadratic scaling
        }

        # Calculate combined mineral factor for mutation probabilities
        mineral_factor = 1.0
        for mineral_type, count in mineral_types.items():
            if mineral_type in mineral_power and count > 0:
                # Apply power scaling to each mineral type
                power = mineral_power.get(mineral_type, 1.0)
                mineral_factor += self.mineral_influence.get(mineral_type, 0.01) * (
                    count**power
                )

        # Apply time-dependent mutation factor (mutations more likely in later generations)
        time_factor = 1.0 + math.log(max(1, current_generation)) * 0.01

        # Apply population trend factor (struggling populations mutate more aggressively)
        # Negative population trend increases mutation rate
        trend_factor = 1.0 - min(0.5, max(-0.5, population_trend)) * 0.5

        return {
            trait: min(0.8, base_rate * mineral_factor * time_factor * trend_factor)
            for trait, base_rate in base_rates.items()
        }

    def apply_differential_evolution(
        self,
        population: int,
        genome: Dict[str, float],
        environment_factors: Dict[str, float],
        generation: int,
    ) -> Dict[str, float]:
        """
        Apply differential evolution algorithm to symbiote genome

        Args:
            population: Current population
            genome: Current genome traits
            environment_factors: Environmental influences
            generation: Current generation number

        Returns:
            Updated genome dictionary
        """
        # Skip if population too low for meaningful evolution
        if population < 3:
            return genome.copy()

        new_genome = genome.copy()

        # Apply differential evolution with three components:
        # 1. Current value
        # 2. Mutation based on difference between best and average traits
        # 3. Random exploration component

        # Simulate idealized "best" genome for this environment
        ideal_genome = {
            "metabolism_rate": 1.0
            + environment_factors.get("resource_abundance", 0) * 0.5,
            "mutation_rate": 0.01
            + environment_factors.get("environmental_variability", 0) * 0.05,
            "aggression": 0.2 + environment_factors.get("competition", 0) * 0.8,
            "intelligence": 1.0 + environment_factors.get("complexity", 0) * 0.5,
            "expansion_drive": 1.0
            + environment_factors.get("available_space", 0) * 0.5,
            "territory_size": 1.0
            + environment_factors.get("resource_distribution", 0) * 0.5,
        }

        # Differential evolution parameters
        crossover_rate = 0.7
        differential_weight = 0.8 * (
            1.0 - math.exp(-generation / 50)
        )  # Increases with generation

        # Apply differential evolution to each trait
        for trait in new_genome:
            if trait in ideal_genome and random.random() < crossover_rate:
                # Calculate the difference vector (ideal - current)
                diff = ideal_genome[trait] - new_genome[trait]

                # Apply weighted difference
                new_genome[trait] += differential_weight * diff

                # Add random mutation (exploration)
                mutation_scale = 0.1 * math.exp(
                    -generation / 100
                )  # Decreases with generation
                new_genome[trait] += random.gauss(0, mutation_scale)

        # Ensure values are within reasonable bounds
        for trait in new_genome:
            new_genome[trait] = max(0.1, min(5.0, new_genome[trait]))

        return new_genome
