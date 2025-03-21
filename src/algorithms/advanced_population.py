"""
advanced_population.py

Provides deeper population and behavior models for symbiotes, integrating
advanced ecological and biological concepts such as multi-species predator-prey,
delayed/lagged growth, and stage-structured evolution. Designed to interoperate
with the existing SymbioteEvolutionAlgorithm or related game logic.

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------
1. MultiSpeciesPopulation:
   - Simulate multiple interacting species (including player or other factions)
     using an extended Lotka–Volterra or competitive-coexistence model.
   - Each species can have its own logistic baseline (growth rate, carrying capacity)
     plus pairwise interaction coefficients (predation, competition, mutualism).
   - Interactions update each timestep, influencing population sizes and resource usage.

2. DelayedGrowthPopulation:
   - Introduces discrete approximations of delay differential equations, modeling
     gestation or resource transport lags.
   - Population at time t depends on population from (t - delay), causing more
     realistic boom-bust cycles if parameters are set appropriately.

3. StageStructuredPopulation:
   - Splits the symbiote life cycle into multiple stages (e.g., Egg, Juvenile, Adult, Boss).
   - Each stage transitions to the next at a configurable rate.
   - Growth and mortality can be stage-specific, allowing deeper lifecycle simulation.

All classes here aim to be production-ready, with clear docstrings, and no usage examples
are provided per instructions. Integrate them into your main game loop or evolutionary
logic by instantiating the relevant class(es) and calling update methods each turn. Tie
them into the existing aggression system, feeding routines, or machine learning memory
if desired.

NOTE: Ensure any references to external methods or data (e.g. feeding logs, environment
factors) are adapted to your codebase. For example, hook MultiSpeciesPopulation into
your SymbioteEvolutionAlgorithm by passing current symbiote populations at each step.

--------------------------------------------------------------------------------
LICENSE / COPYRIGHT NOTICE
--------------------------------------------------------------------------------
Copyright (c) 2025, ...
All rights reserved.
"""

from typing import Dict, List


class MultiSpeciesPopulation:
    """
    Implements a multi-species population model using an extended
    Lotka–Volterra approach. Each species i has:
      - A logistic growth baseline: dN_i/dt = r_i * N_i * (1 - N_i/K_i)
      - Interaction terms for every other species j, scaled by interaction[i][j].
        * Positive values => i preys on j or benefits from j
        * Negative values => i competes with j

    Usage:
        1. Initialize with an interaction matrix describing how species
           affect one another.
        2. On each update, pass in current populations to multi_species_step()
           to get new population levels.

    Integration:
        - Combine with feeding logic or environment checks by adjusting the
          carrying capacities or logistic terms in real time.
        - If the player is considered a "species," you can model their
          population or fleet similarly, though typically they'd have
          separate combat logic.
    """

    def __init__(
        self,
        species_count: int,
        base_growth_rates: List[float],
        carrying_caps: List[float],
        interaction_matrix: List[List[float]],
    ):
        """
        Args:
            species_count: Number of species tracked (including symbiote variants, players, etc.).
            base_growth_rates: Growth rate r_i for each species i (length = species_count).
            carrying_caps: Carrying capacity K_i for each species i (length = species_count).
            interaction_matrix: A 2D matrix [species_count x species_count] describing
                                pairwise interactions. interaction_matrix[i][j] is the
                                coefficient for how species j influences species i.
        """
        assert (
            len(base_growth_rates) == species_count
        ), "Mismatch in species_count vs. growth_rates"
        assert (
            len(carrying_caps) == species_count
        ), "Mismatch in species_count vs. carrying_caps"
        assert (
            len(interaction_matrix) == species_count
        ), "Mismatch in species_count vs. interaction_matrix"

        for row in interaction_matrix:
            assert len(row) == species_count, "Interaction matrix row length mismatch"

        self.species_count = species_count
        self.rates = base_growth_rates
        self.caps = carrying_caps
        self.interaction_matrix = interaction_matrix

        # If desired, you can store additional per-species data (e.g., names, IDs).

    def multi_species_step(
        self, populations: List[float], dt: float = 1.0
    ) -> List[float]:
        """
        Advance each species by one step, returning new population levels.

        Equation used:
            dN_i/dt = r_i*N_i * (1 - (N_i / K_i)) + sum_{j != i}( interaction[i][j] * N_i * N_j )

        The sign and magnitude of interaction[i][j] define how species j affects i:
          - Positive => i benefits from j (e.g., i preys on j).
          - Negative => i is harmed by j (competition).
          - Zero => no direct interaction.

        Args:
            populations: The current populations of each species (length = species_count).
            dt: The timestep for each update.

        Returns:
            A list of floats representing the updated populations for each species.
        """
        new_pops = populations.copy()
        for i in range(self.species_count):
            ni = populations[i]  # noqa: N806 - Using mathematical notation
            if ni <= 0.0:
                new_pops[i] = 0.0
                continue

            r = self.rates[i]
            K = max(1e-9, self.caps[i])  # noqa: N806 - Using mathematical notation

            # Logistic baseline
            logistic_term = r * ni * (1.0 - ni / K)

            # Interaction
            interaction_sum = 0.0
            for j in range(self.species_count):
                if j == i:
                    continue
                nj = populations[j]
                interaction_sum += self.interaction_matrix[i][j] * ni * nj

            dni_dt = logistic_term + interaction_sum
            updated_val = ni + dni_dt * dt

            new_pops[i] = max(0.0, updated_val)

        return new_pops


class DelayedGrowthPopulation:
    """
    Introduces a delay into population growth by referencing previous states
    from a fixed number of steps ago. This approximates certain delay differential
    equations, commonly used when a population's reproductive output depends on
    conditions from earlier periods (gestation, resource acquisition lags, etc.).

    Usage:
        1. Call record_population() each turn before update_population().
        2. update_population() uses population from `delay_steps` turns ago
           to compute the new population. The logic can cause oscillatory
           or lagged responses.

    Integration:
        - You can combine this with logistic or other forms of population updates
          by adjusting the formula in update_population().
        - For synergy with aggression or feeding, pass relevant parameters
          (e.g. food availability) into the growth function as well.
    """

    def __init__(
        self,
        delay_steps: int,
        base_growth_rate: float,
        max_capacity: float,
    ):
        """
        Args:
            delay_steps: How many turns behind the current population
                         influences next growth. Must be >= 1.
            base_growth_rate: Growth rate factor (like r in logistic equations).
            max_capacity: Optional carrying capacity for capping population.
        """
        self.delay_steps = max(1, delay_steps)
        self.base_growth_rate = base_growth_rate
        self.max_capacity = max_capacity
        self.history = []

    def record_population(self, current_pop: float) -> None:
        """
        Call this each turn to store the latest population in the history buffer.
        """
        self.history.append(current_pop)

    def update_population(self, current_pop: float) -> float:
        """
        Computes a new population value using population from `delay_steps` steps ago.
        A simple model: population grows by base_growth_rate * old_pop,
        then gets capped at max_capacity if specified.

        Returns:
            The updated population after applying the delayed growth term.
        """
        if current_pop < 0.1:
            new_pop = 0.0
        else:
            if len(self.history) <= self.delay_steps:
                # Not enough history to apply delay, so do a minimal update
                growth = self.base_growth_rate * current_pop
            else:
                # Use population from delay_steps ago
                old_pop = self.history[-(self.delay_steps)]
                growth = self.base_growth_rate * old_pop
            new_pop = current_pop + growth
        # Enforce carrying capacity if applicable
        if self.max_capacity > 0.0:
            new_pop = min(new_pop, self.max_capacity)

        return max(0.0, new_pop)


class StageStructuredPopulation:
    """
    Splits population into multiple life stages, each with different
    transition, mortality, or reproduction rates. For example:
      - Egg -> Larva -> Adult -> Boss
    At each update, a fraction of individuals in each stage transition
    to the next stage, possibly with unique growth or mortality.

    Usage:
        1. Provide a list of stage names (strings) in order.
        2. Provide transitions dict with rates for how many from each stage
           become the next each turn.
        3. Provide any optional mortality/growth logic in update_stages().
    """

    def __init__(
        self,
        stages: List[str],
        transitions: Dict[str, float],
        stage_mortality: Dict[str, float] = None,
        stage_growth: Dict[str, float] = None,
    ):
        """
        Args:
            stages: Ordered list of stage names (e.g., ["egg", "juvenile", "adult", "boss"]).
            transitions: Fraction of each stage that moves to the next stage per update,
                         e.g., {"egg": 0.3, "juvenile": 0.5, "adult": 0.1, "boss": 0.0}.
            stage_mortality: Optional dict mapping each stage to a mortality fraction
                             (e.g., 0.05 means 5% die each step).
            stage_growth: Optional dict mapping each stage to an extra birth factor
                          (e.g., how many new eggs each adult creates).
        """
        self.stages = stages
        self.transitions = transitions
        self.stage_mortality = stage_mortality or {}
        self.stage_growth = stage_growth or {}

    def update_stages(self, stage_pops: Dict[str, float]) -> Dict[str, float]:
        """
        Update the stage populations by:
          1. Applying mortality to each stage.
          2. Generating new individuals if certain stages produce offspring.
          3. Transitioning a fraction of each stage to the next stage.

        Returns:
            A new dict of updated stage populations with the same keys as self.stages.
        """
        new_stage_pops = {st: 0.0 for st in self.stages}

        # 1. Apply mortality and record survivors
        survivors = {}
        for st in self.stages:
            pop = stage_pops.get(st, 0.0)
            mort = self.stage_mortality.get(st, 0.0)
            mort = max(mort, 0.0)
            mort = min(mort, 1.0)
            alive = pop * (1.0 - mort)
            survivors[st] = max(0.0, alive)

        # 2. Generate offspring from stage_growth
        #    e.g., adult stage might produce new eggs
        for st in self.stages:
            base_pop = survivors.get(st, 0.0)
            birth_rate = self.stage_growth.get(st, 0.0)
            # If birth_rate is e.g. 1.0 => 1 new egg per adult
            # Add those new to the first stage (e.g. "egg") or however you design
            if birth_rate > 0.0 and len(self.stages) > 0:
                new_offspring = base_pop * birth_rate
                # By default, assume the first stage is the "lowest" stage (like "egg")
                first_stage = self.stages[0]
                new_stage_pops[first_stage] += new_offspring

        # 3. Transition
        for i, st in enumerate(self.stages):
            pop = survivors[st]
            transition_rate = self.transitions.get(st, 0.0)
            remain = pop * (1.0 - transition_rate)
            new_stage_pops[st] += remain

            # Move the transitioning fraction to the next stage (if it exists)
            if i < len(self.stages) - 1:
                next_stage = self.stages[i + 1]
                new_stage_pops[next_stage] += pop * transition_rate

        # Enforce non-negative populations
        for st in new_stage_pops:
            new_stage_pops[st] = max(0.0, new_stage_pops[st])

        return new_stage_pops
