"""
src/algorithms/population_models.py

Population Models for Space Muck

This module implements advanced population models for the simulation of mining races:
- MultiSpeciesPopulation: Implements interactions between multiple species using Lotka-Volterra models
- DelayedGrowthPopulation: Models population growth with delays based on previous states
- StageStructuredPopulation: Divides populations into life stages with unique transition and mortality rates
"""

import collections
from typing import Dict, List, Optional


class MultiSpeciesPopulation:
    """
    Implements a multi-species population model based on Lotka-Volterra equations.
    This model simulates interactions between different species, including:
    - competition (negative interactions)
    - mutualism (positive interactions)
    - predator-prey relationships (mixed interactions)

    The system of equations is as follows:
    dN_i/dt = r_i * N_i * (1 - N_i/K_i + sum_j(alpha_ij * N_j/K_i))

    Where:
    - N_i is the population of species i
    - r_i is the intrinsic growth rate of species i
    - K_i is the carrying capacity of species i
    - alpha_ij is the interaction coefficient between species i and j
      (positive = beneficial, negative = competitive)
    """

    def __init__(
        self,
        species_count: int,
        base_growth_rates: List[float],
        carrying_caps: List[float],
        interaction_matrix: Optional[List[List[float]]] = None,
    ):
        """
        Initialize the multi-species population model.

        Args:
            species_count: Number of species in the system
            base_growth_rates: Intrinsic growth rates for each species
            carrying_caps: Carrying capacities for each species
            interaction_matrix: Matrix of interaction coefficients between species
                (if None, a default slightly competitive matrix is created)
        """
        self.species_count = species_count

        # Validate input lengths
        if len(base_growth_rates) != species_count:
            raise ValueError(
                f"Growth rates list length ({len(base_growth_rates)}) must match species count ({species_count})"
            )
        if len(carrying_caps) != species_count:
            raise ValueError(
                f"Carrying capacity list length ({len(carrying_caps)}) must match species count ({species_count})"
            )

        self.growth_rates = base_growth_rates.copy()
        self.carrying_capacities = carrying_caps.copy()

        # Create default interaction matrix if none provided
        if interaction_matrix is None:
            # Default: slight competition between species (-0.01)
            self.interaction_matrix = [
                [-0.01 if i != j else 0.0 for j in range(species_count)]
                for i in range(species_count)
            ]
        else:
            # Validate matrix dimensions
            if len(interaction_matrix) != species_count:
                raise ValueError(
                    f"Interaction matrix rows ({len(interaction_matrix)}) must match species count ({species_count})"
                )
            for row in interaction_matrix:
                if len(row) != species_count:
                    raise ValueError(
                        f"Interaction matrix columns must match species count ({species_count})"
                    )
            self.interaction_matrix = [row.copy() for row in interaction_matrix]

    def multi_species_step(
        self, populations: List[float], dt: float = 1.0
    ) -> List[float]:
        """
        Calculate one step of the multi-species model.

        Args:
            populations: Current population values for each species
            dt: Time step size

        Returns:
            Updated population values for each species
        """
        # Validate input
        if len(populations) != self.species_count:
            raise ValueError(
                f"Population list length ({len(populations)}) must match species count ({self.species_count})"
            )

        new_populations = []

        for i in range(self.species_count):
            # Skip extinct species
            if populations[i] <= 0:
                new_populations.append(0.0)
                continue

            # Calculate interaction term for species i
            interaction_term = 0.0
            for j in range(self.species_count):
                if (
                    i != j and populations[j] > 0
                ):  # Skip self-interaction and extinct species
                    interaction_term += (
                        self.interaction_matrix[i][j]
                        * populations[j]
                        / self.carrying_capacities[i]
                    )

            # Lotka-Volterra equation for species i
            growth_factor = (
                1.0 - (populations[i] / self.carrying_capacities[i]) + interaction_term
            )

            # Calculate population change
            dn = self.growth_rates[i] * populations[i] * growth_factor * dt

            # Update population with bounds checking
            new_pop = max(0.0, populations[i] + dn)
            new_populations.append(new_pop)

        return new_populations

    def set_interaction(self, species1: int, species2: int, value: float) -> None:
        """
        Set the interaction coefficient between two species.

        Args:
            species1: Index of the first species (affected species)
            species2: Index of the second species (affecting species)
            value: Interaction coefficient (positive = beneficial, negative = competitive)
        """
        if not (
            0 <= species1 < self.species_count and 0 <= species2 < self.species_count
        ):
            raise ValueError(
                f"Species indices must be between 0 and {self.species_count-1}"
            )

        self.interaction_matrix[species1][species2] = value


class DelayedGrowthPopulation:
    """
    Models population growth with time delays. This simulates scenarios where:
    - Resource acquisition has a delayed effect on population growth
    - Past events impact current growth rates
    - Environmental carrying capacity changes over time

    This model uses a time-series approach to track population history
    and apply delayed effects to current growth calculations.
    """

    def __init__(
        self,
        delay_steps: int,
        base_growth_rate: float,
        max_capacity: float,
        history_length: int = None,
    ):
        """
        Initialize the delayed growth population model.

        Args:
            delay_steps: Number of steps before resource effects are realized
            base_growth_rate: Baseline intrinsic growth rate
            max_capacity: Maximum population carrying capacity
            history_length: Maximum history length to store (defaults to 2*delay_steps)
        """
        self.delay_steps = delay_steps
        self.base_growth_rate = base_growth_rate
        self.max_capacity = max_capacity

        # Set default history length if not provided
        if history_length is None:
            history_length = 2 * delay_steps
        self.history_length = max(history_length, delay_steps + 1)

        # Initialize history queue with zeros
        self.history = collections.deque(
            [0.0] * self.history_length, maxlen=self.history_length
        )

        # Multipliers for delayed effects (can be modified)
        self.delay_multipliers = [1.0] * (delay_steps + 1)

    def record_population(self, population: float) -> None:
        """
        Record the current population in the history queue.

        Args:
            population: Current population value to record
        """
        self.history.appendleft(population)

    def update_population(
        self, current_population: float, resources: float = 0.0
    ) -> float:
        """
        Calculate the updated population using the delayed growth model.

        Args:
            current_population: Current population value
            resources: Resource amount available (affects growth)

        Returns:
            Updated population value
        """
        # Ensure history contains at least delay_steps + 1 entries
        if len(self.history) <= self.delay_steps:
            return current_population

        # Calculate delayed effect from history
        delayed_effect = 0.0
        for i in range(min(self.delay_steps + 1, len(self.history))):
            delayed_effect += self.history[i] * self.delay_multipliers[i]
        delayed_effect = delayed_effect / sum(
            self.delay_multipliers[: self.delay_steps + 1]
        )

        # Adjust growth rate based on resources
        adjusted_growth_rate = self.base_growth_rate * (1.0 + resources / 100.0)

        # Calculate logistic growth with delayed effect
        capacity_factor = 1.0 - (delayed_effect / self.max_capacity)
        growth = adjusted_growth_rate * current_population * capacity_factor

        return max(0.0, min(self.max_capacity, current_population + growth))

    def set_delay_multipliers(self, multipliers: List[float]) -> None:
        """
        Set custom multipliers for the delayed effects.

        Args:
            multipliers: List of multipliers for each delay step
                (length should match delay_steps + 1)
        """
        if len(multipliers) != self.delay_steps + 1:
            raise ValueError(
                f"Multiplier list length ({len(multipliers)}) must match delay_steps + 1 ({self.delay_steps + 1})"
            )

        self.delay_multipliers = multipliers.copy()


class StageStructuredPopulation:
    """
    Models a population divided into discrete life stages (age/development classes),
    each with unique traits such as:
    - Stage-specific mortality rates
    - Transition probabilities between stages
    - Stage-specific growth/reproduction rates

    This approach is based on stage-structured population models in ecology,
    similar to Leslie matrices but with more flexible transition rules.
    """

    def __init__(
        self,
        stages: List[str],
        transitions: Dict[str, float],
        stage_mortality: Dict[str, float],
        stage_growth: Dict[str, float],
    ):
        """
        Initialize the stage-structured population model.

        Args:
            stages: List of stage names in order of progression
            transitions: Dict mapping stage name to transition probability to next stage
            stage_mortality: Dict mapping stage name to mortality rate
            stage_growth: Dict mapping stage name to reproduction rate
        """
        self.stages = stages

        # Validate input dictionaries contain all stages
        for stage in stages:
            if stage not in transitions and stage != stages[-1]:
                raise ValueError(f"Missing transition rate for stage '{stage}'")
            if stage not in stage_mortality:
                raise ValueError(f"Missing mortality rate for stage '{stage}'")
            if stage not in stage_growth:
                raise ValueError(f"Missing growth rate for stage '{stage}'")

        self.transitions = transitions.copy()
        self.stage_mortality = stage_mortality.copy()
        self.stage_growth = stage_growth.copy()

    def update_stages(
        self, stage_populations: Dict[str, float], environmental_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Update the population in each life stage for one time step.

        Args:
            stage_populations: Dict mapping stage name to current population
            environmental_factor: Multiplier for growth rates (>1 favorable, <1 unfavorable)

        Returns:
            Updated populations for each stage
        """
        # Validate input dictionary contains all stages
        for stage in self.stages:
            if stage not in stage_populations:
                raise ValueError(f"Missing population for stage '{stage}'")

        # Create a new dictionary for updated populations
        new_populations = {stage: 0.0 for stage in self.stages}

        # Calculate transitions and mortality for each stage
        for i, stage in enumerate(self.stages):
            current_pop = stage_populations[stage]

            # Skip if population is zero
            if current_pop <= 0:
                continue

            # Apply mortality
            survivors = current_pop * (1.0 - self.stage_mortality[stage])

            # Calculate transition to next stage
            if i < len(self.stages) - 1:  # Not the final stage
                next_stage = self.stages[i + 1]
                transition_pop = survivors * self.transitions[stage]
                remaining_pop = survivors - transition_pop

                # Add to appropriate stages
                new_populations[stage] += remaining_pop
                new_populations[next_stage] += transition_pop
            else:  # Final stage
                new_populations[stage] += survivors

        # Calculate reproduction (new individuals added to first stage)
        new_births = 0.0
        for stage in self.stages:
            # Each stage contributes new individuals to the first stage
            stage_pop = stage_populations[stage]
            if stage_pop > 0:
                births = stage_pop * self.stage_growth[stage] * environmental_factor
                new_births += births

        # Add new births to the first stage
        new_populations[self.stages[0]] += new_births

        # Round populations to avoid floating point issues
        for stage in new_populations:
            new_populations[stage] = max(0.0, new_populations[stage])

        return new_populations

    def calculate_total_population(self, stage_populations: Dict[str, float]) -> float:
        """
        Calculate the total population across all stages.

        Args:
            stage_populations: Dict mapping stage name to current population

        Returns:
            Total population sum
        """
        return sum(stage_populations.values())

    def get_stage_distribution(
        self, stage_populations: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate the proportion of population in each stage.

        Args:
            stage_populations: Dict mapping stage name to current population

        Returns:
            Dict mapping stage name to proportion of total population
        """
        total = self.calculate_total_population(stage_populations)
        if total <= 0:
            return {stage: 0.0 for stage in self.stages}

        return {stage: pop / total for stage, pop in stage_populations.items()}
