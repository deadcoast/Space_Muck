"""
Enhanced Miner Entity for Space Muck

This module implements an enhanced version of the MinerEntity that incorporates
advanced population models for more realistic ecological dynamics and life-cycle progression.
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from entities.miner_entity import MinerEntity
from algorithms.population_models import (
    MultiSpeciesPopulation,
    DelayedGrowthPopulation,
    StageStructuredPopulation
)


class EnhancedMinerEntity(MinerEntity):
    """
    Enhanced version of MinerEntity that incorporates advanced population models
    for more realistic ecological dynamics and life-cycle progression.
    """
    
    def __init__(
        self,
        race_id: int,
        color: Optional[Tuple[int, int, int]] = None,
        birth_set: Optional[Set[int]] = None,
        survival_set: Optional[Set[int]] = None,
        initial_density: float = 0.001,
        position: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> None:
        """
        Initialize enhanced miner entity with advanced population models.
        
        Args:
            race_id: Unique identifier for the race
            color: RGB color tuple for visualization
            birth_set: Set of neighbor counts that trigger cell birth
            survival_set: Set of neighbor counts that allow cell survival
            initial_density: Initial population density (0.0-1.0)
            position: (x, y) coordinates of the miner base
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            race_id=race_id,
            color=color,
            birth_set=birth_set,
            survival_set=survival_set,
            initial_density=initial_density,
            position=position,
            **kwargs
        )
        
        # Initialize multi-species population model for interaction with other races
        self.multi_species_model = MultiSpeciesPopulation(
            species_count=4,  # Default max number of races
            base_growth_rates=[0.05 * self.genome["metabolism_rate"]] * 4,
            carrying_caps=[1000, 1000, 1000, 1000],  # Default carrying capacities
            interaction_matrix=self._initialize_interaction_matrix()
        )
        
        # Initialize delayed growth for resource acquisition lag simulation
        self.delayed_growth = DelayedGrowthPopulation(
            delay_steps=3,  # Resource processing takes time
            base_growth_rate=0.05 * self.genome["metabolism_rate"],
            max_capacity=1000
        )
        
        # Initialize life stage progression for miners
        self.stage_population = StageStructuredPopulation(
            stages=["juvenile", "worker", "specialized", "elder"],
            transitions={
                "juvenile": 0.2,  # 20% of juveniles become workers each turn
                "worker": 0.05,  # 5% of workers become specialized each turn
                "specialized": 0.02,  # 2% of specialized become elders each turn
                "elder": 0.0  # Elders do not transition further
            },
            stage_mortality={
                "juvenile": 0.05,  # 5% juvenile mortality
                "worker": 0.02,  # 2% worker mortality
                "specialized": 0.01,  # 1% specialized mortality 
                "elder": 0.1  # 10% elder mortality
            },
            stage_growth={
                "juvenile": 0.0,  # Juveniles don't reproduce
                "worker": 0.2,  # Each worker has 20% chance to produce juvenile
                "specialized": 0.1,  # Specialized have 10% chance
                "elder": 0.05  # Elders have 5% chance
            }
        )
        
        # Initial stage populations
        self.stage_populations = {
            "juvenile": int(self.population * 0.4),  # 40% juveniles
            "worker": int(self.population * 0.3),  # 30% workers
            "specialized": int(self.population * 0.2),  # 20% specialized
            "elder": int(self.population * 0.1),  # 10% elders
        }
        
        # Ensure total matches initial population
        total = sum(self.stage_populations.values())
        if total > 0:
            # Normalize to match actual population
            factor = self.population / total
            for stage in self.stage_populations:
                self.stage_populations[stage] = int(self.stage_populations[stage] * factor)
            
            # Ensure we don't lose population due to rounding
            remaining = self.population - sum(self.stage_populations.values())
            if remaining > 0:
                self.stage_populations["worker"] += remaining
    
    def _initialize_interaction_matrix(self) -> List[List[float]]:
        """
        Initialize the species interaction matrix based on race trait.
        
        Returns:
            A 2D matrix defining how species interact with each other.
            Positive values mean beneficial interactions, negative mean competition.
        """
        # Create default matrix (slight competition between all species)
        matrix = [[-0.01 for _ in range(4)] for _ in range(4)]
        
        # Set diagonal to zero (no self-competition in this model)
        for i in range(4):
            matrix[i][i] = 0.0
            
        # Modify based on race trait
        trait = getattr(self, "trait", "neutral")
        
        if trait == "adaptive":
            # Adaptive races can benefit from others slightly
            for i in range(4):
                if i != self.race_id - 1:  # Not self
                    matrix[self.race_id - 1][i] = 0.02  # Benefit from other species
        
        elif trait == "expansive":
            # Expansive races compete more strongly
            for i in range(4):
                if i != self.race_id - 1:  # Not self
                    matrix[self.race_id - 1][i] = -0.03  # Stronger competition
                    matrix[i][self.race_id - 1] = -0.02  # Others compete back
        
        elif trait == "selective":
            # Selective races have more specialized interactions
            for i in range(4):
                if i != self.race_id - 1:  # Not self
                    # Random positive or negative interaction
                    matrix[self.race_id - 1][i] = random.uniform(-0.05, 0.05)
        
        return matrix
    
    def update_population(self, field, all_populations: List[float] = None) -> None:
        """
        Update population using advanced models.
        
        Args:
            field: The asteroid field for resource analysis
            all_populations: List of population counts for all races (optional)
        """
        # If no populations provided, just use our own
        if all_populations is None:
            all_populations = [0] * 4
            all_populations[self.race_id - 1] = self.population
        
        # 1. Record current population for delayed growth
        self.delayed_growth.record_population(self.population)
        
        # 2. Process multi-species interactions
        new_populations = self.multi_species_model.multi_species_step(all_populations)
        potential_population = new_populations[self.race_id - 1]
        
        # 3. Apply delayed effect from resource acquisition
        if len(self.delayed_growth.history) > self.delayed_growth.delay_steps:
            potential_population = self.delayed_growth.update_population(
                potential_population, 
                resources=sum(self.mineral_consumption.values()) / 10.0  # Use total minerals as resource metric
            )
        
        # 4. Adjust environmental factor based on asteroid field conditions
        env_factor = 1.0
        if hasattr(field, "get_mineral_density_at") and self.position:
            try:
                # Higher mineral density = better environment
                mineral_density = field.get_mineral_density_at(self.position)
                env_factor = 1.0 + (mineral_density / 100.0)
            except Exception:
                # Fallback if method fails
                env_factor = 1.0
        
        # 5. Update life stage distribution
        self.stage_populations = self.stage_population.update_stages(
            self.stage_populations,
            environmental_factor=env_factor
        )
        
        # 6. Calculate total population and apply it
        new_total = sum(self.stage_populations.values())
        self.population = int(new_total)
        
        # 7. Update mining efficiency based on specialized workers
        specialized_ratio = self.stage_populations.get("specialized", 0) / max(1, new_total)
        self.mining_efficiency = 0.5 + (0.3 * specialized_ratio)
        
        # 8. Update hunger based on elder-to-juvenile ratio (higher ratio = more hunger)
        elder_ratio = self.stage_populations.get("elder", 0) / max(1, new_total)
        juvenile_ratio = self.stage_populations.get("juvenile", 0) / max(1, new_total)
        hunger_adjustment = self.hunger_rate * (1 + elder_ratio - juvenile_ratio)
        self.hunger += hunger_adjustment
        self.hunger = min(1.0, max(0.0, self.hunger))
        
        # Call parent method to handle CA-based distribution
        super().update_population(field)
    
    def process_minerals(self, minerals: Dict[str, int]) -> None:
        """
        Process minerals with different efficiencies based on life stages.
        
        Args:
            minerals: Dictionary of mineral types and amounts
        """
        # First track overall consumption as in parent method
        super().process_minerals(minerals)

        return self.process_minerals_by_stage(minerals)
    
    def process_minerals_by_stage(self, minerals: Dict[str, int]) -> Dict[str, int]:
        """
        Process minerals with different efficiencies based on life stages.
        
        Args:
            minerals: Dictionary of mineral types and amounts
            
        Returns:
            Dictionary of consumed minerals by each life stage
        """
        consumption_by_stage = {stage: 0 for stage in self.stage_populations}
        
        # Different life stages process minerals differently
        for mineral_type, amount in minerals.items():
            # Juveniles prefer common minerals
            if mineral_type == "common":
                juvenile_amount = amount * 0.4  # 40% to juveniles
                consumption_by_stage["juvenile"] += juvenile_amount
            
            # Workers are generalists
            worker_amount = amount * 0.3  # 30% to workers
            consumption_by_stage["worker"] += worker_amount
            
            # Specialized miners are efficient with rare and precious
            if mineral_type in ["rare", "precious"]:
                specialized_amount = amount * 0.5  # 50% to specialized
                consumption_by_stage["specialized"] += specialized_amount
            
            # Elders focus on anomalies
            if mineral_type == "anomaly":
                elder_amount = amount * 0.7  # 70% to elders
                consumption_by_stage["elder"] += elder_amount
        
        # Apply consumption effects to each stage
        for stage, consumed in consumption_by_stage.items():
            if stage == "juvenile":
                # Juveniles grow faster with food
                self.stage_population.stage_growth["juvenile"] *= (1 + consumed * 0.01)
            elif stage == "worker":
                # Workers get more efficient
                self.mining_efficiency += consumed * 0.001
            elif stage == "specialized":
                # Specialized gain evolution points
                self.evolution_points += consumed
            elif stage == "elder":
                # Elders reduce hunger with their wisdom
                self.hunger = max(0, self.hunger - consumed * 0.01)
        
        return consumption_by_stage
    
    def get_stage_distribution(self) -> Dict[str, float]:
        """
        Get the proportion of population in each life stage.
        
        Returns:
            Dictionary mapping stage names to their proportion of total population
        """
        return self.stage_population.get_stage_distribution(self.stage_populations)
    
    def get_stage_populations(self) -> Dict[str, int]:
        """
        Get the actual population count in each life stage.
        
        Returns:
            Dictionary mapping stage names to their population counts
        """
        return self.stage_populations
