"""
Mining System for Space Muck

This module implements the Mining System, which coordinates mining operations
across different miner entities and manages resource extraction, processing,
and interactions between mining races.
"""

import logging
from typing import Dict, List, Tuple, Union, Any

import pygame

from entities.miner_entity import MinerEntity
from entities.enhanced_miner_entity import EnhancedMinerEntity


class MiningSystem:
    """
    Coordinates mining operations across different miner entities and manages
    the interactions between different mining races.
    
    Responsibilities:
    - Track all miner entities in the game
    - Coordinate inter-species population dynamics
    - Manage resource distribution and conflicts
    - Update mining efficiency based on environmental factors
    - Provide visualization tools for mining operations
    """
    
    def __init__(self):
        """Initialize the mining system."""
        # Mapping of race_id to miner entity
        self.miners: Dict[int, Union[MinerEntity, EnhancedMinerEntity]] = {}
        
        # Cached mining territories for visualization
        self.mining_territories: Dict[int, List[Tuple[int, int]]] = {}
        
        # Resource allocation tracking
        self.resource_allocation: Dict[str, Dict[int, float]] = {}
        
        # System-wide mining stats
        self.total_extracted: Dict[str, int] = {
            "common": 0,
            "rare": 0,
            "precious": 0,
            "anomaly": 0
        }
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
    
    def add_miner(self, miner: Union[MinerEntity, EnhancedMinerEntity]) -> None:
        """
        Add a miner entity to the mining system.
        
        Args:
            miner: MinerEntity or EnhancedMinerEntity to add
        """
        if miner.race_id in self.miners:
            self.logger.warning(f"Replacing existing miner with race_id {miner.race_id}")
        
        self.miners[miner.race_id] = miner
        
        # Initialize resource allocation for this race
        for resource_type in self.total_extracted:
            if resource_type not in self.resource_allocation:
                self.resource_allocation[resource_type] = {}
            self.resource_allocation[resource_type][miner.race_id] = 0.0
    
    def remove_miner(self, race_id: int) -> None:
        """
        Remove a miner entity from the mining system.
        
        Args:
            race_id: ID of the miner to remove
        """
        if race_id in self.miners:
            del self.miners[race_id]
            
            # Clean up territory cache
            if race_id in self.mining_territories:
                del self.mining_territories[race_id]
                
            # Clean up resource allocation
            for resource_type in self.resource_allocation:
                if race_id in self.resource_allocation[resource_type]:
                    del self.resource_allocation[resource_type][race_id]
        else:
            self.logger.warning(f"Attempted to remove non-existent miner with race_id {race_id}")
    
    def update(self, asteroid_field: Any) -> None:
        """
        Update all miners and their interactions with the asteroid field.
        
        Args:
            asteroid_field: The asteroid field object containing resources
        """
        # First, collect all population counts for multi-species modeling
        all_populations = self._get_all_populations()
        
        # Update each miner's population and behavior
        for race_id, miner in self.miners.items():
            # For enhanced miners, pass all population data
            if isinstance(miner, EnhancedMinerEntity):
                miner.update_population(asteroid_field, all_populations)
            else:
                miner.update_population(asteroid_field)
        
        # Process mining operations and resource conflicts
        self._process_mining_operations(asteroid_field)
        
        # Update territory cache for visualization
        self._update_territories()
    
    def _get_all_populations(self) -> List[float]:
        """
        Get population counts for all races, ordered by race_id.
        
        Returns:
            List of population values, with index (race_id - 1) for each race
        """
        # Initialize with zeros for all possible race slots (we support up to 4 races)
        populations = [0.0] * 4
        
        # Fill in actual populations
        for race_id, miner in self.miners.items():
            if 1 <= race_id <= 4:  # Ensure race_id is valid
                populations[race_id - 1] = float(miner.population)
        
        return populations
    
    def _process_mining_operations(self, asteroid_field: Any) -> None:
        """
        Process mining operations for all miners, handling resource extraction and conflicts.
        
        Args:
            asteroid_field: The asteroid field object containing resources
        """
        # Step 1: Collect mining targets from all miners
        mining_targets = self._collect_mining_targets()
        
        # Step 2: Resolve mining conflicts where multiple races target the same cell
        cell_claims = self._resolve_mining_claim_conflicts(mining_targets)
        
        # Step 3: Extract resources and distribute based on claims
        self._process_resource_extraction(asteroid_field, cell_claims)
    
    def _collect_mining_targets(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Collect mining targets from all miners.
        
        Returns:
            Dictionary mapping race_id to list of cell coordinates
        """
        mining_targets = {}
        for race_id, miner in self.miners.items():
            # Get cells this miner is targeting for mining
            cells = miner.get_mining_cells()
            mining_targets[race_id] = cells
        
        return mining_targets
    
    def _resolve_mining_claim_conflicts(self, mining_targets: Dict[int, List[Tuple[int, int]]]) -> Dict[Tuple[int, int], List[int]]:
        """
        Resolve mining conflicts where multiple races target the same cell.
        
        Args:
            mining_targets: Dictionary mapping race_id to list of cell coordinates
            
        Returns:
            Dictionary mapping cell coordinates to list of race_ids claiming that cell
        """
        cell_claims = {}  # (x, y) -> [race_ids]
        for race_id, cells in mining_targets.items():
            for cell in cells:
                if cell not in cell_claims:
                    cell_claims[cell] = []
                cell_claims[cell].append(race_id)
        
        return cell_claims
    
    def _process_resource_extraction(self, asteroid_field: Any, cell_claims: Dict[Tuple[int, int], List[int]]) -> None:
        """
        Extract resources and distribute based on claims.
        
        Args:
            asteroid_field: The asteroid field object containing resources
            cell_claims: Dictionary mapping cell coordinates to list of race_ids claiming that cell
        """
        for cell, claiming_races in cell_claims.items():
            # Skip if cell is outside asteroid field
            if not asteroid_field.is_valid_cell(cell[0], cell[1]):
                continue
                
            # Get resources available at this cell
            cell_resources = asteroid_field.extract_resources_at(*cell)
            
            if not cell_resources:
                continue
                
            # If multiple races claim the cell, distribute based on strength
            if len(claiming_races) > 1:
                self._distribute_contested_resources(cell_resources, claiming_races)
            else:
                self._allocate_resources_to_single_race(cell_resources, claiming_races[0])
    
    def _allocate_resources_to_single_race(self, cell_resources: Dict[str, int], race_id: int) -> None:
        """
        Allocate resources to a single race.
        
        Args:
            cell_resources: Dictionary of resource types and amounts
            race_id: ID of the race receiving the resources
        """
        # Single race gets all resources
        miner = self.miners[race_id]
        miner.process_minerals(cell_resources)
        
        # Track allocation
        for resource_type, amount in cell_resources.items():
            self.total_extracted[resource_type] += amount
            self.resource_allocation[resource_type][race_id] += amount
    
    def _distribute_contested_resources(
        self, 
        resources: Dict[str, int], 
        claiming_races: List[int]
    ) -> None:
        """
        Distribute resources among multiple races claiming the same cell.
        
        Args:
            resources: Dictionary of resource types and amounts
            claiming_races: List of race_ids claiming the resources
        """
        # Calculate mining strength for each race
        strengths = {}
        total_strength = 0.0
        
        for race_id in claiming_races:
            miner = self.miners[race_id]
            # Strength is based on population, mining efficiency, and aggression
            strength = miner.population * miner.mining_efficiency * (1 + miner.aggression)
            strengths[race_id] = strength
            total_strength += strength
        
        # Distribute resources proportionally
        for resource_type, amount in resources.items():
            self.total_extracted[resource_type] += amount
            
            # Distribute to each race
            for race_id, strength in strengths.items():
                if total_strength > 0:
                    # Calculate share based on relative strength
                    share = int(amount * (strength / total_strength))
                    
                    # Ensure we don't exceed the total due to rounding
                    share = min(share, amount)
                    
                    # Process the resources for this miner
                    if share > 0:
                        race_resources = {resource_type: share}
                        self.miners[race_id].process_minerals(race_resources)
                        
                        # Track allocation
                        self.resource_allocation[resource_type][race_id] += share
    
    def _update_territories(self) -> None:
        """Update the cached mining territories for visualization."""
        self.mining_territories = {}
        
        for race_id, miner in self.miners.items():
            # Get territory cells claimed by this miner
            if hasattr(miner, "get_territory"):
                try:
                    territory = miner.get_territory()
                    self.mining_territories[race_id] = territory
                except Exception as e:
                    self.logger.error(f"Error getting territory for race {race_id}: {e}")
    
    def get_resource_distribution(self) -> Dict[str, Dict[int, float]]:
        """
        Get the distribution of resources among races.
        
        Returns:
            Dictionary mapping resource types to dictionaries of race_id -> percentage
        """
        result = {}
        
        for resource_type, allocations in self.resource_allocation.items():
            total_allocated = sum(allocations.values())
            result[resource_type] = {}
            
            if total_allocated > 0:
                for race_id, amount in allocations.items():
                    result[resource_type][race_id] = amount / total_allocated
            else:
                for race_id in allocations:
                    result[resource_type][race_id] = 0.0
        
        return result
    
    def get_mining_efficiency_breakdown(self) -> Dict[int, Dict[str, float]]:
        """
        Get a breakdown of mining efficiency factors for each race.
        
        Returns:
            Dictionary mapping race_id to dictionaries of factor -> value
        """
        result = {}
        
        for race_id, miner in self.miners.items():
            result[race_id] = {
                "base_efficiency": miner.mining_efficiency,
                "population": miner.population
            }
            
            # Add enhanced details for EnhancedMinerEntity
            if isinstance(miner, EnhancedMinerEntity):
                stage_distribution = miner.get_stage_distribution()
                result[race_id]["stage_distribution"] = stage_distribution
                
                # Add specialized worker bonus
                specialized_ratio = stage_distribution.get("specialized", 0)
                result[race_id]["specialized_bonus"] = 0.3 * specialized_ratio
        
        return result
    
    def render_mining_territories(self, surface: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0), cell_size: int = 4) -> None:
        """
        Render the mining territories on the given surface.
        
        Args:
            surface: Pygame surface to render on
            camera_offset: (x, y) offset for camera position
            cell_size: Size of each cell in pixels
        """
        for race_id, territory in self.mining_territories.items():
            if race_id not in self.miners:
                continue
                
            miner = self.miners[race_id]
            color = miner.color
            
            # Use translucent color for territory
            territory_color = (*color, 128)  # Add alpha channel
            
            for x, y in territory:
                # Apply camera offset
                screen_x = x * cell_size - camera_offset[0]
                screen_y = y * cell_size - camera_offset[1]
                
                # Draw translucent rectangle
                rect = pygame.Rect(screen_x, screen_y, cell_size, cell_size)
                pygame.draw.rect(surface, territory_color, rect)
    
    def get_miners_by_stage(self) -> Dict[str, int]:
        """
        Get total count of miners in each life stage across all races.
        Only works with EnhancedMinerEntity instances.
        
        Returns:
            Dictionary mapping stage names to total population counts
        """
        result = {"juvenile": 0, "worker": 0, "specialized": 0, "elder": 0}
        
        for miner in self.miners.values():
            if isinstance(miner, EnhancedMinerEntity):
                for stage, count in miner.get_stage_populations().items():
                    result[stage] += count
        
        return result
