"""
Converter and recipe data models for the resource conversion system.

This module defines the core data structures used by the converter management system
to represent different types of converters, recipes, and resource chains.
"""

from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import uuid


class ConverterType(Enum):
    """Types of resource converters available in the game."""
    SMELTER = "smelter"
    REFINERY = "refinery"
    ASSEMBLER = "assembler"
    FABRICATOR = "fabricator"
    SYNTHESIZER = "synthesizer"


class ConverterTier(Enum):
    """Converter quality/tier levels that affect performance."""
    BASIC = 1
    IMPROVED = 2
    ADVANCED = 3
    SUPERIOR = 4
    ULTIMATE = 5


class ResourceType(Enum):
    """Types of resources available in the game."""
    # Raw materials
    IRON_ORE = "iron_ore"
    COPPER_ORE = "copper_ore"
    TITANIUM_ORE = "titanium_ore"
    SILICON = "silicon"
    RARE_EARTH = "rare_earth"
    PRECIOUS_METAL = "precious_metal"
    ORGANIC_COMPOUND = "organic_compound"
    
    # Processed materials
    IRON_INGOT = "iron_ingot"
    COPPER_INGOT = "copper_ingot"
    TITANIUM_ALLOY = "titanium_alloy"
    SILICON_WAFER = "silicon_wafer"
    RARE_EARTH_ALLOY = "rare_earth_alloy"
    PRECIOUS_ALLOY = "precious_alloy"
    SYNTHETIC_COMPOUND = "synthetic_compound"
    
    # Components
    IRON_PLATE = "iron_plate"
    CIRCUIT_BOARD = "circuit_board"
    STRUCTURAL_ELEMENT = "structural_element"
    POWER_CELL = "power_cell"
    ELECTRONIC_COMPONENT = "electronic_component"
    NANOMATERIAL = "nanomaterial"
    
    # Advanced components
    ADVANCED_CIRCUIT = "advanced_circuit"
    FUSION_CORE = "fusion_core"
    QUANTUM_PROCESSOR = "quantum_processor"
    ANTIMATTER_CATALYST = "antimatter_catalyst"


class Recipe:
    """Represents a conversion recipe that transforms input resources into outputs."""
    
    def __init__(
        self,
        id: str,
        name: str,
        inputs: Dict[ResourceType, int],
        outputs: Dict[ResourceType, int],
        processing_time: float,
        required_converter_type: ConverterType,
        min_converter_tier: ConverterTier = ConverterTier.BASIC,
        energy_cost: float = 1.0,
        description: str = ""
    ):
        """
        Initialize a conversion recipe.
        
        Args:
            id: Unique identifier for the recipe
            name: User-friendly name
            inputs: Dictionary mapping resource types to quantities required
            outputs: Dictionary mapping resource types to quantities produced
            processing_time: Base time to complete the conversion in seconds
            required_converter_type: Type of converter needed for this recipe
            min_converter_tier: Minimum converter tier required
            energy_cost: Energy cost per second of processing
            description: Optional description of the recipe
        """
        self.id = id
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.processing_time = processing_time
        self.required_converter_type = required_converter_type
        self.min_converter_tier = min_converter_tier
        self.energy_cost = energy_cost
        self.description = description
        
    def is_compatible_with(self, converter: 'Converter') -> bool:
        """Check if this recipe can be processed by the given converter."""
        return (
            converter.type == self.required_converter_type and
            converter.tier.value >= self.min_converter_tier.value
        )
        
    def get_time_with_efficiency(self, efficiency: float) -> float:
        """Calculate actual processing time based on efficiency."""
        # Lower is better, so divide by efficiency
        return self.processing_time / max(0.1, efficiency)
        
    def get_energy_with_efficiency(self, efficiency: float) -> float:
        """Calculate actual energy cost based on efficiency."""
        # Lower is better, so divide by efficiency
        return self.energy_cost / max(0.1, efficiency)


class Converter:
    """Represents a resource converter that can process recipes."""
    
    def __init__(
        self,
        id: str,
        name: str,
        type: ConverterType,
        tier: ConverterTier = ConverterTier.BASIC,
        base_efficiency: float = 1.0,
        max_queue_size: int = 3,
        energy_capacity: float = 100.0,
        description: str = ""
    ):
        """
        Initialize a converter.
        
        Args:
            id: Unique identifier for the converter
            name: User-friendly name
            type: Type of converter (determines compatible recipes)
            tier: Quality/performance tier
            base_efficiency: Base efficiency factor (higher is better)
            max_queue_size: Maximum recipes that can be queued
            energy_capacity: Maximum energy storage
            description: Optional description of the converter
        """
        self.id = id
        self.name = name
        self.type = type
        self.tier = tier
        self.base_efficiency = base_efficiency
        self.max_queue_size = max_queue_size
        self.energy_capacity = energy_capacity
        self.description = description
        self.current_energy = energy_capacity
        self.active_processes: List[ConversionProcess] = []
        
    def get_overall_efficiency(self, quality_modifier: float = 1.0, technology_modifier: float = 1.0, environmental_modifier: float = 1.0) -> float:
        """Calculate the overall efficiency of this converter."""
        base = self.base_efficiency * (0.8 + 0.1 * self.tier.value)  # Tier bonus
        return base * quality_modifier * technology_modifier * environmental_modifier
        
    def can_process(self, recipe: Recipe) -> bool:
        """Check if this converter can process the given recipe."""
        return (
            recipe.is_compatible_with(self) and
            len(self.active_processes) < self.max_queue_size
        )
        
    def start_process(self, recipe: Recipe, input_resources: Dict[ResourceType, int], quality_modifier: float = 1.0) -> Optional['ConversionProcess']:
        """
        Start a new conversion process if possible.
        
        Args:
            recipe: Recipe to process
            input_resources: Available resources to use (not consumed here, just checked)
            quality_modifier: Quality modifier for the process
            
        Returns:
            ConversionProcess or None if failed
        """
        # Check compatibility
        if not self.can_process(recipe):
            return None
            
        # Check input resources
        for resource_type, amount in recipe.inputs.items():
            if resource_type not in input_resources or input_resources[resource_type] < amount:
                return None
                
        # Create and add the process
        process = ConversionProcess(
            id=str(uuid.uuid4()),
            recipe=recipe,
            converter_id=self.id,
            quality_modifier=quality_modifier
        )
        self.active_processes.append(process)
        return process
        
    def cancel_process(self, process_id: str) -> bool:
        """Cancel a process by its ID."""
        for i, process in enumerate(self.active_processes):
            if process.id == process_id:
                self.active_processes.pop(i)
                return True
        return False
        
    def update(self, delta_time: float, technology_modifier: float = 1.0, environmental_modifier: float = 1.0) -> List['ConversionProcess']:
        """
        Update all active processes and return completed ones.
        
        Args:
            delta_time: Time elapsed since last update in seconds
            technology_modifier: Technology efficiency modifier
            environmental_modifier: Environmental efficiency modifier
            
        Returns:
            List of completed conversion processes
        """
        completed_processes = []
        
        remaining_processes = []
        for process in self.active_processes:
            # Calculate efficiency for this process
            efficiency = self.get_overall_efficiency(
                process.quality_modifier,
                technology_modifier,
                environmental_modifier
            )
            
            # Check if we have enough energy
            energy_needed = process.recipe.get_energy_with_efficiency(efficiency) * delta_time
            if self.current_energy >= energy_needed:
                # Consume energy
                self.current_energy -= energy_needed
                
                # Update progress
                process.progress += delta_time / process.recipe.get_time_with_efficiency(efficiency)
                
                # Check if complete
                if process.progress >= 1.0:
                    process.complete()
                    completed_processes.append(process)
                else:
                    remaining_processes.append(process)
            else:
                # Not enough energy, process stalls
                process.stalled = True
                remaining_processes.append(process)
                
        self.active_processes = remaining_processes
        return completed_processes


class ConversionProcess:
    """Represents an active conversion process."""
    
    def __init__(
        self,
        id: str,
        recipe: Recipe,
        converter_id: str,
        quality_modifier: float = 1.0
    ):
        """
        Initialize a conversion process.
        
        Args:
            id: Unique identifier for this process
            recipe: Recipe being processed
            converter_id: ID of the converter running this process
            quality_modifier: Quality modifier affecting efficiency
        """
        self.id = id
        self.recipe = recipe
        self.converter_id = converter_id
        self.quality_modifier = quality_modifier
        self.progress = 0.0  # 0.0 to 1.0
        self.stalled = False
        self.completed = False
        self.start_time = 0.0  # Will be set when added to converter
        self.completion_time = 0.0  # Will be set when completed
        
    def complete(self) -> None:
        """Mark the process as completed."""
        self.progress = 1.0
        self.completed = True
        self.stalled = False


class ChainStep:
    """Represents a step in a production chain."""
    
    def __init__(
        self,
        converter_id: str,
        recipe_id: str,
        step_order: int
    ):
        """
        Initialize a chain step.
        
        Args:
            converter_id: ID of the converter to use
            recipe_id: ID of the recipe to process
            step_order: Order in the production chain
        """
        self.converter_id = converter_id
        self.recipe_id = recipe_id
        self.step_order = step_order


class ProductionChain:
    """Represents a sequence of conversion processes forming a production line."""
    
    def __init__(
        self,
        id: str,
        name: str,
        steps: List[ChainStep],
        description: str = ""
    ):
        """
        Initialize a production chain.
        
        Args:
            id: Unique identifier for the chain
            name: User-friendly name
            steps: Ordered list of chain steps
            description: Optional description
        """
        self.id = id
        self.name = name
        self.steps = sorted(steps, key=lambda s: s.step_order)
        self.description = description
        self.active = False
        self.paused = False
        self.current_step_index = 0


# Common efficiency factors for converter operations
class EfficiencyFactor:
    """Represents a factor affecting converter efficiency."""
    
    def __init__(
        self,
        name: str,
        value: float,
        description: str = ""
    ):
        """
        Initialize an efficiency factor.
        
        Args:
            name: Factor name
            value: Multiplier value (1.0 is neutral)
            description: Optional description
        """
        self.name = name
        self.value = value
        self.description = description


class OptimizationSuggestion:
    """Represents a suggestion to improve converter efficiency."""
    
    def __init__(
        self,
        description: str,
        potential_gain: float,
        cost: Optional[int] = None,
        prerequisite: Optional[str] = None
    ):
        """
        Initialize an optimization suggestion.
        
        Args:
            description: Description of the suggested improvement
            potential_gain: Potential efficiency gain
            cost: Optional cost to implement
            prerequisite: Optional prerequisite needed
        """
        self.description = description
        self.potential_gain = potential_gain
        self.cost = cost
        self.prerequisite = prerequisite
