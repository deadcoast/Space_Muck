"""
Converter and recipe registry for the resource conversion system.

This module manages the available converters, recipes and provides access to them
throughout the game.
"""

from typing import Dict, List, Optional, Tuple, Set
import uuid

from converters.converter_models import (
    Converter,
    Recipe,
    ConverterType,
    ConverterTier,
    ResourceType,
    ProductionChain,
    ChainStep,
    OptimizationSuggestion,
)


class ConverterRegistry:
    """
    Registry for all converters and recipes in the game.

    This class acts as a centralized store and manager for converters, recipes, and
    production chain templates.
    """

    def __init__(self):
        """Initialize the converter registry."""
        self.converters: Dict[str, Converter] = {}
        self.recipes: Dict[str, Recipe] = {}
        self.chain_templates: Dict[str, ProductionChain] = {}
        self.active_chains: Dict[str, ProductionChain] = {}

        # Initialize with default content
        self._initialize_default_content()

    def _initialize_default_content(self):
        """Initialize the registry with default converters and recipes."""
        # Create default converters
        self._create_default_converters()

        # Create default recipes
        self._create_default_recipes()

        # Create default chain templates
        self._create_default_chain_templates()

    def _create_default_converters(self):
        """Create and register default converters."""
        converters = [
            Converter(
                id="smelter_basic",
                name="Basic Smelter",
                type=ConverterType.SMELTER,
                tier=ConverterTier.BASIC,
                base_efficiency=1.0,
                description="Basic ore smelting furnace for processing raw ores.",
            ),
            Converter(
                id="refinery_basic",
                name="Basic Refinery",
                type=ConverterType.REFINERY,
                tier=ConverterTier.BASIC,
                base_efficiency=1.0,
                description="Basic refinery for purifying and processing resources.",
            ),
            Converter(
                id="assembler_basic",
                name="Basic Assembler",
                type=ConverterType.ASSEMBLER,
                tier=ConverterTier.BASIC,
                base_efficiency=1.0,
                description="Basic assembler for creating simple components.",
            ),
            Converter(
                id="fabricator_basic",
                name="Basic Fabricator",
                type=ConverterType.FABRICATOR,
                tier=ConverterTier.BASIC,
                base_efficiency=1.0,
                description="Basic fabricator for creating advanced components.",
            ),
            Converter(
                id="synthesizer_basic",
                name="Basic Synthesizer",
                type=ConverterType.SYNTHESIZER,
                tier=ConverterTier.BASIC,
                base_efficiency=1.0,
                description="Basic synthesizer for creating synthetic materials.",
            ),
            # Tier 2 converters
            Converter(
                id="smelter_improved",
                name="Improved Smelter",
                type=ConverterType.SMELTER,
                tier=ConverterTier.IMPROVED,
                base_efficiency=1.2,
                max_queue_size=4,
                description="Improved smelter with higher efficiency and queue capacity.",
            ),
            Converter(
                id="assembler_improved",
                name="Improved Assembler",
                type=ConverterType.ASSEMBLER,
                tier=ConverterTier.IMPROVED,
                base_efficiency=1.2,
                max_queue_size=4,
                description="Improved assembler with higher efficiency and queue capacity.",
            ),
        ]

        for converter in converters:
            self.register_converter(converter)

    def _create_default_recipes(self):
        """Create and register default recipes."""
        recipes = [
            # Basic metal processing
            Recipe(
                id="iron_ore_to_ingot",
                name="Smelt Iron Ore",
                inputs={ResourceType.IRON_ORE: 2},
                outputs={ResourceType.IRON_INGOT: 1},
                processing_time=5.0,
                required_converter_type=ConverterType.SMELTER,
                energy_cost=2.0,
                description="Smelt iron ore into iron ingots.",
            ),
            Recipe(
                id="copper_ore_to_ingot",
                name="Smelt Copper Ore",
                inputs={ResourceType.COPPER_ORE: 2},
                outputs={ResourceType.COPPER_INGOT: 1},
                processing_time=4.0,
                required_converter_type=ConverterType.SMELTER,
                energy_cost=1.8,
                description="Smelt copper ore into copper ingots.",
            ),
            # Component crafting
            Recipe(
                id="iron_ingot_to_plate",
                name="Craft Iron Plates",
                inputs={ResourceType.IRON_INGOT: 1},
                outputs={ResourceType.IRON_PLATE: 2},
                processing_time=3.0,
                required_converter_type=ConverterType.ASSEMBLER,
                energy_cost=1.5,
                description="Press iron ingots into iron plates.",
            ),
            Recipe(
                id="circuit_board",
                name="Craft Circuit Boards",
                inputs={ResourceType.COPPER_INGOT: 1, ResourceType.SILICON_WAFER: 1},
                outputs={ResourceType.CIRCUIT_BOARD: 1},
                processing_time=8.0,
                required_converter_type=ConverterType.ASSEMBLER,
                energy_cost=3.0,
                description="Assemble circuit boards from copper and silicon.",
            ),
            # Advanced components
            Recipe(
                id="advanced_circuit",
                name="Advanced Circuit",
                inputs={
                    ResourceType.CIRCUIT_BOARD: 2,
                    ResourceType.RARE_EARTH_ALLOY: 1,
                    ResourceType.ELECTRONIC_COMPONENT: 3,
                },
                outputs={ResourceType.ADVANCED_CIRCUIT: 1},
                processing_time=15.0,
                required_converter_type=ConverterType.FABRICATOR,
                min_converter_tier=ConverterTier.IMPROVED,
                energy_cost=5.0,
                description="Fabricate advanced circuits with enhanced capabilities.",
            ),
            # Special recipes
            Recipe(
                id="synthetic_compound",
                name="Synthetic Compound",
                inputs={ResourceType.ORGANIC_COMPOUND: 2, ResourceType.RARE_EARTH: 1},
                outputs={ResourceType.SYNTHETIC_COMPOUND: 1},
                processing_time=10.0,
                required_converter_type=ConverterType.SYNTHESIZER,
                energy_cost=4.0,
                description="Synthesize organic compounds into advanced synthetic materials.",
            ),
        ]

        for recipe in recipes:
            self.register_recipe(recipe)

    def _create_default_chain_templates(self):
        """Create and register default production chain templates."""
        # Basic Electronics chain
        electronics_chain = ProductionChain(
            id="basic_electronics",
            name="Basic Electronics",
            steps=[
                ChainStep(
                    converter_id="smelter_basic",
                    recipe_id="copper_ore_to_ingot",
                    step_order=0,
                ),
                ChainStep(
                    converter_id="assembler_basic",
                    recipe_id="circuit_board",
                    step_order=1,
                ),
            ],
            description="A simple production chain for basic electronic components.",
        )

        # Basic Construction Materials chain
        construction_chain = ProductionChain(
            id="basic_construction",
            name="Basic Construction Materials",
            steps=[
                ChainStep(
                    converter_id="smelter_basic",
                    recipe_id="iron_ore_to_ingot",
                    step_order=0,
                ),
                ChainStep(
                    converter_id="assembler_basic",
                    recipe_id="iron_ingot_to_plate",
                    step_order=1,
                ),
            ],
            description="A simple production chain for basic construction materials.",
        )

        self.register_chain_template(electronics_chain)
        self.register_chain_template(construction_chain)

    def register_converter(self, converter: Converter) -> None:
        """
        Register a converter in the registry.

        Args:
            converter: The converter to register
        """
        self.converters[converter.id] = converter

    def register_recipe(self, recipe: Recipe) -> None:
        """
        Register a recipe in the registry.

        Args:
            recipe: The recipe to register
        """
        self.recipes[recipe.id] = recipe

    def register_chain_template(self, chain: ProductionChain) -> None:
        """
        Register a production chain template.

        Args:
            chain: The chain template to register
        """
        self.chain_templates[chain.id] = chain

    def get_converter(self, converter_id: str) -> Optional[Converter]:
        """
        Get a converter by ID.

        Args:
            converter_id: The converter ID to look for

        Returns:
            The converter if found, None otherwise
        """
        return self.converters.get(converter_id)

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """
        Get a recipe by ID.

        Args:
            recipe_id: The recipe ID to look for

        Returns:
            The recipe if found, None otherwise
        """
        return self.recipes.get(recipe_id)

    def get_chain_template(self, template_id: str) -> Optional[ProductionChain]:
        """
        Get a chain template by ID.

        Args:
            template_id: The template ID to look for

        Returns:
            The chain template if found, None otherwise
        """
        return self.chain_templates.get(template_id)

    def get_compatible_recipes(self, converter: Converter) -> List[Recipe]:
        """
        Get all recipes compatible with a specific converter.

        Args:
            converter: The converter to check compatibility with

        Returns:
            List of compatible recipes
        """
        return [
            recipe
            for recipe in self.recipes.values()
            if recipe.is_compatible_with(converter)
        ]

    def get_converters_by_type(self, converter_type: ConverterType) -> List[Converter]:
        """
        Get all converters of a specific type.

        Args:
            converter_type: Type of converters to return

        Returns:
            List of converters of the specified type
        """
        return [
            converter
            for converter in self.converters.values()
            if converter.type == converter_type
        ]

    def get_converters_for_recipe(self, recipe: Recipe) -> List[Converter]:
        """
        Get all converters that can process a specific recipe.

        Args:
            recipe: The recipe to check

        Returns:
            List of compatible converters
        """
        return [
            converter
            for converter in self.converters.values()
            if recipe.is_compatible_with(converter)
        ]

    def start_chain(self, template_id: str) -> Optional[str]:
        """
        Start a new production chain from a template.

        Args:
            template_id: ID of the template to use

        Returns:
            ID of the new chain if successful, None otherwise
        """
        template = self.get_chain_template(template_id)
        if not template:
            return None

        # Create a new chain instance
        chain_id = str(uuid.uuid4())
        new_chain = ProductionChain(
            id=chain_id,
            name=template.name,
            steps=template.steps.copy(),
            description=template.description,
        )

        new_chain.active = True
        self.active_chains[chain_id] = new_chain
        return chain_id

    def get_optimization_suggestions(
        self, converter: Converter
    ) -> List[OptimizationSuggestion]:
        """
        Get optimization suggestions for a converter.

        Args:
            converter: The converter to get suggestions for

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Add tier upgrade suggestion if not at maximum
        if converter.tier.value < ConverterTier.ULTIMATE.value:
            next_tier = ConverterTier(converter.tier.value + 1)
            suggestions.append(
                OptimizationSuggestion(
                    description=f"Upgrade converter to Tier {next_tier.value}",
                    potential_gain=0.1,  # 10% base efficiency gain
                    cost=1000 * next_tier.value,  # Higher tiers cost more
                )
            )

        # Add other suggestions based on converter type
        if converter.type == ConverterType.SMELTER:
            suggestions.append(
                OptimizationSuggestion(
                    description="Research 'Advanced Metallurgy'",
                    potential_gain=0.15,
                    prerequisite="Research points",
                )
            )
        elif converter.type == ConverterType.ASSEMBLER:
            suggestions.append(
                OptimizationSuggestion(
                    description="Install 'Precision Assembly Tools'",
                    potential_gain=0.12,
                    cost=800,
                )
            )

        suggestions.extend(
            (
                OptimizationSuggestion(
                    description="Use higher quality resources", potential_gain=0.08
                ),
                OptimizationSuggestion(
                    description="Address nearby hazards", potential_gain=0.05
                ),
            )
        )
        return suggestions


# Global instance for easy access
converter_registry = ConverterRegistry()
