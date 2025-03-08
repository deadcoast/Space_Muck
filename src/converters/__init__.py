"""
Converter management package for Space Muck.

This package provides components for resource conversion processes,
including converters, recipes, and production chains.
"""

from converters.converter_models import (
    Converter, 
    Recipe, 
    ConverterType, 
    ConverterTier, 
    ResourceType,
    ConversionProcess,
    ChainStep,
    ProductionChain,
    EfficiencyFactor,
    OptimizationSuggestion
)

from converters.converter_registry import (
    ConverterRegistry,
    converter_registry
)
