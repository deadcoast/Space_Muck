"""
Converter management package for Space Muck.

This package provides components for resource conversion processes,
including converters, recipes, and production chains.
"""

# Standard library imports

# Third-party library imports

# Local application imports
from converters.converter_models import (
    ChainStep,
    ConversionProcess,
    Converter,
    ConverterTier,
    ConverterType,
    EfficiencyFactor,
    OptimizationSuggestion,
    ProductionChain,
    Recipe,
    ResourceFlowDirection,
    ResourceFlowQuantity,
    ResourceFlowRate,
    ResourceFlowType,
    ResourceFlowUnit,
    ResourceType,
)
from converters.converter_registry import ConverterRegistry, converter_registry

# Define what symbols are exported when using 'from converters import *'
__all__ = [
    "Converter",
    "Recipe",
    "ConverterType",
    "ConverterTier",
    "ResourceType",
    "ResourceFlowType",
    "ResourceFlowDirection",
    "ResourceFlowUnit",
    "ResourceFlowRate",
    "ResourceFlowQuantity",
    "ConversionProcess",
    "ChainStep",
    "ProductionChain",
    "EfficiencyFactor",
    "OptimizationSuggestion",
    "ConverterRegistry",
    "converter_registry",
]
