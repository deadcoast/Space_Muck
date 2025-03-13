#!/usr/bin/env python3
"""
Dependency configuration settings for Space Muck.

This file contains configuration settings for the dependency injection system.
Edit this file to change how dependencies are configured and provided.

The settings in this file control how optional dependencies are handled and
which implementations are used when multiple options are available.
"""

# Standard library imports

# Third-party library imports

# Local application imports
from typing import Dict, Any, Literal

# Noise generator configuration
NOISE_GENERATOR_TYPE: Literal["perlin", "simplex", "fallback", "auto"] = "perlin"
NOISE_GENERATOR_SINGLETON: bool = True  # Whether to use a singleton noise generator

# Logging configuration
LOGGING_SINGLETON: bool = True  # Whether to use a singleton logger

# GPU acceleration settings
GPU_ENABLED: bool = True  # Whether to use GPU acceleration when available
GPU_PROVIDER: Literal["auto", "cupy", "torch", "none"] = (
    "auto"  # Which GPU library to use
)

# Optional dependency fallback behavior
OPTIONAL_DEPENDENCY_BEHAVIOR: Dict[str, Any] = {
    "scipy": {
        "required": False,  # Whether the dependency is required
        "fallback": "numpy",  # What to use if not available
        "warn": True,  # Whether to warn when falling back
    },
    "cupy": {
        "required": False,
        "fallback": "numpy",
        "warn": True,
    },
    "torch": {
        "required": False,
        "fallback": "numpy",
        "warn": True,
    },
    "numba": {
        "required": False,
        "fallback": "python",
        "warn": True,
    },
    "matplotlib": {
        "required": False,
        "fallback": "none",
        "warn": False,
    },
    "perlin_noise": {
        "required": False,
        "fallback": "custom",
        "warn": True,
    },
}
