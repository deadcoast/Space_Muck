#!/usr/bin/env python3
"""
Dependency configuration settings for Space Muck.

This file contains configuration settings for the dependency injection system.
Edit this file to change how dependencies are configured and provided.
"""

# Noise generator configuration
NOISE_GENERATOR_TYPE = "perlin"  # Options: "perlin", "simplex", "fallback", "auto"
NOISE_GENERATOR_SINGLETON = True  # Whether to use a singleton noise generator

# Logging configuration
LOGGING_SINGLETON = True  # Whether to use a singleton logger

# Add more configuration options as needed
