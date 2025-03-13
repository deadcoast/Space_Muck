#!/usr/bin/env python3
"""
Dependency Configuration System for Space Muck.

This module provides a centralized configuration system for managing
dependencies across the codebase. It integrates with the dependency
injection framework to register and configure dependencies.
"""

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from typing import Dict, Any

# Try relative imports first, then fall back to absolute imports
try:
    from .dependency_injection import DependencyContainer
    from .noise_generator import (
        NoiseGenerator,
        PerlinNoiseGenerator,
        SimplexNoiseGenerator,
        get_noise_generator,
    )
except ImportError:
    # Fall back to absolute imports when running as a script
    from utils.dependency_injection import DependencyContainer
    from utils.noise_generator import (
        NoiseGenerator,
        PerlinNoiseGenerator,
        SimplexNoiseGenerator,
        get_noise_generator,
    )

# Create a global container for the application
app_container = DependencyContainer()


# Configuration options
class DependencyConfig:
    """Configuration options for the dependency injection system."""

    # Noise generator configuration
    NOISE_GENERATOR_TYPE: str = (
        "auto"  # Options: "perlin", "simplex", "fallback", "auto"
    )
    NOISE_GENERATOR_SINGLETON: bool = True  # Whether to use a singleton noise generator

    # Logging configuration
    LOGGING_SINGLETON: bool = True  # Whether to use a singleton logger

    # Add more configuration options as needed

    @classmethod
    def update_from_dict(cls, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration values
        """
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                logging.warning(f"Unknown configuration option: {key}")

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of configuration values
        """
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and key.isupper()
        }


def configure_dependencies() -> None:
    """Configure and register all dependencies based on the current configuration."""

    # Register noise generator based on configuration
    register_noise_generator()

    # Register other dependencies as needed

    logging.info("Dependencies configured successfully")


def register_noise_generator() -> None:
    """Register the appropriate noise generator based on configuration."""

    # First, unregister any existing provider
    if NoiseGenerator in app_container._registry:
        del app_container._registry[NoiseGenerator]
        if NoiseGenerator in app_container._instance_cache:
            del app_container._instance_cache[NoiseGenerator]

    # Define the provider function
    def provide_noise_generator() -> NoiseGenerator:
        """Provide the configured noise generator."""
        noise_type = DependencyConfig.NOISE_GENERATOR_TYPE.lower()

        if noise_type == "perlin":
            try:
                return PerlinNoiseGenerator()
            except ImportError:
                logging.warning(
                    "PerlinNoiseGenerator not available, falling back to SimplexNoiseGenerator"
                )
                return SimplexNoiseGenerator()
        elif noise_type == "simplex":
            try:
                return SimplexNoiseGenerator()
            except ImportError:
                logging.warning(
                    "SimplexNoiseGenerator not available, falling back to default noise generator"
                )
                return get_noise_generator()
        elif noise_type == "fallback":
            return SimplexNoiseGenerator()  # Using SimplexNoiseGenerator as fallback
        else:  # "auto" or any other value
            return get_noise_generator()

    # Register the provider with the container
    app_container.register(
        NoiseGenerator,
        provide_noise_generator,
        DependencyConfig.NOISE_GENERATOR_SINGLETON,
    )


def load_config_from_file(file_path: str) -> None:
    """
    Load configuration from a file.

    Args:
        file_path: Path to the configuration file
    """
    try:
        # Simple implementation using exec to load Python file
        # In a production environment, consider using a more secure method
        config_dict = {}
        with open(file_path, "r") as f:
            exec(f.read(), {}, config_dict)

        # Update configuration
        DependencyConfig.update_from_dict(config_dict)

        # Reconfigure dependencies
        configure_dependencies()

        logging.info(f"Configuration loaded from {file_path}")
    except Exception as e:
        logging.error(f"Error loading configuration from {file_path}: {e}")


# Initialize dependencies with default configuration
configure_dependencies()
