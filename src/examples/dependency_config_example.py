#!/usr/bin/env python3
"""
Example script demonstrating the dependency configuration system.

This script shows how to use the dependency configuration system to
configure and inject dependencies into classes.
"""

import os
import sys
import logging

# Add the parent directory to the path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.dependency_injection import inject, container
from src.utils.dependency_config import (
    app_container,
    DependencyConfig,
    load_config_from_file,
)
from src.utils.noise_generator import NoiseGenerator


# Example class that uses dependency injection
# First, set the global container to our app_container
# Save the original container
original_container = container
# Replace it with our app_container
import src.utils.dependency_injection

src.utils.dependency_injection.container = app_container


@inject
class NoiseGeneratorDemo:
    """Demo class that uses a noise generator."""

    def __init__(self, noise_generator: NoiseGenerator):
        """
        Initialize the demo with a noise generator.

        Args:
            noise_generator: The noise generator to use (injected)
        """
        self.noise_generator = noise_generator

    def generate_sample(self, width: int = 10, height: int = 10) -> None:
        """
        Generate and print a sample noise grid.

        Args:
            width: Width of the grid
            height: Height of the grid
        """
        print(f"Using noise generator: {self.noise_generator.__class__.__name__}")

        # Generate noise
        noise_grid = self.noise_generator.generate_noise(width, height)

        # Print a sample of the noise grid
        print("\nNoise Sample (5x5):")
        for y in range(min(5, height)):
            row = [f"{noise_grid[y, x]:.2f}" for x in range(min(5, width))]
            print(" ".join(row))


def main():
    """Main function to demonstrate dependency configuration."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Print current configuration
    print("Current Dependency Configuration:")
    for key, value in DependencyConfig.to_dict().items():
        print(f"  {key}: {value}")

    # Create and use a demo instance with default configuration
    print("\n=== Using Default Configuration ===")
    demo1 = NoiseGeneratorDemo()
    demo1.generate_sample()

    # Change configuration
    print("\n=== Changing Configuration ===")
    DependencyConfig.NOISE_GENERATOR_TYPE = "fallback"

    # Load configuration from file
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "dependency_settings.py"
    )
    if os.path.exists(config_path):
        print(f"\n=== Loading Configuration from {config_path} ===")
        load_config_from_file(config_path)

    # Print updated configuration
    print("\nUpdated Dependency Configuration:")
    for key, value in DependencyConfig.to_dict().items():
        print(f"  {key}: {value}")

    # Create and use a demo instance with updated configuration
    print("\n=== Using Updated Configuration ===")
    demo2 = NoiseGeneratorDemo()
    demo2.generate_sample()


if __name__ == "__main__":
    main()
