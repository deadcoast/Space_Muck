#!/usr/bin/env python3
"""
Unit tests for the dependency configuration system.

This module contains tests for the dependency configuration system,
ensuring that it correctly loads and applies configuration settings.
"""

# Standard library imports
import os
import sys

# Local application imports
from unittest.mock import patch

# Add the parent directory to the path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

# Import the modules after path setup
from utils.dependency_config import (
    DependencyConfig,
    NoiseGenerator,
    app_container,
    configure_dependencies,
    load_config_from_file,
    register_noise_generator,
)

# Third-party library imports


class TestDependencyConfig(unittest.TestCase):
    """Tests for the DependencyConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        # Save original configuration
        self.original_config = {
            "NOISE_GENERATOR_TYPE": DependencyConfig.NOISE_GENERATOR_TYPE,
            "NOISE_GENERATOR_SINGLETON": DependencyConfig.NOISE_GENERATOR_SINGLETON,
            "LOGGING_SINGLETON": DependencyConfig.LOGGING_SINGLETON,
        }

    def tearDown(self):
        """Tear down test fixtures."""
        # Restore original configuration
        DependencyConfig.update_from_dict(self.original_config)
        configure_dependencies()

    def test_update_from_dict(self):
        """Test updating configuration from a dictionary."""
        # Update configuration
        new_config = {
            "NOISE_GENERATOR_TYPE": "simplex",
            "NOISE_GENERATOR_SINGLETON": False,
            "LOGGING_SINGLETON": False,
        }
        DependencyConfig.update_from_dict(new_config)

        # Check that configuration was updated
        self.assertEqual(DependencyConfig.NOISE_GENERATOR_TYPE, "simplex")
        self.assertFalse(DependencyConfig.NOISE_GENERATOR_SINGLETON)
        self.assertFalse(DependencyConfig.LOGGING_SINGLETON)

    def test_to_dict(self):
        """Test converting configuration to a dictionary."""
        # Set known values
        DependencyConfig.NOISE_GENERATOR_TYPE = "perlin"
        DependencyConfig.NOISE_GENERATOR_SINGLETON = True
        DependencyConfig.LOGGING_SINGLETON = True

        # Get dictionary
        config_dict = DependencyConfig.to_dict()

        # Check that dictionary contains expected values
        self.assertEqual(config_dict["NOISE_GENERATOR_TYPE"], "perlin")
        self.assertTrue(config_dict["NOISE_GENERATOR_SINGLETON"])
        self.assertTrue(config_dict["LOGGING_SINGLETON"])

    def test_unknown_config_option(self):
        """Test handling of unknown configuration options."""
        # Update with unknown option
        with self.assertLogs(level="WARNING") as cm:
            DependencyConfig.update_from_dict({"UNKNOWN_OPTION": "value"})

            # Check that warning was logged
            self.assertIn(
                "WARNING:root:Unknown configuration option: UNKNOWN_OPTION", cm.output
            )


class TestNoiseGeneratorRegistration(unittest.TestCase):
    """Tests for noise generator registration."""

    def setUp(self):
        """Set up test fixtures."""
        # Save original configuration
        self.original_config = {
            "NOISE_GENERATOR_TYPE": DependencyConfig.NOISE_GENERATOR_TYPE,
            "NOISE_GENERATOR_SINGLETON": DependencyConfig.NOISE_GENERATOR_SINGLETON,
        }

        # Clear existing registrations
        if NoiseGenerator in app_container._registry:
            del app_container._registry[NoiseGenerator]
        if NoiseGenerator in app_container._instance_cache:
            del app_container._instance_cache[NoiseGenerator]

    def tearDown(self):
        """Tear down test fixtures."""
        # Restore original configuration
        DependencyConfig.update_from_dict(self.original_config)
        configure_dependencies()

    @patch("src.utils.dependency_config.PerlinNoiseGenerator")
    def test_register_perlin_noise_generator(self, mock_perlin):
        """Test registering a PerlinNoiseGenerator."""
        # Configure for PerlinNoiseGenerator
        DependencyConfig.NOISE_GENERATOR_TYPE = "perlin"
        register_noise_generator()

        # Resolve the noise generator
        app_container.resolve(NoiseGenerator)

        # Check that PerlinNoiseGenerator was created
        mock_perlin.assert_called_once()

    @patch("src.utils.dependency_config.PerlinNoiseGenerator", side_effect=ImportError)
    @patch("src.utils.dependency_config.SimplexNoiseGenerator")
    def test_register_perlin_fallback(self, mock_simplex, mock_perlin):
        """Test fallback when PerlinNoiseGenerator is not available."""
        # Configure for PerlinNoiseGenerator
        DependencyConfig.NOISE_GENERATOR_TYPE = "perlin"
        register_noise_generator()

        # Resolve the noise generator
        app_container.resolve(NoiseGenerator)

        # Check that PerlinNoiseGenerator was attempted
        mock_perlin.assert_called_once()

        # Check that SimplexNoiseGenerator was created as fallback
        mock_simplex.assert_called_once()

    @patch("src.utils.dependency_config.SimplexNoiseGenerator")
    def test_register_simplex_noise_generator(self, mock_simplex):
        """Test registering a SimplexNoiseGenerator."""
        # Configure for SimplexNoiseGenerator
        DependencyConfig.NOISE_GENERATOR_TYPE = "simplex"
        register_noise_generator()

        # Resolve the noise generator
        app_container.resolve(NoiseGenerator)

        # Check that SimplexNoiseGenerator was created
        mock_simplex.assert_called_once()

    @patch("src.utils.dependency_config.get_noise_generator")
    def test_register_auto_noise_generator(self, mock_get_noise):
        """Test registering an auto-selected noise generator."""
        # Configure for auto selection
        DependencyConfig.NOISE_GENERATOR_TYPE = "auto"
        register_noise_generator()

        # Resolve the noise generator
        app_container.resolve(NoiseGenerator)

        # Check that get_noise_generator was called
        mock_get_noise.assert_called_once()


class TestConfigFileLoading(unittest.TestCase):
    """Tests for loading configuration from files."""

    def setUp(self):
        """Set up test fixtures."""
        # Save original configuration
        self.original_config = {
            "NOISE_GENERATOR_TYPE": DependencyConfig.NOISE_GENERATOR_TYPE,
            "NOISE_GENERATOR_SINGLETON": DependencyConfig.NOISE_GENERATOR_SINGLETON,
            "LOGGING_SINGLETON": DependencyConfig.LOGGING_SINGLETON,
        }

        # Create a temporary config file
        self.temp_file = "temp_config.py"
        with open(self.temp_file, "w", encoding="utf-8") as f:
            f.write('NOISE_GENERATOR_TYPE = "simplex"\n')
            f.write("NOISE_GENERATOR_SINGLETON = False\n")

    def tearDown(self):
        """Tear down test fixtures."""
        # Restore original configuration
        DependencyConfig.update_from_dict(self.original_config)
        configure_dependencies()

        # Remove temporary file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_load_config_from_file(self):
        """Test loading configuration from a file."""
        # Load configuration from file
        load_config_from_file(self.temp_file)

        # Check that configuration was updated
        self.assertEqual(DependencyConfig.NOISE_GENERATOR_TYPE, "simplex")
        self.assertFalse(DependencyConfig.NOISE_GENERATOR_SINGLETON)

    def test_load_config_from_nonexistent_file(self):
        """Test handling of nonexistent configuration files."""
        # Try to load from nonexistent file
        with self.assertLogs(level="ERROR") as cm:
            load_config_from_file("nonexistent_file.py")

            # Check that error was logged
            self.assertTrue(
                any(
                    "ERROR:root:Error loading configuration from nonexistent_file.py"
                    in msg
                    for msg in cm.output
                )
            )


if __name__ == "__main__":
    unittest.main()
