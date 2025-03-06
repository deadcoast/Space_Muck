#!/usr/bin/env python3
"""
Comprehensive test script to verify imports across the Space Muck codebase.

This test file serves as the primary import verification tool, combining functionality
from multiple import test files. It verifies that imports are working correctly and tests
basic functionality of key components across the codebase.

The test suite includes:
1. Basic import verification for core modules
2. Instantiation tests for key classes
3. Basic functionality tests for critical components
4. Optional dependency detection and reporting
"""

import unittest
import sys

# import os  # Commented out unused import
import importlib

# import traceback  # Commented out unused import
from pathlib import Path

# Removed unused typing imports
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
)  # noqa: F401 - kept for potential future use

import numpy as np

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# These imports are used in individual test methods with dynamic reloading
# to ensure we're using real implementations instead of mocks
# from src.generators.base_generator import BaseGenerator  # noqa: E402, F401
# from src.generators.asteroid_field import AsteroidField  # noqa: E402, F401
# from src.entities.player import Player  # noqa: E402, F401
# from src.config import GRID_WIDTH, GRID_HEIGHT  # noqa: E402, F401


class TestImports(unittest.TestCase):
    """Test class to verify imports are working correctly."""

    def setUp(self):
        """Set up test environment and ensure no mocked modules."""
        self.import_results = {}
        self.optional_dependencies = {
            "scipy": False,
            "cupy": False,
            "torch": False,
            "numba": False,
            "matplotlib": False,
            "perlin_noise": False,
            "pygame": False,
        }

        # Force reload key modules to ensure we're using real implementations
        modules_to_reset = [
            "src.config",
            "src.entities.player",
            "src.generators.base_generator",
            "src.generators.asteroid_field",
        ]

        for module_name in modules_to_reset:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_config_constants(self):
        """Test that config constants can be imported and have valid values."""
        # Ensure we're using the real module, not a mock
        if "src.config" in sys.modules:
            del sys.modules["src.config"]

        # Import directly to get fresh values
        from src.config import GRID_WIDTH as fresh_width, GRID_HEIGHT as fresh_height

        self.assertIsInstance(fresh_width, int, "GRID_WIDTH should be an integer")
        self.assertIsInstance(fresh_height, int, "GRID_HEIGHT should be an integer")
        self.assertGreater(fresh_width, 0, "GRID_WIDTH should be positive")
        self.assertGreater(fresh_height, 0, "GRID_HEIGHT should be positive")
        print(
            f"Config constants verified: GRID_WIDTH={fresh_width}, GRID_HEIGHT={fresh_height}"
        )

    def test_player_import(self):
        """Test that Player class can be imported and instantiated."""
        # Ensure we're using the real Player class, not a mock
        if "src.entities.player" in sys.modules:
            del sys.modules["src.entities.player"]

        from src.entities.player import Player as FreshPlayer

        player = FreshPlayer()
        self.assertIsInstance(player, FreshPlayer)
        print("Player class successfully imported and instantiated")

    def test_asteroid_field_import(self):
        """Test that AsteroidField class can be imported and instantiated."""
        # Ensure we're using the real AsteroidField class, not a mock
        if "src.generators.asteroid_field" in sys.modules:
            del sys.modules["src.generators.asteroid_field"]

        from src.generators.asteroid_field import AsteroidField as FreshAsteroidField

        field = FreshAsteroidField()
        self.assertIsInstance(field, FreshAsteroidField)
        print("AsteroidField class successfully imported and instantiated")

    def test_base_generator(self):
        """Test basic functionality of BaseGenerator."""
        # Ensure we're using the real BaseGenerator class, not a mock
        if "src.generators.base_generator" in sys.modules:
            del sys.modules["src.generators.base_generator"]

        from src.generators.base_generator import BaseGenerator as FreshBaseGenerator

        generator = FreshBaseGenerator(entity_id="test", seed=42, width=50, height=50)
        self.assertIsInstance(generator, FreshBaseGenerator)

        # Test the apply_cellular_automaton method
        grid = np.zeros((50, 50))
        grid[20:30, 20:30] = 1  # Create a square in the middle

        # Test with default parameters
        result = generator.apply_cellular_automaton(grid)
        self._verify_grid_result(result, "Cellular automaton applied, sum: ")

        # Test with custom parameters
        result = generator.apply_cellular_automaton(
            grid, birth_set={2, 3}, survival_set={3, 4, 5}
        )
        self._verify_grid_result(
            result, "Cellular automaton with custom rules applied, sum: "
        )

    def _verify_grid_result(self, result, message):
        """Verify grid result shape and type."""
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (50, 50))
        print(f"{message}{np.sum(result)}")

    def test_optional_dependencies(self):
        """Test for the presence of optional dependencies."""
        for dependency in self.optional_dependencies.keys():
            try:
                importlib.import_module(dependency)
                self.optional_dependencies[dependency] = True
                print(f"✓ Optional dependency '{dependency}' is available")
            except ImportError:
                print(f"✗ Optional dependency '{dependency}' is not available")

        # No assertions here, as these are optional dependencies

    def test_core_modules(self):
        """Test imports for core modules in the codebase."""
        core_modules = [
            "src.config",
            "src.entities.base_entity",
            "src.entities.player",
            "src.generators.base_generator",
            "src.generators.asteroid_field",
            "src.generators.procedural_generator",
            "src.utils.cellular_automaton_utils",
            "src.algorithms.symbiote_algorithm",
            "src.utils.gpu_utils",
        ]

        for module in core_modules:
            # Clear module if it was previously imported to avoid mocks
            if module in sys.modules:
                del sys.modules[module]

            try:
                imported_module = importlib.import_module(module)
                self.assertIsNotNone(imported_module)
                print(f"✓ Core module '{module}' imported successfully")
            except Exception as e:
                self.fail(f"Failed to import core module '{module}': {e}")

    def test_advanced_imports(self):
        """Test imports for more complex modules with dependencies."""
        # Only test these if numpy is available
        if not np:
            return
        advanced_modules = [
            "src.utils.noise_generator",
            "src.utils.pattern_generator",
            "src.utils.value_generator",
        ]

        for module in advanced_modules:
            # Clear module if it was previously imported to avoid mocks
            if module in sys.modules:
                del sys.modules[module]

            try:
                imported_module = importlib.import_module(module)
                self.assertIsNotNone(imported_module)
                print(f"✓ Advanced module '{module}' imported successfully")
            except Exception as e:
                self.fail(f"Failed to import advanced module '{module}': {e}")


def run_tests():
    """Run the import tests and print results."""
    print("\n==== SPACE MUCK IMPORT VERIFICATION TOOL ====\n")
    print(f"Python version: {sys.version}")
    print(f"System path: {sys.path}\n")

    # First check if basic imports work without unittest
    try:
        # Clear the module if it was previously imported to avoid mocks
        if "src.config" in sys.modules:
            del sys.modules["src.config"]

        # Use importlib.util.find_spec to check for availability
        config_spec = importlib.util.find_spec("src.config")
        if config_spec is None:
            raise ImportError("src.config module not found")
        print("✓ Basic imports are working. Running comprehensive tests...\n")
    except ImportError as e:
        display_import_error(e)

    # Run the unittest tests
    unittest.main(verbosity=2)


def display_import_error(e):
    """Display helpful error message for import failures."""
    print(f"✗ Basic imports are failing: {e}")
    print("\nTroubleshooting tips:")
    print("1. Ensure you're running from the project root directory")
    print("2. Check if 'src' directory is in your Python path")
    print("3. Try running with: PYTHONPATH=. python src/tests/test_imports.py")
    sys.exit(1)


if __name__ == "__main__":
    run_tests()
