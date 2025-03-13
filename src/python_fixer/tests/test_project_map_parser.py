#!/usr/bin/env python3
# Standard library imports
import unittest

# Third-party library imports

# Local application imports
from project_map_parser import ProjectMapParser

class TestProjectMapParser(unittest.TestCase):
    """Test suite for the ProjectMapParser class.

    Following Space Muck testing best practices:
    - No loops in test methods (using helper methods when needed)
    - Clear test method names indicating what's being tested
    - Proper type hints and assertions
    - Comprehensive test coverage
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures used by all test methods."""
        cls.test_map = """\
Project Structure:
    /space_muck
    ├── main.py
    ├── generators/
    │   ├── asteroid_generator.py
    │   ├── asteroid_field.py
    │   └── symbiote_evolution_generator.py
    ├── systems/
    │   ├── fleet_manager.py
    │   ├── trading_system.py
    │   └── exploration_manager.py
    └── ui/
        ├── ascii_ui.py
        └── renderers.py

Enhancement Targets:
    1. asteroid_generator.py: Add resource distribution patterns
    2. fleet_manager.py: Implement advanced formation logic
    3. trading_system.py: Add dynamic pricing model
    4. ascii_ui.py: Improve rendering performance

Dependencies Found:
    Primary:
    - numpy
    - networkx
    - pygame
    - scipy

    Secondary:
    - logging
    - pathlib
    - typing
    - random
"""
        cls.parser = ProjectMapParser(cls.test_map)

    def test_structure_parsing(self) -> None:
        """Test that project structure is correctly parsed."""
        expected_paths = [
            "main.py",
            "generators/asteroid_generator.py",
            "generators/asteroid_field.py",
            "generators/symbiote_evolution_generator.py",
            "systems/fleet_manager.py",
            "systems/trading_system.py",
            "systems/exploration_manager.py",
            "ui/ascii_ui.py",
            "ui/renderers.py",
        ]
        actual_paths = self.parser.get_module_paths()
        self.assertEqual(sorted(expected_paths), sorted(actual_paths))

    def test_enhancement_parsing(self) -> None:
        """Test that enhancement targets are correctly parsed."""
        test_cases = [
            ("asteroid_generator.py", ["Add resource distribution patterns"]),
            ("fleet_manager.py", ["Implement advanced formation logic"]),
            ("trading_system.py", ["Add dynamic pricing model"]),
            ("ascii_ui.py", ["Improve rendering performance"]),
        ]

        for module, expected_enhancements in test_cases:
            with self.subTest(module=module):
                actual_enhancements = self.parser.get_enhancements_for_module(module)
                self.assertEqual(expected_enhancements, actual_enhancements)

    def test_dependency_parsing(self) -> None:
        """Test that dependencies are correctly parsed."""
        expected_primary = ["numpy", "networkx", "pygame", "scipy"]
        expected_secondary = ["logging", "pathlib", "typing", "random"]

        actual_primary = self.parser.get_dependencies_by_type("primary")
        actual_secondary = self.parser.get_dependencies_by_type("secondary")

        self.assertEqual(sorted(expected_primary), sorted(actual_primary))
        self.assertEqual(sorted(expected_secondary), sorted(actual_secondary))

    def test_map_generation(self) -> None:
        """Test that generated map matches expected format."""
        generated_map = self.parser.generate_map()

        # Verify key sections are present
        self.assertIn("Project Structure:", generated_map)
        self.assertIn("Enhancement Targets:", generated_map)
        self.assertIn("Dependencies Found:", generated_map)

        # Verify content preservation
        self.assertIn("asteroid_generator.py", generated_map)
        self.assertIn("Add resource distribution patterns", generated_map)
        self.assertIn("numpy", generated_map)
        self.assertIn("logging", generated_map)

if __name__ == "__main__":
    unittest.main()
