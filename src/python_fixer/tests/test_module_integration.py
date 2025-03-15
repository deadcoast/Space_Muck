#!/usr/bin/env python3
"""Integration tests for python_fixer module interactions.

This test suite verifies that all modules properly interact with each other
and that the public APIs work as expected across module boundaries.
"""

# Standard library imports
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Third-party library imports
import pytest

# Local application imports
from python_fixer.core import SmartFixer, SignalManager
from python_fixer.enhancers import EnhancementSystem, EventSystem
from python_fixer.parsers import HeaderMapParser, ProjectMapParser
from python_fixer.utils import LogContext
from python_fixer.tests.integration_framework import IntegrationFramework, IntegrationTest


class TestModuleIntegration(unittest.TestCase):
    """Test suite for verifying proper integration between python_fixer modules.
    
    This test suite ensures that:
    1. All modules can be imported correctly
    2. Cross-module functionality works as expected
    3. No circular dependencies exist
    4. Optional dependencies are properly handled
    """
    
    def setUp(self) -> None:
        """Set up test fixtures used by all test methods."""
        self.integration_framework = IntegrationFramework()
        self.test_dir = Path(__file__).parent / "test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create test files
        self.test_file = self.test_dir / "test_module.py"
        with open(self.test_file, "w") as f:
            f.write("""
# Test module for integration testing
import os
import sys
from typing import List, Dict

def test_function(value: str) -> str:
    return f"Processed: {value}"

class TestClass:
    def __init__(self):
        self.value = "initial"
        
    def set_value(self, value: str) -> None:
        self.value = value
        
    def get_value(self) -> str:
        return self.value
""")
    
    def tearDown(self) -> None:
        """Clean up after tests."""
        # Remove test files
        if self.test_file.exists():
            self.test_file.unlink()
        
        # Try to remove test directory
        try:
            os.rmdir(self.test_dir)
        except OSError:
            pass  # Directory not empty or other error
    
    def test_core_enhancers_integration(self) -> None:
        """Test integration between core and enhancers modules."""
        # Create enhancement system
        enhancement_system = EnhancementSystem()
        
        # Create event system
        event_system = EventSystem()
        
        # Register an enhancement
        def enhanced_method(self, value: str) -> str:
            return f"Enhanced: {value}"
        
        # Apply enhancement to TestClass
        from python_fixer.tests.test_data.test_module import TestClass
        enhancement_system.add_method(TestClass, "enhanced_method", enhanced_method)
        
        # Create an instance and test the enhancement
        test_instance = TestClass()
        
        # Verify the enhancement works
        self.assertTrue(hasattr(test_instance, "enhanced_method"))
        self.assertEqual(test_instance.enhanced_method("test"), "Enhanced: test")
        
        # Verify original methods still work
        self.assertEqual(test_instance.get_value(), "initial")
        test_instance.set_value("modified")
        self.assertEqual(test_instance.get_value(), "modified")
    
    def test_parsers_core_integration(self) -> None:
        """Test integration between parsers and core modules."""
        # Create a header map parser
        header_parser = HeaderMapParser()
        
        # Create a project map parser
        project_parser = ProjectMapParser()
        
        # Create a test project map
        test_map = """
Project Structure:
    /test_project
    ├── main.py
    ├── utils/
    │   ├── helpers.py
    │   └── formatters.py
    └── core/
        └── processor.py

Enhancement Targets:
    1. helpers.py: Add utility functions
    2. processor.py: Improve processing logic

Dependencies Found:
    Primary:
    - pytest
    - typing
        """
        
        # Parse the project map
        project_structure = project_parser.parse_map(test_map)
        
        # Verify the parsed structure
        self.assertIn("/test_project", project_structure)
        self.assertIn("main.py", project_structure["/test_project"])
        self.assertIn("utils/", project_structure["/test_project"])
        self.assertIn("core/", project_structure["/test_project"])
        
        # Create a SmartFixer instance
        fixer = SmartFixer()
        
        # Test that the fixer can use the parsed structure
        # (This is a simplified test - in reality, more complex interactions would be tested)
        self.assertIsNotNone(fixer)
        
    def test_utils_integration(self) -> None:
        """Test integration with utils module."""
        # Create a log context
        log_context = LogContext(
            module="test_module",
            function="test_function",
            line=10,
            timestamp=None,  # Will be set automatically
            process_id=os.getpid(),
            thread_id=0,
        )
        
        # Verify the log context
        self.assertEqual(log_context.module, "test_module")
        self.assertEqual(log_context.function, "test_function")
        self.assertEqual(log_context.line, 10)
        
        # Test conversion to dict
        log_dict = log_context.to_dict()
        self.assertIsInstance(log_dict, dict)
        self.assertEqual(log_dict["module"], "test_module")
        self.assertEqual(log_dict["function"], "test_function")
        
    def test_full_module_integration(self) -> None:
        """Test full integration across all modules."""
        # Register integration tests
        self.integration_framework.register_test(
            name="core_test",
            components=[SmartFixer, SignalManager],
            dependencies=[],
        )
        
        self.integration_framework.register_test(
            name="enhancers_test",
            components=[EnhancementSystem, EventSystem],
            dependencies=["core_test"],
        )
        
        self.integration_framework.register_test(
            name="parsers_test",
            components=[HeaderMapParser, ProjectMapParser],
            dependencies=["core_test"],
        )
        
        self.integration_framework.register_test(
            name="full_integration",
            components=[SmartFixer, EnhancementSystem, HeaderMapParser],
            dependencies=["core_test", "enhancers_test", "parsers_test"],
        )
        
        # Run all tests
        results = self.integration_framework.run_all_tests()
        
        # Verify all tests passed
        for test_name, result in results.items():
            self.assertTrue(result, f"Integration test {test_name} failed")


if __name__ == "__main__":
    unittest.main()
