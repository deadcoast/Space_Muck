#!/usr/bin/env python3
"""Integration tests for the python_fixer.fixers module.

This test suite verifies that the fixers module properly integrates with
other modules in the python_fixer package, ensuring that all components
work together seamlessly.
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
from python_fixer.core import SmartFixer
from python_fixer.enhancers import EnhancementSystem
from python_fixer.parsers import HeaderMapParser, ProjectMapParser
from python_fixer.utils import LogContext
from python_fixer.fixers import FixManager, SmartFixManager, PatchHandler
from python_fixer.tests.integration_framework import IntegrationFramework, IntegrationTest


class TestFixersIntegration(unittest.TestCase):
    """Test suite for verifying proper integration of the fixers module.
    
    This test suite ensures that:
    1. The fixers module properly integrates with other modules
    2. Fixing strategies work correctly with the core functionality
    3. Enhancements can be applied to fixers
    4. Project structure parsing is used correctly
    """
    
    def setUp(self) -> None:
        """Set up test fixtures used by all test methods."""
        self.integration_framework = IntegrationFramework()
        self.test_dir = Path(__file__).parent / "test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create test files
        self.test_file = self.test_dir / "test_fixers.py"
        with open(self.test_file, "w") as f:
            f.write("""
# Test module for fixers integration testing
import os
import sys
from typing import List, Dict

# This import should be fixed
from collections import defaultdict, OrderedDict

def test_function(value: str) -> str:
    return f"Processed: {value}"

class TestClass:
    def __init__(self):
        self.value = "initial"
        # This creates a circular import in the test
        self.dict = defaultdict(list)
        
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
    
    def test_fix_manager_integration(self) -> None:
        """Test integration of FixManager with other modules."""
        # Create a fix manager
        fix_manager = FixManager()
        
        # Create a smart fixer from the core module
        smart_fixer = SmartFixer()
        
        # Register the smart fixer with the fix manager
        fix_manager.register_fixer("smart", smart_fixer)
        
        # Verify the registration worked
        self.assertIn("smart", fix_manager.available_fixers)
        
        # Test fixing a simple issue
        result = fix_manager.apply_fix("smart", str(self.test_file))
        
        # Verify the fix was applied
        self.assertTrue(result.success)
        self.assertGreater(len(result.changes), 0)
    
    def test_smart_fix_manager_integration(self) -> None:
        """Test integration of SmartFixManager with other modules."""
        # Create a smart fix manager
        smart_fix_manager = SmartFixManager()
        
        # Create a header map parser from the parsers module
        header_parser = HeaderMapParser()
        
        # Create a project map parser from the parsers module
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
        
        # Set the project structure in the smart fix manager
        smart_fix_manager.set_project_structure(project_structure)
        
        # Verify the project structure was set
        self.assertEqual(smart_fix_manager.project_structure, project_structure)
        
        # Test fixing a file with the project structure context
        result = smart_fix_manager.fix_file(str(self.test_file))
        
        # Verify the fix was applied
        self.assertTrue(result.success)
    
    def test_patch_handler_integration(self) -> None:
        """Test integration of PatchHandler with other modules."""
        # Create a patch handler
        patch_handler = PatchHandler()
        
        # Create a smart fixer from the core module
        smart_fixer = SmartFixer()
        
        # Generate a fix for the test file
        fix_result = smart_fixer.fix_file(str(self.test_file))
        
        # Create a patch from the fix result
        patch = patch_handler.create_patch(str(self.test_file), fix_result.changes)
        
        # Verify the patch was created
        self.assertIsNotNone(patch)
        self.assertGreater(len(patch), 0)
        
        # Apply the patch to a copy of the file
        backup_file = self.test_dir / "test_fixers.py.bak"
        with open(self.test_file, "r") as src:
            with open(backup_file, "w") as dst:
                dst.write(src.read())
        
        # Apply the patch
        result = patch_handler.apply_patch(str(backup_file), patch)
        
        # Verify the patch was applied
        self.assertTrue(result)
        
        # Clean up the backup file
        if backup_file.exists():
            backup_file.unlink()
    
    def test_enhancers_with_fixers(self) -> None:
        """Test integration of the enhancers module with fixers."""
        # Create an enhancement system
        enhancement_system = EnhancementSystem()
        
        # Create a fix manager
        fix_manager = FixManager()
        
        # Define an enhancement for the fix manager
        def enhanced_fix(self, file_path: str, **kwargs) -> Dict[str, Any]:
            # Call the original method
            result = self.original_apply_fix(file_path, **kwargs)
            
            # Enhance the result
            result["enhanced"] = True
            result["timestamp"] = "2023-01-01"
            
            return result
        
        # Store the original method
        fix_manager.original_apply_fix = fix_manager.apply_fix
        
        # Apply the enhancement
        enhancement_system.enhance_method(fix_manager, "apply_fix", enhanced_fix)
        
        # Test the enhanced method
        result = fix_manager.apply_fix("default", str(self.test_file))
        
        # Verify the enhancement was applied
        self.assertTrue(result.get("enhanced", False))
        self.assertEqual(result.get("timestamp"), "2023-01-01")
    
    def test_full_fixers_integration(self) -> None:
        """Test full integration of the fixers module with all other modules."""
        # Register integration tests
        self.integration_framework.register_test(
            name="core_test",
            components=[SmartFixer],
            dependencies=[],
        )
        
        self.integration_framework.register_test(
            name="enhancers_test",
            components=[EnhancementSystem],
            dependencies=["core_test"],
        )
        
        self.integration_framework.register_test(
            name="parsers_test",
            components=[HeaderMapParser, ProjectMapParser],
            dependencies=["core_test"],
        )
        
        self.integration_framework.register_test(
            name="fixers_test",
            components=[FixManager, SmartFixManager, PatchHandler],
            dependencies=["core_test", "enhancers_test", "parsers_test"],
        )
        
        # Run all tests
        results = self.integration_framework.run_all_tests()
        
        # Verify all tests passed
        for test_name, result in results.items():
            self.assertTrue(result, f"Integration test {test_name} failed")


if __name__ == "__main__":
    unittest.main()
