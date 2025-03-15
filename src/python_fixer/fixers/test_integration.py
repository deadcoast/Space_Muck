"""
Test script to verify the integration of the fixers module with python_fixer.

This script tests the basic functionality of the fixers module and ensures
that it can be properly imported and used within the python_fixer system.
"""

# Standard library imports
import sys
import unittest
from pathlib import Path

# Add the src directory to the path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Third-party library imports

# Local application imports
from python_fixer import FixManager, SmartFixManager, PatchHandler
from python_fixer.fixers import (
    BaseTransformer,
    RelativeImportTransformer,
    CircularDependencyTransformer,
    Fix,
    ExampleFix
)


class TestFixersIntegration(unittest.TestCase):
    """Test the integration of the fixers module with python_fixer."""

    def test_fix_manager_import(self):
        """Test that FixManager can be imported and instantiated."""
        fix_manager = FixManager()
        self.assertIsInstance(fix_manager, FixManager)
    
    def test_smart_fix_manager_import(self):
        """Test that SmartFixManager can be imported and instantiated."""
        smart_fix_manager = SmartFixManager()
        self.assertIsInstance(smart_fix_manager, SmartFixManager)
    
    def test_patch_handler_import(self):
        """Test that PatchHandler can be imported and instantiated."""
        patch_handler = PatchHandler()
        self.assertIsInstance(patch_handler, PatchHandler)
    
    def test_transformers_import(self):
        """Test that transformers can be imported."""
        # Test that BaseTransformer can be imported
        self.assertTrue(issubclass(RelativeImportTransformer, BaseTransformer))
        self.assertTrue(issubclass(CircularDependencyTransformer, BaseTransformer))
    
    def test_fix_import(self):
        """Test that Fix can be imported and extended."""
        # Test that ExampleFix is a subclass of Fix
        self.assertTrue(issubclass(ExampleFix, Fix))
        
        # Create a custom fix
        class CustomFix(Fix):
            def __init__(self):
                super().__init__(
                    id="custom_fix",
                    description="A custom fix for testing"
                )
            
            def detect(self, code: str) -> bool:
                return "import " in code
            
            def apply(self, code: str) -> str:
                return code.replace("import ", "# import ")
        
        # Test that the custom fix can be instantiated
        custom_fix = CustomFix()
        self.assertIsInstance(custom_fix, Fix)
        self.assertEqual(custom_fix.id, "custom_fix")
        self.assertEqual(custom_fix.description, "A custom fix for testing")
        
        # Test the detect and apply methods
        test_code = "import os\nimport sys\n"
        self.assertTrue(custom_fix.detect(test_code))
        self.assertEqual(custom_fix.apply(test_code), "# import os\n# import sys\n")


if __name__ == "__main__":
    unittest.main()
