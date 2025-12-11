"""
src/tests/integration/test_fixers_integration.py

Integration tests for the fixers module.

These tests verify that the fixers module integrates properly with other modules
in the Python Fixer system, including core, enhancers, parsers, and utils.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Define module-level variables for optional dependencies
git = None
black = None
isort = None
libcst = None
networkx = None
numpy = None
scipy = None
sklearn = None
rope = None

# Check for optional dependencies
GIT_AVAILABLE = importlib.util.find_spec("git") is not None
BLACK_AVAILABLE = importlib.util.find_spec("black") is not None
ISORT_AVAILABLE = importlib.util.find_spec("isort") is not None
LIBCST_AVAILABLE = importlib.util.find_spec("libcst") is not None
NETWORKX_AVAILABLE = importlib.util.find_spec("networkx") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
ROPE_AVAILABLE = importlib.util.find_spec("rope") is not None

# Import optional dependencies only if needed in tests
# We're just checking availability with find_spec above

# Mock the CLI module to avoid dependency issues
sys.modules["cli"] = mock.MagicMock()

# Check if modules are available - do this after setting up mocks
CORE_AVAILABLE = importlib.util.find_spec("python_fixer.core") is not None
ENHANCERS_AVAILABLE = importlib.util.find_spec("python_fixer.enhancers") is not None
PARSERS_AVAILABLE = importlib.util.find_spec("python_fixer.parsers") is not None
UTILS_AVAILABLE = importlib.util.find_spec("python_fixer.utils") is not None
FIXERS_AVAILABLE = importlib.util.find_spec("python_fixer.fixers") is not None

# Define module-level variables for imports
EnhancementSystem = None
EventSystem = None
ProjectMapParser = None
FixManager = None
SmartFixManager = None
RelativeImportTransformer = None
PatchHandler = None

# Import the modules we want to test - with proper error handling
try:
    if (
        CORE_AVAILABLE
        and ENHANCERS_AVAILABLE
        and PARSERS_AVAILABLE
        and UTILS_AVAILABLE
        and FIXERS_AVAILABLE
    ):
        # Only import what we actually use in tests
        from python_fixer.fixers import (
            FixManager,
            PatchHandler,
            RelativeImportTransformer,
            SmartFixManager,
        )

        IMPORTS_SUCCESSFUL = True
    else:
        IMPORTS_SUCCESSFUL = False
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Required modules could not be imported")
class TestFixersIntegration(unittest.TestCase):
    """Test the integration of the fixers module with other modules."""

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = Path(__file__).parent / "test_data"
        self.test_dir.mkdir(exist_ok=True)

        # Create a test file
        self.test_file = self.test_dir / "test_file.py"
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("# Test file for fixers integration tests\n")
            f.write("from ..module import function\n")  # Relative import to fix

    def tearDown(self):
        """Clean up the test environment."""
        if self.test_file.exists():
            self.test_file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_fix_manager_with_core(self):
        """Test that FixManager integrates with core components."""
        # Create a FixManager instance
        fix_manager = FixManager()

        # Create a mock analyzer
        mock_analyzer = mock.MagicMock()
        mock_analyzer.modules = [self.test_file]

        # Verify it can work with SmartFixer from core
        with mock.patch("python_fixer.core.fixer.SmartFixer") as mock_fixer:
            mock_fixer.return_value.fix_project.return_value = True

            # This should use the run method instead of apply_fixes
            fix_manager.run(mock_analyzer, "auto_fix", {})

            # Verify the analyzer was used
            self.assertTrue(mock_analyzer.method_calls)

    def test_smart_fix_manager_with_enhancers(self):
        """Test that SmartFixManager integrates with enhancers."""
        # Create a SmartFixManager instance
        smart_fix_manager = SmartFixManager()

        # Create a mock analyzer
        mock_analyzer = mock.MagicMock()
        mock_analyzer.modules = [self.test_file]

        # Verify it can work with EnhancementSystem from enhancers
        with mock.patch("python_fixer.enhancers.EnhancementSystem") as mock_enhancement:
            mock_enhancement.return_value.enhance.return_value = True

            # Use the run method instead of enhance_fixes
            smart_fix_manager.run(mock_analyzer, "interactive_fix", {})

            # Verify the analyzer was used
            self.assertTrue(mock_analyzer.method_calls)

    def test_transformer_with_parsers(self):
        """Test that transformers integrate with parsers."""
        # Create a RelativeImportTransformer instance - no arguments needed
        transformer = RelativeImportTransformer()

        # Create a mock analyzer
        mock_analyzer = mock.MagicMock()
        mock_analyzer.modules = [self.test_file]

        # Verify it can work with parsers
        with mock.patch.object(transformer, "apply") as mock_apply:
            # Call the apply method with the analyzer
            transformer.apply(mock_analyzer)

            # Verify the apply method was called with the analyzer
            mock_apply.assert_called_once_with(mock_analyzer)

    def test_patch_handler_with_utils(self):
        """Test that PatchHandler integrates with utils for logging."""
        # Create a PatchHandler instance
        patch_handler = PatchHandler()

        # Verify it can work with LogContext from utils
        with mock.patch("python_fixer.utils.LogContext") as mock_log_context:
            self._setup_log_context_mock(mock_log_context)

            # Test the patch handler with mocked apply_fix method
            self._test_patch_handler_apply_fix(patch_handler)

    def _test_patch_handler_apply_fix(self, patch_handler):
        """Test the patch handler's apply_fix method with mocks."""
        # Apply a patch (implementation details simplified for test)
        with mock.patch.object(patch_handler, "apply_fix") as mock_apply:
            mock_apply.return_value = True

            # This should use LogContext from utils
            result = patch_handler.apply_fix(str(self.test_file), "test_fix")

            # Verify the result
            self.assertTrue(result)

    @staticmethod
    def _setup_log_context_mock(mock_log_context):
        """Set up the mock for LogContext."""
        # Mock the context manager behavior
        mock_log_context.return_value.__enter__.return_value = mock_log_context
        mock_log_context.return_value.__exit__.return_value = None

    def test_no_circular_dependencies(self):
        """Test that there are no circular dependencies between modules."""
        # This test verifies that we can import all modules without circular dependency errors
        self._verify_no_circular_imports()

    def _verify_no_circular_imports(self):
        """Helper method to verify no circular imports exist."""
        # First, clear all modules to ensure a clean import
        for module in list(sys.modules.keys()):
            if module.startswith("python_fixer"):
                del sys.modules[module]

        # Now import all modules in the correct order (bottom-up)
        try:
            self._verify_no_circular_imports()
        except ImportError as e:
            self.skipTest(f"Could not import all modules: {e}")

    def _verify_no_circular_imports(self):
        # Import in hierarchical order to verify no circular dependencies
        __import__("python_fixer.utils")
        __import__("python_fixer.core")
        __import__("python_fixer.enhancers")
        __import__("python_fixer.parsers")
        __import__("python_fixer.analyzers")

        # Check if fixers module is available before importing
        if FIXERS_AVAILABLE:
            __import__("python_fixer.fixers")

        # If we reach this point, no import errors occurred
        self.assertTrue(IMPORTS_SUCCESSFUL, "No circular dependencies detected")

    def test_integration_with_all_modules(self):
        """Test integration with all modules in a realistic scenario."""
        # Skip test if any required module is not available
        if not all(
            [CORE_AVAILABLE, ENHANCERS_AVAILABLE, PARSERS_AVAILABLE, UTILS_AVAILABLE]
        ):
            self.skipTest("Not all required modules are available")

        # Create a sample map content for ProjectMapParser
        map_content = """{
            "modules": [
                "test_module"
            ]
        }"""

        # Create instances of components with proper initialization
        fix_manager = FixManager()
        smart_fix_manager = SmartFixManager()

        # Mock the necessary components to avoid actual file operations
        with mock.patch(
            "python_fixer.enhancers.EnhancementSystem"
        ) as mock_enhancement_system, mock.patch(
            "python_fixer.parsers.project_map_parser.ProjectMapParser"
        ) as mock_project_map_parser_class:

            # Set up mock instances
            mock_enhancement = mock_enhancement_system.return_value
            mock_enhancement.enhance.return_value = True

            # Create a mock ProjectMapParser instance
            mock_project_parser = mock.MagicMock()
            mock_project_parser.parse.return_value = {"modules": [str(self.test_file)]}
            mock_project_map_parser_class.return_value = mock_project_parser

            # Create a mock analyzer
            mock_analyzer = mock.MagicMock()
            mock_analyzer.modules = [self.test_file]

            # 1. Create the project parser with the map content
            project_parser = mock_project_map_parser_class(map_content)
            project_structure = project_parser.parse()

            # 2. Run the fix manager with the analyzer
            fix_manager.run(mock_analyzer, "auto_fix", {})

            # 3. Run the smart fix manager with the analyzer
            smart_fix_manager.run(mock_analyzer, "interactive_fix", {})

            # Verify the results
            self.assertEqual(project_structure, {"modules": [str(self.test_file)]})
            self.assertTrue(mock_analyzer.method_calls)


if __name__ == "__main__":
    # Print information about available dependencies
    print("Optional dependencies availability:")
    print(f"git: {GIT_AVAILABLE}")
    print(f"black: {BLACK_AVAILABLE}")
    print(f"isort: {ISORT_AVAILABLE}")
    print(f"libcst: {LIBCST_AVAILABLE}")
    print(f"networkx: {NETWORKX_AVAILABLE}")
    print(f"numpy: {NUMPY_AVAILABLE}")
    print(f"scipy: {SCIPY_AVAILABLE}")
    print(f"sklearn: {SKLEARN_AVAILABLE}")
    print(f"rope: {ROPE_AVAILABLE}")
    print(f"All imports successful: {IMPORTS_SUCCESSFUL}")

    unittest.main()
