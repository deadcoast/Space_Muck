"""
Tests for import validation functionality.

This test suite covers:
1. Import validation
2. Error message generation
3. Package path resolution
4. Invalid import detection
"""

# Standard library imports
import os
import sys

# Third-party library imports
import pytest

# Local application imports
from ...invalid_pkg import AnotherClass
from ..parent_pkg.nonexistent import AnotherClass
from ..pkg import SomeClass
from .nonexistent_module import NonexistentClass
from another_nonexistent_package import SomeClass
from pathlib import Path
from python_fixer.core.analyzer import ImportAnalyzer
from python_fixer.core.types import ImportInfo
from typing import List, Optional
import nonexistent_package

# Test fixtures
@pytest.fixture
def invalid_import_structure(tmp_path: Path) -> Path:
    """Create a temporary package structure with invalid imports.

    Structure:
    invalid_pkg/
    ├── __init__.py
    ├── module_a.py  # contains valid imports
    ├── module_b.py  # contains invalid relative import
    └── module_c.py  # contains invalid absolute import
    """
    pkg_root = tmp_path / "invalid_pkg"
    pkg_root.mkdir()

    # Create package __init__.py
    (pkg_root / "__init__.py").write_text("")

    # Create module_a.py with valid imports
    (pkg_root / "module_a.py").write_text(
        """

def func_a():
    pass
"""
    )

    # Create module_b.py with invalid relative import
    (pkg_root / "module_b.py").write_text(
        """

def func_b():
    pass
"""
    )

    # Create module_c.py with invalid absolute import
    (pkg_root / "module_c.py").write_text(
        """

def func_c():
    pass
"""
    )

    return pkg_root

def test_valid_import_validation(invalid_import_structure: Path) -> None:
    """Test validation of valid imports."""
    analyzer = ImportAnalyzer(invalid_import_structure / "module_a.py")
    analyzer.analyze_imports()  # Call the method without storing the unused result

    # Check that all imports are valid
    assert (
        analyzer.validate_all_imports()
    ), "Valid imports should be validated successfully"
    assert (
        len(analyzer.get_invalid_imports()) == 0
    ), "There should be no invalid imports"
    assert len(analyzer.get_import_errors()) == 0, "There should be no import errors"

def test_invalid_relative_import_validation(invalid_import_structure: Path) -> None:
    """Test validation of invalid relative imports."""
    analyzer = ImportAnalyzer(invalid_import_structure / "module_b.py")
    analyzer.analyze_imports()  # Call the method without storing the unused result

    # Check that imports are invalid
    assert not analyzer.validate_all_imports(), "Invalid imports should fail validation"

    # Check that we have the expected number of invalid imports
    invalid_imports = analyzer.get_invalid_imports()
    assert len(invalid_imports) == 2, "There should be 2 invalid imports"

    # Check error messages
    error_messages = analyzer.get_import_errors()
    assert any(
        "nonexistent_module" in msg for msg in error_messages
    ), "Error for nonexistent_module not found"
    assert any(
        "parent_pkg" in msg for msg in error_messages
    ), "Error for parent_pkg not found"

    # Check that the invalid imports have error messages
    for imp in invalid_imports:
        assert not imp.is_valid, "Import should be marked as invalid"
        assert imp.error_message is not None, "Import should have an error message"

def test_invalid_absolute_import_validation(invalid_import_structure: Path) -> None:
    """Test validation of invalid absolute imports."""
    analyzer = ImportAnalyzer(invalid_import_structure / "module_c.py")
    analyzer.analyze_imports()  # Call the method without storing the unused result

    # Check that imports are invalid
    assert not analyzer.validate_all_imports(), "Invalid imports should fail validation"

    # Check that we have the expected number of invalid imports
    invalid_imports = analyzer.get_invalid_imports()
    assert len(invalid_imports) == 2, "There should be 2 invalid imports"

    # Check error messages
    error_messages = analyzer.get_import_errors()
    assert any(
        "nonexistent_package" in msg for msg in error_messages
    ), "Error for nonexistent_package not found"
    assert any(
        "another_nonexistent_package" in msg for msg in error_messages
    ), "Error for another_nonexistent_package not found"

    # Check that the invalid imports have error messages
    for imp in invalid_imports:
        assert not imp.is_valid, "Import should be marked as invalid"
        assert imp.error_message is not None, "Import should have an error message"

def test_package_path_resolution(invalid_import_structure: Path) -> None:
    """Test package path resolution for import validation."""
    # Create a nested package structure
    nested_pkg = invalid_import_structure / "nested" / "pkg"
    nested_pkg.mkdir(parents=True)
    (nested_pkg.parent / "__init__.py").write_text("")
    (nested_pkg / "__init__.py").write_text("")

    # Create a module with relative imports
    (nested_pkg / "module_d.py").write_text(
        """

"""
    )

    analyzer = ImportAnalyzer(nested_pkg / "module_d.py")
    analyzer.analyze_imports()  # Call the method without storing the unused result

    # Check that the package path is correctly resolved
    assert (
        analyzer._package_path == "invalid_pkg.nested.pkg"
    ), "Package path should be correctly resolved"

    # Check that the relative imports are validated correctly
    invalid_imports = analyzer.get_invalid_imports()
    assert len(invalid_imports) == 2, "There should be 2 invalid imports"

    # Check error messages
    error_messages = analyzer.get_import_errors()
    assert any("pkg" in msg for msg in error_messages), "Error for pkg not found"
    assert any(
        "invalid_pkg" in msg for msg in error_messages
    ), "Error for invalid_pkg not found"

def test_direct_import_info_validation() -> None:
    """Test direct validation of ImportInfo objects."""
    # Test absolute import validation
    abs_import = ImportInfo(module="nonexistent_package", is_relative=False)
    assert (
        not abs_import.validate()
    ), "Nonexistent absolute import should fail validation"
    assert abs_import.error_message is not None, "Error message should be set"
    assert (
        "Cannot resolve absolute import" in abs_import.error_message
    ), "Error message should mention absolute import"

    # Test relative import validation without package path
    rel_import = ImportInfo(module="module", is_relative=True, level=1)
    assert (
        not rel_import.validate()
    ), "Relative import without package path should fail validation"
    assert rel_import.error_message is not None, "Error message should be set"
    assert (
        "Cannot resolve relative import without package path"
        in rel_import.error_message
    ), "Error message should mention missing package path"

    # Test relative import validation with invalid level
    rel_import2 = ImportInfo(module="module", is_relative=True, level=3)
    assert not rel_import2.validate(
        "pkg.subpkg"
    ), "Relative import with excessive level should fail validation"
    assert rel_import2.error_message is not None, "Error message should be set"
    assert (
        "Relative import level" in rel_import2.error_message
    ), "Error message should mention level issue"

    # Test valid import
    valid_import = ImportInfo(module="os", is_relative=False)
    assert valid_import.validate(), "Valid import should pass validation"
    assert valid_import.error_message is None, "Error message should not be set"
    assert valid_import.is_valid, "Import should be marked as valid"
