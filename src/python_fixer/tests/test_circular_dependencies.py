#!/usr/bin/env python3
"""Test for circular dependencies in the python_fixer package.

This test suite verifies that no circular dependencies exist between modules
in the python_fixer package, ensuring a clean, hierarchical import structure.
"""

# Standard library imports
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import unittest

# Third-party library imports
import pytest

# Local application imports
# None - we'll import modules dynamically to test for circular dependencies


class TestCircularDependencies(unittest.TestCase):
    """Test suite for detecting circular dependencies in the python_fixer package.
    
    This test suite ensures that:
    1. No circular dependencies exist between modules
    2. Import structure follows a clean hierarchical pattern
    3. All modules can be imported without side effects
    """
    
    def setUp(self) -> None:
        """Set up test fixtures used by all test methods."""
        # Get the project root directory
        self.project_root = Path(__file__).parent.parent.resolve()
        
        # List of all modules to check
        self.modules = [
            "python_fixer.core",
            "python_fixer.enhancers",
            "python_fixer.parsers",
            "python_fixer.utils",
            "python_fixer.fixers",
            "python_fixer.logging",
            "python_fixer.analyzers",
        ]
        
        # Dictionary to track imported modules and their dependencies
        self.imported_modules: Dict[str, Set[str]] = {}
        
        # Dictionary to track import paths (for debugging)
        self.import_paths: Dict[Tuple[str, str], List[str]] = {}
    
    def _get_module_dependencies(self, module_name: str) -> Set[str]:
        """Get the dependencies of a module.
        
        Args:
            module_name: The name of the module to check
            
        Returns:
            A set of module names that the module depends on
        """
        # If we've already checked this module, return the cached result
        if module_name in self.imported_modules:
            return self.imported_modules[module_name]
        
        # Try to import the module
        try:
            module = importlib.import_module(module_name)
            
            # Get the module's dependencies
            dependencies = set()
            
            # Check the module's __all__ attribute if it exists
            if hasattr(module, "__all__"):
                for attr_name in module.__all__:
                    if hasattr(module, attr_name):
                        attr = getattr(module, attr_name)
                        if hasattr(attr, "__module__"):
                            dep_module = attr.__module__
                            if dep_module.startswith("python_fixer") and dep_module != module_name:
                                dependencies.add(dep_module)
            
            # Cache the result
            self.imported_modules[module_name] = dependencies
            return dependencies
            
        except ImportError as e:
            # If the module can't be imported, return an empty set
            self.imported_modules[module_name] = set()
            return set()
    
    def _detect_circular_dependency(
        self, 
        module_name: str, 
        visited: Set[str] = None, 
        path: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """Detect circular dependencies for a module.
        
        Args:
            module_name: The name of the module to check
            visited: Set of modules visited in the current path
            path: Current path of modules being checked
            
        Returns:
            A tuple of (has_circular_dependency, circular_path)
        """
        if visited is None:
            visited = set()
        
        if path is None:
            path = []
        
        # If we've already visited this module in the current path, we have a circular dependency
        if module_name in visited:
            return True, path + [module_name]
        
        # Add the module to the visited set and path
        visited.add(module_name)
        path.append(module_name)
        
        # Get the module's dependencies
        dependencies = self._get_module_dependencies(module_name)
        
        # Check each dependency for circular dependencies
        for dep in dependencies:
            # Store the import path for debugging
            self.import_paths[(module_name, dep)] = path.copy()
            
            # Check for circular dependencies
            has_circular, circular_path = self._detect_circular_dependency(
                dep, visited.copy(), path.copy()
            )
            
            if has_circular:
                return True, circular_path
        
        # No circular dependencies found
        return False, []
    
    def test_no_circular_dependencies(self) -> None:
        """Test that no circular dependencies exist between modules."""
        for module_name in self.modules:
            has_circular, circular_path = self._detect_circular_dependency(module_name)
            
            # If a circular dependency is found, fail the test
            self.assertFalse(
                has_circular,
                f"Circular dependency detected: {' -> '.join(circular_path)}"
            )
    
    def test_import_all_modules(self) -> None:
        """Test that all modules can be imported without errors."""
        for module_name in self.modules:
            try:
                importlib.import_module(module_name)
                # If we get here, the import succeeded
                self.assertTrue(True)
            except ImportError as e:
                # If the import fails, fail the test
                self.fail(f"Failed to import {module_name}: {str(e)}")
    
    def test_hierarchical_imports(self) -> None:
        """Test that imports follow a hierarchical structure."""
        # Define the allowed import directions
        allowed_imports = {
            "python_fixer.core": ["python_fixer.utils"],
            "python_fixer.enhancers": ["python_fixer.core", "python_fixer.utils"],
            "python_fixer.parsers": ["python_fixer.core", "python_fixer.utils"],
            "python_fixer.fixers": ["python_fixer.core", "python_fixer.enhancers", "python_fixer.parsers", "python_fixer.utils"],
            "python_fixer.logging": ["python_fixer.core", "python_fixer.utils"],
            "python_fixer.analyzers": ["python_fixer.core", "python_fixer.enhancers", "python_fixer.parsers", "python_fixer.utils", "python_fixer.fixers", "python_fixer.logging"],
        }
        
        # Check each module's dependencies
        for module_name, allowed in allowed_imports.items():
            dependencies = self._get_module_dependencies(module_name)
            
            for dep in dependencies:
                # Check if the dependency is allowed
                self.assertIn(
                    dep,
                    allowed,
                    f"Unexpected dependency: {module_name} -> {dep}"
                )


if __name__ == "__main__":
    unittest.main()
