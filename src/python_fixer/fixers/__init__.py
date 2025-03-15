"""Python Fixer - Fixers Module

This module provides classes and utilities for fixing Python code issues,
including import optimization, dependency resolution, and code transformation.

The fixers module integrates with other python_fixer modules to provide
comprehensive code fixing capabilities.
"""

# Standard library imports
import contextlib
import importlib.util
from typing import TYPE_CHECKING

# Local application imports - these need to be at the top level for proper module structure
from .manager import FixManager
from .smart import SmartFixManager, Fix, ExampleFix
from .transformers import BaseTransformer, RelativeImportTransformer, CircularDependencyTransformer
from .patch import PatchHandler

# Check for optional dependencies
PATCH_AVAILABLE = importlib.util.find_spec("patch") is not None
QUESTIONARY_AVAILABLE = importlib.util.find_spec("questionary") is not None

# Define module-level variables for optional dependencies
patch = None
questionary = None

# For type checking only
if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        import patch  # type: ignore
        import questionary  # type: ignore

# Import optional dependencies at runtime
if PATCH_AVAILABLE:
    with contextlib.suppress(ImportError):
        import patch as _patch  # Imported but not used directly
        patch = _patch  # Assign to module-level variable

if QUESTIONARY_AVAILABLE:
    with contextlib.suppress(ImportError):
        import questionary as _questionary  # Imported but not used directly
        questionary = _questionary  # Assign to module-level variable

__all__ = [
    "FixManager", 
    "SmartFixManager", 
    "Fix", 
    "ExampleFix",
    "BaseTransformer",
    "RelativeImportTransformer", 
    "CircularDependencyTransformer",
    "PatchHandler",
    "PATCH_AVAILABLE",
    "QUESTIONARY_AVAILABLE"
]