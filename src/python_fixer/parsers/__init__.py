"""
Parsers for Python Import Fixer.

This module provides parsers for analyzing and manipulating project structure,
module headers, and other code elements to support the Python Import Fixer.
"""

# Standard library imports
import importlib.util
from typing import TYPE_CHECKING

# For type checking only - these imports are not executed at runtime
if TYPE_CHECKING:
    try:
        import yaml  # type: ignore
    except ImportError:
        pass

# Check for optional dependencies
YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None

# Import optional dependencies at runtime
yaml = None  # Define at module level

if YAML_AVAILABLE:
    try:
        import yaml
    except ImportError:
        pass

# Local application imports
from .header_map_parser import HeaderMapParser
from .project_map_parser import ProjectMapParser

__all__ = [
    "HeaderMapParser",
    "ProjectMapParser",
]
