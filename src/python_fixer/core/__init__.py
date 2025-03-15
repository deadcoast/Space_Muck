"""
Core functionality for Python Import Fixer.

This module provides the core types, signals, and fixer functionality
for the Python Import Fixer system.
"""

# Standard library imports
import importlib.util
from typing import TYPE_CHECKING

# For type checking only - these imports are not executed at runtime
if TYPE_CHECKING:
    try:
        import ast  # type: ignore
    except ImportError:
        pass

# Check for optional dependencies
AST_AVAILABLE = importlib.util.find_spec("ast") is not None

# Import optional dependencies at runtime
ast = None  # Define at module level

if AST_AVAILABLE:
    try:
        import ast
    except ImportError:
        pass

# Local application imports
from .fixer import SmartFixer, FixOperation, FixStrategy
from .signals import SignalManager
from .signatures import (
    CodeSignature, 
    SignatureComponent,
    TypeInfo,
    SignatureMetrics
)
from .types import (
    TypeCheckable,
    TypeCheckResult,
    validate_type,
    validate_protocol,
    ImportInfo
)

__all__ = [
    # Fixer classes
    "SmartFixer",
    "FixOperation",
    "FixStrategy",
    
    # Signal handling
    "SignalManager",
    
    # Signature analysis
    "CodeSignature",
    "SignatureComponent",
    "TypeInfo",
    "SignatureMetrics",
    
    # Type validation
    "TypeCheckable",
    "TypeCheckResult",
    "validate_type",
    "validate_protocol",
    "ImportInfo"
]
