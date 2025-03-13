"""
Logging and reporting functionality for Python Import Fixer.
"""

# Standard library imports

# Third-party library imports

# Local application imports
from .structured import StructuredLogger
from typing import Any, Dict

__all__ = [
    "get_formatters",
    "get_structured",
    "StructuredLogger",
]


# Lazy imports to avoid circular dependencies
def get_formatters() -> Dict[str, Any]:
    """Get formatter classes lazily to avoid circular imports."""
    from .formatters import (
        ConsoleFormatter,
        EnhancedFormatter,
        FileOperationFormatter,
        JSONFormatter,
    )

    return {
        "ConsoleFormatter": ConsoleFormatter,
        "EnhancedFormatter": EnhancedFormatter,
        "FileOperationFormatter": FileOperationFormatter,
        "JSONFormatter": JSONFormatter,
    }


def get_structured() -> Dict[str, Any]:
    """Get structured logging classes lazily to avoid circular imports."""
    from .structured import LogContext, LogMetrics

    return {
        "LogContext": LogContext,
        "LogMetrics": LogMetrics,
    }
