"""
Variant Loggers - Bridge module between standard logging and enhanced logging.

This module provides a compatibility layer for code that expects a variant_loggers
interface but needs to work with the python_fixer logging system.
"""

# Standard library imports
import logging
from typing import Any, Dict, Optional

# Third-party library imports

# Local application imports
from .enhanced import StructuredLogger

# Define constants to match logging module
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Create a module-level logger
_logger = logging.getLogger("variant_loggers")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger
        
    Returns:
        A logger instance
    """
    return logging.getLogger(name)

def basic_config(level: int = INFO, format: Optional[str] = None, 
                 datefmt: Optional[str] = None, **kwargs: Any) -> None:
    """
    Configure the basic parameters of the logging system.
    
    Args:
        level: The root logger level
        format: The format string for the handler
        datefmt: The date format string for the handler
        **kwargs: Additional parameters to pass to logging.basicConfig
    """
    logging.basicConfig(level=level, format=format, datefmt=datefmt, **kwargs)

# Convenience functions that match the logging module interface
def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message."""
    _logger.debug(msg, *args, **kwargs)

def info(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message."""
    _logger.info(msg, *args, **kwargs)

def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message."""
    _logger.warning(msg, *args, **kwargs)

def error(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an error message."""
    _logger.error(msg, *args, **kwargs)

def critical(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a critical message."""
    _logger.critical(msg, *args, **kwargs)

# Enhanced logging functions that use StructuredLogger if available
def log_with_context(level: int, msg: str, context: Dict[str, Any]) -> None:
    """
    Log a message with additional context data.
    
    Args:
        level: The logging level
        msg: The message to log
        context: Additional context data to include in the log
    """
    try:
        # Try to use StructuredLogger if available
        structured_logger = StructuredLogger.get_instance()
        if level == DEBUG:
            structured_logger.debug(msg, extra={"context": context})
        elif level == INFO:
            structured_logger.info(msg, extra={"context": context})
        elif level == WARNING:
            structured_logger.warning(msg, extra={"context": context})
        elif level == ERROR:
            structured_logger.error(msg, extra={"context": context})
        elif level == CRITICAL:
            structured_logger.critical(msg, extra={"context": context})
    except Exception:
        # Fall back to standard logging if StructuredLogger is not available
        _logger.log(level, msg)
        if context:
            _logger.log(level, f"Context: {context}")
