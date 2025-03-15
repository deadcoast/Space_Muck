"""
Complete, working structured logger with JSON formatting and file output.
"""

import json
import logging
import sys

# Standard library imports
from datetime import datetime

# Local application imports
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party library imports


class JSONFormatter(logging.Formatter):
    """Format log records as JSON"""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add any extra fields
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics

        if hasattr(record, "file_info"):
            log_data["file_info"] = record.file_info

        # Include traceback for errors
        if record.exc_info:
            log_data["error"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_data)


class StructuredLogger:
    """Enhanced logger with structured output and convenience methods"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, log_level: str = "INFO") -> logging.Logger:
        """Get a configured logger instance.
        
        Args:
            name: Name of the logger
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
            
        # Create logger
        logger = logging.getLogger(name)
        level = getattr(logging, log_level.upper())
        logger.setLevel(level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Create console handler with JSON formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        logger.addHandler(console_handler)
        
        # Store logger in cache
        cls._loggers[name] = logger
        
        return logger
        
    @classmethod
    def setup_file_logging(cls, log_dir: Path, name: str = "python_fixer") -> None:
        """Set up file logging.
        
        Args:
            log_dir: Directory to store log files
            name: Base name for log files
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # Add file handler with JSON formatter
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    def __init__(self, name: str, log_file: Optional[Path] = None, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Create formatters
        json_formatter = JSONFormatter()

        # Console handler (with colors for better readability)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(json_formatter)
        self.logger.addHandler(console)

        # File handler if requested
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message with optional extra data"""
        self.logger.info(message, extra=extra, **kwargs)

    def error(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Log error message with optional exception info"""
        self.logger.error(message, exc_info=exc_info, extra=extra, **kwargs)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=extra, **kwargs)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=extra, **kwargs)

    def file_operation(
        self, file_path: Path, operation: str, details: Optional[str] = None
    ):
        """Log file operation with details"""
        file_info = {
            "path": str(file_path),
            "operation": operation,
            "size": file_path.stat().st_size if file_path.exists() else None,
        }
        if details:
            file_info["details"] = details

        self.info(
            f"File operation: {operation} on {file_path}",
            extra={"file_info": file_info},
        )
