"""
Complete set of log formatters for structured logging with different output formats.
Includes JSON, colored console output, and detailed error formatting.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ColorFormatter(logging.Formatter):
    """
    Formatter that adds ANSI colors to console output for better readability.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors based on log level"""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Basic message with timestamp
        formatted = f"{datetime.fromtimestamp(record.created).isoformat()} "
        formatted += f"{color}[{record.levelname}]{reset} "
        formatted += f"{record.getMessage()}"

        # Add location info
        formatted += f" ({record.module}:{record.funcName}:{record.lineno})"

        # Add any extra data
        if hasattr(record, "metrics"):
            formatted += f"\nMetrics: {json.dumps(record.metrics, indent=2)}"

        if hasattr(record, "file_info"):
            formatted += f"\nFile: {json.dumps(record.file_info, indent=2)}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{color}Exception:{reset}\n"
            formatted += self.formatException(record.exc_info)

        return formatted


class DetailedJSONFormatter(logging.Formatter):
    """
    Enhanced JSON formatter with comprehensive metadata and context.
    """
    def __init__(self, fmt: Optional[str] = None, **kwargs: Dict[str, Any]):
        super().__init__(fmt=fmt)
        self.additional_fields: Dict[str, Any] = kwargs

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as detailed JSON"""
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "location": {
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "path": record.pathname,
            },
            "process": {"id": record.process, "name": record.processName},
            "thread": {"id": record.thread, "name": record.threadName},
        }

        # Add extra context if present
        if hasattr(record, "context"):
            log_data["context"] = record.context

        # Add metrics if present
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics

        # Add file operation info if present
        if hasattr(record, "file_info"):
            log_data["file_info"] = record.file_info

        # Add error details if present
        if record.exc_info:
            log_data["error"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_data)


class FileOperationFormatter(logging.Formatter):
    """
    Specialized formatter for logging file operations with detailed stats.
    """
    def __init__(self, fmt: Optional[str] = None, include_stats: bool = True):
        super().__init__(fmt=fmt)
        self.include_stats = include_stats

    def format(self, record: logging.LogRecord) -> str:
        """Format file operations with statistics"""
        if not hasattr(record, "file_info"):
            return super().format(record)

        file_info = record.file_info
        file_path = Path(file_info["path"])

        # Gather file statistics
        stats = {
            "size": file_path.stat().st_size if file_path.exists() else None,
            "modified": (
                datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                if file_path.exists()
                else None
            ),
            "operation": file_info.get("operation"),
            "details": file_info.get("details", ""),
        }

        # Create formatted output
        output = [
            f"File Operation: {stats['operation']}",
            f"Path: {file_path}",
            f"Size: {stats['size']} bytes",
            f"Last Modified: {stats['modified']}",
            f"Details: {stats['details']}",
        ]

        return "\n".join(output)


class StructuredFormatter:
    """
    Factory class for creating appropriate formatters based on output type.
    """
    @staticmethod
    def create(format_type: str = "json", **kwargs: Dict[str, Any]) -> logging.Formatter:
        """Create a formatter of the specified type.

        Args:
            format_type: One of 'json', 'color', 'file', or 'detailed'
            **kwargs: Additional configuration for the formatter

        Returns:
            Appropriate formatter instance
        """
        if format_type == "json":
            return DetailedJSONFormatter(**kwargs)
        elif format_type == "color":
            return ColorFormatter(**kwargs)
        elif format_type == "file":
            return FileOperationFormatter(**kwargs)
        else:
            return logging.Formatter(**kwargs)


# Example usage:
if __name__ == "__main__":
    # Setup logger with different formatters
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)

    # Console handler with color formatting
    console = logging.StreamHandler()
    console.setFormatter(StructuredFormatter.create("color"))
    logger.addHandler(console)

    # File handler with JSON formatting
    file_handler = logging.FileHandler("test.log")
    file_handler.setFormatter(StructuredFormatter.create("json"))
    logger.addHandler(file_handler)

    # Test logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error(
            "Test error occurred",
            extra={"file_info": {"path": "test.py", "operation": "test"}},
            exc_info=e,
        )
