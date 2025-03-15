# Standard library imports
import json
from datetime import datetime
from typing import Any, Dict

import variant_loggers

# Local application imports
from rich.console import Console
from variant_loggers import LogRecord

# Third-party library imports


console = Console()


class JsonFormatter:
    """JSON output formatter for structured variant_loggers"""

    def format(self, record: LogRecord) -> str:
        """Format record as JSON string"""
        data = {
            "timestamp": record.timestamp,
            "level": record.get_level_name(),
            "message": record.get_message(),
            "module": record.module,
            "function": record.function,
            "line": record.line,
        }

        # Add exception info if present
        if "exception" in record.extra:
            data["exception"] = record.extra["exception"]

        # Add context if available
        if "context" in record.extra:
            data["context"] = record.extra["context"]

        return json.dumps(data)

    def format_exception(self, exc_info):
        """
        Formats the given exception information into a formatted string.

        :param exc_info: Exception information tuple as returned by sys.exc_info().
        :type exc_info: tuple

        :return: A formatted string that represents the provided exception details.
        :rtype: str
        """
        return "".join(variant_loggers.Formatter.format_exception(self, exc_info))


def get_level_name(self) -> str:
    """Returns the name of the variant_loggers level."""
    return variant_loggers.get_level_name(self.level)


def get_message(self) -> str:
    """Returns the log message."""
    return self.message


@classmethod
def from_dict(cls, record_data: Dict[str, Any]) -> "LogRecord":
    """Creates a LogRecord instance from a dictionary."""
    return cls(
        timestamp=record_data.get("timestamp", datetime.now().timestamp()),
        level=record_data.get("level", "INFO"),
        message=record_data.get("message", ""),
        module=record_data.get("module", "unknown"),
        func_name=record_data.get("func_name"),
        lineno=record_data.get("lineno"),
        process=record_data.get("process"),
        thread=record_data.get("thread"),
        extra=record_data.get("extra"),
        stack_trace=record_data.get("stack_trace"),
    )
