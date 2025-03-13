# Standard library imports
from datetime import datetime
import json

# Third-party library imports

# Local application imports
from rich.console import Console
from typing import Any, Dict
from variant_loggers import LogRecord
import variant_loggers

console = Console()


class JsonFormatter:
    """JSON output formatter for structured variant_loggers"""

    def format(self, record: LogRecord) -> str:
        """Format record as JSON string"""
        data = {
            "timestamp": record.timestamp,
            "level": record.getLevelName(),
            "message": record.getMessage(),
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

    def formatException(self, exc_info):
        """
        Formats the given exception information into a formatted string.

        :param exc_info: Exception information tuple as returned by sys.exc_info().
        :type exc_info: tuple

        :return: A formatted string that represents the provided exception details.
        :rtype: str
        """
        return "".join(variant_loggers.Formatter.formatException(self, exc_info))


def getLevelName(self) -> str:
    """Returns the name of the variant_loggers level."""
    return variant_loggers.getLevelName(self.level)

def getMessage(self) -> str:
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
