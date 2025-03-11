import json
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Base logging levels
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRITICAL = 50

# Web dashboard configuration
DEFAULT_DASHBOARD_HOST = "localhost"
DEFAULT_DASHBOARD_PORT = 8000
DEFAULT_DASHBOARD_RELOAD = False


def getLevelName(level: int) -> str:
    """Convert numeric level to string representation."""
    levels = {
        DEBUG: "DEBUG",
        INFO: "INFO",
        WARNING: "WARNING",
        ERROR: "ERROR",
        CRITICAL: "CRITICAL",
    }
    return levels.get(level, str(level))


@dataclass
class LogContext:
    """Structure to hold context information for log entries."""

    timestamp: datetime
    module: str
    function: str
    line: int
    extra: Dict[str, Any]
    process_id: int
    thread_id: int
    thread_name: str = threading.current_thread().name
    log_path: Optional[Path] = None
    context_id: str = str(uuid.uuid4())


class LogRecord:
    """Container for log record information."""

    def __init__(
        self,
        level: int,
        message: str,
        timestamp: str,
        module: str = "unknown",
        function: str = "unknown",
        line: int = 0,
        extra: Optional[Dict[str, Any]] = None,
        web_visible: bool = True,
        log_path: Optional[Path] = None,
        context_id: str = str(uuid.uuid4()),
    ):
        self.level = level
        self.message = message
        self.timestamp = timestamp
        self.module = module
        self.function = function
        self.line = line
        self.extra = extra or {}
        # Merge context data if available
        if hasattr(self, "_context_stack") and self._context_stack:
            context = self._context_stack[-1]
            self.extra.update(context.extra)
        self.log_path = log_path
        self.context_id = context_id
        self.thread_name = threading.current_thread().name

    def to_dict(self) -> Dict[str, Any]:
        """Convert log record to dictionary format."""
        return {
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp,
            "module": self.module,
            "function": self.function,
            "line": self.line,
            "extra": self.extra,
            "web_visible": getattr(self, "web_visible", True),
            "thread_name": self.thread_name,
            "log_path": str(self.log_path) if self.log_path else None,
            "context_id": self.context_id,
        }


class Handler:
    """Base class for log handlers."""

    def __init__(self):
        self.level = INFO
        self.formatter = None

    def setLevel(self, level: int):
        """Set the logging level for this handler."""
        self.level = level

    def setFormatter(self, formatter):
        """Set the formatter for this handler."""
        self.formatter = formatter

    def handle(self, record: LogRecord):
        """Handle a log record."""
        raise NotImplementedError

    def close(self):
        """Clean up handler resources."""
        pass


class StreamHandler(Handler):
    """Handler for logging to streams (e.g., stdout)."""

    def __init__(self, stream=sys.stdout):
        super().__init__()
        self.stream = stream

    def handle(self, record: LogRecord):
        if record.level >= self.level:
            msg = (
                self.formatter.format(record) if self.formatter else str(record.message)
            )
            self.stream.write(msg + "\n")
            self.stream.flush()


class FileHandler(Handler):
    """Handler for logging to files."""

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.file = open(filename, "a")

    def handle(self, record: LogRecord):
        if record.level >= self.level:
            msg = (
                self.formatter.format(record) if self.formatter else str(record.message)
            )
            self.file.write(msg + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


class Formatter:
    """Formatter for log records."""

    def __init__(self, fmt: str):
        self.fmt = fmt

    def format(self, record: LogRecord) -> str:
        """Format a log record according to the format string."""
        try:
            fmt_config = json.loads(self.fmt)
            format_str = fmt_config.get("format", "{timestamp} - {level} - {message}")

            # Check for level-specific format rules
            if "format_rules" in fmt_config:
                for rule in fmt_config["format_rules"]:
                    if getLevelName(record.level) == rule["level"]:
                        format_str = rule["format"]
                        break

            return format_str.format(
                timestamp=record.timestamp,
                level=getLevelName(record.level),
                module=record.module,
                function=record.function,
                line=record.line,
                message=record.message,
            )
        except json.JSONDecodeError:
            # If fmt is not JSON, use it as a simple format string
            return self.fmt.format(
                timestamp=record.timestamp,
                level=getLevelName(record.level),
                module=record.module,
                function=record.function,
                line=record.line,
                message=record.message,
            )


class Logger:
    """Main logger class."""

    def __init__(self, name: str):
        self.name = name
        self.level = INFO
        self.handlers: List[Handler] = []
        self._context_stack: List[LogContext] = []

    @asynccontextmanager
    async def context(self, **kwargs) -> LogContext:
        """Create an async logging context with additional metadata.

        Args:
            **kwargs: Additional context information to include in logs

        Example:
            async with logger.context(task_id='123'):
                logger.info('Processing task')
        """
        context = LogContext(
            timestamp=datetime.now(),
            module=self.name,
            function="",
            line=0,
            extra=kwargs,
            process_id=0,
            thread_id=0,
        )
        self._context_stack.append(context)
        try:
            yield context
        finally:
            self._context_stack.pop()

    def setLevel(self, level: int):
        """Set the logging level for this logger."""
        self.level = level

    def addHandler(self, handler: Handler):
        """Add a handler to this logger."""
        self.handlers.append(handler)

    def _get_current_context(self) -> Optional[LogContext]:
        """Get the current logging context if available."""
        return self._context_stack[-1] if self._context_stack else None

    def removeHandler(self, handler: Handler):
        """Remove a handler from this logger."""
        if handler in self.handlers:
            self.handlers.remove(handler)

    def log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log a message with the specified level."""
        if level >= self.level:
            record = LogRecord(
                level=level,
                message=message,
                timestamp=datetime.now().isoformat(),
                extra=extra,
            )
            for handler in self.handlers:
                handler.handle(record)


# Logger cache to avoid creating multiple loggers with the same name
_loggers = {}


def getLogger(name: str) -> Logger:
    """Get or create a logger with the specified name."""
    if name not in _loggers:
        _loggers[name] = Logger(name)
    return _loggers[name]


def debug(msg: str, *args, **kwargs):
    """Log a debug message."""
    getLogger("root").log(DEBUG, msg, kwargs.get("extra"))


def info(msg: str, *args, **kwargs):
    """Log an info message."""
    getLogger("root").log(INFO, msg, kwargs.get("extra"))


def warning(msg: str, *args, **kwargs):
    """Log a warning message."""
    getLogger("root").log(WARNING, msg, kwargs.get("extra"))


def error(msg: str, *args, **kwargs):
    """Log an error message."""
    getLogger("root").log(ERROR, msg, kwargs.get("extra"))


def critical(msg: str, *args, **kwargs):
    """Log a critical message."""
    getLogger("root").log(CRITICAL, msg, kwargs.get("extra"))
