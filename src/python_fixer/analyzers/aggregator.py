# Standard library imports
import importlib.util
import inspect
import logging as _logging
from collections import defaultdict

# Local application imports
from contextlib import suppress
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List

# Third-party library imports
import numpy as np
import pandas as pd

# Check for optional dependencies
VARIANT_LOGGERS_AVAILABLE = importlib.util.find_spec("variant_loggers") is not None

# For type checking only
if TYPE_CHECKING:
    with suppress(ImportError):
        from variant_loggers import LogCorrelator, LogRecord  # type: ignore
# Import optional dependencies at runtime
LogCorrelator = None
LogRecord = None
log_correlator = None

if VARIANT_LOGGERS_AVAILABLE:
    with suppress(ImportError):
        from variant_loggers import LogCorrelator, LogRecord

        log_correlator = LogCorrelator()


# Existing Aggregator Classes
class Aggregator:
    """Aggregate log entries for analysis"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.entries: List[LogRecord] = []
        self.pattern_counts = defaultdict(int)
        self.error_count = 0
        self.processing_times: List[float] = []
        self.counts_over_time = []

    def add_entry(self, record: LogRecord):
        """Add new log entry"""
        self.entries.append(record)
        if record.level == 40:  # ERROR
            self.error_count += 1
        if "patterns" in record.extra:
            pid = record.extra["patterns"]["pattern_id"]
            self.pattern_counts[pid] += 1
        if "metrics" in record.extra:
            if pt := record.extra["metrics"].get("processing_time"):
                self.processing_times.append(pt)
        self.counts_over_time.append(1)
        if len(self.counts_over_time) > self.window_size:
            self.counts_over_time.pop(0)
        if len(self.entries) > self.window_size:
            self._trim_window()

    def _trim_window(self):
        """Trim aggregation window by removing excess entries and updating counts."""
        excess = len(self.entries) - self.window_size
        if excess <= 0:
            return

        # Get entries to remove
        removed = self.entries[:excess]
        self.entries = self.entries[excess:]

        # Process each removed entry
        for record in removed:
            self._update_counts_for_removed_record(record)

    def _update_counts_for_removed_record(self, record):
        """Update counts when a record is removed from the window.

        Args:
            record: The log record being removed
        """
        # Update error count
        if record.level == 40:  # ERROR
            self.error_count -= 1

        # Update pattern counts
        self._update_pattern_counts(record)

        # Update processing times
        self._update_processing_times(record)

        # Update time-based counts
        if self.counts_over_time:
            self.counts_over_time.pop(0)

    def _update_pattern_counts(self, record):
        """Update pattern counts for a removed record.

        Args:
            record: The log record being removed
        """
        if "patterns" not in record.extra:
            return

        pid = record.extra["patterns"]["pattern_id"]
        self.pattern_counts[pid] -= 1
        if self.pattern_counts[pid] == 0:
            del self.pattern_counts[pid]

    def _update_processing_times(self, record):
        """Update processing times for a removed record.

        Args:
            record: The log record being removed
        """
        if "metrics" not in record.extra:
            return

        pt = record.extra["metrics"].get("processing_time")
        if pt and pt in self.processing_times:
            self.processing_times.remove(pt)

    def get_aggregations(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        return {
            "entries": self.entries.copy(),
            "timestamps": [e.timestamp for e in self.entries],
            "counts_over_time": self.counts_over_time.copy(),
            "counts": len(self.entries),
            "error_rate": self.error_count / len(self.entries) if self.entries else 0,
            "patterns": dict(self.pattern_counts),
            "avg_processing_time": (
                float(np.mean(self.processing_times)) if self.processing_times else 0.0
            ),
            "correlations": self._analyze_correlations(),
        }


def _analyze_correlations(self) -> Dict[str, Any]:
    """Analyze correlations between log attributes"""
    if not self.entries:
        return {}
    df = pd.DataFrame(
        [
            {
                "level": e.getLevelName(),
                "pattern": (
                    e.extra["patterns"]["pattern_id"] if "patterns" in e.extra else -1
                ),
                "has_correlation": 1 if "correlations" in e.extra else 0,
            }
            for e in self.entries
        ]
    )
    return {} if df.empty else df.corr().to_dict(orient="dict")


class LogAggregator:
    """Real-time log aggregation and analysis"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.entries: List[LogRecord] = []
        self.pattern_counts = defaultdict(int)
        self.error_count = 0
        self.processing_times: List[float] = []
        self.counts_over_time = []

    def add_entry(self, record: LogRecord):
        """Add new log entry"""
        self.entries.append(record)

        # Update counts
        if record.level == _logging.ERROR:
            self.error_count += 1

        if "patterns" in record.extra:
            pid = record.extra["patterns"].get("pattern_id")
            if pid is not None:
                self.pattern_counts[pid] += 1

        if "metrics" in record.extra:
            pt = record.extra["metrics"].get("processing_time")
            if pt is not None:
                self.processing_times.append(pt)

        # Update counts over time for timeline visualization
        self.counts_over_time.append(1)  # Increment count
        if len(self.counts_over_time) > self.window_size:
            self.counts_over_time.pop(0)

        # Maintain window size
        if len(self.entries) > self.window_size:
            self._trim_window()

    def _trim_window(self):
        """Trim aggregation window by removing excess entries and updating counts."""
        excess = len(self.entries) - self.window_size
        if excess <= 0:
            return

        # Get entries to remove
        removed = self.entries[:excess]
        self.entries = self.entries[excess:]

        # Process each removed entry
        for record in removed:
            self._update_counts_for_removed_record(record)

    def _update_counts_for_removed_record(self, record):
        """Update counts when a record is removed from the window.

        Args:
            record: The log record being removed
        """
        # Update error count
        if record.level == _logging.ERROR:
            self.error_count -= 1

        # Update pattern counts
        self._update_pattern_counts(record)

        # Update processing times
        self._update_processing_times(record)

        # Update time-based counts
        if self.counts_over_time:
            self.counts_over_time.pop(0)

    def _update_pattern_counts(self, record):
        """Update pattern counts for a removed record.

        Args:
            record: The log record being removed
        """
        if "patterns" not in record.extra:
            return

        pid = record.extra["patterns"].get("pattern_id")
        if pid is not None:
            self.pattern_counts[pid] -= 1
            if self.pattern_counts[pid] == 0:
                del self.pattern_counts[pid]

    def _update_processing_times(self, record):
        """Update processing times for a removed record.

        Args:
            record: The log record being removed
        """
        if "metrics" not in record.extra:
            return

        pt = record.extra["metrics"].get("processing_time")
        if pt in self.processing_times:
            self.processing_times.remove(pt)

    def get_aggregations(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        return {
            "entries": self.entries.copy(),
            "timestamps": [e.timestamp for e in self.entries],
            "counts_over_time": self.counts_over_time.copy(),
            "counts": len(self.entries),
            "error_rate": self.error_count / len(self.entries) if self.entries else 0,
            "patterns": dict(self.pattern_counts),
            "avg_processing_time": (
                float(np.mean(self.processing_times)) if self.processing_times else 0.0
            ),
            "correlations": self._analyze_correlations(),
        }

    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between log attributes"""
        if not self.entries:
            return {}

        # Create correlation matrix from log entries
        df = self._create_correlation_dataframe()

        # Return empty dict if no data, otherwise return correlation matrix
        return {} if df.empty else df.corr().to_dict(orient="dict")

    def _create_correlation_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame for correlation analysis from log entries.

        Returns:
            pd.DataFrame: DataFrame with log attributes for correlation analysis
        """
        data = []

        for entry in self.entries:
            # Extract entry data into a dictionary
            entry_data = {
                "level": entry.getLevelName(),
                "pattern": self._extract_pattern_id(entry),
                "has_correlation": self._has_correlation(entry),
            }
            data.append(entry_data)

        return pd.DataFrame(data)

    def _extract_pattern_id(self, entry) -> int:
        """Extract pattern ID from a log entry.

        Args:
            entry: Log entry to extract pattern ID from

        Returns:
            int: Pattern ID or -1 if not found
        """
        # Early returns for missing data
        if "patterns" not in entry.extra:
            return -1

        patterns = entry.extra["patterns"]
        return patterns.get("pattern_id", -1)

    def _has_correlation(self, entry) -> int:
        """Check if entry has correlation ID.

        Args:
            entry: Log entry to check

        Returns:
            int: 1 if has correlation, 0 otherwise
        """
        return 1 if entry.extra.get("correlation_id") else 0


class ProjectAnalyzer:
    """Analyze Python project structure and dependencies."""

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.logger = _logging.getLogger(__name__)

    def initialize_project(self) -> None:
        """Initialize project analysis."""
        self.logger.info(f"Initializing project at {self.project_path}")

    def analyze_project(self) -> Dict[str, Any]:
        """Analyze project structure and dependencies.

        Returns:
            Dict[str, Any]: Analysis results
        """
        self.logger.info(f"Analyzing project at {self.project_path}")
        return {"status": "success"}

    def fix_project(self, mode: str = "safe") -> Dict[str, Any]:
        """Fix project issues.

        Args:
            mode (str): Fix mode - 'safe' or 'aggressive'

        Returns:
            Dict[str, Any]: Fix results
        """
        self.logger.info(f"Fixing project at {self.project_path} (mode: {mode})")
        return {"status": "success"}


# --- Logging Functions with Decorator ---


# Configure the basic logger
def basic_config(
    level: str = "INFO",
    format: str = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
):
    """
    Configures the basic settings for the logging system.

    Args:
        level (str): The logging level as a string (e.g., "DEBUG", "INFO").
        format (str): The format string for log messages.
        datefmt (str): The date format string.
    """
    numeric_level = getattr(_logging, level.upper(), _logging.INFO)
    _logging.basicConfig(level=numeric_level, format=format, datefmt=datefmt)


# Get logger instance
def get_logger_instance(name: str) -> _logging.Logger:
    """
    Retrieves a logger instance with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    return _logging.getLogger(name)


# Get level name
def get_level_name(level: int) -> str:
    """
    Retrieves the textual representation of a logging level.

    Args:
        level (int): The logging level as an integer.

    Returns:
        str: The name of the logging level.
    """
    return _logging.getLevelName(level)


def _get_frame_info() -> tuple:
    """
    Retrieves the name of the caller function and the line number from which the logging function was called.

    Returns:
        tuple: A tuple containing the function name and line number. Returns (None, None) if unavailable.
    """
    frame = inspect.currentframe()
    if frame is not None:
        caller_frame = frame.f_back
        if caller_frame is not None:
            function_name = caller_frame.f_code.co_name
            line_number = caller_frame.f_lineno
            return function_name, line_number
    return None, None


def log(level: int, msg: str, function_name: str, line_number: int, **kwargs):
    """
    Handles the logging logic, including creating a LogRecord and correlating it.

    Args:
        level (int): The logging level.
        msg (str): The log message.
        function_name (str): Name of the caller function.
        line_number (int): Line number in the caller function.
        **kwargs: Additional keyword arguments for logging.
    """
    logger = get_logger_instance(__name__)
    logger.log(level, msg, extra=kwargs)

    # Create a LogRecord instance
    record = LogRecord(
        level=level,
        message=msg,
        timestamp=datetime.now(timezone.utc).isoformat() + "Z",
        module=__name__,
        function=function_name,
        line=line_number,
        extra=kwargs,
    )

    # Correlate the log record
    log_correlator.correlate(record)


def log_decorator(level: int):
    """
    A decorator factory that creates decorators for logging functions.

    Args:
        level (int): The logging level to be used by the decorated function.

    Returns:
        function: The decorator.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(msg: str, **kwargs):
            function_name, line_number = _get_frame_info()
            log(level, msg, function_name, line_number, **kwargs)

        return wrapper

    return decorator


@log_decorator(_logging.DEBUG)
def debug(msg: str, **kwargs):
    """
    Logs a debug-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging


@log_decorator(_logging.INFO)
def info(msg: str, **kwargs):
    """
    Logs an info-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging


@log_decorator(_logging.WARNING)
def warning(msg: str, **kwargs):
    """
    Logs a warning-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging


@log_decorator(_logging.ERROR)
def error(msg: str, **kwargs):
    """
    Logs an error-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging


@log_decorator(_logging.CRITICAL)
def critical(msg: str, **kwargs):
    """
    Logs a critical-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging
