"""
Logging configuration for Space Muck.

This module provides advanced logging capabilities including:
- Console logging with colored output
- File logging with rotation
- Performance tracking
- Exception handling with detailed traceback
- Memory usage monitoring
"""

import logging
import logging.handlers
import os
import sys
import time
import traceback
import gc
from datetime import datetime
from typing import Optional, Dict, Any

# Try to import colorama for cross-platform colored console output
try:
    import colorama
    from colorama import Fore, Back, Style

    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class MemoryHandler(logging.Handler):
    """Custom handler that keeps recent log records in memory for in-game display."""

    def __init__(self, capacity: int = 100):
        """Initialize with a maximum capacity of records to store."""
        super().__init__()
        self.capacity = capacity
        self.records = []

    def emit(self, record):
        """Store the record in memory, respecting capacity limits."""
        self.records.append(self.format(record))
        if len(self.records) > self.capacity:
            self.records.pop(0)

    def get_records(self):
        """Return all stored records."""
        return self.records


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to console output."""

    def __init__(self, fmt: str):
        """Initialize with the specified format string."""
        super().__init__(fmt)

        # Define color mappings for different log levels
        self.colors = {
            "DEBUG": Fore.CYAN if COLORAMA_AVAILABLE else "",
            "INFO": Fore.GREEN if COLORAMA_AVAILABLE else "",
            "WARNING": Fore.YELLOW if COLORAMA_AVAILABLE else "",
            "ERROR": Fore.RED if COLORAMA_AVAILABLE else "",
            "CRITICAL": Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else "",
            "RESET": Style.RESET_ALL if COLORAMA_AVAILABLE else "",
        }

    def format(self, record):
        """Format the record with appropriate colors."""
        levelname = record.levelname
        message = super().format(record)

        if COLORAMA_AVAILABLE:
            return f"{self.colors.get(levelname, '')}{message}{self.colors['RESET']}"
        return message


class PerformanceLogFilter(logging.Filter):
    """Filter that adds performance metrics to log records."""

    def __init__(self):
        """Initialize with starting time and memory usage."""
        super().__init__()
        self.start_time = time.time()
        self.last_time = self.start_time

    def filter(self, record):
        """Add performance attributes to the record."""
        current_time = time.time()

        # Add elapsed time since program start
        record.elapsed = current_time - self.start_time

        # Add time since last log
        record.since_last = current_time - self.last_time
        self.last_time = current_time

        # Try to get memory usage information
        try:
            import psutil

            process = psutil.Process(os.getpid())
            record.memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except (ImportError, AttributeError):
            record.memory = 0

        return True


def setup_logging(
    log_to_file: bool = True, log_level: int = logging.INFO
) -> logging.Logger:
    """
    Configure the logging system for the game with advanced features.

    Args:
        log_to_file: Whether to save logs to file
        log_level: Minimum log level to record

    Returns:
        Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_to_file and not os.path.exists("logs"):
        os.makedirs("logs")

    # Get the logger
    logger = logging.getLogger("space_muck")
    logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter and add it to the handler
    console_format = "[%(levelname)s] %(message)s"
    if COLORAMA_AVAILABLE:
        console_formatter = ColoredFormatter(console_format)
    else:
        console_formatter = logging.Formatter(console_format)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file:
        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = f"logs/space_muck_{timestamp}.log"

        # Create file handler which logs even debug messages
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)

        # Create more detailed formatter for file logs
        file_format = (
            "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
        )
        if log_level <= logging.DEBUG:
            file_format += " (elapsed: %(elapsed).3f, memory: %(memory).2f MB)"

        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Log the startup event
        logger.info(f"Logging started, saving to {log_file}")

    # Create in-memory handler for showing logs in-game
    memory_handler = MemoryHandler(capacity=100)
    memory_handler.setLevel(logging.INFO)
    memory_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S")
    )
    logger.addHandler(memory_handler)

    # Add performance filter if in debug mode
    if log_level <= logging.DEBUG:
        perf_filter = PerformanceLogFilter()
        console_handler.addFilter(perf_filter)
        if log_to_file:
            file_handler.addFilter(perf_filter)

    return logger


def log_exception(e: Exception, critical: bool = True) -> None:
    """
    Log an exception with full traceback and context information.

    Args:
        e: The exception to log
        critical: Whether to log as CRITICAL (True) or ERROR (False)
    """
    # Get the logger
    logger = logging.getLogger("space_muck")

    # Prepare exception details
    exc_type = type(e).__name__
    exc_message = str(e)
    exc_traceback = traceback.format_exc()

    # Log the exception with appropriate level
    if critical:
        logger.critical(f"Unhandled {exc_type}: {exc_message}")
        logger.critical(f"Traceback:\n{exc_traceback}")
    else:
        logger.error(f"{exc_type}: {exc_message}")
        logger.error(f"Traceback:\n{exc_traceback}")

    # Log memory stats in debug mode
    if logger.level <= logging.DEBUG:
        # Try to get detailed memory info
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory = process.memory_info().rss / (1024 * 1024)
            logger.debug(f"Memory usage at exception: {memory:.2f} MB")

            # Run garbage collection and log again
            gc.collect()
            memory_after = process.memory_info().rss / (1024 * 1024)
            logger.debug(
                f"Memory after gc.collect(): {memory_after:.2f} MB, delta: {memory_after - memory:.2f} MB"
            )
        except ImportError:
            pass


def get_in_memory_logs() -> list:
    """
    Get logs stored in memory for in-game display.

    Returns:
        list: List of log messages
    """
    logger = logging.getLogger("space_muck")

    return next(
        (
            handler.get_records()
            for handler in logger.handlers
            if isinstance(handler, MemoryHandler)
        ),
        [],
    )


def log_performance_start(tag: str) -> float:
    """
    Start timing a code section.

    Args:
        tag: Identifier for the timing operation

    Returns:
        float: Start time
    """
    start_time = time.time()
    if logging.getLogger("space_muck").level <= logging.DEBUG:
        logging.debug(f"Performance '{tag}' started")
    return start_time


def log_performance_end(tag: str, start_time: float) -> float:
    """
    End timing a code section and log the duration.

    Args:
        tag: Identifier for the timing operation
        start_time: Start time from log_performance_start

    Returns:
        float: Duration in seconds
    """
    duration = time.time() - start_time
    if logging.getLogger("space_muck").level <= logging.DEBUG:
        logging.debug(f"Performance '{tag}' completed in {duration * 1000:.2f}ms")
    return duration


def log_memory_usage(tag: str = "Memory check") -> Optional[float]:
    """
    Log current memory usage.

    Args:
        tag: Description for the log entry

    Returns:
        Optional[float]: Memory usage in MB, or None if not available
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / (1024 * 1024)
        logging.debug(f"{tag}: {memory:.2f} MB")
        return memory
    except ImportError:
        logging.debug(f"{tag}: psutil not available")
        return None


class LogContext:
    """Context manager for tracking performance and logging exceptions."""

    def __init__(self, context_name: str, log_level: int = logging.DEBUG):
        """
        Initialize with a context name and log level.

        Args:
            context_name: Name for this context/operation
            log_level: Level to log at (e.g., logging.DEBUG)
        """
        self.context_name = context_name
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.time()
        if logging.getLogger("space_muck").level <= self.log_level:
            logging.log(self.log_level, f"{self.context_name} started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log timing and any exceptions when exiting the context."""
        duration = time.time() - self.start_time

        if exc_type is not None:
            # An exception occurred
            logging.error(
                f"{self.context_name} failed after {duration * 1000:.2f}ms: {exc_val}"
            )
            log_exception(exc_val, critical=False)
            return False  # Let the exception propagate

        if logging.getLogger("space_muck").level <= self.log_level:
            logging.log(
                self.log_level,
                f"{self.context_name} completed in {duration * 1000:.2f}ms",
            )

        return True
