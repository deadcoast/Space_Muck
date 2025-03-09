import inspect
import logging
from datetime import datetime, timezone
from functools import wraps
from logging import LogRecord, getLogger

from python_fixer.logging.correlator import log_correlator

# Configure the basic logger
def basicConfig(
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
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format=format, datefmt=datefmt)


# Get logger instance
def getLoggerInstance(name: str) -> logging.Logger:
    """
    Retrieves a logger instance with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    return getLogger(name)


# Get level name
def getLevelName(level: int) -> str:
    """
    Retrieves the textual representation of a logging level.

    Args:
        level (int): The logging level as an integer.

    Returns:
        str: The name of the logging level.
    """
    return logging.getLevelName(level)


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
    logger = getLoggerInstance(__name__)
    logger.log(level, msg, extra=kwargs)

    # Create a LogRecord instance
    record = LogRecord(
        level=level,
        message=msg,
        timestamp=f"{datetime.now(timezone.utc).isoformat()}Z",
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


@log_decorator(logging.DEBUG)
def debug(msg: str, **kwargs):
    """
    Logs a debug-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging


@log_decorator(logging.INFO)
def info(msg: str, **kwargs):
    """
    Logs an info-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging


@log_decorator(logging.WARNING)
def warning(msg: str, **kwargs):
    """
    Logs a warning-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging


@log_decorator(logging.ERROR)
def error(msg: str, **kwargs):
    """
    Logs an error-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging


@log_decorator(logging.CRITICAL)
def critical(msg: str, **kwargs):
    """
    Logs a critical-level message.

    Args:
        msg (str): The log message.
        **kwargs: Additional keyword arguments for logging.
    """
    pass  # The decorator handles the logging
