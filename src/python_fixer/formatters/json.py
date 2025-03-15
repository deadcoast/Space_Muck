"""
JSON formatter for structured logging.
"""

import json
import logging
from typing import Dict, Any


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after formatting the log record.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the formatter with specified keyword arguments.
        """
        super().__init__()
        self.kwargs = kwargs

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record as JSON.
        """
        log_data = self._get_record_data(record)
        return json.dumps(log_data)
    
    def _get_record_data(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Get a dictionary with all the log record attributes.
        """
        # Standard log record attributes
        data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
        }

        # Add any extra attributes
        if hasattr(record, 'extra') and record.extra:
            data |= record.extra

        # Include any custom kwargs passed to the formatter
        if self.kwargs:
            data.update(self.kwargs)

        return data
