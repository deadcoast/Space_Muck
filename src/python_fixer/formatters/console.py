# Standard library imports

# Third-party library imports

from typing import Any, Dict

# Local application imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import logging
from logging import LogRecord

console = Console()


class ConsoleFormatter:
    """Rich console output formatter"""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record into a rich console output string.
        
        Args:
            record: The log record to format
            
        Returns:
            A string representation of the formatted log record
        """
        # Create a panel for the message
        message_panel = Panel(
            self._format_message(record),
            title=self._get_level_text(logging.getLevelName(record.levelno)),
            style=self._get_level_style(logging.getLevelName(record.levelno)),
        )
        
        # Use StringIO to capture the rendered output as a string
        from io import StringIO
        string_io = StringIO()
        temp_console = Console(file=string_io, width=100)
        
        # Add context if available
        if hasattr(record, 'extra') and record.extra:
            context_table = self._create_context_table(record.extra)
            temp_console.print(message_panel)
            temp_console.print(context_table)
        else:
            temp_console.print(message_panel)
            
        return string_io.getvalue()

    def _format_message(self, record: logging.LogRecord) -> str:
        """Format the log message"""
        return f"{record.getMessage()}"

    def _get_level_text(self, levelname: str) -> str:
        """Get formatted level name"""
        return f"[bold]{levelname}[/bold]"

    def _get_level_style(self, levelname: str) -> str:
        """Get style based on log level"""
        return {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
        }.get(levelname, "white")

    def _create_context_table(self, context: Dict[str, Any]) -> Table:
        """Create rich table for context data"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Key", style="dim", width=12)
        table.add_column("Value", style="dim")

        for key, value in context.items():
            table.add_row(str(key), str(value))

        return table
