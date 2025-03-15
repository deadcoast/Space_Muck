# Standard library imports

# Third-party library imports

from typing import Any, Dict

# Local application imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from variant_loggers import LogRecord

console = Console()


class ConsoleFormatter:
    """Rich console output formatter"""

    def format(self, record: LogRecord) -> str:
        message_panel = Panel(
            self._format_message(record),
            title=self._get_level_text(record.getLevelName()),
            style=self._get_level_style(record.getLevelName()),
        )

        # Add context if available
        if record.extra:
            context_table = self._create_context_table(record.extra)
            return f"{message_panel}\n{context_table}"

        return str(message_panel)

    def _format_message(self, record: LogRecord) -> str:
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
