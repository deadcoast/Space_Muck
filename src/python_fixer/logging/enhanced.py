# Standard library imports
import json
import logging
import os
import sys
import threading
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL, FileHandler, StreamHandler, LogRecord, getLevelName
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, NamedTuple

# We'll import analyzer and formatter classes inside methods to avoid circular imports
# This helps maintain modularity while preventing import cycles


# JSONFormatter has been moved to python_fixer.formatters.json


class StructuredLogger:
    """Enhanced logger with structured output and convenience methods"""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str, log_level: str = "INFO") -> logging.Logger:
        """Get a configured logger instance.

        Args:
            name: Name of the logger
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Returns:
            logging.Logger: Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]

        # Create logger
        logger = logging.getLogger(name)
        level = getattr(logging, log_level.upper())
        logger.setLevel(level)

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create console handler with JSON formatter
        console_handler = logging.StreamHandler(sys.stdout)
        from python_fixer.formatters.json import JsonFormatter
        console_handler.setFormatter(JsonFormatter())
        logger.addHandler(console_handler)

        # Store logger in cache
        cls._loggers[name] = logger

        return logger

    @classmethod
    def setup_file_logging(cls, log_dir: Path, name: str = "python_fixer") -> None:
        """Set up file logging with enhanced formatting capabilities.

        Args:
            log_dir: Directory to store log files
            name: Base name for log files
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"

        # Get root logger
        root_logger = logging.getLogger()
        
        # Create formatters - use enhanced formatter if possible
        from python_fixer.formatters.json import JsonFormatter
        json_formatter = JsonFormatter()
        
        from contextlib import suppress
        
        # Try to use the enhanced formatter if the output directory exists
        with suppress(Exception):
            # Import here to avoid circular imports
            from python_fixer.formatters.formatter import EnhancedFormatter
            enhanced_formatter = EnhancedFormatter(output_dir=log_dir)
            # Add console handler with enhanced formatter for rich output
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(enhanced_formatter)
            root_logger.addHandler(console_handler)

        # Add file handler with JSON formatter for structured logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)

    def __init__(self, name: str, log_file: Optional[Path] = None, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.name = name
        self.log_dir: Optional[Path] = Path("./logs")
        self.level: int = INFO
        self.enable_metrics: bool = True
        self.enable_ml: bool = True
        # Use string type annotations to avoid circular imports
        self.pattern_analyzer: Optional['PatternAnalyzer'] = None
        self.correlator: Optional['Correlator'] = None
        self.aggregator: Optional['Aggregator'] = None
        self.handlers: Optional[List[Any]] = None
        self.report_path: Optional[Path] = None
        self.json_formatter: Optional['JsonFormatter'] = None
        # Use string type annotations to avoid circular imports
        self.enhanced_formatter: Optional['EnhancedFormatter'] = None
        self.console_handler: Optional[StreamHandler] = None
        self.file_handler: Optional[FileHandler] = None
        self.console_formatter: Optional['ConsoleFormatter'] = None
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.log_dir / "report.html"

        # Initialize formatters
        from python_fixer.formatters.json import JsonFormatter
        self.json_formatter = JsonFormatter()
        
        # Initialize console formatter for human-readable output
        from python_fixer.formatters.console import ConsoleFormatter
        self.console_formatter = ConsoleFormatter()
        
        # Initialize enhanced formatter if output directory exists
        if self.log_dir.exists():
            # Import here to avoid circular imports
            from python_fixer.formatters.formatter import EnhancedFormatter
            self.enhanced_formatter = EnhancedFormatter(output_dir=self.log_dir)

        # Setup console handler with ConsoleFormatter for human-readable output
        self.console_handler = StreamHandler(stream=sys.stdout)
        self.console_handler.setFormatter(self.console_formatter)
        self.console_handler.setLevel(self.level)
        
        # Setup structured console handler with JsonFormatter for machine-readable output
        self.structured_console_handler = StreamHandler(stream=sys.stderr)
        self.structured_console_handler.setFormatter(self.json_formatter)
        self.structured_console_handler.setLevel(self.level)

        # Setup file handler
        self.file_handler = FileHandler(filename=str(self.log_dir / f"{self.name}.log"))
        self.file_handler.setFormatter(self.json_formatter)
        self.file_handler.setLevel(self.level)

        # Configure base logger with all handlers
        self.logger.setLevel(self.level)
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.structured_console_handler)
        self.logger.addHandler(self.file_handler)

        # Initialize analysis components
        # Import here to avoid circular imports
        from python_fixer.analyzers.pattern import PatternAnalyzer
        from python_fixer.analyzers.correlator import Correlator
        from python_fixer.analyzers.aggregator import Aggregator
        
        self.pattern_analyzer = self.pattern_analyzer or PatternAnalyzer()
        self.correlator = self.correlator or Correlator()
        self.aggregator = self.aggregator or Aggregator()

        # Add additional handlers if provided
        if self.handlers:
            for handler in self.handlers:
                self.logger.addHandler(handler)

        # Initialize console formatter for report generation
        from python_fixer.formatters.console import ConsoleFormatter
        self.console_formatter = ConsoleFormatter()

        # Add enhanced formatter handler if available
        if self.enhanced_formatter:
            enhanced_console_handler = logging.StreamHandler(sys.stdout)
            enhanced_console_handler.setFormatter(self.enhanced_formatter)
            self.logger.addHandler(enhanced_console_handler)

        # File handler if requested
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self.json_formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message with optional extra data"""
        self.logger.info(message, extra=extra, **kwargs)

    def error(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Log error message with optional exception info"""
        self.logger.error(message, exc_info=exc_info, extra=extra, **kwargs)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=extra, **kwargs)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=extra, **kwargs)

    def file_operation(
        self, file_path: Path, operation: str, details: Optional[str] = None
    ):
        """Log file operation with details"""
        file_info = {
            "path": str(file_path),
            "operation": operation,
            "size": file_path.stat().st_size if file_path.exists() else None,
        }
        if details:
            file_info["details"] = details

        self.info(
            f"File operation: {operation} on {file_path}",
            extra={"file_info": file_info},
        )

    def is_enabled_for(self, level: int) -> bool:
        """Check if logger is enabled for specified level

        Args:
            level: The logging level to check

        Returns:
            bool: True if the logger is enabled for the specified level
        """
        return self.logger.isEnabledFor(level)


    def _create_context(self, **kwargs) -> Dict[str, Any]:
        """Create a structured context for the log entry."""
        return {
            "timestamp": datetime.now(timezone.utc),
            "module": kwargs.get("module", "unknown"),
            "function": kwargs.get("function", "unknown"),
            "line": kwargs.get("line", 0),
            "extra": kwargs.get("extra", {}),
            "process_id": os.getpid(),
            "thread_id": threading.get_ident()
        }

    def _level_str_to_num(self, level_str: str) -> int:
        """Convert log level from string to numeric value."""
        level_map = {
            "DEBUG": DEBUG,
            "INFO": INFO,
            "WARNING": WARNING,
            "ERROR": ERROR,
            "CRITICAL": CRITICAL,
        }
        return level_map.get(level_str.upper(), INFO)

    def _update_metrics(self, record: LogRecord):
        """Update relevant metrics based on the log record."""
        if self.enable_metrics and self.aggregator:
            self.aggregator.add_entry(record)

    async def _analyze_patterns(self, record: LogRecord) -> Any:
        """Analyze log message patterns."""
        if self.enable_ml and self.pattern_analyzer:
            return await self.pattern_analyzer.analyze(record.message)
        return None

    def _write_entry(self, record: LogRecord):
        """Emit the log record to configured handlers."""
        log_message = (
            f"{record.created} - {logging.getLevelName(record.levelno)} - "
            f"{record.module}.{record.funcName}:{record.lineno} - {record.getMessage()}"
        )
        extra = getattr(record, 'extra', {}) or {}
        self.logger.log(record.levelno, log_message, extra=extra)

    async def _update_visualizations(self):
        """Update visual dashboards with the latest log information."""
        # The EnhancedFormatter already handles visualization updates
        # through its internal visualization_engine when available
        await asyncio.sleep(0)  # Avoid blocking

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return uuid.uuid4().hex

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return uuid.uuid4().hex

    # Define SpanContext as a NamedTuple for simplicity
    class SpanContext(NamedTuple):
        process_id: int
        thread_id: int
        is_remote: bool
        span_id: str
        trace_id: str

    # Define Span class for tracing
    class Span:
        def __init__(self, name: str, context: 'StructuredLogger.SpanContext'):
            self.name = name
            self.context = context
            self.start_time = datetime.now()
            self.end_time = None

    @asynccontextmanager
    async def start_as_current_span(
        self, name: str, context: Optional['StructuredLogger.SpanContext'] = None
    ) -> 'StructuredLogger.Span':
        """Start a new tracing span as the current span."""
        if context is None:
            context = self.SpanContext(
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
                is_remote=False,
                span_id=self._generate_span_id(),
                trace_id=self._generate_trace_id(),
            )
        span = self.Span(name=name, context=context)
        # Start span (simplified without external tracer)
        self.logger.info(f"Starting span: {name}")
        try:
            yield span
        finally:
            # End span (simplified without external tracer)
            self.logger.info(f"Ending span: {name}")

    async def log(self, level: Union[int, str], message: str, **kwargs):
        """Enhanced asynchronous logging with context and analysis."""
        # Create structured context
        context = self._create_context(**kwargs)

        # Convert log level if necessary
        level_num = self._level_str_to_num(level) if isinstance(level, str) else level

        # Create log record
        record = LogRecord(
            level=level_num,
            message=message,
            timestamp=context.timestamp.isoformat(),
            module=context.module,
            function=context.function,
            line=context.line,
            extra=context.extra.copy(),
        )

        # Start tracing span
        async with self.start_as_current_span("log_entry") as span:
            # Set tracing attributes
            span.set_attribute("log.module", record.module)
            span.set_attribute("log.function", record.function)
            span.set_attribute("log.line", record.line)
            span.set_attribute("log.message", record.message)
            span.set_attribute("log.level", logging.getLevelName(record.level))

            # Update metrics
            if self.enable_metrics:
                self._update_metrics(record)

            # Analyze patterns
            if self.enable_ml:
                patterns = await self._analyze_patterns(record)
                if patterns:
                    record.extra["patterns"] = patterns

            # Handle correlations
            if self.correlator:
                if correlations := self.correlator.correlate(record):
                    record.extra["correlations"] = [r.to_dict() for r in correlations]

            # Write the log entry
            self._write_entry(record)

            # Update visualizations
            await self._update_visualizations()

    async def generate_report(self, format: str = "html") -> Path:
        """Generate a report based on aggregated log data."""
        if not self.aggregator:
            raise ValueError("Aggregator is not initialized")
            
        # Use EnhancedFormatter's visualization if available
        if self.enhanced_formatter:
            # The EnhancedFormatter has more advanced visualization capabilities
            report_path = self.report_path
            if format.lower() == "json":
                report_path = report_path.with_suffix(".json")
                
            # Delegate to the EnhancedFormatter's report generation
            from contextlib import suppress
            
            # Import here to avoid circular imports
            with suppress(AttributeError, NotImplementedError):
                # We need to check if the method exists at runtime
                if hasattr(self.enhanced_formatter, "generate_report"):
                    return await self.enhanced_formatter.generate_report(report_path, format)
            # If we get here, the enhanced report generation failed, so fall back to default
        
        # Default implementation
        if format.lower() == "html":
            report_content = self._generate_html_report()
        elif format.lower() == "json":
            report_content = self._generate_json_report()
        else:
            raise ValueError(f"Unsupported report format: {format}")

        # Write report to file
        report_path = self.report_path
        if format.lower() == "json":
            report_path = report_path.with_suffix(".json")

        # Use regular file operations instead of aiofiles
        with open(report_path, "w") as f:
            f.write(report_content)
        await asyncio.sleep(0)  # Yield control to event loop

        return report_path

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        agg_data = self.aggregator.get_aggregations()

        html_content = [
            "<html><head><title>Log Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style></head><body>",
            f"<h1>{self.name} Log Report</h1>",
            "<h2>Summary</h2>",
            f"<p>Total Logs: {agg_data['counts']}</p>",
            f"<p>Error Rate: {agg_data['error_rate']:.2%}</p>",
            "<h2>Log Entries</h2>",
            "<table><tr><th>Timestamp</th><th>Level</th><th>Message</th></tr>",
        ]

        html_content.extend(
            f"<tr><td>{entry.timestamp}</td><td>{getLevelName(entry.level)}</td><td>{entry.message}</td></tr>"
            for entry in agg_data["entries"]
        )
        html_content.extend(["</table></body></html>"])
        return "\n".join(html_content)

    def _generate_json_report(self) -> str:
        """Generate JSON report content."""
        agg_data = self.aggregator.get_aggregations()
        report_data = {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_logs": agg_data["counts"],
                "error_rate": agg_data["error_rate"],
                "patterns": agg_data["patterns"],
            },
            "entries": [entry.to_dict() for entry in agg_data["entries"]],
        }
        return json.dumps(report_data, indent=2)

    def close(self):
        """Close all handlers associated with the logger."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


if __name__ == "__main__":

    async def main():
        # Example usage
        log_dir = Path("./logs")
        logger = StructuredLogger(
            name="MyAppLogger", log_dir=log_dir, enable_metrics=True, enable_ml=True
        )

        # Log some test messages
        await logger.log(
            INFO,
            "Application started",
            module="app",
            function="main",
            line=1,
            extra={"tags": ["startup"]},
        )

        await logger.log(
            ERROR,
            "Test error message",
            module="app",
            function="process",
            line=2,
            extra={"correlation_id": "test-123"},
        )

        # Generate and save report
        report_path = await logger.generate_report("html")
        print(f"Report generated at: {report_path}")

        # Clean up
        logger.close()

    # Run the example
    if sys.platform != "win32":
        asyncio.run(main())
    else:
        # Windows-specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
