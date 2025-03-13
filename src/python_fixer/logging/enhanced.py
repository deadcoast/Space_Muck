# Standard library imports
from datetime import datetime, timezone
from logging import INFO
import json
import os
import sys
import uuid

# Third-party library imports

# Local application imports
from formatters.json import JsonFormatter
from variant_loggers import variant_loggers, Path
from contextlib import asynccontextmanager
from logger_refactor.logger_analysis import PatternAnalyzer, Correlator, Aggregator
from logger_refactor.logger_record import (
    Span,
    SpanContext,
    Path,
    Console,
    tracer,
    Any,
    List,
    Optional,
    Union,
    aiofiles,
    asyncio,
    threading,
    StreamHandler,
    FileHandler,
    DEBUG,
    WARNING,
    ERROR,
    CRITICAL,
    LogRecord,
    getLevelName,
)


class EnhancedLogger:
    """Enhanced logging system with advanced features."""

    def __init__(
        self,
        name: str = "EnhancedLogger",
        log_dir: Optional[Path] = None,
        level: int = INFO,
        enable_metrics: bool = True,
        enable_ml: bool = True,
        pattern_analyzer: Optional[PatternAnalyzer] = None,
        correlator: Optional[Correlator] = None,
        aggregator: Optional[Aggregator] = None,
        handlers: Optional[List[Any]] = None,
    ):
        """Initialize the EnhancedLogger."""
        self.name = name
        self.log_dir = log_dir or Path("./logs")
        self.level = level
        self.enable_metrics = enable_metrics
        self.enable_ml = enable_ml

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.log_dir / "report.html"

        # Initialize formatter with JSON config
        self.formatter = JsonFormatter(
            fmt=json.dumps(
                {
                    "format": "{timestamp} - {level} - {module}.{function}:{line} - {message}",
                    "format_rules": [
                        {
                            "level": "ERROR",
                            "format": "{timestamp} - {level} - {message}",
                        },
                        {
                            "level": "INFO",
                            "format": "{timestamp} - {level} - {module}.{function} - {message}",
                        },
                    ],
                    "is_active": True,
                }
            )
        )

        # Setup console handler
        self.console_handler = StreamHandler(stream=sys.stdout)
        self.console_handler.setFormatter(self.formatter)
        self.console_handler.setLevel(self.level)

        # Setup file handler
        self.file_handler = FileHandler(filename=str(self.log_dir / f"{self.name}.log"))
        self.file_handler.setFormatter(self.formatter)
        self.file_handler.setLevel(self.level)

        # Configure base logger
        self.logger = variant_loggers.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)

        # Initialize analysis components
        self.pattern_analyzer = pattern_analyzer or PatternAnalyzer()
        self.correlator = correlator or Correlator()
        self.aggregator = aggregator or Aggregator()

        # Add additional handlers if provided
        if handlers:
            for handler in handlers:
                self.logger.addHandler(handler)

        # Initialize console for report generation
        self.console = Console()

    def _create_context(self, **kwargs) -> variant_loggers.LogContext:
        """Create a structured context for the log entry."""
        return variant_loggers.LogContext(
            timestamp=datetime.now(timezone.utc),
            module=kwargs.get("module", "unknown"),
            function=kwargs.get("function", "unknown"),
            line=kwargs.get("line", 0),
            extra=kwargs.get("extra", {}),
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
        )

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
            f"{record.timestamp} - {variant_loggers.getLevelName(record.level)} - "
            f"{record.module}.{record.function}:{record.line} - {record.message}"
        )
        extra = record.extra or {}
        self.logger.log(record.level, log_message, extra=extra)

    async def _update_visualizations(self, record: LogRecord):
        """Update visual dashboards with the latest log information."""
        # Placeholder for visualization updates
        await asyncio.sleep(0)  # Avoid blocking

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return uuid.uuid4().hex

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return uuid.uuid4().hex

    @asynccontextmanager
    async def start_as_current_span(
        self, name: str, context: Optional[SpanContext] = None
    ) -> Span:
        """Start a new tracing span as the current span."""
        if context is None:
            context = SpanContext(
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
                is_remote=False,
                span_id=self._generate_span_id(),
                trace_id=self._generate_trace_id(),
            )
        span = Span(name=name, context=context)
        tracer.start_span(span)
        try:
            yield span
        finally:
            tracer.end_span(span)

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
            span.set_attribute("log.level", variant_loggers.getLevelName(record.level))

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
            await self._update_visualizations(record)

    async def generate_report(self, format: str = "html") -> Path:
        """Generate a report based on aggregated log data."""
        if not self.aggregator:
            raise ValueError("Aggregator is not initialized")

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

        async with aiofiles.open(report_path, "w") as f:
            await f.write(report_content)

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
        logger = EnhancedLogger(
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
