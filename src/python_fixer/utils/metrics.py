# Standard library imports
from collections import Counter

# Third-party library imports

# Local application imports
from altair import Dict
from dataclasses import dataclass, field
from prometheus_client import Gauge, Histogram


@dataclass
class LogMetrics:
    """Real-time metrics for log analysis"""

    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    import_fixes: int = 0
    files_processed: int = 0
    average_processing_time: float = 0.0
    pattern_frequencies: Dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """Collect and track metrics using Prometheus"""

    def __init__(self):
        self.error_counter = Counter("log_errors_total", "Total error count")
        self.warning_counter = Counter("log_warnings_total", "Total warning count")
        self.log_size = Gauge("log_size_bytes", "Total log size in bytes")
        self.processing_time = Histogram(
            "log_processing_seconds",
            "Time spent processing logs",
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
        )
