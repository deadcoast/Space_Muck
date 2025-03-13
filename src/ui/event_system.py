"""
Event system for Space Muck UI components.

This module provides a centralized event system for handling UI events,
metrics collection, and real-time updates.
"""

# Standard library imports
import time

# Third-party library imports

# Local application imports
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Any, Callable, Optional


class EventType(Enum):
    """Types of events that can be emitted."""

    CONVERTER_SELECTED = auto()
    PROCESS_STARTED = auto()
    PROCESS_COMPLETED = auto()
    PROCESS_CANCELLED = auto()
    CHAIN_CREATED = auto()
    CHAIN_MODIFIED = auto()
    CHAIN_DELETED = auto()
    EFFICIENCY_UPDATED = auto()
    METRIC_UPDATED = auto()


@dataclass
class EventData:
    """Container for event data."""

    type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: float = time.time()


class MetricType(Enum):
    """Types of metrics that can be tracked."""

    THROUGHPUT = auto()
    ENERGY_USAGE = auto()
    UTILIZATION = auto()
    EFFICIENCY = auto()
    PROCESS_COUNT = auto()


@dataclass
class MetricData:
    """Container for metric data."""

    type: MetricType
    value: float
    unit: str
    source: str
    timestamp: float = time.time()


class EventSystem:
    """Central event system for UI components."""

    def __init__(self):
        """Initialize the event system."""
        self.handlers: Dict[EventType, List[Callable[[EventData], None]]] = {}
        self.metric_handlers: Dict[MetricType, List[Callable[[MetricData], None]]] = {}
        self.metrics_cache: Dict[str, List[MetricData]] = {}
        self.cache_duration: float = 3600.0  # 1 hour

    def subscribe(
        self, event_type: EventType, handler: Callable[[EventData], None]
    ) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Function to call when event occurs
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def subscribe_metric(
        self, metric_type: MetricType, handler: Callable[[MetricData], None]
    ) -> None:
        """
        Subscribe to a metric type.

        Args:
            metric_type: Type of metric to subscribe to
            handler: Function to call when metric is updated
        """
        if metric_type not in self.metric_handlers:
            self.metric_handlers[metric_type] = []
        self.metric_handlers[metric_type].append(handler)

    def emit(self, event_type: EventType, source: str, data: Dict[str, Any]) -> None:
        """
        Emit an event.

        Args:
            event_type: Type of event
            source: Source of the event
            data: Event data
        """
        event = EventData(event_type, source, data)
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(event)

    def update_metric(
        self, metric_type: MetricType, value: float, unit: str, source: str
    ) -> None:
        """
        Update a metric value.

        Args:
            metric_type: Type of metric
            value: New value
            unit: Unit of measurement
            source: Source of the metric
        """
        metric = MetricData(metric_type, value, unit, source)

        # Cache the metric
        cache_key = f"{source}_{metric_type.name}"
        if cache_key not in self.metrics_cache:
            self.metrics_cache[cache_key] = []
        self.metrics_cache[cache_key].append(metric)

        # Notify handlers
        if metric_type in self.metric_handlers:
            for handler in self.metric_handlers[metric_type]:
                handler(metric)

        # Clean old cache entries
        self._clean_cache()

    def get_metric_history(
        self, metric_type: MetricType, source: str, duration: Optional[float] = None
    ) -> List[MetricData]:
        """
        Get historical metric data.

        Args:
            metric_type: Type of metric
            source: Source of the metric
            duration: Optional duration in seconds to limit history

        Returns:
            List of metric data points
        """
        cache_key = f"{source}_{metric_type.name}"
        if cache_key not in self.metrics_cache:
            return []

        if duration is None:
            return self.metrics_cache[cache_key]

        current_time = time.time()
        return [
            metric
            for metric in self.metrics_cache[cache_key]
            if current_time - metric.timestamp <= duration
        ]

    def _clean_cache(self) -> None:
        """Clean old entries from the metrics cache."""
        current_time = time.time()
        for key in list(self.metrics_cache.keys()):
            self.metrics_cache[key] = [
                metric
                for metric in self.metrics_cache[key]
                if current_time - metric.timestamp <= self.cache_duration
            ]
            if not self.metrics_cache[key]:
                del self.metrics_cache[key]
