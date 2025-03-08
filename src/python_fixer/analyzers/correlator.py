from typing import Any, Dict, List

# Assuming LogRecord and CustomFormatter are defined elsewhere in your codebase
from variant_loggers import LogRecord


class Correlator:
    """Correlate related log entries"""

    def __init__(self, related_event: str, max_distance: int = 5):
        self.max_distance = max_distance
        self.recent_entries: List[LogRecord] = []
        self.related_event = related_event

    def correlate(self, record: LogRecord) -> List[LogRecord]:
        """Find correlations with recent entries"""
        correlations = [
            recent
            for recent in self.recent_entries[-self.max_distance:]
            if self._are_related(record, recent)
        ]
        self.recent_entries.append(record)
        if len(self.recent_entries) > self.max_distance * 2:
            self.recent_entries = self.recent_entries[-self.max_distance:]
        return correlations

    def _are_related(self, record1: LogRecord, record2: LogRecord) -> bool:
        """Determine if two records are related"""
        cid1 = record1.extra.get("correlation_id")
        cid2 = record2.extra.get("correlation_id")
        if cid1 and cid1 == cid2:
            return True
        tags1 = set(record1.extra.get("tags", []))
        tags2 = set(record2.extra.get("tags", []))
        if tags1 & tags2:
            return True
        p1 = record1.extra.get("patterns", {}).get("pattern_id")
        p2 = record2.extra.get("patterns", {}).get("pattern_id")
        return p1 is not None and p2 is not None and p1 == p2

    def to_dict(self) -> Dict[str, Any]:
        """Convert the correlation to a dictionary."""
        return {"related_event": self.related_event}


class LogCorrelator:
    """Correlate related log entries."""

    def __init__(self, max_distance: int = 5):
        self.max_distance = max_distance
        self.recent_entries: List[LogRecord] = []

    def correlate(self, record: LogRecord) -> List[LogRecord]:
        """
        Find correlations with recent entries.

        Args:
            record (LogRecord): The current log record to correlate.

        Returns:
            List[LogRecord]: A list of related log records.
        """
        correlations = [
            recent
            for recent in self.recent_entries[-self.max_distance :]
            if self._are_related(record, recent)
        ]
        # Add current record to recent entries
        self.recent_entries.append(record)
        if len(self.recent_entries) > self.max_distance * 2:
            self.recent_entries = self.recent_entries[-self.max_distance :]

        return correlations

    def _are_related(self, record1: LogRecord, record2: LogRecord) -> bool:
        """
        Determine if two records are related.

        Args:
            record1 (LogRecord): The first log record.
            record2 (LogRecord): The second log record.

        Returns:
            bool: True if related, False otherwise.
        """
        # Check correlation ID
        cid1 = record1.extra.get("correlation_id")
        cid2 = record2.extra.get("correlation_id")
        if cid1 and cid1 == cid2:
            return True

        # Check for shared tags
        tags1 = set(record1.extra.get("tags", []))
        tags2 = set(record2.extra.get("tags", []))
        if tags1 & tags2:  # If they share any tags
            return True

        # Check for similar patterns
        p1 = record1.extra.get("patterns", {}).get("pattern_id")
        p2 = record2.extra.get("patterns", {}).get("pattern_id")
        return p1 is not None and p2 is not None and p1 == p2


