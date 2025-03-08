from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Set

from rich import Console

cli = Console()


@dataclass
class LogContext:
    """Enhanced context for structured variant_loggers"""

    module: str
    function: str
    line: int
    timestamp: datetime
    process_id: int
    thread_id: int
    extra: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "function": self.function,
            "line": self.line,
            "timestamp": self.timestamp.isoformat(),
            "process_id": self.process_id,
            "thread_id": self.thread_id,
            "extra": self.extra,
            "metrics": self.metrics,
            "tags": list(self.tags),
            "correlation_id": self.correlation_id,
        }
