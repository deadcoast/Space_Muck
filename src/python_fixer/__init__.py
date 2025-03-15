"""Python Import Fixer - Advanced Python import and dependency analyzer/fixer."""

# Standard library imports
import importlib.util
from typing import TYPE_CHECKING, List, Dict, Any, Optional

# Third-party library imports

# Local application imports
# Core components
from .core import SmartFixer, FixOperation, FixStrategy, SignalManager
from .core import CodeSignature, SignatureComponent, TypeInfo, SignatureMetrics
from .core import TypeCheckable, TypeCheckResult, validate_type, validate_protocol, ImportInfo

# Enhancers
from .enhancers import EnhancementSystem, EventSystem, EventType, Event

# Parsers
from .parsers import HeaderMapParser, ProjectMapParser

# Utils
from .utils import LogContext, LogMetrics, MetricsCollector

# Existing imports
from .analyzers.project_analysis import ProjectAnalyzer
from .logging.enhanced import StructuredLogger
from .fixers import FixManager, SmartFixManager, PatchHandler

__version__ = "0.1.0"
__author__ = "Codeium"

# Expose main classes for easy import
__all__ = [
    # Core components
    "SmartFixer",
    "FixOperation",
    "FixStrategy",
    "SignalManager",
    "CodeSignature",
    "SignatureComponent",
    "TypeInfo",
    "SignatureMetrics",
    "TypeCheckable",
    "TypeCheckResult",
    "validate_type",
    "validate_protocol",
    "ImportInfo",
    
    # Enhancers
    "EnhancementSystem",
    "EventSystem",
    "EventType",
    "Event",
    
    # Parsers
    "HeaderMapParser",
    "ProjectMapParser",
    
    # Utils
    "LogContext",
    "LogMetrics",
    "MetricsCollector",
    
    # Analyzers
    "ProjectAnalyzer",
    
    # Logging
    "StructuredLogger",
    
    # Fixers
    "FixManager",
    "SmartFixManager",
    "PatchHandler",
]
