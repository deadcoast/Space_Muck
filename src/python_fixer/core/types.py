"""Type definitions and constants for the python_fixer package."""

import contextlib
import importlib.util
from typing import Any, Dict, Optional

# Dictionary to store optional dependencies
OPTIONAL_DEPS: Dict[str, Any] = {}

# Import optional dependencies
def import_optional_dep(name: str) -> Optional[Any]:
    """Import an optional dependency.

    Args:
        name: Name of the dependency to import

    Returns:
        The imported module or None if not available
    """
    with contextlib.suppress(ImportError):
        if importlib.util.find_spec(name) is not None:
            return importlib.import_module(name)
    return None

# Initialize optional dependencies
OPTIONAL_DEPS.update({
    "libcst": import_optional_dep("libcst"),
    "networkx": import_optional_dep("networkx"),
    "matplotlib": import_optional_dep("matplotlib.pyplot"),
    "mypy": import_optional_dep("mypy.api"),
    "rope": import_optional_dep("rope"),
})
