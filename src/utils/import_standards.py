#!/usr/bin/env python3
"""
Import Standards for Space Muck.

This module defines the standard import ordering and patterns to be used
throughout the Space Muck codebase.

Standard import order:
1. Standard library imports (alphabetically sorted)
2. Third-party library imports (alphabetically sorted)
3. Local application imports (alphabetically sorted by module)

Example:
```python
# Standard library imports

# Third-party library imports

try:
    import scipy.ndimage as ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Local application imports

```

Guidelines:
1. Always use absolute imports (starting with 'src.') for local modules
2. Group imports by category with a blank line between categories
3. Sort imports alphabetically within each category
4. Use try/except for optional dependencies
5. Define constants for dependency availability (e.g., SCIPY_AVAILABLE)
6. Import specific classes/functions rather than entire modules when possible
7. Avoid circular imports by using type hints and forward references
"""

# ========================================================================
# EXAMPLE IMPORTS BELOW - These demonstrate the correct style but are not used
# These are intentionally marked as noqa to suppress linter warnings
# ========================================================================

# Standard library imports
import abc  # noqa: F401 - Example import
import logging  # noqa: F401 - Example import
import math  # noqa: F401 - Example import
import random  # noqa: F401 - Example import
from typing import (  # noqa: F401 - Example imports
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# Third-party library imports
import numpy as np  # noqa: F401 - Example import

# Local application imports - specific imports are preferred over wildcard imports
from config import COLOR_BG, COLOR_FG, GAME_TITLE  # noqa: F401 - Example import
from entities.base_entity import BaseEntity  # noqa: F401 - Example import
from utils.dependency_injection import inject  # noqa: F401 - Example import
from utils.noise_generator import (  # noqa: F401 - Example import
    NoiseGenerator,
    get_noise_generator,
)

# ========================================================================
# END OF EXAMPLE IMPORTS
# ========================================================================

# Standard import categories
STDLIB_MODULES = [
    "abc",
    "argparse",
    "collections",
    "copy",
    "datetime",
    "functools",
    "inspect",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "random",
    "re",
    "sys",
    "time",
    "uuid",
]

# Common third-party libraries
THIRD_PARTY_MODULES = [
    "numpy",
    "scipy",
    "perlin_noise",
    "matplotlib",
    "pandas",
    "pytest",
]

# Optional dependencies and their fallbacks
OPTIONAL_DEPENDENCIES = {
    "perlin_noise": {
        "import_statement": "from perlin_noise import PerlinNoise",
        "availability_flag": "PERLIN_AVAILABLE",
        "fallback_message": "PerlinNoise package is not available. Using fallback noise generator.",
    },
    "scipy": {
        "import_statement": "import scipy.ndimage as ndimage",
        "availability_flag": "SCIPY_AVAILABLE",
        "fallback_message": "scipy not available, using fallback implementation.",
    },
}


def generate_standard_imports(
    stdlib_imports: List[str],
    third_party_imports: List[str],
    local_imports: List[str],
    optional_imports: List[str] = None,
) -> str:
    """
    Generate standardized import statements.

    Args:
        stdlib_imports: List of standard library imports
        third_party_imports: List of third-party library imports
        local_imports: List of local application imports
        optional_imports: List of optional dependencies to handle with try/except

    Returns:
        String containing formatted import statements
    """
    import_str = "# Standard library imports\n"
    if stdlib_imports:
        import_str += "\n".join(sorted(stdlib_imports)) + "\n"

    import_str += "\n# Third-party library imports\n"
    if third_party_imports:
        import_str += "\n".join(sorted(third_party_imports)) + "\n"

    if optional_imports:
        import_str += "\n# Optional dependencies\n"
        for opt_import in optional_imports:
            if opt_import in OPTIONAL_DEPENDENCIES:
                dep_info = OPTIONAL_DEPENDENCIES[opt_import]
                import_str += f"try:\n    {dep_info['import_statement']}\n"
                import_str += f"    {dep_info['availability_flag']} = True\n"
                import_str += "except ImportError:\n"
                import_str += f"    {dep_info['availability_flag']} = False\n"
                import_str += f'    print("{dep_info["fallback_message"]}")\n\n'

    import_str += "\n# Local application imports\n"
    if local_imports:
        import_str += "\n".join(sorted(local_imports)) + "\n"

    return import_str
