# Python Fixer - Fixers Module

## Overview

The `fixers` module provides a comprehensive framework for detecting and fixing common Python code issues. It's designed with modularity, separation of concerns, and optional integration in mind, allowing users to leverage specific fix capabilities as needed.

This module integrates with other components of the Python Fixer system, following the hierarchical dependency structure defined in the project architecture.

## Key Components

### FixManager

The `FixManager` class serves as the base manager for handling code fixes. It provides:

- Configuration loading and validation
- Basic fix application infrastructure
- Logging and reporting capabilities

```python
from python_fixer import FixManager

# Initialize with a configuration file
fix_manager = FixManager(config_path="fixes_config.json")

# Apply fixes to a specific file
fix_manager.apply_fixes("path/to/file.py")
```

### SmartFixManager

The `SmartFixManager` extends the base `FixManager` with enhanced capabilities:

- Automatic fix detection and application
- Interactive fixing with user prompts
- Integration with the enhanced analyzer for deeper code understanding

```python
from python_fixer import SmartFixManager, ProjectAnalyzer

# Initialize the analyzer
analyzer = ProjectAnalyzer("path/to/project")

# Initialize the smart fix manager
smart_fix = SmartFixManager()

# Apply automatic fixes
await smart_fix.automatic_fix(analyzer)

# Or use interactive mode
await smart_fix.interactive_fix(analyzer)
```

### Transformers

The module includes several transformers that modify code to fix specific issues:

- `BaseTransformer`: Abstract base class for all transformers
- `RelativeImportTransformer`: Converts relative imports to absolute imports
- `CircularDependencyTransformer`: Resolves circular import dependencies

```python
from python_fixer import RelativeImportTransformer

# Create a transformer
transformer = RelativeImportTransformer()

# Apply the transformation to a file
transformer.apply(analyzer)
```

### PatchHandler

The `PatchHandler` applies pre-defined patches to code files:

```python
from python_fixer import PatchHandler

# Initialize the patch handler
patch_handler = PatchHandler()

# Apply a specific fix to a file
await patch_handler.apply_fix("path/to/file.py", "fix_identifier")
```

## Integration with Python Fixer

The fixers module is fully integrated with the python_fixer system, allowing seamless use of all components. The main classes are exposed through the package's `__init__.py` file, so you can import them directly:

```python
from python_fixer import FixManager, SmartFixManager, PatchHandler
```

### Module Dependencies

The fixers module follows the project's hierarchical dependency structure:

```
utils (base utilities)
  ↑
core (core functionality)
  ↑
enhancers, parsers (specialized components)
  ↑
fixers (application-specific functionality)
```

### Integration Points

- **Core Module**: The fixers module uses the core module's `SmartFixer` for fundamental fixing operations and code analysis
- **Enhancers Module**: Extends fixing capabilities through method enhancement using the `EnhancementSystem`
- **Parsers Module**: Uses `HeaderMapParser` and `ProjectMapParser` to understand project structure and module relationships
- **Utils Module**: Depends on utilities for logging, metrics, and context management

### Optional Dependencies

The fixers module handles optional dependencies using the standard pattern defined in the project:

```python
import importlib.util
from typing import TYPE_CHECKING

# Check for optional dependencies
DEPENDENCY_AVAILABLE = importlib.util.find_spec("dependency_name") is not None

# For type checking only
if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        import dependency_name  # type: ignore

# Import optional dependencies at runtime
dependency_name = None  # Define at module level
if DEPENDENCY_AVAILABLE:
    with contextlib.suppress(ImportError):
        import dependency_name
```

## Extending with Custom Fixes

You can create custom fixes by extending the `Fix` class:

```python
from python_fixer.fixers import Fix

class MyCustomFix(Fix):
    """Custom fix implementation"""
    
    def __init__(self):
        super().__init__(
            id="my_custom_fix",
            description="Fixes a specific issue in the code"
        )
    
    def detect(self, code: str) -> bool:
        # Implement detection logic
        return "issue_pattern" in code
    
    def apply(self, code: str) -> str:
        # Implement fix logic
        return code.replace("issue_pattern", "fixed_pattern")
```

## Logging

The fixers module integrates with the python_fixer logging system through the `variant_loggers` bridge, ensuring consistent logging across the entire system:

```python
from python_fixer.logging import variant_loggers

# Log messages with context
variant_loggers.log_with_context(
    variant_loggers.INFO,
    "Applied fix to file",
    {"file": "path/to/file.py", "fix_id": "my_custom_fix"}
)
```

### Structured Logging

The module follows the project's structured logging approach, which provides:

1. **Consistent Format**: All log messages follow a consistent JSON format
2. **Context Enrichment**: Logs include context information like module, function, and line number
3. **Metrics Collection**: Integration with the metrics collection system
4. **Log Levels**: Support for standard log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

This ensures that all fixes and transformations are properly tracked and can be analyzed for effectiveness.

## Best Practices

1. **Incremental Changes**: Apply fixes incrementally, testing after each step
2. **Modularity**: Keep each fix focused on a single responsibility
3. **Documentation**: Document all fixes and their expected behavior
4. **Error Handling**: Implement robust error handling in all fixes
5. **Testing**: Write comprehensive tests for all fixes
6. **Optional Dependencies**: Follow the project's pattern for handling optional dependencies
7. **No Circular Dependencies**: Ensure no circular dependencies are introduced
8. **Consistent Logging**: Use the structured logging system consistently
9. **Type Hints**: Provide proper type hints for all public methods
10. **Maximum 200 Lines**: Keep each file under 200 lines, splitting functionality if needed

## Contributing

When adding new fixes to the system:

1. Create a new class extending `Fix` or `BaseTransformer`
2. Implement the required methods (`detect`, `apply`, etc.)
3. Add tests for the new fix
4. Update documentation to reflect the new functionality
5. Expose the new fix in the appropriate `__init__.py` file
6. Update integration tests to verify proper integration with other modules
7. Document integration points in the module_integration.md file
8. Verify no circular dependencies are introduced
9. Follow the project's coding standards for docstrings, type hints, and error handling
10. Update the System_Scratchpad.md file to reflect your changes
