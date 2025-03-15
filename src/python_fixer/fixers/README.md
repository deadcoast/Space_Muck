# Python Fixer - Fixers Module

## Overview

The `fixers` module provides a comprehensive framework for detecting and fixing common Python code issues. It's designed with modularity, separation of concerns, and optional integration in mind, allowing users to leverage specific fix capabilities as needed.

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

## Best Practices

1. **Incremental Changes**: Apply fixes incrementally, testing after each step
2. **Modularity**: Keep each fix focused on a single responsibility
3. **Documentation**: Document all fixes and their expected behavior
4. **Error Handling**: Implement robust error handling in all fixes
5. **Testing**: Write comprehensive tests for all fixes

## Contributing

When adding new fixes to the system:

1. Create a new class extending `Fix` or `BaseTransformer`
2. Implement the required methods (`detect`, `apply`, etc.)
3. Add tests for the new fix
4. Update documentation to reflect the new functionality
5. Expose the new fix in the appropriate `__init__.py` file
