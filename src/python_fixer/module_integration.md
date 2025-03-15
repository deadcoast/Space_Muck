# Python Fixer - Module Integration

This document outlines the integration points between the different modules in the Python Fixer system, ensuring proper dependency management and avoiding circular dependencies.

## Module Hierarchy

The Python Fixer system follows a hierarchical dependency structure:

```
utils (base utilities)
  ↑
core (core functionality)
  ↑
enhancers, parsers, analyzers (specialized components)
  ↑
fixers (application-specific functionality)
```

This hierarchy ensures that:
1. Lower-level modules don't depend on higher-level modules
2. Modules at the same level don't depend on each other (except in special cases)
3. Dependencies flow in one direction, preventing circular dependencies

## Module Responsibilities

### utils
- Provides base utilities used by all other modules
- Contains logging, metrics collection, and context management
- No dependencies on other Python Fixer modules

### core
- Implements core functionality and data structures
- Depends only on utils
- Provides base classes and interfaces for other modules

### enhancers
- Extends core functionality with specialized enhancements
- Depends on core and utils
- Provides event system and enhancement capabilities

### parsers
- Implements parsing capabilities for project structure
- Depends on core and utils
- Provides header and project mapping capabilities

### analyzers
- Implements code analysis capabilities
- Depends on core and utils
- Provides import and code structure analysis

### fixers
- Implements specific code fixing capabilities
- Depends on core, enhancers, parsers, analyzers, and utils
- Provides high-level fixing functionality for end users

## Integration Points

### fixers ↔ core
- `FixManager` extends core's base functionality
- `SmartFixManager` uses core's `SmartFixer` for advanced fixing capabilities
- Transformers use core's AST manipulation capabilities

### fixers ↔ enhancers
- `FixManager` uses the `EnhancementSystem` for method enhancement
- `SmartFixManager` integrates with the event system for fix notifications
- Transformers use enhancement capabilities for advanced code transformations

### fixers ↔ parsers
- `FixManager` uses `HeaderMapParser` to understand module structure
- `SmartFixManager` uses `ProjectMapParser` for project-wide fixes
- Transformers use parsing capabilities to understand code structure

### fixers ↔ analyzers
- `FixManager` uses analyzers to detect code issues
- `SmartFixManager` uses analyzers for advanced code analysis
- Transformers use analyzers to understand code structure before transformation

### fixers ↔ utils
- All fixers components use the logging system for consistent logging
- Metrics collection is used to track fix performance and success rates
- Context management is used for proper error handling and resource management

## Optional Dependencies

Each module handles optional dependencies using the standard pattern:

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

## Testing Integration

Integration tests verify that modules work together properly:

1. **Core + Utils Tests**: Verify base functionality works correctly
2. **Enhancers + Core Tests**: Verify enhancement system integrates with core
3. **Parsers + Core Tests**: Verify parsing capabilities work with core
4. **Analyzers + Core Tests**: Verify analysis capabilities work with core
5. **Fixers + All Tests**: Verify fixers integrate with all other modules

## Best Practices

1. **Follow the Hierarchy**: Always respect the module hierarchy
2. **Avoid Circular Dependencies**: Never create circular dependencies between modules
3. **Use Optional Dependencies**: Handle third-party dependencies as optional when possible
4. **Document Integration Points**: Always document how modules integrate
5. **Test Integration**: Write tests that verify proper integration
6. **Consistent Logging**: Use the logging system consistently across all modules
7. **Error Handling**: Implement proper error handling in all public methods
8. **Type Hints**: Provide proper type hints for all public methods
9. **Maximum 200 Lines**: Keep each file under 200 lines, splitting functionality if needed
10. **One Component Type Per File**: Each component type gets its own dedicated file
