"""Type definitions and constants for the python_fixer package."""

import contextlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
from typeguard import typechecked

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

def check_dependency_available(name: str) -> bool:
    """Check if a dependency is available without importing it.
    
    Args:
        name: Name of the dependency to check
        
    Returns:
        True if the dependency is available, False otherwise
    """
    return importlib.util.find_spec(name) is not None

# Initialize optional dependencies
OPTIONAL_DEPS.update({
    "libcst": import_optional_dep("libcst"),
    "networkx": import_optional_dep("networkx"),
    "matplotlib": import_optional_dep("matplotlib.pyplot"),
    "mypy": import_optional_dep("mypy.api"),
    "rope": import_optional_dep("rope"),
    "typeguard": import_optional_dep("typeguard"),
})

@runtime_checkable
class TypeCheckable(Protocol):
    """Protocol for objects that support runtime type checking."""
    __annotations__: Dict[str, Any]

@dataclass
class TypeCheckResult:
    """Result of a type check operation.

    Attributes:
        is_valid: Whether the type check passed
        errors: List of type check errors if any
        context: Optional context information about where the error occurred
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    context: Optional[str] = None

@typechecked
def validate_type(value: Any, expected_type: Any, context: Optional[str] = None) -> TypeCheckResult:
    """Validate that a value matches an expected type.

    Args:
        value: The value to check
        expected_type: The expected type annotation
        context: Optional context information for error messages

    Returns:
        TypeCheckResult with validation status and any errors
    """
    try:
        # Import internal typeguard functions for type checking
        from typeguard._checkers import check_type_internal, TypeCheckError

        # Convert expected_type to string representation for error messages
        type_str = _get_type_str(expected_type)
        if hasattr(expected_type, '__origin__'):
            origin = expected_type.__origin__
            args = expected_type.__args__
            if origin is Union:
                type_str = f"Union[{', '.join(f'<class \'{arg.__name__}\'' for arg in args)}]"
            elif origin is list:
                type_str = f"<class 'list[{', '.join(str(args[0]))}]'>"
            else:
                type_str = f"<class '{origin.__name__}[{', '.join(str(args[0]))}]'>"
        elif hasattr(expected_type, '__name__'):
            type_str = f"<class '{expected_type.__name__}'>"

        # Perform type check
        try:
            check_type_internal(value, expected_type, {})
            return TypeCheckResult(is_valid=True)
        except TypeCheckError:
            error_msg = f"expected {type_str}, got {type(value).__name__}"
            if context:
                error_msg = f"{context}: {error_msg}"
            return TypeCheckResult(is_valid=False, errors=[error_msg], context=context)
    except Exception as e:
        error_msg = str(e)
        if context:
            error_msg = f"{context}: {error_msg}"
        return TypeCheckResult(is_valid=False, errors=[error_msg], context=context)


@typechecked
def _get_type_str(type_obj: Any) -> str:
    """Get a string representation of a type object.

    Args:
        type_obj: The type object to convert to string

    Returns:
        String representation of the type
    """
    if hasattr(type_obj, "__origin__"):
        origin = type_obj.__origin__
        args = type_obj.__args__

        if origin is Union:
            type_strs = []
            for arg in args:
                type_str = _get_type_str(arg)
                type_strs.append(type_str)
            return f"Union[{', '.join(type_strs)}]"
        elif origin is list:
            if len(args) == 1:
                inner_type = _get_type_str(args[0])
                return f"<class 'list[{inner_type}]'>"
            return f"<class 'list[{', '.join(_get_type_str(arg) for arg in args)}]'>"
        else:
            if hasattr(origin, "__module__") and origin.__module__ != "builtins":
                return f"<class '{origin.__module__}.{origin.__name__}'>"
            return f"<class '{origin.__name__}'>"

    if hasattr(type_obj, "__module__") and type_obj.__module__ != "builtins":
        return f"<class '{type_obj.__module__}.{type_obj.__name__}'>"
    return f"<class '{type_obj.__name__}'>"


@typechecked
def validate_protocol(value: Any, protocol_type: Any) -> TypeCheckResult:
    """Validate that a value implements a protocol correctly.

    Args:
        value: The value to check
        protocol_type: The protocol class to check against

    Returns:
        TypeCheckResult with validation status and any errors
    """
    # Initialize variables before conditional blocks
    missing_methods: List[str] = []
    incorrect_types: List[str] = []
    error_messages: List[str] = []
    value_method: Optional[Any] = None
    value_annotations: Dict[str, Any] = {}
    
    try:
        # First check if the value implements the protocol
        if not isinstance(value, protocol_type):
            # Try to get a helpful error message about why it doesn't match
            for name, method in protocol_type.__dict__.items():
                if name.startswith('_') or not hasattr(method, '__annotations__'):
                    continue
                    
                if not hasattr(value, name):
                    missing_methods.append(name)
                    continue
                    
                value_method = getattr(value, name)
                if not callable(value_method):
                    incorrect_types.append(
                        f"{name} (expected callable, got {type(value_method).__name__})"
                    )
                    continue
                    
                # Check method signature
                value_annotations = getattr(value_method, '__annotations__', {})
                for param_name, expected_type in method.__annotations__.items():
                    if param_name == 'return':
                        continue
                        
                    # Handle missing annotations
                    if param_name not in value_annotations:
                        incorrect_types.append(
                            f"{name}.{param_name} (missing type annotation)"
                        )
                        continue
                        
                    # Handle type mismatches
                    if value_annotations[param_name] != expected_type:
                        type_str = _get_type_str(expected_type)
                        value_type_str = _get_type_str(value_annotations[param_name])
                        incorrect_types.append(
                            f"{name}.{param_name} (expected {type_str}, got {value_type_str})"
                        )
                
                # Check return type
                if 'return' in method.__annotations__:
                    expected_return = method.__annotations__['return']
                    if 'return' not in value_annotations:
                        incorrect_types.append(f"{name} return (missing type annotation)")
                        continue
                    
                    # Handle return type mismatch
                    if value_annotations['return'] != expected_return:
                        type_str = _get_type_str(expected_return)
                        value_return_str = _get_type_str(value_annotations['return'])
                        incorrect_types.append(
                            f"{name} return (expected {type_str}, got {value_return_str})"
                        )
            
            # Build error messages
            if missing_methods:
                error_messages.append(
                    f"Missing required methods: {', '.join(missing_methods)}"
                )
            if incorrect_types:
                error_messages.append(
                    f"Type mismatches: {', '.join(incorrect_types)}"
                )
            
            return TypeCheckResult(
                is_valid=False,
                errors=[f"Object does not implement protocol {protocol_type.__name__}:"] + error_messages
            )
        
        # Check if all required methods have correct signatures
        for name, method in protocol_type.__dict__.items():
            if name.startswith('_') or not hasattr(method, '__annotations__'):
                continue
            
            value_method = getattr(value, name)
            value_annotations = getattr(value_method, '__annotations__', {})
            
            # Check method signature
            for param_name, expected_type in method.__annotations__.items():
                if param_name == 'return':
                    continue
                
                if param_name not in value_annotations:
                    return TypeCheckResult(
                        is_valid=False,
                        errors=[f"Missing type annotation for {name}.{param_name}"]
                    )
                
                if value_annotations[param_name] != expected_type:
                    type_str = _get_type_str(expected_type)
                    value_type_str = _get_type_str(value_annotations[param_name])
                    return TypeCheckResult(
                        is_valid=False,
                        errors=[f"Type mismatch in {name}.{param_name}: expected {type_str}, got {value_type_str}"]
                    )
            
            # Check return type
            if 'return' in method.__annotations__:
                expected_return = method.__annotations__['return']
                if 'return' not in value_annotations:
                    return TypeCheckResult(
                        is_valid=False,
                        errors=[f"Missing return type annotation for {name}"]
                    )
                elif value_annotations['return'] != expected_return:
                    type_str = _get_type_str(expected_return)
                    value_return_str = _get_type_str(value_annotations['return'])
                    return TypeCheckResult(
                        is_valid=False,
                        errors=[f"Return type mismatch in {name}: expected {type_str}, got {value_return_str}"]
                    )
        
        # All checks passed
        return TypeCheckResult(is_valid=True)
    except Exception as e:
        return TypeCheckResult(
            is_valid=False,
            errors=[f"Error validating protocol implementation: {str(e)}"]
        )

@dataclass
class ImportInfo:
    """Information about a Python import statement.
    
    Attributes:
        module: The module being imported (e.g., 'os.path' in 'from os.path import join')
        imported_names: List of names being imported (e.g., ['join', 'dirname'])
        is_relative: Whether this is a relative import (starts with dots)
        level: Number of dots in relative import (0 for absolute imports)
        is_valid: Whether the import is valid (can be resolved)
        error_message: Error message if the import is invalid
    """
    module: Optional[str]
    imported_names: List[str] = field(default_factory=list)
    is_relative: bool = False
    level: int = 0
    is_valid: bool = True
    error_message: Optional[str] = None
    
    def validate(self, package_path: Optional[str] = None) -> bool:
        """Validate that this import can be resolved.
        
        Args:
            package_path: Optional package path for resolving relative imports
            
        Returns:
            True if the import is valid, False otherwise
        """
        if not self.module and not self.imported_names:
            self.is_valid = False
            self.error_message = "Import has no module or imported names"
            return False
            
        try:
            if self.is_relative:
                if not package_path:
                    self.is_valid = False
                    self.error_message = "Cannot resolve relative import without package path"
                    return False
                    
                # Calculate the parent package for relative imports
                parts = package_path.split('.')
                if self.level > len(parts):
                    self.is_valid = False
                    self.error_message = f"Relative import level {self.level} exceeds package depth {len(parts)}"
                    return False
                    
                if self.level == len(parts):
                    parent_pkg = ""
                else:
                    parent_pkg = '.'.join(parts[:-self.level])
                    
                # Construct the full module name
                if self.module:
                    if parent_pkg:
                        full_module = f"{parent_pkg}.{self.module}"
                    else:
                        full_module = self.module
                else:
                    full_module = parent_pkg
                    
                # Check if the module exists
                if not check_dependency_available(full_module):
                    self.is_valid = False
                    self.error_message = f"Cannot resolve relative import: {full_module}"
                    return False
            else:
                # For absolute imports, just check if the module exists
                if self.module and not check_dependency_available(self.module):
                    self.is_valid = False
                    self.error_message = f"Cannot resolve absolute import: {self.module}"
                    return False
                    
            self.is_valid = True
            self.error_message = None
            return True
        except Exception as e:
            self.is_valid = False
            self.error_message = f"Error validating import: {str(e)}"
            return False
            
    def get_full_import_path(self, package_path: Optional[str] = None) -> Optional[str]:
        """Get the full import path for this import.
        
        Args:
            package_path: Optional package path for resolving relative imports
            
        Returns:
            Full import path or None if it cannot be resolved
        """
        if not self.validate(package_path):
            return None
            
        if self.is_relative:
            if not package_path:
                return None
                
            # Calculate the parent package for relative imports
            parts = package_path.split('.')
            if self.level > len(parts):
                return None
                
            if self.level == len(parts):
                parent_pkg = ""
            else:
                parent_pkg = '.'.join(parts[:-self.level])
                
            # Construct the full module name
            if self.module:
                if parent_pkg:
                    return f"{parent_pkg}.{self.module}"
                else:
                    return self.module
            else:
                return parent_pkg
        else:
            return self.module
