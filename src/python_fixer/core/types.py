"""Type definitions and constants for the python_fixer package."""

# Standard library imports

# Third-party library imports

import contextlib
import importlib.util

# Local application imports
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
OPTIONAL_DEPS.update(
    {
        "libcst": import_optional_dep("libcst"),
        "networkx": import_optional_dep("networkx"),
        "matplotlib": import_optional_dep("matplotlib.pyplot"),
        "mypy": import_optional_dep("mypy.api"),
        "rope": import_optional_dep("rope"),
        "typeguard": import_optional_dep("typeguard"),
    }
)


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
def validate_type(
    value: Any, expected_type: Any, context: Optional[str] = None
) -> TypeCheckResult:
    """Validate that a value matches an expected type.

    Args:
        value: The value to check
        expected_type: The expected type annotation
        context: Optional context information for error messages

    Returns:
        TypeCheckResult with validation status and any errors
    """
    try:
        # Get string representation of the expected type
        type_str = _get_type_representation(expected_type)

        # Perform the actual type check
        return _perform_type_check(value, expected_type, type_str, context)
    except Exception as e:
        return _create_exception_result(e, context)


def _get_type_representation(expected_type: Any) -> str:
    """Get a string representation of a type for error messages.

    Args:
        expected_type: The type to represent as a string

    Returns:
        String representation of the type
    """
    type_str = _get_type_str(expected_type)

    if hasattr(expected_type, "__origin__"):
        return _get_generic_type_representation(expected_type)
    elif hasattr(expected_type, "__name__"):
        return f"<class '{expected_type.__name__}'>"

    return type_str


def _get_generic_type_representation(expected_type: Any) -> str:
    """Get a string representation of a generic type.

    Args:
        expected_type: The generic type to represent as a string

    Returns:
        String representation of the generic type
    """
    origin = expected_type.__origin__
    args = expected_type.__args__

    if origin is Union:
        return f"Union[{', '.join(f"<class '{arg.__name__}'" for arg in args)}]"
    elif origin is list:
        return f"<class 'list[{', '.join(str(args[0]))}]'>"
    else:
        return f"<class '{origin.__name__}[{', '.join(str(args[0]))}]'>"


def _perform_type_check(
    value: Any, expected_type: Any, type_str: str, context: Optional[str] = None
) -> TypeCheckResult:
    """Perform the actual type check using typeguard.

    Args:
        value: The value to check
        expected_type: The expected type
        type_str: String representation of the expected type
        context: Optional context information for error messages

    Returns:
        TypeCheckResult with validation status and any errors
    """
    try:
        from typeguard._checkers import TypeCheckError, check_type_internal

        check_type_internal(value, expected_type, {})
        return TypeCheckResult(is_valid=True)
    except TypeCheckError:
        error_msg = f"expected {type_str}, got {type(value).__name__}"
        if context:
            error_msg = f"{context}: {error_msg}"
        return TypeCheckResult(is_valid=False, errors=[error_msg], context=context)


def _create_exception_result(
    exception: Exception, context: Optional[str] = None
) -> TypeCheckResult:
    """Create a TypeCheckResult for an exception during type checking.

    Args:
        exception: The exception that occurred
        context: Optional context information for error messages

    Returns:
        TypeCheckResult with validation status and error message
    """
    error_msg = str(exception)
    if context:
        error_msg = f"{context}: {error_msg}"
    return TypeCheckResult(is_valid=False, errors=[error_msg], context=context)


@typechecked
def _format_union_type(args: tuple) -> str:
    """Format a Union type into a string representation.

    Args:
        args: The type arguments of the Union

    Returns:
        String representation of the Union type
    """
    type_strs = [_get_type_str(arg) for arg in args]
    return f"Union[{', '.join(type_strs)}]"


@typechecked
def _format_list_type(args: tuple) -> str:
    """Format a list type into a string representation.

    Args:
        args: The type arguments of the list

    Returns:
        String representation of the list type
    """
    if len(args) == 1:
        inner_type = _get_type_str(args[0])
        return f"<class 'list[{inner_type}]'>"
    return f"<class 'list[{', '.join(_get_type_str(arg) for arg in args)}]'>"


@typechecked
def _format_class_type(cls: Any) -> str:
    """Format a class type into a string representation.

    Args:
        cls: The class to format

    Returns:
        String representation of the class
    """
    if hasattr(cls, "__module__") and cls.__module__ != "builtins":
        return f"<class '{cls.__module__}.{cls.__name__}'>"
    return f"<class '{cls.__name__}'>"


@typechecked
def _get_type_str(type_obj: Any) -> str:
    """Get a string representation of a type object.

    Args:
        type_obj: The type object to convert to string

    Returns:
        String representation of the type
    """
    # Handle parameterized types (generics)
    if hasattr(type_obj, "__origin__"):
        origin = type_obj.__origin__
        args = type_obj.__args__

        if origin is Union:
            return _format_union_type(args)
        elif origin is list:
            return _format_list_type(args)
        else:
            return _format_class_type(origin)

    # Handle regular classes
    return _format_class_type(type_obj)


@typechecked
def validate_protocol(value: Any, protocol_type: Any) -> TypeCheckResult:
    """Validate that a value implements a protocol correctly.

    Args:
        value: The value to check
        protocol_type: The protocol class to check against

    Returns:
        TypeCheckResult with validation status and any errors
    """
    try:
        # First check if the value implements the protocol
        if not isinstance(value, protocol_type):
            return _check_protocol_compatibility(value, protocol_type)

        # Check if all required methods have correct signatures
        return _validate_protocol_methods(value, protocol_type)
    except Exception as e:
        return TypeCheckResult(
            is_valid=False,
            errors=[f"Error validating protocol implementation: {str(e)}"],
        )


def _check_protocol_compatibility(value: Any, protocol_type: Any) -> TypeCheckResult:
    """Check why a value doesn't implement a protocol and provide detailed error messages.

    Args:
        value: The value to check
        protocol_type: The protocol class to check against

    Returns:
        TypeCheckResult with validation status and detailed error messages
    """
    missing_methods: List[str] = []
    incorrect_types: List[str] = []
    error_messages: List[str] = []

    # Analyze each method in the protocol
    for name, method in protocol_type.__dict__.items():
        if name.startswith("_") or not hasattr(method, "__annotations__"):
            continue

        # Check if method exists
        if not hasattr(value, name):
            missing_methods.append(name)
            continue

        # Check if method is callable
        value_method = getattr(value, name)
        if not callable(value_method):
            incorrect_types.append(
                f"{name} (expected callable, got {type(value_method).__name__})"
            )
            continue

        # Check method signature
        _check_method_signature_compatibility(
            name, method, value_method, incorrect_types
        )

    # Build error messages
    if missing_methods:
        error_messages.append(f"Missing required methods: {', '.join(missing_methods)}")
    if incorrect_types:
        error_messages.append(f"Type mismatches: {', '.join(incorrect_types)}")

    return TypeCheckResult(
        is_valid=False,
        errors=[f"Object does not implement protocol {protocol_type.__name__}:"]
        + error_messages,
    )


def _check_method_signature_compatibility(
    name: str, method: Any, value_method: Any, incorrect_types: List[str]
) -> None:
    """Check if a method's signature is compatible with a protocol method.

    Args:
        name: The method name
        method: The protocol method
        value_method: The implementation method
        incorrect_types: List to collect type mismatch errors
    """
    value_annotations = getattr(value_method, "__annotations__", {})

    # Check parameter types
    for param_name, expected_type in method.__annotations__.items():
        if param_name == "return":
            continue

        # Handle missing annotations
        if param_name not in value_annotations:
            incorrect_types.append(f"{name}.{param_name} (missing type annotation)")
            continue

        # Handle type mismatches
        if value_annotations[param_name] != expected_type:
            type_str = _get_type_str(expected_type)
            value_type_str = _get_type_str(value_annotations[param_name])
            incorrect_types.append(
                f"{name}.{param_name} (expected {type_str}, got {value_type_str})"
            )

    # Check return type
    _check_return_type_compatibility(name, method, value_annotations, incorrect_types)


def _check_return_type_compatibility(
    name: str,
    method: Any,
    value_annotations: Dict[str, Any],
    incorrect_types: List[str],
) -> None:
    """Check if a method's return type is compatible with a protocol method.

    Args:
        name: The method name
        method: The protocol method
        value_annotations: The implementation method's annotations
        incorrect_types: List to collect type mismatch errors
    """
    if "return" in method.__annotations__:
        expected_return = method.__annotations__["return"]
        if "return" not in value_annotations:
            incorrect_types.append(f"{name} return (missing type annotation)")
            return

        # Handle return type mismatch
        if value_annotations["return"] != expected_return:
            type_str = _get_type_str(expected_return)
            value_return_str = _get_type_str(value_annotations["return"])
            incorrect_types.append(
                f"{name} return (expected {type_str}, got {value_return_str})"
            )


def _validate_protocol_methods(value: Any, protocol_type: Any) -> TypeCheckResult:
    """Validate that all methods in a protocol are correctly implemented.

    Args:
        value: The value to check
        protocol_type: The protocol class to check against

    Returns:
        TypeCheckResult with validation status and any errors
    """
    for name, method in protocol_type.__dict__.items():
        if name.startswith("_") or not hasattr(method, "__annotations__"):
            continue

        value_method = getattr(value, name)
        value_annotations = getattr(value_method, "__annotations__", {})

        # Check parameter types
        result = _validate_parameter_types(name, method, value_annotations)
        if not result.is_valid:
            return result

        # Check return type
        result = _validate_return_type(name, method, value_annotations)
        if not result.is_valid:
            return result

    # All checks passed
    return TypeCheckResult(is_valid=True)


def _validate_parameter_types(
    name: str, method: Any, value_annotations: Dict[str, Any]
) -> TypeCheckResult:
    """Validate parameter types for a method against a protocol method.

    Args:
        name: The method name
        method: The protocol method
        value_annotations: The implementation method's annotations

    Returns:
        TypeCheckResult with validation status and any errors
    """
    for param_name, expected_type in method.__annotations__.items():
        if param_name == "return":
            continue

        if param_name not in value_annotations:
            return TypeCheckResult(
                is_valid=False,
                errors=[f"Missing type annotation for {name}.{param_name}"],
            )

        if value_annotations[param_name] != expected_type:
            type_str = _get_type_str(expected_type)
            value_type_str = _get_type_str(value_annotations[param_name])
            return TypeCheckResult(
                is_valid=False,
                errors=[
                    f"Type mismatch in {name}.{param_name}: expected {type_str}, got {value_type_str}"
                ],
            )

    return TypeCheckResult(is_valid=True)


def _validate_return_type(
    name: str, method: Any, value_annotations: Dict[str, Any]
) -> TypeCheckResult:
    """Validate return type for a method against a protocol method.

    Args:
        name: The method name
        method: The protocol method
        value_annotations: The implementation method's annotations

    Returns:
        TypeCheckResult with validation status and any errors
    """
    # If no return type annotation in protocol, it's valid
    if "return" not in method.__annotations__:
        return TypeCheckResult(is_valid=True)

    expected_return = method.__annotations__["return"]

    # Check if return type annotation is missing
    if "return" not in value_annotations:
        return TypeCheckResult(
            is_valid=False,
            errors=[f"Missing return type annotation for {name}"],
        )

    # Check if return type matches
    if value_annotations["return"] != expected_return:
        type_str = _get_type_str(expected_return)
        value_return_str = _get_type_str(value_annotations["return"])
        return TypeCheckResult(
            is_valid=False,
            errors=[
                f"Return type mismatch in {name}: expected {type_str}, got {value_return_str}"
            ],
        )

    return TypeCheckResult(is_valid=True)


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
        # Check if import has required components
        if not self._has_valid_components():
            return self._set_validation_result(
                False, "Import has no module or imported names"
            )

        try:
            # Handle relative imports
            if self.is_relative:
                return self._validate_relative_import(package_path)
            # Handle absolute imports
            elif self.module:
                return self._validate_absolute_import()

            return self._set_validation_result(True, None)
        except Exception as e:
            return self._set_validation_result(
                False, f"Error validating import: {str(e)}"
            )

    def _has_valid_components(self) -> bool:
        """
        Check if the import has valid components (module or imported names).

        Returns:
            True if the import has valid components, False otherwise
        """
        return bool(self.module or self.imported_names)

    def _validate_relative_import(self, package_path: Optional[str]) -> bool:
        """
        Validate a relative import.

        Args:
            package_path: Package path for resolving relative imports

        Returns:
            True if the import is valid, False otherwise
        """
        # Relative imports require a package path
        if not package_path:
            return self._set_validation_result(
                False, "Cannot resolve relative import without package path"
            )

        if full_module := self._resolve_relative_module_path(package_path):
            # Check if the module exists
            return (
                True
                if check_dependency_available(full_module)
                else self._set_validation_result(
                    False, f"Cannot resolve relative import: {full_module}"
                )
            )
        else:
            return False  # Error already set in _resolve_relative_module_path

    def _resolve_relative_module_path(self, package_path: str) -> Optional[str]:
        """
        Resolve the full module path for a relative import.

        Args:
            package_path: Package path for resolving relative imports

        Returns:
            Full module path if resolved successfully, None otherwise
        """
        parts = package_path.split(".")

        # Check if relative import level is valid
        if self.level > len(parts):
            self._set_validation_result(
                False,
                f"Relative import level {self.level} exceeds package depth {len(parts)}",
            )
            return None

        # Calculate parent package
        parent_pkg = "" if self.level == len(parts) else ".".join(parts[: -self.level])

        # Construct the full module name
        if self.module:
            return f"{parent_pkg}.{self.module}" if parent_pkg else self.module
        else:
            return parent_pkg

    def _validate_absolute_import(self) -> bool:
        """
        Validate an absolute import.

        Returns:
            True if the import is valid, False otherwise
        """
        if not check_dependency_available(self.module):
            return self._set_validation_result(
                False, f"Cannot resolve absolute import: {self.module}"
            )
        return True

    def _set_validation_result(
        self, is_valid: bool, error_message: Optional[str]
    ) -> bool:
        """
        Set the validation result and error message.

        Args:
            is_valid: Whether the import is valid
            error_message: Error message if the import is invalid

        Returns:
            The is_valid parameter (for method chaining)
        """
        self.is_valid = is_valid
        self.error_message = error_message
        return is_valid

    def get_full_import_path(self, package_path: Optional[str] = None) -> Optional[str]:
        """Get the full import path for this import.

        Args:
            package_path: Optional package path for resolving relative imports

        Returns:
            Full import path or None if it cannot be resolved
        """
        if not self.validate(package_path):
            return None

        if not self.is_relative:
            return self.module
        if not package_path:
            return None

        # Calculate the parent package for relative imports
        parts = package_path.split(".")
        if self.level > len(parts):
            return None

        parent_pkg = "" if self.level == len(parts) else ".".join(parts[: -self.level])
        # Construct the full module name
        if self.module:
            return f"{parent_pkg}.{self.module}" if parent_pkg else self.module
        else:
            return parent_pkg
