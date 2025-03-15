"""
Tests for runtime type checking functionality.

This test suite covers:
1. Runtime type validation of function arguments
2. Runtime type validation of return values
3. Protocol implementation validation
4. Optional type handling
5. Error handling and edge cases
"""

# Standard library imports

import importlib.util
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# Third-party library imports
import pytest
from typeguard import typechecked

# Local application imports
from python_fixer.core.types import validate_protocol, validate_type

# Check for optional dependencies
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# For type checking only - these imports are not executed at runtime


# Test fixtures
@pytest.fixture
def sample_protocol_class():
    """Create a sample protocol class for testing."""

    @runtime_checkable
    class SampleProtocol(Protocol):
        def method_a(self, x: int) -> str:
            """Test method with basic types."""
            ...

        def method_b(self, y: Optional[float], *, z: str = "default") -> List[int]:
            """Test method with optional param and keyword-only args."""
            ...

        def method_c(self) -> Optional[Dict[str, Any]]:
            """Test method with optional return type."""
            ...

    return SampleProtocol


@pytest.fixture
def valid_implementation():
    """Create a valid implementation of the sample protocol."""

    class ValidImpl:
        def method_a(self, x: int) -> str:
            return str(x)

        def method_b(self, y: Optional[float], *, z: str = "default") -> List[int]:
            return [] if y is None else [int(y)]

        def method_c(self) -> Optional[Dict[str, Any]]:
            return {"status": "ok"}

    return ValidImpl()


@pytest.fixture
def invalid_implementation():
    """Create an invalid implementation of the sample protocol."""

    class InvalidImpl:
        def method_a(self, x: str) -> int:  # Wrong parameter and return types
            return 42

        def method_b(
            self, y: float
        ) -> List[str]:  # Missing Optional, wrong return type
            return ["invalid"]

        # method_c missing entirely to test protocol completeness

    return InvalidImpl()


# Basic runtime type checking tests
@pytest.mark.unit
def test_basic_type_validation():
    """Test basic runtime type validation."""

    @typechecked
    def typed_function(x: int, y: str = "default") -> bool:
        return bool(x)

    # Valid cases
    assert typed_function(42) is True
    assert typed_function(0, "test") is False

    # Invalid cases
    result = validate_type("not an int", int)
    assert not result.is_valid
    assert "expected <class 'int'>, got str" in result.errors[0]

    result = validate_type(42, str)
    assert not result.is_valid
    assert "expected <class 'str'>, got int" in result.errors[0]


@pytest.mark.unit
def test_optional_type_validation():
    """Test runtime validation of Optional types."""

    @typechecked
    def optional_func(x: Optional[int], y: str = "default") -> Optional[bool]:
        return None if x is None else bool(x)

    # Valid cases
    assert optional_func(None) is None
    assert optional_func(42) is True
    assert optional_func(0) is False

    # Invalid cases
    result = validate_type("not an int", Optional[int])
    assert not result.is_valid
    assert (
        "expected Union[<class 'int'>, <class 'NoneType'>], got str" in result.errors[0]
    )


# Protocol implementation tests
@pytest.mark.unit
def test_protocol_implementation(
    sample_protocol_class, valid_implementation, invalid_implementation
):
    """Test runtime validation of protocol implementations."""
    # Initialize test variables before conditional blocks
    result = None
    error_messages = []

    # Test valid implementation
    result = validate_protocol(valid_implementation, sample_protocol_class)
    assert result.is_valid, f"Valid implementation failed: {result.errors}"

    # Test invalid implementation
    result = validate_protocol(invalid_implementation, sample_protocol_class)
    assert not result.is_valid, "Invalid implementation should fail validation"
    assert len(result.errors) > 0, "Should have validation errors"

    # Store error messages for validation
    error_messages = result.errors

    # Verify specific error messages with descriptive assertions
    assert any(
        "method_a" in msg and "expected <class 'int'>" in msg for msg in error_messages
    ), "Missing type mismatch error for method_a parameter type"
    assert any(
        "method_b" in msg
        and "expected Union[<class 'float'>, <class 'NoneType'>]" in msg
        for msg in error_messages
    ), "Missing Optional type error for method_b parameter"
    assert any("method_c" in msg for msg in error_messages), (
        "Missing error for missing method_c implementation"
    )

    # Test method calls with runtime type checking
    result = validate_type(valid_implementation, sample_protocol_class)
    assert result.is_valid, "Valid implementation should pass type check"

    result = validate_type(invalid_implementation, sample_protocol_class)
    assert not result.is_valid, "Invalid implementation should fail type check"
    assert "Object does not implement protocol" in result.errors[0], (
        "Missing or incorrect error message for protocol implementation failure"
    )


# Edge cases and error handling
@pytest.mark.unit
def test_type_error_messages():
    """Test that type error messages are clear and helpful."""
    # Test with invalid input type
    result = validate_type("not a list", List[Dict[str, Any]], context="complex_types")
    assert not result.is_valid
    assert "complex_types" in result.errors[0]
    assert "expected <class 'list[<class 'dict[str, typing.Any]'>>" in result.errors[0]
    assert "got str" in result.errors[0]

    # Test with invalid nested type
    result = validate_type([{"key": 42}, "not a dict"], List[Dict[str, Any]])
    assert not result.is_valid
    assert "expected <class 'dict'>" in result.errors[0]
    assert "got str" in result.errors[0]

    # Test with context information
    result = validate_type(123, str, context="string_field")
    assert not result.is_valid
    assert "string_field" in result.errors[0]
    assert "expected <class 'str'>" in result.errors[0]
    assert "got int" in result.errors[0]


@pytest.mark.unit
def test_runtime_type_inference():
    """Test runtime type checking with inferred types."""
    # Initialize variables before conditional blocks
    obj = None

    class DynamicClass:
        def __init__(self):
            self.value: Any = 42

        @typechecked
        def set_value(self, new_value: Any) -> None:
            self.value = new_value

        @typechecked
        def get_value(self) -> Any:
            return self.value

    obj = DynamicClass()

    # Any type should be accepted
    obj.set_value(123)
    obj.set_value("string")
    obj.set_value([1, 2, 3])

    # Return type Any should allow any value
    assert obj.get_value() == [1, 2, 3]


@pytest.mark.unit
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not available")
def test_numpy_type_validation():
    """Test runtime type validation with numpy arrays."""
    import numpy as np

    # Initialize variables before conditional blocks
    array_1d = np.array([1, 2, 3])
    array_2d = np.array([[1, 2], [3, 4]])

    @typechecked
    def process_array(x: np.ndarray) -> np.ndarray:
        return x * 2

    # Test valid numpy array inputs
    result = validate_type(array_1d, np.ndarray)
    assert result.is_valid, "1D numpy array validation failed"

    result = validate_type(array_2d, np.ndarray)
    assert result.is_valid, "2D numpy array validation failed"

    # Test invalid inputs
    result = validate_type([1, 2, 3], np.ndarray)
    assert not result.is_valid, "List should not validate as numpy array"
    assert "expected <class 'ndarray'>, got list" in result.errors[0]


@pytest.mark.unit
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_torch_type_validation():
    """Test runtime type validation with torch tensors and ML models."""
    import torch
    import torch.nn as nn

    # Initialize variables before conditional blocks
    tensor_1d = torch.tensor([1, 2, 3])
    tensor_2d = torch.tensor([[1, 2], [3, 4]])
    simple_model = nn.Linear(2, 1)

    @typechecked
    def process_tensor(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    @typechecked
    def process_model(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return model(x)

    # Test valid torch tensor inputs
    result = validate_type(tensor_1d, torch.Tensor)
    assert result.is_valid, "1D torch tensor validation failed"

    result = validate_type(tensor_2d, torch.Tensor)
    assert result.is_valid, "2D torch tensor validation failed"

    # Test torch model validation
    result = validate_type(simple_model, nn.Module)
    assert result.is_valid, "PyTorch model validation failed"

    # Test model parameter validation
    for param in simple_model.parameters():
        result = validate_type(param, torch.nn.Parameter)
        assert result.is_valid, "Model parameter validation failed"

    # Test invalid inputs
    result = validate_type([1, 2, 3], torch.Tensor)
    assert not result.is_valid, "List should not validate as torch tensor"
    assert "expected <class 'Tensor'>, got list" in result.errors[0]

    # Test invalid model type
    class FakeModel:
        def __call__(self, x):
            return x

    fake_model = FakeModel()
    result = validate_type(fake_model, nn.Module)
    assert not result.is_valid, "Invalid model should not validate as nn.Module"
    assert "expected <class 'Module'>, got FakeModel" in result.errors[0]


@pytest.mark.unit
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_torch_type_inference_fallback():
    """Test graceful fallback when torch-specific type inference fails."""
    import torch
    import torch.nn as nn

    # Initialize test variables
    tensor = torch.randn(2, 2)
    model = nn.Linear(2, 2)

    # Test fallback for invalid tensor operations
    result = validate_type(tensor, Dict[str, Any])
    assert not result.is_valid
    assert "expected <class 'dict'>, got Tensor" in result.errors[0]

    # Test fallback for invalid model operations
    result = validate_type(model, List[int])
    assert not result.is_valid
    assert "expected <class 'list'>, got Module" in result.errors[0]

    # Test graceful handling of non-tensor attributes
    result = validate_type(model.in_features, int)
    assert result.is_valid, "Model integer attribute validation failed"

    # Test proper error message for tensor shape mismatch
    @typechecked
    def requires_specific_shape(x: torch.Tensor) -> None:
        assert x.shape == (3, 3), "Expected 3x3 tensor"

    with pytest.raises(AssertionError, match="Expected 3x3 tensor"):
        requires_specific_shape(tensor)


# Integration tests
@pytest.mark.integration
def test_complex_type_validation():
    """Test runtime type checking with complex type annotations."""
    from typing import Generic, TypeVar, Union

    T = TypeVar("T")

    @typechecked
    class Container(Generic[T]):
        def __init__(self, value: T):
            self.value = value

        def get(self) -> T:
            return self.value

        def set(self, value: T) -> None:
            self.value = value

    # Test with different type parameters
    int_container = Container[int](42)
    assert int_container.get() == 42

    # Test invalid constructor argument
    result = validate_type("not an int", int)
    assert not result.is_valid
    assert "expected <class 'int'>, got str" in result.errors[0]
    result = validate_type("not an int", Container[int])
    assert not result.is_valid
    assert "expected <class 'Container[<class 'int'>]'>, got str" in result.errors[0]

    # Test invalid set value
    with pytest.raises(TypeError) as exc_info:
        int_container.set("not an int")
    assert "expected int" in str(exc_info.value).lower()
    assert "got str" in str(exc_info.value).lower()

    # Test with union types
    @typechecked
    def union_func(x: Union[int, str]) -> Union[float, bool]:
        return float(x) if isinstance(x, int) else bool(x)

    assert isinstance(union_func(42), float)
    assert isinstance(union_func("test"), bool)

    result = validate_type([], Union[int, str], context="union_type")
    assert not result.is_valid
    assert "union_type" in result.errors[0]
    assert "expected Union[int, str]" in result.errors[0]
    assert "got list" in result.errors[0]

    # Test invalid return type
    @typechecked
    def bad_return(x: int) -> str:
        return x  # Should raise TypeError

    with pytest.raises(TypeError) as exc_info:
        bad_return(42)
    assert "expected str" in str(exc_info.value).lower()
    assert "got int" in str(exc_info.value).lower()
