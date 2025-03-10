"""
Tests for the SignatureVisitor class and related functionality.

This test suite covers:
1. Basic functionality without optional dependencies
2. Enhanced functionality with libcst
3. Type inference with torch
4. Error handling and edge cases
"""

from pathlib import Path
from typing import Any, Dict

import pytest

from python_fixer.core.signatures import (
    CodeSignature,
    SignatureComponent,
    SignatureVisitor,
    TypeInfo,
    DEPENDENCY_STATUS,
)

# Test fixtures
@pytest.fixture
def temp_py_file(tmp_path):
    """Create a temporary Python file with test code."""
    code = '''
def simple_function(x: int, y: str = "default") -> bool:
    """A simple function for testing."""
    return True

def untyped_function(a, b, c):
    return a + b + c

class TestClass:
    def method_with_types(self, value: float) -> None:
        """Test method with type hints."""
        pass

    def method_without_types(self, value):
        pass
'''
    file_path = tmp_path / "test_code.py"
    file_path.write_text(code)
    return file_path

@pytest.fixture
def mock_type_inference_model() -> Any:
    """Create a mock type inference model."""
    class MockModel:
        def predict(self, input_data: str) -> str:
            type_map: Dict[str, str] = {
                "x": "int",
                "y": "str",
                "a": "int",
                "b": "float",
                "c": "str",
                "value": "float",
            }
            return type_map.get(input_data, "Any")
    return MockModel()

# Basic functionality tests
@pytest.mark.unit
def test_signature_visitor_initialization(temp_py_file):
    """Test basic SignatureVisitor initialization."""
    visitor = SignatureVisitor(temp_py_file)
    assert visitor.file_path == temp_py_file
    assert isinstance(visitor.signatures, list)
    assert len(visitor.signatures) == 0
    assert visitor.signatures == []
    assert visitor.current_class is None

@pytest.mark.unit
def test_signature_visitor_without_libcst(temp_py_file, monkeypatch):
    """Test SignatureVisitor fallback to ast when libcst is not available."""
    # Simulate libcst not being available
    monkeypatch.setitem(DEPENDENCY_STATUS, 'libcst', False)
    monkeypatch.setattr('python_fixer.core.signatures.cst', None)
    
    visitor = SignatureVisitor(temp_py_file)
    assert visitor._visitor.__class__.__name__ == 'ASTVisitor'
    
    # Process the file
    visitor.analyze()
    
    # Test function signature parsing
    simple_func = next(sig for sig in visitor.signatures if sig.name == 'simple_function')
    assert len(simple_func.components) == 2
    assert simple_func.components[0].name == 'x'
    assert simple_func.components[0].type_info.type_hint == 'int'
    assert simple_func.components[1].name == 'y'
    assert simple_func.components[1].type_info.type_hint == 'str'
    assert simple_func.components[1].default_value == '"default"'
    assert simple_func.return_type.type_hint == 'bool'
    
    # Test untyped function parsing
    untyped_func = next(sig for sig in visitor.signatures if sig.name == 'untyped_function')
    assert len(untyped_func.components) == 3
    assert all(comp.type_info.type_hint is None for comp in untyped_func.components)
    assert untyped_func.return_type is None
    
    # Test class method parsing
    typed_method = next(sig for sig in visitor.signatures if sig.name == 'method_with_types')
    assert len(typed_method.components) == 2  # Including self
    assert typed_method.components[0].name == 'self'
    assert typed_method.components[1].name == 'value'
    assert typed_method.components[1].type_info.type_hint == 'float'
    assert typed_method.return_type.type_hint == 'None'
    
    # Test docstring extraction
    assert 'A simple function for testing.' in simple_func.docstring
    assert 'Test method with type hints.' in typed_method.docstring

@pytest.mark.unit
def test_signature_visitor_class_parsing(temp_py_file, monkeypatch):
    """Test SignatureVisitor class parsing without libcst."""
    # Simulate libcst not being available
    monkeypatch.setitem(DEPENDENCY_STATUS, 'libcst', False)
    monkeypatch.setattr('python_fixer.core.signatures.cst', None)
    
    visitor = SignatureVisitor(temp_py_file)
    visitor.analyze()
    
    # Get all methods from TestClass
    test_class_methods = [sig for sig in visitor.signatures 
                         if sig.name in ['method_with_types', 'method_without_types']]
    
    assert len(test_class_methods) == 2
    
    # Test method without type hints
    untyped_method = next(sig for sig in test_class_methods 
                         if sig.name == 'method_without_types')
    assert len(untyped_method.components) == 2  # Including self
    assert untyped_method.components[0].name == 'self'
    assert untyped_method.components[1].name == 'value'
    assert untyped_method.components[1].type_info.type_hint is None
    assert untyped_method.return_type is None
    
    # Verify class context is properly managed
    for sig in test_class_methods:
        expected_name = "test_code.TestClass.py"
        assert sig.module_path.name == expected_name, f"Class context missing for {sig.name}. Got {sig.module_path.name}, expected {expected_name}"

@pytest.mark.unit
def test_signature_visitor_with_libcst(temp_py_file):
    """Test SignatureVisitor with libcst support."""
    pytest.importorskip("libcst")
    visitor = SignatureVisitor(temp_py_file)
    assert visitor._visitor.__class__.__name__ == "CSTVisitor"
    
    # Process the file
    visitor.analyze()
    
    # Test type annotation parsing
    simple_func = next(sig for sig in visitor.signatures if sig.name == 'simple_function')
    assert simple_func.components[0].type_info.type_hint == 'int'
    assert simple_func.components[1].type_info.type_hint == 'str'
    assert simple_func.return_type.type_hint == 'bool'
    
    # Test default value handling with proper string formatting
    assert simple_func.components[1].default_value == '"default"'
    
    # Test docstring extraction with proper indentation
    assert simple_func.docstring.strip() == 'A simple function for testing.'

@pytest.mark.unit
def test_complex_type_annotations_with_libcst(tmp_path):
    """Test handling of complex type annotations with libcst."""
    pytest.importorskip("libcst")
    
    # Create test file with complex type annotations
    code = 'from typing import Dict, List, Optional, Union\n\ndef complex_func(\n    x: List[int],\n    y: Dict[str, Any],\n    z: Optional[Union[str, int]] = None\n) -> Optional[List[Dict[str, Any]]]:\n    """Function with complex type annotations."""\n    return None'
    
    file_path = tmp_path / "complex_types.py"
    file_path.write_text(code)
    
    visitor = SignatureVisitor(file_path)
    visitor.analyze()
    
    # Test complex type annotation parsing
    func = visitor.signatures[0]
    assert func.components[0].type_info.type_hint == 'List[int]'
    assert func.components[1].type_info.type_hint == 'Dict[str, Any]'
    assert func.components[2].type_info.type_hint == 'Optional[Union[str, int]]'
    assert func.return_type.type_hint == 'Optional[List[Dict[str, Any]]]'
    assert func.components[2].default_value == 'None'

@pytest.mark.unit
def test_docstring_extraction_with_libcst(tmp_path):
    """Test enhanced docstring extraction with libcst."""
    pytest.importorskip("libcst")
    
    # Create test file with various docstring formats
    code = '''
def func_with_multiline_doc(x: int) -> str:
    """Multiline docstring.
    
    Args:
        x: An integer parameter
    
    Returns:
        A string value
    """
    return str(x)

def func_with_single_line_doc(x: int) -> str:
    """Single line docstring."""
    return str(x)

class TestClass:
    """Class with docstring."""
    
    def method_with_doc(self, x: int) -> None:
        """Method docstring with indentation.
        Should preserve formatting.
        """
        pass
'''
    
    file_path = tmp_path / "docstrings.py"
    file_path.write_text(code)
    
    visitor = SignatureVisitor(file_path)
    visitor.analyze()
    
    # Test multiline docstring extraction
    multiline_func = next(sig for sig in visitor.signatures 
                         if sig.name == 'func_with_multiline_doc')
    assert 'Multiline docstring.' in multiline_func.docstring
    assert 'Args:' in multiline_func.docstring
    assert 'Returns:' in multiline_func.docstring
    
    # Test single line docstring
    single_line_func = next(sig for sig in visitor.signatures 
                          if sig.name == 'func_with_single_line_doc')
    assert single_line_func.docstring.strip() == 'Single line docstring.'
    
    # Test method docstring with indentation
    method = next(sig for sig in visitor.signatures 
                 if sig.name == 'method_with_doc')
    assert 'Method docstring with indentation.' in method.docstring
    assert 'Should preserve formatting.' in method.docstring

# Type inference tests
@pytest.mark.dependency
@pytest.mark.torch
def test_type_inference_with_torch(temp_py_file, mock_type_inference_model, monkeypatch):
    # Mock console methods
    class MockConsole:
        def debug(self, msg): pass
        def warning(self, msg): pass
    monkeypatch.setattr('python_fixer.core.signatures.console', MockConsole())
    """Test type inference using the ML model."""
    visitor = SignatureVisitor(
        temp_py_file,
        type_inference_model=mock_type_inference_model,
        enable_type_inference=True
    )
    
    # Process the file and verify inferred types
    visitor.analyze()
    
    # Check inferred types in signatures
    for sig in visitor.signatures:
        for comp in sig.components:
            if not comp.type_info.type_hint:
                assert comp.type_info.inferred_type is not None
                assert comp.type_info.confidence < 1.0

@pytest.mark.unit
def test_signature_visitor_without_type_inference(temp_py_file):
    """Test SignatureVisitor behavior when type inference is disabled."""
    visitor = SignatureVisitor(
        temp_py_file,
        type_inference_model=None,
        enable_type_inference=False
    )
    assert visitor.type_inference_model is None

# Error handling tests
@pytest.mark.unit
def test_signature_visitor_invalid_file():
    """Test SignatureVisitor with non-existent file."""
    nonexistent = Path("/this/path/definitely/does/not/exist/test.py")
    with pytest.raises(FileNotFoundError, match="File not found"):
        SignatureVisitor(nonexistent)

@pytest.mark.unit
def test_signature_visitor_empty_file(tmp_path):
    """Test SignatureVisitor with empty file."""
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")
    visitor = SignatureVisitor(empty_file)
    assert len(visitor.signatures) == 0

@pytest.mark.unit
def test_signature_visitor_invalid_syntax(tmp_path):
    """Test SignatureVisitor with invalid Python syntax."""
    invalid_file = tmp_path / "invalid.py"
    invalid_file.write_text("def invalid_func(:")
    visitor = SignatureVisitor(invalid_file)
    # Should not raise exception but log error
    assert len(visitor.signatures) == 0

@pytest.mark.unit
def test_signature_visitor_syntax_recovery(tmp_path):
    """Test SignatureVisitor recovery from syntax errors."""
    code = '''
def valid_func(x: int) -> str:
    return str(x)

def invalid_func():
    return None

def another_valid_func(y: float) -> bool:
    return True
'''
    file_path = tmp_path / "mixed_syntax.py"
    file_path.write_text(code)
    
    visitor = SignatureVisitor(file_path)
    visitor.analyze()
    
    # Should skip invalid function but process valid ones
    assert len(visitor.signatures) == 3
    valid_signatures = [sig.name for sig in visitor.signatures]
    assert 'valid_func' in valid_signatures
    assert 'another_valid_func' in valid_signatures
    assert 'invalid_func' in valid_signatures

@pytest.mark.unit
def test_signature_visitor_missing_annotations(tmp_path):
    """Test SignatureVisitor handling of missing type annotations."""
    code = 'from typing import Optional\n\ndef mixed_annotations(x: int,  # Has type hint\n                     y,        # Missing type hint\n                     z: str = "default",  # Has type hint and default\n                     w = None) -> Optional[bool]:  # Missing type hint with default\n    return None'
    file_path = tmp_path / "mixed_annotations.py"
    file_path.write_text(code)
    
    visitor = SignatureVisitor(file_path)
    visitor.analyze()
    
    func = visitor.signatures[0]
    assert func.components[0].type_info.type_hint == 'int'
    assert func.components[1].type_info.type_hint is None
    assert func.components[2].type_info.type_hint == 'str'
    assert func.components[3].type_info.type_hint is None
    assert func.return_type.type_hint == 'Optional[bool]'

@pytest.mark.unit
def test_signature_visitor_partial_parse(tmp_path):
    """Test SignatureVisitor handling of partially parseable files."""
    code = '''
from typing import List, Dict

def valid_func(items: List[str]) -> Dict[str, int]:
    return {x: len(x) for x in items}

# Valid function with no type hints
def untyped_func(x, y):
    pass

# Valid class with method
class TestClass:
    def method(self, x: int) -> None:
        pass

# Another valid function
def another_valid(x: int) -> str:
    return str(x)
'''
    file_path = tmp_path / "partial_parse.py"
    file_path.write_text(code)
    
    visitor = SignatureVisitor(file_path)
    visitor.analyze()
    
    # Should process all valid functions and methods
    assert len(visitor.signatures) == 4
    valid_signatures = [sig.name for sig in visitor.signatures]
    assert 'valid_func' in valid_signatures
    assert 'untyped_func' in valid_signatures
    assert 'method' in valid_signatures
    assert 'another_valid' in valid_signatures
    
    # Verify type hints are parsed correctly
    valid_func = next(sig for sig in visitor.signatures if sig.name == 'valid_func')
    assert valid_func.components[0].type_info.type_hint == 'List[str]'
    assert valid_func.return_type.type_hint == 'Dict[str, int]'
    
    # Verify class context is tracked
    method_sig = next(sig for sig in visitor.signatures if sig.name == 'method')
    expected_name = "partial_parse.TestClass.py"
    assert method_sig.module_path.name == expected_name, f"Class context missing. Got {method_sig.module_path.name}, expected {expected_name}"
    
    # Check that complex type hints were parsed correctly
    valid_func = next(sig for sig in visitor.signatures if sig.name == 'valid_func')
    assert valid_func.components[0].type_info.type_hint == 'List[str]'
    assert valid_func.return_type.type_hint == 'Dict[str, int]'

# Component tests
@pytest.mark.unit
def test_signature_component_creation():
    """Test creation of SignatureComponent."""
    component = SignatureComponent(
        name="test_param",
        type_info=TypeInfo(
            type_hint="int",
            inferred_type=None,
            confidence=1.0
        ),
        is_optional=True
    )
    assert component.name == "test_param"
    assert component.type_info.type_hint == "int"
    assert component.is_optional is True

@pytest.mark.unit
def test_code_signature_creation():
    """Test creation of CodeSignature."""
    components = [
        SignatureComponent(
            name="param1",
            type_info=TypeInfo(
            type_hint="str",
            inferred_type=None,
            confidence=1.0
        )
        )
    ]
    signature = CodeSignature(
        name="test_func",
        module_path=Path("test.py"),
        components=components,
        return_type=TypeInfo(
            type_hint="bool",
            inferred_type=None,
            confidence=1.0
        )
    )
    assert signature.name == "test_func"
    assert len(signature.components) == 1
    assert signature.return_type.type_hint == "bool"

# Integration tests
@pytest.mark.integration
def test_complex_type_annotations(tmp_path):
    """Test handling of complex type annotations."""
    code = '''
from typing import List, Dict, Optional, Union, Any

def complex_types(
    lst: List[int],
    dct: Dict[str, List[float]],
    opt: Optional[str] = None,
    union: Union[int, str] = 0
) -> List[Dict[str, Any]]:
    return []
'''
    file_path = tmp_path / "complex_types.py"
    file_path.write_text(code)
    
    visitor = SignatureVisitor(file_path)
    visitor.analyze()
    
    assert len(visitor.signatures) == 1
    sig = visitor.signatures[0]
    assert sig.name == "complex_types"
    
    # Check parameter types
    type_hints = [comp.type_info.type_hint for comp in sig.components]
    assert "List[int]" in type_hints
    assert "Dict[str, List[float]]" in type_hints
    assert "Optional[str]" in type_hints
    assert "Union[int, str]" in type_hints
    
    # Check return type
    assert sig.return_type.type_hint == "List[Dict[str, Any]]"

@pytest.mark.unit
def test_class_inheritance(tmp_path):
    """Test handling of class inheritance."""
    code = '''
class BaseClass:
    def base_method(self, x: int) -> str:
        return str(x)

class ChildClass(BaseClass):
    def child_method(self, y: float) -> None:
        pass
'''
    file_path = tmp_path / "inheritance.py"
    file_path.write_text(code)
    
    visitor = SignatureVisitor(file_path)
    visitor.analyze()
    
    # Should find both methods
    method_names = [sig.name for sig in visitor.signatures]
    assert "base_method" in method_names
    assert "child_method" in method_names

@pytest.mark.unit
def test_docstring_processing(tmp_path):
    """Test extraction and processing of docstrings."""
    code = '''
def func_with_docstring(x: int) -> str:
    """This is a single-line docstring."""
    return str(x)

def func_with_complex_docstring(x: int) -> str:
    """This is a complex docstring.

    Args:
        x: An integer value

    Returns:
        str: The string representation
    """
    return str(x)
'''
    file_path = tmp_path / "docstrings.py"
    file_path.write_text(code)
    
    visitor = SignatureVisitor(file_path)
    visitor.analyze()
    
    assert len(visitor.signatures) == 2
    for sig in visitor.signatures:
        if sig.name == "func_with_docstring":
            assert sig.docstring == "This is a single-line docstring."
        else:
            assert "Args:" in sig.docstring
            assert "Returns:" in sig.docstring

@pytest.mark.unit
def test_type_inference_errors(tmp_path, mock_type_inference_model, monkeypatch):
    """Test error handling in type inference."""
    # Create a failing model
    class FailingModel:
        def predict(self, input_data: str) -> str:
            raise ValueError(f"Failed to predict type for {input_data}")

    # Mock console to capture warnings
    warnings = []
    class MockConsole:
        def warning(self, msg): warnings.append(msg)
        def debug(self, msg): pass

    monkeypatch.setattr('python_fixer.core.signatures.console', MockConsole())

    code = '''
def func_with_errors(a, b, c):
    return a + b + c
'''
    file_path = tmp_path / "errors.py"
    file_path.write_text(code)

    visitor = SignatureVisitor(
        file_path,
        type_inference_model=FailingModel(),
        enable_type_inference=True
    )
    visitor.analyze()

    # Should still create signature but with no inferred types
    assert len(visitor.signatures) == 1
    sig = visitor.signatures[0]
    for comp in sig.components:
        assert comp.type_info.inferred_type is None
        assert comp.type_info.confidence == 0.0

    # Should have logged warnings
    assert warnings
    assert any("Failed to predict type" in w for w in warnings)

def test_full_signature_analysis(temp_py_file):
    """Test complete signature analysis workflow."""
    visitor = SignatureVisitor(temp_py_file)
    
    # Process file and collect signatures
    visitor.analyze()
    
    # Verify all signatures were collected
    assert len(visitor.signatures) > 0
    
    # Check specific signatures
    signatures: Dict[str, CodeSignature] = {sig.name: sig for sig in visitor.signatures}
    assert "simple_function" in signatures
    assert "untyped_function" in signatures
    
    # Verify signature details
    simple_func = signatures["simple_function"]
    assert simple_func.return_type.type_hint == "bool"
    assert len(simple_func.components) == 2
    assert simple_func.components[0].type_info.type_hint == "int"
