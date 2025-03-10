"""Tests for type hint validation functionality."""

import ast
# These typing imports are used in test code strings that are parsed with ast.parse()
# The linter may incorrectly mark them as unused since they appear in string literals
from typing import Dict, List, Optional, Protocol, TypeVar, Union  # noqa: F401
from unittest import TestCase

from python_fixer.core.analyzer import TypeAnnotationVisitor


class TestTypeAnnotationVisitor(TestCase):
    """Test suite for TypeAnnotationVisitor."""

    def setUp(self):
        self.visitor = TypeAnnotationVisitor()

    def _parse_and_visit(self, code: str) -> TypeAnnotationVisitor:
        """Parse code and visit with TypeAnnotationVisitor.
        
        Args:
            code: Python code to parse
            
        Returns:
            TypeAnnotationVisitor instance after visiting the AST
        """
        tree = ast.parse(code)
        self.visitor.visit(tree)
        return self.visitor

    def _assert_type_metrics(self, visitor: TypeAnnotationVisitor, total: int, valid: int, coverage: float = 100.0, error_count: int = 0):
        """Assert type annotation metrics match expected values.
        
        Args:
            visitor: TypeAnnotationVisitor instance
            total: Expected total annotations
            valid: Expected valid annotations
            coverage: Expected type coverage percentage
            error_count: Expected number of type errors
        """
        self.assertEqual(visitor.total_annotations, total)
        self.assertEqual(visitor.valid_annotations, valid)
        self.assertEqual(visitor.type_coverage, coverage)
        self.assertEqual(len(visitor.type_errors), error_count)

    def test_simple_type_annotations(self):
        # sourcery skip: class-extract-method
        """Test basic type annotations."""
        code = """
x: int = 1
y: str = "hello"
z: bool = True
"""
        visitor = self._parse_and_visit(code)
        self._assert_type_metrics(visitor, total=3, valid=3)

    def test_complex_type_annotations(self):
        """Test complex type annotations with generics."""
        code = """
x: List[int] = [1, 2, 3]
y: Dict[str, List[int]] = {"a": [1, 2]}
z: Optional[str] = None
"""
        visitor = self._parse_and_visit(code)
        self._assert_type_metrics(visitor, total=3, valid=3)

    def test_union_types(self):
        """Test union type annotations."""
        code = """
# Using Union
x: Union[int, str] = "1"
# Using | syntax (PEP 604)
y: int | str = 2
"""
        visitor = self._parse_and_visit(code)
        self._assert_type_metrics(visitor, total=2, valid=2)

    def test_function_annotations(self):
        """Test function type annotations."""
        code = """
def greet(name: str, count: int = 1) -> str:
    return name * count

def process(*args: str, **kwargs: int) -> None:
    pass
"""
        visitor = self._parse_and_visit(code)
        self._assert_type_metrics(visitor, total=6, valid=6)  # name, count, return, args, kwargs, return

    def test_forward_references(self):
        """Test forward reference type annotations."""
        code = """
class Tree:
    def __init__(self) -> None:
        self.left: "Tree" = None
        self.right: "Tree" = None
"""
        visitor = self._parse_and_visit(code)
        self._assert_type_metrics(visitor, total=3, valid=3)  # return, left, right

    def test_protocol_and_typevar(self):
        """Test Protocol and TypeVar annotations."""
        code = """
T = TypeVar('T')

class Printable(Protocol):
    def print(self) -> None: ...
    
def process(item: T) -> T:
    return item
    
x: Printable
"""
        visitor = self._parse_and_visit(code)
        self._assert_type_metrics(visitor, total=3, valid=3)  # print return, process param+return, x

    def test_invalid_annotations(self):
        """Test invalid type annotations."""
        code = """
x: 123 = "invalid"  # Invalid: annotation is a literal
y: [int] = []  # Invalid: using list literal as type
"""
        visitor = self._parse_and_visit(code)
        self._assert_type_metrics(visitor, total=2, valid=0, coverage=0.0, error_count=1)

    def test_missing_annotations(self):
        """Test handling of missing annotations."""
        code = """
x = 1  # No annotation
def func(a, b):  # No annotations
    pass
"""
        visitor = self._parse_and_visit(code)
        self._assert_type_metrics(visitor, total=0, valid=0, coverage=0.0)
