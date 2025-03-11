"""
Enhanced Python code signature analyzer with advanced type inference and analysis.
Features:
- Deep signature analysis using advanced AST traversal
- Type inference using machine learning and pattern matching
- Function complexity and cohesion metrics
- Signature clustering and similarity analysis
- Advanced visualization capabilities
- Import dependency tracking with graph theory
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union

import networkx as nx
import typeguard
from pydantic import BaseModel, Field
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from rich.console import Console
from rich.syntax import Syntax
from rich.tree import Tree
from typing_extensions import Protocol, runtime_checkable

# Import libcst with error handling
try:
    import libcst as cst
except ImportError:
    cst = None

# Optional dependency configuration
OPTIONAL_DEPENDENCIES = {
    'libcst': {
        'module': 'libcst',
        'import_as': 'cst',
        'required_for': ['AST parsing', 'code transformation'],
    },
    'numpy': {
        'module': 'numpy',
        'import_as': 'np',
        'required_for': ['numerical computations'],
    },
    'rustworkx': {
        'module': 'rustworkx',
        'import_as': 'rx',
        'required_for': ['graph operations'],
    },
    'sympy': {
        'module': 'sympy',
        'import_as': 'sympy',
        'required_for': ['symbolic mathematics'],
    },
    'torch': {
        'module': 'torch',
        'import_as': 'torch',
        'submodules': [('torch.nn.functional', 'F')],
        'required_for': ['type inference', 'model predictions'],
    },
    'scipy': {
        'module': 'scipy.spatial.distance',
        'imports': ['cosine'],
        'required_for': ['similarity calculations'],
    },
    'sklearn': {
        'module': 'sklearn',
        'imports': [
            ('sklearn.cluster', ['DBSCAN']),
            ('sklearn.feature_extraction.text', ['TfidfVectorizer']),
        ],
        'required_for': ['clustering', 'text vectorization'],
    },
}

# Initialize console before dependency checks
console = Console()

# Check and import optional dependencies
DEPENDENCY_STATUS = {}
IMPORTED_MODULES = {}

# Type checking imports
if TYPE_CHECKING:
    import libcst as cst
    import numpy as np
    import rustworkx as rx
    import sympy
    import torch
    import torch.nn.functional as F
    from scipy.spatial.distance import cosine
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer

# Runtime dependency management
for dep_name, dep_info in OPTIONAL_DEPENDENCIES.items():
    # Check availability
    is_available = importlib.util.find_spec(dep_info['module'].split('.')[0]) is not None
    DEPENDENCY_STATUS[dep_name] = is_available
    
    if is_available:
        try:
            # Import main module if specified
            if 'import_as' in dep_info:
                module = __import__(dep_info['module'], fromlist=['*'])
                IMPORTED_MODULES[dep_info['import_as']] = module
                globals()[dep_info['import_as']] = module
            
            # Import submodules if specified
            if 'submodules' in dep_info:
                for submodule, alias in dep_info['submodules']:
                    sub_mod = __import__(submodule, fromlist=['*'])
                    IMPORTED_MODULES[alias] = sub_mod
                    globals()[alias] = sub_mod
            
            # Import specific items if specified
            if 'imports' in dep_info:
                for imp in dep_info['imports']:
                    if isinstance(imp, tuple):
                        module, items = imp
                        mod = __import__(module, fromlist=items)
                        for item in items:
                            IMPORTED_MODULES[item] = getattr(mod, item)
                            globals()[item] = getattr(mod, item)
                    else:
                        module = dep_info['module']
                        mod = __import__(module, fromlist=[imp])
                        IMPORTED_MODULES[imp] = getattr(mod, imp)
                        globals()[imp] = getattr(mod, imp)
        except ImportError as e:
            DEPENDENCY_STATUS[dep_name] = False
            console.debug(f"Failed to import {dep_name}: {e}")
    else:
        # Initialize empty placeholders for unavailable modules
        if 'import_as' in dep_info:
            globals()[dep_info['import_as']] = None
        if 'submodules' in dep_info:
            for _, alias in dep_info['submodules']:
                globals()[alias] = None
        if 'imports' in dep_info:
            for imp in dep_info['imports']:
                name = imp[1][0] if isinstance(imp, tuple) else imp
                globals()[name] = None

console = Console()

T = TypeVar("T")


@runtime_checkable
class TypeAnnotated(Protocol):
    """Protocol for objects with type annotations"""

    __annotations__: Dict[str, Any]


@runtime_checkable
class Documented(Protocol):
    """Protocol for objects with docstrings"""

    __doc__: Optional[str]


@runtime_checkable
class Callable(Protocol):
    """Protocol for callable objects with signature information"""

    __signature__: inspect.Signature
    __name__: str
    __module__: str
    __qualname__: str
    __annotations__: Dict[str, Any]
    __doc__: Optional[str]


@runtime_checkable
class SignatureComparable(Protocol):
    """Protocol for objects that can be compared based on their signatures"""

    def is_compatible_with(self, other: Any) -> bool:
        """Check if this signature is compatible with another"""
        ...

    def similarity_score(self, other: Any) -> float:
        """Calculate similarity score with another signature"""
        ...


@runtime_checkable
class SignatureValidatable(Protocol):
    """Protocol for objects that can validate their signatures"""

    def validate(self) -> bool:
        """Validate the signature"""
        ...

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        ...


@runtime_checkable
class SignatureProvider(Protocol):
    """Protocol for objects that can provide signature information"""

    def get_signature(self) -> CodeSignature:
        """Get the signature information for this object"""
        ...

    def get_type_info(self) -> Dict[str, TypeInfo]:
        """Get type information for all components"""
        ...

    def get_metrics(self) -> SignatureMetrics:
        """Get signature metrics"""
        ...


@dataclass
class TypeInfo(SignatureValidatable):
    """Enhanced type information with inference confidence

    Implements SignatureValidatable protocol for validation support.
    Provides comprehensive type information including:
    - Static type hints from code
    - Inferred types from ML model
    - Confidence scores for inference
    - Source locations for type usage
    - Type constraints for validation
    """

    type_hint: Optional[str]
    inferred_type: Optional[str]
    confidence: float
    source_locations: Set[str] = field(default_factory=set)
    constraints: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.validate()

    def validate(self) -> bool:
        """Validate type consistency and constraints"""
        # Check basic type presence
        if not self.type_hint and not self.inferred_type:
            self.confidence *= 0.5
            return False

        # Validate type hint consistency
        if self.type_hint:
            try:
                typeguard.check_type(value="", expected_type=eval(self.type_hint))
            except Exception:
                self.confidence *= 0.5
                return False

        # Validate type hint and inferred type consistency
        if (
            self.type_hint
            and self.inferred_type
            and self.type_hint != self.inferred_type
        ):
            self.confidence *= 0.8
            return False

        # Validate constraints if present
        for constraint in self.constraints:
            try:
                eval(constraint)
            except Exception:
                self.confidence *= 0.9
                return False

        return True

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        errors = []

        # Check type presence
        if not self.type_hint and not self.inferred_type:
            errors.append("No type information available")

        # Validate type hint
        if self.type_hint:
            try:
                typeguard.check_type(value="", expected_type=eval(self.type_hint))
            except Exception as e:
                errors.append(f"Invalid type hint '{self.type_hint}': {str(e)}")

        # Check type consistency
        if (
            self.type_hint
            and self.inferred_type
            and self.type_hint != self.inferred_type
        ):
            errors.append(
                f"Type hint '{self.type_hint}' conflicts with inferred type '{self.inferred_type}'"
            )

        # Check constraints
        for constraint in self.constraints:
            try:
                eval(constraint)
            except Exception as e:
                errors.append(f"Constraint validation failed: {constraint} - {str(e)}")

        return errors


class SignatureMetrics(BaseModel):
    """Advanced metrics for code signatures

    Provides comprehensive metrics for analyzing code signatures including:
    - Structural metrics (complexity, cohesion, coupling)
    - Quality metrics (maintainability, documentation)
    - Type safety metrics (type hints, inference, constraints)
    - Validation metrics (validation score, coverage, compatibility)

    All metrics are normalized between 0.0 and 1.0
    """

    # Structural Metrics
    complexity: float = Field(
        0.0, ge=0.0, le=1.0, description="Cyclomatic complexity normalized to [0,1]"
    )
    cohesion: float = Field(
        0.0, ge=0.0, le=1.0, description="Method and attribute cohesion score"
    )
    coupling: float = Field(
        0.0, ge=0.0, le=1.0, description="Coupling with other components"
    )

    # Quality Metrics
    maintainability: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall maintainability index"
    )
    documentation_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Documentation completeness and quality"
    )

    # Type Safety Metrics
    type_safety: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall type safety score"
    )
    type_hint_coverage: float = Field(
        0.0, ge=0.0, le=1.0, description="Proportion of components with type hints"
    )
    type_inference_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Average confidence of inferred types"
    )
    constraint_coverage: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of components with type constraints",
    )

    # Validation Metrics
    validation_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall validation success rate"
    )
    validation_coverage: float = Field(
        0.0, ge=0.0, le=1.0, description="Proportion of components validated"
    )
    compatibility_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Compatibility with related signatures"
    )
    error_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Rate of validation errors"
    )


@dataclass
class SignatureComponent(
    TypeAnnotated, SignatureProvider, SignatureValidatable, SignatureComparable
):
    """Component of a signature with enhanced analysis"""

    name: str
    type_info: TypeInfo
    default_value: Optional[str] = None
    is_optional: bool = False
    constraints: List[str] = field(default_factory=list)
    usage_locations: Set[str] = field(default_factory=set)
    _type_annotations: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize type annotations and other post-init setup."""
        # Initialize type annotations
        self._type_annotations = {}
        if self.type_info is not None:
            self._type_annotations[self.name] = self.type_info.type_hint
            # Initialize other annotations from type info
            if self.type_info.type_hint:
                try:
                    type_obj = eval(self.type_info.type_hint)
                    self.__dict__['__annotations__'] = {self.name: type_obj}
                except (NameError, SyntaxError):
                    self.__dict__['__annotations__'] = {self.name: Any}
        
        # Handle default values
        if (self.default_value is not None and isinstance(self.default_value, str) and 
            not (self.default_value.startswith('"') and self.default_value.endswith('"'))):
            if self.default_value.startswith("'") and self.default_value.endswith("'"):
                # Convert single quotes to double quotes
                self.default_value = f'"{self.default_value[1:-1]}"'
            elif self.type_info and self.type_info.type_hint == 'str':
                # Add quotes for string literals if type hint is str
                self.default_value = f'"{self.default_value}"'

    def get_signature(self) -> CodeSignature:
        """Get a minimal signature for this component"""
        return CodeSignature(
            name=self.name,
            module_path=Path(),
            components=[],
            return_type=self.type_info,
        )

    def get_type_info(self) -> Dict[str, TypeInfo]:
        """Get type information for this component"""
        return {self.name: self.type_info}

    def get_metrics(self) -> SignatureMetrics:
        """Get basic metrics for this component"""
        return SignatureMetrics(
            complexity=0.1, type_safety=self.type_info.confidence  # Base complexity
        )

    def validate(self) -> bool:
        """Validate the component"""
        return len(self.get_validation_errors()) == 0

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        errors = []
        if not self.type_info.type_hint and not self.type_info.inferred_type:
            errors.append(f"No type information for {self.name}")
        if self.type_info.confidence < 0.5:
            errors.append(
                f"Low type confidence ({self.type_info.confidence}) for {self.name}"
            )
        return errors

    def is_compatible_with(self, other: Any) -> bool:
        """Check if this component is compatible with another"""
        if not isinstance(other, SignatureComponent):
            return False
        # Check type compatibility
        if self.type_info.type_hint and other.type_info.type_hint:
            return self.type_info.type_hint == other.type_info.type_hint
        # If no type hints, fall back to inferred types
        if self.type_info.inferred_type and other.type_info.inferred_type:
            return self.type_info.inferred_type == other.type_info.inferred_type
        return True  # No type information to compare

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type_info.type_hint or self.type_info.inferred_type,
            "confidence": self.type_info.confidence,
            "optional": self.is_optional,
            "constraints": self.constraints,
        }


@dataclass
class CodeSignature(
    TypeAnnotated,
    Documented,
    SignatureProvider,
    SignatureValidatable,
    SignatureComparable,
):
    """Enhanced code signature with comprehensive analysis"""

    name: str
    module_path: Path
    components: List[SignatureComponent]
    return_type: Optional[TypeInfo] = None
    docstring: Optional[str] = None
    metrics: SignatureMetrics = field(default_factory=SignatureMetrics)
    dependencies: Set[str] = field(default_factory=set)
    call_graph: Optional[nx.DiGraph] = None
    _type_annotations: Dict[str, Any] = field(default_factory=dict, init=False)
    __doc__: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize type annotations and docstring after dataclass initialization."""
        # Initialize type annotations
        self._type_annotations = {}
        for comp in self.components:
            if comp.type_info and comp.type_info.type_hint:
                try:
                    type_obj = eval(comp.type_info.type_hint)
                    self._type_annotations[comp.name] = type_obj
                except (NameError, SyntaxError):
                    self._type_annotations[comp.name] = Any

        # Add return type annotation if present
        if self.return_type and self.return_type.type_hint:
            try:
                type_obj = eval(self.return_type.type_hint)
                self._type_annotations["return"] = type_obj
            except (NameError, SyntaxError):
                self._type_annotations["return"] = Any

        # Set annotations in __dict__ to avoid dataclass issues
        self.__dict__["__annotations__"] = self._type_annotations
        self.__doc__ = self.docstring

    def get_signature(self) -> "CodeSignature":
        """Get this signature"""
        return self

    def get_type_info(self) -> Dict[str, TypeInfo]:
        """Get type information for all components"""
        type_info = {comp.name: comp.type_info for comp in self.components}
        if self.return_type:
            type_info["return"] = self.return_type
        return type_info

    def get_metrics(self) -> SignatureMetrics:
        """Get signature metrics"""
        return self.metrics

    def validate(self) -> bool:
        """Validate the signature"""
        return len(self.get_validation_errors()) == 0

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        errors = []
        # Check component types
        for comp in self.components:
            if not comp.validate():
                errors.extend(comp.get_validation_errors())
        # Check return type
        if self.return_type and not self.return_type.type_hint:
            errors.append(f"Missing return type hint for {self.name}")
        return errors

    def is_compatible_with(self, other: Any) -> bool:
        """Check if this signature is compatible with another"""
        if not isinstance(other, CodeSignature):
            return False
        # Check return type compatibility
        if (
            self.return_type
            and other.return_type
            and self.return_type.type_hint != other.return_type.type_hint
        ):
            return False
        # Check parameter compatibility (order matters)
        if len(self.components) != len(other.components):
            return False
        return all(
            s.is_compatible_with(o) for s, o in zip(self.components, other.components)
        )

    def similarity_score(self, other: Any) -> float:
        """Calculate signature similarity using TF-IDF and cosine similarity"""
        if not isinstance(other, CodeSignature):
            return 0.0
        vectorizer = TfidfVectorizer()
        signatures = [
            f"{self.name} {' '.join(c.name for c in self.components)}",
            f"{other.name} {' '.join(c.name for c in other.components)}",
        ]
        tfidf_matrix = vectorizer.fit_transform(signatures)
        return 1 - cosine(tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1])


class MathExpressionEvaluator:
    """
    Utilizes `sympy` to evaluate and analyze mathematical expressions within signature components.
    """

    @staticmethod
    def is_math_expression(expression: str) -> bool:
        try:
            sympy.sympify(expression)
            return True
        except sympy.SympifyError:
            return False

    @staticmethod
    def simplify_expression(expression: str) -> Optional[str]:
        """
        Simplifies a given mathematical expression.

        Args:
            expression: A string representing a mathematical expression.

        Returns:
            Simplified mathematical expression as a string, or None if invalid.
        """
        try:
            simplified = sympy.simplify(expression)
            return str(simplified)
        except sympy.SympifyError:
            return None

    @staticmethod
    def solve_expression(equation: str, variable: str = "x") -> Optional[str]:
        """
        Solves the equation for a given variable.

        Args:
            equation: The string representation of the equation.
            variable: The variable to solve for.

        Returns:
            The solution as a string, or None if unsolvable.
        """
        try:
            expr = sympy.Eq(sympy.sympify(equation), 0)
            solutions = sympy.solve(expr, sympy.Symbol(variable))
            return str(solutions)
        except (sympy.SympifyError, ValueError):
            return None


class CodeHighlighter:
    """
    Provides syntax highlighting for Python code using `pygments`.
    """

    @staticmethod
    def highlight_code(code: str) -> str:
        """
        Highlights given Python code using `pygments`.

        Args:
            code: A string containing Python code.

        Returns:
            Syntax-highlighted HTML representation of the code.
        """
        formatter = HtmlFormatter(full=True, linenos=True, style="colorful")
        return highlight(code, PythonLexer(), formatter)


class SyntaxTreeVisualizer:
    """
    Uses `rich` to provide tree visualization for Python syntax elements.
    Requires the `rich` package for tree visualization.
    """

    @staticmethod
    def visualize_ast(node: ast.AST, name: str = "AST Tree") -> Optional[Tree]:
        """
        Builds a tree visualization of the AST.

        Args:
            node: The input `ast` node to visualize.
            name: The tree root name.

        Returns:
            A `rich.tree.Tree` object representing the AST, or None if visualization fails.

        Raises:
            TypeError: If node is not an AST node.
        """
        if not isinstance(node, ast.AST):
            raise TypeError(f"Expected ast.AST, got {type(node).__name__}")

        try:
            tree = Tree(name)
            SyntaxTreeVisualizer._build_ast_tree(tree, node)
            return tree
        except Exception as e:
            console.warning(f"Failed to visualize AST: {e}")
            return None

    @staticmethod
    def _build_ast_tree(tree: Tree, node: ast.AST) -> None:
        """
        Recursively builds the AST as a tree structure.

        Args:
            tree: The current `Tree` object being populated.
            node: The AST node being inspected.

        Note:
            Handles AST nodes, lists of nodes, and primitive values.
            Skips None values and empty lists for cleaner visualization.
        """
        if not isinstance(node, ast.AST):
            console.warning(f"Invalid node type: {type(node).__name__}")
            return

        try:
            for field_name, value in ast.iter_fields(node):
                # Skip None values and empty lists
                if value is None or (isinstance(value, list) and not value):
                    continue

                if isinstance(value, ast.AST):
                    subtree = tree.add(f"{field_name}: {type(value).__name__}")
                    SyntaxTreeVisualizer._build_ast_tree(subtree, value)
                elif isinstance(value, list):
                    subtree = tree.add(f"{field_name}: list[{len(value)}]")
                    for item in value:
                        if isinstance(item, ast.AST):
                            item_tree = subtree.add(f"{type(item).__name__}")
                            SyntaxTreeVisualizer._build_ast_tree(item_tree, item)
                        elif item is not None:
                            subtree.add(str(item))
                else:
                    # Safely convert value to string, truncate if too long
                    str_value = str(value)
                    if len(str_value) > 100:
                        str_value = f"{str_value[:97]}..."
                    tree.add(f"{field_name}: {str_value}")
        except Exception as e:
            console.warning(f"Error building AST tree: {e}")

    @staticmethod
    def render_code_tree(code: str) -> None:
        """
        Parses Python code and renders the syntax tree using `rich`.

        Args:
            code: Input Python code as a string.

        Note:
            Handles syntax errors and invalid input gracefully.
            Skips visualization if AST parsing fails.
        """
        if not code or not code.strip():
            console.warning("Empty code string provided")
            return

        try:
            parsed_ast = ast.parse(code)
            if tree := SyntaxTreeVisualizer.visualize_ast(parsed_ast):
                console.print(tree)
            else:
                console.warning("Failed to generate AST visualization")
        except SyntaxError as e:
            console.print(f"[red]Syntax Error:[/] {str(e)}")
            # Show the problematic line if available
            if text := getattr(e, 'text', None):
                console.print(f"[yellow]Line {e.lineno}:[/] {text.rstrip()}")
                if e.offset:
                    console.print(f"{' ' * (e.offset + 14)}[red]^[/]")
        except Exception as e:
            console.print(f"[red]Error parsing code:[/] {str(e)}")


class RichSyntaxHighlighter:
    """
    Uses `rich.syntax.Syntax` to display colorful syntax-highlighted code in the console.
    Requires the `rich` package for syntax highlighting.
    """

    @staticmethod
    def display_code(code: str, theme: str = "monokai", show_line_numbers: bool = True) -> None:
        """
        Displays syntax-highlighted Python code in the console.

        Args:
            code: The Python code as a string.
            theme: The syntax highlighting theme to use (default: monokai).
            show_line_numbers: Whether to display line numbers (default: True).

        Note:
            Handles empty input and syntax errors gracefully.
            Supports various rich themes for syntax highlighting.
        """
        if not code or not code.strip():
            console.warning("Empty code string provided")
            return

        try:
            # Validate Python syntax before highlighting
            ast.parse(code)
            
            # Create syntax object with error handling
            try:
                syntax = Syntax(
                    code,
                    lexer="python",
                    theme=theme,
                    line_numbers=show_line_numbers,
                    word_wrap=True
                )
                console.print(syntax)
            except Exception as e:
                console.warning(f"Error highlighting code: {e}")
                # Fallback to plain text if highlighting fails
                console.print(code)
        except SyntaxError as e:
            console.print(f"[red]Syntax Error:[/] {str(e)}")
            # Show the problematic line if available
            if text := getattr(e, 'text', None):
                console.print(f"[yellow]Line {e.lineno}:[/] {text.rstrip()}")
                if e.offset:
                    console.print(f"{' ' * (e.offset + 14)}[red]^[/]")
        except Exception as e:
            console.print(f"[red]Error parsing code:[/] {str(e)}")


class SignatureVisitor:
    """
    Represents a visitor for inspecting and processing function or class
    signatures in Python code. Uses libcst if available, otherwise falls back
    to ast module for basic functionality.

    This visitor collects information about the code structure, including
    functions, methods, classes, and their type annotations.

    Attributes:
        file_path: The path of the file being analyzed.
        type_inference_model: Machine learning model for type inference (optional).
        signatures: A list to store `CodeSignature` objects based on discovered elements.

    Requirements:
        - libcst (optional): For enhanced AST traversal and code analysis
        - torch (optional): For type inference model support
    """

    def __init__(
        self, 
        file_path: Path, 
        type_inference_model: Optional[Any] = None,
        enable_type_inference: bool = True
    ) -> None:
        # Validate file path
        if not isinstance(file_path, Path):
            raise TypeError(f"Expected Path object, got {type(file_path).__name__}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path exists but is not a file: {file_path}")
        """Initialize the SignatureVisitor.

        Args:
            file_path: Path to the file being analyzed.
            type_inference_model: Optional ML model for type inference.
            enable_type_inference: Whether to use type inference (default: True).
        """
        self.file_path = file_path
        self.signatures: List[CodeSignature] = []
        self.current_class: Optional[str] = None
        self.enable_type_inference = enable_type_inference
        
        # Initialize type inference
        self.type_inference_model = None
        if enable_type_inference and type_inference_model is not None:
            if not DEPENDENCY_STATUS.get('torch'):
                console.warning(
                    "Type inference requested but torch not available. "
                    "Install torch for ML-based type inference support."
                )
            else:
                try:
                    self.type_inference_model = type_inference_model
                    console.debug("Type inference model initialized successfully")
                except Exception as e:
                    console.warning(f"Failed to initialize type inference model: {e}")
        
        # Initialize base class based on available dependencies
        if cst is not None:
            self._visitor = self._create_cst_visitor()
        else:
            self._visitor = self._create_ast_visitor()
            
    def analyze(self) -> None:
        """Analyze the file and collect signatures."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        try:
            source = self.file_path.read_text()
            
            if isinstance(self._visitor, ast.NodeVisitor):
                tree = ast.parse(source)
                self._visitor.visit(tree)
            else:
                tree = cst.parse_module(source)
                tree.visit(self._visitor)
        except (OSError, SyntaxError) as e:
            console.warning(f"Error analyzing file {self.file_path}: {e}")
            raise
            
    def _process_annotation(self, node: cst.BaseExpression) -> Optional[str]:
        """Process a type annotation node to get its string representation."""
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            return cst.Module([]).code_for_node(node)
        elif isinstance(node, cst.Subscript):
            # Handle complex types like List[int], Dict[str, int], etc.
            if isinstance(node.value, (cst.Name, cst.Attribute)):
                return self._extracted_from__process_annotation_10(node)
        return None

    # TODO Rename this here and in `_process_annotation`
    def _extracted_from__process_annotation_10(self, node):
        base_type = self._process_annotation(node.value)
        if not base_type:
            return None

        # Process type parameters
        params = []
        for param in node.slice:
            if isinstance(param.slice.value, (cst.Name, cst.Attribute, cst.Subscript)):
                if param_type := self._process_annotation(param.slice.value):
                    params.append(param_type)

        return f"{base_type}[{', '.join(params)}]" if params else base_type

    def _create_cst_visitor(self) -> Any:
        """Create a libcst-based visitor if available."""
        parent = self
        class CSTVisitor(cst.CSTVisitor):
            def __init__(self):
                super().__init__()
                self.parent = parent
                
            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                # Extract function signature info using libcst
                name = node.name.value
                components = []
                
                # Get parameters
                for param in node.params.params:
                    param_name = param.name.value
                    type_hint = None
                    default_value = None
                    
                    # Get type hint if available
                    if param.annotation:
                        type_hint = self.parent._process_annotation(param.annotation.annotation)
                    
                    # Handle default value if present
                    if param.default:
                        try:
                            default_value = cst.Module([]).code_for_node(param.default)
                            # Handle string literals
                            if default_value.startswith("'") and default_value.endswith("'"):
                                default_value = f'"{default_value[1:-1]}"'
                            elif type_hint == 'str' and not (default_value.startswith('"') and default_value.endswith('"')):
                                default_value = f'"{default_value}"'
                        except Exception as e:
                            console.warning(f"Failed to get default value for {param_name}: {e}")
                    
                    # Handle type inference if needed
                    inferred_type = None
                    confidence = 0.0
                    if not type_hint and self.parent.type_inference_model and self.parent.enable_type_inference:
                        try:
                            inferred_type = self.parent.type_inference_model.predict(param_name)
                            confidence = 0.5  # Mock confidence value
                        except Exception as e:
                            console.warning(f"Type inference failed for parameter {param_name}: {e}")
                            inferred_type = None
                            confidence = 0.0
                    
                    components.append(SignatureComponent(
                        name=param_name,
                        type_info=TypeInfo(
                            type_hint=type_hint,
                            inferred_type=inferred_type,
                            confidence=confidence
                        ),
                        default_value=default_value
                    ))
                    
                # Get return type
                return_type = None
                if node.returns:
                    type_hint = self.parent._process_annotation(node.returns.annotation)
                    return_type = TypeInfo(type_hint=type_hint, inferred_type=None, confidence=0.0)
                    
                # Create signature
                sig = CodeSignature(
                    name=name,
                    module_path=self.parent.file_path,
                    components=components,
                    return_type=return_type,
                    docstring=node.get_docstring()
                )
                self.parent.signatures.append(sig)
                
        return CSTVisitor()
        
    def _create_ast_visitor(self) -> Any:
        """Create an ast-based visitor as fallback."""
        parent = self


        class ASTVisitor(ast.NodeVisitor):
            def __init__(self):
                super().__init__()
                self.parent = parent
                self.current_class = None

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                # Extract function signature info using ast
                name = node.name
                components = []

                # Get parameters and their defaults
                defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
                for arg, default in zip(node.args.args, defaults):
                    param_name = arg.arg
                    type_hint = None
                    default_value = None

                    # Get type hint if available
                    if arg.annotation:
                        type_hint = ast.unparse(arg.annotation)

                    # Handle default value if present
                    if default is not None:
                        try:
                            default_value = ast.unparse(default)
                            # Handle string literals
                            if default_value.startswith("'") and default_value.endswith("'"):
                                default_value = f'"{default_value[1:-1]}"'
                            elif type_hint == 'str' and not (default_value.startswith('"') and default_value.endswith('"')):
                                default_value = f'"{default_value}"'
                        except Exception as e:
                            console.warning(f"Failed to unparse default value for {param_name}: {e}")

                    # Handle type inference if needed
                    inferred_type = None
                    confidence = 0.0
                    if not type_hint and self.parent.type_inference_model and self.parent.enable_type_inference:
                        try:
                            inferred_type = self.parent.type_inference_model.predict(param_name)
                            confidence = 0.5  # Mock confidence value
                        except Exception as e:
                            console.warning(f"Type inference failed for parameter {param_name}: {e}")
                            inferred_type = None
                            confidence = 0.0

                    components.append(SignatureComponent(
                        name=param_name,
                        type_info=TypeInfo(
                            type_hint=type_hint,
                            inferred_type=inferred_type,
                            confidence=confidence
                        ),
                        default_value=default_value
                    ))

                # Get return type
                return_type = None
                if hasattr(node, 'returns') and node.returns:
                    type_hint = ast.unparse(node.returns)
                    return_type = TypeInfo(type_hint=type_hint, inferred_type=None, confidence=0.0)

                # Create signature with class context if inside a class
                module_path = self.parent.file_path
                if self.current_class:
                    # Ensure consistent module path format for class methods
                    module_path = module_path.with_name(f"{module_path.stem}.{self.current_class}.py")

                sig = CodeSignature(
                    name=name,
                    module_path=module_path,
                    components=components,
                    return_type=return_type,
                    docstring=ast.get_docstring(node)
                )
                self.parent.signatures.append(sig)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                # Store current class name for method context
                prev_class = self.current_class
                self.current_class = node.name

                try:
                    # Create class signature
                    sig = CodeSignature(
                        name=node.name,
                        module_path=self.parent.file_path,
                        components=[],
                        docstring=ast.get_docstring(node),
                        dependencies={
                            base.id
                            for base in node.bases
                            if isinstance(base, ast.Name)
                        },
                    )
                    self.parent.signatures.append(sig)

                    # Visit class body to process methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            try:
                                self.visit(item)
                            except Exception as e:
                                console.warning(
                                    f"Error processing method {item.name} "
                                    f"in class {node.name}: {e}"
                                )
                finally:
                    # Restore previous class context
                    self.current_class = prev_class


        return ASTVisitor()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """
        Processes `FunctionDef` nodes (functions) and collects their relevant information.

        Args:
            node: A `libcst.FunctionDef` object representing a function definition.

        Note:
            Handles errors gracefully and logs warnings for invalid nodes.
            Uses type inference if enabled and no explicit type hints are present.
        """
        try:
            # Extract basic function information
            name = node.name.value
            if not name:
                console.warning("Found function definition with empty name")
                return

            # Get docstring and return type
            docstring = self._get_docstring(node)
            return_type = self._infer_type(node.returns)
            
            # Create return type info
            inferred_return_type = None
            if not return_type and self.type_inference_model is not None:
                inferred_return_type = self._infer_type_from_model(f"return_{name}")

            type_info = TypeInfo(
                type_hint=return_type,
                inferred_type=inferred_return_type,
                confidence=1.0 if return_type else 0.7
            )

            # Extract and process parameters
            try:
                parameters = self._extract_parameters(node.params)
            except Exception as e:
                console.warning(f"Error extracting parameters from {name}: {e}")
                parameters = []

            # Create components with proper error handling
            components: List[SignatureComponent] = []
            for param_name, param_type, is_optional in parameters:
                if not param_name:
                    console.warning(f"Skipping parameter with empty name in {name}")
                    continue

                # Infer type if needed and model is available
                inferred_type = None
                if not param_type and self.type_inference_model is not None:
                    inferred_type = self._infer_type_from_model(param_name)

                try:
                    component = SignatureComponent(
                        name=param_name,
                        type_info=TypeInfo(
                            type_hint=param_type,
                            inferred_type=inferred_type,
                            confidence=0.9 if param_type else 0.7,
                        ),
                        is_optional=is_optional,
                    )
                    components.append(component)
                except Exception as e:
                    console.warning(f"Error creating component for {param_name}: {e}")

            # Create and validate signature
            try:
                # Create signature with class context if inside a class
                module_path = self.file_path
                if self.current_class:
                    # Ensure consistent module path format for class methods
                    module_path = module_path.with_name(f"{module_path.stem}.{self.current_class}.py")
                
                code_signature = CodeSignature(
                    name=name,
                    module_path=module_path,
                    components=components,
                    return_type=type_info,
                    docstring=docstring,
                )
                self.signatures.append(code_signature)
            except Exception as e:
                console.warning(f"Error creating signature for {name}: {e}")

        except Exception as e:
            console.warning(f"Error processing function definition: {e}")

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """
        Processes `ClassDef` nodes (classes) to capture relevant signature-like information.

        Args:
            node: A `libcst.ClassDef` object representing a class definition.

        Note:
            Handles class inheritance and method signatures.
            Provides graceful error handling for invalid class definitions.
        """
        try:
            # Extract basic class information
            class_name = node.name.value
            if not class_name:
                console.warning("Found class definition with empty name")
                return

            # Get docstring and base classes
            docstring = self._get_docstring(node)
            base_classes: List[str] = []
            
            # Extract base classes with error handling
            try:
                for base in node.bases:
                    if isinstance(base.value, cst.Name):
                        base_classes.append(base.value.value)
                    elif isinstance(base.value, cst.Attribute):
                        # Handle qualified names (e.g., module.class)
                        base_classes.append(cst.Module([]).code_for_node(base.value))
            except Exception as e:
                console.warning(f"Error extracting base classes for {class_name}: {e}")

            # Store current class name for method context
            prev_class = self.current_class
            self.current_class = class_name

            try:
                # Create class signature
                code_signature = CodeSignature(
                    name=class_name,
                    module_path=self.file_path,
                    components=[],  # Methods will be added during traversal
                    docstring=docstring,
                    dependencies=set(base_classes) if base_classes else set()
                )
                self.signatures.append(code_signature)

                # Visit class body to process methods
                if node.body.body:
                    for statement in node.body.body:
                        if isinstance(statement, cst.FunctionDef):
                            try:
                                self.visit_FunctionDef(statement)
                            except Exception as e:
                                console.warning(
                                    f"Error processing method {statement.name.value} "
                                    f"in class {class_name}: {e}"
                                )
            finally:
                # Restore previous class context
                self.current_class = prev_class

        except Exception as e:
            console.warning(f"Error processing class definition: {e}")

    def _extract_parameters(
        self, params: cst.Parameters
    ) -> List[Tuple[str, Optional[str], bool]]:
        """
        Extract parameter details from a function definition.

        Args:
            params: The parameters node from a function definition.

        Returns:
            A list of tuples containing:
                - Parameter name (str)
                - Type annotation if present (Optional[str])
                - Whether parameter is optional (bool)

        Note:
            Handles special cases like *args, **kwargs, and positional-only parameters.
            Skips invalid parameters and logs warnings.
        """
        extracted_params: List[Tuple[str, Optional[str], bool]] = []

        try:
            # Process positional-only parameters
            if params.posonly_params:
                for param in params.posonly_params:
                    try:
                        self._process_parameter(param, extracted_params)
                    except Exception as e:
                        console.warning(f"Error processing positional-only parameter: {e}")

            # Process regular parameters
            for param in params.params:
                try:
                    self._process_parameter(param, extracted_params)
                except Exception as e:
                    console.warning(f"Error processing parameter: {e}")

            # Process *args parameter
            if params.star_arg:
                try:
                    self._process_parameter(params.star_arg, extracted_params)
                except Exception as e:
                    console.warning(f"Error processing *args parameter: {e}")

            # Process **kwargs parameter
            if params.kwonly_params:
                for param in params.kwonly_params:
                    try:
                        self._process_parameter(param, extracted_params)
                    except Exception as e:
                        console.warning(f"Error processing keyword-only parameter: {e}")

            return extracted_params

        except Exception as e:
            console.warning(f"Error extracting parameters: {e}")
            return []

    def _process_type_parameters(
        self,
        base_type: str,
        slice_nodes: Sequence[cst.SubscriptElement]
    ) -> Optional[str]:
        """
        Process type parameters from subscription nodes.

        Args:
            base_type: Base type string (e.g., 'List' for List[int])
            slice_nodes: Sequence of subscription elements

        Returns:
            Complete type string if valid, None otherwise

        Note:
            - Handles simple and complex type parameters
            - Validates parameter nodes
            - Constructs proper type string format
        """
        try:
            # Extract type parameters
            type_params = []
            for slice_item in slice_nodes:
                # Validate and process slice item
                if not (param_value := self._validate_slice_item(slice_item)):
                    continue

                # Process parameter value
                if param_str := self._process_param_value(param_value):
                    type_params.append(param_str)

            # Construct final type string
            return self._construct_type_string(base_type, type_params)

        except Exception as e:
            console.warning(f"Error processing type parameters: {e}")
            return None

    def _process_complex_type(self, node: cst.Subscript) -> Optional[str]:
        """
        Process a complex type annotation node.

        Args:
            node: Subscript node containing the complex type

        Returns:
            Complete type string if valid, None otherwise

        Note:
            - Handles generic types (List[T], Dict[K, V])
            - Validates base type and parameters
            - Provides comprehensive error handling
        """
        try:
            # Get base type
            if not (base_type := self._get_base_type(node.value)):
                return None

            # Get and process type parameters
            if not (type_str := self._process_type_parameters(base_type, node.slice)):
                return None

            return type_str

        except Exception as e:
            console.warning(f"Failed to process complex type annotation: {e}")
            return None

    def _validate_annotation_node(
        self,
        annotation: Any
    ) -> Optional[Union[cst.Name, cst.Subscript, cst.Attribute]]:
        """
        Validate an annotation node and extract its type information.

        Args:
            annotation: The annotation object to validate

        Returns:
            Valid annotation node if present, None otherwise

        Note:
            - Validates annotation object
            - Checks for supported node types
            - Provides error handling
        """
        try:
            # Check for annotation attribute
            if not hasattr(annotation, 'annotation'):
                console.warning(
                    f"Invalid annotation object: {type(annotation).__name__}"
                )
                return None

            # Get and validate node type
            node = annotation.annotation
            if not isinstance(node, (cst.Name, cst.Subscript, cst.Attribute)):
                console.debug(
                    f"Unsupported annotation type: {type(node).__name__}"
                )
                return None

            return node

        except Exception as e:
            console.warning(f"Error validating annotation node: {e}")
            return None

    def _validate_type_name(self, node: cst.Name) -> Optional[str]:
        """
        Validate and extract a simple type name.

        Args:
            node: Name node containing the type

        Returns:
            Type name if valid, None otherwise

        Note:
            - Validates type name
            - Handles empty values
            - Provides error handling
        """
        try:
            type_value = node.value
            if not type_value:
                console.warning("Empty type annotation value")
                return None

            return type_value

        except Exception as e:
            console.warning(f"Error validating type name: {e}")
            return None

    def _construct_type_string(self, base_type: str, type_params: List[str]) -> str:
        """
        Construct a complete type string from base type and parameters.

        Args:
            base_type: Base type name (e.g., 'List')
            type_params: List of type parameter strings

        Returns:
            Complete type string (e.g., 'List[int]' or 'Dict[str, int]')

        Note:
            - Handles empty parameter lists
            - Formats complex types correctly
            - Maintains proper spacing
        """
        return f"{base_type}[{', '.join(type_params)}]" if type_params else base_type

    def _validate_slice_item(self, slice_item: Any) -> Optional[cst.Name | cst.Attribute]:
        """
        Validate a slice item and extract its parameter value.

        Args:
            slice_item: Slice item to validate

        Returns:
            Parameter value if valid, None otherwise

        Note:
            - Validates slice item type
            - Extracts parameter value
            - Provides error handling
        """
        try:
            # Validate slice item type
            if not isinstance(slice_item, cst.SubscriptElement):
                console.warning(
                    f"Invalid slice item type: {type(slice_item).__name__}"
                )
                return None

            return slice_item.slice.value

        except Exception as e:
            console.warning(f"Error validating slice item: {e}")
            return None

    def _process_param_value(self, param_value: Union[cst.Name, cst.Attribute]) -> Optional[str]:
        """
        Process a parameter value node to extract its string representation.

        Args:
            param_value: Parameter value node to process

        Returns:
            String representation if valid, None otherwise

        Note:
            - Handles simple names and attributes
            - Validates node types
            - Provides error handling
        """
        try:
            # Validate parameter type
            if not isinstance(param_value, (cst.Name, cst.Attribute)):
                console.warning(
                    f"Unsupported parameter type: {type(param_value).__name__}"
                )
                return None

            # Convert to string representation
            param_str = cst.Module([]).code_for_node(param_value)
            if not param_str:
                console.warning("Empty parameter string")
                return None

            return param_str

        except Exception as e:
            console.warning(f"Error processing parameter value: {e}")
            return None

    def _get_qualified_name(self, node: cst.Attribute) -> Optional[str]:
        """
        Get qualified name from an attribute node.

        Args:
            node: Attribute node to process

        Returns:
            Qualified name if valid, None otherwise
        """
        try:
            return cst.Module([]).code_for_node(node)
        except Exception as e:
            console.warning(f"Failed to process qualified type name: {e}")
            return None

    def _get_base_type(self, node: Union[cst.Name, cst.Attribute]) -> Optional[str]:
        """
        Extract base type from a type annotation node.

        Args:
            node: The node containing the base type information

        Returns:
            Base type string if valid, None otherwise

        Note:
            - Handles simple names (str, int)
            - Handles qualified names (typing.List)
            - Validates node types
        """
        try:
            if isinstance(node, cst.Name):
                return node.value
            elif isinstance(node, cst.Attribute):
                return cst.Module([]).code_for_node(node)
            else:
                console.warning(f"Unsupported base type: {type(node).__name__}")
                return None
        except Exception as e:
            console.warning(f"Error extracting base type: {e}")
            return None

    def _process_parameter(
        self, param: Union[cst.Param, cst.ParamStar], params_list: List[Tuple[str, Optional[str], bool]]
    ) -> None:
        """
        Process a single parameter and add it to the parameters list.

        Args:
            param: The parameter node to process.
            params_list: List to append the processed parameter to.

        Note:
            Handles error cases and parameter validation.
            Logs warnings for invalid parameters.
        """
        if not isinstance(param, (cst.Param, cst.ParamStar)):
            console.warning(f"Invalid parameter type: {type(param).__name__}")
            return

        try:
            # Get parameter name
            param_name = param.name.value if isinstance(param, cst.Param) else "*args"
            if not param_name:
                console.warning("Empty parameter name")
                return

            # Get type annotation
            param_type = None
            if isinstance(param, cst.Param) and param.annotation:
                param_type = self._infer_type(param.annotation)

            # Check if parameter is optional (has default value)
            is_optional = False
            if isinstance(param, cst.Param):
                is_optional = param.default is not None

            params_list.append((param_name, param_type, is_optional))

        except Exception as e:
            console.warning(f"Error processing parameter {getattr(param, 'name', '<unknown>')}: {e}")

    def _infer_type(self, annotation: Optional[cst.Annotation]) -> Optional[str]:
        """
        Infers the type from an annotation node.

        Args:
            annotation: The annotation node representing the type hint.

        Returns:
            The inferred type as a string, or None if no type is specified.

        Note:
            Handles various type hint formats:
            - Simple types (int, str)
            - Complex types (List[int], Dict[str, int])
            - Qualified names (typing.List)
            - Union types (Union[str, int])
            - Optional types (Optional[str])
            - Custom types (MyClass)
            Returns None if annotation is invalid or missing.
        """
        if not annotation:
            return None

        try:
            # Validate and get annotation node
            if not (node := self._validate_annotation_node(annotation)):
                return None

            # Process simple type names (int, str, etc.)
            if isinstance(node, cst.Name) and (type_value := self._validate_type_name(node)):
                return type_value

            # Process complex types (List[int], Dict[str, int], etc.)
            elif isinstance(node, cst.Subscript) and (type_str := self._process_complex_type(node)):
                return type_str

            # Process qualified names (typing.List, etc.)
            elif isinstance(node, cst.Attribute) and (qualified_name := self._get_qualified_name(node)):
                return qualified_name

        except Exception as e:
            console.warning(f"Error processing type annotation: {e}")
            return None

        return None

    def _validate_model_inputs(self, name: str) -> Optional[str]:
        """
        Validate inputs for the type inference model.

        Args:
            name: Variable name to validate

        Returns:
            Validated and stripped name if valid, None otherwise
        """
        # Check model availability
        if self.type_inference_model is None:
            console.debug("Type inference model not available")
            return None

        # Validate input name
        if not isinstance(name, str):
            console.warning(f"Invalid input type for name: {type(name).__name__}")
            return None

        name = name.strip()
        if not name:
            console.debug("Empty variable name provided for type inference")
            return None

        return name

    def _infer_type_from_model(self, name: str) -> Optional[str]:
        """
        Uses the ML type inference model to predict a type for a variable name.

        Args:
            name: The name of the variable for which to infer the type.

        Returns:
            The inferred type as a string, or None if no model is specified or no prediction is possible.

        Note:
            - Requires PyTorch to be installed and model to be initialized
            - Uses a character-level tokenization approach
            - Handles various error cases gracefully
            - Applies confidence thresholds for predictions
        """
        # Validate inputs
        if not (validated_name := self._validate_model_inputs(name)):
            return None

        try:
            # Process input through model pipeline
            if not (predicted_type := self._process_input_through_model(validated_name)):
                return None

            # Log successful prediction
            console.debug(
                f"Inferred type '{predicted_type}' for variable '{validated_name}'"
            )
            return predicted_type

        except Exception as e:
            console.warning(f"Unexpected error in type inference: {e}")
            return None

    def _process_input_through_model(self, name: str) -> Optional[str]:
        """
        Process input through the complete model pipeline.

        Args:
            name: Variable name to process

        Returns:
            Predicted type if successful, None otherwise

        Note:
            - Handles tokenization
            - Gets model prediction
            - Processes prediction with confidence check
        """
        try:
            # Tokenize input
            if not (tokenized_input := self._tokenize_input(name)):
                return None

            # Get model prediction
            if not (prediction := self._get_model_prediction(
                tokenized_input,
                len(self.COMMON_TYPES),
                name
            )):
                return None

            # Process prediction
            return self._process_type_prediction(
                prediction=prediction,
                name=name,
                type_mapping=self.COMMON_TYPES
            )

        except Exception as e:
            console.warning(f"Error in model pipeline for '{name}': {e}")
            return None

    def _get_model_prediction(
        self,
        tokenized_input: torch.Tensor,
        num_types: int,
        name: str
    ) -> Optional[torch.Tensor]:
        """
        Get prediction from the type inference model.

        Args:
            tokenized_input: Tokenized input tensor
            num_types: Expected number of type classes
            name: Variable name for error reporting

        Returns:
            Model prediction tensor if successful, None otherwise

        Note:
            - Disables gradient computation
            - Validates output shape
            - Handles various error cases
        """
        # Run model inference
        prediction = self._run_model_inference(tokenized_input, name)
        if prediction is None:
            return None
            
        # Validate prediction shape
        if not self._validate_prediction_shape(prediction, num_types):
            return None
            
        return prediction

    def _run_model_inference(self, tokenized_input: torch.Tensor, name: str) -> Optional[torch.Tensor]:
        """
        Run inference using the type inference model.

        Args:
            tokenized_input: Tokenized input tensor
            name: Variable name for error reporting

        Returns:
            Model prediction if successful, None otherwise

        Note:
            - Disables gradient computation
            - Handles model-specific errors
            - Provides detailed error messages
        """
        if not DEPENDENCY_STATUS.get('torch'):
            console.warning("PyTorch not available for model inference")
            return None
            
        try:
            # Run model inference with gradient computation disabled
            with torch.no_grad():
                return self.type_inference_model(tokenized_input)
        except RuntimeError as e:
            console.warning(f"Model inference failed for '{name}': {e}")
            return None
        except Exception as e:
            console.warning(f"Unexpected error during model inference: {e}")
            return None

    def _validate_prediction_shape(
        self,
        prediction: torch.Tensor,
        expected_num_types: int
    ) -> bool:
        """
        Validate the shape of a model prediction tensor.

        Args:
            prediction: Model prediction tensor
            expected_num_types: Expected number of type classes

        Returns:
            True if shape is valid, False otherwise

        Note:
            - Checks tensor dimensions
            - Validates output size
            - Provides detailed error messages
        """
        try:
            # Check tensor dimensions
            if prediction.dim() != 2:
                console.warning(
                    f"Invalid prediction dimensions: {prediction.dim()}, expected 2"
                )
                return False

            # Validate number of type classes
            if prediction.size(1) != expected_num_types:
                console.warning(
                    f"Invalid number of type classes: {prediction.size(1)}, "
                    f"expected {expected_num_types}"
                )
                return False

            return True

        except Exception as e:
            console.warning(f"Error validating prediction shape: {e}")
            return False

    def _tokenize_input(self, name: str) -> Optional[torch.Tensor]:
        """
        Tokenize a variable name for input to the type inference model.

        Args:
            name: Variable name to tokenize

        Returns:
            Tokenized input tensor if successful, None otherwise

        Note:
            - Uses character-level tokenization
            - Limits vocabulary size to 1000 characters
            - Moves tensor to correct device
            - Adds batch dimension
        """
        try:
            # Get tokens and create tensor
            return tensor if (tokens := self._get_char_tokens(name)) and (tensor := self._create_token_tensor(tokens)) else None
        except Exception as e:
            console.warning(f"Tokenization failed for '{name}': {e}")
            return None

    def _get_char_tokens(self, name: str) -> Optional[List[int]]:
        """
        Convert input string to character-level tokens.

        Args:
            name: Input string to tokenize

        Returns:
            List of token indices if successful, None otherwise

        Note:
            - Converts to lowercase
            - Limits token values
            - Validates output
        """
        try:
            # Convert to character tokens
            return tokens if (tokens := [min(ord(char), 999) for char in name.lower()]) else None

        except Exception as e:
            console.warning(f"Character tokenization failed: {e}")
            return None

    def _create_token_tensor(self, tokens: List[int]) -> Optional[torch.Tensor]:
        """
        Create tensor from token indices.

        Args:
            tokens: List of token indices

        Returns:
            Tensor on correct device if successful, None otherwise

        Note:
            - Sets correct dtype
            - Moves to model device
            - Adds batch dimension
        """
        try:
            # Create tensor and move to device
            return torch.tensor(
                tokens,
                dtype=torch.long,
                device=next(self.type_inference_model.parameters()).device
            ).unsqueeze(0)

        except Exception as e:
            console.warning(f"Error creating token tensor: {e}")
            return None

    def _process_type_prediction(
        self,
        prediction: torch.Tensor,
        name: str,
        type_mapping: List[str],
        confidence_threshold: float = 0.5
    ) -> Optional[str]:
        """
        Process model prediction to get inferred type with confidence check.

        Args:
            prediction: Raw model prediction tensor from the type inference model
            name: Variable name being predicted
            type_mapping: List of possible type names
            confidence_threshold: Minimum confidence threshold for predictions (default: 0.5)

        Returns:
            Inferred type string if confidence threshold met, None otherwise

        Note:
            - Applies softmax to get probability distribution
            - Uses argmax to get most likely type
            - Checks confidence threshold
            - Handles edge cases and numerical stability
        """
        try:
            # Validate inputs
            if not self._validate_prediction_inputs(prediction, name, type_mapping):
                return None

            # Get prediction probabilities
            if not (prob_info := self._compute_probabilities(prediction)):
                return None
            confidence, type_index = prob_info

            # Check confidence threshold
            if not self._check_confidence_threshold(confidence, name, confidence_threshold):
                return None

            # Get and validate inferred type
            if not (inferred_type := self._get_inferred_type(type_index, type_mapping)):
                return None

            # Log successful prediction
            console.debug(
                f"Inferred type '{inferred_type}' for variable '{name}' "
                f"(confidence: {confidence:.2f})"
            )
            return inferred_type

        except Exception as e:
            console.warning(f"Error processing type prediction: {e}")
            return None

    def _validate_prediction_inputs(
        self,
        prediction: torch.Tensor,
        name: str,
        type_mapping: List[str]
    ) -> bool:
        """
        Validate inputs for type prediction processing.

        Args:
            prediction: Model prediction tensor
            name: Variable name
            type_mapping: Type mapping list

        Returns:
            True if inputs are valid, False otherwise

        Note:
            - Validates tensor type
            - Checks variable name
            - Validates type mapping
        """
        try:
            if not isinstance(prediction, torch.Tensor):
                console.warning(f"Invalid prediction type: {type(prediction).__name__}")
                return False

            if not isinstance(name, str) or not name.strip():
                console.warning("Invalid or empty variable name")
                return False

            if not isinstance(type_mapping, list) or not type_mapping:
                console.warning("Invalid or empty type mapping")
                return False

            return True

        except Exception as e:
            console.warning(f"Error validating prediction inputs: {e}")
            return False

    def _compute_probabilities(
        self,
        prediction: torch.Tensor
    ) -> Optional[Tuple[float, int]]:
        """
        Compute probabilities from model prediction.

        Args:
            prediction: Model prediction tensor

        Returns:
            Tuple of (confidence, type_index) if successful, None otherwise

        Note:
            - Moves tensor to CPU
            - Applies softmax
            - Handles numerical stability
        """
        try:
            # Move prediction to CPU for numpy operations
            logits = prediction.cpu()
            # Apply softmax with numerical stability
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, type_index = torch.max(probabilities, dim=1)
            return confidence.item(), type_index.item()

        except Exception as e:
            console.warning(f"Error computing probabilities: {e}")
            return None

    def _check_confidence_threshold(
        self,
        confidence: float,
        name: str,
        threshold: float
    ) -> bool:
        """
        Check if prediction confidence meets threshold.

        Args:
            confidence: Prediction confidence
            name: Variable name for logging
            threshold: Confidence threshold

        Returns:
            True if confidence meets threshold, False otherwise

        Note:
            - Validates confidence value
            - Provides debug logging
        """
        if confidence < threshold:
            console.debug(
                f"Low confidence ({confidence:.2f}) for variable '{name}' "
                f"(threshold: {threshold})"
            )
            return False
        return True

    def _get_inferred_type(
        self,
        type_index: int,
        type_mapping: List[str]
    ) -> Optional[str]:
        """
        Get inferred type from type mapping.

        Args:
            type_index: Index into type mapping
            type_mapping: List of possible types

        Returns:
            Inferred type string if valid, None otherwise

        Note:
            - Validates index range
            - Checks type string
            - Provides error messages
        """
        try:
            # Validate type index
            if not 0 <= type_index < len(type_mapping):
                console.warning(
                    f"Invalid type index {type_index} "
                    f"(max allowed: {len(type_mapping) - 1})"
                )
                return None

            # Get and validate inferred type
            inferred_type = type_mapping[type_index]
            if not isinstance(inferred_type, str):
                console.warning(
                    f"Invalid type mapping entry: {type(inferred_type).__name__}"
                )
                return None

            return inferred_type

        except Exception as e:
            console.warning(f"Error getting inferred type: {e}")
            return None

    def _get_docstring(
        self, node: Union[cst.FunctionDef, cst.ClassDef]
    ) -> Optional[str]:
        """
        Retrieves and processes the docstring for a function or class definition.

        Args:
            node: The node representing the function or class.

        Returns:
            The processed docstring as a string, or None if no docstring is present.

        Note:
            Handles both single and multi-line docstrings.
            Preserves docstring formatting while removing unnecessary whitespace.
            Returns None for invalid or empty docstrings.
        """
        try:
            # Validate node and get body
            if not (body := self._validate_docstring_node(node)):
                return None

            # Extract docstring from body
            if not (docstring := self._extract_docstring_from_body(body)):
                return None

            # Process and clean docstring
            return cleaned if (cleaned := self._clean_docstring(docstring)) else None
        except Exception as e:
            console.warning(f"Error processing docstring: {e}")
            return None

    def _validate_docstring_node(
        self,
        node: Union[cst.FunctionDef, cst.ClassDef]
    ) -> Optional[cst.IndentedBlock]:
        """
        Validate a node for docstring extraction.

        Args:
            node: Node to validate

        Returns:
            Node body if valid, None otherwise

        Note:
            - Validates node type
            - Checks body presence
            - Verifies indentation
        """
        try:
            # Validate input
            if not node or not isinstance(node, (cst.FunctionDef, cst.ClassDef)):
                console.warning(f"Invalid node type for docstring: {type(node).__name__}")
                return None

            # Check for body and proper indentation
            if not node.body or not isinstance(node.body, cst.IndentedBlock):
                return None

            return node.body

        except Exception as e:
            console.warning(f"Error validating docstring node: {e}")
            return None

    def _extract_docstring_from_body(
        self,
        body: cst.IndentedBlock
    ) -> Optional[str]:
        """
        Extract docstring from node body.

        Args:
            body: Node body to process

        Returns:
            Raw docstring if found, None otherwise

        Note:
            - Validates statement structure
            - Extracts string content
            - Handles evaluation errors
        """
        try:
            # Get first statement and validate structure
            if not body.body or not (expr := self._get_docstring_expr(body.body[0])):
                return None

            # Extract raw docstring
            if not (raw := self._extract_raw_docstring(expr)):
                return None

            # Format docstring
            return self._format_docstring(raw)

        except Exception as e:
            console.warning(f"Error extracting docstring: {e}")
            return None

    def _get_docstring_expr(
        self,
        stmt: cst.SimpleStatementLine
    ) -> Optional[Union[cst.SimpleString, cst.ConcatenatedString]]:
        """
        Get docstring expression from statement.

        Args:
            stmt: Statement to process

        Returns:
            Docstring expression if valid, None otherwise

        Note:
            - Validates statement type
            - Checks expression type
            - Handles different string types
        """
        try:
            # Validate statement type and structure
            if not isinstance(stmt, cst.SimpleStatementLine) or not stmt.body:
                return None

            # Get and validate expression
            expr = stmt.body[0]
            return expr.value if isinstance(expr, cst.Expr) and isinstance(
                expr.value, (cst.SimpleString, cst.ConcatenatedString)
            ) else None

        except Exception as e:
            console.warning(f"Error getting docstring expression: {e}")
            return None

    def _extract_raw_docstring(
        self,
        expr: Union[cst.SimpleString, cst.ConcatenatedString]
    ) -> Optional[str]:
        """
        Extract raw docstring from expression.

        Args:
            expr: Expression containing docstring

        Returns:
            Raw docstring if valid, None otherwise

        Note:
            - Handles simple strings
            - Handles concatenated strings
            - Validates output
        """
        try:
            if isinstance(expr, cst.SimpleString):
                raw = expr.value
            else:  # ConcatenatedString
                raw = "".join(
                    part.value for part in expr.parts
                    if isinstance(part, cst.SimpleString)
                )

            return raw or None

        except Exception as e:
            console.warning(f"Error extracting raw docstring: {e}")
            return None

    def _format_docstring(self, raw: str) -> str:
        """
        Format and clean a raw docstring.

        Args:
            raw: Raw docstring to format

        Returns:
            Formatted docstring

        Note:
            - Removes quotes
            - Normalizes line endings
            - Handles indentation
            - Preserves structure
        """
        try:
            # Remove quotes and normalize line endings
            docstring = raw.strip('"\'\'"').replace('\r\n', '\n')
            lines = docstring.split('\n')

            # Handle multi-line docstrings
            if len(lines) > 1:
                # Process subsequent lines
                if min_indent := self._find_min_indent(lines[1:]):
                    lines[1:] = [
                        line[min_indent:] if line.strip() else ''
                        for line in lines[1:]
                    ]

            # Join and normalize
            return '\n'.join(line.rstrip() for line in lines).strip()

        except Exception as e:
            console.warning(f"Error formatting docstring: {e}")
            return raw

    def _find_min_indent(self, lines: List[str]) -> int:
        """
        Find minimum indentation in a list of lines.

        Args:
            lines: Lines to process

        Returns:
            Minimum indentation level

        Note:
            - Skips empty lines
            - Handles mixed indentation
            - Returns 0 if no valid indent found
        """
        try:
            min_indent = float('inf')
            for line in lines:
                if stripped := line.lstrip():
                    indent = len(line) - len(stripped)
                    min_indent = min(min_indent, indent)

            return min(min_indent, float('inf')) if min_indent < float('inf') else 0

        except Exception as e:
            console.warning(f"Error finding minimum indentation: {e}")
            return 0

        except Exception as e:
            console.warning(f"Error extracting docstring: {e}")
            return None


class SignatureAnalyzer:
    """Advanced signature analyzer with ML-enhanced type inference"""

    # Common Python types for type inference
    COMMON_TYPES = [
        # Built-in types
        'str', 'int', 'float', 'bool',
        # Container types
        'list', 'dict', 'tuple', 'set',
        # Special types
        'datetime', 'Path', 'Optional', 'Any', 'None',
        # Additional types can be added here
    ]

    def __init__(self, root_path: Path):
        """
        Initialize the SignatureAnalyzer.

        Args:
            root_path: Root path of the project to analyze

        Raises:
            ImportError: If required dependencies are not available
        """
        # Validate root path
        if not isinstance(root_path, Path):
            raise TypeError(f"Expected Path object, got {type(root_path).__name__}")
        if not root_path.exists():
            raise FileNotFoundError(f"Root path does not exist: {root_path}")

        # Check required dependencies
        if not DEPENDENCY_STATUS.get('libcst'):
            raise ImportError(
                "libcst is required for SignatureAnalyzer. "
                "Please install it with: pip install libcst"
            )

        # Initialize instance variables
        self.root = root_path
        self.signatures: Dict[str, CodeSignature] = {}
        self.dependency_graph = nx.DiGraph()
        self.validation_errors: Dict[str, List[str]] = {}
        self.incompatible_pairs: List[Tuple[str, str]] = []

        # Initialize type inference model if available
        self.type_inference_model = None
        if DEPENDENCY_STATUS.get('torch'):
            try:
                self.type_inference_model = self._initialize_type_inference()
            except Exception as e:
                console.warning(
                    f"Failed to initialize type inference model: {e}. "
                    "Type inference will be disabled."
                )

    def validate_all_signatures(self) -> bool:
        """Validate all signatures in the project"""
        self.validation_errors.clear()
        is_valid = True

        for name, sig in self.signatures.items():
            if not sig.validate():
                self.validation_errors[name] = sig.get_validation_errors()
                is_valid = False

        return is_valid

    def check_signature_compatibility(self) -> bool:
        """Check compatibility between related signatures"""
        self.incompatible_pairs.clear()
        is_compatible = True

        # Check compatibility between connected signatures in dependency graph
        for source, target in self.dependency_graph.edges():
            source_sig = self.signatures.get(source)
            target_sig = self.signatures.get(target)
            if (
                source_sig
                and target_sig
                and not source_sig.is_compatible_with(target_sig)
            ):
                self.incompatible_pairs.append((source, target))
                is_compatible = False

        return is_compatible

    def find_similar_signatures(
        self, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """Find similar signatures based on similarity score"""
        similar_pairs = []
        seen = set()

        for name1, sig1 in self.signatures.items():
            for name2, sig2 in self.signatures.items():
                if name1 != name2 and (name2, name1) not in seen:
                    similarity = sig1.similarity_score(sig2)
                    if similarity >= threshold:
                        similar_pairs.append((name1, name2, similarity))
                        seen.add((name1, name2))

        return sorted(similar_pairs, key=lambda x: x[2], reverse=True)

    def analyze_project(self) -> Dict[str, Any]:
        """Perform comprehensive signature analysis of project"""
        console.print("[bold blue]Starting signature analysis...")

        # Analyze all Python files
        for py_file in self.root.rglob("*.py"):
            self._analyze_file(py_file)

        # Build dependency graph
        self._build_dependency_graph()

        # Calculate advanced metrics
        self._calculate_metrics()

        # Validate signatures and check compatibility
        is_valid = self.validate_all_signatures()
        is_compatible = self.check_signature_compatibility()
        similar_signatures = self.find_similar_signatures()

        # Generate clusters
        clusters = self._cluster_signatures()

        return {
            "signatures": self.signatures,
            "metrics": self._generate_project_metrics(),
            "clusters": clusters,
            "visualizations": self._generate_visualizations(),
            "validation": {
                "is_valid": is_valid,
                "validation_errors": self.validation_errors,
                "is_compatible": is_compatible,
                "incompatible_pairs": self.incompatible_pairs,
                "similar_signatures": similar_signatures,
            },
        }

    def _analyze_file(self, file_path: Path):
        """Analyze signatures in a single file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            module = cst.parse_module(source)
            visitor = SignatureVisitor(file_path, self.type_inference_model)
            module.visit(visitor)

            # Add signatures to collection
            for sig in visitor.signatures:
                qualified_name = f"{file_path.stem}.{sig.name}"
                self.signatures[qualified_name] = sig

        except Exception as e:
            console.print(f"[red]Error analyzing {file_path}: {str(e)}")

    def _build_dependency_graph(self):
        """Build comprehensive dependency graph using rustworkx if available, otherwise networkx"""
        if DEPENDENCY_STATUS.get('rustworkx'):
            try:
                graph = rx.PyDiGraph()
                node_map = {name: graph.add_node(name) for name, sig in self.signatures.items()}
                # Add edges with weights based on similarity
                for name1, sig1 in self.signatures.items():
                    for name2, sig2 in self.signatures.items():
                        if name1 != name2:
                            similarity = sig1.similarity_score(sig2)
                            if similarity > 0.5:
                                graph.add_edge(node_map[name1], node_map[name2], similarity)

                # Convert to networkx for additional algorithms
                self.dependency_graph = nx.DiGraph(graph.edge_list())
            except Exception as e:
                console.warning(f"Error using rustworkx, falling back to networkx: {e}")
                self._build_dependency_graph_networkx()
        else:
            self._build_dependency_graph_networkx()

    def _build_dependency_graph_networkx(self):
        """Build dependency graph using only networkx"""
        graph = nx.DiGraph()
        for name in self.signatures:
            graph.add_node(name)
        # Add edges with weights based on similarity
        for name1, sig1 in self.signatures.items():
            for name2, sig2 in self.signatures.items():
                if name1 != name2:
                    similarity = sig1.similarity_score(sig2)
                    if similarity > 0.5:
                        graph.add_edge(name1, name2, weight=similarity)
        self.dependency_graph = graph

    def _calculate_metrics(self):
        """Calculate advanced metrics for all signatures"""
        for name, signature in self.signatures.items():
            # Calculate structural metrics
            complexity = self._calculate_complexity(signature)
            cohesion = self._calculate_cohesion(signature)
            coupling = self._calculate_coupling(name)

            # Calculate quality metrics
            maintainability = self._calculate_maintainability(signature)
            documentation_score = self._calculate_doc_score(signature)

            # Calculate type safety metrics
            type_safety = self._calculate_type_safety(signature)
            if signature.components:
                n_components = len(signature.components)
                type_hint_coverage = (
                    len([c for c in signature.components if c.type_info.type_hint])
                    / n_components
                )
                type_inference_confidence = (
                    sum(c.type_info.confidence for c in signature.components)
                    / n_components
                )
                constraint_coverage = (
                    len([c for c in signature.components if c.constraints])
                    / n_components
                )
            else:
                type_hint_coverage = type_inference_confidence = constraint_coverage = (
                    0.0
                )

            # Calculate validation metrics
            validation_score = float(signature.validate())
            validation_errors = signature.get_validation_errors()
            error_rate = (
                len(validation_errors) / len(signature.components)
                if signature.components
                else 0.0
            )
            validation_coverage = 1.0 - error_rate

            # Calculate compatibility with related signatures
            related_sigs = {
                s
                for s in self.signatures.values()
                if s.name != signature.name
                and (
                    s.name in signature.dependencies or signature.name in s.dependencies
                )
            }
            compatibility_score = 1.0
            if related_sigs:  # Only calculate if there are related signatures
                compatibility_score = len(
                    [s for s in related_sigs if signature.is_compatible_with(s)]
                ) / len(related_sigs)

            # Update metrics
            signature.metrics = SignatureMetrics(
                # Structural metrics
                complexity=complexity,
                cohesion=cohesion,
                coupling=coupling,
                # Quality metrics
                maintainability=maintainability,
                documentation_score=documentation_score,
                # Type safety metrics
                type_safety=type_safety,
                type_hint_coverage=type_hint_coverage,
                type_inference_confidence=type_inference_confidence,
                constraint_coverage=constraint_coverage,
                # Validation metrics
                validation_score=validation_score,
                validation_coverage=validation_coverage,
                compatibility_score=compatibility_score,
                error_rate=error_rate,
            )

    def _calculate_complexity(self, signature: CodeSignature) -> float:
        """Calculate signature complexity using advanced metrics"""
        # Base complexity from number of components
        base_complexity = len(signature.components) / 10

        # Adjust for type complexity
        type_complexity = sum(
            1 + len(c.constraints)
            for c in signature.components
            if c.type_info.type_hint
        ) / (len(signature.components) or 1)

        # Adjust for dependency complexity
        dep_complexity = len(signature.dependencies) / 5

        # Combine using weighted sum
        return min(
            1.0, (base_complexity * 0.4 + type_complexity * 0.4 + dep_complexity * 0.2)
        )

    def _calculate_cohesion(self, signature: CodeSignature) -> float:
        """Calculate signature cohesion using spectral graph theory"""
        if not signature.call_graph:
            return 0.0

        try:
            # Calculate using normalized Laplacian eigenvalues
            laplacian = nx.normalized_laplacian_matrix(signature.call_graph)
            eigenvals = np.linalg.eigvals(laplacian.toarray())

            # Use algebraic connectivity (second smallest eigenvalue)
            return float(sorted(np.real(eigenvals))[1])

        except Exception:
            return 0.0

    def _calculate_coupling(self, signature_name: str) -> float:
        """Calculate coupling using graph centrality"""
        try:
            centrality = nx.eigenvector_centrality(self.dependency_graph)
            return centrality.get(signature_name, 0.0)
        except Exception:
            return 0.0

    def _calculate_maintainability(self, signature: CodeSignature) -> float:
        """Calculate maintainability index"""
        factors = [
            1 - signature.metrics.complexity,
            signature.metrics.cohesion,
            1 - signature.metrics.coupling,
            signature.metrics.documentation_score,
        ]
        return sum(factors) / len(factors)

    def _calculate_type_safety(self, signature: CodeSignature) -> float:
        """Calculate type safety score based on type hints, validation, and constraints"""
        type_scores = []
        for component in signature.components:
            # Base score from type hint presence and validation
            score = 0.0
            if component.type_info.type_hint:
                score += 0.4  # Type hint present
                if component.validate():  # Type validates correctly
                    score += 0.3

            # Adjust for type confidence and inference
            score += 0.2 * component.type_info.confidence
            if component.type_info.inferred_type:
                score += 0.1 * (
                    1.0
                    if component.type_info.inferred_type
                    == component.type_info.type_hint
                    else 0.5
                )

            # Bonus for having constraints and validation
            if component.constraints:
                score *= 1.1  # Smaller bonus as we now factor in validation
                if not component.get_validation_errors():  # All constraints satisfied
                    score *= 1.1

            type_scores.append(min(1.0, score))

        return sum(type_scores) / (len(type_scores) or 1)

    def _calculate_doc_score(self, signature: CodeSignature) -> float:
        """Calculate documentation quality score"""
        if not signature.docstring:
            return 0.0

        # Calculate using TF-IDF similarity with components
        vectorizer = TfidfVectorizer()
        docs = [signature.docstring, " ".join(c.name for c in signature.components)]
        try:
            tfidf_matrix = vectorizer.fit_transform(docs)
            return 1 - cosine(tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1])
        except Exception:
            return 0.0

    def _cluster_signatures(self) -> Dict[str, List[str]]:
        """Cluster similar signatures using DBSCAN if sklearn and numpy are available"""
        if not DEPENDENCY_STATUS.get('sklearn') or not DEPENDENCY_STATUS.get('numpy'):
            console.warning("sklearn and numpy required for signature clustering")
            return {}

        try:
            # Create feature vectors for signatures
            features = []
            sig_names = []

            for name, sig in self.signatures.items():
                # Create feature vector from signature metrics
                feature_vector = [
                    sig.metrics.complexity,
                    sig.metrics.cohesion,
                    sig.metrics.coupling,
                    sig.metrics.maintainability,
                    sig.metrics.type_safety,
                    len(sig.components) / 10,
                    len(sig.dependencies) / 5,
                ]
                features.append(feature_vector)
                sig_names.append(name)

            if not features:
                return {}

            # Convert to numpy array for DBSCAN
            features_array = np.array(features)

            # Cluster using DBSCAN with error handling
            try:
                clustering = DBSCAN(eps=0.3, min_samples=2)
                labels = clustering.fit_predict(features_array)
            except Exception as e:
                console.warning(f"DBSCAN clustering failed: {e}")
                return {}

            # Group signatures by cluster
            clusters = defaultdict(list)
            for name, label in zip(sig_names, labels):
                clusters[f"cluster_{label}"].append(name)

            return dict(clusters)
        except Exception as e:
            console.warning(f"Error during signature clustering: {e}")
            return {}

    def _initialize_type_inference(self) -> Optional[torch.nn.Module]:
        """Initialize neural type inference model if torch is available"""
        if not DEPENDENCY_STATUS.get('torch'):
            console.warning("PyTorch is required for type inference model. Running without ML-based type inference.")
            return None

        try:
            class TypeInferenceModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = torch.nn.Embedding(
                        1000, 64
                    )  # Vocabulary size, embedding dim
                    self.lstm = torch.nn.LSTM(64, 128, batch_first=True)
                    self.fc = torch.nn.Linear(128, 50)  # 50 common Python types

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    try:
                        x = self.embedding(x)
                        lstm_out, _ = self.lstm(x)
                        return F.softmax(self.fc(lstm_out[:, -1]), dim=1)
                    except Exception as e:
                        console.warning(f"Error in type inference model forward pass: {e}")
                        return torch.zeros(x.size(0), 50)  # Return zero probabilities as fallback

            return TypeInferenceModel()
        except Exception as e:
            console.warning(f"Failed to initialize type inference model: {e}")
            return None

    def _generate_project_metrics(self) -> Dict[str, float]:
        """Generate comprehensive project-wide metrics"""
        metrics = defaultdict(float)
        n_sigs = len(self.signatures) or 1

        # Calculate standard metrics
        for sig in self.signatures.values():
            for metric_name, value in sig.metrics.model_dump().items():
                metrics[f"avg_{metric_name}"] += value
            # Add validation score (1.0 if valid, 0.0 if invalid)
            metrics["validation_score"] += float(sig.validate())

        # Calculate compatibility metrics
        total_possible_pairs = (n_sigs * (n_sigs - 1)) / 2
        incompatible_count = len(self.incompatible_pairs)
        metrics["compatibility_score"] = 1.0 - (
            incompatible_count / total_possible_pairs if total_possible_pairs > 0 else 0
        )
        metrics["total_incompatible_pairs"] = float(incompatible_count)
        metrics["validation_coverage"] = len(
            [s for s in self.signatures.values() if s.validate()]
        )

        return {
            k: v / n_sigs
            for k, v in metrics.items()
            if k not in {"compatibility_score", "total_incompatible_pairs"}
        } | {
            "compatibility_score": metrics["compatibility_score"],
            "total_incompatible_pairs": metrics["total_incompatible_pairs"],
            "total_signatures": float(n_sigs),
        }

    def _generate_visualizations(self) -> Dict[str, str]:
        """Generate visualization outputs"""
        visualizations = {}

        # Generate dependency graph visualization
        with contextlib.suppress(Exception):
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.dependency_graph)
            nx.draw(
                self.dependency_graph,
                pos,
                with_labels=True,
                node_color="lightblue",
                font_size=8,
                arrows=True,
            )
            plt.savefig("signature_dependencies.png")
            plt.close()
            visualizations["dependency_graph"] = "signature_dependencies.png"
        return visualizations
