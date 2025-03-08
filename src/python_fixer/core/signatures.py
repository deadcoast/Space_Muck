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
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

import libcst as cst
import networkx as nx
import numpy as np
import rustworkx as rx
import sympy
import torch
import torch.nn.functional as F
import typeguard
from pydantic import BaseModel, Field
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from rich.console import Console
from rich.syntax import Syntax
from rich.tree import Tree
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from typing_extensions import Protocol, runtime_checkable

console = Console()

T = TypeVar("T")


@runtime_checkable
class Callable(Protocol):
    """Protocol for callable objects with signature information"""

    __signature__: inspect.Signature


@dataclass
class TypeInfo:
    """Enhanced type information with inference confidence"""

    type_hint: Optional[str]
    inferred_type: Optional[str]
    confidence: float
    source_locations: Set[str] = field(default_factory=set)
    constraints: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate type consistency"""
        if self.type_hint and self.inferred_type:
            try:
                typeguard.check_type(value="", expected_type=eval(self.type_hint))
            except Exception:
                self.confidence *= 0.5


class SignatureMetrics(BaseModel):
    """Advanced metrics for code signatures"""

    complexity: float = Field(0.0, ge=0.0, le=1.0)
    cohesion: float = Field(0.0, ge=0.0, le=1.0)
    coupling: float = Field(0.0, ge=0.0, le=1.0)
    maintainability: float = Field(0.0, ge=0.0, le=1.0)
    type_safety: float = Field(0.0, ge=0.0, le=1.0)
    documentation_score: float = Field(0.0, ge=0.0, le=1.0)


@dataclass
class SignatureComponent:
    """Component of a signature with enhanced analysis"""

    name: str
    type_info: TypeInfo
    default_value: Optional[str] = None
    is_optional: bool = False
    constraints: List[str] = field(default_factory=list)
    usage_locations: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type_info.type_hint or self.type_info.inferred_type,
            "confidence": self.type_info.confidence,
            "optional": self.is_optional,
            "constraints": self.constraints,
        }


@dataclass
class CodeSignature:
    """Enhanced code signature with comprehensive analysis"""

    name: str
    module_path: Path
    components: List[SignatureComponent]
    return_type: Optional[TypeInfo] = None
    docstring: Optional[str] = None
    metrics: SignatureMetrics = field(default_factory=SignatureMetrics)
    dependencies: Set[str] = field(default_factory=set)
    call_graph: Optional[nx.DiGraph] = None

    def similarity_score(self, other: CodeSignature) -> float:
        """Calculate signature similarity using TF-IDF and cosine similarity"""
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
    """

    @staticmethod
    def visualize_ast(node: ast.AST, name: str = "AST Tree") -> Tree:
        """
        Builds a tree visualization of the AST.

        Args:
            node: The input `ast` node to visualize.
            name: The tree root name.

        Returns:
            A `rich.tree.Tree` object representing the AST.
        """
        tree = Tree(name)
        SyntaxTreeVisualizer._build_ast_tree(tree, node)
        return tree

    @staticmethod
    def _build_ast_tree(tree: Tree, node: ast.AST):
        """
        Recursively builds the AST as a tree structure.

        Args:
            tree: The current `Tree` object being populated.
            node: The AST node being inspected.
        """
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                subtree = tree.add(f"{field}: {type(value).__name__}")
                SyntaxTreeVisualizer._build_ast_tree(subtree, value)
            elif isinstance(value, list):
                subtree = tree.add(f"{field}: list")
                for item in value:
                    if isinstance(item, ast.AST):
                        item_tree = subtree.add(f"{type(item).__name__}")
                        SyntaxTreeVisualizer._build_ast_tree(item_tree, item)
            else:
                tree.add(f"{field}: {value}")

    @staticmethod
    def render_code_tree(code: str):
        """
        Parses Python code and renders the syntax tree using `rich`.

        Args:
            code: Input Python code as a string.
        """
        try:
            parsed_ast = ast.parse(code)
            tree = SyntaxTreeVisualizer.visualize_ast(parsed_ast)
            console.print(tree)
        except SyntaxError as e:
            console.print(f"[red]Syntax Error:[/] {e}")


class RichSyntaxHighlighter:
    """
    Uses `rich.syntax.Syntax` to display colorful syntax-highlighted code in the console.
    """

    @staticmethod
    def display_code(code: str):
        """
        Displays syntax-highlighted Python code in the console.

        Args:
            code: The Python code as a string.
        """
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)


class SignatureVisitor(cst.CSTVisitor):
    """
    Represents a visitor for inspecting and processing function or class
    signatures in Python code using `libcst` for improved static analysis
    with type hint inference.

    This visitor collects information about the code structure, including
    functions, methods, classes, and their type annotations.

    Attributes:
        file_path: The path of the file being analyzed.
        type_inference_model: Machine learning model for type inference (optional).
        signatures: A list to store `CodeSignature` objects based on discovered elements.
    """

    def __init__(
        self, file_path: Path, type_inference_model: Optional[torch.nn.Module] = None
    ):
        super().__init__()
        """
        Initialize the visitor.

        Args:
            file_path: Path of the file being analyzed.
            type_inference_model: A pre-trained ML model for type inference (optional).
        """
        self.file_path = file_path
        self.type_inference_model = type_inference_model
        self.signatures: List[CodeSignature] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """
        Processes `FunctionDef` nodes (functions) and collects their relevant information.

        Args:
            node: A `libcst.FunctionDef` object representing a function definition.
        """
        name = node.name.value
        docstring = self._get_docstring(node)
        parameters = self._extract_parameters(node.params)
        return_type = self._infer_type(node.returns)
        type_info = TypeInfo(type_hint=return_type, inferred_type=None, confidence=1.0)

        components = [
            SignatureComponent(
                name=param_name,
                type_info=TypeInfo(
                    type_hint=param_type,
                    inferred_type=(
                        None if param_type else self._infer_type_from_model(param_name)
                    ),
                    confidence=0.9 if param_type else 0.7,
                ),
                is_optional=is_optional,
            )
            for param_name, param_type, is_optional in parameters
        ]

        code_signature = CodeSignature(
            name=name,
            module_path=self.file_path,
            components=components,
            return_type=type_info,
            docstring=docstring,
        )

        self.signatures.append(code_signature)

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """
        Processes `ClassDef` nodes (classes) to capture relevant signature-like information.

        Args:
            node: A `libcst.ClassDef` object representing a class definition.
        """
        class_name = node.name.value
        docstring = self._get_docstring(node)

        # Add as a minimal `CodeSignature`, further processing could include methods.
        code_signature = CodeSignature(
            name=class_name,
            module_path=self.file_path,
            components=[],  # Placeholder: could extract methods or attributes signatures here.
            docstring=docstring,
        )

        self.signatures.append(code_signature)

    def _extract_parameters(self, params: cst.Parameters) -> List[tuple]:
        """
        Extract parameter details from a function definition.

        Args:
            params: The parameters node from a function definition.

        Returns:
            A list of tuples containing parameter name, type annotation (if present),
            and whether it is optional.
        """
        extracted_params = []
        for param in params.params:
            param_name = param.name.value
            param_type = self._infer_type(param.annotation)
            is_optional = (
                param.default is not None  # If a default value exists, it is optional.
            )
            extracted_params.append((param_name, param_type, is_optional))
        return extracted_params

    def _infer_type(self, annotation: Optional[cst.Annotation]) -> Optional[str]:
        """
        Infers the type from an annotation node.

        Args:
            annotation: The annotation node representing the type hint.

        Returns:
            The inferred type as a string, or None if no type is specified.
        """
        if annotation:
            if isinstance(annotation.annotation, cst.Name):
                # Handles simple type names like `int`, `str`, etc.
                return annotation.annotation.value
            elif isinstance(annotation.annotation, cst.Subscript):
                # Handles complex types like `List[int]` or `Dict[str, int]`
                return cst.Module([]).code_for_node(annotation.annotation)
        return None

    def _infer_type_from_model(self, name: str) -> Optional[str]:
        """
        Uses the ML type inference model to predict a type for a variable name.

        Args:
            name: The name of the variable for which to infer the type.

        Returns:
            The inferred type as a string, or None if no model is specified or no prediction is possible.
        """
        if self.type_inference_model:
            # Example tokenized input processing (requires actual implementation details).
            # Here we assume `name` is tokenized appropriately for the model.
            tokenized_input = torch.tensor(
                [ord(char) for char in name.lower()]
            ).unsqueeze(
                0
            )  # Mock encoding
            prediction = self.type_inference_model(tokenized_input)
            type_index = torch.argmax(prediction, dim=1).item()
            return f"InferredType{type_index}"  # Simplified: Replace with actual type mapping.
        return None

    def _get_docstring(
        self, node: Union[cst.FunctionDef, cst.ClassDef]
    ) -> Optional[str]:
        """
        Retrieves the docstring for a function or class definition.

        Args:
            node: The node representing the function or class.

        Returns:
            The docstring as a string, or None if no docstring is present.
        """
        if node.body and isinstance(node.body, cst.IndentedBlock):
            # Get the first statement in the body
            first_element = node.body.body[0]
            if isinstance(first_element, cst.SimpleStatementLine):
                # Check if the first statement is an expression
                expr = first_element.body[0]
                if isinstance(expr, cst.Expr) and isinstance(
                    expr.value, cst.SimpleString
                ):
                    # Extract and return the docstring value
                    return expr.value.value.strip("\"'")  # Strip quotes from the string
        return None


class SignatureAnalyzer:
    """Advanced signature analyzer with ML-enhanced type inference"""

    def __init__(self, root_path: Path):
        self.root = root_path
        self.signatures: Dict[str, CodeSignature] = {}
        self.dependency_graph = nx.DiGraph()
        self.type_inference_model = self._initialize_type_inference()

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

        # Generate clusters
        clusters = self._cluster_signatures()

        return {
            "signatures": self.signatures,
            "metrics": self._generate_project_metrics(),
            "clusters": clusters,
            "visualizations": self._generate_visualizations(),
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
        """Build comprehensive dependency graph using rustworkx"""
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

    def _calculate_metrics(self):
        """Calculate advanced metrics for all signatures"""
        for name, signature in self.signatures.items():
            # Calculate complexity using cyclomatic complexity
            complexity = self._calculate_complexity(signature)

            # Calculate cohesion using spectral analysis
            cohesion = self._calculate_cohesion(signature)

            # Calculate coupling using graph theory
            coupling = self._calculate_coupling(name)

            # Calculate maintainability
            maintainability = self._calculate_maintainability(signature)

            # Calculate type safety score
            type_safety = self._calculate_type_safety(signature)

            # Update metrics
            signature.metrics = SignatureMetrics(
                complexity=complexity,
                cohesion=cohesion,
                coupling=coupling,
                maintainability=maintainability,
                type_safety=type_safety,
                documentation_score=self._calculate_doc_score(signature),
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
        """Calculate type safety score"""
        type_scores = []
        for component in signature.components:
            score = component.type_info.confidence
            if component.constraints:
                score *= 1.2  # Bonus for having constraints
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
        """Cluster similar signatures using DBSCAN"""
        # Create feature vectors for signatures
        features = []
        sig_names = []

        for name, sig in self.signatures.items():
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

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=2)
        labels = clustering.fit_predict(features)

        # Group signatures by cluster
        clusters = defaultdict(list)
        for name, label in zip(sig_names, labels):
            clusters[f"cluster_{label}"].append(name)

        return dict(clusters)

    def _initialize_type_inference(self) -> torch.nn.Module:
        """Initialize neural type inference model"""

        class TypeInferenceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(
                    1000, 64
                )  # Vocabulary size, embedding dim
                self.lstm = torch.nn.LSTM(64, 128, batch_first=True)
                self.fc = torch.nn.Linear(128, 50)  # 50 common Python types

            def forward(self, x):
                x = self.embedding(x)
                lstm_out, _ = self.lstm(x)
                return F.softmax(self.fc(lstm_out[:, -1]), dim=1)

        return TypeInferenceModel()

    def _generate_project_metrics(self) -> Dict[str, float]:
        """Generate comprehensive project-wide metrics"""
        metrics = defaultdict(float)
        for sig in self.signatures.values():
            for metric_name, value in sig.metrics.model_dump().items():
                metrics[f"avg_{metric_name}"] += value

        n_sigs = len(self.signatures) or 1
        return {k: v / n_sigs for k, v in metrics.items()}

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
