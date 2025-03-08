"""
Enhanced Python Project Analyzer with advanced static analysis and dependency tracking.
Features:
- Static analysis using libcst for accurate code parsing
- Dynamic import resolution and validation
- Cyclomatic complexity analysis
- Dead code detection
- Import graph visualization
- Type hint validation
- Dependency chain analysis
"""

import ast
import contextlib
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Core dependencies that are always required
from rich.console import Console
from rich.table import Table

# Optional dependencies - these will be imported lazily
import matplotlib.pyplot as plt
import sympy
import toml
import libcst as cst
import mypy.api as mypy
from radon import complexity as radon_cc
from rope.base import project as rope_project

# Initialize optional dependencies
OPTIONAL_DEPS = {
    "libcst": cst,
    "matplotlib": plt,
    "mypy": mypy,
    "networkx": None,  # Will be imported on demand
    "numpy": None,  # Will be imported on demand
    "sympy": sympy,
    "toml": toml,
    "radon": radon_cc,
    "rope": rope_project,
}

# Try to import numpy for type hints
try:
    import numpy
    import numpy.typing as npt

    OPTIONAL_DEPS["numpy"] = numpy
    NDArray = npt.NDArray[numpy.float64]
except ImportError:
    NDArray = None  # Type will be Any

# Advanced console for rich output
console = Console()


@dataclass
class CodeModule:
    """Represents a Python module with comprehensive metadata."""

    name: str
    path: Path
    # Analysis metrics
    complexity: float = 0.0
    maintainability: float = 0.0
    cyclomatic_complexity: int = 0
    loc: int = 0
    docstring_coverage: float = 0.0
    cohesion_score: float = 0.0
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    imported_by: Set[str] = field(default_factory=set)
    circular_deps: List[str] = field(default_factory=list)
    # Type information
    type_errors: List[str] = field(default_factory=list)
    type_coverage: float = 0.0
    # Fix tracking
    fixes_applied: List[str] = field(default_factory=list)
    last_modified: datetime = field(default_factory=datetime.now)
    backup_path: Optional[Path] = None


@dataclass
class ProjectMetrics:
    """Comprehensive project-wide metrics."""

    # Module statistics
    total_modules: int = 0
    total_imports: int = 0
    total_loc: int = 0
    # Complexity metrics
    avg_complexity: float = 0.0
    max_complexity: float = 0.0
    complexity_distribution: Dict[str, float] = field(default_factory=dict)
    # Dependency metrics
    circular_deps: List[List[str]] = field(default_factory=list)
    import_depth: Dict[str, int] = field(default_factory=dict)
    modularity_score: float = 0.0
    coupling_matrix: Optional[NDArray] = None
    # Type system metrics
    type_coverage: float = 0.0
    type_error_count: int = 0
    # Performance metrics
    analysis_time: float = 0.0
    optimization_time: float = 0.0
    # Fix metrics
    fixes_applied: int = 0
    fixes_failed: int = 0
    files_modified: Set[str] = field(default_factory=set)


class ASTImportVisitor(ast.NodeVisitor):
    """Basic AST visitor for collecting imports."""

    def __init__(self):
        super().__init__()
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        for name in node.names:
            self.imports.add(name.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.add(node.module)


class CodebaseOptimizer:
    """Unified system for Python codebase analysis and optimization."""

    def __init__(
        self, root_path: Path, config_path: Optional[Path] = None, backup: bool = True
    ):
        # Initialize paths and configuration
        self.root = root_path.resolve()
        self.backup = backup
        self.config = self._load_config(config_path) if config_path else {}

        # Initialize core components
        self.modules: Dict[str, CodeModule] = {}
        self.metrics = ProjectMetrics()

        # Initialize graphs if networkx is available
        self.dependency_graph = None
        try:
            import networkx

            OPTIONAL_DEPS["networkx"] = networkx
            self.dependency_graph = networkx.DiGraph()
        except ImportError:
            console.print(
                "[yellow]networkx not available - some features will be disabled"
            )

        # Initialize rope project if available
        self.rope_project = None
        if OPTIONAL_DEPS["rope"]:
            try:
                self.rope_project = OPTIONAL_DEPS["rope"].Project(str(self.root))
            except AttributeError:
                console.print(
                    "[yellow]rope project initialization failed - some features will be disabled"
                )
        else:
            console.print("[yellow]rope not available - some features will be disabled")

        # Initialize optimization components
        self.fix_strategies = self._initialize_fix_strategies()
        self.module_clusters = None
        self.optimal_order = None

        # Performance settings
        self.max_workers = self.config.get("max_workers", 4)
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_dir = Path(self.config.get("cache_dir", ".python_fixer_cache"))

        # Analysis settings
        self.enable_type_checking = self.config.get("enable_type_checking", True)
        self.enable_complexity_analysis = self.config.get(
            "enable_complexity_analysis", True
        )

        # Setup logging
        self._setup_logging()

    def analyze_project(self) -> ProjectMetrics:
        """Perform comprehensive project analysis"""
        try:
            return self._execute_project_analysis()
        except Exception as e:
            console.print(f"[red]Error during project analysis: {str(e)}")
            logging.error("Project analysis failed", exc_info=e)
            return ProjectMetrics()

    def _execute_project_analysis(self):
        console.rule("[bold blue]Starting Enhanced Project Analysis")

        # Parallel file discovery and initial parsing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            python_files = list(self.root.rglob("*.py"))
            list(executor.map(self._analyze_file, python_files))

        # Build dependency graph
        if self.dependency_graph is not None:
            self._build_dependency_graph()
            self._analyze_circular_dependencies()
        else:
            console.print(
                "[yellow]Skipping dependency analysis - networkx not available"
            )

        # Advanced analysis steps
        self._calculate_cohesion_metrics()

        # Type analysis
        try:
            self._analyze_type_coverage()
        except (ImportError, AttributeError):
            console.print("[yellow]Skipping type analysis - mypy not available")

        # Complexity analysis
        try:
            self._calculate_complexity_metrics()
        except (ImportError, AttributeError):
            console.print("[yellow]Skipping complexity analysis - radon not available")

        # Generate comprehensive metrics
        self._generate_metrics()

        # Visualize results if matplotlib is available
        if OPTIONAL_DEPS["matplotlib"]:
            self._generate_visualizations()
        else:
            console.print("[yellow]Skipping visualizations - matplotlib not available")

        return self.metrics

    def _analyze_type_coverage(self):
        """Analyze type coverage across all modules in the project."""
        total_annotations = 0
        valid_annotations = 0

        for module_name, node in self.modules.items():
            try:
                with open(node.path, "r", encoding="utf-8") as file:
                    source_code = file.read()

                # Parse the source using ast to count type annotations
                tree = ast.parse(source_code)
                visitor = TypeAnnotationVisitor()
                visitor.visit(tree)

                # Update counts
                total_annotations += visitor.total_annotations
                valid_annotations += visitor.valid_annotations

            except Exception as e:
                console.print(
                    f"[red]Error analyzing type coverage for {module_name}: {e}"
                )

        # Calculate and assign type coverage metrics
        self.metrics.type_coverage = (
            valid_annotations / total_annotations if total_annotations > 0 else 0.0
        )

        console.print(
            f"Type Coverage: {self.metrics.type_coverage:.2%} "
            f"({valid_annotations}/{total_annotations})"
        )

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file using AST or libcst if available"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            # Use libcst if available, otherwise fallback to ast
            if OPTIONAL_DEPS["libcst"]:
                cst = OPTIONAL_DEPS["libcst"]["libcst"]
                module = cst.parse_module(source)
                visitor = ImportCollectorVisitor()
                module.visit(visitor)
            else:
                # Fallback to basic AST analysis
                tree = ast.parse(source)
                visitor = ASTImportVisitor()
                visitor.visit(tree)

            console.print(f"[green]Successfully analyzed {file_path}")

        except Exception as e:
            console.print(f"[red]Error analyzing {file_path}: {str(e)}")
            logging.error(f"File analysis failed for {file_path}", exc_info=e)

            # Calculate complexity metrics if radon is available
            if OPTIONAL_DEPS["radon"]:
                cc_visit = OPTIONAL_DEPS["radon"]["radon.complexity"].cc_visit
                mi_visit = OPTIONAL_DEPS["radon"]["radon.metrics"].mi_visit
                complexity = cc_visit(source)
                maintainability = mi_visit(source, multi=True)

                # Create module node with complexity metrics
                module_name = self._get_module_name(file_path)
                if OPTIONAL_DEPS["numpy"]:
                    np = OPTIONAL_DEPS["numpy"]
                    node = CodeModule(
                        name=module_name,
                        path=file_path,
                        dependencies=visitor.imports,
                        complexity=float(np.mean([c.complexity for c in complexity])),
                        maintainability=np.mean(maintainability),
                        cyclomatic_complexity=sum(c.complexity for c in complexity),
                    )
                else:
                    # Fallback without numpy
                    complexities = [c.complexity for c in complexity]
                    node = CodeModule(
                        name=module_name,
                        path=file_path,
                        dependencies=visitor.imports,
                        complexity=(
                            sum(complexities) / len(complexities)
                            if complexities
                            else 0.0
                        ),
                        maintainability=(
                            sum(maintainability) / len(maintainability)
                            if maintainability
                            else 0.0
                        ),
                        cyclomatic_complexity=sum(c.complexity for c in complexity),
                    )
            else:
                # Create basic module node without complexity metrics
                module_name = self._get_module_name(file_path)
                node = CodeModule(
                    name=module_name,
                    path=file_path,
                    dependencies=visitor.imports,
                )

            self.modules[module_name] = node

            # Type checking if enabled and mypy is available
            if self.enable_type_checking and OPTIONAL_DEPS["mypy"]:
                self._check_types(file_path, node)

        except Exception as e:
            console.print(f"[red]Error analyzing {file_path}: {str(e)}")

    def _build_dependency_graph(self):
        """Build comprehensive dependency graph using networkx"""
        if not OPTIONAL_DEPS["networkx"] or not self.dependency_graph:
            console.print("[yellow]Skipping dependency graph - networkx not available")
            return

        for module_name, node in self.modules.items():
            self.dependency_graph.add_node(
                module_name,
                complexity=node.complexity,
                maintainability=node.maintainability,
            )

            for dep in node.dependencies:
                self.dependency_graph.add_edge(module_name, dep)

        # Create a partition using a community detection algorithm
        try:
            from networkx.algorithms.community import (
                greedy_modularity_communities,
                modularity,
            )

            communities = list(greedy_modularity_communities(self.dependency_graph))
            self.metrics.modularity_score = modularity(
                self.dependency_graph, communities
            )
        except ImportError:
            console.print(
                "[yellow]Skipping community detection - algorithm not available"
            )

        # Calculate coupling matrix if numpy is available
        if OPTIONAL_DEPS["numpy"]:
            self._calculate_coupling_matrix()

    def _calculate_coupling_matrix(self):
        np = OPTIONAL_DEPS["numpy"]["numpy"]
        n = len(self.modules)
        coupling_matrix = np.zeros((n, n))
        module_indices = {name: i for i, name in enumerate(self.modules)}

        for module_name, node in self.modules.items():
            i = module_indices[module_name]
            for dep in node.dependencies:
                if dep in module_indices:
                    coupling_matrix[i, module_indices[dep]] = 1

        self.metrics.coupling_matrix = coupling_matrix

    def _analyze_circular_dependencies(self):
        """Detect and analyze circular dependencies with advanced cycle detection."""
        with contextlib.suppress(OPTIONAL_DEPS["networkx"].NetworkXNoCycle):
            if cycles := list(
                OPTIONAL_DEPS["networkx"].simple_cycles(self.dependency_graph)
            ):
                self.metrics.circular_deps = [
                    cycle for cycle in cycles if len(cycle) > 1
                ]

            # Calculate cycle metrics
            if cycle_lengths := [len(cycle) for cycle in self.metrics.circular_deps]:
                if OPTIONAL_DEPS["numpy"]:
                    np = OPTIONAL_DEPS["numpy"]
                    self.metrics.complexity_distribution["cycle_length_avg"] = np.mean(
                        cycle_lengths
                    )
                    self.metrics.complexity_distribution["cycle_length_max"] = max(
                        cycle_lengths
                    )
                else:
                    # Fallback without numpy
                    self.metrics.complexity_distribution["cycle_length_avg"] = (
                        sum(cycle_lengths) / len(cycle_lengths)
                        if cycle_lengths
                        else 0.0
                    )
                    self.metrics.complexity_distribution["cycle_length_max"] = max(
                        cycle_lengths, default=0.0
                    )

            # Find strongly connected components (groups of mutually dependent modules)
            if sccs := list(
                OPTIONAL_DEPS["networkx"].strongly_connected_components(
                    self.dependency_graph
                )
            ):
                large_sccs = [scc for scc in sccs if len(scc) > 1]

            # Calculate component coupling scores
            for scc in large_sccs:
                subgraph = self.dependency_graph.subgraph(scc)
                coupling_score = OPTIONAL_DEPS["networkx"].density(subgraph)
                for module_name in scc:
                    if module_name in self.modules:
                        self.modules[module_name].cohesion_score = coupling_score

            # Advanced circular dependency analysis
            if self.metrics.circular_deps:
                self._suggest_dependency_fixes()

    def _suggest_dependency_fixes(self):
        """Generate intelligent suggestions for fixing circular dependencies."""
        suggestions = []
        for cycle in self.metrics.circular_deps:
            # Analyze cycle characteristics
            cycle_subgraph = self.dependency_graph.subgraph(cycle)
            most_connected = max(cycle, key=lambda n: cycle_subgraph.degree(n))
            edge_count = cycle_subgraph.number_of_edges()

            # Calculate cycle-specific metrics
            target_module = max(
                (
                    (module, self.modules[module].complexity)
                    for module in cycle
                    if module in self.modules
                ),
                key=lambda x: x[1],
                default=(most_connected, 0),
            )[0]

            # Generate targeted suggestions
            if edge_count > len(cycle) * 1.5:  # Dense cycle
                # Analyze common interfaces and generate suggestions
                if interfaces := self._extract_common_interfaces(cycle):
                    interface_suggestions = [
                        {
                            "type": "extract_interface",
                            "modules": cycle,
                            "interface_name": interface_name,
                            "methods": methods,
                            "suggestion": f"Create Protocol '{interface_name}' with methods {', '.join(methods)}",
                        }
                        for interface_name, methods in interfaces.items()
                    ]
                    suggestions.extend(interface_suggestions)
                    for suggestion_data in interface_suggestions:
                        console.print(
                            f"[yellow]Suggestion: {suggestion_data['suggestion']}"
                        )
                else:
                    suggestion = f"Create an interface to abstract common functionality from {' -> '.join(cycle)}"
                    suggestions.append(
                        {
                            "type": "extract_interface",
                            "modules": cycle,
                            "suggestion": suggestion,
                        }
                    )
                    console.print(f"[yellow]Suggestion: {suggestion}")
            else:  # Sparse cycle
                suggestion = f"Split {target_module} into smaller modules to break the dependency cycle"
                suggestions.append(
                    {
                        "type": "split_module",
                        "module": target_module,
                        "suggestion": suggestion,
                    }
                )
                console.print(f"[yellow]Suggestion: {suggestion}")

            # Check for common anti-patterns
            if deps := [
                self.modules[m].dependencies for m in cycle if m in self.modules
            ]:
                if common_imports := set.intersection(*deps):
                    suggestion = f"Extract common dependencies: {common_imports}"
                    suggestions.append(
                        {
                            "type": "extract_common",
                            "modules": cycle,
                            "dependencies": list(common_imports),
                            "suggestion": suggestion,
                        }
                    )
                    console.print(f"  {suggestion}")

        return suggestions

    def _extract_common_interfaces(self, modules: List[str]) -> Dict[str, List[str]]:
        """Extract common interfaces from a set of modules.

        Args:
            modules: List of module names to analyze

        Returns:
            Dictionary mapping interface names to lists of method names
        """
        interfaces = {}

        # Collect all class definitions and method signatures
        for module_name in modules:
            if module_name not in self.modules:
                continue

            module_path = self.modules[module_name].path
            try:
                with open(module_path, "r") as f:
                    tree = ast.parse(f.read())

                # Find all classes and their methods
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        methods = []
                        for child in node.body:
                            if isinstance(child, ast.FunctionDef):
                                # Extract method signature
                                args = [
                                    arg.arg
                                    for arg in child.args.args
                                    if arg.arg != "self"
                                ]
                                returns = None
                                if child.returns:
                                    returns = ast.unparse(child.returns)

                                method_sig = {
                                    "name": child.name,
                                    "args": args,
                                    "returns": returns,
                                    "docstring": ast.get_docstring(child),
                                }
                                methods.append(method_sig)

                        # Group similar methods across classes
                        if methods:
                            interface_name = f"{node.name}Interface"
                            interfaces.setdefault(interface_name, []).extend(
                                [m["name"] for m in methods]
                            )

            except Exception as e:
                console.print(f"[red]Error analyzing {module_name}: {e}")

        # Filter to find common methods
        common_interfaces = {}
        for interface_name, methods in interfaces.items():
            # Count method occurrences and keep methods that appear in multiple classes
            if common_methods := [
                m for m, count in Counter(methods).items() if count > 1
            ]:
                common_interfaces[interface_name] = common_methods

        return common_interfaces

    def _generate_protocol(self, interface_name: str, methods: List[Dict]) -> str:
        """Generate a Protocol class definition for an interface.

        Args:
            interface_name: Name of the interface
            methods: List of method signatures

        Returns:
            String containing the Protocol class definition
        """
        lines = [
            "from typing import Protocol",
            "",
            f"class {interface_name}(Protocol):",
            f'    """Protocol defining the interface for {interface_name}."""\n',
        ]

        for method in methods:
            args_str = ", ".join(["self"] + method["args"])
            returns_str = f" -> {method['returns']}" if method["returns"] else ""

            if method["docstring"]:
                lines.extend(
                    [
                        f"    def {method['name']}({args_str}){returns_str}:",
                        f'        """{method["docstring"]}"""',
                        "        ...",
                    ]
                )
            else:
                lines.extend(
                    [f"    def {method['name']}({args_str}){returns_str}: ...", ""]
                )

        return "\n".join(lines)

    def _calculate_cohesion_metrics(self):
        """Calculate advanced cohesion metrics using spectral graph theory"""
        for module_name, node in self.modules.items():
            # Create subgraph and calculate metrics if dependencies exist
            if deps := list(node.dependencies):
                if (
                    subgraph := self.dependency_graph.subgraph([module_name] + deps)
                ) and len(subgraph) > 1:
                    # Calculate Laplacian matrix and eigenvalues
                    if (
                        eigenvals := sympy.Matrix(
                            OPTIONAL_DEPS["networkx"]
                            .laplacian_matrix(subgraph)
                            .todense()
                        ).eigenvals()
                    ) and (
                        sorted_eigenvals := sorted(
                            float(v.real) for v in eigenvals.keys()
                        )
                    ):
                        node.cohesion_score = (
                            sorted_eigenvals[1] if len(sorted_eigenvals) > 1 else 0.0
                        )

    def _check_types(self, file_path: Path, node: CodeModule):
        """Perform type checking and validation"""
        if OPTIONAL_DEPS["mypy"]:
            results = OPTIONAL_DEPS["mypy"].run([str(file_path)])
            if results[0]:  # mypy output
                if errors := results[0].split("\n"):
                    node.type_errors = [
                        error
                        for error in errors
                        if error and not error.startswith("Found")
                    ]
        else:
            console.print("[yellow]Skipping type checking - mypy not available")

    def _calculate_complexity_metrics(self):
        """Calculate advanced complexity and maintainability metrics"""
        if not OPTIONAL_DEPS["radon"]:
            console.print("[yellow]Skipping complexity metrics - radon not available")
            return

        complexities = []
        for node in self.modules.values():
            # Calculate weighted complexity based on multiple factors
            weighted_complexity = (
                node.cyclomatic_complexity * 0.4
                + (1 - node.maintainability / 100) * 0.3
                + len(node.dependencies) * 0.3
            )
            complexities.append(weighted_complexity)
            node.complexity = weighted_complexity

        if OPTIONAL_DEPS["numpy"]:
            self.metrics.avg_complexity = (
                OPTIONAL_DEPS["numpy"].mean(complexities) if complexities else 0.0
            )
        else:
            # Fallback without numpy
            self.metrics.avg_complexity = (
                sum(complexities) / len(complexities) if complexities else 0.0
            )
        # Calculate import depth using longest paths
        for module_name in self.modules:
            try:
                paths = OPTIONAL_DEPS["networkx"].single_source_shortest_path_length(
                    self.dependency_graph, module_name
                )
                self.metrics.import_depth[module_name] = max(paths.values())
            except OPTIONAL_DEPS["networkx"].NetworkXError:
                self.metrics.import_depth[module_name] = 0

    def _generate_metrics(self):
        """Generate comprehensive analysis metrics"""
        self.metrics.total_modules = len(self.modules)
        self.metrics.total_imports = self.dependency_graph.number_of_edges()

        # Calculate additional graph metrics
        with contextlib.suppress(OPTIONAL_DEPS["networkx"].NetworkXError):
            # Eigenvector centrality for module importance
            centrality = OPTIONAL_DEPS["networkx"].eigenvector_centrality_numpy(
                self.dependency_graph
            )

            # Update module scores with centrality
            for module_name, score in centrality.items():
                if module_name in self.modules:
                    self.modules[module_name].cohesion_score = score

    def _generate_visualizations(self):
        """Generate advanced visualizations of the dependency graph"""
        if not (OPTIONAL_DEPS["matplotlib"] and OPTIONAL_DEPS["networkx"]):
            console.print(
                "[yellow]Skipping visualization - matplotlib or networkx not available"
            )
            return

        plt = OPTIONAL_DEPS["matplotlib"].pyplot
        plt.figure(figsize=(15, 10))

        # Use force-directed layout for better visualization
        pos = OPTIONAL_DEPS["networkx"].kamada_kawai_layout(self.dependency_graph)

        # Node sizes based on complexity
        node_sizes = [
            (self.modules[node].complexity * 1000 if node in self.modules else 100)
            for node in self.dependency_graph.nodes()
        ]

        # Node colors based on type errors
        node_colors = [
            (
                "red"
                if node in self.modules and self.modules[node].type_errors
                else "lightblue"
            )
            for node in self.dependency_graph.nodes()
        ]

        # Draw the graph
        OPTIONAL_DEPS["networkx"].draw(
            self.dependency_graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            font_weight="bold",
            arrows=True,
            edge_color="gray",
            alpha=0.7,
        )

        # Add title and metadata
        plt.title(
            "Module Dependency Graph\n(Node size = complexity, Red = type errors)"
        )

        # Save with high resolution
        plt.savefig(
            self.root / "dependency_graph.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    def generate_report(self) -> str:
        """Generate a comprehensive analysis report with Rich formatting"""
        # Create tables for different aspects of the analysis
        module_table = Table(title="Module Analysis")
        module_table.add_column("Module", justify="left", style="cyan")
        module_table.add_column("Complexity", justify="right")
        module_table.add_column("Dependencies", justify="right")
        module_table.add_column("Type Errors", justify="right", style="red")

        for name, node in sorted(self.modules.items()):
            module_table.add_row(
                name,
                f"{node.complexity:.2f}",
                str(len(node.dependencies)),
                str(len(node.type_errors)),
            )

        # Create summary table
        summary_table = Table(title="Analysis Summary")
        summary_table.add_column("Metric", style="blue")
        summary_table.add_column("Value")

        summary_table.add_row("Total Modules", str(self.metrics.total_modules))
        summary_table.add_row("Total Dependencies", str(self.metrics.total_imports))
        summary_table.add_row(
            "Average Complexity", f"{self.metrics.avg_complexity:.2f}"
        )
        summary_table.add_row("Type Coverage", f"{self.metrics.type_coverage:.1%}")
        summary_table.add_row(
            "Modularity Score", f"{self.metrics.modularity_score:.2f}"
        )

        # Print tables
        console.print(summary_table)
        console.print(module_table)

        # Return markdown version for file output
        return self._generate_markdown_report()

    def _generate_markdown_report(self) -> str:
        """Generate a markdown version of the report"""
        sections = [
            "# Python Project Analysis Report\n",
            "## Project Overview",
            f"- **Total Modules:** {self.metrics.total_modules}",
            f"- **Total Dependencies:** {self.metrics.total_imports}",
            *(
                f"- **Average Complexity:** {self.metrics.avg_complexity:.2f}",
                f"- **Type Coverage:** {self.metrics.type_coverage:.1%}",
                f"- **Modularity Score:** {self.metrics.modularity_score:.2f}\n",
            ),
        ]

        # Circular dependencies section
        if self.metrics.circular_deps:
            sections.append("## Circular Dependencies")
            sections.extend(
                f"- `{' -> '.join(cycle)} -> {cycle[0]}`"
                for cycle in self.metrics.circular_deps
            )
            sections.append("")

        # Module details section
        sections.append("## Module Details")
        for name, node in sorted(self.modules.items()):
            return self.named_dependencies(sections, name, node)

    def named_dependencies(self, sections, name, node):
        sections.extend((f"\n### {name}", f"- **Complexity:** {node.complexity:.2f}"))
        sections.append(f"- **Dependencies:** {len(node.dependencies)}")
        if node.dependencies:
            sections.append("  ```")
            sections.extend(f"  - {dep}" for dep in sorted(node.dependencies))
            sections.append("  ```")
        if node.type_errors:
            sections.extend(("- **Type Errors:**", "  ```"))
            sections.extend(f"  - {error}" for error in node.type_errors)
            sections.extend(("  ```", "- **Type Errors:**", "  ```"))
            sections.extend(f"  - {error}" for error in node.type_errors)
            sections.append("  ```")
        sections.extend(
            (
                f"- **Import Depth:** {self.metrics.import_depth[name]}",
                f"- **Cohesion Score:** {node.cohesion_score:.2f}",
            )
        )
        # Generate recommendations section
        sections.append("\n## Recommendations")

        if high_complexity := [
            (name, node)
            for name, node in self.modules.items()
            if node.complexity > self.metrics.avg_complexity * 1.5
        ]:
            sections.append("\n### High Complexity Modules")
            sections.append("Consider refactoring these modules to reduce complexity:")
            for name, node in sorted(
                high_complexity, key=lambda x: x[1].complexity, reverse=True
            ):
                sections.append(f"- `{name}` (Complexity: {node.complexity:.2f})")

        if modules_with_type_errors := [
            name for name, node in self.modules.items() if node.type_errors
        ]:
            sections.append("\n### Type Checking Issues")
            sections.append("Add type hints to improve code safety in:")
            for name in sorted(modules_with_type_errors):
                sections.append(f"- `{name}`")

        return "\n".join(sections)

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        """Load configuration from TOML file"""
        try:
            return toml.load(config_path)
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not load config from {config_path}: {e}"
            )
            return {}

    @staticmethod
    def _get_module_name(file_path: Path) -> str:
        """Convert file path to module name"""
        parts = list(file_path.parts)
        if "__init__.py" in parts:
            parts.remove("__init__.py")
        return ".".join(parts[:-1] + [file_path.stem])


class ImportCollectorVisitor(cst.CSTVisitor):
    """Visitor to collect imports using libcst"""

    def __init__(self):
        super().__init__()
        self.imports: Set[str] = set()
        self.type_errors: List[str] = []
        self.type_checking_enabled = False
        self.type_checking_errors = []
        self.type_checking_warnings = []
        self.type_checking_ignored = []

    def visit_Import(self, node: cst.Import):
        for name in node.names:
            self.imports.add(name.name.value)

    def visit_ImportFrom(self, node: cst.ImportFrom):
        if node.module:
            self.imports.add(node.module.value)


class TypeAnnotationVisitor(ast.NodeVisitor):
    """Visitor to analyze type annotations in the AST."""

    def __init__(self):
        super().__init__()
        self.total_annotations = 0
        self.valid_annotations = 0

    def visit_AnnAssign(self, node):
        """Handles variable annotations."""
        self.total_annotations += 1
        if isinstance(node.annotation, ast.Name):
            self.valid_annotations += 1

    def visit_FunctionDef(self, node):
        """Handles function argument and return type annotations."""
        # Check return type annotation
        if node.returns:
            self.total_annotations += 1
            if isinstance(node.returns, ast.Name):
                self.valid_annotations += 1

        # Check argument type annotations
        for arg in node.args.args:
            if arg.annotation:
                self.total_annotations += 1
                if isinstance(arg.annotation, ast.Name):
                    self.valid_annotations += 1
