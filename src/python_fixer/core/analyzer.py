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
import importlib.util
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Core dependencies that are always required
from rich.console import Console
from rich.table import Table

from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple

# Optional dependencies
from python_fixer.core.types import OPTIONAL_DEPS, Any

# Type hints for optional dependencies
if TYPE_CHECKING:
    import numpy
    import numpy.typing as npt
    NDArray = npt.NDArray[numpy.float64]
else:
    NDArray = Any  # type: ignore

# Lazy imports to avoid circular dependencies
_logging = None
if importlib.util.find_spec("logging") is not None:
    import logging as _logging

# Import optional dependencies with detailed error messages
def _import_optional_dependency(name: str, import_path: str, features: str) -> Any:
    """Import an optional dependency with detailed error handling.

    Args:
        name: Name of the dependency
        import_path: Import path to use
        features: Description of features that require this dependency

    Returns:
        The imported module or None if not available
    """
    try:
        if importlib.util.find_spec(import_path) is not None:
            module = __import__(import_path, fromlist=["*"])
            OPTIONAL_DEPS[name] = module
            return module
        print(f"\033[93m⚠ {name} not available - {features} will be disabled\033[0m")
        print(f"  To enable these features, install {name} with: pip install {name}")
    except ImportError as e:
        print(f"\033[93m⚠ Failed to import {name} - {features} will be disabled\033[0m")
        print(f"  Error: {e}")
        print(f"  To enable these features, install {name} with: pip install {name}")
    return None

# Import libcst for code parsing
_libcst = _import_optional_dependency(
    "libcst",
    "libcst",
    "advanced code parsing and transformation"
)

# Import matplotlib for visualization
_matplotlib = _import_optional_dependency(
    "matplotlib",
    "matplotlib.pyplot",
    "dependency graph visualization"
)

# Import mypy for type checking
_mypy = _import_optional_dependency(
    "mypy",
    "mypy.api",
    "static type checking"
)

# Import networkx for graph analysis
_networkx = _import_optional_dependency(
    "networkx",
    "networkx",
    "dependency graph analysis"
)

# Import numpy for numerical computations
_numpy = _import_optional_dependency(
    "numpy",
    "numpy",
    "advanced metrics and analysis"
)

# Import radon for complexity analysis
_radon = _import_optional_dependency(
    "radon",
    "radon.complexity",
    "code complexity analysis"
)

# Import rope for refactoring
_rope = _import_optional_dependency(
    "rope",
    "rope.base.project",
    "code refactoring"
)

# Import sympy for symbolic computation
_sympy = _import_optional_dependency(
    "sympy",
    "sympy",
    "advanced type inference"
)

# Import toml for configuration
_toml = _import_optional_dependency(
    "toml",
    "toml",
    "configuration file parsing"
)

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


class ProjectAnalyzer:
    """Unified system for Python codebase analysis and optimization."""

    def __init__(
        self, root_path: str, config: Optional[Dict[str, Any]] = None, backup: bool = True
    ):
        """Initialize the ProjectAnalyzer.

        Args:
            root_path: Path to the project root directory
            config: Optional configuration dictionary
            backup: Whether to create backups before modifying files

        Raises:
            ValueError: If root_path is invalid or inaccessible
            KeyError: If required config values are missing or invalid
            RuntimeError: If critical dependencies are missing
            Exception: For other initialization errors
        """
        self.logger = _logging.getLogger(__name__) if _logging else None

        try:
            # Validate and initialize paths
            if not root_path:
                raise ValueError("root_path cannot be empty")

            self.root = Path(root_path).resolve()
            if not self.root.exists():
                raise ValueError(f"Project path does not exist: {self.root}")
            if not self.root.is_dir():
                raise ValueError(f"Project path is not a directory: {self.root}")
            if not any(self.root.glob("**/*.py")):
                raise ValueError(f"No Python files found in project: {self.root}")

            # Check write permissions if backup is enabled
            self.backup = backup
            if self.backup and not os.access(self.root, os.W_OK):
                raise ValueError(f"No write permission in project directory: {self.root}")

            # Initialize and validate configuration
            self.config = self._validate_config(config or {})
            if self.logger:
                self.logger.info(
                    "Initializing ProjectAnalyzer",
                    extra={
                        "root_path": str(self.root),
                        "backup_enabled": self.backup,
                        "config": self.config,
                    },
                )

            # Initialize core components
            self.modules: Dict[str, CodeModule] = {}
            self.metrics = ProjectMetrics()

            # Initialize optional components with detailed logging
            self._initialize_optional_components()

            # Initialize optimization components
            self.fix_strategies = self._initialize_fix_strategies()
            self.module_clusters = None
            self.optimal_order = None

            # Set up performance settings
            self._configure_performance_settings()

            if self.logger:
                self.logger.info(
                    "ProjectAnalyzer initialization complete",
                    extra={
                        "available_features": list(OPTIONAL_DEPS.keys()),
                        "max_workers": self.max_workers,
                        "caching_enabled": self.enable_caching,
                    },
                )

        except KeyboardInterrupt:
            if self.logger:
                self.logger.warning("Initialization interrupted by user")
            print("\n\033[93m⚠ Initialization interrupted by user\033[0m")
            raise
        except ValueError as e:
            if self.logger:
                self.logger.error(f"Validation error during initialization: {e}")
            print(f"\n\033[91m✗ {str(e)}\033[0m")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error("Unexpected error during initialization", exc_info=e)
            print(f"\n\033[91m✗ Initialization failed: {e}\033[0m")
            raise

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize configuration.

        Args:
            config: Raw configuration dictionary

        Returns:
            Validated configuration with defaults applied

        Raises:
            KeyError: If required config values are missing or invalid
        """
        validated = config.copy()

        # Validate max_workers
        max_workers = validated.get("max_workers", 4)
        if not isinstance(max_workers, int) or max_workers < 1:
            raise KeyError(f"Invalid max_workers value: {max_workers}. Must be a positive integer.")
        validated["max_workers"] = max_workers

        # Validate enable_caching
        enable_caching = validated.get("enable_caching", True)
        if not isinstance(enable_caching, bool):
            raise KeyError(f"Invalid enable_caching value: {enable_caching}. Must be a boolean.")
        validated["enable_caching"] = enable_caching

        # Validate cache_dir
        cache_dir = validated.get("cache_dir", ".python_fixer_cache")
        if not isinstance(cache_dir, (str, Path)):
            raise KeyError(f"Invalid cache_dir value: {cache_dir}. Must be a string or Path.")
        validated["cache_dir"] = cache_dir

        return validated

    def _initialize_optional_components(self) -> None:
        """Initialize optional dependencies and components."""
        # Initialize graphs if networkx is available
        self.dependency_graph: Optional[Any] = None
        if "networkx" in OPTIONAL_DEPS:
            try:
                self.dependency_graph = OPTIONAL_DEPS["networkx"].DiGraph()
                if self.logger:
                    self.logger.info("Initialized networkx dependency graph")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to initialize networkx: {e}")
                print("\033[93m⚠ networkx initialization failed - graph features disabled\033[0m")

        # Initialize rope project if available
        self.rope_project: Optional[Any] = None
        if "rope" in OPTIONAL_DEPS:
            try:
                self.rope_project = OPTIONAL_DEPS["rope"].Project(str(self.root))
                if self.logger:
                    self.logger.info("Initialized rope project")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to initialize rope: {e}")
                print("\033[93m⚠ rope initialization failed - some features disabled\033[0m")

    def _configure_performance_settings(self) -> None:
        """Configure performance-related settings."""
        self.max_workers = self.config["max_workers"]
        self.enable_caching = self.config["enable_caching"]
        self.cache_dir = Path(self.config["cache_dir"])

        if self.enable_caching:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                if self.logger:
                    self.logger.info(f"Created cache directory: {self.cache_dir}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to create cache directory: {e}")
                print(f"\033[93m⚠ Failed to create cache directory: {e}\033[0m")
                self.enable_caching = False
            raise



    def _initialize_modules(self) -> None:
        """Initialize module tracking."""
        for py_file in self.root.rglob("*.py"):
            module_name = str(py_file.relative_to(self.root)).replace("/", ".")
            self.modules[module_name] = CodeModule(
                name=module_name,
                path=py_file,
            )

    def _initialize_dependency_graph(self) -> None:
        """Initialize the dependency graph if networkx is available."""
        if self.dependency_graph is not None:
            for module in self.modules.values():
                self.dependency_graph.add_node(module.name)

    def _initialize_metrics(self) -> None:
        """Initialize project-wide metrics."""
        self.metrics.total_modules = len(self.modules)

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
            _logging.error("Project analysis failed", exc_info=e)
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

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file using AST or libcst if available.

        This method performs comprehensive analysis of a Python file, including:
        - Import collection using libcst (preferred) or ast
        - Complexity metrics calculation using radon
        - Type checking using mypy
        - Module dependency tracking

        Args:
            file_path: Path to the Python file to analyze
        """
        if not file_path.exists():
            if self.logger:
                self.logger.error(f"File not found: {file_path}")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            module_name = self._get_module_name(file_path)
            imports = self._collect_imports(source)
            metrics = self._calculate_file_metrics(source)

            # Create module node with available metrics
            node = CodeModule(
                name=module_name,
                path=file_path,
                dependencies=imports,
                **metrics
            )
            self.modules[module_name] = node

            # Perform type checking if enabled
            if self.enable_type_checking:
                self._check_types(file_path, node)

            if self.logger:
                self.logger.debug(
                    f"Analyzed {module_name}",
                    extra={
                        "imports": len(imports),
                        "metrics": metrics,
                        "type_errors": len(node.type_errors)
                    }
                )

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error analyzing {file_path}",
                    exc_info=e,
                    extra={"module": module_name if 'module_name' in locals() else None}
                )
            console.print(f"[red]Error analyzing {file_path}: {str(e)}")

    def _collect_imports(self, source: str) -> Set[str]:
        """Collect imports from source code using libcst or ast.

        Args:
            source: Python source code to analyze

        Returns:
            Set of imported module names
        """
        try:
            # Use libcst if available for more accurate parsing
            if OPTIONAL_DEPS["libcst"]:
                cst = OPTIONAL_DEPS["libcst"]["libcst"]
                module = cst.parse_module(source)
                visitor = ImportCollectorVisitor()
                module.visit(visitor)
                return visitor.imports

            # Fallback to basic AST analysis
            tree = ast.parse(source)
            visitor = ASTImportVisitor()
            visitor.visit(tree)
            return visitor.imports

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Import collection failed, using empty set",
                    exc_info=e
                )
            return set()

    def _calculate_file_metrics(self, source: str) -> Dict[str, float]:
        """Calculate complexity and maintainability metrics for source code.

        Args:
            source: Python source code to analyze

        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            "complexity": 0.0,
            "maintainability": 0.0,
            "cyclomatic_complexity": 0
        }

        if not OPTIONAL_DEPS["radon"]:
            return metrics

        try:
            cc_visit = OPTIONAL_DEPS["radon"]["radon.complexity"].cc_visit
            mi_visit = OPTIONAL_DEPS["radon"]["radon.metrics"].mi_visit

            complexity = cc_visit(source)
            maintainability = mi_visit(source, multi=True)
            complexities = [c.complexity for c in complexity]

            if OPTIONAL_DEPS["numpy"]:
                np = OPTIONAL_DEPS["numpy"]
                metrics.update({
                    "complexity": float(np.mean(complexities)) if complexities else 0.0,
                    "maintainability": float(np.mean(maintainability)),
                    "cyclomatic_complexity": int(sum(complexities))
                })
            else:
                metrics.update({
                    "complexity": sum(complexities) / len(complexities) if complexities else 0.0,
                    "maintainability": sum(maintainability) / len(maintainability) if maintainability else 0.0,
                    "cyclomatic_complexity": sum(complexities)
                })

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Metrics calculation failed",
                    exc_info=e
                )

        return metrics

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
                        eigenvals := OPTIONAL_DEPS["sympy"].Matrix(
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

    def _get_mypy_options(self, file_path: Path) -> List[str]:
        """Get mypy configuration options for type checking.

        Args:
            file_path: Path to the Python file to type check

        Returns:
            List of mypy command line options
        """
        return [
            "--strict",  # Enable all strict type checking options
            "--show-error-codes",  # Show error codes in output
            "--pretty",  # Use prettier output
            "--show-column-numbers",  # Show column numbers in errors
            "--check-untyped-defs",  # Check function bodies with no type annotations
            "--warn-redundant-casts",  # Warn about casting that doesn't change type
            "--warn-return-any",  # Warn about returning Any from non-Any typed function
            "--warn-unreachable",  # Warn about unreachable code
            str(file_path)
        ]

    def _process_mypy_output(self, output: str) -> Tuple[List[str], int, int]:
        """Process mypy output and classify errors.

        Args:
            output: Raw mypy output string

        Returns:
            Tuple containing (error_lines, error_count, warning_count)
        """
        errors = []
        error_count = 0
        warning_count = 0

        for line in output.split("\n"):
            if not line or line.startswith("Found"):
                continue

            # Classify error type
            if "error:" in line:
                error_count += 1
            elif "warning:" in line:
                warning_count += 1

            errors.append(line)

        return errors, error_count, warning_count

    def _log_type_check_results(self, node: CodeModule, errors: List[str], error_count: int, warning_count: int) -> None:
        """Log type checking results and update metrics.

        Args:
            node: CodeModule instance being checked
            errors: List of error messages
            error_count: Number of errors found
            warning_count: Number of warnings found
        """
        if self.logger:
            self.logger.info(
                f"Type checking completed for {node.name}",
                extra={
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "total_issues": len(errors),
                    "first_error": errors[0] if errors else None
                }
            )

        # Update project-wide metrics
        self.metrics.type_error_count += error_count

    def _check_types(self, file_path: Path, node: CodeModule) -> None:
        """Perform type checking and validation using mypy.

        This method runs mypy with strict type checking options and collects detailed error information.
        It handles missing dependencies gracefully and provides comprehensive logging.

        Args:
            file_path: Path to the Python file to type check
            node: CodeModule instance to store type checking results
        """
        if not OPTIONAL_DEPS["mypy"]:
            if self.logger:
                self.logger.debug(
                    "Skipping type checking - mypy not available",
                    extra={"module": node.name}
                )
            return

        try:
            # Run mypy with configured options
            results = OPTIONAL_DEPS["mypy"].run(self._get_mypy_options(file_path))

            # Process mypy output
            if results[0]:  # mypy output is available
                errors, error_count, warning_count = self._process_mypy_output(results[0])
                node.type_errors = errors
                self._log_type_check_results(node, errors, error_count, warning_count)

        except FileNotFoundError:
            if self.logger:
                self.logger.error(
                    "mypy executable not found in PATH",
                    extra={"module": node.name}
                )
            console.print("[red]Error: mypy not found in PATH. Please install mypy.")

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Type checking failed",
                    exc_info=e,
                    extra={"module": node.name}
                )
            console.print(f"[red]Error during type checking of {node.name}: {str(e)}")

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
            if "toml" not in OPTIONAL_DEPS:
                console.print("[yellow]Warning: toml not available, using empty config")
                return {}
            return OPTIONAL_DEPS["toml"].load(config_path)
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


    def initialize_project(self) -> None:
        """Initialize a new project for analysis.

        This method sets up the necessary project structure and configuration.
        """
        try:
            self._initialize_project_components()
        except KeyboardInterrupt:
            print("\nProject initialization interrupted by user")
            raise
        except Exception as e:
            print(f"Error during project initialization: {str(e)}")
            if _logging:
                _logging.error("Project initialization failed", exc_info=True)
            raise

    def _initialize_project_components(self) -> None:
        """Initialize all project components including cache, modules, and backups."""
        try:
            print(f"Initializing project at {self.root}")
            print(f"Config: {self.config}")
            print(f"Cache enabled: {self.enable_caching}")

            # Create cache directory if enabled
            if self.enable_caching:
                self._create_cache_directory()

            # Initialize core components
            self._initialize_modules()
            self._initialize_dependency_graph()
            self._initialize_metrics()

            # Create backup if enabled
            if self.backup:
                self._create_backup_directory()

            print("Project initialization complete")

        except KeyboardInterrupt:
            print("\nProject component initialization interrupted by user")
            raise
        except Exception as e:
            print(f"Error during project component initialization: {str(e)}")
            raise

    def _create_cache_directory(self) -> None:
        """Create a cache directory for the project."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created cache directory at {self.cache_dir}")

    def _create_backup_directory(self) -> None:
        """Create a backup directory for the project."""
        backup_dir = self.root / ".python_fixer_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created backup directory at {backup_dir}")

    def _initialize_fix_strategies(self) -> Dict[str, Any]:
        """Initialize the available fix strategies.

        Returns:
            Dictionary mapping strategy names to their implementations.
        """
        return {
            "circular_deps": self._fix_circular_dependencies,
            "unused_imports": self._fix_unused_imports,
            "type_hints": self._fix_type_hints,
            "docstrings": self._fix_docstrings,
        }

    def _fix_circular_dependencies(self, module: CodeModule) -> bool:
        """Fix circular dependencies in a module.

        Args:
            module: Module to fix

        Returns:
            True if any fixes were applied
        """
        # Placeholder for circular dependency fixes
        return False

    def _fix_unused_imports(self, module: CodeModule) -> bool:
        """Fix unused imports in a module.

        Args:
            module: Module to fix

        Returns:
            True if any fixes were applied
        """
        # Placeholder for unused import fixes
        return False

    def _fix_type_hints(self, module: CodeModule) -> bool:
        """Fix type hints in a module.

        Args:
            module: Module to fix

        Returns:
            True if any fixes were applied
        """
        # Placeholder for type hint fixes
        return False

    def _fix_docstrings(self, module: CodeModule) -> bool:
        """Fix docstrings in a module.

        Args:
            module: Module to fix

        Returns:
            True if any fixes were applied
        """
        # Placeholder for docstring fixes
        return False

    def _setup_logging(self) -> None:
        """Setup logging configuration for the analyzer."""
        if _logging is not None:
            _logging.basicConfig(
                level=_logging.DEBUG if self.config.get("verbose") else _logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    _logging.StreamHandler(),
                    _logging.FileHandler("python_fixer.log"),
                ],
            )


class ImportCollectorVisitor(_libcst.CSTVisitor if _libcst is not None else object):
    """Visitor to collect imports using libcst"""

    def __init__(self):
        if _libcst is not None:
            super().__init__()
        self.imports: Set[str] = set()
        self.type_errors: List[str] = []
        self.type_checking_enabled = False
        self.type_checking_errors = []
        self.type_checking_warnings = []
        self.type_checking_ignored = []

    def visit_Import(self, node: '_libcst.Import') -> None:
        if OPTIONAL_DEPS["libcst"] is not None:
            for name in node.names:
                self.imports.add(name.name.value)

    def visit_ImportFrom(self, node: '_libcst.ImportFrom') -> None:
        if OPTIONAL_DEPS["libcst"] is not None and node.module:
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
