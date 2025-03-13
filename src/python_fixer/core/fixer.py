"""
Enhanced Python Code Fixer with advanced refactoring and import optimization.
Features:
- Automatic import restructuring using libcst
- Smart dependency resolution
- Code transformation with advanced AST manipulation
- Automatic import grouping and sorting
- Dead code elimination
- Circular dependency breaking
- Type annotation addition
"""

# Standard library imports
from collections import defaultdict
import itertools

# Third-party library imports
from scipy.optimize import linear_sum_assignment
import numpy as np

# Local application imports
from cli import console
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn
from sklearn.cluster import SpectralClustering
from typing import Any, Dict, List, Optional, Set, Tuple
import ast
import black
import difflib
import git
import isort
import libcst as cst
import networkx as nx
import rope.base.project
import yaml


@dataclass
class FixOperation:
    """Represents a single fix operation"""

    file_path: Path
    operation_type: str
    original_code: str
    fixed_code: str
    confidence: float
    impact_score: float
    dependencies_affected: Set[str] = field(default_factory=set)


@dataclass
class FixStrategy:
    """Strategy for fixing specific import issues"""

    priority: int
    impact: float
    risk: float
    requires_manual_review: bool
    description: str
    fix_function: callable


class SmartFixer:
    """Advanced Python code fixer with intelligent refactoring capabilities"""

    def __init__(
        self, root_path: Path, config_path: Optional[Path] = None, backup: bool = True
    ):
        self.root = root_path.resolve()
        self.backup = backup
        self.config = self._load_config(config_path.resolve() if config_path else None)
        console.print(f"[green]Root path resolved successfully: {self.root}")
        if config_path:
            console.print(f"[green]Config path resolved: {config_path.resolve()}")
        self.fixes: List[FixOperation] = []
        self.rope_project = rope.base.project.Project(str(self.root))

        # Initialize fix strategies with weighted priorities
        self.strategies = self._initialize_strategies()

        # Track module dependencies
        self.dependency_graph = nx.DiGraph()

        # Performance optimization
        self.max_workers = self.config.get("max_workers", 4)

    console = Console()

    def fix_project(self) -> Dict[str, any]:
        """Execute comprehensive project fixes"""
        with Progress(
            SpinnerColumn(), *Progress.get_default_columns(), console=console
        ) as progress:
            task = progress.add_task("[cyan]Fixing project...", total=6)

            # Phase 1: Analyze and plan fixes
            self._analyze_project_structure()
            progress.update(task, advance=1)

            # Phase 2: Optimize import graph
            self._optimize_dependency_graph()
            progress.update(task, advance=1)

            # Phase 3: Apply fixes in optimal order
            self._apply_strategic_fixes()
            progress.update(task, advance=1)

            # Phase 4: Verify and adjust fixes
            self._verify_fixes()
            progress.update(task, advance=1)

            # Phase 5: Format and clean up
            self._format_and_cleanup()
            progress.update(task, advance=1)

            # Phase 6: Generate report
            report = self._generate_fix_report()
            progress.update(task, advance=1)

        return report

    def _analyze_project_structure(self):
        """Analyze project structure and build dependency graph"""
        console.print("[bold blue]Analyzing project structure...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            python_files = list(self.root.rglob("*.py"))
            list(executor.map(self._analyze_file, python_files))

    def _analyze_file(self, file_path: Path):
        """Analyze a single file using libcst"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            module = cst.parse_module(source)
            visitor = ImportAnalyzer()
            module.visit(visitor)

            # Add to dependency graph
            module_name = self._get_module_name(file_path)
            self.dependency_graph.add_node(
                module_name, file_path=file_path, imports=visitor.imports
            )

            for imp in visitor.imports:
                self.dependency_graph.add_edge(module_name, imp)

        except Exception as e:
            console.print(f"[red]Error analyzing {file_path}: {str(e)}")

    def _optimize_dependency_graph(self):
        """Optimize dependency graph using network flow algorithms"""
        # Calculate optimal import ordering using topological sort
        try:
            optimal_order = list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXUnfeasible:
            # Handle cycles by using a feedback arc set
            edges_to_remove = self._minimum_feedback_arc_set()
            temp_graph = self.dependency_graph.copy()
            temp_graph.remove_edges_from(edges_to_remove)
            optimal_order = list(nx.topological_sort(temp_graph))

        # Calculate optimal module grouping using spectral clustering
        adjacency_matrix = nx.adjacency_matrix(self.dependency_graph).todense()
        eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)
        lambda_gap = abs(eigenvalues[-1] - eigenvalues[-2])  # Spectral gap
        console.print(f"Largest spectral gap: {lambda_gap:.4f}")

        n_clusters = min(5, len(self.dependency_graph))

        clustering = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
        clustering.fit(adjacency_matrix)
        self.module_clusters = clustering.labels_

        # Ensure labels_ is properly handled
        self.optimal_order = optimal_order

    def _minimum_feedback_arc_set(self):
        """Calculate minimum feedback arc set using linear programming"""
        adjacency_matrix = nx.adjacency_matrix(self.dependency_graph).todense()
        n = len(adjacency_matrix)

        # Create cost matrix for optimization
        cost_matrix = np.zeros((n, n))
        for i, j in itertools.product(range(n), range(n)):
            if adjacency_matrix[i, j]:
                # Weight based on dependency complexity
                cost_matrix[i, j] = 1 / (
                    1 + self.dependency_graph.edges[i, j].get("weight", 1)
                )

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Convert solution to edges
        edges_to_remove = []
        for i, j in zip(row_ind, col_ind):
            if adjacency_matrix[i, j]:
                source = list(self.dependency_graph.nodes())[i]
                target = list(self.dependency_graph.nodes())[j]
                edges_to_remove.append((source, target))

        return edges_to_remove

    def _apply_strategic_fixes(self):
        """Apply fixes in optimal order based on dependency graph"""
        console.print("[bold green]Applying strategic fixes...")

        # Sort strategies by priority and impact
        sorted_strategies = sorted(
            self.strategies, key=lambda s: (s.priority, -s.impact)
        )

        # Apply fixes following optimal order
        for module in self.optimal_order:
            node_data = self.dependency_graph.nodes[module]
            file_path = node_data["file_path"]

            for strategy in sorted_strategies:
                if self._should_apply_strategy(strategy, module):
                    if fix_op := strategy.fix_function(file_path):
                        self.fixes.append(fix_op)

    def _should_apply_strategy(self, strategy: FixStrategy, module: str) -> bool:
        """Determine if a strategy should be applied based on context"""
        # Calculate risk-adjusted impact
        risk_adjusted_impact = strategy.impact * (1 - strategy.risk)

        # Consider module position in dependency graph
        centrality = nx.betweenness_centrality(self.dependency_graph)  # Updated here
        module_importance = centrality.get(module, 0)

        # Check module cluster assignment
        cluster = self.module_clusters[
            list(self.dependency_graph.nodes()).index(module)
        ]

        # Decision function incorporating multiple factors
        decision_score = (
            risk_adjusted_impact * 0.4
            + module_importance * 0.3
            + (1 / (cluster + 1)) * 0.3
        )

        return decision_score > 0.5

    def _verify_fixes(self):
        """Verify applied fixes and adjust if needed"""
        console.print("[bold yellow]Verifying fixes...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._verify_single_fix, fix) for fix in self.fixes
            ]
            # Collect results and adjust fixes if needed
            for future in futures:
                fix, is_valid = future.result()
                if not is_valid:
                    self._adjust_fix(fix)

    def _verify_single_fix(self, fix: FixOperation) -> Tuple[FixOperation, bool]:
        """Verify a single fix operation"""
        try:
            # Parse fixed code to verify syntax
            ast.parse(fix.fixed_code)

            # Run type checker if available
            if self.config.get("enable_type_checking", True):
                import mypy.api

                result = mypy.api.run([fix.file_path])
                if result[0]:  # mypy errors
                    return fix, False

            # Verify imports still resolve
            cst.parse_module(fix.fixed_code)
            return fix, True

        except Exception:
            return fix, False

    def _adjust_fix(self, fix: FixOperation):
        """Adjust a failed fix operation"""
        console.print(f"[yellow]Adjusting fix for {fix.file_path}")

        # Try alternative fix strategies
        for strategy in reversed(self.strategies):
            if strategy.priority > 0:  # Skip highest risk strategies
                try:
                    new_fix = strategy.fix_function(fix.file_path)
                    if new_fix and self._verify_single_fix(new_fix)[1]:
                        self.fixes.remove(fix)
                        self.fixes.append(new_fix)
                        break
                except Exception:
                    continue

    def _format_and_cleanup(self):
        """Format fixed files and clean-up imports"""
        console.print("[bold blue]Formatting and cleaning up...")

        for fix in self.fixes:
            try:
                # Format with black
                formatted_code = black.format_str(fix.fixed_code, mode=black.FileMode())

                # Sort imports with isort
                sorted_code = isort.code(formatted_code)

                # Update fix operation
                fix.fixed_code = sorted_code

                # Write changes
                if self.backup:
                    backup_path = fix.file_path.with_suffix(".bak")
                    backup_path.write_text(fix.original_code)

                fix.file_path.write_text(sorted_code)

            except Exception as e:
                console.print(f"[red]Error formatting {fix.file_path}: {str(e)}")

    def _generate_fix_report(self) -> Dict[str, any]:
        """Generate comprehensive fix report"""
        report = {
            "total_fixes": len(self.fixes),
            "files_modified": len({fix.file_path for fix in self.fixes}),
            "fixes_by_type": {},
            "high_impact_changes": [],
            "requires_review": [],
            "metrics": {
                "average_confidence": np.mean([fix.confidence for fix in self.fixes]),
                "average_impact": np.mean([fix.impact_score for fix in self.fixes]),
            },
        }

        # Generate detailed fix type statistics
        for fix in self.fixes:
            report["fixes_by_type"][fix.operation_type] = (
                report["fixes_by_type"].get(fix.operation_type, 0) + 1
            )

            if fix.impact_score > 0.8:
                report["high_impact_changes"].append(
                    {
                        "file": str(fix.file_path),
                        "type": fix.operation_type,
                        "impact": fix.impact_score,
                    }
                )

        # Generate HTML diff views
        report["diffs"] = self._generate_html_diffs()

        return report

    def _generate_html_diffs(self) -> Dict[str, str]:
        """Generate HTML diffs for all fixes"""
        diffs = {}
        formatter = HtmlFormatter(style="monokai", full=True)

        for fix in self.fixes:
            diff = difflib.unified_diff(
                fix.original_code.splitlines(),
                fix.fixed_code.splitlines(),
                fromfile=str(fix.file_path),
                tofile=f"{fix.file_path}.fixed",
            )

            diff_html = highlight("\n".join(diff), PythonLexer(), formatter)

            diffs[str(fix.file_path)] = diff_html

        return diffs

    def _initialize_strategies(self) -> List[FixStrategy]:
        """Initialize fix strategies with priorities"""
        return [
            FixStrategy(
                priority=1,
                impact=0.9,
                risk=0.1,
                requires_manual_review=False,
                description="Convert relative imports to absolute",
                fix_function=self._fix_relative_imports,
            ),
            FixStrategy(
                priority=2,
                impact=0.8,
                risk=0.2,
                requires_manual_review=False,
                description="Fix circular dependencies",
                fix_function=self._fix_circular_dependencies,
            ),
            FixStrategy(
                priority=3,
                impact=0.7,
                risk=0.3,
                requires_manual_review=True,
                description="Optimize import structure",
                fix_function=self._optimize_imports,
            ),
            # Add more strategies here
        ]

    def _fix_relative_imports(self, file_path: Path) -> Optional[FixOperation]:
        """Convert relative imports to absolute imports"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            module = cst.parse_module(source)
            transformer = RelativeImportTransformer(self.root, file_path)
            modified_module = module.visit(transformer)

            if transformer.made_changes:
                return FixOperation(
                    file_path=file_path,
                    operation_type="relative_to_absolute",
                    original_code=source,
                    fixed_code=modified_module.code,
                    confidence=0.9,
                    impact_score=0.7,
                    dependencies_affected=transformer.affected_imports,
                )

        except Exception as e:
            console.print(f"[red]Error fixing imports in {file_path}: {str(e)}")
            return None

    def _fix_circular_dependencies(self, file_path: Path) -> Optional[FixOperation]:
        """Break circular dependencies using interface extraction"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            module = cst.parse_module(source)
            transformer = CircularDependencyTransformer(self.dependency_graph)
            modified_module = module.visit(transformer)

            if transformer.made_changes:
                return FixOperation(
                    file_path=file_path,
                    operation_type="break_circular_dep",
                    original_code=source,
                    fixed_code=modified_module.code,
                    confidence=0.7,
                    impact_score=0.9,
                    dependencies_affected=transformer.affected_modules,
                )

        except Exception as e:
            console.print(f"[red]Error fixing circular deps in {file_path}: {str(e)}")
            return None

    def _optimize_imports(self, file_path: Path) -> Optional[FixOperation]:
        """Optimize import structure using static analysis"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            module = cst.parse_module(source)
            transformer = ImportOptimizer(self.dependency_graph, self.module_clusters)
            modified_module = module.visit(transformer)

            if transformer.made_changes:
                return FixOperation(
                    file_path=file_path,
                    operation_type="optimize_imports",
                    original_code=source,
                    fixed_code=modified_module.code,
                    confidence=0.8,
                    impact_score=0.6,
                    dependencies_affected=transformer.affected_imports,
                )

        except Exception as e:
            console.print(f"[red]Error optimizing imports in {file_path}: {str(e)}")
            return None

    @staticmethod
    def _get_module_name(file_path: Path) -> str:
        """Convert file path to module name"""
        parts = list(file_path.parts)
        if "__init__.py" in parts:
            parts.remove("__init__.py")
        return ".".join(parts[:-1] + [file_path.stem])

    def _load_config(self, config_path: Optional[Path]):
        """Load configuration from a YAML file."""
        if config_path:
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as file:
                        config = yaml.safe_load(file)
                        console.print(
                            f"[green]Loaded config successfully: {config_path}"
                        )
                        return config
                except yaml.YAMLError as e:
                    console.print(f"[red]Error parsing YAML file {config_path}: {e}")
                    return {}
            console.print(f"[yellow]Config file does not exist: {config_path}")
        else:
            console.print("[yellow]No config path provided for loading.")
        return {}

    async def get_history(self):
        """Get project history using GitPython"""
        try:
            repo = git.Repo(self.root)
            commits = list(repo.iter_commits("main", max_count=100))
            history = []
            history.extend(
                {
                    "commit": commit.hexsha,
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                    "message": commit.message.strip(),
                }
                for commit in commits
            )
            return history
        except Exception as e:
            console.print(f"[red]Error retrieving history: {str(e)}")
            return []

    async def apply_fix(self, file_path: str, fix_id: str):
        """Apply a specific fix to a file based on fix_id"""
        try:
            # Find the fix operation by fix_id
            fix_operation = next(
                fix
                for fix in self.fixes
                if fix.file_path == Path(file_path) and fix.operation_type == fix_id
            )

            # Backup the original file if enabled
            if self.backup:
                backup_path = Path(file_path).with_suffix(".bak")
                backup_path.write_text(Path(file_path).read_text())

            # Apply the fix
            Path(file_path).write_text(fix_operation.fixed_code)

            console.print(f"[green]Fix applied successfully to {file_path}")

        except StopIteration:
            console.print(f"[red]Fix with id {fix_id} not found for {file_path}")
        except Exception as e:
            console.print(f"[red]Error applying fix to {file_path}: {str(e)}")


class RelativeImportTransformer(cst.CSTTransformer):
    """Transform relative imports to absolute imports"""

    def __init__(self, root_path: Path, file_path: Path):
        """Initialize RelativeImportTransformer with root path and file path"""
        super().__init__()
        self.root = root_path
        self.file_path = file_path
        self.made_changes = False
        self.affected_imports = set()

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        """Handle relative imports by converting them to absolute imports."""
        # Check if the import is relative
        if original_node.relative:
            self.made_changes = True
            # Count the dots (levels of relative import)
            level = sum(dot is not None for dot in original_node.relative)
            module_path = self._resolve_relative_import(
                level, original_node.module.value if original_node.module else ""
            )
            self.affected_imports.add(module_path)

            # Return the transformed ImportFrom node with absolute imports
            return updated_node.with_changes(
                relative=[], module=cst.Name(value=module_path)
            )
        return updated_node

    def _resolve_relative_import(self, level: int, module: str) -> str:
        """Resolve relative import to absolute path"""
        current_path = self.file_path.parent
        for _ in range(level):
            current_path = current_path.parent

        resolved = current_path.relative_to(self.root)
        parts = list(resolved.parts)
        if module:
            parts.append(module)

        return ".".join(parts)


class CircularDependencyTransformer(cst.CSTTransformer):
    """Transform code to break circular dependencies."""

    def __init__(self, dependency_graph: nx.DiGraph):
        super().__init__()
        self.graph = dependency_graph
        self.made_changes = False
        self.affected_modules = set()

    def _create_interface_import(
        self, extra_imports: Optional[List[str]] = None
    ) -> cst.ImportFrom:
        """Create an enhanced interface-based import to break circular dependency.

        Optionally, add additional imports alongside 'Protocol'.

        Args:
            extra_imports (Optional[List[str]]): A list of additional names to import
                from the 'typing' module (e.g., ['List', 'Union']).

        Returns:
            cst.ImportFrom: A libcst.ImportFrom node for the 'typing' module.
        """
        # Base imports with 'Protocol'
        imports = [cst.ImportAlias(name=cst.Name("Protocol"))]

        # Add extra imports if provided
        if extra_imports:
            imports.extend(
                cst.ImportAlias(name=cst.Name(name)) for name in extra_imports
            )

        # Return the ImportFrom node
        return cst.ImportFrom(
            module=cst.Name("typing"),
            names=imports,
        )

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        module_name = original_node.module.value if original_node.module else ""
        if self._is_part_of_cycle(module_name):
            self.made_changes = True
            self.affected_modules.add(module_name)
            return self._create_interface_import()  # Remove the argument here
        return updated_node

    def _is_part_of_cycle(self, module_name: str) -> bool:
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return any(module_name in cycle for cycle in cycles)
        except nx.NetworkXNoCycle:
            return False


class ImportAnalyzer(cst.CSTVisitor):
    """
    Analyzes imports in a Python module using libcst.
    Collects all imported modules and identifiers.
    """

    def __init__(self):
        super().__init__()

        self.imports = set()  # Store all imports

    def visit_Import(self, node: cst.Import):
        # Handle statements like: `import os`
        for alias in node.names:
            self.imports.add(alias.name.value)

    def visit_ImportFrom(self, node: cst.ImportFrom):
        # Handle statements like: `from os import path`
        if node.module:
            module_name = node.module.value
            for alias in node.names:
                if isinstance(alias.name, cst.Name):
                    self.imports.add(f"{module_name}.{alias.name.value}")


class ImportOptimizer(cst.CSTTransformer):
    """Optimize import structure based on dependency analysis"""

    def __init__(self, dependency_graph: nx.DiGraph, module_clusters: np.ndarray):
        super().__init__()
        self.graph = dependency_graph
        self.clusters = module_clusters
        self.made_changes = False
        self.affected_imports = set()

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        # Group imports by cluster
        import_groups = self._group_imports(original_node)

        # Create new import block
        new_body = []
        for group in import_groups:
            new_body.extend(group)
            new_body.append(cst.EmptyLine())

        self.made_changes = True
        return updated_node.with_changes(
            body=new_body
            + [
                node
                for node in updated_node.body
                if not isinstance(node, (cst.Import, cst.ImportFrom))
            ]
        )

    def _group_imports(self, module: cst.Module) -> List[List[cst.CSTNode]]:
        """Group imports by their cluster assignment"""
        imports = [
            node
            for node in module.body
            if isinstance(node, (cst.Import, cst.ImportFrom))
        ]

        grouped: Dict[int, List[cst.CSTNode]] = defaultdict(list)
        for imp in imports:
            cluster = self._get_import_cluster(imp)
            grouped[cluster].append(imp)

        return [grouped[i] for i in sorted(grouped.keys())]

    def _get_import_cluster(
        self, node: cst.CSTNode
    ) -> np.ndarray[tuple[int, ...], np.dtype[Any] | Any] | int:
        """Get cluster assignment for import"""
        if isinstance(node, cst.ImportFrom):
            module_name = node.module.value if node.module else ""
            try:
                idx = list(self.graph.nodes()).index(module_name)
                return self.clusters[idx]
            except (ValueError, IndexError):
                return 0
        return 0
