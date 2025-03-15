# src/analysis/core/project_analysis.py

# -----------------------------
# PROJECT ANALYSIS MODULE
# -----------------------------

import ast

# Standard library imports
import logging

# Local application imports
from pathlib import Path
from typing import Dict, List

import networkx as nx
import toml

# Third-party library imports


class ProjectAnalyzer:
    """Analyzes Python project structure and dependencies.

    MAP: /project_root/analysis/core
    EFFECT: Creates comprehensive project analysis including dependencies and enhancement targets
    NAMING: ProjectAnalyzer

    Attributes:
        project_path: Root path of the project to analyze
        dependency_graph: NetworkX graph of module dependencies
        modules: Dictionary of analyzed module information
    """

    def __init__(
        self, project_path: str, config_path: Path = None, backup: bool = True
    ):
        self.project_path = Path(project_path)
        self.config_path = config_path
        self.backup = backup
        self.dependency_graph = nx.DiGraph()
        self.modules: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_config() if config_path else self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration settings."""
        return {
            "max_workers": 4,
            "backup": True,
            "fix_mode": "interactive",
            "git_integration": True,
            "exclude_patterns": ["venv", "*.pyc", "__pycache__"],
        }

    def _load_config(self) -> Dict:
        """Load configuration from file."""

        try:
            return toml.load(self.config_path)
        except Exception as e:
            self.logger.warning(f"Error loading config: {e}. Using defaults.")
            return self._default_config()

    def initialize_project(self) -> None:
        """Initialize project with default configuration."""
        if not self.project_path.exists():
            self.project_path.mkdir(parents=True)

        config_path = self.project_path / "python_fixer.toml"
        if not config_path.exists():
            config_path.write_text(toml.dumps(self._default_config()))
            self.logger.info(f"Created default config at {config_path}")

    def fix_project(self, mode: str = "interactive") -> Dict:
        """Fix import issues in the project.

        Args:
            mode: Fix mode ('interactive' or 'automatic')

        Returns:
            Dict containing fix metrics
        """
        analysis = self.analyze_project()
        fixes = {"imports_fixed": 0, "files_modified": set(), "backup_created": False}

        # Create backups if needed
        if self.backup:
            fixes["backup_created"] = self._create_backups(analysis)

        # Apply fixes to modules with issues
        self._apply_fixes_to_modules(analysis, fixes)

        return fixes

    def _create_backups(self, analysis: Dict) -> bool:
        """Create backups of files that will be modified.

        Args:
            analysis: Project analysis dictionary

        Returns:
            True if backups were created, False otherwise
        """
        # Create backup directory
        backup_dir = self.project_path / "python_fixer_backup"
        if not backup_dir.exists():
            backup_dir.mkdir(parents=True)
            self.logger.info(f"Created backup directory at {backup_dir}")

        # Get files that need backup
        files_to_backup = [
            Path(module_path)
            for module_path, info in analysis["dependencies"].items()
            if info.get("cycles") or info.get("unused_imports")
        ]

        # Create the actual backups
        for file_path in files_to_backup:
            if file_path.exists():
                backup_path = backup_dir / file_path.name
                backup_path.write_text(file_path.read_text())
                self.logger.info(f"Created backup of {file_path} at {backup_path}")

        return True

    def _apply_fixes_to_modules(self, analysis: Dict, fixes: Dict) -> None:
        """Apply fixes to modules with issues.

        Args:
            analysis: Project analysis dictionary
            fixes: Dictionary to track fix metrics
        """
        for module_path, info in analysis["dependencies"].items():
            if info.get("cycles") or info.get("unused_imports"):
                self.logger.info(f"Found issues in {module_path}")
                self._fix_module(module_path, info, fixes)

    def _fix_module(self, module_path: str, info: Dict, fixes: Dict) -> None:
        """Apply fixes to a single module.

        Args:
            module_path: Path to the module
            info: Module analysis information
            fixes: Dictionary to track fix metrics
        """
        file_path = Path(module_path)
        if not file_path.exists():
            self.logger.warning(f"File {module_path} not found, skipping fixes")
            return

        try:
            file_content = file_path.read_text()
            tree = ast.parse(file_content)
            if self._apply_module_fixes(tree, file_content, info, file_path):
                fixes["imports_fixed"] += 1
                fixes["files_modified"].add(module_path)
        except Exception as e:
            self.logger.error(f"Error fixing {module_path}: {e}")

    def _apply_module_fixes(
        self, tree: ast.AST, file_content: str, info: Dict, file_path: Path
    ) -> bool:
        """Apply specific fixes to a module.

        Args:
            tree: AST of the module
            file_content: Content of the file
            info: Module analysis information
            file_path: Path to the module file

        Returns:
            True if the module was modified, False otherwise
        """
        modified = False

        # Fix unused imports
        if info.get("unused_imports"):
            modified = self._remove_unused_imports(
                tree, file_content, info["unused_imports"], file_path
            )

        # Fix import cycles
        if info.get("cycles"):
            cycle_fix = self._fix_import_cycles(
                tree, file_content, info["cycles"], file_path
            )
            modified = modified or cycle_fix

        return modified

    def analyze_project(self) -> Dict:
        """Performs complete project analysis.

        Returns:
            Dict containing:
                - structure: Project file structure
                - dependencies: Module dependency information
                - enhancements: Potential enhancement targets
        """
        try:
            structure = self._scan_structure()
            dependencies = self._analyze_dependencies()
            enhancements = self._identify_enhancements()

            return {
                "structure": structure,
                "dependencies": dependencies,
                "enhancements": enhancements,
            }
        except Exception as e:
            self.logger.error(f"Error during project analysis: {str(e)}")
            raise

    def _scan_structure(self) -> Dict:
        """Scans project structure and builds module map.

        Returns:
            Dict containing module paths and their contents
        """
        structure = {"modules": [], "packages": []}
        excluded_dirs = {".venv", ".git", "__pycache__", "tests", "docs"}

        for path in self.project_path.rglob("*.py"):
            # Skip excluded directories
            if any(part in excluded_dirs for part in path.parts):
                continue

            if path.is_file():
                try:
                    # Skip files larger than 1MB to avoid memory issues
                    if path.stat().st_size > 1_000_000:
                        self.logger.warning(f"Skipping large file {path}")
                        continue

                    module_info = self._analyze_module(path)
                    rel_path = path.relative_to(self.project_path)
                    self.modules[str(rel_path)] = module_info
                    structure["modules"].append(module_info)
                except UnicodeDecodeError:
                    self.logger.warning(f"Skipping file with encoding issues: {path}")
                except Exception as e:
                    self.logger.warning(f"Error analyzing module {path}: {str(e)}")

        return structure

    def _analyze_module(self, path: Path) -> Dict:
        """Analyzes individual module contents.

        Args:
            path: Path to the Python module

        Returns:
            Dict containing module analysis information
        """
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        tree = ast.parse(content)
        analyzer = ModuleAnalyzer()
        analyzer.visit(tree)

        return {
            "path": str(path),
            "classes": analyzer.classes,
            "functions": analyzer.functions,
            "imports": analyzer.imports,
            "docstring": ast.get_docstring(tree),
        }

    def _analyze_dependencies(self) -> Dict:
        """Analyzes project dependencies and builds dependency graph.

        Returns:
            Dict containing dependency analysis results
        """
        try:
            # Build dependency graph
            for module_path, info in self.modules.items():
                self.dependency_graph.add_node(module_path)
                for imp in info["imports"]:
                    # Only add edges for internal project dependencies
                    if any(str(imp["module"]).startswith(mod) for mod in self.modules):
                        self.dependency_graph.add_edge(module_path, imp["module"])

            # Get cycles with a timeout to prevent hanging on complex graphs
            try:
                cycles = list(nx.simple_cycles(self.dependency_graph))
            except nx.NetworkXUnfeasible:
                cycles = []
                self.logger.warning("Could not compute dependency cycles")

            dependency_info = self._get_dependency_info()
            return {
                "primary": dependency_info["primary"],
                "secondary": dependency_info["secondary"],
                "cycles": cycles,
                "graph": self.dependency_graph,
            }
        except Exception as e:
            self.logger.error(f"Error in dependency analysis: {str(e)}")
            return {
                "primary": [],
                "secondary": [],
                "cycles": [],
                "graph": self.dependency_graph,
            }

    def _identify_enhancements(self) -> List[Dict]:
        """Identifies potential enhancement targets.

        Returns:
            List of dicts containing enhancement suggestions
        """
        enhancements = []
        for module_path, info in self.modules.items():
            if module_enhancements := self._analyze_enhancement_needs(info):
                enhancements.append(
                    {"module": module_path, "suggestions": module_enhancements}
                )
        return enhancements

    def _analyze_enhancement_needs(self, module_info: Dict) -> List[str]:
        """Analyzes module for potential enhancements.

        Args:
            module_info: Dictionary containing module analysis information

        Returns:
            List of suggested enhancements
        """
        needs = []

        # Check for missing docstrings
        if not module_info["docstring"]:
            needs.append("add_module_docstring")

        # Check classes and functions for documentation
        needs.extend(
            f"add_class_docstring:{cls['name']}"
            for cls in module_info["classes"]
            if not cls["docstring"]
        )
        needs.extend(
            f"add_function_docstring:{func['name']}"
            for func in module_info["functions"]
            if not func["docstring"]
        )

        # Check for high dependency count
        if len(module_info["imports"]) > 10:
            needs.append("reduce_dependencies")

        return needs

    def _get_dependency_info(self) -> Dict:
        """Extracts detailed dependency information.

        Returns:
            Dict containing primary and secondary dependencies
        """
        try:
            nodes = list(self.dependency_graph.nodes())
            return {
                "primary": [n for n in nodes if self.dependency_graph.in_degree(n) > 2],
                "secondary": [
                    n for n in nodes if self.dependency_graph.in_degree(n) <= 2
                ],
            }
        except Exception as e:
            self.logger.error(f"Error getting dependency info: {str(e)}")
            return {"primary": [], "secondary": []}


class ModuleAnalyzer(ast.NodeVisitor):
    """Analyzes Python module contents using AST.

    Attributes:
        classes: List of class definitions found
        functions: List of function definitions found
        imports: List of import statements found
    """

    def __init__(self):
        self.classes: List[Dict] = []
        self.functions: List[Dict] = []
        self.imports: List[Dict] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Analyzes class definitions."""
        self.classes.append(
            {
                "name": node.name,
                "methods": len(
                    [n for n in node.body if isinstance(n, ast.FunctionDef)]
                ),
                "docstring": ast.get_docstring(node),
                "decorators": [
                    d.id for d in node.decorator_list if isinstance(d, ast.Name)
                ],
            }
        )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyzes function definitions."""
        self.functions.append(
            {
                "name": node.name,
                "args": len(node.args.args),
                "docstring": ast.get_docstring(node),
                "decorators": [
                    d.id for d in node.decorator_list if isinstance(d, ast.Name)
                ],
            }
        )

    def visit_Import(self, node: ast.Import) -> None:
        """Analyzes import statements."""
        for name in node.names:
            self.imports.append({"module": name.name, "alias": name.asname})

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Analyzes from-import statements."""
        if node.module:
            for name in node.names:
                self.imports.append(
                    {"module": f"{node.module}.{name.name}", "alias": name.asname}
                )
