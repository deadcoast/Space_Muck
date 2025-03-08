# src/analysis/core/project_analysis.py

# -----------------------------
# PROJECT ANALYSIS MODULE
# -----------------------------

# Parent: None
# Dependencies: ast, networkx, pathlib, typing

import ast
import logging
from pathlib import Path
from typing import Dict, List
import networkx as nx

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

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.dependency_graph = nx.DiGraph()
        self.modules: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)

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
                'structure': structure,
                'dependencies': dependencies,
                'enhancements': enhancements
            }
        except Exception as e:
            self.logger.error(f"Error during project analysis: {str(e)}")
            raise

    def _scan_structure(self) -> Dict:
        """Scans project structure and builds module map.

        Returns:
            Dict containing module paths and their contents
        """
        structure = {'modules': [], 'packages': []}
        excluded_dirs = {'.venv', '.git', '__pycache__', 'tests', 'docs'}

        for path in self.project_path.rglob('*.py'):
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
                    structure['modules'].append(module_info)
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
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()

        tree = ast.parse(content)
        analyzer = ModuleAnalyzer()
        analyzer.visit(tree)

        return {
            'path': str(path),
            'classes': analyzer.classes,
            'functions': analyzer.functions,
            'imports': analyzer.imports,
            'docstring': ast.get_docstring(tree)
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
                for imp in info['imports']:
                    # Only add edges for internal project dependencies
                    if any(str(imp['module']).startswith(mod) for mod in self.modules):
                        self.dependency_graph.add_edge(module_path, imp['module'])

            # Get cycles with a timeout to prevent hanging on complex graphs
            try:
                cycles = list(nx.simple_cycles(self.dependency_graph))
            except nx.NetworkXUnfeasible:
                cycles = []
                self.logger.warning("Could not compute dependency cycles")

            dependency_info = self._get_dependency_info()
            return {
                'primary': dependency_info['primary'],
                'secondary': dependency_info['secondary'],
                'cycles': cycles,
                'graph': self.dependency_graph
            }
        except Exception as e:
            self.logger.error(f"Error in dependency analysis: {str(e)}")
            return {
                'primary': [],
                'secondary': [],
                'cycles': [],
                'graph': self.dependency_graph
            }

    def _identify_enhancements(self) -> List[Dict]:
        """Identifies potential enhancement targets.

        Returns:
            List of dicts containing enhancement suggestions
        """
        enhancements = []
        for module_path, info in self.modules.items():
            if module_enhancements := self._analyze_enhancement_needs(info):
                enhancements.append({
                    'module': module_path,
                    'suggestions': module_enhancements
                })
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
        if not module_info['docstring']:
            needs.append('add_module_docstring')

        # Check classes and functions for documentation
        needs.extend(f'add_class_docstring:{cls["name"]}' 
                    for cls in module_info['classes'] if not cls['docstring'])
        needs.extend(f'add_function_docstring:{func["name"]}' 
                    for func in module_info['functions'] if not func['docstring'])

        # Check for high dependency count
        if len(module_info['imports']) > 10:
            needs.append('reduce_dependencies')

        return needs

    def _get_dependency_info(self) -> Dict:
        """Extracts detailed dependency information.

        Returns:
            Dict containing primary and secondary dependencies
        """
        try:
            nodes = list(self.dependency_graph.nodes())
            return {
                'primary': [n for n in nodes if self.dependency_graph.in_degree(n) > 2],
                'secondary': [n for n in nodes if self.dependency_graph.in_degree(n) <= 2]
            }
        except Exception as e:
            self.logger.error(f"Error getting dependency info: {str(e)}")
            return {'primary': [], 'secondary': []}

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
        self.classes.append({
            'name': node.name,
            'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
            'docstring': ast.get_docstring(node),
            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        })
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyzes function definitions."""
        self.functions.append({
            'name': node.name,
            'args': len(node.args.args),
            'docstring': ast.get_docstring(node),
            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        })

    def visit_Import(self, node: ast.Import) -> None:
        """Analyzes import statements."""
        for name in node.names:
            self.imports.append({
                'module': name.name,
                'alias': name.asname
            })

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Analyzes from-import statements."""
        if node.module:
            for name in node.names:
                self.imports.append({
                    'module': f"{node.module}.{name.name}",
                    'alias': name.asname
                })