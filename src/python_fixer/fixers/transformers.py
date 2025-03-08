import ast
from pathlib import Path
from typing import Dict, List, Optional

import astor
import variant_loggers
from fixers.fix_enhanced_analyzer import EnhancedAnalyzer


# Base Transformer Class
class BaseTransformer:
    """
    Abstract base class for all transformers.
    """

    def apply(self, analyzer: EnhancedAnalyzer):
        """
        Apply the transformation using the analyzer's data.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the apply method.")


# RelativeImportTransformer Class
class RelativeImportTransformer(BaseTransformer):
    """
    Transforms relative imports to absolute imports based on the project structure.
    """

    def apply(self, analyzer: EnhancedAnalyzer):
        variant_loggers.info("Applying RelativeImportTransformer...")
        for module_path in analyzer.modules:
            with open(module_path, "r", encoding="utf-8") as file:
                source = file.read()
            tree = ast.parse(source, filename=str(module_path))
            modified = False

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    if absolute_module := self._resolve_absolute_import(
                        analyzer, module_path, node
                    ):
                        node.module = absolute_module
                        node.level = 0
                        modified = True
                        variant_loggers.debug(
                            f"Converted relative import in {module_path}: "
                            f"{astor.to_source(node).strip()}"
                        )

            if modified:
                new_source = astor.to_source(tree)
                with open(module_path, "w", encoding="utf-8") as file:
                    file.write(new_source)
                variant_loggers.info(f"Updated imports in {module_path}.")

    def _resolve_absolute_import(
        self, analyzer: EnhancedAnalyzer, module_path: Path, node: ast.ImportFrom
    ) -> Optional[str]:
        """
        Resolve the absolute module name from a relative import.
        """
        current_module = analyzer._module_name(module_path)
        parts = current_module.split(".")
        target = node.module.split(".") if node.module else []
        # Calculate the base for the absolute import
        base = parts[: -node.level]
        absolute_module = ".".join(base + target)
        return absolute_module or None


# CircularDependencyTransformer Class
class CircularDependencyTransformer(BaseTransformer):
    """
    Detects and attempts to resolve circular dependencies within the project.
    """

    def apply(self, analyzer: EnhancedAnalyzer):
        variant_loggers.info("Applying CircularDependencyTransformer...")
        import_graph = analyzer.import_graph
        cycles = self._find_cycles(import_graph)
        variant_loggers.debug(f"Found cycles: {cycles}")

        for cycle in cycles:
            if success := self._resolve_cycle(analyzer, cycle):
                variant_loggers.info(f"Resolved cycle: {' -> '.join(cycle)}")
            else:
                variant_loggers.warning(f"Could not resolve cycle: {' -> '.join(cycle)}")

    def _find_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """
        Detect cycles in the import graph using Depth-First Search (DFS).
        """
        visited = set()
        stack = []
        cycles = []

        def dfs(node, path):
            visited.add(node)
            path.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in path:
                    cycle_start_index = path.index(neighbor)
                    cycle = path[cycle_start_index:]
                    if cycle not in cycles:
                        cycles.append(cycle)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _resolve_cycle(self, analyzer, cycle):
        pass


def _resolve_cycle(self, analyzer: EnhancedAnalyzer, cycle: List[str]) -> bool:
    """
    Attempt to resolve a circular dependency cycle.
    Strategy:
        - Refactor imports to use dependency injection or local imports.
        - This is a complex task and may require manual intervention.
        - Here, we'll demonstrate by converting top-level imports to local imports.
    """
    # For demonstration, attempt to convert the first import in the cycle to a local import
    # WARNING: This is a naive implementation and may not work for all cases
    if not cycle:
        return False

    # Select the module to modify (e.g., the first module in the cycle)
    module_to_modify = cycle[0]
    module_path = self._get_module_path(analyzer, module_to_modify)
    if not module_path:
        variant_loggers.error(f"Module path not found for {module_to_modify}.")
        return False

    with open(module_path, "r", encoding="utf-8") as file:
        source = file.read()
    tree = ast.parse(source, filename=str(module_path))
    modified = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_module = node.module
            if imported_module in cycle:
                if enclosing_function := self._get_enclosing_function(node, tree):
                    # Create a new ImportFrom node with absolute import
                    new_import = ast.ImportFrom(
                        module=imported_module, names=node.names, level=0
                    )
                    # Insert the new import at the beginning of the function body
                    enclosing_function.body.insert(0, new_import)
                    # Remove the original import from the function body
                    enclosing_function.body.remove(node)
                    modified = True
                    variant_loggers.debug(
                        f"Converted top-level import in {module_path}: {astor.to_source(node).strip()} "
                        f"to local import inside function '{enclosing_function.name}'."
                    )
                else:
                    # If not inside a function, cannot convert
                    variant_loggers.warning(
                        f"Cannot convert import in {module_path} as it's not inside a function."
                    )

    if modified:
        new_source = astor.to_source(tree)
        with open(module_path, "w", encoding="utf-8") as file:
            file.write(new_source)
        return True
    return False
