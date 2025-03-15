#!/usr/bin/env python3

# -----------------------------
# PROJECT MAP GENERATOR
# -----------------------------
#
# Parent: analysis.docs
# Dependencies: sphinx, networkx, graphviz, typing, logging
#
# MAP: /project_root/analysis/docs
# EFFECT: Generates project structure maps and analysis
# NAMING: ProjectMap[Type]Generator

# Standard library imports
import logging

# Local application imports
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx
from docutils import nodes
from graphviz import Digraph
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

# Third-party library imports


@dataclass
class ProjectMap:
    """Container for project map data."""

    name: str
    description: str
    components: Set[str]
    dependencies: Dict[str, List[str]]
    enhancement_targets: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class ProjectMapGenerator:
    """Generates project structure maps and analysis.

    This system provides:
    1. Project structure documentation
    2. Enhancement target identification
    3. Dependency mapping
    4. Component relationship analysis
    """

    def __init__(self, output_dir: Path):
        """Initialize the project map generator.

        Args:
            output_dir: Directory to write maps to
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.project_maps = {}  # type: Dict[str, ProjectMap]
        self.dependency_graph = nx.DiGraph()

    def document_structure(
        self, root_dir: Path, output_file: str = "structure.md"
    ) -> None:
        """Document project directory structure.

        Args:
            root_dir: Root directory of the project
            output_file: Name of the output file
        """
        try:
            content = ["# Project Structure", "", "## Directory Tree", ""]

            def format_tree(path: Path, prefix: str = "") -> List[str]:
                """Format directory tree for documentation."""
                lines = []
                for item in sorted(path.iterdir()):
                    if item.name.startswith("."):
                        continue

                    lines.append(f"{prefix}* {item.name}")
                    if item.is_dir():
                        lines.extend(format_tree(item, f"{prefix}  "))
                return lines

            content.extend(format_tree(root_dir))

            # Add component relationships
            content.extend(
                ["", "## Component Relationships", "", "```mermaid", "graph LR"]
            )

            # Add nodes and edges to mermaid diagram
            for source, targets in self.dependency_graph.edges():
                for target in targets:
                    content.append(f"    {source}-->{target}")

            content.extend(["```", "", "## Enhancement Targets", ""])

            # Add enhancement targets
            for map_name, project_map in self.project_maps.items():
                if project_map.enhancement_targets:
                    content.extend(
                        [
                            f"### {map_name}",
                            "",
                            *[
                                f"* {target}"
                                for target in project_map.enhancement_targets
                            ],
                            "",
                        ]
                    )

            # Write documentation
            output_path = self.output_dir / output_file
            output_path.write_text("\n".join(content))

        except Exception as e:
            self.logger.error(f"Error documenting project structure: {str(e)}")

    def identify_enhancement_targets(
        self, source_files: Dict[str, str], complexity_threshold: int = 50
    ) -> List[str]:
        """Identify potential enhancement targets based on code analysis.

        Args:
            source_files: Dictionary mapping file paths to their source code
            complexity_threshold: Complexity threshold for identifying targets

        Returns:
            List of identified enhancement targets
        """
        try:
            from radon.complexity import cc_visit

            targets = []
            for path, code in source_files.items():
                try:
                    # Calculate cyclomatic complexity
                    complexity_results = cc_visit(code)
                    total_complexity = sum(
                        result.complexity for result in complexity_results
                    )

                    # Check complexity threshold
                    if total_complexity > complexity_threshold:
                        targets.append(path)
                        self.logger.info(
                            f"Enhancement target identified: {path} "
                            f"(complexity: {total_complexity})"
                        )
                except Exception as e:
                    self.logger.warning(f"Error analyzing {path}: {str(e)}")

            return targets

        except Exception as e:
            self.logger.error(f"Error identifying enhancement targets: {str(e)}")
            return []

    def map_dependencies(
        self,
        module_dependencies: Dict[str, List[str]],
        output_file: str = "dependencies.pdf",
    ) -> None:
        """Generate dependency map using Graphviz.

        Args:
            module_dependencies: Dictionary mapping modules to their dependencies
            output_file: Name of the output file
        """
        try:
            # Create graph
            dot = Digraph(comment="Module Dependencies")
            dot.attr(rankdir="LR")

            # Add nodes and edges
            for module, deps in module_dependencies.items():
                dot.node(module, module)
                for dep in deps:
                    dot.edge(module, dep)

            # Save graph
            output_path = self.output_dir / output_file
            dot.render(str(output_path), cleanup=True)

            # Update dependency graph
            self.dependency_graph.clear()
            for module, deps in module_dependencies.items():
                self.dependency_graph.add_node(module)
                for dep in deps:
                    self.dependency_graph.add_edge(module, dep)

        except Exception as e:
            self.logger.error(f"Error mapping dependencies: {str(e)}")

    def create_project_map(
        self,
        name: str,
        description: str,
        components: Set[str],
        dependencies: Dict[str, List[str]],
        enhancement_targets: Optional[List[str]] = None,
        notes: Optional[List[str]] = None,
    ) -> Optional[ProjectMap]:
        """Create a project map.

        Args:
            name: Name of the map
            description: Description of the map
            components: Set of component names
            dependencies: Dictionary mapping components to their dependencies
            enhancement_targets: Optional list of enhancement targets
            notes: Optional list of additional notes

        Returns:
            Created project map
        """
        try:
            # Create map
            project_map = ProjectMap(
                name=name,
                description=description,
                components=components,
                dependencies=dependencies,
                enhancement_targets=enhancement_targets or [],
                notes=notes or [],
            )

            # Store map
            self.project_maps[name] = project_map

            # Update dependency graph
            for source, targets in dependencies.items():
                self.dependency_graph.add_node(source)
                for target in targets:
                    self.dependency_graph.add_edge(source, target)

            return project_map

        except Exception as e:
            self.logger.error(f"Error creating project map {name}: {str(e)}")
            return None

    def clear_maps(self) -> None:
        """Clear all project maps."""
        self.project_maps.clear()
        self.dependency_graph.clear()

    def generate_sphinx_docs(self, output_file: str = "project_maps.rst") -> None:
        """Generate Sphinx documentation for project maps.

        Args:
            output_file: Name of the output file
        """
        try:
            content = [
                "Project Maps",
                "============",
                "",
                ".. contents::",
                "   :local:",
                "",
            ]

            # Document each project map
            for map_name, project_map in self.project_maps.items():
                content.extend(
                    [
                        map_name,
                        "-" * len(map_name),
                        "",
                        project_map.description,
                        "",
                        "Components",
                        "~~~~~~~~~",
                        "",
                        ".. code-block:: text",
                        "",
                        *[
                            f"    * {component}"
                            for component in sorted(project_map.components)
                        ],
                        "",
                        "Dependencies",
                        "~~~~~~~~~~~",
                        "",
                        ".. mermaid::",
                        "",
                        "    graph LR",
                    ]
                )

                # Add dependencies to mermaid diagram
                content.extend(
                    f"        {source}-->{target}"
                    for source, targets in project_map.dependencies.items()
                    for target in targets
                )

                # Add enhancement targets if present
                if project_map.enhancement_targets:
                    content.extend(
                        [
                            "",
                            "Enhancement Targets",
                            "~~~~~~~~~~~~~~~~~~",
                            "",
                            *[
                                f"* {target}"
                                for target in project_map.enhancement_targets
                            ],
                            "",
                        ]
                    )

                # Add notes if present
                if project_map.notes:
                    content.extend(
                        [
                            "Notes",
                            "~~~~~",
                            "",
                            *[f"* {note}" for note in project_map.notes],
                            "",
                        ]
                    )

            # Write documentation
            output_path = self.output_dir / output_file
            output_path.write_text("\n".join(content))

        except Exception as e:
            self.logger.error(f"Error generating Sphinx documentation: {str(e)}")


class ProjectMapDirective(SphinxDirective):
    """Sphinx directive for including project maps in documentation."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = False

    def run(self) -> List[nodes.Node]:
        """Run the directive."""
        map_name = self.arguments[0]

        # Create a new section for the project map
        section = nodes.section()
        section += nodes.title(text=map_name)

        # Get the project map
        project_map = self.env.app.project_maps.get(map_name)
        if project_map is None:
            return [section]

        # Add description
        section += nodes.paragraph(text=project_map.description)

        # Add components
        comp_section = nodes.section()
        comp_section += nodes.title(text="Components")
        comp_list = nodes.bullet_list()
        for component in sorted(project_map.components):
            comp_list += nodes.list_item(text=component)
        comp_section += comp_list
        section += comp_section

        # Add dependencies
        dep_section = nodes.section()
        dep_section += nodes.title(text="Dependencies")
        dep_list = nodes.bullet_list()
        for source, targets in project_map.dependencies.items():
            for target in targets:
                dep_list += nodes.list_item(text=f"{source} → {target}")
        dep_section += dep_list
        section += dep_section

        return [section]


def setup(app: Sphinx) -> Dict[str, bool]:
    """Set up the Sphinx extension."""
    app.add_directive("project-map", ProjectMapDirective)
    app.add_node(nodes.section)
    app.add_node(nodes.title)
    app.add_node(nodes.paragraph)
    app.add_node(nodes.bullet_list)
    app.add_node(nodes.list_item)

    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
