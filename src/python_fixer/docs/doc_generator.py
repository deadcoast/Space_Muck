#!/usr/bin/env python3

# -----------------------------
# DOCUMENTATION GENERATOR
# -----------------------------
#
# Parent: analysis.docs
# Dependencies: sphinx, graphviz, typing, logging
#
# MAP: /project_root/analysis/docs
# EFFECT: Generates documentation for enhanced methods and project structure
# NAMING: Doc[Type]Generator

# Standard library imports
import inspect
import logging

# Third-party library imports

# Local application imports
from dataclasses import dataclass, field
from graphviz import Digraph
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MethodDoc:
    """Container for method documentation."""

    name: str
    description: str
    parameters: Dict[str, str]
    returns: str
    examples: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class DocGenerator:
    """Generates documentation for enhanced methods and project structure.

    This system provides:
    1. Automatic method documentation
    2. Dependency graph generation
    3. Configuration documentation
    4. Project structure documentation
    """

    def __init__(self, output_dir: Path):
        """Initialize the documentation generator.

        Args:
            output_dir: Directory to write documentation to
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.method_docs = {}  # type: Dict[str, MethodDoc]
        self.dependencies = {}  # type: Dict[str, List[str]]

    def document_method(
        self,
        method: Any,
        class_name: str,
        method_name: str,
        examples: Optional[List[str]] = None,
        notes: Optional[List[str]] = None,
    ) -> MethodDoc:
        """Generate documentation for a method.

        Args:
            method: Method to document
            class_name: Name of the class containing the method
            method_name: Name of the method
            examples: Optional list of usage examples
            notes: Optional list of additional notes

        Returns:
            Generated method documentation
        """
        try:
            # Get method signature and docstring
            sig = inspect.signature(method)
            doc = inspect.getdoc(method) or ""

            # Parse parameters
            params = {}
            for name, param in sig.parameters.items():
                if name != "self":
                    annotation = param.annotation
                    if annotation != inspect.Parameter.empty:
                        params[name] = str(annotation)
                    else:
                        params[name] = "Any"

            # Parse return type
            returns = (
                str(sig.return_annotation)
                if sig.return_annotation != inspect.Parameter.empty
                else "None"
            )

            # Create documentation
            doc_id = f"{class_name}.{method_name}"
            method_doc = MethodDoc(
                name=method_name,
                description=doc,
                parameters=params,
                returns=returns,
                examples=examples or [],
                notes=notes or [],
            )

            # Store documentation
            self.method_docs[doc_id] = method_doc

            return method_doc

        except Exception as e:
            self.logger.error(
                f"Error documenting method {class_name}.{method_name}: {str(e)}"
            )
            return None

    def generate_dependency_graph(
        self,
        module_dependencies: Dict[str, List[str]],
        output_file: str = "dependencies",
    ) -> Path:
        """Generate a dependency graph using Graphviz.

        Args:
            module_dependencies: Dictionary mapping modules to their dependencies
            output_file: Name of the output file (without extension)

        Returns:
            Path to the generated graph file
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
            output_path = self.output_dir / f"{output_file}.pdf"
            dot.render(str(output_path), cleanup=True)

            return output_path

        except Exception as e:
            self.logger.error(f"Error generating dependency graph: {str(e)}")
            return None

    def generate_sphinx_docs(self) -> None:
        """Generate Sphinx documentation."""
        try:
            # Create Sphinx configuration
            conf_py = self.output_dir / "conf.py"
            conf_content = [
                "project = 'Space Muck'",
                "copyright = '2025, Space Muck Team'",
                "author = 'Space Muck Team'",
                "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']",
                "html_theme = 'sphinx_rtd_theme'",
                "master_doc = 'index'",
            ]
            conf_py.write_text("\n".join(conf_content))

            # Create index file
            index_rst = self.output_dir / "index.rst"
            index_content = [
                "Space Muck Documentation",
                "=====================",
                "",
                ".. toctree::",
                "   :maxdepth: 2",
                "   :caption: Contents:",
                "",
                "   modules",
                "   dependencies",
            ]
            index_rst.write_text("\n".join(index_content))

            # Create module documentation
            modules_rst = self.output_dir / "modules.rst"
            module_content = [
                "Modules",
                "=======",
                "",
                ".. toctree::",
                "   :maxdepth: 4",
                "",
                "   api",
            ]
            modules_rst.write_text("\n".join(module_content))

            # Create API documentation
            api_rst = self.output_dir / "api.rst"
            api_content = ["API Reference", "============", ""]

            # Add method documentation
            for doc_id, doc in self.method_docs.items():
                api_content.extend(
                    [
                        doc_id,
                        "-" * len(doc_id),
                        "",
                        doc.description,
                        "",
                        "Parameters:",
                        "~~~~~~~~~~",
                        "",
                    ]
                )

                api_content.extend(
                    f"* **{name}** (*{type_info}*)"
                    for name, type_info in doc.parameters.items()
                )

                api_content.extend(["", "Returns:", "~~~~~~~~", f"*{doc.returns}*", ""])

                if doc.examples:
                    api_content.extend(
                        [
                            "Examples:",
                            "~~~~~~~~~",
                            ".. code-block:: python",
                            "",
                            *[f"    {line}" for line in doc.examples],
                            "",
                        ]
                    )

                if doc.notes:
                    api_content.extend(["Notes:", "~~~~~~", "", *doc.notes, ""])

            api_rst.write_text("\n".join(api_content))

        except Exception as e:
            self.logger.error(f"Error generating Sphinx docs: {str(e)}")

    def document_configuration(
        self, config: Dict[str, Any], output_file: str = "configuration.rst"
    ) -> None:
        """Document configuration options.

        Args:
            config: Dictionary of configuration options
            output_file: Name of the output file
        """
        try:
            content = ["Configuration", "=============", ""]

            def format_value(value: Any) -> str:
                """Format a configuration value for documentation."""
                if isinstance(value, dict):
                    return (
                        "{\n"
                        + "\n".join(
                            f"    {k}: {format_value(v)}" for k, v in value.items()
                        )
                        + "\n}"
                    )
                elif isinstance(value, list):
                    return (
                        "[\n"
                        + "\n".join(f"    {format_value(v)}" for v in value)
                        + "\n]"
                    )
                else:
                    return str(value)

            # Document each configuration option
            for key, value in config.items():
                content.extend([key, "-" * len(key), "", format_value(value), ""])

            # Write documentation
            output_path = self.output_dir / output_file
            output_path.write_text("\n".join(content))

        except Exception as e:
            self.logger.error(f"Error documenting configuration: {str(e)}")

    def document_project_structure(
        self, root_dir: Path, output_file: str = "structure.rst"
    ) -> None:
        """Document project directory structure.

        Args:
            root_dir: Root directory of the project
            output_file: Name of the output file
        """
        try:
            content = ["Project Structure", "================", ""]

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

            # Write documentation
            output_path = self.output_dir / output_file
            output_path.write_text("\n".join(content))

        except Exception as e:
            self.logger.error(f"Error documenting project structure: {str(e)}")
