#!/usr/bin/env python3

# -----------------------------
# HEADER MAP PARSER
# -----------------------------

# Parent: analysis.parsers
# Dependencies: ast, pathlib, typing, logging

class HeaderMapParser:
    """Parser for maintaining and updating module header information.

    This parser maintains standardized headers across the Space Muck project,
    tracking module relationships, dependencies, and enhancement targets.

    Header Format:
    # -----------------------------
    # MODULE NAME
    # -----------------------------
    #
    # Parent: [parent module]
    # Dependencies: [comma-separated list]
    #
    # MAP: [module location in project]
    # EFFECT: [module's primary purpose]
    # NAMING: [naming convention explanation]
    """

# Standard library imports
import logging
import re

# Third-party library imports

# Local application imports
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import ast

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._header_pattern = re.compile(
            r"#\s*-+\s*\n"  # Opening line
            r"#\s*([^#]+)\s*\n"  # Module name
            r"#\s*-+\s*\n"  # Closing line
            r"(?:#\s*\n)?"  # Optional empty line
            r"(?:#\s*Parent:\s*([^\n]*)\n)?"  # Parent
            r"(?:#\s*Dependencies:\s*([^\n]*)\n)?"  # Dependencies
            r"(?:#\s*\n)?"  # Optional empty line
            r"(?:#\s*MAP:\s*([^\n]*)\n)?"  # Module location
            r"(?:#\s*EFFECT:\s*([^\n]*)\n)?"  # Module purpose
            r"(?:#\s*NAMING:\s*([^\n]*)\n)?",  # Naming convention
            re.MULTILINE,
        )

    def parse_header(self, content: str) -> Dict[str, Union[str, List[str]]]:
        """Parse module header information from content.

        Args:
            content: String containing the module's content

        Returns:
            Dictionary containing parsed header information:
            {
                'module_name': str,
                'parent': str,
                'dependencies': List[str],
                'map': str,
                'effect': str,
                'naming': str
            }
        """
        match = self._header_pattern.search(content)
        if not match:
            return self._create_empty_header_info()

        module_name, parent, deps, map_loc, effect, naming = match.groups()

        return {
            "module_name": module_name.strip() if module_name else "",
            "parent": parent.strip() if parent else "",
            "dependencies": [d.strip() for d in deps.split(",")] if deps else [],
            "map": map_loc.strip() if map_loc else "",
            "effect": effect.strip() if effect else "",
            "naming": naming.strip() if naming else "",
        }

    def generate_header(self, info: Dict[str, Union[str, List[str]]]) -> str:
        """Generate a standardized header from information.

        Args:
            info: Dictionary containing header information

        Returns:
            Formatted header string
        """
        template = (
            "# -----------------------------\n"
            "# {module_name}\n"
            "# -----------------------------\n"
            "#\n"
            "# Parent: {parent}\n"
            "# Dependencies: {dependencies}\n"
            "#\n"
            "# MAP: {map}\n"
            "# EFFECT: {effect}\n"
            "# NAMING: {naming}\n"
        )

        return template.format(
            module_name=info["module_name"],
            parent=info["parent"],
            dependencies=", ".join(info["dependencies"]),
            map=info["map"],
            effect=info["effect"],
            naming=info["naming"],
        )

    def update_header(
        self, content: str, updates: Dict[str, Union[str, List[str]]]
    ) -> Tuple[str, bool]:
        """Update existing header with new information.

        Args:
            content: Current module content
            updates: Dictionary containing fields to update

        Returns:
            Tuple of (updated content, whether changes were made)
        """
        current_info = self.parse_header(content)
        if not current_info["module_name"]:  # No existing header
            current_info = self._create_empty_header_info()

        # Update fields
        current_info.update(updates)

        # Generate new header
        new_header = self.generate_header(current_info)

        if not (match := self._header_pattern.search(content)):
            return new_header + "\n" + content, True
        new_content = content[: match.start()] + new_header + content[match.end() :]
        return new_content, True

    def analyze_imports(self, content: str) -> List[str]:
        """Analyze module imports to maintain dependency list.

        Args:
            content: Module content to analyze

        Returns:
            List of imported module names
        """
        try:
            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return sorted(set(imports))

        except Exception as e:
            self.logger.error(f"Error analyzing imports: {str(e)}")
            return []

    def _create_empty_header_info(self) -> Dict[str, Union[str, List[str]]]:
        """Create empty header information structure.

        Returns:
            Dictionary with empty header fields
        """
        return {
            "module_name": "",
            "parent": "",
            "dependencies": [],
            "map": "",
            "effect": "",
            "naming": "",
        }

    def extract_module_info(
        self, module_path: Path
    ) -> Dict[str, Union[str, List[str], Optional[str]]]:
        """Extract complete module information including docstrings.

        Args:
            module_path: Path to the module file

        Returns:
            Dictionary containing module information
        """
        try:
            with open(module_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get header info
            header_info = self.parse_header(content)

            # Get actual imports
            actual_deps = self.analyze_imports(content)

            # Parse module for additional info
            tree = ast.parse(content)
            module_doc = ast.get_docstring(tree) or ""

            return {
                **header_info,
                "actual_dependencies": actual_deps,
                "docstring": module_doc,
                "path": str(module_path),
            }

        except Exception as e:
            self.logger.error(
                f"Error extracting module info from {module_path}: {str(e)}"
            )
            return self._create_empty_header_info()
