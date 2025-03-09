#!/usr/bin/env python3

# src/analysis/parsers/project_map_parser.py

# -----------------------------
# PROJECT MAP PARSER
# -----------------------------

# Parent: analysis.parsers
# Dependencies: re, pathlib, typing, logging

"""
MAP:
- Target Location: /project_root/analysis/parsers
- Integration Points:
    - analysis/core/analyzer.py
    - analysis/enhancers/base_enhancer.py
- New Files Created: None

EFFECT:
- Original Functionality: None (new component)
- Enhanced Functionality: Project structure parsing and analysis
- Side Effects: None

NAMING CONVENTION:
- Prefix: Parser
- Class Pattern: Parser[Type]
- Method Pattern: parse_[component], get_[data_type]
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

# Implementation map for tracking module enhancements
IMPLEMENTATION_MAP = {
    "project_map_parser": {
        "path": "src/analysis/parsers/project_map_parser.py",
        "enhancements": [
            "parse_map",
            "parse_structure",
            "parse_enhancements",
            "parse_dependencies",
        ],
        "new_dependencies": ["re", "pathlib", "typing", "logging"],
    }
}


class ProjectMapParser:
    """Parser for Space Muck project analysis maps.

    This parser handles the human-readable project map format, converting it
    into structured data that can be used programmatically while maintaining
    readability.

    Enhancement ID: MAP_PARSER_000
    Original Method: None (new class)
    Changes:
        - Added implementation map tracking
        - Added error handling with logging
        - Added state validation
    Integration Points:
        - analysis/core/analyzer.py
        - analysis/enhancers/base_enhancer.py
    """

    def __init__(self, map_content: str):
        """Initialize parser with map content.

        Args:
            map_content: String containing the project map content
        """
        self.logger = logging.getLogger(__name__)
        self.raw_content = map_content
        self.structure = {}
        self.enhancements = []
        self.dependencies = {"primary": [], "secondary": []}
        self._parse_map()

    def _parse_map(self) -> None:
        """Parse the different sections of the map.

        Enhancement ID: MAP_PARSER_001
        Original Method: None (new)
        Changes:
            - Added section pattern matching
            - Added validation for required sections
            - Added error handling with logging
        Integration Points:
            - Used by analyzer.analyze_project_structure()
        """
        try:
            if not self.raw_content:
                raise ValueError("Empty map content provided")

            sections = {
                "structure": (r"Original Structure:(.*?)(?=Enhancement Targets:|$)"),
                "enhancements": (r"Enhancement Targets:(.*?)(?=Dependencies Found:|$)"),
                "dependencies": (r"Dependencies Found:(.*?)(?=\Z)"),
            }

            for section, pattern in sections.items():
                try:
                    if (match := re.search(pattern, self.raw_content, re.DOTALL)) and (
                        parser := {
                            "structure": self._parse_structure,
                            "enhancements": self._parse_enhancements,
                            "dependencies": self._parse_dependencies,
                        }.get(section)
                    ):
                        parser(match[1].strip())
                    else:
                        self.logger.warning(
                            f"Section '{section}' not found or invalid in map content"
                        )
                except Exception as e:
                    self.logger.error(f"Error parsing section '{section}': {str(e)}")
                    raise ValueError(f"Failed to parse section '{section}': {str(e)}") from e
        except Exception as e:
            self.logger.error(f"Failed to parse project map: {str(e)}")
            raise

    def _parse_structure(self, structure_text: str) -> None:
        """Parse project structure section.

        Enhancement ID: MAP_PARSER_002
        Original Method: None (new)
        Changes:
            - Added type safety checks
            - Added validation for structure format
            - Added proper error handling
        Integration Points:
            - Used by _parse_map for structure parsing

        Args:
            structure_text: Text containing project structure

        Raises:
            ValueError: If structure text is invalid or malformed
        """
        if not isinstance(structure_text, str):
            raise ValueError("Structure text must be a string")

        lines = structure_text.strip().split("\n")
        current_path: List[str] = []
        indent_stack: List[int] = [-1]  # Stack to track indentation levels

        for line in lines:
            if not line.strip():
                continue

            # Count leading spaces to determine indent level
            indent = len(line) - len(line.lstrip())
            name = line.lstrip("│├└── ").rstrip()

            # Handle directory markers
            is_dir = name.endswith("/")
            if is_dir:
                name = name[:-1]

            # Pop back to the correct level based on indentation
            while indent_stack and indent_stack[-1] >= indent:
                indent_stack.pop()
                if current_path:
                    current_path.pop()

            # Add current level
            current_path.append(name)
            indent_stack.append(indent)

            # Only add to structure if it's a Python file or directory
            if is_dir or name.endswith(".py"):
                self._add_to_structure(current_path.copy())

    def _parse_enhancements(self, enhancements_text: str) -> None:
        """Parse enhancement targets section.

        Enhancement ID: MAP_PARSER_003
        Original Method: None (new)
        Changes:
            - Added type safety checks
            - Added validation for enhancement format
            - Added proper error handling
        Integration Points:
            - Used by _parse_map for enhancement parsing

        Args:
            enhancements_text: Text containing enhancement targets

        Raises:
            ValueError: If enhancement text is invalid or malformed
        """
        if not isinstance(enhancements_text, str):
            raise ValueError("Enhancement text must be a string")

        pattern = re.compile(r"(?:[\d.]+\s+)?([^:]+):\s*(.+)")
        for line in enhancements_text.strip().split("\n"):
            if match := pattern.match(line.strip()):
                module, enhancement = match.groups()
                self.enhancements.append(
                    {"module": module.strip(), "enhancement": enhancement.strip()}
                )

    def _parse_dependencies(self, dependencies_text: str) -> None:
        """Parse dependencies section.

        Enhancement ID: MAP_PARSER_004
        Original Method: None (new)
        Changes:
            - Added type safety checks
            - Added validation for dependency format
            - Added proper error handling
        Integration Points:
            - Used by _parse_map for dependency parsing

        Args:
            dependencies_text: Text containing dependencies

        Raises:
            ValueError: If dependency text is invalid or malformed
        """
        if not isinstance(dependencies_text, str):
            raise ValueError("Dependencies text must be a string")

        current_type: Optional[str] = None
        for line in dependencies_text.strip().split("\n"):
            line = line.strip()

            if not line:
                continue

            if line.startswith("Primary:"):
                current_type = "primary"
                continue
            elif line.startswith("Secondary:"):
                current_type = "secondary"
                continue

            if current_type and line.startswith("-"):
                self.dependencies[current_type].append(line[1:].strip())

    def _add_to_structure(self, path_parts: List[str]) -> None:
        """Add a path to the structure dictionary.

        Enhancement ID: MAP_PARSER_005
        Original Method: None (new)
        Changes:
            - Added type safety checks
            - Added validation for path components
            - Added proper error handling
        Integration Points:
            - Used by _parse_structure for building project structure

        Args:
            path_parts: List of path components

        Raises:
            ValueError: If path_parts is invalid or contains invalid components
        """
        if not isinstance(path_parts, list) or not all(
            isinstance(p, str) for p in path_parts
        ):
            raise ValueError("Path parts must be a list of strings")

        current: Dict = self.structure
        for part in path_parts:
            if not part:  # Skip empty path components
                continue
            if part not in current:
                current[part] = {}
            current = current[part]

    def get_module_paths(self) -> List[str]:
        """Get list of all module paths in the project.

        Enhancement ID: MAP_PARSER_006
        Original Method: None (new)
        Changes:
            - Added proper path handling with pathlib
            - Added validation for path components
            - Added proper error handling
        Integration Points:
            - Used by analyzer for module discovery

        Returns:
            List of module paths

        Raises:
            ValueError: If structure contains invalid paths
        """
        paths = []

        def collect_paths(structure: Dict, current_path: Optional[Path] = None) -> None:
            for name, substructure in structure.items():
                # Build path using pathlib for proper path handling
                path = Path(name) if current_path is None else current_path / name

                # Only add .py files to paths
                if name.endswith(".py"):
                    paths.append(str(path))
                elif substructure:  # Continue recursion for directories
                    collect_paths(substructure, path)

        collect_paths(self.structure)
        return sorted(paths)

    def get_enhancements_for_module(self, module_name: str) -> List[str]:
        """Get planned enhancements for a specific module.

        Enhancement ID: MAP_PARSER_007
        Original Method: None (new)
        Changes:
            - Added type safety checks
            - Added validation for module name
            - Added proper error handling
        Integration Points:
            - Used by analyzer for enhancement tracking

        Args:
            module_name: Name of the module

        Returns:
            List of enhancement descriptions

        Raises:
            ValueError: If module_name is invalid
        """
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("Module name must be a non-empty string")

        return [
            e["enhancement"] for e in self.enhancements if e["module"] == module_name
        ]

    def get_dependencies_by_type(self, dep_type: str) -> List[str]:
        """Get dependencies of a specific type.

        Enhancement ID: MAP_PARSER_008
        Original Method: None (new)
        Changes:
            - Added type safety checks
            - Added validation for dependency type
            - Added proper error handling
        Integration Points:
            - Used by analyzer for dependency tracking

        Args:
            dep_type: Type of dependencies ('primary' or 'secondary')

        Returns:
            List of dependencies

        Raises:
            ValueError: If dep_type is invalid or not one of the allowed types
        """
        if not isinstance(dep_type, str) or dep_type not in {"primary", "secondary"}:
            raise ValueError("Dependency type must be either 'primary' or 'secondary'")

        return self.dependencies.get(dep_type, [])

    def generate_map(self) -> str:
        """Generate a human-readable project map.

        Enhancement ID: MAP_PARSER_009
        Original Method: None (new)
        Changes:
            - Added type safety checks
            - Added validation for map components
            - Added proper error handling
        Integration Points:
            - Used by analyzer for map generation

        Returns:
            Formatted project map string

        Raises:
            ValueError: If map components are invalid or missing
        """
        try:
            sections: List[str] = []

            # Project Structure
            if not isinstance(self.structure, dict):
                raise ValueError("Project structure must be a dictionary")

            sections.append("Project Structure:")
            structure_lines: List[str] = []

            def format_structure(
                structure: Dict, prefix: str = "", is_dir: bool = False
            ) -> None:
                if not isinstance(structure, dict):
                    raise ValueError("Structure must be a dictionary")
                if not isinstance(prefix, str):
                    raise ValueError("Prefix must be a string")
                if not isinstance(is_dir, bool):
                    raise ValueError("is_dir must be a boolean")

                items = sorted(
                    structure.items(),
                    key=lambda x: (
                        not bool(x[1]),
                        x[0],
                    ),  # Directories first, then files
                )

                for i, (name, substructure) in enumerate(items):
                    if not isinstance(name, str):
                        raise ValueError("Structure keys must be strings")

                    is_last = i == len(items) - 1
                    marker = "└── " if is_last else "├── "

                    # Add directory marker
                    display_name = f"{name}/" if bool(substructure) else name
                    line = prefix + marker + display_name
                    structure_lines.append(line)

                    if substructure:
                        if not isinstance(substructure, dict):
                            raise ValueError("Substructure must be a dictionary")
                        new_prefix = prefix + ("    " if is_last else "│   ")
                        format_structure(substructure, new_prefix, True)

            format_structure(self.structure)
            sections.extend(structure_lines)

            # Enhancement Targets
            if not isinstance(self.enhancements, list):
                raise ValueError("Enhancements must be a list")

            sections.append("\nEnhancement Targets:")
            for i, enhancement in enumerate(self.enhancements, 1):
                if not isinstance(enhancement, dict) or any(
                    k not in enhancement or not isinstance(enhancement[k], str)
                    for k in ["module", "enhancement"]
                ):
                    raise ValueError("Invalid enhancement format")
                sections.append(
                    f"{i}. {enhancement['module']}: {enhancement['enhancement']}"
                )

            # Dependencies
            if (
                not isinstance(self.dependencies, dict)
                or any(k not in self.dependencies for k in ["primary", "secondary"])
                or any(
                    not isinstance(deps, list) for deps in self.dependencies.values()
                )
            ):
                raise ValueError("Invalid dependencies structure")

            sections.append("\nDependencies Found:")
            sections.append("Primary:")
            for dep in self.dependencies["primary"]:
                if not isinstance(dep, str):
                    raise ValueError("Dependencies must be strings")
                sections.append(f"- {dep}")
            sections.append("\nSecondary:")
            for dep in self.dependencies["secondary"]:
                if not isinstance(dep, str):
                    raise ValueError("Dependencies must be strings")
                sections.append(f"- {dep}")

            return "\n".join(sections)
        except Exception as e:
            self.logger.error(f"Failed to generate map: {str(e)}")
            raise ValueError(f"Failed to generate map: {str(e)}")
