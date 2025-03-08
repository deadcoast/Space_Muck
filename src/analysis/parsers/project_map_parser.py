#!/usr/bin/env python3

# -----------------------------
# PROJECT MAP PARSER
# -----------------------------
#
# Parent: analysis.parsers
# Dependencies: re, pathlib, typing, logging
#
# MAP: /project_root/analysis/parsers
# EFFECT: Parses and maintains project analysis maps
# NAMING: Parser[Type]

import logging
import re
from typing import Dict, List

class ProjectMapParser:
    """Parser for Space Muck project analysis maps.
    
    This parser handles the human-readable project map format, converting it
    into structured data that can be used programmatically while maintaining
    readability.
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
        self.dependencies = {
            'primary': [],
            'secondary': []
        }
        self._parse_map()

    def _parse_map(self) -> None:
        """Parse the different sections of the map."""
        sections = {
            'structure': (r'Project Structure:(.*?)(?=Enhancement Targets:|$)'),
            'enhancements': (r'Enhancement Targets:(.*?)(?=Dependencies Found:|$)'),
            'dependencies': (r'Dependencies Found:(.*?)(?=\Z)')
        }
        
        for section, pattern in sections.items():
            if match := re.search(pattern, self.raw_content, re.DOTALL):
                content = match.group(1).strip()
                parser = {
                    'structure': self._parse_structure,
                    'enhancements': self._parse_enhancements,
                    'dependencies': self._parse_dependencies
                }.get(section)
                
                if parser:
                    parser(content)

    def _parse_structure(self, structure_text: str) -> None:
        """Parse project structure section.
        
        Args:
            structure_text: Text containing project structure
        """
        lines = structure_text.strip().split('\n')
        current_path = []
        indent_stack = [-1]  # Stack to track indentation levels
        
        for line in lines:
            if not line.strip():
                continue
            
            # Count leading spaces to determine indent level
            indent = len(line) - len(line.lstrip())
            name = line.lstrip('│├└── ').rstrip()
            
            # Handle directory markers
            is_dir = name.endswith('/')
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
            if is_dir or name.endswith('.py'):
                self._add_to_structure(current_path.copy())

    def _parse_enhancements(self, enhancements_text: str) -> None:
        """Parse enhancement targets section.
        
        Args:
            enhancements_text: Text containing enhancement targets
        """
        pattern = re.compile(r'(?:[\d.]+\s+)?([^:]+):\s*(.+)')
        for line in enhancements_text.strip().split('\n'):
            if line.strip() and (match := pattern.match(line.strip())):
                module, enhancement = match.groups()
                self.enhancements.append({
                    'module': module.strip(),
                    'enhancement': enhancement.strip()
                })

    def _parse_dependencies(self, dependencies_text: str) -> None:
        """Parse dependencies section.
        
        Args:
            dependencies_text: Text containing dependencies
        """
        current_type = None
        for line in dependencies_text.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if line.startswith('Primary:'):
                current_type = 'primary'
                continue
            elif line.startswith('Secondary:'):
                current_type = 'secondary'
                continue
                
            if current_type and line.startswith('-'):
                self.dependencies[current_type].append(line[1:].strip())

    def _add_to_structure(self, path_parts: List[str]) -> None:
        """Add a path to the structure dictionary.
        
        Args:
            path_parts: List of path components
        """
        current = self.structure
        for part in path_parts:
            if part not in current:
                current[part] = {}
            current = current[part]

    def get_module_paths(self) -> List[str]:
        """Get list of all module paths in the project.
        
        Returns:
            List of module paths
        """
        paths = []
        
        def collect_paths(structure: Dict, current_path: str = "") -> None:
            for name, substructure in structure.items():
                # Build path, handling root directory specially
                path = f"{current_path}/{name}" if current_path else name
                
                # Only add .py files to paths
                if name.endswith('.py'):
                    paths.append(path)
                elif substructure:  # Continue recursion for directories
                    collect_paths(substructure, path)
        
        collect_paths(self.structure)
        return sorted(paths)

    def get_enhancements_for_module(self, module_name: str) -> List[str]:
        """Get planned enhancements for a specific module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            List of enhancement descriptions
        """
        return [
            e['enhancement'] 
            for e in self.enhancements 
            if e['module'] == module_name
        ]

    def get_dependencies_by_type(self, dep_type: str) -> List[str]:
        """Get dependencies of a specific type.
        
        Args:
            dep_type: Type of dependencies ('primary' or 'secondary')
            
        Returns:
            List of dependencies
        """
        return self.dependencies.get(dep_type, [])

    def generate_map(self) -> str:
        """Generate a human-readable project map.
        
        Returns:
            Formatted project map string
        """
        sections = []
        
        # Project Structure
        sections.append("Project Structure:")
        structure_lines = []
        
        def format_structure(structure: Dict, prefix: str = "", is_dir: bool = False) -> None:
            items = sorted(
                structure.items(),
                key=lambda x: (not bool(x[1]), x[0])  # Directories first, then files
            )
            
            for i, (name, substructure) in enumerate(items):
                is_last = i == len(items) - 1
                marker = "└── " if is_last else "├── "
                
                # Add directory marker
                display_name = f"{name}/" if bool(substructure) else name
                line = prefix + marker + display_name
                structure_lines.append(line)
                
                if substructure:
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    format_structure(substructure, new_prefix, True)
        
        format_structure(self.structure)
        sections.extend(structure_lines)
        
        # Enhancement Targets
        sections.append("\nEnhancement Targets:")
        for i, enhancement in enumerate(self.enhancements, 1):
            sections.append(
                f"{i}. {enhancement['module']}: {enhancement['enhancement']}"
            )
        
        # Dependencies
        sections.append("\nDependencies Found:")
        sections.append("Primary:")
        for dep in self.dependencies['primary']:
            sections.append(f"- {dep}")
        sections.append("\nSecondary:")
        for dep in self.dependencies['secondary']:
            sections.append(f"- {dep}")
        
        return "\n".join(sections)
