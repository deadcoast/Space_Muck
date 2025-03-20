## Parsers for Project Mapping

### Header Map Parser
**Purpose:**
The header parser works as part of your Python analysis and enhancement system to:
1. Maintain a standardized way to track module information
2. Automatically update module relationships as code changes
3. Keep documentation synchronized with actual code structure

**When to Use:**
Use this system when you need to:
1. Track module dependencies actively
2. Maintain clear documentation of module relationships
3. Automatically update module headers during enhancement
4. Keep track of module lineage (parent/child relationships)

**How it Works:**
```python

```

**Integration Points:**
- Place the header parser in `modules/core/analysis/`
- Use it in conjunction with your enhancement system
- It works with both your project analysis map and module enhancements

**Key Features:**
1. Maintains consistent header format
2. Updates automatically with code changes
3. Tracks dependencies actively
4. Preserves module relationships

### Project Analysis Map Parser

This parser:
- Takes the human-readable map format
- Parses it into structured data
- Provides methods to work with the parsed information
- Can be used programmatically while maintaining the readable format
- Place the Project Analysis parser in `modules/core/analysis/`
- Use it in conjunction with your enhancement system
- It works with both your project analysis map and module enhancements

```python
# modules/core/analysis/map_parser.py
# 
# -----------------------------
#  PROJECT MAP PARSER
# -----------------------------
# 
# Parent: analysis
# Dependencies: database

import re
from pathlib import Path
from typing import Dict, List, Optional

class ProjectMapParser:
    """
    Parses and works with human-readable Project Analysis Maps
    """
    def __init__(self, map_content: str):
        self.raw_content = map_content
        self.structure = {}
        self.enhancements = []
        self.dependencies = {
            'primary': [],
            'secondary': []
        }
        self._parse_map()

    def _parse_map(self):
        """Parse the different sections of the map"""
        # Extract structure section
        structure_match = re.search(
            r'Original Structure:(.*?)(?=Enhancement Targets:)', 
            self.raw_content, 
            re.DOTALL
        )
        if structure_match:
            self._parse_structure(structure_match.group(1))

        # Extract enhancement targets
        enhancements_match = re.search(
            r'Enhancement Targets:(.*?)(?=Dependencies Found:)',
            self.raw_content,
            re.DOTALL
        )
        if enhancements_match:
            self._parse_enhancements(enhancements_match.group(1))

        # Extract dependencies
        dependencies_match = re.search(
            r'Dependencies Found:(.*?)$',
            self.raw_content,
            re.DOTALL
        )
        if dependencies_match:
            self._parse_dependencies(dependencies_match.group(1))

    def _parse_structure(self, structure_text: str):
        """Parse project structure from text"""
        lines = structure_text.strip().split('\n')
        current_path = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            indent_level = (len(line) - len(line.lstrip('│├└── ')))
            name = line.lstrip('│├└── ')
            
            # Adjust current path based on indent level
            current_path = current_path[:indent_level]
            current_path.append(name)
            
            # Build structure dict
            self._add_to_structure(current_path)

    def _parse_enhancements(self, enhancements_text: str):
        """Parse enhancement targets from text"""
        lines = enhancements_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                module, enhancement = line.split(':', 1)
                self.enhancements.append({
                    'module': module.strip(),
                    'enhancement': enhancement.strip()
                })

    def _parse_dependencies(self, dependencies_text: str):
        """Parse dependencies from text"""
        lines = dependencies_text.strip().split('\n')
        current_type = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('- Primary:'):
                current_type = 'primary'
                continue
            elif line.startswith('- Secondary:'):
                current_type = 'secondary'
                continue
                
            if current_type and line.startswith('- '):
                self.dependencies[current_type].append(line[2:].strip())

    def _add_to_structure(self, path_parts: List[str]):
        """Add a path to the structure dict"""
        current = self.structure
        for part in path_parts:
            if part not in current:
                current[part] = {}
            current = current[part]

    def get_module_paths(self) -> List[str]:
        """Get list of all module paths"""
        paths = []
        def collect_paths(structure, current_path=""):
            for name, substructure in structure.items():
                path = f"{current_path}/{name}" if current_path else name
                if path.endswith('.py'):
                    paths.append(path)
                collect_paths(substructure, path)
        
        collect_paths(self.structure)
        return paths

    def get_enhancements_for_module(self, module_name: str) -> List[str]:
        """Get planned enhancements for a specific module"""
        return [
            e['enhancement'] 
            for e in self.enhancements 
            if e['module'] == module_name
        ]

    def get_dependencies_by_type(self, dep_type: str) -> List[str]:
        """Get dependencies of a specific type"""
        return self.dependencies.get(dep_type, [])

# Usage example:
if __name__ == "__main__":
    map_content = '''
    PROJECT ANALYSIS MAP

    Original Structure:
        /project_root
        ├── main.py
        ├── utils/
        │   └── helpers.py
        └── modules/
            ├── user.py
            └── data.py

    Enhancement Targets:
        1. user.py: Add authentication
        2. data.py: Add validation

    Dependencies Found:
        - Primary:
            - database
            - config
        - Secondary:
            - logging
            - utils
    '''

    parser = ProjectMapParser(map_content)
    
    # Get all module paths
    print("Module paths:", parser.get_module_paths())
    
    # Get enhancements for a module
    print("Enhancements for user.py:", 
          parser.get_enhancements_for_module("user.py"))
    
    # Get primary dependencies
    print("Primary dependencies:", 
          parser.get_dependencies_by_type("primary"))
```
