#!/usr/bin/env python3
"""
Import Fixer for Space Muck.

This script fixes imports in Python files to follow the project's import standards.
It converts relative imports to absolute imports and organizes imports according
to the standards defined in import_standards.py.

Usage:
    python -m src.utils.fix_imports [file_or_directory]

If no file or directory is specified, it will fix imports in the entire src directory.
"""

# Standard library imports
import argparse
import os
import re
import sys

# Third-party library imports

# Local application imports
from typing import List, Dict, Set, Optional, Tuple
import importlib.util

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the import standards

    STDLIB_MODULES,
    THIRD_PARTY_MODULES,
    OPTIONAL_DEPENDENCIES,
    generate_standard_imports,
)

def is_stdlib_import(module_name: str) -> bool:
    """Check if a module is from the standard library."""
    # Strip away any submodules
    base_module = module_name.split('.')[0]
    return base_module in STDLIB_MODULES

def is_third_party_import(module_name: str) -> bool:
    """Check if a module is from a third-party library."""
    # Strip away any submodules
    base_module = module_name.split('.')[0]
    return base_module in THIRD_PARTY_MODULES

def is_local_import(module_name: str) -> bool:
    """Check if a module is a local import."""
    return not (is_stdlib_import(module_name) or is_third_party_import(module_name))

def extract_imports(content: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Extract import statements from Python content.
    
    Args:
        content: Python file content
        
    Returns:
        Tuple of (import lines, import info)
    """
    import_lines = []
    import_info = {
        "stdlib": [],
        "third_party": [],
        "local": [],
        "optional": [],
    }
    
    # Regular expressions for import statements
    import_regex = re.compile(r'^import\s+([\w.]+)(?:\s+as\s+[\w.]+)?(?:\s*,\s*([\w.]+)(?:\s+as\s+[\w.]+)?)*', re.MULTILINE)
    from_import_regex = re.compile(r'^from\s+([\w.]+)\s+import\s+(.+)', re.MULTILINE)
    
    # Find all import statements
    for match in import_regex.finditer(content):
        line = match.group(0)
        import_lines.append(line)
        
        # Process each module in the import statement
        modules = re.findall(r'([\w.]+)(?:\s+as\s+[\w.]+)?', line.replace('import ', '', 1))
        for module in modules:
            if is_stdlib_import(module):
                import_info["stdlib"].append(line)
            elif is_third_party_import(module):
                import_info["third_party"].append(line)
            else:
                import_info["local"].append(line)
    
    # Find all from ... import statements
    for match in from_import_regex.finditer(content):
        line = match.group(0)
        import_lines.append(line)
        
        module = match.group(1)
        if is_stdlib_import(module):
            import_info["stdlib"].append(line)
        elif is_third_party_import(module):
            import_info["third_party"].append(line)
            # Check if it's an optional dependency
            if module in OPTIONAL_DEPENDENCIES:
                import_info["optional"].append(module)
        else:
            # Convert relative imports to absolute
            if not module.startswith('src.'):
                # This is a relative import, needs to be fixed
                import_info["local"].append(line)
    
    return import_lines, import_info

def fix_imports_in_file(file_path: str, dry_run: bool = False) -> bool:
    """
    Fix imports in a Python file.
    
    Args:
        file_path: Path to the Python file
        dry_run: If True, don't modify the file, just print changes
        
    Returns:
        True if changes were made, False otherwise
    """
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract imports
    import_lines, import_info = extract_imports(content)
    
    if not import_lines:
        print(f"No imports found in {file_path}")
        return False
    
    # Generate new imports
    new_imports = generate_standard_imports(
        import_info["stdlib"],
        import_info["third_party"],
        import_info["local"],
        import_info["optional"]
    )
    
    # Replace imports in the content
    new_content = content
    for line in import_lines:
        new_content = new_content.replace(line, '')
    
    # Find the insertion point for imports
    # Look for the module docstring or the first non-comment, non-empty line
    docstring_end = re.search(r'""".*?"""|\'\'\'.*?\'\'\'', new_content, re.DOTALL)
    if docstring_end:
        insertion_point = docstring_end.end()
    else:
        # Find the first non-comment, non-empty line
        lines = new_content.split('\n')
        insertion_point = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                insertion_point = sum(len(l) + 1 for l in lines[:i])
                break
    
    # Insert new imports
    new_content = new_content[:insertion_point] + '\n\n' + new_imports + '\n' + new_content[insertion_point:]
    
    # Clean up multiple blank lines
    new_content = re.sub(r'\n{3,}', '\n\n', new_content)
    
    if new_content != content:
        if not dry_run:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Fixed imports in {file_path}")
        else:
            print(f"Would fix imports in {file_path}")
        return True
    else:
        print(f"No changes needed in {file_path}")
        return False

def fix_imports_in_directory(directory: str, dry_run: bool = False) -> int:
    """
    Fix imports in all Python files in a directory.
    
    Args:
        directory: Directory to process
        dry_run: If True, don't modify files, just print changes
        
    Returns:
        Number of files modified
    """
    modified_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_imports_in_file(file_path, dry_run):
                    modified_count += 1
    
    return modified_count

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Fix imports in Python files')
    parser.add_argument('path', nargs='?', default=os.path.join(project_root, 'src'),
                        help='File or directory to process (default: src directory)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Don\'t modify files, just print changes')
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        if fix_imports_in_file(args.path, args.dry_run):
            print("1 file modified")
        else:
            print("No files modified")
    elif os.path.isdir(args.path):
        modified_count = fix_imports_in_directory(args.path, args.dry_run)
        print(f"{modified_count} files modified")
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
