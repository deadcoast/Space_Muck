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

# Local application imports
from typing import Dict, List, Tuple

from . import import_standards

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Get constants and functions from import_standards
STDLIB_MODULES = import_standards.STDLIB_MODULES
THIRD_PARTY_MODULES = import_standards.THIRD_PARTY_MODULES
OPTIONAL_DEPENDENCIES = import_standards.OPTIONAL_DEPENDENCIES
generate_standard_imports = import_standards.generate_standard_imports


def is_stdlib_import(module_name: str) -> bool:
    """Check if a module is from the standard library."""
    # Strip away any submodules
    base_module = module_name.split(".")[0]
    return base_module in STDLIB_MODULES


def is_third_party_import(module_name: str) -> bool:
    """Check if a module is from a third-party library."""
    # Strip away any submodules
    base_module = module_name.split(".")[0]
    return base_module in THIRD_PARTY_MODULES


def is_local_import(module_name: str) -> bool:
    """Check if a module is a local import."""
    return not (is_stdlib_import(module_name) or is_third_party_import(module_name))


def _categorize_import(
    module: str, line: str, import_info: Dict[str, List[str]]
) -> None:
    """Categorize an import based on its module name."""
    if is_stdlib_import(module):
        import_info["stdlib"].append(line)
    elif is_third_party_import(module):
        import_info["third_party"].append(line)
        # Check if it's an optional dependency
        if module in OPTIONAL_DEPENDENCIES:
            import_info["optional"].append(module)
    else:
        import_info["local"].append(line)


def _process_simple_import(
    match: re.Match, import_lines: List[str], import_info: Dict[str, List[str]]
) -> None:
    """Process a simple 'import x' statement."""
    line = match[0]
    import_lines.append(line)

    # Simpler regex to extract module names
    modules = re.findall(r"([\w.]+)(?:\s+as\s+[\w.]+)?", line.replace("import ", "", 1))
    for module in modules:
        _categorize_import(module, line, import_info)


def _process_from_import(
    match: re.Match, import_lines: List[str], import_info: Dict[str, List[str]]
) -> None:
    """Process a 'from x import y' statement."""
    line = match[0]
    import_lines.append(line)

    module = match[1]
    _categorize_import(module, line, import_info)


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

    # Simplified regular expressions for import statements
    import_regex = re.compile(r"^import\s+([\w.\s,]+)", re.MULTILINE)
    from_import_regex = re.compile(r"^from\s+([\w.]+)\s+import", re.MULTILINE)

    # Process all import statements
    for match in import_regex.finditer(content):
        _process_simple_import(match, import_lines, import_info)

    # Process all from ... import statements
    for match in from_import_regex.finditer(content):
        _process_from_import(match, import_lines, import_info)

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

    with open(file_path, "r", encoding="utf-8") as f:
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
        import_info["optional"],
    )

    # Replace imports in the content
    new_content = content
    for line in import_lines:
        new_content = new_content.replace(line, "")

    if docstring_end := re.search(r'""".*?"""|\'\'\'.*?\'\'\'', new_content, re.DOTALL):
        insertion_point = docstring_end.end()
    else:
        # Find the first non-comment, non-empty line
        lines = new_content.split("\n")
        insertion_point = next(
            (
                sum(len(line_item) + 1 for line_item in lines[:i])
                for i, line in enumerate(lines)
                if line.strip() and not line.strip().startswith("#")
            ),
            0,
        )
    # Insert new imports
    new_content = (
        new_content[:insertion_point]
        + "\n\n"
        + new_imports
        + "\n"
        + new_content[insertion_point:]
    )

    # Clean up multiple blank lines
    new_content = re.sub(r"\n{3,}", "\n\n", new_content)

    if new_content != content:
        if not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
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
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_imports_in_file(file_path, dry_run):
                    modified_count += 1

    return modified_count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix imports in Python files")
    parser.add_argument(
        "path",
        nargs="?",
        default=os.path.join(project_root, "src"),
        help="File or directory to process (default: src directory)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't modify files, just print changes"
    )
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


if __name__ == "__main__":
    sys.exit(main())
