#!/usr/bin/env python3
"""
UI Import Fixer for Space Muck.

This script specifically addresses import issues in the UI components of Space Muck,
following the strict UI implementation rules:
1. Strict modularity - each file has ONE clear responsibility
2. No monolithic files - files must be kept small and focused
3. Clear separation - base classes, components, and utilities must remain separate
4. Proper import structure - following project import standards

Usage:
    python -m src.utils.fix_ui_imports [file_or_directory]

If no file or directory is specified, it will fix imports in the entire src/ui directory.
"""

# Standard library imports
import logging
import os
import re
import sys

# Third-party library imports

# Local application imports
from typing import List, Dict, Set, Tuple, Optional

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# UI module structure based on the memories
UI_STRUCTURE = {
    "base_modules": [
        "src.ui.ui_base.ascii_base",  # Contains only base classes and enums
    ],
    "component_modules": [
        "src.ui.ui_base.ascii_ui",  # Contains specific ASCII UI components
    ],
    "specialized_modules": [
        "src.ui.ui_element",  # Directory for specialized complex components
    ],
    "utility_modules": [
        "src.ui.draw_utils",  # Drawing utilities
    ],
}

# Mapping of common UI imports to their correct absolute paths
UI_IMPORT_MAP = {
    # Base classes and enums
    "UIStyle": "src.ui.ui_base.ascii_base",
    "AnimationStyle": "src.ui.ui_base.ascii_base",
    "UIElement": "src.ui.ui_base.ascii_base",
    "COLOR_TEXT": "src.ui.ui_base.ascii_base",
    "COLOR_BG": "src.ui.ui_base.ascii_base",
    "COLOR_HIGHLIGHT": "src.ui.ui_base.ascii_base",
    # ASCII UI components
    "ASCIIBox": "src.ui.ui_base.ascii_ui",
    "ASCIIPanel": "src.ui.ui_base.ascii_ui",
    "ASCIIButton": "src.ui.ui_base.ascii_ui",
    "ASCIIProgressBar": "src.ui.ui_base.ascii_ui",
    # Drawing utilities
    "draw_panel": "src.ui.draw_utils",
    "draw_histogram": "src.ui.draw_utils",
    "draw_tooltip": "src.ui.draw_utils",
}


def find_ui_imports(content: str) -> List[Tuple[str, str, str]]:
    """
    Find UI-related imports in the content.

    Args:
        content: Python file content

    Returns:
        List of tuples (import_line, module, imported_item)
    """
    ui_imports = []

    # Regular expressions for import statements
    import_regex = re.compile(r"^import\s+([\w.]+)(?:\s+as\s+[\w.]+)?", re.MULTILINE)
    from_import_regex = re.compile(r"^from\s+([\w.]+)\s+import\s+(.+)", re.MULTILINE)

    # Find all import statements
    for match in import_regex.finditer(content):
        line = match.group(0)
        module = match.group(1)

        # Check if it's a UI-related import
        if "ui" in module:
            ui_imports.append((line, module, module))

    # Find all from ... import statements
    for match in from_import_regex.finditer(content):
        line = match.group(0)
        module = match.group(1)
        imported_items = match.group(2)

        # Check if it's a UI-related import
        if "ui" in module:
            # Split imported items
            for item in re.split(r",\s*", imported_items):
                item = item.strip()
                # Remove 'as' alias if present
                if " as " in item:
                    item = item.split(" as ")[0].strip()
                ui_imports.append((line, module, item))

    return ui_imports


def fix_ui_imports(file_path: str, dry_run: bool = False) -> bool:
    """
    Fix UI imports in a Python file.

    Args:
        file_path: Path to the Python file
        dry_run: If True, don't modify the file, just print changes

    Returns:
        True if changes were made, False otherwise
    """
    logging.info(f"Processing {file_path}...")

    with open(file_path, "r") as f:
        content = f.read()

    # Find UI imports
    ui_imports = find_ui_imports(content)

    if not ui_imports:
        logging.info(f"No UI imports found in {file_path}")
        return False

    # Track changes
    changes_made = False
    new_content = content

    # Fix imports
    for import_line, module, imported_item in ui_imports:
        # Check if the import needs to be fixed
        if imported_item in UI_IMPORT_MAP:
            correct_module = UI_IMPORT_MAP[imported_item]

            # If the module is already correct, skip
            if module == correct_module:
                continue

            # Create the correct import statement
            if module == imported_item:  # It's an 'import X' statement
                correct_import = f"import {correct_module}"
            else:  # It's a 'from X import Y' statement
                correct_import = f"from {correct_module} import {imported_item}"

            # Replace the import
            new_content = new_content.replace(import_line, correct_import)
            changes_made = True
            logging.info(f"  Fixed: {import_line} -> {correct_import}")

        # Fix relative imports to absolute
        elif "ui" in module and not module.startswith("src."):
            # Convert to absolute import
            absolute_module = f"src.{module}"

            # Create the correct import statement
            if module == imported_item:  # It's an 'import X' statement
                correct_import = f"import {absolute_module}"
            else:  # It's a 'from X import Y' statement
                correct_import = f"from {absolute_module} import {imported_item}"

            # Replace the import
            new_content = new_content.replace(import_line, correct_import)
            changes_made = True
            logging.info(f"  Fixed: {import_line} -> {correct_import}")

    # Write changes to file
    if changes_made:
        if not dry_run:
            with open(file_path, "w") as f:
                f.write(new_content)
            logging.info(f"Fixed UI imports in {file_path}")
        else:
            logging.info(f"Would fix UI imports in {file_path}")
        return True
    else:
        logging.info(f"No changes needed in {file_path}")
        return False


def fix_ui_imports_in_directory(directory: str, dry_run: bool = False) -> int:
    """
    Fix UI imports in all Python files in a directory.

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
                if fix_ui_imports(file_path, dry_run):
                    modified_count += 1

    return modified_count


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Fix UI imports in Python files")
    parser.add_argument(
        "path",
        nargs="?",
        default=os.path.join(project_root, "src", "ui"),
        help="File or directory to process (default: src/ui directory)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't modify files, just print changes"
    )
    args = parser.parse_args()

    if os.path.isfile(args.path):
        if fix_ui_imports(args.path, args.dry_run):
            logging.info("1 file modified")
        else:
            logging.info("No files modified")
    elif os.path.isdir(args.path):
        modified_count = fix_ui_imports_in_directory(args.path, args.dry_run)
        logging.info(f"{modified_count} files modified")
    else:
        logging.error(f"Error: {args.path} is not a valid file or directory")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
