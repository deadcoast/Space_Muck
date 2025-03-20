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

# Local application imports
from typing import List, Tuple

# Third-party library imports


# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define module path constants to avoid duplication
MODULE_ASCII_BASE = "src.ui.ui_base.ascii_base"
MODULE_ASCII_UI = "src.ui.ui_base.ascii_ui"
MODULE_UI_ELEMENT = "src.ui.ui_element"
MODULE_DRAW_UTILS = "src.ui.draw_utils"

# UI module structure based on the memories
UI_STRUCTURE = {
    "base_modules": [
        MODULE_ASCII_BASE,  # Contains only base classes and enums
    ],
    "component_modules": [
        MODULE_ASCII_UI,  # Contains specific ASCII UI components
    ],
    "specialized_modules": [
        MODULE_UI_ELEMENT,  # Directory for specialized complex components
    ],
    "utility_modules": [
        MODULE_DRAW_UTILS,  # Drawing utilities
    ],
}

# Mapping of common UI imports to their correct absolute paths
UI_IMPORT_MAP = {
    # Base classes and enums
    "UIStyle": MODULE_ASCII_BASE,
    "AnimationStyle": MODULE_ASCII_BASE,
    "UIElement": MODULE_ASCII_BASE,
    "COLOR_TEXT": MODULE_ASCII_BASE,
    "COLOR_BG": MODULE_ASCII_BASE,
    "COLOR_HIGHLIGHT": MODULE_ASCII_BASE,
    # ASCII UI components
    "ASCIIBox": MODULE_ASCII_UI,
    "ASCIIPanel": MODULE_ASCII_UI,
    "ASCIIButton": MODULE_ASCII_UI,
    "ASCIIProgressBar": MODULE_ASCII_UI,
    # Drawing utilities
    "draw_panel": MODULE_DRAW_UTILS,
    "draw_histogram": MODULE_DRAW_UTILS,
    "draw_tooltip": MODULE_DRAW_UTILS,
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


def _create_correct_import(module: str, imported_item: str, correct_module: str) -> str:
    """Create the corrected import statement based on import type."""
    if module == imported_item:  # It's an 'import X' statement
        return f"import {correct_module}"
    else:  # It's a 'from X import Y' statement
        return f"from {correct_module} import {imported_item}"


def _process_mapped_import(import_line: str, module: str, imported_item: str, content: str) -> tuple[str, bool]:
    """Process imports that exist in the UI_IMPORT_MAP."""
    correct_module = UI_IMPORT_MAP[imported_item]
    
    # If the module is already correct, skip
    if module == correct_module:
        return content, False
        
    correct_import = _create_correct_import(module, imported_item, correct_module)
    new_content = content.replace(import_line, correct_import)
    logging.info(f"  Fixed: {import_line} -> {correct_import}")
    return new_content, True


def _process_relative_import(import_line: str, module: str, imported_item: str, content: str) -> tuple[str, bool]:
    """Process UI-related relative imports that need to be made absolute."""
    if "ui" not in module or module.startswith("src."):
        return content, False

    absolute_module = f"src.{module}"
    correct_import = _create_correct_import(module, imported_item, absolute_module)
    new_content = content.replace(import_line, correct_import)
    logging.info(f"  Fixed: {import_line} -> {correct_import}")
    return new_content, True


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

    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return False

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
        # Process imports that exist in the UI_IMPORT_MAP
        if imported_item in UI_IMPORT_MAP:
            new_content, changed = _process_mapped_import(
                import_line, module, imported_item, new_content
            )
        else:
            # Process relative imports that need to be made absolute
            new_content, changed = _process_relative_import(
                import_line, module, imported_item, new_content
            )
        changes_made = changes_made or changed
    # Write changes to file if needed
    if not changes_made:
        logging.info(f"No changes needed in {file_path}")
        return False

    if not dry_run:
        try:
            with open(file_path, "w") as f:
                f.write(new_content)
            logging.info(f"Fixed UI imports in {file_path}")
        except Exception as e:
            logging.error(f"Error writing to file {file_path}: {e}")
            return False
    else:
        logging.info(f"Would fix UI imports in {file_path}")

    return True


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
