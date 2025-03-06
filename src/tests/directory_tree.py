#!/usr/bin/env python3
"""
Directory Tree Generator

A simple script to generate a tree-like representation of the directory structure
starting from the directory where this script is placed or a specified directory.

Usage:
    1. Place this script anywhere in your project
    2. Run it with: python directory_tree.py
    3. Optional arguments:
       - Max depth: python directory_tree.py 3
       - Output file: python directory_tree.py -o tree_output.txt
       - Exclude patterns: python directory_tree.py -e "*.pyc" "custom_pattern"
       - Use default ignore list: python directory_tree.py --use-ignore
       - Use both default ignore and custom excludes: python directory_tree.py --use-ignore -e "custom_pattern"
       - Specify a custom directory: python directory_tree.py -c /path/to/directory

Author: Space Muck Team
"""

import os
import argparse
from pathlib import Path

# Default ignore patterns - modify this list to add your commonly ignored patterns
DEFAULT_IGNORE_PATTERNS = [
    "__pycache__",
    ".ruff_cache",
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    "__init__.py",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.exe",
    "*.egg-info",
    "*.egg",
    "*.whl",
    "build",
    "dist",
    "venv",
    "env",
    ".env",
    ".tox",
    ".coverage",
    "htmlcov",
    ".DS_Store",
]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a directory tree structure.")
    parser.add_argument(
        "max_depth",
        nargs="?",
        type=int,
        default=None,
        help="Maximum depth to traverse (default: unlimited)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file to save the tree (default: print to console)",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        nargs="+",
        default=[],
        help='Patterns to exclude (e.g., "*.pyc" "__pycache__")',
    )
    parser.add_argument(
        "--use-ignore",
        action="store_true",
        help="Use the default ignore patterns defined in the script",
    )
    parser.add_argument(
        "--show-ignore",
        action="store_true",
        help="Show the current default ignore patterns and exit",
    )
    parser.add_argument(
        "--add-ignore",
        nargs="+",
        default=[],
        help="Add patterns to the default ignore list for this run only",
    )
    parser.add_argument(
        "-c",
        "--custom-dir",
        type=str,
        default=None,
        help="Specify a custom directory to map instead of the script directory",
    )
    return parser.parse_args()


def should_exclude(path, exclude_patterns):
    """Check if the path should be excluded based on patterns."""
    path_str = str(path)
    for pattern in exclude_patterns:
        # Simple wildcard matching
        if pattern.startswith("*") and pattern[1:] in path_str:
            return True
        elif pattern in path_str:
            return True
    return False


def generate_tree(
    directory, prefix="", max_depth=None, current_depth=0, exclude_patterns=None
):
    """
    Generate a tree representation of the directory structure.

    Args:
        directory (Path): The directory to start from
        prefix (str): Prefix for the current line (used for formatting)
        max_depth (int, optional): Maximum depth to traverse
        current_depth (int): Current depth in the traversal
        exclude_patterns (list): Patterns to exclude

    Returns:
        str: The tree representation
    """
    if exclude_patterns is None:
        exclude_patterns = []

    if max_depth is not None and current_depth > max_depth:
        return ""

    # Skip if this path should be excluded
    if should_exclude(directory, exclude_patterns):
        return ""

    # Get the directory name
    directory_name = directory.name or str(directory)

    # Start with the current directory
    result = f"{prefix}├── {directory_name}/\n"

    # Get all items in the directory
    try:
        items = list(directory.iterdir())
        items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return f"{prefix}├── {directory_name}/ [Permission Denied]\n"

    # Process each item
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        new_prefix = prefix + ("    " if is_last else "│   ")

        if item.is_dir():
            # For directories, recursively generate the tree
            result += generate_tree(
                item,
                prefix=new_prefix,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                exclude_patterns=exclude_patterns,
            )
        elif not should_exclude(item, exclude_patterns):
            result += f"{new_prefix}├── {item.name}\n"

    return result


def show_ignore_patterns():
    """Display the current default ignore patterns."""
    print("Current default ignore patterns:")
    for pattern in DEFAULT_IGNORE_PATTERNS:
        print(f"  - {pattern}")
    print("\nYou can modify these patterns directly in the script.")
    print("Look for the DEFAULT_IGNORE_PATTERNS list near the top of the file.")


def main():
    """Main function to generate and output the directory tree."""
    args = parse_arguments()

    # Show ignore patterns if requested
    if args.show_ignore:
        show_ignore_patterns()
        return

    # Prepare exclude patterns
    exclude_patterns = list(
        args.exclude
    )  # Convert to list to avoid modifying the original

    # Add default ignore patterns if requested
    if args.use_ignore:
        exclude_patterns.extend(DEFAULT_IGNORE_PATTERNS)

    # Add additional ignore patterns for this run
    if args.add_ignore:
        exclude_patterns.extend(args.add_ignore)

    # Determine which directory to use
    if args.custom_dir:
        target_dir = Path(args.custom_dir)
        if not target_dir.exists() or not target_dir.is_dir():
            print(
                f"Error: The specified directory '{args.custom_dir}' does not exist or is not a directory."
            )
            return
    else:
        # Get the directory where the script is located
        target_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Generate the tree
    tree_output = f"Directory Tree for: {target_dir}\n"
    if exclude_patterns:
        tree_output += f"Excluded patterns: {', '.join(exclude_patterns)}\n"
    tree_output += generate_tree(
        target_dir, max_depth=args.max_depth, exclude_patterns=exclude_patterns
    )

    # Output the tree
    if args.output:
        with open(args.output, "w") as f:
            f.write(tree_output)
        print(f"Tree saved to {args.output}")
    else:
        print(tree_output)


if __name__ == "__main__":
    main()
