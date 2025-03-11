#!/usr/bin/env python3
"""
Simple Directory Tree Generator

This script generates a clean, properly formatted directory tree structure
with consistent indentation and Unicode characters.

Usage:
    python directory_tree_simple.py [options]

Options:
    --path PATH         Directory to generate tree for (default: current script location)
    --output FILE       Output file (default: directory_tree.md in script directory)
    --ignore PATTERN    Patterns to ignore (can be specified multiple times)
    --max-depth N       Maximum depth to display

Author: Space Muck Team
"""

import os
import glob
import argparse
from pathlib import Path

# Default patterns to ignore
DEFAULT_IGNORE = [
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    ".pytest_cache",
    "*.egg-info",
    "dist",
    "build",
    "venv",
    "env",
    ".env",
    "__init__.py",
    ".tox",
    ".coverage",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a directory tree")
    parser.add_argument("--path", type=str, help="Directory to generate tree for")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument(
        "--ignore", action="append", help="Patterns to ignore", default=[]
    )
    parser.add_argument("--max-depth", type=int, help="Maximum depth to display")
    parser.add_argument(
        "--use-default-ignore", action="store_true", help="Use default ignore patterns"
    )
    return parser.parse_args()


def should_ignore(path, ignore_patterns):
    """Check if path should be ignored based on patterns."""
    path_str = str(path)

    for pattern in ignore_patterns:
        # Handle glob patterns
        if "*" in pattern or "?" in pattern:
            if any(glob.fnmatch.fnmatch(path_str, p) for p in glob.glob(pattern)):
                return True
        # Direct substring match
        elif pattern in path_str:
            return True

    return False


def tree(
    directory, ignore_patterns=None, max_depth=None, prefix="", is_last=False, depth=0
):
    """Generate directory tree string with proper formatting."""
    if ignore_patterns is None:
        ignore_patterns = []

    if max_depth is not None and depth > max_depth:
        return ""

    # Get directory name for display
    display_name = str(directory) if depth == 0 else directory.name
    # Start with current directory
    lines = []

    # Add directory name with proper prefix for first line
    if depth == 0:
        lines.append(f"Directory Tree for: {display_name}")
        if ignore_patterns:
            lines.append(f"Excluded patterns: {', '.join(ignore_patterns)}")
        connector = "├── "
    else:
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{display_name}/")

    # Get all entries in the directory
    try:
        entries = sorted(
            [e for e in directory.iterdir() if not should_ignore(e, ignore_patterns)],
            key=lambda x: (not x.is_dir(), x.name.lower()),
        )
    except (PermissionError, FileNotFoundError):
        return "\n".join(lines)

    # Process all entries
    for i, entry in enumerate(entries):
        is_entry_last = i == len(entries) - 1

        # Create appropriate prefix for children
        if depth == 0:
            new_prefix = ""
        else:
            new_prefix = f"{prefix}    " if is_last else f"{prefix}│   "
        # Handle directories recursively
        if entry.is_dir():
            if subtree := tree(
                entry,
                ignore_patterns,
                max_depth,
                new_prefix,
                is_entry_last,
                depth + 1,
            ):
                lines.append(subtree)
        else:
            # Handle files
            file_prefix = "└── " if is_entry_last else "├── "
            lines.append(f"{new_prefix}{file_prefix}{entry.name}")

    return "\n".join(lines)


def main():
    """Main function."""
    args = parse_args()

    # Set up ignore patterns
    ignore_patterns = args.ignore.copy()
    if args.use_default_ignore:
        ignore_patterns.extend(DEFAULT_IGNORE)

    # Determine the directory to process
    if args.path:
        directory = Path(args.path)
    else:
        directory = Path(os.path.dirname(os.path.abspath(__file__)))

    # Generate the tree
    result = tree(directory, ignore_patterns, args.max_depth)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        dir_name = directory.name
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"directory_tree_{dir_name}.md"
        )

    # Write to file
    with open(output_file, "w") as f:
        f.write("```\n" + result + "\n```\n")

    print(f"Tree saved to {output_file}")
    print("```")
    print(result)
    print("```")


if __name__ == "__main__":
    main()
