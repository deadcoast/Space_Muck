#!/usr/bin/env python3
"""
main.py

Entry point for the lazy formatting conversion tool with an interactive ASCII menu.
"""

import importlib.util
import os
import sys

# Dynamically import the lazy_menu module from the current directory
try:
    # Try direct import first for package usage
    from . import lazy_menu
except ImportError:
    # For direct script execution
    module_path = os.path.join(os.path.dirname(__file__), "lazy_menu.py")
    spec = importlib.util.spec_from_file_location("lazy_menu", module_path)
    lazy_menu = importlib.util.module_from_spec(spec)
    sys.modules["lazy_menu"] = lazy_menu
    spec.loader.exec_module(lazy_menu)


def main():
    """Main entry point for the lazy formatting conversion tool."""
    lazy_menu.run_interactive_menu()


if __name__ == "__main__":
    main()
