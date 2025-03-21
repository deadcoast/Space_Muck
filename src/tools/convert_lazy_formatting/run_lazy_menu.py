#!/usr/bin/env python3
"""
run_lazy_menu.py

Simple executable script to run the lazy formatting conversion tool with an interactive ASCII menu.
"""

import os
import sys
import importlib.util

def run_lazy_menu():
    """Import and run the lazy_menu module's interactive menu."""
    # Import the lazy_menu module dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(script_dir, "lazy_menu.py")
    
    # Make sure we can import the module
    if not os.path.exists(module_path):
        print(f"Error: Could not find lazy_menu.py at {module_path}")
        return False
    
    # Import lazy_menu.py
    spec = importlib.util.spec_from_file_location("lazy_menu", module_path)
    lazy_menu = importlib.util.module_from_spec(spec)
    sys.modules["lazy_menu"] = lazy_menu
    spec.loader.exec_module(lazy_menu)
    
    # Run the interactive menu
    try:
        lazy_menu.run_interactive_menu()
        return True
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return False
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return False

if __name__ == "__main__":
    sys.exit(0 if run_lazy_menu() else 1)
