#!/usr/bin/env python3
"""
Simple test script to check config imports.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    print("Attempting to import directly from config.py...")
    import src.config
    print("Successfully imported src.config")
    print(f"GRID_WIDTH = {src.config.GRID_WIDTH}")
    print(f"GRID_HEIGHT = {src.config.GRID_HEIGHT}")
    print(f"CELL_SIZE = {src.config.CELL_SIZE}")
except Exception as e:
    print(f"Error importing src.config: {e}")
    import traceback
    traceback.print_exc()

print("\nImport test complete")
