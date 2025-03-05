#!/usr/bin/env python3
"""
Test script to check imports.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    print("Attempting to import from src.generators...")
    from src.generators import AsteroidField
    print("Successfully imported AsteroidField")
except Exception as e:
    print(f"Error importing AsteroidField: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nAttempting to import from src.entities...")
    from src.entities.player import Player
    print("Successfully imported Player")
except Exception as e:
    print(f"Error importing Player: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nAttempting to import from src.config...")
    from src.config import GRID_WIDTH, GRID_HEIGHT
    print(f"Successfully imported config constants: GRID_WIDTH={GRID_WIDTH}, GRID_HEIGHT={GRID_HEIGHT}")
except Exception as e:
    print(f"Error importing config constants: {e}")
    import traceback
    traceback.print_exc()

print("\nImport test complete")
