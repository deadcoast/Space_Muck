"""
Simple test script to verify basic import fixes.
This script attempts to import a minimal set of modules to check import paths.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("Testing basic imports...")
    
    # Test config import
    try:
        from src.config import GRID_WIDTH, GRID_HEIGHT
        print(f"✓ Config constants imported successfully: GRID_WIDTH={GRID_WIDTH}, GRID_HEIGHT={GRID_HEIGHT}")
    except Exception as e:
        print(f"✗ Failed to import config constants: {e}")
    
    # Test base_generator import
    try:
        from src.generators.base_generator import BaseGenerator
        print("✓ BaseGenerator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import BaseGenerator: {e}")
    
    print("\nBasic import test complete!")

if __name__ == "__main__":
    main()
