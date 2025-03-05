"""
Simple test script to verify imports one by one.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_imports():
    print("Testing imports one by one...")
    
    # Test procedural_generator imports
    try:
        from src.generators.procedural_generator import ProceduralGenerator
        print("✓ ProceduralGenerator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ProceduralGenerator: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Test asteroid_field imports
    try:
        from src.generators.asteroid_field import AsteroidField
        print("✓ AsteroidField imported successfully")
    except Exception as e:
        print(f"✗ Failed to import AsteroidField: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Test base_generator imports
    try:
        from src.generators.base_generator import BaseGenerator
        print("✓ BaseGenerator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import BaseGenerator: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_imports()
