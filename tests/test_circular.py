"""
Test for circular imports
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

print("Testing procedural_generator import...")
from src.generators.procedural_generator import ProceduralGenerator
print("ProceduralGenerator imported successfully")

print("\nTesting asteroid_field import...")
from src.generators.asteroid_field import AsteroidField
print("AsteroidField imported successfully")
