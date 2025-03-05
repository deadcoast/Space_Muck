"""
Test individual imports one by one
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# First, test base_entity import
print("Testing base_entity import...")
try:
    from src.entities.base_entity import BaseEntity
    print("✓ BaseEntity imported successfully")
except Exception as e:
    print(f"✗ Failed to import BaseEntity: {e}")
    import traceback
    print(traceback.format_exc())

# Then, test base_generator import
print("\nTesting base_generator import...")
try:
    from src.generators.base_generator import BaseGenerator
    print("✓ BaseGenerator imported successfully")
except Exception as e:
    print(f"✗ Failed to import BaseGenerator: {e}")
    import traceback
    print(traceback.format_exc())
