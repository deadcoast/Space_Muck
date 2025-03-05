"""
Test a single import
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

print("Testing import...")
import src.entities.base_entity
print("Import successful")
