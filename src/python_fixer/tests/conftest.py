"""Configure pytest for python_fixer tests."""

import os
import sys
from pathlib import Path

# Get the project root directory (parent of tests directory)
project_root = Path(__file__).parent.parent

# Add the project root to Python path so tests can import python_fixer
sys.path.insert(0, str(project_root))
