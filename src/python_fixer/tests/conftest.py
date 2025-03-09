"""Configure pytest for python_fixer tests."""

import os
import sys
from pathlib import Path

import pytest

# Get the project root directory (parent of tests directory)
project_root = Path(__file__).parent.parent
src_root = project_root.parent

# Add the src directory to Python path so tests can import python_fixer
sys.path.insert(0, str(src_root))

# Configure test environment
os.environ["PYTHONPATH"] = str(src_root)

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment for all tests."""
    # Add any test setup code here if needed
    yield
    # Add any test cleanup code here if needed

@pytest.fixture(autouse=True)
def python_fixer_path():
    """Add python_fixer package to Python path."""
    old_path = sys.path.copy()
    sys.path.insert(0, str(project_root))
    yield
    sys.path[:] = old_path
