"""Configure pytest for python_fixer tests."""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest

# Get the project root directory and source root
TESTS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_DIR.parent.resolve()
SRC_ROOT = PROJECT_ROOT.parent.resolve()

# Ensure paths exist
assert TESTS_DIR.exists(), f"Tests directory not found: {TESTS_DIR}"
assert PROJECT_ROOT.exists(), f"Project root not found: {PROJECT_ROOT}"
assert SRC_ROOT.exists(), f"Source root not found: {SRC_ROOT}"

# Configure test environment
os.environ["PYTHONPATH"] = str(SRC_ROOT)
os.environ["PYTHON_FIXER_TEST"] = "1"

@pytest.fixture(autouse=True)
def setup_test_env() -> Generator[None, None, None]:
    """Set up test environment for all tests.
    
    This fixture runs automatically for all tests and handles:
    1. Environment setup before each test
    2. Cleanup after each test
    """
    # Setup: Store original environment
    original_env = os.environ.copy()
    original_path = sys.path.copy()
    
    try:
        # Add project paths
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        
        yield
    finally:
        # Cleanup: Restore original state
        os.environ.clear()
        os.environ.update(original_env)
        sys.path[:] = original_path

@pytest.fixture
def temp_test_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for test files.
    
    Args:
        tmp_path: Pytest's temporary path fixture
    
    Returns:
        Path to temporary test directory
    """
    test_dir = tmp_path / "test_files"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
