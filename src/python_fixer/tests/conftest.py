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
ROOT_DIR = SRC_ROOT.parent.resolve()

# Verify critical paths and files
def verify_project_structure():
    """Verify project structure and critical files exist."""
    critical_paths = {
        'tests': TESTS_DIR,
        'project': PROJECT_ROOT,
        'src': SRC_ROOT,
        'init': PROJECT_ROOT / '__init__.py',
        'core': PROJECT_ROOT / 'core' / '__init__.py',
        'pyproject': ROOT_DIR / 'pyproject.toml'
    }
    
    for name, path in critical_paths.items():
        if not path.exists():
            raise RuntimeError(f"{name.title()} not found: {path}")
            
    return True

# Verify project structure
verify_project_structure()

# Configure test environment
os.environ["PYTHONPATH"] = str(ROOT_DIR)
os.environ["PYTHON_FIXER_TEST"] = "1"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"  # Prevent unwanted plugin loading

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
        # Add project paths in correct order
        paths_to_add = [
            str(ROOT_DIR),
            str(SRC_ROOT),
            str(PROJECT_ROOT),
            str(TESTS_DIR)
        ]
        
        # Remove existing paths to avoid duplicates
        sys.path[:] = [p for p in sys.path if p not in paths_to_add]
        
        # Add paths in correct order
        for path in reversed(paths_to_add):
            sys.path.insert(0, path)
            
        # Verify module availability
        import importlib.util
        required_modules = ['python_fixer', 'python_fixer.core.signatures']
        for module in required_modules:
            if not importlib.util.find_spec(module):
                raise RuntimeError(f"Required module not found: {module}")
            
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
    
    # Create test file structure
    (test_dir / "python_files").mkdir(exist_ok=True)
    (test_dir / "test_data").mkdir(exist_ok=True)
    (test_dir / "fixtures").mkdir(exist_ok=True)
    
    yield test_dir
    
    # Cleanup any remaining files
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
