"""
Simple script to test imports and diagnose issues.
"""

# Standard library imports
import os
import sys

# Third-party library imports

# Local application imports
import traceback

# Print current directory and Python path
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Try different import approaches
try:
    print("\nTrying relative imports...")
    from ui.draw_utils import draw_text
    print("✓ Successfully imported draw_text from ui.draw_utils")
except Exception as e:
    print(f"✗ Error importing draw_text from ui.draw_utils: {e}")
    traceback.print_exc()

try:
    print("\nTrying absolute imports...")
    # Import draw_text directly from the module file, not the package
    from src.ui.draw_utils import draw_text  # This is importing from draw_utils.py file
    print("✓ Successfully imported draw_text from src.ui.draw_utils")
except Exception as e:
    print(f"✗ Error importing draw_text from src.ui.draw_utils: {e}")
    traceback.print_exc()

# Check if ui is a package
try:
    print("\nChecking if ui is a package...")
    import ui
    print(f"✓ ui is a package, located at: {ui.__file__}")
except Exception as e:
    print(f"✗ Error importing ui: {e}")
    traceback.print_exc()

# Check if src.ui is a package
try:
    print("\nChecking if src.ui is a package...")
    import src.ui
    print(f"✓ src.ui is a package, located at: {src.ui.__file__}")
except Exception as e:
    print(f"✗ Error importing src.ui: {e}")
    traceback.print_exc()

print("\nTest complete.")
