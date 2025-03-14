#!/usr/bin/env python3
"""
Test runner for Space Muck tests.
This script adds the src directory to the Python path and runs the specified tests.
"""

import os
import sys
import unittest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

if __name__ == "__main__":
    # Discover and run tests directly
    test_dir = os.path.join(os.path.dirname(__file__), "src", "ui", "tests")
    test_suite = unittest.defaultTestLoader.discover(test_dir, pattern="test_*.py")

    # Run the tests
    unittest.TextTestRunner(verbosity=2).run(test_suite)
