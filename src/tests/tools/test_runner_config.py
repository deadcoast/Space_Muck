"""
Test runner for Space Muck.

This script runs all tests and generates a coverage report.
"""

import os
import sys
import unittest
import coverage

# Setup coverage
cov = coverage.Coverage(
    source=["src"], omit=["*/tests/*", "*/docs/*", "*/__pycache__/*"]
)
cov.start()

# Import test modules
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import test modules
from tests.unit_tests import *
from tests.integration_tests import *
from tests.performance_tests import *
from tests.regression_tests import *


def run_tests():
    """Run all tests and generate coverage report."""
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add tests from all modules
    test_suite.addTests(test_loader.loadTestsFromModule(sys.modules[__name__]))

    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)

    # Stop coverage and generate report
    cov.stop()
    cov.save()

    # Print report
    print("\nCoverage Summary:")
    cov.report()

    # Generate HTML report
    cov_dir = os.path.join(os.path.dirname(__file__), "coverage")
    os.makedirs(cov_dir, exist_ok=True)
    cov.html_report(directory=cov_dir)
    print(f"HTML report saved to {os.path.abspath(cov_dir)}")

    # Return result for CI systems
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
