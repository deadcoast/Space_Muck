"""
Linting check utility for Space Muck codebase.

This script provides tools to check Python files for syntax errors, import issues,
and other common linting problems. It can be used as a standalone script or imported
as a module in other testing tools.
"""

# Standard library imports
import os
import sys

# Third-party library imports

# Local application imports
from typing import List, Tuple
import ast
import importlib.util

# Standard library imports

# Third-party imports
# Use importlib.util.find_spec to check if pylint is available
PYLINT_AVAILABLE = importlib.util.find_spec("pylint") is not None

# Import pylint if available
if PYLINT_AVAILABLE:
    from pylint import lint
else:
    import logging

    logging.warning("pylint not available. Some linting features will be limited.")


def check_syntax(file_path: str) -> bool:
    """Check a Python file for syntax errors.

    Args:
        file_path: Path to the Python file to check

    Returns:
        bool: True if no syntax errors were found, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the file to check for syntax errors
        ast.parse(content)
        print(f"✅ {file_path} is syntactically correct")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return False
    except Exception as e:
        print(f"❌ Error checking {file_path}: {e}")
        return False


def check_import_patterns(file_path: str) -> Tuple[bool, List[str]]:
    """Check if import statements follow the standardized pattern.

    The standard pattern is:
    1. Standard library imports
    2. Third-party imports
    3. Local application imports with 'src.' prefix

    Args:
        file_path: Path to the Python file to check

    Returns:
        Tuple[bool, List[str]]: (success, list of issues found)
    """
    issues = []
    try:
        return _extracted_from_check_import_patterns_(file_path, issues)
    except Exception as e:
        issues.append(f"Error checking import patterns: {e}")
        return False, issues


# TODO Rename this here and in `check_import_patterns`
def _extracted_from_check_import_patterns_(file_path, issues):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the file
    tree = ast.parse(content)

    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    # Check for 'src.' prefix in local imports
    for node in imports:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (
                not node.module.startswith("src.")
                and node.module not in ("builtins", "typing")
                and "." in node.module
            )
        ):
            issues.append(f"Local import '{node.module}' should use 'src.' prefix")

        # Check for import grouping with comments
    if imports:
        has_stdlib_comment = "# Standard library imports" in content
        has_thirdparty_comment = "# Third-party imports" in content
        has_local_comment = "# Local application imports" in content

        if (
            not has_stdlib_comment
            or not has_thirdparty_comment
            or not has_local_comment
        ):
            issues.append(
                "Missing import section comments. Should have '# Standard library imports', '# Third-party imports', and '# Local application imports'"
            )

    return not issues, issues


def check_file(file_path: str) -> bool:
    """Check a Python file for syntax errors and import pattern issues.

    Args:
        file_path: Path to the Python file to check

    Returns:
        bool: True if no issues were found, False otherwise
    """
    syntax_ok = check_syntax(file_path)
    if not syntax_ok:
        return False

    import_ok, import_issues = check_import_patterns(file_path)
    if not import_ok:
        print(f"❌ Import pattern issues in {file_path}:")
        for issue in import_issues:
            print(f"  - {issue}")
        return False

    print(f"✅ {file_path} follows standardized import patterns")
    return True


def run_pylint(file_path: str) -> bool:
    """Run pylint on a Python file if available.

    Args:
        file_path: Path to the Python file to check

    Returns:
        bool: True if pylint passed or is not available, False if pylint found issues
    """
    if not PYLINT_AVAILABLE:
        print("⚠️ Pylint not available, skipping detailed linting")
        return True

    try:
        # Run pylint with our project's configuration
        pylint_args = [file_path, "--rcfile=pylintrc"]

        print(f"Running pylint on {file_path}...")
        pylint_runner = lint.Run(pylint_args, exit=False)
        score = pylint_runner.linter.stats.global_note

        if score >= 8.0:  # Consider 8.0+ as passing
            print(f"✅ Pylint score: {score:.1f}/10")
            return True
        else:
            print(f"❌ Pylint score: {score:.1f}/10 (threshold is 8.0)")
            return False
    except Exception as e:
        print(f"❌ Error running pylint: {e}")
        return False


if __name__ == "__main__":
    # Default file to check if none provided
    default_file = "src/tests/benchmarks/benchmark_comprehensive_gpu.py"

    # Get file path from command line arguments or use default
    file_path = sys.argv[1] if len(sys.argv) > 1 else default_file

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    # Run checks
    syntax_ok = check_syntax(file_path)
    import_ok, import_issues = check_import_patterns(file_path)

    if not syntax_ok:
        print("❌ Syntax check failed")
        sys.exit(1)

    if not import_ok:
        print("❌ Import pattern check failed:")
        for issue in import_issues:
            print(f"  - {issue}")
        sys.exit(1)

    # Run pylint if available
    if PYLINT_AVAILABLE:
        pylint_ok = run_pylint(file_path)
        if not pylint_ok:
            print("❌ Pylint check failed")
            sys.exit(1)

    print("✅ All checks passed!")
    sys.exit(0)
