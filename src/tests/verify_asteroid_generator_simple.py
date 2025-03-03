#!/usr/bin/env python3
"""
Simple verification script for the refactored AsteroidGenerator class.
This script just verifies the file structure and class inheritance without actually importing.
"""

import sys
import os
import importlib.util
import inspect

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

print(f"Python path: {sys.path}")


def check_file_exists(file_path):
    """Check if a file exists."""
    return os.path.isfile(file_path)


def main():
    """Main verification function."""
    print("=== Simple AsteroidGenerator Verification ===")

    # Check if the files exist
    base_generator_path = os.path.join(
        parent_dir, "src", "entities", "base_generator.py"
    )
    asteroid_generator_path = os.path.join(
        parent_dir, "src", "generators", "asteroid_generator.py"
    )

    if not check_file_exists(base_generator_path):
        print(f"Error: BaseGenerator file does not exist at {base_generator_path}")
        return False
    print(f"BaseGenerator file exists at {base_generator_path}")

    if not check_file_exists(asteroid_generator_path):
        print(
            f"Error: AsteroidGenerator file does not exist at {asteroid_generator_path}"
        )
        return False
    print(f"AsteroidGenerator file exists at {asteroid_generator_path}")

    # Check file contents to verify inheritance
    try:
        with open(asteroid_generator_path, "r") as f:
            content = f.read()
            if "class AsteroidGenerator(BaseGenerator)" in content:
                print("AsteroidGenerator properly inherits from BaseGenerator")
            else:
                print("Error: AsteroidGenerator does not inherit from BaseGenerator")
                return False
    except Exception as e:
        print(f"Error reading AsteroidGenerator file: {e}")
        return False

    # Check for required methods in the file content
    required_methods = ["generate_field", "generate_values", "generate_rare_resources"]

    # Check for method calls to BaseGenerator methods
    required_method_calls = [
        "generate_noise_layer",
        "apply_cellular_automaton",
        "create_clusters",
    ]

    try:
        with open(asteroid_generator_path, "r") as f:
            content = f.read()

            if missing_methods := [
                method for method in required_methods if f"def {method}" not in content
            ]:
                print(
                    f"Error: AsteroidGenerator missing required methods: {', '.join(missing_methods)}"
                )
                return False
            else:
                print(
                    f"Success: AsteroidGenerator has all required methods: {', '.join(required_methods)}"
                )

            if missing_calls := [
                method_call
                for method_call in required_method_calls
                if f"self.{method_call}" not in content
            ]:
                print(
                    f"Warning: AsteroidGenerator may not be using BaseGenerator methods: {', '.join(missing_calls)}"
                )
                print(
                    "This might be OK if the methods are called differently or reimplemented."
                )
            else:
                print(
                    f"Success: AsteroidGenerator uses BaseGenerator methods: {', '.join(required_method_calls)}"
                )
    except Exception as e:
        print(f"Error reading AsteroidGenerator file: {e}")
        return False

    print("\n=== Simple verification completed successfully! ===")
    print(
        "The AsteroidGenerator class is properly structured and inherits from BaseGenerator."
    )
    print("Note: Full functionality testing requires numpy, scipy, and matplotlib.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
