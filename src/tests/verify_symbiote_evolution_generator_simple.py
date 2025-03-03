"""
Simple verification script for the SymbioteEvolutionGenerator class.

This script checks that the SymbioteEvolutionGenerator class is properly structured
and inherits from BaseGenerator. It verifies the existence of required methods
and calls to parent class methods.
"""

import os
import sys

# Get the parent directory to access the src folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


def check_file_exists(file_path):
    """Check if a file exists at the given path."""
    return os.path.isfile(file_path)


def main():
    """Main verification function."""
    print("=== Simple SymbioteEvolutionGenerator Verification ===")

    # Check if the files exist
    base_generator_path = os.path.join(
        parent_dir, "src", "entities", "base_generator.py"
    )
    symbiote_generator_path = os.path.join(
        parent_dir, "src", "generators", "symbiote_evolution_generator.py"
    )
    symbiote_algorithm_path = os.path.join(
        parent_dir, "src", "algorithms", "symbiote_algorithm.py"
    )

    if not check_file_exists(base_generator_path):
        print(f"Error: BaseGenerator file does not exist at {base_generator_path}")
        return False

    if not check_file_exists(symbiote_generator_path):
        print(
            f"Error: SymbioteEvolutionGenerator file does not exist at {symbiote_generator_path}"
        )
        return False

    if not check_file_exists(symbiote_algorithm_path):
        print(
            f"Error: SymbioteEvolutionAlgorithm file does not exist at {symbiote_algorithm_path}"
        )
        return False

    # Check file contents to verify inheritance
    try:
        with open(symbiote_generator_path, "r") as f:
            content = f.read()
            if "class SymbioteEvolutionGenerator(BaseGenerator)" not in content:
                print(
                    "Error: SymbioteEvolutionGenerator does not inherit from BaseGenerator"
                )
                return False
            print(
                "Inheritance check passed: SymbioteEvolutionGenerator inherits from BaseGenerator"
            )
    except Exception as e:
        print(f"Error reading SymbioteEvolutionGenerator file: {e}")
        return False

    # Check for required methods in the file content
    required_methods = [
        "generate_initial_colonies",
        "generate_mineral_distribution",
        "simulate_evolution",
        "generate_mutation_map",
    ]

    # Check for method calls to BaseGenerator methods
    required_method_calls = [
        "generate_noise_layer",
        "apply_cellular_automaton",
        "create_clusters",
    ]

    try:
        with open(symbiote_generator_path, "r") as f:
            content = f.read()

            if missing_methods := [
                method for method in required_methods if f"def {method}" not in content
            ]:
                print(
                    f"Error: SymbioteEvolutionGenerator missing required methods: {', '.join(missing_methods)}"
                )
                return False
            else:
                print("Method check passed: All required methods are implemented")

            if missing_calls := [
                method_call
                for method_call in required_method_calls
                if f"self.{method_call}" not in content
            ]:
                print(
                    f"Warning: SymbioteEvolutionGenerator may not be using BaseGenerator methods: {', '.join(missing_calls)}"
                )
                print(
                    "This might be OK if the methods are called differently or reimplemented."
                )
            else:
                print(
                    "Method call check passed: All expected BaseGenerator method calls are present"
                )
    except Exception as e:
        print(f"Error reading SymbioteEvolutionGenerator file: {e}")
        return False

    # Check for SymbioteEvolutionAlgorithm integration
    try:
        with open(symbiote_generator_path, "r") as f:
            content = f.read()
            if (
                "from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm"
                not in content
            ):
                print(
                    "Error: SymbioteEvolutionGenerator does not import SymbioteEvolutionAlgorithm"
                )
                return False
            if "self.evolution_algorithm = SymbioteEvolutionAlgorithm" not in content:
                print(
                    "Error: SymbioteEvolutionGenerator does not initialize SymbioteEvolutionAlgorithm"
                )
                return False
            print(
                "Integration check passed: SymbioteEvolutionAlgorithm is properly integrated"
            )
    except Exception as e:
        print(f"Error checking SymbioteEvolutionAlgorithm integration: {e}")
        return False

    print("\n=== Simple verification completed successfully! ===")
    print(
        "The SymbioteEvolutionGenerator class is properly structured and inherits from BaseGenerator."
    )
    print("Note: Full functionality testing requires numpy, scipy, and perlin_noise.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
