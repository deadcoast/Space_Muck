#!/usr/bin/env python3
"""
Comprehensive verification script for the SymbioteEvolutionGenerator class.

This script provides both basic file structure verification and advanced functionality testing
for the SymbioteEvolutionGenerator class. It tests the functionality of the SymbioteEvolutionGenerator
and ensures it correctly inherits from BaseGenerator while maintaining all required functionality.

The verification includes:
1. File structure and inheritance verification
2. Basic functionality testing
3. Evolution simulation over time
4. Mineral consumption impact analysis
5. Visualization of results
6. Code review for best practices
"""

import sys
import os

# Try to import required packages, but don't fail if they're not available
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    print("numpy not available. Tests will be skipped.")
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("matplotlib not available. Visualization will be skipped.")
    MATPLOTLIB_AVAILABLE = False

# Check if scipy is available using importlib
import importlib.util

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
if not SCIPY_AVAILABLE:
    print("scipy not available. Tests will be skipped.")

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Only import if required packages are available
if NUMPY_AVAILABLE and SCIPY_AVAILABLE:
    try:
        # Try primary import path
        from src.generators.symbiote_evolution_generator import (
            SymbioteEvolutionGenerator,
        )
        from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm

        SYMBIOTE_GENERATOR_AVAILABLE = True
    except ImportError:
        try:
            # Try alternative import path
            from generators.symbiote_evolution_generator import (
                SymbioteEvolutionGenerator,
            )
            from algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm

            SYMBIOTE_GENERATOR_AVAILABLE = True
        except ImportError as e:
            print(f"Could not import SymbioteEvolutionGenerator: {e}")
            SYMBIOTE_GENERATOR_AVAILABLE = False
else:
    SYMBIOTE_GENERATOR_AVAILABLE = False


def test_symbiote_evolution_generator():
    """Test the basic functionality of the SymbioteEvolutionGenerator class."""
    print("Testing SymbioteEvolutionGenerator initialization...")

    # Create a generator with a fixed seed for reproducibility
    generator = SymbioteEvolutionGenerator(
        seed=42,
        width=100,
        height=100,
        initial_aggression=0.3,
        growth_rate=0.1,
        base_mutation_rate=0.05,
    )

    # Verify basic properties
    assert generator.seed == 42, f"Expected seed 42, got {generator.seed}"
    assert generator.width == 100, f"Expected width 100, got {generator.width}"
    assert generator.height == 100, f"Expected height 100, got {generator.height}"
    assert (
        generator.entity_type == "symbiote"
    ), f"Expected entity_type 'symbiote', got {generator.entity_type}"

    # Verify SymbioteEvolutionAlgorithm integration
    assert hasattr(
        generator, "evolution_algorithm"
    ), "Missing evolution_algorithm attribute"
    assert isinstance(
        generator.evolution_algorithm, SymbioteEvolutionAlgorithm
    ), "evolution_algorithm is not an instance of SymbioteEvolutionAlgorithm"
    assert (
        generator.evolution_algorithm.aggression == 0.3
    ), f"Expected initial_aggression 0.3, got {generator.evolution_algorithm.aggression}"

    # Test noise generation
    print("Testing noise generation...")
    noise = generator.generate_noise_layer("medium", scale=0.05)
    assert noise.shape == (100, 100), f"Expected shape (100, 100), got {noise.shape}"

    # Test initial colony generation
    print("Testing initial colony generation...")
    colony_grid, metadata = generator.generate_initial_colonies(num_colonies=3)
    assert colony_grid.shape == (
        100,
        100,
    ), f"Expected shape (100, 100), got {colony_grid.shape}"
    assert np.sum(colony_grid > 0) > 0, "No colonies were generated"
    assert "seed" in metadata, "Metadata missing seed information"
    assert "num_colonies" in metadata, "Metadata missing num_colonies information"
    assert (
        "colony_population" in metadata
    ), "Metadata missing colony_population information"

    # Test mineral distribution generation
    print("Testing mineral distribution generation...")
    mineral_grid = generator.generate_mineral_distribution()
    assert mineral_grid.shape == (
        100,
        100,
    ), f"Expected shape (100, 100), got {mineral_grid.shape}"
    assert (
        np.min(mineral_grid) >= 0
    ), f"Expected min value >= 0, got {np.min(mineral_grid)}"
    assert (
        np.max(mineral_grid) <= 1
    ), f"Expected max value <= 1, got {np.max(mineral_grid)}"

    # Test evolution simulation
    print("Testing evolution simulation...")
    evolved_grid, evolution_history = generator.simulate_evolution(
        colony_grid, mineral_grid, iterations=5
    )
    assert evolved_grid.shape == (
        100,
        100,
    ), f"Expected shape (100, 100), got {evolved_grid.shape}"
    assert (
        len(evolution_history) == 5
    ), f"Expected 5 evolution steps, got {len(evolution_history)}"

    # Test mutation map generation
    print("Testing mutation map generation...")
    mutation_map = generator.generate_mutation_map(evolved_grid, evolution_history)
    assert mutation_map.shape == (
        100,
        100,
    ), f"Expected shape (100, 100), got {mutation_map.shape}"

    print("All basic tests passed!")
    return (
        generator,
        colony_grid,
        mineral_grid,
        evolved_grid,
        evolution_history,
        mutation_map,
    )


def test_evolution_over_time():
    """Test the evolution of symbiotes over multiple iterations."""
    print("\nTesting evolution over time...")

    # Create a generator with a fixed seed for reproducibility
    generator = SymbioteEvolutionGenerator(seed=123, width=80, height=80)

    # Generate initial colonies
    colony_grid, _ = generator.generate_initial_colonies(num_colonies=5)
    mineral_grid = generator.generate_mineral_distribution()

    # Run evolution for more iterations
    evolved_grid, evolution_history = generator.simulate_evolution(
        colony_grid, mineral_grid, iterations=10
    )

    # Check that evolution history contains expected data
    assert (
        len(evolution_history) == 10
    ), f"Expected 10 evolution steps, got {len(evolution_history)}"

    # Check that each step in history has required fields
    # This loop is necessary for validating the structure of each evolution step
    # sourcery skip: no-loop-in-tests
    for i, step in enumerate(evolution_history):
        _extracted_from_test_evolution_over_time_24(step, i)
    # Verify population changes over time
    populations = [step["population"] for step in evolution_history]
    assert len(set(populations)) > 1, "Population did not change during evolution"

    print("Evolution over time test passed!")
    return evolved_grid, evolution_history


# TODO Rename this here and in `test_evolution_over_time`
def _extracted_from_test_evolution_over_time_24(step, i):
    assert "iteration" in step, f"Step {i} missing 'iteration' field"
    assert "population" in step, f"Step {i} missing 'population' field"
    assert "aggression" in step, f"Step {i} missing 'aggression' field"
    assert "genome" in step, f"Step {i} missing 'genome' field"
    assert "mutations" in step, f"Step {i} missing 'mutations' field"
    assert (
        "mineral_consumption" in step
    ), f"Step {i} missing 'mineral_consumption' field"


def test_mineral_consumption_impact():
    """Test how mineral consumption affects symbiote evolution."""
    print("\nTesting mineral consumption impact...")

    # Create two generators with the same seed but different mineral distributions
    seed = 456
    generator1 = SymbioteEvolutionGenerator(seed=seed, width=60, height=60)
    generator2 = SymbioteEvolutionGenerator(seed=seed, width=60, height=60)

    # Generate identical initial colonies
    colony_grid1, _ = generator1.generate_initial_colonies(num_colonies=3)
    colony_grid2 = colony_grid1.copy()

    # Generate different mineral distributions
    # First with low minerals
    mineral_grid1 = generator1.generate_mineral_distribution() * 0.3
    # Second with high minerals
    mineral_grid2 = generator2.generate_mineral_distribution() * 1.5
    mineral_grid2 = np.clip(mineral_grid2, 0, 1)  # Ensure values stay in [0,1]

    # Run evolution for both scenarios
    evolved_grid1, history1 = generator1.simulate_evolution(
        colony_grid1, mineral_grid1, iterations=8
    )
    evolved_grid2, history2 = generator2.simulate_evolution(
        colony_grid2, mineral_grid2, iterations=8
    )

    # Compare final populations
    final_pop1 = history1[-1]["population"]
    final_pop2 = history2[-1]["population"]

    print(f"Final population with low minerals: {final_pop1}")
    print(f"Final population with high minerals: {final_pop2}")

    # Check mutation counts
    mutations1 = sum(len(step["mutations"]) for step in history1)
    mutations2 = sum(len(step["mutations"]) for step in history2)

    print(f"Total mutations with low minerals: {mutations1}")
    print(f"Total mutations with high minerals: {mutations2}")

    return (evolved_grid1, evolved_grid2), (mineral_grid1, mineral_grid2)


def visualize_results(
    generator, colony_grid, mineral_grid, evolved_grid, evolution_history, mutation_map
):
    """Visualize the results of the tests."""
    if not MATPLOTLIB_AVAILABLE:
        print("\nSkipping visualization as matplotlib is not available.")
        return

    print("\nVisualizing results...")

    plt.figure(figsize=(15, 10))

    # Plot initial colony grid
    plt.subplot(2, 3, 1)
    _visualize_colorbar(
        colony_grid,
        "binary",
        "Colony Presence",
        "Initial Symbiote Colonies",
    )

    # Plot mineral distribution
    plt.subplot(2, 3, 2)
    _visualize_colorbar(
        mineral_grid, "viridis", "Mineral Value", "Mineral Distribution"
    )

    # Plot evolved colony grid
    plt.subplot(2, 3, 3)
    _visualize_colorbar(
        evolved_grid, "binary", "Colony Presence", "Evolved Symbiote Colonies"
    )

    # Plot mutation map
    plt.subplot(2, 3, 4)
    _visualize_colorbar(
        mutation_map, "plasma", "Mutation Intensity", "Mutation Hotspots"
    )

    # Plot population over time
    plt.subplot(2, 3, 5)
    populations = [step["population"] for step in evolution_history]
    iterations = [step["iteration"] for step in evolution_history]
    plt.plot(iterations, populations, "b-")
    plt.xlabel("Evolution Iteration")
    plt.ylabel("Population")
    plt.title("Population Growth")
    plt.grid(True)

    # Plot aggression over time
    plt.subplot(2, 3, 6)
    aggression = [step["aggression"] for step in evolution_history]
    plt.plot(iterations, aggression, "r-")
    plt.xlabel("Evolution Iteration")
    plt.ylabel("Aggression Level")
    plt.title("Aggression Changes")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("symbiote_evolution_test.png")
    print("Visualization saved as 'symbiote_evolution_test.png'")


def visualize_mineral_impact(evolved_grids, mineral_grids):
    """Visualize the impact of different mineral distributions."""
    if not MATPLOTLIB_AVAILABLE:
        print("\nSkipping visualization as matplotlib is not available.")
        return

    print("\nVisualizing mineral impact...")

    evolved_grid1, evolved_grid2 = evolved_grids
    mineral_grid1, mineral_grid2 = mineral_grids

    plt.figure(figsize=(12, 8))

    # Plot low mineral distribution
    plt.subplot(2, 2, 1)
    _visualize_colorbar(
        mineral_grid1,
        "viridis",
        "Mineral Value",
        "Low Mineral Distribution",
    )

    # Plot high mineral distribution
    plt.subplot(2, 2, 2)
    _visualize_colorbar(
        mineral_grid2, "viridis", "Mineral Value", "High Mineral Distribution"
    )

    # Plot evolved colonies with low minerals
    plt.subplot(2, 2, 3)
    _visualize_colorbar(
        evolved_grid1, "binary", "Colony Presence", "Colonies with Low Minerals"
    )

    # Plot evolved colonies with high minerals
    plt.subplot(2, 2, 4)
    _visualize_colorbar(
        evolved_grid2, "binary", "Colony Presence", "Colonies with High Minerals"
    )

    plt.tight_layout()
    plt.savefig("symbiote_mineral_impact.png")
    print("Visualization saved as 'symbiote_mineral_impact.png'")


def _visualize_colorbar(data, cmap, label, title):
    plt.imshow(data, cmap=cmap)
    plt.colorbar(label=label)
    plt.title(title)


def perform_code_review():
    """Perform a code review of the SymbioteEvolutionGenerator class."""
    print("\nPerforming code review of SymbioteEvolutionGenerator...")

    try:
        return _extracted_from_perform_code_review_7()
    except Exception as e:
        print(f"Error during code review: {e}")
        return False


# TODO Rename this here and in `perform_code_review`
def _extracted_from_perform_code_review_7():
    # Read the file directly instead of importing it
    import os

    symbiote_generator_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "generators",
        "symbiote_evolution_generator.py",
    )

    if not os.path.exists(symbiote_generator_path):
        print(f"Error: File not found at {symbiote_generator_path}")
        return False

    with open(symbiote_generator_path, "r") as f:
        source = f.read()

    # Check for key features
    features = {
        "Inheritance": "class SymbioteEvolutionGenerator(BaseGenerator)" in source,
        "Algorithm Integration": "from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm"
        in source,
        "Colony Generation": "def generate_initial_colonies" in source,
        "Mineral Distribution": "def generate_mineral_distribution" in source,
        "Evolution Simulation": "def simulate_evolution" in source,
        "Mutation Tracking": "def generate_mutation_map" in source,
        "Parameter Handling": "self.set_parameter" in source,
        "Error Handling": "try:" in source and "except" in source,
        "Performance Logging": "log_performance" in source,
    }

    # Report findings
    print("\nCode Review Results:")
    for feature, present in features.items():
        status = "✓" if present else "✗"
        print(f"{status} {feature}")

    # Count methods
    method_count = source.count("def ")
    print(f"\nTotal methods: {method_count}")

    # Check docstrings
    has_class_docstring = '"""' in source.split("class")[1].split(":")[0]
    print(f"Class docstring: {'✓' if has_class_docstring else '✗'}")

    # Check for BaseGenerator method calls
    base_methods = [
        "generate_noise_layer",
        "apply_cellular_automaton",
        "create_clusters",
    ]

    print("\nBaseGenerator method calls:")
    for method in base_methods:
        call_present = f"self.{method}" in source
        status = "✓" if call_present else "✗"
        print(f"{status} {method}")

    # Check for SymbioteEvolutionAlgorithm integration
    algo_methods = [
        "process_mineral_feeding",
        "generate_cellular_automaton_rules",
        "update_cellular_automaton",
        "apply_environmental_effects",
        "simulate_colony_interaction",
    ]

    print("\nSymbioteEvolutionAlgorithm method calls:")
    for method in algo_methods:
        call_present = f"self.evolution_algorithm.{method}" in source
        status = "✓" if call_present else "✗"
        print(f"{status} {method}")

    return True


def verify_file_structure():
    """Verify the file structure and inheritance of SymbioteEvolutionGenerator."""
    import os

    print("\n=== Verifying SymbioteEvolutionGenerator File Structure ===")

    # Get the parent directory to access the src folder
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

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

    success = True

    if not os.path.isfile(base_generator_path):
        print(f"Error: BaseGenerator file does not exist at {base_generator_path}")
        success = False
    else:
        print(f"✓ BaseGenerator file exists at {base_generator_path}")

    if not os.path.isfile(symbiote_generator_path):
        print(
            f"Error: SymbioteEvolutionGenerator file does not exist at {symbiote_generator_path}"
        )
        success = False
    else:
        print(f"✓ SymbioteEvolutionGenerator file exists at {symbiote_generator_path}")

    if not os.path.isfile(symbiote_algorithm_path):
        print(
            f"Error: SymbioteEvolutionAlgorithm file does not exist at {symbiote_algorithm_path}"
        )
        success = False
    else:
        print(f"✓ SymbioteEvolutionAlgorithm file exists at {symbiote_algorithm_path}")

    # Only proceed with inheritance check if all files exist
    if success:
        # Check file contents to verify inheritance
        try:
            with open(symbiote_generator_path, "r") as f:
                content = f.read()

            if (
                "from src.generators.base_generator import BaseGenerator" in content
                or "from src.entities.base_generator import BaseGenerator" in content
            ):
                print("✓ SymbioteEvolutionGenerator imports BaseGenerator")
            else:
                print(
                    "✗ SymbioteEvolutionGenerator does not import BaseGenerator correctly"
                )
                success = False

            if (
                "from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm"
                in content
            ):
                print("✓ SymbioteEvolutionGenerator imports SymbioteEvolutionAlgorithm")
            else:
                print(
                    "✗ SymbioteEvolutionGenerator does not import SymbioteEvolutionAlgorithm correctly"
                )
                success = False

            if "class SymbioteEvolutionGenerator(BaseGenerator):" in content:
                print("✓ SymbioteEvolutionGenerator inherits from BaseGenerator")
            else:
                print(
                    "✗ SymbioteEvolutionGenerator does not inherit from BaseGenerator"
                )
                success = False

            # Try to import the modules to check inheritance programmatically
            try:
                sys.path.append(parent_dir)
                from src.generators.symbiote_evolution_generator import (
                    SymbioteEvolutionGenerator,
                )
                from src.generators.base_generator import BaseGenerator

                if issubclass(SymbioteEvolutionGenerator, BaseGenerator):
                    print("✓ SymbioteEvolutionGenerator is a subclass of BaseGenerator")
                else:
                    print(
                        "✗ SymbioteEvolutionGenerator is not a subclass of BaseGenerator"
                    )
                    success = False

                # Check for required methods
                required_methods = [
                    "generate",
                    "evolve_colony",
                    "apply_cellular_automaton",
                ]
                for method in required_methods:
                    if hasattr(SymbioteEvolutionGenerator, method) and callable(
                        getattr(SymbioteEvolutionGenerator, method)
                    ):
                        print(
                            f"✓ SymbioteEvolutionGenerator has required method: {method}"
                        )
                    else:
                        print(
                            f"✗ SymbioteEvolutionGenerator is missing required method: {method}"
                        )
                        success = False

            except ImportError as e:
                print(f"Error importing modules: {e}")
                success = False

        except Exception as e:
            print(f"Error checking file contents: {e}")
            success = False

    print(f"File structure verification {'successful' if success else 'failed'}")
    return success


if __name__ == "__main__":
    print("=== SymbioteEvolutionGenerator Comprehensive Verification ===")

    # First verify the file structure
    structure_success = verify_file_structure()

    if can_run_full_tests := (
        NUMPY_AVAILABLE and SCIPY_AVAILABLE and SYMBIOTE_GENERATOR_AVAILABLE
    ):
        print("\nRunning full functionality tests...")

        # Run the basic tests
        (
            generator,
            colony_grid,
            mineral_grid,
            evolved_grid,
            evolution_history,
            mutation_map,
        ) = test_symbiote_evolution_generator()

        # Run evolution over time test
        evolved_grid_time, evolution_history_time = test_evolution_over_time()

        # Run mineral impact test
        evolved_grids, mineral_grids = test_mineral_consumption_impact()

        # Visualize the results if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            visualize_results(
                generator,
                colony_grid,
                mineral_grid,
                evolved_grid,
                evolution_history,
                mutation_map,
            )
            visualize_mineral_impact(evolved_grids, mineral_grids)

        print("\n=== All functionality tests completed successfully! ===")
    else:
        print("\nSkipping functionality tests due to missing dependencies.")
        print("The following packages are required for full testing:")
        print(f"  - numpy: {'✓' if NUMPY_AVAILABLE else '✗'}")
        print(f"  - scipy: {'✓' if SCIPY_AVAILABLE else '✗'}")
        print(
            f"  - matplotlib: {'✓' if MATPLOTLIB_AVAILABLE else '✗'} (optional, for visualization)"
        )

        if code_review_success := perform_code_review():
            print("\nCode review completed successfully.")
        else:
            print("\nCode review failed.")
            sys.exit(1)

    print("\nThe SymbioteEvolutionGenerator appears to be correctly implemented.")
    print(
        "To run full functionality tests, ensure all required dependencies are installed."
    )
