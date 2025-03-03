#!/usr/bin/env python3
"""
Simple demonstration of the SymbioteEvolutionGenerator.
This script creates a basic symbiote evolution simulation and displays
the results in text format, requiring minimal dependencies.
"""

import os
import sys
import time
import random
from pprint import pprint

# Add the parent directory to the path to make imports work
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_dir)

# Try to import numpy, but provide fallback
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("numpy not available. Using fallback random generation.")

# Import the generator
try:
    from src.generators.symbiote_evolution_generator import SymbioteEvolutionGenerator

    GENERATOR_AVAILABLE = True
    print("Successfully imported SymbioteEvolutionGenerator")
except ImportError as e:
    print(f"Error importing SymbioteEvolutionGenerator: {e}")
    print("Python path:")
    for p in sys.path:
        print(f"  {p}")
    print("\nTrying alternative import paths...")

    try:
        # Try direct import
        import sys

        sys.path.append("/Users/deadcoast/PycharmProjects/Space_Muck")
        from src.generators.symbiote_evolution_generator import (
            SymbioteEvolutionGenerator,
        )

        GENERATOR_AVAILABLE = True
        print("Successfully imported using absolute path")
    except ImportError as e2:
        GENERATOR_AVAILABLE = False
        print(f"All import attempts failed. Error: {e2}")
        print("Please check your installation and import paths.")
        sys.exit(1)


def create_text_visualization(grid, title="Grid"):
    """Create a text-based visualization of a grid."""
    if not NUMPY_AVAILABLE:
        print(f"\n{title} (text representation):")
        for row in grid:
            print("".join(["#" if cell > 0.5 else "." for cell in row]))
        return

    print(f"\n{title} (text representation):")
    symbols = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]

    for row in grid:
        line = ""
        for cell in row:
            # Map cell value (0-1) to a symbol
            # Handle numpy scalar values
            value = float(cell)
            idx = min(int(value * len(symbols)), len(symbols) - 1)
            line += symbols[idx]
        print(line)


def fallback_grid(size=10):
    """Create a fallback grid if numpy is not available."""
    return [[random.random() for _ in range(size)] for _ in range(size)]


def run_demo():
    """Run a simple demonstration of the SymbioteEvolutionGenerator."""
    print("=== SymbioteEvolutionGenerator Demo ===\n")

    # Create a generator with a fixed seed for reproducibility
    seed = 42
    print(f"Using seed: {seed}")

    generator = SymbioteEvolutionGenerator(seed=seed)

    # Set parameters
    generator.set_parameter("grid_size", 20)
    generator.set_parameter("initial_colony_count", 3)
    generator.set_parameter("mineral_richness", 0.7)
    generator.set_parameter("evolution_rate", 0.3)
    generator.set_parameter("mutation_rate", 0.1)

    print("\nGenerator parameters:")
    pprint(generator.parameters)

    # Generate initial state
    print("\nGenerating initial state...")
    start_time = time.time()

    if NUMPY_AVAILABLE:
        # The generator may return a tuple with metadata
        colony_result = generator.generate_initial_colonies()
        if isinstance(colony_result, tuple):
            colony_grid = colony_result[0]  # Extract just the grid
        else:
            colony_grid = colony_result

        mineral_grid = generator.generate_mineral_distribution()
    else:
        print("Using fallback random grids...")
        colony_grid = fallback_grid(generator.parameters["grid_size"])
        mineral_grid = fallback_grid(generator.parameters["grid_size"])

    print(f"Generation completed in {time.time() - start_time:.2f} seconds")

    # Debug information
    print(f"Colony grid type: {type(colony_grid)}")
    if hasattr(colony_grid, "shape"):
        print(f"Colony grid shape: {colony_grid.shape}")

    # Visualize initial state
    print("\nInitial Colony Grid (text representation):")
    if NUMPY_AVAILABLE:
        for row in colony_grid:
            print("".join(["#" if float(cell) > 0.5 else "." for cell in row]))
    else:
        for row in colony_grid:
            print("".join(["#" if cell > 0.5 else "." for cell in row]))

    print("\nMineral Distribution Grid (text representation):")
    if NUMPY_AVAILABLE:
        for row in mineral_grid:
            print("".join([str(min(9, int(float(cell) * 10))) for cell in row]))
    else:
        for row in mineral_grid:
            print("".join([str(min(9, int(cell * 10))) for cell in row]))

    # Simulate evolution
    print("\nSimulating evolution...")
    start_time = time.time()

    if NUMPY_AVAILABLE:
        try:
            # Try with the expected signature
            evolved_grid, evolution_history = generator.simulate_evolution(
                colony_grid, mineral_grid, iterations=5
            )

            print(
                f"Evolution simulation completed in {time.time() - start_time:.2f} seconds"
            )
            print(f"Evolution history type: {type(evolution_history)}")
            print(
                f"Evolution history length: {len(evolution_history) if isinstance(evolution_history, list) else 'N/A'}"
            )

            # Generate mutation map if available
            try:
                mutation_map = generator.generate_mutation_map(
                    colony_grid, evolution_history
                )
            except TypeError:
                try:
                    # Alternative signature
                    mutation_map = generator.generate_mutation_map(evolution_history)
                except Exception as e:
                    print(f"Could not generate mutation map: {e}")
                    mutation_map = np.zeros_like(colony_grid)
        except Exception as e:
            print(f"Error during evolution simulation: {e}")
            evolved_grid = colony_grid.copy()
            evolution_history = [colony_grid.copy()]
            mutation_map = np.zeros_like(colony_grid)
    else:
        print("Using fallback evolution simulation...")
        evolved_grid = fallback_grid(generator.parameters["grid_size"])
        evolution_history = [
            fallback_grid(generator.parameters["grid_size"]) for _ in range(5)
        ]
        mutation_map = fallback_grid(generator.parameters["grid_size"])
        print(
            f"Evolution simulation completed in {time.time() - start_time:.2f} seconds"
        )

    # Visualize final state
    print("\nEvolved Colony Grid (text representation):")
    if NUMPY_AVAILABLE:
        for row in evolved_grid:
            print("".join(["#" if float(cell) > 0.5 else "." for cell in row]))
    else:
        for row in evolved_grid:
            print("".join(["#" if cell > 0.5 else "." for cell in row]))

    print("\nMutation Intensity Map (text representation):")
    if NUMPY_AVAILABLE:
        for row in mutation_map:
            print("".join([str(min(9, int(float(cell) * 10))) for cell in row]))
    else:
        for row in mutation_map:
            print("".join([str(min(9, int(cell * 10))) for cell in row]))

    # Display evolution statistics
    if NUMPY_AVAILABLE:
        print("\nEvolution Statistics:")
        try:
            if isinstance(evolution_history, list):
                for i, history_entry in enumerate(evolution_history):
                    if isinstance(history_entry, dict):
                        # Extract information from the history entry dictionary
                        population = history_entry.get("population", "N/A")
                        aggression = history_entry.get("aggression", "N/A")
                        mutations = history_entry.get("mutations", [])
                        mineral_consumption = history_entry.get(
                            "mineral_consumption", {}
                        )

                        print(f"Iteration {i}:")
                        print(f"  Population: {population}")
                        print(
                            f"  Aggression: {aggression:.4f}"
                            if isinstance(aggression, (int, float))
                            else f"  Aggression: {aggression}"
                        )
                        print(f"  Mutations: {len(mutations)}")

                        # Display mineral consumption if available
                        if mineral_consumption:
                            print(f"  Mineral consumption:")
                            for mineral, amount in mineral_consumption.items():
                                print(f"    {mineral}: {amount:.4f}")
                    else:
                        print(
                            f"Iteration {i}: Entry type {type(history_entry)} not compatible with statistics display"
                        )
            else:
                print(
                    f"Evolution history type {type(evolution_history)} not compatible with statistics calculation"
                )
        except Exception as e:
            print(f"Error displaying evolution statistics: {e}")

    print("\nDemo completed successfully!")
    print(
        "For full functionality with visualizations, install numpy, scipy, and matplotlib."
    )


if __name__ == "__main__":
    run_demo()
