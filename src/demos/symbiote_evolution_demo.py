#!/usr/bin/env python3
"""
Simple demonstration of the SymbioteEvolutionGenerator.
This script creates a basic symbiote evolution simulation and displays
the results in text format, requiring minimal dependencies.
"""

# Standard library imports
import os
import random
import sys
import time

# Local application imports
from pprint import pprint

# Third-party library imports


# Add the parent directory to the path to make imports work
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_dir)

# Try to import numpy, but provide fallback
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Define np as None to avoid unbound variable errors
    np = None
    print("numpy not available. Using fallback random generation.")

# Import the generator
try:
    from generators.symbiote_evolution_generator import SymbioteEvolutionGenerator

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

        sys.path.append("/Users/deadcoast/PycharmProjects/Space_Muck")
        from generators.symbiote_evolution_generator import (
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


def initialize_generator(seed=42):
    """Initialize the SymbioteEvolutionGenerator with parameters.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Configured SymbioteEvolutionGenerator instance
    """
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

    return generator


def generate_initial_state(generator):
    """Generate initial colony and mineral grids.

    Args:
        generator: Configured SymbioteEvolutionGenerator

    Returns:
        Tuple of (colony_grid, mineral_grid)
    """
    print("\nGenerating initial state...")
    start_time = time.time()

    if NUMPY_AVAILABLE:
        # The generator may return a tuple with metadata
        colony_result = generator.generate_initial_colonies()
        colony_grid = (
            colony_result[0] if isinstance(colony_result, tuple) else colony_result
        )
        mineral_grid = generator.generate_mineral_distribution()
    else:
        print("Using fallback random grids...")
        colony_grid = fallback_grid(generator.parameters["grid_size"])
        mineral_grid = fallback_grid(generator.parameters["grid_size"])

    print(f"Generation completed in {time.time() - start_time:.2f} seconds")
    return colony_grid, mineral_grid


def print_grid_info(colony_grid):
    """Print debug information about the colony grid.

    Args:
        colony_grid: The colony grid to analyze
    """
    print(f"Colony grid type: {type(colony_grid)}")

    if NUMPY_AVAILABLE and np is not None and isinstance(colony_grid, np.ndarray):
        print(f"Colony grid shape: {colony_grid.shape}")
    else:
        print(
            f"Colony grid dimensions: {len(colony_grid)}x{len(colony_grid[0]) if colony_grid else 0}"
        )


def create_empty_mutation_map(colony_grid):
    """Create an empty mutation map based on colony grid dimensions.

    Args:
        colony_grid: The colony grid to match dimensions with

    Returns:
        Empty mutation map with same dimensions as colony_grid
    """
    if NUMPY_AVAILABLE and np is not None and isinstance(colony_grid, np.ndarray):
        return np.zeros_like(colony_grid)
    else:
        return [
            [0.0 for _ in range(len(colony_grid[0]))] for _ in range(len(colony_grid))
        ]


def simulate_evolution_with_numpy(generator, colony_grid, mineral_grid):
    """Simulate evolution using numpy implementation.

    Args:
        generator: Configured SymbioteEvolutionGenerator
        colony_grid: Initial colony grid
        mineral_grid: Mineral distribution grid

    Returns:
        Tuple of (evolved_grid, evolution_history, mutation_map)
    """
    try:
        # Try with the expected signature
        evolved_grid, evolution_history = generator.simulate_evolution(
            colony_grid, mineral_grid, iterations=5
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
                mutation_map = create_empty_mutation_map(colony_grid)

    except Exception as e:
        print(f"Error during evolution simulation: {e}")
        # Handle both numpy arrays and regular lists
        if hasattr(colony_grid, "copy"):
            evolved_grid = colony_grid.copy()
            evolution_history = [colony_grid.copy()]
        else:
            # Deep copy for regular lists
            evolved_grid = [row[:] for row in colony_grid]
            evolution_history = [[row[:] for row in colony_grid]]

        mutation_map = create_empty_mutation_map(colony_grid)

    return evolved_grid, evolution_history, mutation_map


def simulate_evolution_fallback(generator):
    """Simulate evolution using fallback implementation when numpy is not available.

    Args:
        generator: Configured SymbioteEvolutionGenerator

    Returns:
        Tuple of (evolved_grid, evolution_history, mutation_map)
    """
    print("Using fallback evolution simulation...")
    grid_size = generator.parameters["grid_size"]

    evolved_grid = fallback_grid(grid_size)
    evolution_history = [fallback_grid(grid_size) for _ in range(5)]
    mutation_map = fallback_grid(grid_size)

    return evolved_grid, evolution_history, mutation_map


def display_evolution_statistics(evolution_history):
    """Display statistics from the evolution history.

    Args:
        evolution_history: List of evolution history entries
    """
    if not NUMPY_AVAILABLE:
        return

    print("\nEvolution Statistics:")
    try:
        if not isinstance(evolution_history, list):
            print(
                f"Evolution history type {type(evolution_history)} not compatible with statistics calculation"
            )
            return

        for i, entry in enumerate(evolution_history):
            display_history_entry(i, entry)

    except Exception as e:
        print(f"Error displaying evolution statistics: {e}")


def display_history_entry(index, history_entry):
    """Display a single evolution history entry.

    Args:
        index: Iteration index
        history_entry: Evolution history entry data
    """
    if not isinstance(history_entry, dict):
        print(
            f"Iteration {index}: Entry type {type(history_entry)} not compatible with statistics display"
        )
        return

    # Extract information from the history entry dictionary
    population = history_entry.get("population", "N/A")
    aggression = history_entry.get("aggression", "N/A")
    mutations = history_entry.get("mutations", [])
    mineral_consumption = history_entry.get("mineral_consumption", {})

    print(f"Iteration {index}:")
    print(f"  Population: {population}")

    if isinstance(aggression, (int, float)):
        print(f"  Aggression: {aggression:.4f}")
    else:
        print(f"  Aggression: {aggression}")

    print(f"  Mutations: {len(mutations)}")

    # Display mineral consumption if available
    if mineral_consumption:
        print("  Mineral consumption:")
        for mineral, amount in mineral_consumption.items():
            print(f"    {mineral}: {amount:.4f}")


def _format_colony_row(row):
    """Format a single row of colony data for visualization.

    Args:
        row: A row of colony data

    Returns:
        Formatted string representation of the row
    """
    if NUMPY_AVAILABLE:
        return "".join(["#" if float(cell) > 0.5 else "." for cell in row])
    else:
        return "".join(["#" if cell > 0.5 else "." for cell in row])


def _format_mineral_row(row):
    """Format a single row of mineral data for visualization.

    Args:
        row: A row of mineral data

    Returns:
        Formatted string representation of the row
    """
    if NUMPY_AVAILABLE:
        return "".join([str(min(9, int(float(cell) * 10))) for cell in row])
    else:
        return "".join([str(min(9, int(cell * 10))) for cell in row])


def _visualize_state_handler(title1, grid1, title2, grid2):
    """Visualize two grids with titles in text format.

    Args:
        title1: Title for the first grid
        grid1: First grid data (colony)
        title2: Title for the second grid
        grid2: Second grid data (mineral/mutation)
    """
    # Visualize first grid (colony)
    print(title1)
    for row in grid1:
        print(_format_colony_row(row))

    # Visualize second grid (mineral/mutation)
    print(title2)
    for row in grid2:
        print(_format_mineral_row(row))


def run_demo():
    """Run a simple demonstration of the SymbioteEvolutionGenerator."""
    print("=== SymbioteEvolutionGenerator Demo ===\n")

    # Initialize generator
    generator = initialize_generator(seed=42)

    # Generate initial state
    colony_grid, mineral_grid = generate_initial_state(generator)
    print_grid_info(colony_grid)

    # Visualize initial state
    _visualize_state_handler(
        "\nInitial Colony Grid (text representation):",
        colony_grid,
        "\nMineral Distribution Grid (text representation):",
        mineral_grid,
    )

    # Simulate evolution
    print("\nSimulating evolution...")
    start_time = time.time()

    if NUMPY_AVAILABLE:
        evolved_grid, evolution_history, mutation_map = simulate_evolution_with_numpy(
            generator, colony_grid, mineral_grid
        )
    else:
        evolved_grid, evolution_history, mutation_map = simulate_evolution_fallback(
            generator
        )

    print(f"Evolution simulation completed in {time.time() - start_time:.2f} seconds")

    if NUMPY_AVAILABLE:
        print(f"Evolution history type: {type(evolution_history)}")
        if isinstance(evolution_history, list):
            print(f"Evolution history length: {len(evolution_history)}")
        else:
            print("Evolution history length: N/A")

    # Visualize evolved state
    _visualize_state_handler(
        "\nEvolved Colony Grid (text representation):",
        evolved_grid,
        "\nMutation Intensity Map (text representation):",
        mutation_map,
    )

    # Display evolution statistics
    display_evolution_statistics(evolution_history)

    print("\nDemo completed successfully!")
    print(
        "For full functionality with visualizations, install numpy, scipy, and matplotlib."
    )


if __name__ == "__main__":
    run_demo()
