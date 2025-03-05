#!/usr/bin/env python3
"""
Test script to verify imports and fix linting issues.
"""

# Import the base_generator to check for issues
from src.generators.base_generator import BaseGenerator


# Create an instance and test the methods
def test_base_generator():
    """Test basic functionality of BaseGenerator."""
    generator = BaseGenerator(entity_id="test", seed=42, width=50, height=50)

    # Test the apply_cellular_automaton method
    import numpy as np

    grid = np.zeros((50, 50))
    grid[20:30, 20:30] = 1  # Create a square in the middle

    # Test with default parameters
    result = generator.apply_cellular_automaton(grid)
    print(f"Cellular automaton applied, sum: {np.sum(result)}")

    # Test with custom parameters
    result = generator.apply_cellular_automaton(
        grid, birth_set={2, 3}, survival_set={3, 4, 5}
    )
    print(f"Cellular automaton with custom rules applied, sum: {np.sum(result)}")

    print("All tests completed successfully!")


if __name__ == "__main__":
    test_base_generator()
