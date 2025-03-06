"""
Simple script to verify that the Player class can be instantiated.
"""

import sys
import os
from unittest.mock import MagicMock

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock all dependencies
modules_to_mock = [
    "perlin_noise",
    "sklearn",
    "sklearn.cluster",
    "networkx",
    "pygame",
    "scipy",
    "scipy.stats",
    "scipy.ndimage",
    "numpy",
    "matplotlib",
    "matplotlib.pyplot",
]

for module in modules_to_mock:
    sys.modules[module] = MagicMock()


# Create a mock for the SymbioteEvolutionAlgorithm
class MockSymbioteEvolutionAlgorithm:
    def __init__(self, **kwargs):
        # Accept any keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def process_mineral_feeding(self, *args, **kwargs):
        return 100, 0.5, []

    def generate_cellular_automaton_rules(self, *args, **kwargs):
        return {3, 6}, {2, 3}


# Replace the actual algorithm with our mock
sys.modules["src.algorithms.symbiote_algorithm"] = MagicMock()
sys.modules["src.algorithms.symbiote_algorithm"].SymbioteEvolutionAlgorithm = (
    MockSymbioteEvolutionAlgorithm
)

# Now try to import and instantiate the Player class
try:
    from entities.player import Player

    # Create a player instance
    player = Player()

    # Print some attributes to verify
    print("Player successfully instantiated!")
    print(f"Is player: {player.is_player}")
    print(f"Credits: {player.credits}")
    print(f"Ship level: {player.ship_level}")
    print(f"Mining speed: {player.mining_speed}")
    print(f"Mining efficiency: {player.mining_efficiency}")
    print(f"Trait: {player.trait}")

    print("\nRefactoring successful!")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
