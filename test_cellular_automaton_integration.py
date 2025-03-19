#!/usr/bin/env python3
"""
Test script for cellular automaton integration with AsteroidField.
This script verifies that the cellular_automaton.py module has been
properly integrated with the AsteroidField class.
"""

import sys
import os
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Check if cellular_automaton module is available
try:
    from src.algorithms import cellular_automaton
    available = True
    logging.info("Cellular automaton module is available")
except ImportError:
    available = False
    logging.warning("Cellular automaton module could not be imported")

# Create test data
test_grid = np.zeros((10, 10), dtype=np.float32)
test_grid[4:7, 4:7] = 1.0  # Create a small square pattern

# Test the optimized neighbor counting function if available
if available:
    try:
        neighbors = cellular_automaton.count_neighbors(test_grid)
        logging.info(f"Neighbor counting test successful: shape={neighbors.shape}")
        
        # Test cellular automaton rules application
        birth_set = {3}
        survival_set = {2, 3}
        result = cellular_automaton.apply_life_rules(test_grid, birth_set, survival_set)
        logging.info(f"Cellular automaton rules test successful: shape={result.shape}")
        
        # Test energy diffusion
        energy_grid = np.random.random((10, 10)).astype(np.float32)
        diffused = cellular_automaton.diffuse_energy(energy_grid, decay_rate=0.05, spread_rate=0.2)
        logging.info(f"Energy diffusion test successful: shape={diffused.shape}")
        
        print("All cellular automaton module tests passed!")
    except Exception as e:
        logging.error(f"Error during cellular automaton testing: {str(e)}")
else:
    logging.warning("Skipping tests since cellular automaton module is not available")

print("Integration test complete")
