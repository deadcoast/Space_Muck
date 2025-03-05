"""
Test script to verify import fixes across the Space Muck codebase.
This script attempts to import key modules and classes to ensure imports are working correctly.
"""

import sys
import os
import traceback
from pathlib import Path

def run_import_test():
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))

    success_count = 0
    failure_count = 0

    # Test basic imports
    print("Testing basic imports...")
    try:
        import pygame
        print("✓ pygame imported successfully")
        success_count += 1
    except Exception as e:
        print(f"✗ Failed to import pygame: {e}")
        failure_count += 1

    # Test config imports
    print("\nTesting config imports...")
    try:
        from src.config import (
            GRID_WIDTH, GRID_HEIGHT, CELL_SIZE, 
            COLOR_PLAYER, COLOR_ASTEROID,
            COMBAT_BASE_ATTACK_POWER, COMBAT_BASE_SHIELD_STRENGTH,
            PLAYER_START_CURRENCY
        )
        print("✓ Config constants imported successfully")
        print(f"  - GRID_WIDTH: {GRID_WIDTH}")
        print(f"  - COMBAT_BASE_ATTACK_POWER: {COMBAT_BASE_ATTACK_POWER}")
        print(f"  - PLAYER_START_CURRENCY: {PLAYER_START_CURRENCY}")
        success_count += 1
    except Exception as e:
        print(f"✗ Failed to import config constants: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        failure_count += 1

    # Test generator imports
    print("\nTesting generator imports...")
    try:
        from src.generators.base_generator import BaseGenerator
        print("✓ BaseGenerator imported successfully")
        success_count += 1
    except Exception as e:
        print(f"✗ Failed to import BaseGenerator: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        failure_count += 1

    try:
        from src.generators.procedural_generator import ProceduralGenerator
        print("✓ ProceduralGenerator imported successfully")
        success_count += 1
    except Exception as e:
        print(f"✗ Failed to import ProceduralGenerator: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        failure_count += 1

    try:
        from src.generators.asteroid_field import AsteroidField
        print("✓ AsteroidField imported successfully")
        success_count += 1
    except Exception as e:
        print(f"✗ Failed to import AsteroidField: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        failure_count += 1

    # Test entity imports
    print("\nTesting entity imports...")
    try:
        from src.entities.miner_entity import MinerEntity
        print("✓ MinerEntity imported successfully")
        success_count += 1
    except Exception as e:
        print(f"✗ Failed to import MinerEntity: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        failure_count += 1

    # Test utility imports
    print("\nTesting utility imports...")
    try:
        from src.utils.noise_generator import NoiseGenerator
        print("✓ NoiseGenerator imported successfully")
        success_count += 1
    except Exception as e:
        print(f"✗ Failed to import NoiseGenerator: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        failure_count += 1

    # Print summary
    print("\n" + "=" * 50)
    print(f"Import test complete! Success: {success_count}, Failures: {failure_count}")
    print("=" * 50)

    return success_count, failure_count

if __name__ == "__main__":
    try:
        success_count, failure_count = run_import_test()
        sys.exit(0 if failure_count == 0 else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print(traceback.format_exc())
        sys.exit(1)
