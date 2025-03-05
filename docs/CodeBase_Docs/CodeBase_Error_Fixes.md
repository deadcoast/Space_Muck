# CodeBase Error Fixes

This document logs common issues encountered during development and their solutions to avoid repeating the same mistakes.

## Table of Contents

1. [Import Issues](#import-issues)
2. [Linting Issues](#linting-issues)
3. [GPU Acceleration Issues](#gpu-acceleration-issues)
4. [Game Loop Implementation](#game-loop-implementation)
5. [Code Duplication Issues](#code-duplication-issues)
6. [Code Quality Issues](#code-quality-issues)

## Import Issues

### Import Structure Inconsistencies

**Issue**: Inconsistent import patterns across the codebase, including:
1. Mixing absolute imports with 'src' prefix and relative imports
2. Duplicate imports of the same module
3. Inconsistent import organization
4. Using wildcard imports (`from module import *`)

**Solution**: Standardize import structure following these guidelines:

1. **Organize imports** in a consistent order:
   ```python
   # Standard library imports
   import logging
   import math
   
   # Third-party library imports
   import numpy as np
   import pygame
   
   # Local application imports
   from entities.base_entity import BaseEntity
   ```

2. **Use relative imports** for local application modules:
   ```python
   # Incorrect - using absolute imports with 'src' prefix
   from src.utils.gpu_utils import apply_noise_generation_gpu
   
   # Correct - using relative imports
   from ..utils.gpu_utils import apply_noise_generation_gpu
   ```
   
   Note: The number of dots in relative imports depends on the directory depth. Use one dot for imports from the same directory, two dots for imports from the parent directory, and so on.

3. **Avoid duplicate imports** by checking if a module is already imported:
   ```python
   # Incorrect - duplicate import
   import scipy
   import scipy.signal
   ```
   
4. **Fix incorrect import paths** to ensure modules are found correctly:
   ```python
   # Incorrect - importing from wrong location
   from src.entities.base_generator import BaseGenerator
   
   # Correct - importing from actual location
   from src.generators.base_generator import BaseGenerator
   ```
   
   # Correct - single import
   import scipy
   from scipy import signal
   ```

### Test Import Path Issues

**Issue**: Test files were importing modules from incorrect paths, such as:
1. Importing from `src.world.asteroid_field` when the module is actually in `src.generators.asteroid_field`
2. Importing from `src.world.procedural_generation` when the module is actually in `src.generators.asteroid_generator`

**Solution**: Update import paths in test files to match the actual module structure:

1. **Update import statements** in test files:
   ```python
   # Incorrect - importing from non-existent path
   from src.world.asteroid_field import AsteroidField
   
   # Correct - importing from actual module location
   from src.generators.asteroid_field import AsteroidField
   ```

2. **Update function references** when module structure changes:
   ```python
   # Incorrect - using function from non-existent module
   field_data = generate_field(width, height)
   
   # Correct - using class method from actual module
   generator = AsteroidGenerator(width=width, height=height)
   asteroid_grid, metadata = generator.generate_field()
   ```

3. **Ensure proper path setup** in test files:
   ```python
   import os
   import sys
   
   # Add the src directory to the path
   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
   ```

### Optional Dependencies

**Issue**: Missing imports for optional dependencies like `cupy` and `torch` causing linting errors.

**Solution**: Use a robust pattern for handling optional dependencies with proper type checking:

```python
import importlib.util
from typing import TYPE_CHECKING

# Check for optional dependencies
CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# For type checking only - these imports are not executed at runtime
if TYPE_CHECKING:
    try:
        import cupy  # type: ignore
        import torch  # type: ignore
    except ImportError:
        pass

# Import optional dependencies at runtime
cp = None  # Define cp at module level
torch = None  # Define torch at module level

if CUPY_AVAILABLE:
    try:
        import cupy as cp
    except ImportError:
        pass

if TORCH_AVAILABLE:
    try:
        import torch
    except ImportError:
        pass
```

Then use these flags to conditionally execute code:

```python
if backend in ["cuda", "cupy"] and CUPY_AVAILABLE:
    # CUDA/CuPy specific code
elif backend == "mps" and TORCH_AVAILABLE:
    # MPS/PyTorch specific code
```

This approach:
1. Uses `importlib.util.find_spec()` to check if the dependency is installed
2. Defines module-level variables to prevent undefined variable errors
3. Uses `TYPE_CHECKING` with try/except for type hints without runtime errors
4. Adds extra error handling with try/except during imports
5. Avoids unused import warnings by not importing unused typing modules

### Wildcard Import Issues

**Issue**: Using wildcard imports (`from module import *`) causes several problems:
1. Makes it unclear which names are imported and where they come from
2. Can lead to name conflicts if multiple modules define the same name
3. Causes linting warnings (F403, F405 in flake8/ruff)
4. Makes it difficult to track which constants are actually used

**Solution**: Replace wildcard imports with explicit imports:

```python
# Bad - wildcard import
from config import *

# Good - explicit imports
from config import (
    # Version information
    VERSION,
    # Window settings
    WINDOW_WIDTH, WINDOW_HEIGHT,
    # Game states
    STATE_PLAY, STATE_SHOP,
    # Colors
    COLOR_BG, COLOR_TEXT
)
```

Benefits:
1. Makes code more readable by showing exactly which constants are used
2. Prevents name conflicts and undefined name errors
3. Allows linters to correctly identify unused imports
4. Makes it easier to refactor and maintain the codebase

## Linting Issues

### Unused Imports

**Issue**: Unused imports causing linting warnings.

**Solution**: Remove unused imports or use them if they're necessary. Common unused imports found:
- `argparse` when not using command-line arguments
- `from typing import Tuple, Optional` when not using these types
- Specific GPU utility imports when not using them

### Unused Variables

**Issue**: Variables assigned but never used.

**Solution**: Remove the variable assignment if the value isn't needed, or use the variable:

```python
# Before
start_time = time.time()
result = func(*args, **kwargs)  # result is never used
end_time = time.time()

# After
start_time = time.time()
func(*args, **kwargs)  # Don't assign if not using the result
end_time = time.time()
```

## Type Errors

### Tuple vs Integer Operations

**Issue**: Attempting to perform integer operations on tuple values:
```python
# Error: TypeError: unsupported operand type(s) for //: 'tuple' and 'int'
center_x, center_y = GAME_MAP_SIZE // 2, GAME_MAP_SIZE // 2
```

**Solution**: Unpack the tuple first, then perform operations on individual components:
```python
# Correct approach
grid_width, grid_height = GAME_MAP_SIZE
center_x, center_y = grid_width // 2, grid_height // 2
```

### Function Parameter Mismatches

**Issue**: Calling functions with incorrect parameter names or extra parameters:
```python
# Error: TypeError: add_value_clusters() got multiple values for argument 'num_clusters'
add_value_clusters(
    value_grid=value_grid,
    num_clusters=num_clusters,
    cluster_value_multiplier=1.0 + cluster_tendency,  # Wrong parameter name
    width=self.width,  # Extra parameter
    height=self.height,  # Extra parameter
    random_generator=self.random,  # Extra parameter
)
```

**Solution**: Check the function signature and use only the parameters defined in the function:
```python
# Correct approach
add_value_clusters(
    value_grid=value_grid,
    num_clusters=num_clusters,
    cluster_radius=int(min(self.width, self.height) * 0.05),
    value_multiplier=1.0 + cluster_tendency,  # Correct parameter name
)
```

## GPU Acceleration Issues

### Handling Backend Failures

**Issue**: Repetitive code for handling failed backends in benchmark functions.

**Solution**: Extract common patterns into helper functions:

```python
def _time_handler(results, backend, prefix):
    results["times"][backend].append(float("inf"))
    results["speedup"][backend].append(0.0)
    logging.info(f"{prefix}{backend}: Failed")

def _bandwidth_handler(results, backend, prefix):
    results["times"][backend].append(float("inf"))
    results["bandwidth"][backend].append(0.0)
    logging.info(f"{prefix}{backend}: Failed")
```

## Game Loop Implementation

### Missing Entry Point

**Issue**: The main.py file had a Game class defined but lacked a proper entry point and game loop implementation. This made it impossible to run the game directly.

**Solution**: Implement a proper main() function and entry point:

## Code Quality Issues

### Low Code Quality in Large Methods

**Issue**: Several methods in the Game class had low code quality scores (below 15%) due to excessive length, complex conditionals, and multiple responsibilities within a single method.

**Solution**: Refactor large methods by extracting smaller, focused methods with single responsibilities:

1. **Extract Methods for Event Handling**:
   ```python
   # Before: Large monolithic event handling method
   def handle_events(self) -> None:
       for event in pygame.event.get():
           # Many lines of code with different responsibilities
           # ...
   
   # After: Smaller, focused methods
   def handle_events(self) -> None:
       for event in pygame.event.get():
           # Handle UI component events first
           if self.notifier.handle_event(event):
               continue
           # ...
           elif event.type == pygame.MOUSEMOTION:
               self.handle_mouse_motion(event)
           elif event.type == pygame.MOUSEBUTTONDOWN:
               self.handle_mouse_button_down(event)
   
   def handle_mouse_motion(self, event) -> None:
       # Only handle mouse motion logic
       self.hover_position = event.pos
       self.cursor_over_ui = self.check_cursor_over_ui()
   ```

2. **Use Early Returns to Reduce Nesting**:
   ```python
   # Before: Deeply nested conditionals
   def check_for_discoveries(self) -> None:
       if self.frame_counter % 120 == 0:  # Every 2 seconds at 60 FPS
           for dy, dx in itertools.product(range(-3, 4), range(-3, 4)):
               nx, ny = self.player.x + dx, self.player.y + dy
               if (condition1 and condition2 and condition3):
                   # Do something
   
   # After: Early returns to reduce nesting
   def check_for_discoveries(self) -> None:
       if self.frame_counter % 120 != 0:  # Every 2 seconds at 60 FPS
           return
           
       for dy, dx in itertools.product(range(-3, 4), range(-3, 4)):
           nx, ny = self.player.x + dx, self.player.y + dy
           if (condition1 and condition2 and condition3):
               # Do something
   ```

3. **Use Meaningful Method Names**:
   ```python
   # Before: Generic or unclear method name
   def _extracted_from_regenerate_field_12(self):
       # Method implementation
   
   # After: Descriptive method name that explains purpose
   def setup_new_field(self):
       # Method implementation
   ```

### Benefits of Code Quality Improvements

1. **Improved Readability**: Smaller methods with clear purposes are easier to understand
2. **Better Maintainability**: Isolated functionality makes future changes safer
3. **Easier Testing**: Single-responsibility methods are easier to test
4. **Reduced Cognitive Load**: Developers can focus on one aspect at a time
5. **Better Code Reuse**: Extracted methods can be reused in other contexts

```python
def run_game_loop(game: Game) -> None:
    """Run the main game loop.
    
    Args:
        game: Initialized Game instance
    """
    running = True
    try:
        # Main game loop
        while running:
            # Start performance timing for this frame
            frame_start = log_performance_start("Frame")
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    game.handle_events()
            
            # Update game state
            game.update()
            
            # Render the game
            game.draw()
            
            # Maintain frame rate
            game.clock.tick(FPS)
            
            # End performance timing for this frame
            log_performance_end("Frame", frame_start)
            
    except Exception as e:
        # Log any unhandled exceptions
        log_exception("Unhandled exception in main game loop", e)
        raise
    finally:
        # Clean up resources
        pygame.quit()
        logging.info("Game terminated")


def main() -> None:
    """Initialize and run the game."""
    try:
        # Set up logging
        setup_logging(log_level=logging.INFO)
        logging.info(f"Starting Space Muck v{VERSION}")
        
        # Initialize the game
        game = Game()
        logging.info("Game initialized successfully")
        
        # Run the game loop
        run_game_loop(game)
        
    except Exception as e:
        # Log any unhandled exceptions during initialization
        log_exception("Failed to initialize or run game", e)
        raise


# Entry point
if __name__ == "__main__":
    main()
```

### Duplicate Imports

**Issue**: The main.py file had duplicate imports, with some using absolute paths with 'src' prefix and others using relative imports.

**Solution**: Organize imports following the project's best practices:

```python
# Standard library imports
import gc
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Union

# Third-party library imports
import numpy as np
import pygame

# Local application imports
from config import *
from generators.asteroid_field import AsteroidField
from generators.procedural_generator import create_field_with_multiple_algorithms
from algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
from entities.player import Player
from entities.miner_entity import MinerEntity
```

## Code Quality Issues

### Poorly Named Functions

**Issue**: Functions with generic extracted names (e.g., `_extracted_from_function_name_123`) that lack descriptive meaning.

**Solution**: Rename these functions with descriptive names that indicate their purpose and add proper docstrings:

```python
# Before
# TODO Rename this here and in `apply_noise_generation_gpu`
def _extracted_from_apply_noise_generation_gpu_49(seed, height, width):
    logging.warning("Noise generator not available. Using random noise.")
    rng = np.random.RandomState(seed)
    return rng.random((height, width))

# After
def _generate_fallback_noise(seed, height, width):
    """Generate random noise as a fallback when noise generators are unavailable.
    
    Args:
        seed: Random seed for reproducibility
        height: Height of the noise grid
        width: Width of the noise grid
        
    Returns:
        np.ndarray: Random noise grid
    """
    logging.warning("Noise generator not available. Using random noise.")
    rng = np.random.RandomState(seed)
    return rng.random((height, width))
```

Also update all references to the renamed function:

```python
# Before
return _extracted_from_apply_noise_generation_gpu_49(seed, height, width)

# After
return _generate_fallback_noise(seed, height, width)
```

### Monolithic Methods

**Issue**: Large methods that handle multiple responsibilities, making them difficult to understand, test, and maintain.

**Solution**: Refactor large methods into smaller, more focused methods with clear responsibilities:

```python
# Before: Single method handling multiple generator types
def _asteroid_handler(self):
    # Common parameters setup
    # ...
    
    if ASTEROID_GENERATOR_AVAILABLE:
        # Optimized generator logic
        # ...
    else:
        # Procedural generator logic
        # ...
    
    # Common post-processing
    # ...

# After: Split into multiple methods with clear responsibilities
def _asteroid_handler(self) -> None:
    """Generate asteroid field using available generators."""
    # Common parameters setup
    # ...
    
    try:
        if ASTEROID_GENERATOR_AVAILABLE:
            self._generate_asteroid_field_with_optimized_generator(common_params, seed)
        else:
            self._generate_asteroid_field_with_procedural_generator(common_params, seed)
        
        # Common post-processing
        # ...
    except Exception as e:
        # Error handling
        # ...

def _generate_asteroid_field_with_optimized_generator(self, common_params: dict, seed: int) -> None:
    """Generate asteroid field using the optimized AsteroidGenerator."""
    # Optimized generator logic
    # ...

def _generate_asteroid_field_with_procedural_generator(self, common_params: dict, seed: int) -> None:
    """Generate asteroid field using the ProceduralGenerator."""
    # Procedural generator logic
    # ...
```

This approach:
1. Improves readability by breaking down complex logic into smaller, focused methods
2. Enhances maintainability by isolating different generator implementations
3. Makes testing easier by allowing each method to be tested independently
4. Provides clear documentation through method names and docstrings
5. Centralizes error handling in the parent method

## Code Duplication Issues

### Duplicate Class Implementations

**Issue**: The main.py file contained two separate implementations of the Game class, each with slightly different functionality and initialization logic. This created confusion about which implementation was being used and led to potential bugs when features were added to one implementation but not the other.

**Solution**: Consolidate the duplicate implementations into a single, comprehensive Game class that incorporates all necessary functionality:

1. **Identify all unique features** from both implementations
2. **Create a unified implementation** that includes all necessary functionality
3. **Ensure consistent naming** of methods and attributes
4. **Update all references** to use the consolidated implementation

**Benefits**:
1. Eliminates confusion about which implementation is being used
2. Ensures all features are available in a single place
3. Simplifies maintenance and future development
4. Reduces the risk of bugs from inconsistent implementations

**Example of consolidated initialization**:
```python
def __init__(self) -> None:
    """Initialize the game environment, field, and entities."""
    # Initialize Pygame
    pygame.init()
    pygame.mixer.init()  # Initialize sound system
    pygame.font.init()  # Ensure fonts are initialized

    # Create display surface
    self.screen: pygame.Surface = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )
    pygame.display.set_caption(f"Space Muck v{VERSION} - Procedural Generation Edition")

    # Setup logging
    setup_logging(log_level=logging.INFO)
    logging.info(f"Starting Space Muck v{VERSION} ({BUILD_DATE})")
    
    # Initialize field with advanced procedural generation
    with LogContext("Field Initialization"):
        self.field: AsteroidField = create_field_with_multiple_algorithms(
            width=GRID_WIDTH,
            height=GRID_HEIGHT,
            seed=self.seed,
            rare_chance=RARE_THRESHOLD,
            rare_bonus=RARE_BONUS_MULTIPLIER,
        )
```
