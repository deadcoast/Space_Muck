---
CODEBASE MAPPING REFERENCE
---

## Table of Contents

1. [mapping_template_example.py](#code_base_template)
2. [base_entity.py](#entity_system)
3. [miner_entity.py](#entity_system)
4. [player.py](#entity_system)
5. [fleet.py](#entity_system)
6. [base_generator.py](#entity_system)
7. [procedural_generator.py](#generator_system)
8. [asteroid_generator.py](#generator_system)
9. [symbiote_evolution_generator.py](#generator_system)
10. [verify_base_entity.py](#testing)
11. [verify_fleet.py](#testing)
12. [verify_procedural_generator.py](#testing)
13. [verify_asteroid_generator_simple.py](#testing)
14. [verify_asteroid_generator.py](#testing)
15. [verify_base_generator_optimizations.py](#testing)
16. [benchmark_base_generator.py](#testing)
17. [benchmark_parallel_processing.py](#testing)
18. [verify_symbiote_evolution_generator.py](#testing)
19. [symbiote_evolution_demo.py](#demo)
20. [asteroid_field.py](#world_system)
21. [test_base_entity.py](#unit_testing)
22. [test_base_generator.py](#unit_testing)
23. [test_miner_entity.py](#unit_testing)
24. [test_player.py](#unit_testing)
25. [test_fleet.py](#unit_testing)
26. [test_procedural_generator.py](#unit_testing)
27. [test_asteroid_generator.py](#unit_testing)
28. [test_symbiote_evolution_generator.py](#unit_testing)
29. [dependency_injection.py](#utils)
30. [noise_generator.py](#utils)
31. [dependency_config.py](#utils)
32. [pattern_generator.py](#utils)
33. [value_generator.py](#utils)
34. [cellular_automaton_utils.py](#utils)
35. [import_standards.py](#utils)
36. [visualization.py](#utils)
37. [test_pattern_generator.py](#unit_testing)
38. [test_value_generator.py](#unit_testing)
39. [test_visualization.py](#testing)
40. [dependency_config_example.py](#examples)

---

## 1. [mapping_template_example.py](#mapping_template_example)

- `path/to/file.py`
  - **Purpose**: What does the code base asset do?
  - **File Imports**: List any Python Codebase Dependencies in the Format below.
    - from src.entities.base_entity import BaseEntity
  - **File Dependencies**: List any Python Library Dependencies in the format below.
    - from sklearn.cluster import KMeans
    - import pygame
  - **Required Components**: List any specific components or systems that the code base asset is designed to work with.

## 2. [base_entity.py](#entity_system)

- `src/entities/base_entity.py`
  - **Purpose**: Root class for all entities in the game, providing common functionality
  - **File Imports**: 
    - from typing import Tuple, Optional, Dict, Any
  - **File Dependencies**: 
    - import logging
    - import uuid
  - **Required Components**: 
    - Entity identification (ID, type, tags)
    - Position management
    - Lifecycle management (activate/deactivate)
    - Serialization/deserialization

## 3. [miner_entity.py](#entity_system)

- `src/entities/miner_entity.py`
  - **Purpose**: Base class for entities that can mine resources and evolve using cellular automaton patterns
  - **File Imports**: 
    - from src.config import (COLOR_RACE_1, COLOR_RACE_2, COLOR_RACE_3, etc.)
    - from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
    - from src.utils.logging_setup import log_exception
    - from src.entities.base_entity import BaseEntity
  - **File Dependencies**: 
    - import numpy as np
    - import networkx as nx
    - from perlin_noise import PerlinNoise (optional with fallback)
    - import scipy.stats as stats (for statistical analysis and distributions)
    - from sklearn.cluster import KMeans
    - import pygame
  - **Required Components**: 
    - AsteroidField
    - BaseEntity
    - SymbioteEvolutionAlgorithm
    - config constants
  - **Key Features**:
    - Territory analysis using scipy.stats (skewness, kurtosis)
    - Dedicated mutate() method using scipy.stats distributions
    - Behavior model using probability distributions (beta, normal, logistic)
    - Population statistics with confidence intervals
    - Entity placement using truncated normal distributions

## 4. [player.py](#entity_system)

- `src/entities/player.py`
  - **Purpose**: Player character class with specific player functionality
  - **File Imports**: 
    - from src.entities.miner_entity import MinerEntity
    - from src.config import COLOR_PLAYER
  - **File Dependencies**: 
    - Inherits from MinerEntity (which inherits from BaseEntity)
    - Uses config constants for player visualization
  - **Required Components**: 
    - AsteroidField
    - Inventory system
    - Quest system (placeholder for future implementation)

## 5. [fleet.py](#entity_system)

- `src/entities/fleet.py`
  - **Purpose**: Fleet management for player and AI entities
  - **File Imports**: 
    - from src.entities.base_entity import BaseEntity
  - **File Dependencies**: 
    - import uuid
    - import logging
  - **Required Components**: 
    - BaseEntity
    - Ship management system
    - Movement and pathfinding system

## 6. [base_generator.py](#entity_system)

- `src/entities/base_generator.py`
  - **Purpose**: Base class for all procedural generator entities in the game
  - **File Imports**: 
    - from src.entities.base_entity import BaseEntity
    - from src.utils import cellular_automaton_utils
    - from src.utils import value_generator
    - from src.entities.noise_generator import NoiseGenerator
  - **File Dependencies**: 
    - import random
    - import logging
    - import numpy as np
    - from typing import Any, Dict, List, Optional, Set, Tuple
    - from perlin_noise import PerlinNoise (optional with fallback)
  - **Required Components**: 
    - BaseEntity class
    - Noise generation
    - Cellular automaton
  - **Key Features**:
    - Modular integration with utility modules
    - Robust fallback mechanisms for missing dependencies
    - Intelligent caching system for performance optimization
    - Parameter validation and error handling
    - Comprehensive logging

## 7. [procedural_generator.py](#generator_system)

- `src/generators/procedural_generator.py`
  - **Purpose**: Procedural generator class for creating asteroid fields, inherits from BaseGenerator
  - **File Imports**: 
    - from src.entities.base_generator import BaseGenerator
    - from src.world.asteroid_field import AsteroidField
    - from src.utils.logging_setup import log_performance_start, log_performance_end, log_exception
  - **File Dependencies**: 
    - import numpy as np
    - import scipy.ndimage as ndimage
    - import scipy.stats as stats
    - from perlin_noise import PerlinNoise
    - from skimage import measure
    - from typing import Dict, List, Tuple, Optional, Any
  - **Required Components**: 
    - BaseGenerator class
    - AsteroidField class
  - **Key Methods**:
    - `generate()`: Creates complete asteroid field with grid, rare_grid, and energy_grid
    - `generate_asteroid_field()`: Creates the main asteroid grid
    - `generate_multi_layer_asteroid_field()`: Generates asteroid field with multiple noise layers
    - `generate_rare_minerals()`: Distributes rare minerals in the asteroid field
    - `generate_tiered_mineral_distribution()`: Creates tiered distribution of rare minerals
    - `apply_edge_degradation()`: Applies edge degradation for more natural appearance

## 8. [verify_base_entity.py](#testing)

- `src/tests/verify_base_entity.py`
  - **Purpose**: Verification script to test BaseEntity class functionality
  - **File Imports**: 
    - from src.entities.base_entity import BaseEntity
  - **File Dependencies**: 
    - None
  - **Required Components**: 
    - BaseEntity class

## 9. [verify_fleet.py](#testing)

- `src/tests/verify_fleet.py`
  - **Purpose**: Verification script to test Fleet class functionality
  - **File Imports**: 
    - from src.entities.fleet import Fleet
    - from src.entities.base_entity import BaseEntity
  - **File Dependencies**: 
    - None
  - **Required Components**: 
    - Fleet class
    - BaseEntity class

## 10. [verify_procedural_generator.py](#testing)

- `src/tests/verify_procedural_generator.py`
  - **Purpose**: Verification script for the ProceduralGenerator class
  - **File Imports**: 
    - from src.generators.procedural_generator import ProceduralGenerator, create_field_with_multiple_algorithms
    - from src.world.asteroid_field import AsteroidField
  - **File Dependencies**: 
    - import numpy as np
    - import matplotlib.pyplot as plt
  - **Required Components**: 
    - ProceduralGenerator class
    - AsteroidField class

## 11. [asteroid_generator.py](#generator_system)

- `src/generators/asteroid_generator.py`
  - **Purpose**: Procedural generator class for creating asteroid fields with complex patterns, inherits from BaseGenerator
  - **File Imports**: 
    - from src.entities.base_generator import BaseGenerator
  - **File Dependencies**: 
    - import numpy as np
    - import scipy.ndimage as ndimage
    - from perlin_noise import PerlinNoise
    - import random
  - **Required Components**: 
    - BaseGenerator class
    - Pattern generation algorithms
  - **Key Methods**:
    - `generate_field()`: Generates a complete asteroid field with optional pattern weighting
    - `generate_values()`: Generates resource values for each asteroid in the field
    - `generate_rare_resources()`: Generates rare resource distribution across the asteroid field
    - `_spiral_pattern()`: Creates spiral patterns in the asteroid field
    - `_ring_pattern()`: Creates ring patterns in the asteroid field
    - `_gradient_pattern()`: Creates gradient patterns in the asteroid field
    - `_void_pattern()`: Creates void areas in the asteroid field

## 12. [verify_asteroid_generator.py](#testing)

- `src/tests/verify_asteroid_generator.py`
  - **Purpose**: Verification script for the AsteroidGenerator class
  - **File Imports**: 
    - from src.generators.asteroid_generator import AsteroidGenerator
  - **File Dependencies**: 
    - import numpy as np
    - import matplotlib.pyplot as plt
  - **Required Components**: 
    - AsteroidGenerator class

## 13. [symbiote_evolution_generator.py](#generator_system)

- `src/generators/symbiote_evolution_generator.py`
  - **Purpose**: Specialized generator for symbiote evolution with colony growth and mutation patterns
  - **File Imports**: 
    - from src.entities.base_generator import BaseGenerator
    - from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
    - from src.utils.logging_setup import log_performance_start, log_performance_end, log_exception, LogContext
  - **File Dependencies**: 
    - import numpy as np
    - from perlin_noise import PerlinNoise
    - import scipy.ndimage as ndimage
    - import random
    - import math
  - **Required Components**: 
    - BaseGenerator class
    - SymbioteEvolutionAlgorithm class
    - Cellular automaton implementation
    - Mutation and evolution mechanics
  - **Key Methods**:
    - `generate_initial_colonies()`: Creates initial symbiote colonies on a grid
    - `generate_colony_distribution()`: Generates the distribution of colonies based on noise patterns
    - `generate_mineral_distribution()`: Creates a mineral distribution map
    - `generate_mineral_concentration_map()`: Generates detailed mineral concentrations
    - `simulate_evolution()`: Simulates symbiote evolution over time
    - `generate_mutation_map()`: Creates a map showing mutation hotspots

## 14. [verify_base_generator_optimizations.py](#testing)

- `src/tests/verify_base_generator_optimizations.py`
  - **Purpose**: Verification script for testing the optimized methods in BaseGenerator
  - **File Imports**: 
    - from entities.base_generator import BaseGenerator
    - from utils.visualization import GeneratorVisualizer
  - **File Dependencies**: 
    - import numpy as np
    - import matplotlib.pyplot as plt (optional)
    - import time
  - **Required Components**: 
    - BaseGenerator class
    - GeneratorVisualizer class
  - **Key Features**:
    - Tests noise layer generation with caching
    - Tests cellular automaton implementation with performance metrics
    - Tests clustering implementation with performance metrics
    - Tests thresholding implementation with performance metrics
    - Visualizes results when matplotlib is available

## 15. [benchmark_base_generator.py](#testing)

- `src/tests/benchmark_base_generator.py`
  - **Purpose**: Benchmark script for measuring BaseGenerator performance
  - **File Imports**: 
    - from entities.base_generator import BaseGenerator
    - from utils.noise_generator import NoiseGenerator
  - **File Dependencies**: 
    - import numpy as np
    - import time
    - import logging
  - **Required Components**: 
    - BaseGenerator class
  - **Key Features**:
    - Measures execution time of cellular automaton operations
    - Measures execution time of clustering operations
    - Compares performance across different grid sizes
    - Provides formatted benchmark results

## 18. [verify_symbiote_evolution_generator.py](#testing)

- `src/tests/verify_symbiote_evolution_generator.py`
  - **Purpose**: Comprehensive verification script for the SymbioteEvolutionGenerator class
  - **File Imports**: 
    - from src.generators.symbiote_evolution_generator import SymbioteEvolutionGenerator
    - from src.algorithms.symbiote_algorithm import SymbioteEvolutionAlgorithm
  - **File Dependencies**: 
    - import numpy as np
    - import matplotlib.pyplot as plt
    - import scipy
  - **Required Components**: 
    - SymbioteEvolutionGenerator class
    - SymbioteEvolutionAlgorithm class
    - Visualization capabilities

## 16. [asteroid_field.py](#world_system)

- `src/world/asteroid_field.py`
  - **Purpose**: Represents the asteroid field in the game, with grid-based asteroid distribution and properties
  - **File Imports**: 
    - from src.generators.procedural_generator import ProceduralGenerator
    - from src.generators.symbiote_evolution_generator import SymbioteEvolutionGenerator
    - from src.utils.logging_setup import log_performance_start, log_performance_end, log_exception
  - **File Dependencies**: 
    - import numpy as np
    - import random
    - import logging
    - from typing import Dict, List, Tuple, Optional, Any
  - **Required Components**: 
    - ProceduralGenerator class
    - SymbioteEvolutionGenerator class
  - **Key Methods**:
    - `generate_field()`: Generates the asteroid field using ProceduralGenerator
    - `generate_symbiote_evolution()`: Generates symbiote evolution using SymbioteEvolutionGenerator

## 15. [symbiote_evolution_demo.py](#demo)

- `src/demos/symbiote_evolution_demo.py`
  - **Purpose**: Demonstration script for the SymbioteEvolutionGenerator with text-based visualization
  - **File Imports**: 
    - from src.generators.symbiote_evolution_generator import SymbioteEvolutionGenerator
  - **File Dependencies**: 
    - import numpy as np
    - import time
    - import random
    - import pprint
  - **Required Components**: 
    - SymbioteEvolutionGenerator class
    - Fallback mechanisms for missing dependencies
    - Text-based visualization
## 14. [verify_asteroid_generator.py](#testing)

- `src/tests/verify_asteroid_generator.py`
  - **Purpose**: Comprehensive verification script for AsteroidGenerator performance and optimization
  - **File Imports**: 
    - from src.generators.asteroid_generator import AsteroidGenerator
    - from src.entities.base_generator import BaseGenerator
  - **File Dependencies**: 
    - import numpy as np
    - import time
    - import matplotlib.pyplot as plt (optional)
    - import logging
    - from typing import Dict, List, Tuple, Optional, Any, Callable
  - **Required Components**: 
    - AsteroidGenerator class
    - BaseGenerator class with caching mechanism
    - Performance measurement utilities
    - Visualization capabilities
  - **Key Features**:
    - Performance testing for generator methods
    - Caching verification
    - Visualization of generator outputs
    - Comparison of cached vs. non-cached operations
    - Flexible seed-based testing

## 15. [verify_base_generator_optimizations.py](#testing)

- `src/tests/verify_base_generator_optimizations.py`
  - **Purpose**: Standalone verification script for BaseGenerator optimization techniques
  - **File Imports**: 
    - None (implements a standalone BaseGenerator)
  - **File Dependencies**: 
    - import numpy as np
    - import time
    - import matplotlib.pyplot as plt (optional)
    - import logging
    - from typing import Dict, List, Tuple, Optional, Any, Callable
    - from functools import lru_cache, wraps
  - **Required Components**: 
    - Standalone BaseGenerator implementation
    - Caching mechanism
    - Performance measurement utilities
    - Visualization capabilities
  - **Key Features**:
    - Benchmarking of caching mechanisms
    - Comparison of different optimization techniques
    - Visualization of performance improvements
    - Flexible seed-based testing
    - Isolated testing environment

## 18. [verify_symbiote_evolution_generator.py](#testing)

- `src/tests/verify_symbiote_evolution_generator.py`
  - **Purpose**: Comprehensive verification script for SymbioteEvolutionGenerator performance and optimization
  - **File Imports**: 
    - from src.generators.symbiote_evolution_generator import SymbioteEvolutionGenerator
    - from src.entities.base_generator import BaseGenerator
  - **File Dependencies**: 
    - import numpy as np
    - import time
    - import matplotlib.pyplot as plt (optional)
    - import logging
    - from typing import Dict, List, Tuple, Optional, Any, Callable
  - **Required Components**: 
    - SymbioteEvolutionGenerator class
    - BaseGenerator class with caching mechanism
    - Performance measurement utilities
    - Visualization capabilities
  - **Key Features**:
    - Performance testing for symbiote evolution methods
    - Caching verification
    - Visualization of evolution patterns
    - Comparison of cached vs. non-cached operations
    - Flexible seed-based testing

## 17. [test_base_entity.py](#unit_testing)

- `src/tests/test_base_entity.py`
  - **Purpose**: Comprehensive unit tests for the BaseEntity class
  - **File Imports**: 
    - from entities.base_entity import BaseEntity
  - **File Dependencies**: 
    - import unittest
    - from unittest.mock import patch, MagicMock
  - **Required Components**: 
    - BaseEntity class
  - **Test Coverage**:
    - Initialization with default and custom values
    - Activation and deactivation
    - Position management
    - Tag management
    - Serialization and deserialization
    - Edge cases and error handling

## 18. [test_base_generator.py](#unit_testing)

- `src/tests/test_base_generator.py`
  - **Purpose**: Comprehensive unit tests for the BaseGenerator class
  - **File Imports**: 
    - from src.entities.base_generator import BaseGenerator
    - from src.entities.noise_generator import NoiseGenerator
  - **File Dependencies**: 
    - import unittest
    - import numpy as np
    - from unittest.mock import MagicMock, patch
  - **Required Components**: 
    - BaseGenerator class
    - NoiseGenerator class
  - **Key Features**:
    - Tests for apply_cellular_automaton with and without utility modules
    - Tests for create_clusters with and without utility modules
    - Parameter validation tests
    - Caching tests
    - Mock testing for utility module integration
  - **Required Components**: 
    - BaseGenerator class
    - NoiseGenerator class
  - **Test Coverage**:
    - Initialization with default and custom values
    - Noise generation methods
    - Cellular automaton methods
    - Clustering methods
    - Inheritance from BaseEntity
    - Serialization and deserialization

## 19. [test_miner_entity.py](#unit_testing)

- `src/tests/test_miner_entity.py`
  - **Purpose**: Comprehensive unit tests for the MinerEntity class
  - **File Imports**: 
    - from entities.miner_entity import MinerEntity
  - **File Dependencies**: 
    - import unittest
    - import numpy as np
    - from unittest.mock import patch, MagicMock
  - **Required Components**: 
    - MinerEntity class
    - BaseEntity class
  - **Test Coverage**:
    - Initialization with default and custom values
    - Genome generation and manipulation
    - Population methods
    - Resource gathering methods
    - Trait-specific behaviors
    - Inheritance from BaseEntity
    - Serialization and deserialization

## 20. [test_player.py](#unit_testing)

- `src/tests/test_player.py`
  - **Purpose**: Comprehensive unit tests for the Player class
  - **File Imports**: 
    - from entities.player import Player
  - **File Dependencies**: 
    - import unittest
    - from unittest.mock import patch, MagicMock
  - **Required Components**: 
    - Player class
    - MinerEntity class
  - **Test Coverage**:
    - Initialization with default and custom values
    - Player-specific attributes (credits, inventory)
    - Movement methods
    - Mining methods
    - Inheritance from MinerEntity
    - Serialization and deserialization

## 21. [test_fleet.py](#unit_testing)

- `src/tests/test_fleet.py`
  - **Purpose**: Comprehensive unit tests for the Fleet class
  - **File Imports**: 
    - from entities.fleet import Fleet
  - **File Dependencies**: 
    - import unittest
    - from unittest.mock import patch, MagicMock
  - **Required Components**: 
    - Fleet class
    - BaseEntity class
  - **Test Coverage**:
    - Initialization with default and custom values
    - Ship management methods
    - Movement and destination setting
    - Fuel consumption
    - Formation management
    - Inheritance from BaseEntity
    - Serialization and deserialization

## 22. [test_procedural_generator.py](#unit_testing)

- `src/tests/test_procedural_generator.py`
  - **Purpose**: Comprehensive unit tests for the ProceduralGenerator class
  - **File Imports**: 
    - from generators.procedural_generator import ProceduralGenerator, create_field_with_multiple_algorithms
  - **File Dependencies**: 
    - import unittest
    - import numpy as np
    - from unittest.mock import patch, MagicMock
  - **Required Components**: 
    - ProceduralGenerator class
    - BaseGenerator class
    - AsteroidField class
  - **Test Coverage**:
    - Initialization with default and custom values
    - Asteroid field generation methods
    - Rare mineral generation
    - Energy source generation
    - Multi-layer field generation
    - Tiered mineral distribution
    - Inheritance from BaseGenerator
    - Integration with AsteroidField

## 23. [test_asteroid_generator.py](#unit_testing)

- `src/tests/test_asteroid_generator.py`
  - **Purpose**: Comprehensive unit tests for the AsteroidGenerator class
  - **File Imports**: 
    - from generators.asteroid_generator import AsteroidGenerator
  - **File Dependencies**: 
    - import unittest
    - import numpy as np
    - from unittest.mock import patch, MagicMock
  - **Required Components**: 
    - AsteroidGenerator class
    - BaseGenerator class
    - AsteroidField class
  - **Test Coverage**:
    - Initialization with default and custom values
    - Asteroid belt generation
    - Asteroid cluster generation
    - Field type selection
    - Mineral distribution methods
    - Energy field generation
    - Inheritance from BaseGenerator
    - Integration with AsteroidField

## 24. [test_symbiote_evolution_generator.py](#unit_testing)

- `src/tests/test_symbiote_evolution_generator.py`
  - **Purpose**: Comprehensive unit tests for the SymbioteEvolutionGenerator class
  - **File Imports**: 
    - from generators.symbiote_evolution_generator import SymbioteEvolutionGenerator
  - **File Dependencies**: 
    - import unittest
    - import numpy as np
    - from unittest.mock import patch, MagicMock
  - **Required Components**: 
    - SymbioteEvolutionGenerator class
    - BaseGenerator class
  - **Test Coverage**:
    - Initialization with default and custom values
    - Initial population generation
    - Genome generation and manipulation
    - Population evolution methods
    - Parent selection algorithms
    - Crossover and mutation methods
    - Fitness calculation
    - Symbiote species generation
    - Inheritance from BaseGenerator
    - Serialization and deserialization

## 25. [dependency_injection.py](#utils)

- `src/utils/dependency_injection.py`
  - **Purpose**: Provides a dependency injection framework for the codebase
  - **File Imports**: 
    - None
  - **File Dependencies**: 
    - import inspect
    - from typing import Dict, Type, Callable, Any, Optional, Set, List
    - import functools
  - **Required Components**: 
    - None (standalone utility)
  - **Key Features**:
    - DependencyContainer class for managing dependencies
    - inject decorator for automatic dependency resolution
    - provides decorator for registering dependency providers
    - Support for singleton and transient dependency registration
    - Automatic dependency resolution based on type annotations
    - Circular dependency detection

## 26. [noise_generator.py](#utils)

- `src/utils/noise_generator.py`
  - **Purpose**: Abstract noise generator interface with concrete implementations
  - **File Imports**: 
    - None
  - **File Dependencies**: 
    - import abc
    - import logging
    - import random
    - import numpy as np
    - from typing import Optional, List, Tuple
  - **Required Components**: 
    - Optional: perlin_noise library (for PerlinNoiseGenerator)
    - Optional: noise library (for SimplexNoiseGenerator)
  - **Key Features**:
    - NoiseGenerator abstract base class
    - PerlinNoiseGenerator implementation
    - SimplexNoiseGenerator implementation
    - FallbackNoiseGenerator for when dependencies are missing
    - get_noise_generator factory function
    - Support for multi-octave noise generation
    - Graceful degradation when libraries are unavailable

## 27. [dependency_config.py](#utils)

- `src/utils/dependency_config.py`
  - **Purpose**: Centralized configuration system for managing dependencies
  - **File Imports**: 
    - from src.utils.dependency_injection import DependencyContainer, provides
    - from src.utils.noise_generator import NoiseGenerator, PerlinNoiseGenerator, SimplexNoiseGenerator, FallbackNoiseGenerator, get_noise_generator
  - **File Dependencies**: 
    - import logging
    - from typing import Dict, Any, Type, Optional, List, Callable
  - **Required Components**: 
    - dependency_injection.py
    - noise_generator.py
  - **Key Features**:
    - DependencyConfig class for centralized configuration options
    - app_container global DependencyContainer instance
    - configure_dependencies function for setting up dependencies
    - register_noise_generator function for configuring noise generators
    - load_config_from_file function for loading configuration from external files
    - Support for updating configuration at runtime

## 28. [dependency_settings.py](#config)

- `src/config/dependency_settings.py`
  - **Purpose**: Configuration settings for the dependency injection system
  - **File Imports**: 
    - None
  - **File Dependencies**: 
    - None
  - **Required Components**: 
    - None
  - **Key Features**:
    - NOISE_GENERATOR_TYPE setting for selecting noise generator implementation
    - NOISE_GENERATOR_SINGLETON setting for controlling singleton behavior
    - LOGGING_SINGLETON setting for controlling logger singleton behavior
    - Extensible design for adding more configuration options

## 29. [pattern_generator.py](#utils)

- `src/utils/pattern_generator.py`
  - **Purpose**: Utility functions for generating various patterns used in procedural generation
  - **File Imports**: None
  - **File Dependencies**: 
    - import math
    - import numpy as np
    - from typing import List, Optional, Tuple, Any
  - **Required Components**: None
  - **Key Features**:
    - generate_spiral_pattern: Creates spiral patterns with configurable density
    - generate_ring_pattern: Creates concentric ring patterns with configurable number and width
    - generate_gradient_pattern: Creates directional gradients with optional noise
    - generate_void_pattern: Creates areas of emptiness with soft edges
    - apply_weighted_patterns: Combines multiple patterns with weights

## 30. [value_generator.py](#utils)

- `src/utils/value_generator.py`
  - **Purpose**: Utility functions for generating value distributions and rare resource patterns
  - **File Imports**: None
  - **File Dependencies**: 
    - import numpy as np

- `src/utils/import_standards.py`
  - **Purpose**: Defines standardized import mechanisms and helper functions for the codebase
  - **File Imports**: None
  - **File Dependencies**: 
    - import inspect
    - import os
    - import re
    - from typing import Dict, List, Optional, Set, Tuple
    - from typing import Optional, Set, Union, List, Tuple, Any
  - **Required Components**: None
  - **Key Features**:
    - generate_value_distribution: Creates value distributions based on noise and statistical parameters
    - add_value_clusters: Adds clusters of higher values to a grid
    - generate_rare_resource_distribution: Creates distributions of rare resources

## 31. [cellular_automaton_utils.py](#utils)

- `src/utils/cellular_automaton_utils.py`
  - **Purpose**: Utility functions for cellular automaton operations used in procedural generation
  - **File Imports**: None
  - **File Dependencies**: 
    - import numpy as np
    - from typing import Set, Dict, Any, Tuple
    - import scipy.ndimage as ndimage
  - **Required Components**: None
  - **Key Features**:

## 32. [import_standards.py](#utils)

- `src/utils/import_standards.py`
  - **Purpose**: Defines standardized import mechanisms and helper functions for the codebase
  - **File Imports**: None
  - **File Dependencies**: 
    - import inspect
    - import os
    - import re
    - from typing import Dict, List, Optional, Set, Tuple
  - **Required Components**: None
  - **Key Features**:
    - Standardized import order (stdlib, third-party, optional, local)
    - Helper functions for generating standardized imports

## 33. [renderers.py](#ui)

- `src/ui/renderers.py`
  - **Purpose**: Specialized rendering components for Space Muck game elements
  - **File Imports**: 
    - from src.config import COLOR_ASTEROID_RARE, COLOR_BG, COLOR_GRID, COLOR_RACE_1, COLOR_RACE_2, COLOR_RACE_3
    - from src.ui.draw_utils import draw_text, draw_minimap
  - **File Dependencies**: 
    - import math
    - import random
    - from typing import List, Tuple
    - import numpy as np
    - import pygame
  - **Required Components**: None
  - **Key Features**:
    - AsteroidFieldRenderer: Renders asteroid fields with optimized performance
    - PlayerRenderer: Handles player ship and fleet rendering
    - Notification system for in-game messages
    - Color interpolation and caching for performance
    - Pre-rendered surfaces for common game elements
    - _render_surface_handler: Extracted method for fade overlay rendering

## 34. [shop.py](#ui)

- `src/ui/shop.py`
  - **Purpose**: Implements the in-game shop interface for Space Muck
  - **File Imports**: 
    - from src.config import COLOR_RACE_1, COLOR_RACE_2, COLOR_RACE_3, COLOR_TEXT, COLOR_UI_BUTTON, COLOR_UI_BUTTON_HOVER, WINDOW_WIDTH, WINDOW_HEIGHT
    - from src.ui.draw_utils import draw_button, draw_panel, draw_text, draw_tooltip
  - **File Dependencies**: 
    - import random
    - from typing import Any, Dict, List, Tuple
    - import pygame
  - **Required Components**: None
  - **Key Features**:
    - Shop: Main shop interface with multiple tabs and categories
    - ShipUpgradeShop: Specialized shop for ship upgrades
    - RaceDiscoveryShop: Interface for discovering and interacting with symbiotic races
    - EvolutionShop: Interface for evolving symbiotic races
    - _update_handler: Consolidated update logic for shop interactions
    - Tooltip system for displaying item information
    - Support for handling optional dependencies
    - Constants for categorizing common libraries
    - apply_cellular_automaton: Applies cellular automaton rules using a traditional implementation
    - apply_cellular_automaton_optimized: Applies rules using an optimized convolution-based approach
    - generate_cellular_automaton_rules: Generates rule sets based on entity parameters
    - apply_environmental_effects: Applies environmental effects to entities

## 32. [dependency_config_example.py](#examples)

## 32. [visualization.py](#utils)

- `src/utils/visualization.py`
  - **Purpose**: Comprehensive visualization tools for generator outputs and grid analysis
  - **File Imports**:
    - None (standalone utility module)
  - **File Dependencies**:
    - import os
    - import logging
    - import numpy as np
    - import matplotlib.pyplot as plt (optional with fallback)
    - from matplotlib import colors, cm (optional with fallback)
    - from PIL import Image (optional with fallback)
  - **Required Components**:
    - None (works with any numpy grid data)
  - **Key Features**:
    - Grid visualization with customizable colormaps
    - Multi-grid comparison for side-by-side analysis
    - Evolution visualization with animation support
    - Image export capabilities
    - Fallback mechanisms for environments without matplotlib/PIL

## 33. [test_pattern_generator.py](#unit_testing)

- `src/tests/test_pattern_generator.py`
  - **Purpose**: Comprehensive unit tests for pattern generation utilities
  - **File Imports**:
    - from src.utils.pattern_generator import (
        generate_circle_pattern,
        generate_rectangle_pattern,
        generate_random_pattern,
        generate_perlin_noise_pattern,
        generate_cellular_automaton_pattern
    )
  - **File Dependencies**:
    - import unittest
    - import numpy as np
  - **Required Components**:
    - pattern_generator.py utility module
  - **Key Tests**:
    - Test pattern dimensions and shapes
    - Test parameter validation
    - Test pattern characteristics (density, distribution)
    - Test edge cases and boundary conditions

## 34. [test_value_generator.py](#unit_testing)

- `src/tests/test_value_generator.py`
  - **Purpose**: Comprehensive unit tests for value generation utilities
  - **File Imports**:
    - from src.utils.value_generator import (
        generate_value_grid,
        generate_clustered_values,
        generate_resource_distribution,
        apply_distance_falloff
    )
  - **File Dependencies**:
    - import unittest
    - import numpy as np
  - **Required Components**:
    - value_generator.py utility module
  - **Key Tests**:
    - Test grid properties (dimensions, value ranges)
    - Test clustering behavior
    - Test resource distribution patterns
    - Test falloff application
    - Test statistical properties of generated values

## 35. [test_visualization.py](#testing)

- `src/tests/test_visualization.py`
  - **Purpose**: Demonstration and validation of visualization capabilities
  - **File Imports**:
    - from src.utils.visualization import GeneratorVisualizer
    - from src.utils.pattern_generator import generate_perlin_noise_pattern
  - **File Dependencies**:
    - import numpy as np
    - import os
  - **Required Components**:
    - visualization.py utility module
    - pattern_generator.py for test data generation
  - **Key Features**:
    - Demonstrates various visualization methods
    - Tests export capabilities
    - Validates colormap support
    - Shows evolution visualization

## 36. [dependency_config_example.py](#examples)

- `src/examples/dependency_config_example.py`
  - **Purpose**: Example script demonstrating the dependency configuration system
  - **File Imports**: 
    - from utils.dependency_injection import inject
    - from utils.dependency_config import app_container, DependencyConfig, load_config_from_file
    - from utils.noise_generator import NoiseGenerator
  - **File Dependencies**: 
    - import os
    - import sys
    - import logging
  - **Required Components**: 
    - dependency_injection.py
    - dependency_config.py
    - noise_generator.py
    - dependency_settings.py
  - **Key Features**:
    - NoiseGeneratorDemo class showing dependency injection in action
    - Example of loading configuration from a file
    - Demonstration of changing configuration at runtime
    - Comparison of different noise generator implementations
