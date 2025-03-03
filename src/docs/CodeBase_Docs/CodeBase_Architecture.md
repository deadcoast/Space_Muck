---
CODEBASE SYSTEM ARCHITECTURE
---

This file should be updated with information about the files design, structure, functionality, and architecture. The goal of this file is to provide a living document as the codebase evolves. The (#architecture_tag) is a string that can be used to identify how the file works as a code base asset. For example, a UI asset would have the tag "(#ui)".

## Table of Contents

1. [documentation_template.py](#code_base_template)
2. [base_entity.py](#entity_system)
3. [miner_entity.py](#entity_system)
4. [player.py](#entity_system)
5. [fleet.py](#entity_system)
6. [Entity System Architecture](#entity_system)
7. [Generator System Architecture](#generator_system)
8. [Performance Optimization Techniques](#optimization)
9. [World System Architecture](#world_system)
10. [Combat System Architecture](#combat_system)
11. [Trading System Architecture](#trading_system)
12. [GPU Acceleration Architecture](#gpu_acceleration)
13. [Fleet Combat System Architecture](#fleet_combat_system)

---

## 1. [documentation_template.py](#code_base_template)

### Key Design Principles

- **Code Reuse**: Common functionality is implemented in parent classes and inherited by child classes
- **Override Pattern**: Child classes override parent methods only when necessary
- **Attribute Specialization**: Child classes can override default attribute values from parent classes
- **Clear Responsibility**: Each class has a well-defined responsibility

#### Architecture Specifics

The game uses an inheritance-based architecture for entities to reduce code duplication and improve maintainability. The entity system follows this hierarchy:

```
BaseEntity
├── MinerEntity
│   └── Player
└── Fleet
```

Entities in the game interact through several established patterns:

1. **Parent-Child Relationships**: Inheritance provides specialized behavior while maintaining core functionality
2. **Position-Based Interactions**: Entities interact based on their positions in the game world
3. **Tag-Based Filtering**: The tagging system allows for flexible entity categorization and filtering
4. **Event Propagation**: Events can propagate up and down the inheritance chain as needed

### Entity Testing Architecture

The entity testing system follows a comprehensive approach to ensure all entity classes function correctly:

#### Testing Principles

1. **Inheritance-Aware Testing**: Tests verify both class-specific functionality and inherited behavior
2. **Attribute Validation**: Comprehensive testing of default and custom attribute initialization
3. **Mock Dependencies**: External dependencies are mocked to isolate entity behavior
4. **Behavior Verification**: Entity methods are tested with various inputs to verify correct behavior
5. **Edge Case Coverage**: Tests include boundary conditions and edge cases

#### Player Testing Framework

The Player class testing framework includes:

1. **Attribute Testing**:
   - Default attribute initialization
   - Custom attribute initialization
   - Inherited attribute verification
   - Level and XP attribute verification

2. **Functionality Testing**:
   - Credit management and transactions
   - Mining operations with different asteroid types
   - Movement and boundary conditions
   - Inventory management
   - Anomaly discovery
   - Quest handling
   - Level progression and XP gain

3. **Level Progression Testing**:
   - XP calculation for different mineral types
   - XP accumulation without level-up
   - Single level-up mechanics
   - Multiple level-ups from a single XP gain
   - Level cap enforcement
   - Level bonus application
   - Mining with XP gain
   - Level progress reporting

4. **Future Expansion**:
   - Ship upgrade system tests (placeholder)

5. **Dependency Mocking**:
   - External libraries (pygame, scipy, sklearn)
   - Game world components (AsteroidField)

This testing architecture ensures that the Player class and its inheritance chain are thoroughly tested, maintaining code quality and preventing regressions.

## 2. [base_entity.py](#entity_system)

### Key Design Principles

- **Core Entity Functionality**: Provides common attributes and methods for all entities
- **Lifecycle Management**: Handles entity activation and deactivation
- **Position System**: Manages entity positioning in the game world
- **Tagging System**: Implements categorization through entity tags
- **Serialization**: Provides to_dict() and from_dict() methods for data persistence

#### Architecture Specifics

BaseEntity serves as the root class for all entities in the game. It provides fundamental functionality that all entities need, including:

- Entity identification (ID, type, tags)
- Position management and movement
- Lifecycle control (activate/deactivate)
- Health and level attributes
- Serialization/deserialization for saving/loading

The class uses UUID for generating unique entity identifiers and implements a tagging system for flexible entity categorization. It's designed to be extended by more specialized entity classes.

## 3. [miner_entity.py](#entity_system)

### Key Design Principles

- **Cellular Automaton**: Implements cellular automaton logic for entity populations
- **Resource Gathering**: Handles territory management and resource collection
- **Evolution Mechanics**: Implements adaptation and evolution through traits
- **Population Management**: Contains methods for population growth and decline

#### Architecture Specifics

MinerEntity extends BaseEntity to create entities capable of mining resources and evolving using cellular automaton patterns. Key features include:

- Race-based attributes and behaviors
- Cellular automaton rules through birth and survival sets
- Territory management on the asteroid field
- Resource gathering and efficiency calculations
- Evolution and adaptation through the SymbioteEvolutionAlgorithm

The class integrates with external libraries like numpy, networkx, and perlin_noise to implement complex behaviors and visualizations.

## 4. [player.py](#entity_system)

### Key Design Principles

- **Player-Specific Attributes**: Adds inventory, credits, and other player resources
- **Quest Management**: Handles player progression through quests
- **Enhanced Mining**: Implements specialized mining mechanics for the player
- **Ship Management**: Manages player ship upgrades and capabilities
- **Level Progression**: Implements an XP-based level system with performance bonuses

#### Architecture Specifics

Player extends MinerEntity to represent the player character in the game. It inherits all the mining and evolution capabilities while adding player-specific features:

- Player identification (is_player flag, race_id 0)
- Economic attributes (credits, inventory)
- Progression tracking (discovered_anomalies, quests)
- Enhanced mining capabilities (mining_speed, efficiency)
- Ship management (ship_level)
- Level progression system with XP tracking

The Player class always starts with the "adaptive" trait and has higher mining efficiency (0.8) compared to the base MinerEntity.

The Player class inherits from MinerEntity but has several player-specific attributes:
- `is_player`: Boolean flag (always true)
- `credits`: Starting value of 1000
- `ship_level`: Starting at level 1
- `mining_speed`: Base multiplier of 1.0
- `inventory`: Dictionary for storing items
- `discovered_anomalies`: Set of discovered anomalies
- `current_quest`: Current active quest
- `completed_quests`: List of completed quests
- `level`: Current player level (starts at 1)
- `xp`: Current experience points
- `xp_to_next_level`: XP threshold required for next level
- `level_bonuses`: Dictionary of level-specific bonuses

Additionally, it overrides some MinerEntity defaults:
- `trait`: Always starts as "adaptive"
- `mining_efficiency`: Higher than base MinerEntity (0.8 vs default)
- `race_id`: Typically 0 for the player

#### Level Progression System

The Player class implements a comprehensive level progression system with the following components:

1. **XP Tracking**:
   - `xp`: Tracks current experience points
   - `xp_to_next_level`: Threshold for leveling up
   - XP is gained through mining activities

2. **Level-Up Mechanics**:
   - Player starts at level 1 with 0 XP
   - Each level requires progressively more XP (20% increase per level)
   - XP resets to 0 after level-up
   - Maximum level cap at 5 (configurable for future expansion)

3. **XP Calculation**:
   - Base XP gain is 10% of mining value
   - Multipliers applied based on mineral rarity:
     * Common minerals: 1.0x multiplier
     * Rare minerals: 1.5x multiplier
     * Anomalies: 3.0x multiplier

4. **Level Bonuses**:
   - Mining efficiency increases with level (+0.05 per level)
   - Mining speed increases with level (+0.1 per level)
   - Bonuses are cumulative across levels
   - Level 5 provides maximum bonuses:
     * +0.2 mining efficiency (total 1.0)
     * +0.4 mining speed (total 1.4)

5. **Implementation Details**:
   - `_calculate_xp_gain()`: Determines XP based on mining value and type
   - `_add_xp()`: Handles level-up logic and applies bonuses
   - `get_level_progress()`: Returns current level progression details
   - Multiple level-ups can occur from a single large XP gain
   - Level-up events are logged for monitoring

#### Ship Upgrade System

The Player class implements a tiered ship upgrade system with the following components:

1. **Ship Levels**:
   - Ships start at level 1 (base level)
   - Maximum ship level is 5
   - Each level provides increasing benefits

2. **Upgrade Costs**:
   - Level 2: 2,000 credits
   - Level 3: 5,000 credits
   - Level 4: 12,000 credits
   - Level 5: 25,000 credits

3. **Upgrade Benefits**:
   - Increased cargo capacity:
     * Level 2: +10 capacity
     * Level 3: +20 capacity
     * Level 4: +35 capacity
     * Level 5: +50 capacity
   - Improved mining speed:
     * Level 2: +0.2 speed
     * Level 3: +0.3 speed
     * Level 4: +0.4 speed
     * Level 5: +0.5 speed

4. **Implementation Details**:
   - `upgrade_ship()`: Handles ship upgrades with cost validation
   - Upgrades are cumulative with level progression bonuses
   - Ship upgrades require sufficient credits
   - Upgrade events are logged for monitoring

#### Quest Assignment System

The Player class implements a dynamic quest system with the following components:

1. **Quest Types**:
   - Mining quests: Collect a target amount of minerals
   - Exploration quests: Discover a target number of anomalies
   - Combat quests: Defeat a target number of enemy ships

2. **Quest Generation**:
   - Quests are dynamically generated based on player level
   - Difficulty scales with player level
   - Rewards scale with player level
   - Each quest has a unique ID for tracking

3. **Difficulty Scaling**:
   - Base difficulty multiplier: 1.0 + (level - 1) * 0.5
   - Base reward multiplier: 1.0 + (level - 1) * 0.7
   - Target requirements increase with player level

4. **Quest Completion**:
   - Mining progress updates during mining operations
   - Exploration progress updates when discovering anomalies
   - Combat progress updates during combat (not yet implemented)
   - Completing quests awards credits and bonus XP

5. **Implementation Details**:
   - `assign_quest()`: Assigns or generates a new quest
   - `_generate_quest()`: Creates a quest appropriate for player level
   - `complete_quest()`: Handles quest completion and rewards
   - Quest progress is tracked automatically during gameplay

#### Reputation System

The Player class implements a comprehensive reputation system with the following components:

1. **Factions**:
   - Miners Guild: Mining-focused faction
   - Explorers Union: Exploration-focused faction
   - Galactic Navy: Combat-focused faction
   - Traders Coalition: Trading-focused faction
   - Fringe Colonies: Outlaw/independent faction

2. **Reputation Levels**:
   - Hostile (-100 to -51): Cannot accept quests, 50% price markup
   - Unfriendly (-50 to -11): Limited quest access, 20% price markup
   - Neutral (-10 to 10): Standard interactions, normal prices
   - Friendly (11 to 50): Better quest rewards, 10% price discount
   - Allied (51 to 100): Best quest rewards, 20% price discount

3. **Reputation Effects**:
   - Trading prices adjusted based on reputation level
   - Quest availability filtered by reputation level
   - Quest rewards improved with higher reputation
   - Faction-specific quest types and descriptions

4. **Reputation Changes**:
   - Completing faction quests increases reputation
   - Reputation gain scales with quest difficulty
   - Reputation changes trigger level transition events
   - Reputation is bounded between -100 and 100

5. **Implementation Details**:
   - `get_reputation_level()`: Determines current standing with a faction
   - `change_reputation()`: Updates reputation values with bounds checking
   - `get_faction_price_modifier()`: Calculates price adjustments for trading
   - Faction quest completion automatically updates reputation
   - Reputation changes are logged for monitoring

## 5. [fleet.py](#entity_system)

### Key Design Principles

- **Ship Collection**: Manages a collection of ships that act as a unit
- **Formation Management**: Handles different fleet formations
- **Movement System**: Implements fleet movement and pathfinding
- **Resource Management**: Tracks fleet resources like fuel
- **Combat Capabilities**: Provides methods for calculating fleet strength

#### Architecture Specifics

Fleet extends BaseEntity to represent a collection of ships that move and act as a unit. Key features include:

- Ship management (adding, removing, organizing)
- Fleet movement and pathfinding
- Formation control
- Resource tracking (fuel consumption)
- Strength calculation based on ship composition

The Fleet class implements a basic pathfinding algorithm and handles movement through the game world, with provisions for more complex algorithms in the future.

## 6. [Generator System Architecture](#generator_system)

### Key Design Principles

- **Procedural Generation**: Generators create game content dynamically at runtime
- **Inheritance Hierarchy**: All generators inherit from BaseGenerator
- **Modular Implementation**: Complex generation logic is extracted into helper methods
- **Fallback Mechanisms**: Robust error handling with fallback generation when errors occur
- **Parameter Flexibility**: Generators accept parameters with sensible defaults
- **Dependency Injection**: Utility modules are injected for better testability and maintainability
- **Caching**: Performance optimization through intelligent caching of expensive operations
- **Vectorization**: Optimized operations using NumPy vectorization where possible

#### Architecture Specifics

The game uses an inheritance-based architecture for generators to provide common functionality while allowing specialized behavior. The generator system follows this hierarchy:

```
BaseEntity
└── BaseGenerator
    ├── ProceduralGenerator
    ├── AsteroidGenerator
    └── SymbioteEvolutionGenerator
```

Generators in the game share several common patterns:

1. **Method Extraction**: Complex generation logic is extracted into helper methods for better maintainability
2. **Performance Logging**: All generation methods include performance tracking
3. **Exception Handling**: Robust error handling with fallback generation
4. **Parameter System**: Flexible parameter handling with sensible defaults
5. **Noise Generation**: Common noise generation techniques shared across generators
6. **Utility Integration**: Integration with specialized utility modules for optimized operations
7. **Caching System**: Intelligent caching of expensive operations for performance
8. **Graceful Degradation**: Fallback to internal implementations when dependencies are unavailable

### BaseGenerator Refactoring

The BaseGenerator class has been significantly refactored to improve performance, maintainability, and flexibility:

#### Key Improvements

1. **Modular Integration with Utility Modules**:
   - Integrated with `cellular_automaton_utils.py` for optimized cellular automaton operations
   - Integrated with `value_generator.py` for efficient clustering operations
   - Implemented fallback mechanisms for both modules to maintain compatibility

2. **Enhanced Error Handling**:
   - Added parameter validation with appropriate warnings
   - Implemented try/except blocks with informative error messages
   - Added graceful degradation paths when optimal implementations fail

3. **Performance Optimizations**:
   - Maintained and improved caching mechanisms
   - Used vectorized operations where possible
   - Added performance logging
   - Calculated optimal parameters dynamically (e.g., cluster_radius)

4. **Code Quality Improvements**:
   - Enhanced docstrings with parameter and return value documentation
   - Standardized method signatures
   - Improved code organization with helper methods
   - Added comprehensive logging

5. **Testing and Verification**:
   - Added comprehensive unit tests for all methods
   - Created benchmark scripts for performance measurement
   - Implemented mock testing for utility module integration
   - Added parameter validation tests

The refactoring follows best practices for maintainability, performance, and code quality while preserving backward compatibility with existing game systems.

### 7. [procedural_generator.py](#generator_system)

#### Key Design Principles

- **Asteroid Field Generation**: Creates asteroid fields with varying densities and patterns
- **Mineral Distribution**: Handles rare mineral placement within asteroid fields
- **Cellular Automaton**: Uses cellular automaton for natural-looking patterns
- **Value Distribution**: Implements lognormal distribution for asteroid values
- **Unified Generation**: Provides a single method to generate all required grids

#### Architecture Specifics

The ProceduralGenerator extends BaseGenerator to create asteroid fields and mineral distributions. Key features include:

- Multiple noise layers for complex patterns
- Cellular automaton for natural-looking asteroid clusters
- Value distribution for asteroid resources
- Rare mineral generation with multiple rarity tiers
- Void space creation for visual interest
- Complete grid generation with a single method call

The class implements method extraction for better maintainability, with complex generation logic moved to descriptive helper methods like `generate_multi_layer_asteroid_field` and `generate_tiered_mineral_distribution`.

The new `generate()` method provides a unified interface for creating all required grids:
- `grid`: Main asteroid distribution
- `rare_grid`: Rare mineral distribution
- `energy_grid`: Energy levels for asteroids

This method includes robust error handling with fallback generation and applies rare mineral bonuses to the energy grid.

### 8. [symbiote_evolution_generator.py](#generator_system)

#### Key Design Principles

- **Colony Generation**: Creates initial symbiote colonies
- **Evolution Simulation**: Simulates evolution over multiple generations
- **Mineral Influence**: Mineral distribution affects evolution patterns
- **Mutation Mechanics**: Implements genome-based mutation system

## 8. [Performance Optimization Techniques](#optimization)

### Key Optimization Strategies

- **Caching Mechanism**: Implemented in BaseGenerator and utility classes
- **Vectorized Operations**: Replaced loops with NumPy operations
- **Fallback Implementations**: Graceful degradation when optimal libraries are unavailable
- **Dependency Injection**: Flexible component integration
- **Performance Benchmarking**: Systematic measurement of improvements

#### Architecture Specifics

1. **Caching System**
   - Implemented in BaseGenerator with _cache dictionary
   - Uses _get_cache_key for generating unique cache keys
   - _cache_result and _get_cached_result methods for cache management
   - Provides 10-100x speedup for repeated operations
   - Cache invalidation based on parameter changes

2. **Vectorized Grid Processing**
   - Replaced nested loops with NumPy array operations
   - Used convolution for cellular automaton neighbor counting
   - Implemented vectorized thresholding and clustering
   - Optimized grid creation with array broadcasting

3. **Optimized Cellular Automaton**
   - Modular implementation with focused helper methods
   - SciPy-based implementation with fallback to manual processing
   - Vectorized rule application for Conway's Game of Life
   - Parameter validation with appropriate warnings

4. **Efficient Clustering**
   - Dynamic calculation of optimal cluster parameters
   - Vectorized distance calculations
   - Efficient cluster identification and labeling
   - Memory-efficient implementation for large grids

5. **Graceful Degradation**
   - Fallback implementations when optimal libraries are unavailable
   - Consistent API regardless of available dependencies
   - Informative warnings when using fallback methods
   - Performance logging for identifying bottlenecks

6. **Verification Framework**
   - Comprehensive verification scripts (verify_base_generator_optimizations.py)
   - Benchmark scripts for measuring performance (benchmark_base_generator.py)
   - Visualization support for debugging and analysis
   - Comparison tools for validating optimized implementations

7. **Parallel Processing Framework**:
   - Implemented in BaseGenerator for computationally intensive operations:
     * Cellular automaton evolution with _apply_cellular_automaton_parallel
     * Value clustering with _apply_parallel_clustering
   - Configurable activation thresholds (default: 40,000 cells)
     * _parallel_ca_threshold for cellular automaton operations
     * _parallel_clustering_threshold for clustering operations
   - Dynamic worker allocation based on available CPU cores (max 8 workers)
   - Chunk-based processing for optimal load distribution
     * Grid-based chunking for cellular automaton operations
     * Task-based chunking for clustering operations
   - Reproducibility through worker-specific seeded random number generation
   - Graceful fallback to sequential processing on errors
   - Benchmark results (from benchmark_parallel_processing.py):
     * Performance improvements more noticeable for larger grid sizes
     * Speedup factors range from 0.98x (small grids) to 2.46x (large grids)
     * Minimal overhead for smaller grids (< 100x100)
     * Optimal performance achieved with grid sizes > 200x200
   - Future improvement opportunities:
     * GPU acceleration for even larger grid sizes
     * Adaptive threshold selection based on hardware capabilities
     * Hybrid processing model combining parallel CPU and GPU operations
     * Dynamic load balancing for heterogeneous workloads
   - Maintains consistent API across different processing methods
   - Comprehensive benchmarking with benchmark_parallel_processing.py
     * Performance measurement across grid sizes (50x50 to 500x500)
     * Speedup visualization for different operations
     * Median-based timing to eliminate outliers

These optimization techniques have been systematically applied throughout the codebase, with particular focus on the BaseGenerator class and its utility modules. The improvements maintain backward compatibility while significantly enhancing performance for procedural generation operations.

## 9. [World System Architecture](#world_system)

### Key Design Principles

- **Grid-Based World**: The game world is represented as a grid of cells
- **Procedural Generation**: World content is generated procedurally
- **Entity Management**: World objects manage entities within them
- **Resource Distribution**: Resources are distributed based on procedural algorithms

#### Architecture Specifics

The world system provides the environment in which entities exist and interact. It uses grid-based representations for various world properties and leverages the generator system for content creation.

### 9. [asteroid_field.py](#world_system)

#### Key Design Principles

- **Grid Representation**: Represents the asteroid field as a grid of cells
- **Generator Integration**: Uses generator classes for content creation
- **Multiple Grid Layers**: Maintains separate grids for different properties
- **Entity Tracking**: Tracks entities within the asteroid field

#### Architecture Specifics

The AsteroidField class represents the asteroid field in the game, with multiple grid layers:

- `grid`: Main asteroid distribution with asteroid values
- `rare_grid`: Rare mineral distribution
- `energy_grid`: Energy levels for asteroids
- `entity_grid`: Entities present in the field
- `influence_grids`: Influence maps for different factions

The class integrates with generator classes for content creation:

- Uses `ProceduralGenerator.generate()` for creating the asteroid field
- Uses `SymbioteEvolutionGenerator` for simulating symbiote evolution

The updated implementation uses the new generator methods for more streamlined and maintainable code, with proper parameter handling and robust error handling.

The SymbioteEvolutionGenerator extends BaseGenerator to simulate symbiote evolution with colonies and mutations. Key features include:

- Initial colony placement using noise-based distribution
- Mineral distribution that influences evolution
- Genome-based mutation system
- Evolution history tracking
- Cellular automaton for colony growth patterns

The class implements method extraction for better maintainability, with complex generation logic moved to helper methods like `_colony_handler` and `_extracted_from_generate_mineral_distribution_12`.

## 9. [Dependency Injection System](#utils)

### Key Design Principles

- **Inversion of Control**: Dependencies are provided to objects rather than created within them
- **Type-Based Resolution**: Dependencies are resolved based on type annotations
- **Singleton Support**: Dependencies can be registered as singletons or transient instances
- **Decorator-Based API**: Simple decorator-based API for registering and injecting dependencies
- **Circular Dependency Detection**: Automatic detection of circular dependencies

#### Architecture Specifics

The dependency injection system provides a flexible framework for managing dependencies throughout the codebase. It consists of:

- `DependencyContainer`: Central registry for dependencies with methods for registration and resolution
- `provides` decorator: Registers a function or class as a provider for a specific type
- `inject` decorator: Automatically injects dependencies into function or method parameters

The system supports two dependency lifetimes:
- **Singleton**: One instance shared across the application
- **Transient**: New instance created each time the dependency is requested

Dependencies are resolved based on type annotations, allowing for clean, declarative code. The system includes circular dependency detection to prevent infinite recursion during resolution.

Example usage:

```python
from src.utils.dependency_injection import DependencyContainer, provides, inject

container = DependencyContainer()

@provides(Database, container=container, singleton=True)
def provide_database() -> Database:
    return Database(connection_string="...")

class UserService:
    @inject(container=container)
    def __init__(self, database: Database):
        self.database = database
```

## 10. [Noise Generator Abstraction](#utils)

### Key Design Principles

- **Abstract Interface**: Common interface for all noise generators
- **Multiple Implementations**: Support for different noise algorithms
- **Graceful Degradation**: Fallback mechanisms when dependencies are unavailable
- **Multi-Octave Support**: Enhanced noise generation with multiple octaves
- **Factory Pattern**: Simple factory function for obtaining appropriate generators

#### Architecture Specifics

The noise generator abstraction provides a unified interface for generating noise used in procedural generation. Key components include:

- `NoiseGenerator`: Abstract base class defining the interface for all noise generators
- `PerlinNoiseGenerator`: Implementation using the perlin_noise library
- `SimplexNoiseGenerator`: Implementation using the noise library
- `FallbackNoiseGenerator`: Simple implementation for when external libraries are unavailable
- `get_noise_generator`: Factory function that returns the best available generator

The system supports multi-octave noise generation for more natural-looking results, with configurable persistence, lacunarity, and octave count. It gracefully degrades when dependencies are missing, ensuring that procedural generation can continue even without optimal noise libraries.

This abstraction is integrated with the dependency injection system, allowing generator classes to request noise generators without needing to know the specific implementation details.

## 11. [Import Standards](#utils)

### Import Standardization

The codebase follows a standardized import pattern to improve readability, maintainability, and consistency. The standard is defined in `src/utils/import_standards.py` and includes:

#### Import Order

1. **Standard library imports** (alphabetically sorted)
   ```python
   # Standard library imports
   import logging
   import math
   import random
   from typing import Dict, List, Optional
   ```

2. **Third-party library imports** (alphabetically sorted)
   ```python
   # Third-party library imports
   import numpy as np
   import scipy.ndimage as ndimage
   ```

3. **Optional dependencies** (with fallback handling)
   ```python
   # Optional dependencies
   try:
       from perlin_noise import PerlinNoise
       PERLIN_AVAILABLE = True
   except ImportError:
       PERLIN_AVAILABLE = False
       print("PerlinNoise package is not available. Using fallback noise generator.")
   ```

4. **Local application imports** (alphabetically sorted)
   ```python
   # Local application imports
   from src.config import *
   from src.entities.base_entity import BaseEntity
   from src.utils.dependency_injection import inject
   ```

#### Import Guidelines

- Always use absolute imports (starting with 'src.') for local modules
- Group imports by category with a blank line between categories
- Sort imports alphabetically within each category
- Use try/except for optional dependencies with clear fallback messages
- Define constants for dependency availability (e.g., SCIPY_AVAILABLE)
- Import specific classes/functions rather than entire modules when possible
- Avoid circular imports by using type hints and forward references

### Helper Functions

The `import_standards.py` module provides helper functions for generating standardized import statements, handling optional dependencies, and maintaining consistent import patterns across the codebase.

## 12. [Pattern and Value Generation Utilities](#utils)

### Key Design Principles

- **Centralized Functionality**: Common pattern and value generation functions in dedicated modules
- **Reusability**: Functions designed to be reused across different generator classes
- **Configurability**: Highly configurable parameters for flexible generation
- **Separation of Concerns**: Clear separation between pattern generation and value distribution

### Pattern Generator

The `pattern_generator` module provides functions for generating various patterns used in procedural generation:

- `generate_spiral_pattern`: Creates spiral patterns with configurable density
- `generate_ring_pattern`: Creates concentric ring patterns with configurable number and width
- `generate_gradient_pattern`: Creates directional gradients with optional noise
- `generate_void_pattern`: Creates areas of emptiness with soft edges
- `apply_weighted_patterns`: Combines multiple patterns with weights

These functions eliminate code duplication across generator classes and provide a standardized way to create and combine patterns.

### Value Generator

The `value_generator` module provides functions for generating value distributions:

- `generate_value_distribution`: Creates value distributions based on noise and statistical parameters
- `add_value_clusters`: Adds clusters of higher values to a grid
- `generate_rare_resource_distribution`: Creates distributions of rare resources

These utilities ensure consistent value generation across different generator types and reduce code duplication.

### Cellular Automaton Utilities

The `cellular_automaton_utils` module provides functions for cellular automaton operations used in procedural generation:

- `apply_cellular_automaton`: Applies cellular automaton rules to a grid using a traditional implementation
- `apply_cellular_automaton_optimized`: Applies rules using an optimized convolution-based approach
- `generate_cellular_automaton_rules`: Generates rule sets based on entity parameters
- `apply_environmental_effects`: Applies environmental effects to entities based on mineral distribution

#### Fallback Strategy

The module implements a robust fallback strategy for handling missing dependencies:

- Attempts to import scipy for optimized convolution operations
- If scipy is not available, falls back to the standard implementation
- Provides a warning message when using the fallback implementation
- Ensures consistent behavior regardless of available dependencies

#### Unit Testing

Comprehensive unit tests have been created for the cellular automaton utilities:

- Tests for standard cellular automaton implementation
- Tests for optimized implementation with fallback handling
- Tests for rule generation based on different genome parameters
- Tests for environmental effects with different hostility levels

The tests ensure that the utilities work correctly and consistently across different environments and configurations.ed on external factors

These utilities provide consistent cellular automaton operations across different generator classes and offer both standard and optimized implementations for different use cases.

## Dependency Configuration System

The Space Muck codebase includes a flexible configuration system for managing dependencies. This system allows for centralized configuration of dependencies, making it easy to change implementations or behavior without modifying code.

### DependencyConfig

The `DependencyConfig` class provides a centralized location for configuration options. It includes settings for noise generator type, singleton behavior, and more.

```python
# Get current configuration
config_dict = DependencyConfig.to_dict()

# Update configuration
DependencyConfig.NOISE_GENERATOR_TYPE = "simplex"
DependencyConfig.NOISE_GENERATOR_SINGLETON = True
```

### Configuration Files

The system supports loading configuration from external files, making it easy to change behavior without modifying code.

```python
# Load configuration from a file
load_config_from_file("/path/to/config.py")
```

A typical configuration file might look like this:

```python
# dependency_settings.py
NOISE_GENERATOR_TYPE = "perlin"  # Options: "perlin", "simplex", "fallback", "auto"
NOISE_GENERATOR_SINGLETON = True  # Whether to use a singleton noise generator
LOGGING_SINGLETON = True  # Whether to use a singleton logger
```

### Dynamic Dependency Registration

The system includes functions for dynamically registering dependencies based on configuration:

```python
# Configure all dependencies
configure_dependencies()

# Register a specific noise generator
register_noise_generator()
```

### Global Container

The system provides a global `app_container` instance that can be used throughout the application:

```python
# Use the global container
@inject(container=app_container)
def my_function(noise_generator: NoiseGenerator):
    # Use the noise generator
    pass
```

### Example Usage

Here's an example of how to use the dependency configuration system:

```python
# Load configuration from a file
load_config_from_file("config/dependency_settings.py")

# Use a class with injected dependencies
@inject(container=app_container)
class NoiseGeneratorDemo:
    def __init__(self, noise_generator: NoiseGenerator):
        self.noise_generator = noise_generator
    
    def generate_sample(self):
        # Use the noise generator
        noise = self.noise_generator.generate_noise(10, 10)
        return noise

# Create an instance (dependencies will be injected)
demo = NoiseGeneratorDemo()
sample = demo.generate_sample()
```

---

## 10. [Combat System Architecture](#combat_system)

The Space Muck combat system provides a comprehensive framework for player-enemy ship interactions, encounter generation, and combat resolution. The system is designed to be modular, extensible, and integrated with the game's reputation and quest systems.

### Key Components

1. **Enemy Ship Generation**
   - Procedural enemy ship creation based on player level and location
   - Faction-aware generation with appropriate attributes
   - Difficulty scaling based on game progression
   - Loot generation system for defeated enemies

2. **Combat Mechanics**
   - Turn-based combat system with initiative
   - Weapon and defense systems with multiple upgrade levels
   - Critical hit mechanics and damage calculation
   - Shield and hull management
   - Player flee mechanics with success probability

3. **Encounter Generation**
   - Zone-based danger levels across the game map
   - Faction-aware encounter generation
   - Quest-specific encounter support
   - Encounter cooldown system to prevent spam

### System Architecture

The combat system follows this architecture:

```
CombatSystem
├── Manages combat encounters
├── Handles attack and defense mechanics
├── Implements combat turn resolution
└── Manages player flee mechanics

EncounterGenerator
├── Generates combat encounters
├── Implements zone-based danger levels
└── Supports faction-based and quest-specific encounters

EnemyShip
├── Procedural enemy ship generation
├── Combat attributes and mechanics
└── Loot generation system
```

### Integration Points

1. **Player Integration**
   - Combat stats stored in Player class
   - Reputation impact from combat actions
   - Experience and rewards from combat

2. **Quest System Integration**
   - Combat-specific quests
   - Quest progression through combat
   - Special enemy types for quests

3. **Game World Integration**
   - Location-based encounter generation
   - Zone danger levels affecting encounter frequency
   - Faction territories influencing enemy types

### Configuration

The combat system is highly configurable through constants in the `config.py` file:

```python
# Combat system settings
COMBAT_BASE_ATTACK_POWER: int = 10  # Base attack power for level 1 weapons
COMBAT_BASE_ATTACK_SPEED: float = 1.0  # Base attacks per time unit
COMBAT_BASE_WEAPON_RANGE: int = 5  # Base weapon range in grid units
COMBAT_BASE_CRIT_CHANCE: float = 0.05  # 5% base critical hit chance
COMBAT_CRIT_MULTIPLIER: float = 2.0  # Critical hits do double damage

COMBAT_BASE_SHIELD_STRENGTH: int = 50  # Base shield points
COMBAT_BASE_SHIELD_RECHARGE: float = 1.0  # Shield points recharged per time unit
COMBAT_BASE_HULL_STRENGTH: int = 100  # Base hull integrity points
COMBAT_BASE_EVASION: float = 0.1  # 10% base chance to evade attacks
COMBAT_BASE_ARMOR: float = 0.05  # 5% base damage reduction

COMBAT_WEAPON_UPGRADE_COST: List[int] = [0, 1500, 4000, 10000, 20000]  # Costs for weapon levels 1-5
COMBAT_SHIELD_UPGRADE_COST: List[int] = [0, 2000, 5000, 12000, 25000]  # Costs for shield levels 1-5
COMBAT_HULL_UPGRADE_COST: List[int] = [0, 3000, 7000, 15000, 30000]  # Costs for hull levels 1-5

COMBAT_ENEMY_TYPES: List[str] = ["pirate", "patrol", "mercenary", "elite"]  # Types of enemy ships
COMBAT_DIFFICULTY_MULTIPLIER: Dict[str, float] = {  # Difficulty multipliers for enemy stats
    "easy": 0.8,
    "medium": 1.0,
    "hard": 1.3,
    "elite": 1.8
}
```

### Future Expansion

The combat system is designed to be easily expandable with:

1. **Advanced Tactical Features**
   - Special abilities and combat maneuvers
   - Environmental effects on combat
   - Fleet combat with multiple ships

2. **Enhanced Enemy Variety**
   - More enemy ship types and factions
   - Special boss encounters
   - Unique enemy abilities

3. **Deeper Quest Integration**
   - Multi-stage combat quests
   - Unique combat scenarios
   - Story-driven encounters

---

## 11. [Trading System Architecture](#trading_system)

### Key Design Principles

- **Dynamic Economy**: Implements a realistic, responsive market economy with supply and demand mechanics
- **Location-Based Pricing**: Prices vary based on station location and local conditions
- **Faction Integration**: Faction reputation affects trading prices and available quests
- **Procedural Generation**: Trading stations and market events are procedurally generated
- **Quest Integration**: Trading system generates and validates trading-specific quests

### System Components

1. **Commodity System**
   - Five primary commodity types with base prices
   - Dynamic price fluctuations based on supply and demand
   - Price volatility modeling for realistic market behavior
   - Rarity-based price modifiers

2. **Trading Station System**
   - Procedural station generation with location-based attributes
   - Station inventory management with periodic restocking
   - Station specialization affecting available commodities
   - Faction affiliation affecting prices and available quests

3. **Transaction System**
   - Buy/sell functionality with inventory validation
   - Reputation-based price modifiers
   - Transaction logging for debugging and analytics
   - Market manipulation detection

4. **Market Event System**
   - Procedural market event generation (shortages, surpluses, etc.)
   - Time-based event duration and intensity
   - Event propagation across nearby stations
   - Event-based quest generation

5. **Quest Generation System**
   - Multiple quest types (delivery, procurement, market manipulation, rare commodity)
   - Difficulty scaling based on player level
   - Faction-specific quest generation
   - Comprehensive quest completion validation

### Integration Points

1. **Player Integration**
   - Inventory management for commodities
   - Credits management for transactions
   - Reputation impact from trading activities
   - Quest rewards and progression

2. **Faction System Integration**
   - Faction reputation affects prices
   - Faction-specific trading quests
   - Faction territory influences available commodities
   - Faction conflicts create market events

3. **Game World Integration**
   - Trading stations placed throughout game world
   - Location affects available commodities and prices
   - Distance-based price differences encourage travel
   - Regional market events

### Configuration

The trading system is configurable through constants that could be added to the `config.py` file:

```python
# Trading system settings
TRADING_BASE_PRICE_FLUCTUATION: float = 0.2  # Base price fluctuation (±20%)
TRADING_STATION_RESTOCK_INTERVAL: int = 24  # Hours between station restocks
TRADING_MAX_INVENTORY_PER_COMMODITY: int = 1000  # Maximum inventory per commodity
TRADING_MIN_INVENTORY_PER_COMMODITY: int = 50  # Minimum inventory per commodity
TRADING_EVENT_DURATION_RANGE: Tuple[int, int] = (24, 72)  # Event duration in hours (min, max)
TRADING_EVENT_INTENSITY_RANGE: Tuple[float, float] = (0.1, 0.5)  # Event price impact (min, max)

# Commodity base prices
TRADING_COMMODITY_BASE_PRICES: Dict[str, int] = {
    "minerals": 100,
    "technology": 250,
    "luxury_goods": 500,
    "medical_supplies": 350,
    "contraband": 750
}

# Faction reputation price modifiers
TRADING_REPUTATION_PRICE_MODIFIERS: Dict[str, float] = {
    "hostile": 1.5,      # 50% markup
    "unfriendly": 1.25,  # 25% markup
    "neutral": 1.0,      # No modifier
    "friendly": 0.9,     # 10% discount

---

## 12. [GPU Acceleration Architecture](#gpu_acceleration)

### Overview

The GPU Acceleration system provides hardware-accelerated implementations of computationally intensive operations in Space Muck, with graceful fallbacks for systems without GPU support. This architecture enables significant performance improvements for large-scale operations while maintaining compatibility across different hardware configurations.

### Key Design Principles

1. **Abstraction Layer**: Provides a unified API that abstracts away the underlying GPU implementation details.
2. **Graceful Degradation**: Automatically falls back to CPU implementations when GPU support is unavailable.
3. **Multiple Backend Support**: Supports multiple GPU acceleration libraries (CUDA via Numba, CuPy) for maximum compatibility.
4. **Performance Optimization**: Optimized implementations for common operations like cellular automaton and noise generation.
5. **Benchmarking Tools**: Includes tools for measuring and comparing performance across different backends.

### Components

#### 1. GPU Utilities Module (`gpu_utils.py`)

The core module providing GPU-accelerated implementations with CPU fallbacks:

```python
# Core functionality
is_gpu_available()  # Check if any GPU acceleration is available
get_available_backends()  # Get list of available GPU backends
to_gpu(array)  # Transfer array to GPU
to_cpu(array)  # Transfer array from GPU to CPU

# Accelerated operations
apply_cellular_automaton_gpu()  # GPU-accelerated cellular automaton
apply_noise_generation_gpu()  # GPU-accelerated noise generation
```

#### 2. Backend Detection and Selection

The system automatically detects available GPU backends at runtime:

- **CUDA via Numba**: For NVIDIA GPUs with CUDA support
- **CuPy**: For more general GPU support with a NumPy-compatible API
- **CPU Fallback**: Automatically used when no GPU is available

#### 3. Memory Management

Efficient transfer of data between CPU and GPU memory:

- **Lazy Transfer**: Data is only transferred when necessary
- **Memory Optimization**: Minimizes redundant memory transfers
- **Array Compatibility**: Maintains compatibility with NumPy arrays

### Accelerated Operations

#### 1. Cellular Automaton

Provides GPU-accelerated implementations of cellular automaton operations:

- **CUDA Implementation**: Uses CUDA kernels for parallel processing on NVIDIA GPUs
- **CuPy Implementation**: Uses CuPy's FFT-based convolution for efficient computation
- **CPU Fallback**: Uses the existing optimized CPU implementation

#### 2. Noise Generation

Provides GPU-accelerated implementations of noise generation:

- **CuPy Implementation**: Uses CuPy's FFT capabilities for efficient noise generation
- **CPU Fallback**: Uses the existing CPU-based noise generator

### Performance Considerations

1. **Grid Size Thresholds**: GPU acceleration provides the most benefit for large grids (typically >256x256)
2. **Memory Transfer Overhead**: For small operations, the overhead of transferring data to/from GPU may outweigh benefits
3. **Batch Processing**: Operations are batched where possible to maximize GPU utilization
4. **Hardware Variability**: Performance varies based on GPU capabilities and driver versions

### Testing and Benchmarking

1. **Unit Tests**: Comprehensive tests ensure consistent results across backends
2. **Benchmark Suite**: Tools for measuring performance across different grid sizes and configurations
3. **Visualization**: Performance comparison plots to identify optimal configurations

### Integration Points

The GPU acceleration system integrates with the following components:

1. **Cellular Automaton Utilities**: Accelerates cellular automaton operations used in procedural generation
2. **Noise Generation**: Accelerates noise generation used in terrain and asteroid field creation
3. **Base Generator**: Provides accelerated alternatives to computationally intensive operations

### Future Expansion

1. **Additional Operations**: Extend GPU acceleration to more operations (clustering, pathfinding)
2. **Multi-GPU Support**: Support for systems with multiple GPUs
3. **Adaptive Selection**: Dynamically choose optimal backend based on operation characteristics
4. **Tensor Operations**: Support for machine learning operations using GPU acceleration
    "allied": 0.75      # 25% discount
}
```

### Future Expansion

The trading system is designed to be easily expandable with:

1. **Advanced Economic Features**
   - Market simulation with AI traders
   - Economic cycles and trends
   - Player-owned trading stations
   - Commodity futures and investments

2. **Enhanced Trading Mechanics**
   - More commodity types and specializations
   - Black market trading with high risk/reward
   - Trade route optimization
   - Commodity transformation (manufacturing)

3. **Deeper Faction Integration**
   - Faction-specific trade goods
   - Trade embargoes between hostile factions
   - Trade agreements and treaties
   - Economic warfare mechanics

---

## 13. [Fleet Combat System Architecture](#fleet_combat_system)

### Overview

The Fleet Combat System provides a comprehensive framework for fleet-to-fleet engagements in Space Muck, with formation-based tactics, stance-based combat mechanics, and realistic damage distribution. This system integrates with the Fleet Management System to create dynamic and strategic combat encounters.

### Key Design Principles

1. **Formation-Based Combat**: Different fleet formations provide unique combat advantages and disadvantages
2. **Stance-Based Tactics**: Combat stances determine engagement distance, damage output, and defensive capabilities
3. **Dynamic Positioning**: Fleets adjust position during combat based on their stance
4. **Realistic Damage Distribution**: Damage is distributed among ships based on formation type
5. **Autonomous Engagement**: Optional auto-engagement system for AI-controlled fleets

### Components

#### 1. Combat Engagement System

Manages the initiation and termination of combat between fleets:

```python
# Core engagement functionality
engage_fleet(target_fleet, stance)  # Initiate combat with another fleet
disengage()  # Exit combat state
set_auto_engagement(auto_engage, attack_same_faction)  # Configure auto-engagement settings
```

#### 2. Combat Stance System

Provides different tactical options for fleet combat:

```python
# Combat stance configurations
stance_damage_multipliers = {
    "balanced": 1.0,     # Balanced damage
    "aggressive": 1.5,   # High damage
    "defensive": 0.7,    # Low damage
    "evasive": 0.4       # Very low damage
}

stance_defense_multipliers = {
    "balanced": 1.0,     # Balanced defense
    "aggressive": 0.7,   # Low defense
    "defensive": 1.5,    # High defense
    "evasive": 1.3       # Good defense
}

ideal_distances = {
    "balanced": 5.0,    # Medium range
    "aggressive": 2.0,  # Close range
    "defensive": 8.0,   # Long range
    "evasive": 10.0     # Very long range
}
```

#### 3. Formation-Based Combat

Implements formation-specific combat mechanics and damage distribution:

```python
# Formation-specific damage distribution functions
_distribute_damage_evenly()        # Line formation: even distribution
_distribute_damage_front_heavy()   # Column formation: front ship takes 40%
_distribute_damage_point_heavy()   # Wedge formation: point ship takes 50%
_distribute_damage_flank_heavy()   # Echelon formation: flank ship takes 40%
_distribute_damage_flagship_protected()  # Circle formation: flagship takes only 10%
_distribute_damage_randomly()      # Scatter formation: random distribution
```

#### 4. Combat Positioning

Manages fleet positioning during combat based on stance:

```python
# In _handle_combat method
if abs(distance - ideal_distance) > 0.5:
    # Calculate direction vector
    dx = target_fleet.position[0] - self.position[0]
    dy = target_fleet.position[1] - self.position[1]
    
    # Normalize
    if distance > 0:
        dx /= distance
        dy /= distance
        
    # Determine if we need to move closer or further
    if distance > ideal_distance:
        # Move closer
        move_distance = min(speed * delta_time, distance - ideal_distance)
    else:
        # Move away
        move_distance = min(speed * delta_time, ideal_distance - distance)
        dx = -dx
        dy = -dy
```

#### 5. Damage System

Calculates and applies damage during combat:

```python
# Fleet strength calculation
def get_fleet_strength(self) -> float:
    if not self.ships:
        return 0.0
        
    # Base strength is the sum of all ships' attack power
    base_strength = sum(ship.attack_power for ship in self.ships)
    
    # Apply formation multipliers
    formation_mult = formation_multipliers.get(self.formation, 1.0)
    
    # Apply commander level bonus
    commander_mult = 1.0 + (self.commander_level - 1) * 0.05
    
    # Apply morale modifier
    morale_mult = self.morale
    
    # Calculate final strength
    strength = base_strength * formation_mult * commander_mult * morale_mult
    
    return strength
```

### Integration Points

The Fleet Combat System integrates with the following components:

1. **Fleet Management System**: Builds on the fleet management functionality for movement and formation
2. **Enemy Ship System**: Interacts with individual ships for damage application and destruction
3. **Faction System**: Uses faction relationships to determine engagement rules

### Future Expansion

1. **Advanced Combat Tactics**
   - Special maneuvers (flanking, pincer movements)
   - Coordinated multi-fleet tactics
   - Environmental factors affecting combat

2. **Enhanced Ship Roles**
   - Specialized ship roles within formations
   - Role-specific combat bonuses
   - Officer assignments affecting performance

3. **Strategic Combat Elements**
   - Supply lines and logistics
   - Fleet morale system affecting performance
   - Retreat and reinforcement mechanics
