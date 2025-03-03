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
8. [World System Architecture](#world_system)

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

#### Architecture Specifics

Player extends MinerEntity to represent the player character in the game. It inherits all the mining and evolution capabilities while adding player-specific features:

- Player identification (is_player flag, race_id 0)
- Economic attributes (credits, inventory)
- Progression tracking (discovered_anomalies, quests)
- Enhanced mining capabilities (mining_speed, efficiency)
- Ship management (ship_level)

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

Additionally, it overrides some MinerEntity defaults:
- `trait`: Always starts as "adaptive"
- `mining_efficiency`: Higher than base MinerEntity (0.8 vs default)
- `race_id`: Typically 0 for the player

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

## 8. [World System Architecture](#world_system)

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

