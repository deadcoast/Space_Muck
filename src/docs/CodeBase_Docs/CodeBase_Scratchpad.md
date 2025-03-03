---
CODEBASE SCRATCHPAD
---

# Scratchpad

## Primary Task: Removing Duplicated Code & Code Refactoring

1. **DELETING IMPLEMENTATIONS OR SILENCING METHODS IS NOT AN ACCEPTABLE FIX FOR TYPESCRIPT ERRORS**
2. **THE DEFINITION OF FIXING ERRORS IS IMPLEMENTING THEM PROPERLY, WITH THE REST OF THE CODE BASE**

- Review the python files and identify duplicated code.
- Identify the implementations **ALL CURRENT IMPLEMENTATIONS MUST BE RETAINED**
- Identify the most appropriate way to refactor the code.
- Implement the refactoring.

### Next Steps
1. [ ] Implement performance optimizations for entity operations
2. [X] Standardize import mechanisms across the codebase
3. [X] Develop more robust fallback strategies for missing dependencies
4. [ ] Create comprehensive documentation for the generator classes
5. [ ] Implement visualization improvements for generator outputs
6. [X] Refactor AsteroidGenerator to use utility modules
7. [X] Refactor SymbioteEvolutionGenerator to use utility modules
8. [X] Create unit tests for utility modules (cellular_automaton_utils)
9. [ ] Create unit tests for other utility modules (pattern_generator, value_generator)
10. [X] Fix linting issues in base_generator.py (missing imports, mutable defaults)
11. [X] Add fallback for PerlinNoise in miner_entity.py
12. [X] Fix linting issues in renderers.py (import * issues, undefined variables)
13. [X] Fix linting issues in shop.py (import * issues, unused imports, undefined variables)
14. [X] Implement scipy.stats in miner_entity.py for statistical analysis and distributions
15. [X] Fix missing List import in import_standards.py
16. [X] Fix undefined variables in asteroid_field.py (_update_entities_implementation method)
17. [X] Fix linting issues in asteroid_field.py (unused imports, star imports, unused variable)
18. [X] Further optimize imports in asteroid_field.py to only include used constants
19. [X] Fix unused variable 'noise_gen' in base_generator.py by implementing proper fallback

### Import Standardization Task

Standardizing import mechanisms across the codebase to improve maintainability and readability.

#### Completed
- Created `src/utils/import_standards.py` to define consistent import mechanisms
- Implemented standardized import order across multiple files
- Added helper functions for generating standardized imports
- Fixed import-related linting issues in base_generator.py
- Added proper handling of optional dependencies
- Standardized imports in shop.py
- Standardized imports in miner_entity.py with fallback for PerlinNoise
- Replaced import * with explicit imports from config in miner_entity.py
- Implemented entity behavior colors from config.py in miner_entity.py
- Fixed missing List import in import_standards.py to resolve type hint errors

### Entity Behavior Color Implementation

#### Completed
- Added entity behavior color constants to config.py
- Implemented behavior colors in miner_entity.py's draw method
- Updated draw_population method to display behavior colors
- Added behavior information to entity statistics display
- Applied color coding based on entity behaviors:
  - Default: Gray (180, 180, 180)
  - Feeding: Green (0, 200, 0)
  - Expanding: Blue (0, 100, 255)
  - Migrating: Orange (255, 165, 0)
  - Aggressive: Red (255, 0, 0)
- Implemented race color assignment in MinerEntity constructor:
  - Added default race colors (COLOR_RACE_1, COLOR_RACE_2, COLOR_RACE_3)
  - Made color parameter optional with automatic race color assignment
  - Added fallback for additional race IDs with random color generation
- Added status bar visualization using WINDOW_WIDTH:
  - Created draw_status_bar method that uses WINDOW_WIDTH for positioning
  - Implemented hunger and aggression bars
  - Added race and behavior indicators
  - Integrated status bar with main draw method

### Code Quality Improvements Task

Improving code quality and fixing linting issues across the codebase.

#### Completed
- Fixed mutable default argument issues in base_generator.py
- Added missing List import from typing module
- Refactored apply_cellular_automaton method for better readability
- Fixed import * issues in shop.py by replacing with explicit imports
- Removed unused imports (gc, logging) from shop.py
- Fixed unused variable warnings by renaming results → _results and metrics → _metrics
- Moved inline random imports to the top of the file
- Added missing mouse_pos variable in shop.py
- Extracted neighbor counting logic to a separate method
- Added proper typing annotations for all parameters
- Fixed import * issues in renderers.py by explicitly importing needed constants
- Fixed undefined variable issues in renderers.py (ship_size vs ship_scale)
- Extracted duplicated fade overlay rendering logic into a helper method (_render_surface_handler)
- Reordered conditional statements for better readability in notification positioning
- Documented fixes in CodeBase_Error_Fixes.md

#### Import Standards
- [X] Create import_standards.py with guidelines and helper functions
- [X] Define standard import order (stdlib, third-party, local)
- [X] Implement proper handling of optional dependencies
- [X] Define consistent approach to import grouping and spacing

#### Files to Update
- [X] src/utils/cellular_automaton_utils.py
- [X] src/utils/pattern_generator.py
- [X] src/utils/value_generator.py
- [X] src/utils/noise_generator.py
- [X] src/entities/base_generator.py
- [X] src/generators/asteroid_generator.py
- [X] src/generators/symbiote_evolution_generator.py
- [X] src/ui/renderers.py
- [X] src/ui/shop.py
- [X] src/entities/miner_entity.py

### Refactoring Tasks

2. **ProceduralGenerator** (needs refactoring)
   - Currently a standalone class in procedural_generation.py
   - Handles procedural generation for the asteroid field and symbiote races
   - Contains noise generators and statistical parameters
   - Should be refactored to inherit from BaseGenerator

3. **AsteroidGenerator** (refactored ✓)
   - Refactored to use the pattern_generator and value_generator utility modules
   - Removed duplicated code for pattern generation and value distribution
   - Now uses dependency injection for noise generation
   - Maintains all existing functionality with improved modularity

4. **SymbioteEvolutionGenerator** (refactored ✓)
   - Refactored to use the pattern_generator, value_generator, and cellular_automaton_utils utility modules
   - Removed duplicated code for pattern generation, value distribution, and cellular automaton operations
   - Now uses dependency injection for noise generation
   - Maintains all existing functionality with improved modularity

### Statistical Analysis Implementation

1. **MinerEntity** (enhanced with scipy.stats ✓)
   - Implemented scipy.stats for statistical analysis of entity distributions
   - Added a dedicated `mutate()` method using scipy.stats distributions
   - Enhanced territory analysis with statistical measures (skewness, kurtosis)
   - Improved behavior model using probability distributions
   - Added visualization of population statistics with confidence intervals
   - Used truncated normal distributions for more natural entity placement

### Refactoring Plan for Generator Classes

#### Generator Class Refactoring
- Identified three generator classes that need to be refactored: ProceduralGenerator, AsteroidGenerator, and SymbioteEvolutionGenerator
- These classes share common patterns and functionality that can benefit from inheriting from BaseGenerator
- The refactoring will maintain all existing functionality while reducing code duplication
- Next step is to implement ProceduralGenerator class that inherits from BaseGenerator

1. **ProceduralGenerator Refactoring**
   - [ ] Create new file src/generators/procedural_generator.py
   - [ ] Implement ProceduralGenerator class inheriting from BaseGenerator
   - [ ] Move relevant code from procedural_generation.py
   - [ ] Adapt to use BaseGenerator's methods where appropriate
   - [ ] Ensure all existing functionality is preserved
   - [ ] Implement BaseGenerator class that inherits from BaseEntity
   - [ ] Refactor ProceduralGenerator to inherit from BaseGenerator
   - [ ] Refactor AsteroidGenerator to inherit from BaseGenerator
   - [ ] Refactor SymbioteEvolutionGenerator to inherit from BaseGenerator

2. **SymbioteEvolutionGenerator Refactoring**
   - [ ] Create new file src/generators/symbiote_evolution_generator.py
   - [ ] Implement SymbioteEvolutionGenerator class inheriting from BaseGenerator
   - [ ] Move relevant code from procedural_generation.py
   - [ ] Adapt to use BaseGenerator's methods where appropriate
   - [ ] Ensure all existing functionality is preserved

3. **Verification and Testing**
   - [ ] Create verification script for each generator class
   - [ ] Test each generator class to ensure functionality is preserved
   - [ ] Update AsteroidField to work with the new generator classes

### Notes
- BaseGenerator is already implemented and inherits from BaseEntity
- BaseGenerator provides common functionality like seed management, noise generation, and cellular automaton logic
- The new inheritance structure will be: BaseEntity → BaseGenerator → [ProceduralGenerator, AsteroidGenerator, SymbioteEvolutionGenerator]
- All existing functionality must be preserved during the refactoring
- The refactoring will reduce code duplication and improve maintainability

#### Future Improvements
- After successfully implementing a robust dependency injection and configuration framework and integrating it with the BaseGenerator class, the current focus is on extending this pattern to other generator classes and components. The dependency injection system has been fully documented in the CodeBase_Architecture.md and CodeBase_Mapping.md files.
[ ] Implement dependency injection for other components (e.g., logging, serialization)
[ ] Create verification scripts for the refactored generator classes
[ ] Update AsteroidField to work with the new generator classes
[ ] Update documentation to reflect the new generator inheritance structure
[ ] Add comprehensive unit tests for all entity classes
[ ] Implement more robust error handling
[ ] Add more comprehensive logging
[ ] Review and optimize performance of cellular automaton logic