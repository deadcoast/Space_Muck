# CodeBase Scratchpad: Space Muck Test Suite Refactoring

- **NEVER CREATE MOCK TESTS OR USE MOCKING IN TESTS**
- **ALWAYS USE ACTUAL IMPLEMENTATIONS INSTEAD OF MOCKS**
- **NEVER CORRECT CODE BY COMMENTING IT OUT OR REMOVING IT**
- **WHEN FINDING MOCKS IN TESTS, REPLACE THEM WITH ACTUAL IMPLEMENTATIONS**
- **TESTS SHOULD VERIFY REAL BEHAVIOR, NOT MOCK BEHAVIOR**
- **AVOID TEST UTILITIES THAT CREATE MOCK IMPLEMENTATIONS**
- **AVOID CREATING NEW FILES WHEN ASKED TO CORRECT A FILE**
- **NEVER CREATE ANY FUNCTIONS THAT DON'T EXIST IN THE ORIGINAL CODEBASE**
- **ONLY CREATE ACTUAL INSTANCES OF REAL CLASSES AND USE THEIR EXISTING METHODS DIRECTLY**
- **NEVER ADD WRAPPER FUNCTIONS AROUND EXISTING METHODS**
- **NEVER ABSTRACT OR ALTER EXISTING FUNCTIONALITY**
- **NEVER CREATE YOUR OWN IMPLEMENTATIONS OF EXISTING METHODS**
- **ALWAYS EXAMINE THE ACTUAL CODEBASE BEFORE WRITING ANY TEST CODE**
- **ONLY ADD MINIMAL SETUP/TEARDOWN FUNCTIONALITY IF ABSOLUTELY REQUIRED**
- **NEVER MAKE ASSUMPTIONS ABOUT THE CODE - ALWAYS VERIFY FIRST**
- **TEST FILES SHOULD ONLY CREATE REAL INSTANCES OF EXISTING CLASSES**

## Current Task
Systematically repairing and improving the Space Muck test suite by eliminating mock-based testing approaches, replacing simulated behaviors with real implementations, and ensuring comprehensive test coverage.

## Project Guiding Principles
- **NO MOCKING**: Never create mock tests or use mocking in tests
- **ALWAYS USE ACTUAL IMPLEMENTATIONS INSTEAD OF MOCKS**
- **NEVER CORRECT CODE BY COMMENTING IT OUT WITH "\_"**
- **WHEN FINDING MOCKS IN TESTS, REPLACE THEM WITH ACTUAL IMPLEMENTATIONS**
- **TESTS SHOULD VERIFY REAL BEHAVIOR, NOT MOCK BEHAVIOR**
- **AVOID CREATING TEST UTILITIES THAT CREATE MOCK IMPLEMENTATIONS**

## Test Suite Status

### Current Statistics
- **Total Tests**: 228 tests
- **Passing Tests**: 192 tests (84%)
- **Failing Tests**: 20 tests (9%)
- **Skipped Tests**: 16 tests (7%)

### Project Structure
The Space Muck project follows a modular structure:

- `src/` - Main source code directory
  - `algorithms/` - Core algorithmic implementations (noise, patterns, etc.)
  - `entities/` - Game entity classes (player, asteroids, etc.)
  - `generators/` - Procedural generation systems
  - `tests/` - Comprehensive test suite
  - `ui/` - User interface components
  - `utils/` - Utility functions and helpers
- `docs/` - Project documentation
  - `CodeBase_Docs/` - Detailed codebase documentation
- `test_visualizations/` - Visual test outputs
- `visualizations/` - Generated visualization files

## 🔄 In Progress Tasks

#### Standardization and Consolidation
1. **Test File Organization**
   - ✅ Symbiote Evolution Generator: Merged verify_symbiote_evolution_generator.py into test_symbiote_evolution_generator.py
   - ✅ Asteroid Generator: Merged verify_asteroid_generator.py into test_asteroid_generator.py
   - ✅ Base Entity: Merged verify_base_entity.py into test_base_entity.py
   - Procedural Generator: Merge verify_procedural_generator.py into test_procedural_generator.py
   - Fleet: Merge verify_fleet.py into test_fleet.py
   - Player: Merge verify_player.py into test_player.py

2. **Test Utilities Standardization**
   - Create centralized test helper functions
   - Implement common assertion utilities
   - Standardize test setup and teardown procedures

#### Dependency Management
- Audit all dependencies and their usage
- Ensure consistent handling of optional dependencies:
  - GPU libraries (cupy, torch, numba)
  - Visualization libraries (matplotlib)
  - Specialized libraries (perlin-noise, scipy)
- Create or update requirements.txt with precise version specifications
- Create optional-requirements.txt for GPU acceleration dependencies
- Document environment setup process with clear installation instructions

#### Documentation Updates
- Update CodeBase_Mapping.md with new or modified files
- Update CodeBase_Error_Fixes.md with solutions to common issues
  - Document the base_generator location issue
  - Document the GAME_MAP_SIZE tuple vs int issue
  - Document the add_value_clusters parameter issue
- Ensure all public functions, classes, and modules have proper docstrings

## Current Challenges
- Terminal commands for running tests and linting are not executing properly
- Need to review and refactor remaining test files that use unittest.mock

## Next Steps

### Priority Task List
1. **Continue Test Suite Refactoring**
   - Focus on refactoring remaining test files using unittest.mock:
     1. test_procedural_generator.py (core generation logic)
     2. test_value_generator_gpu.py (specialized GPU code)
     3. test_fleet_manager.py and test_fleet.py (fleet management)
     4. test_player.py (core game entity)
     5. test_gpu_utils.py (utility functions)
   - Apply the same refactoring patterns used with test_symbiote_evolution_generator.py

2. **Run Comprehensive Tests After Each Refactoring**
   - Ensure all tests pass after each test file refactoring
   - Check for regression issues
   - Maintain detailed logs of test coverage and results
   - Focus on one test file at a time for manageable changes

3. **Address Verification and Tool Test Files**
   - After main test files, review verification files
   - Then refactor tool test files

4. **Linting and Code Quality**
   - Address potential linting issues in all files
   - Check for unused imports
   - Verify proper type hints
   - Ensure consistent code style
   - Rename poorly named functions like _extracted_from_evolve_26

5. **Documentation Updates**
   - Update documentation with any new patterns identified
   - Ensure all docstrings are complete and accurate

## Testing Environment

### Testing Commands
```bash
# Run all tests
python -m pytest src/tests

# Run specific test categories
python -m pytest src/tests/test_base_generator.py
python -m pytest src/tests/test_gpu_*.py

# Run with verbose output
python -m pytest src/tests -v

# Run with coverage report
python -m pytest src/tests --cov=src

# Run specific import tests
python -m src.tests.test_imports

# Run tests with specific environment settings
PYTHONPATH=. python src/tests/test_imports.py

# Run specific test file with unittest
python -m unittest src/tests/test_imports.py

# Run tests with coverage report
python -m coverage run -m unittest discover src/tests
python -m coverage report
```

### Linting Commands
```bash
# Install linting tools
pip install flake8 pylint ruff sourcery-cli

# Run linting tools
flake8 src/
pylint src/
ruff check src/
sourcery login
sourcery run src/
```