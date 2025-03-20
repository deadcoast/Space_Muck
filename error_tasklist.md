# Space Muck Error Tasklist

This document tracks linting errors that need to be addressed throughout the codebase. The approach should be to **properly integrate** functionality rather than just commenting out code or silencing warnings.

## Unused Imports (F401)

These imports are declared but never used. Each should be either properly integrated into the code with meaningful functionality or removed.

- [ ] **src/entities/miner_entity.py**
  - [ ] Line 74: `utils.gpu_utils.apply_dbscan_clustering_gpu`

- [ ] **src/ui/ui_base/ui_main.py**
  - [ ] Line 10: `config.COLOR_TEXT`
  - [ ] Line 10: `config.COLOR_BG`
  - [ ] Line 10: `config.COLOR_HIGHLIGHT`

- [ ] **src/utils/fix_ui_imports.py**
  - [ ] Line 25: `typing.Dict`
  - [ ] Line 25: `typing.Optional`
  - [ ] Line 25: `typing.Set`

- [ ] **tests/integration/test_fixers_integration.py**
  - [ ] Line 14: `typing.TYPE_CHECKING`

## Unused Variables (F841)

These variables are assigned but never used. They should be either properly utilized in the code or removed.

## Undefined Names (F821)

These names are referenced but not defined. They require proper imports or definitions.

- [ ] **src/entities/miner_entity.py**
  - [ ] Line 81: `MinerEntity`
  - [ ] Line 120: `MultiSpeciesPopulation`
  - [ ] Line 128: `DelayedGrowthPopulation`
  - [ ] Line 135: `StageStructuredPopulation`

- [ ] **src/ui/ui_base/event_system.py**
  - [ ] Lines 65-72: `EventType` (multiple instances)

- [ ] **src/ui/ui_base/ui_main.py**
  - [ ] Line 44: `Menu`
  - [ ] Line 58: `UIStyle`
  - [ ] Line 66: `FleetDisplay`
  - [ ] Line 70: `AsteroidFieldVisualizer`
  - [ ] Line 74: `SymbioteEvolutionMonitor`
  - [ ] Line 85: `MiningStatus`
  - [ ] Lines 175-183: `AnimationStyle` (multiple instances)

## Import Organization Issues (E402)

These imports are not at the top of the file and need to be reorganized.

- [ ] **src/generators/symbiote_evolution_generator.py**
  - [ ] Lines 40-62: Multiple imports not at top of file

- [ ] **src/tests/benchmarks/benchmark_base_generator.py**
  - [ ] Line 26: `entities.base_generator.BaseGenerator`

- [ ] **src/tests/verifications/verify_base_generator_optimizations.py**
  - [ ] Lines 27-28: Multiple imports not at top of file

## Syntax Errors

These files contain syntax errors that need to be fixed.

- [ ] **src/utils/fix_imports.py**
  - [ ] Lines 39-39: Expected statement (likely incomplete code)

- [ ] **src/ui/__init__.py**
  - [ ] Line 9: Expected identifier (possible typo in import path)

## Wildcard Imports (F403)

These imports use wildcards which make it difficult to determine which names are in the namespace.

- [ ] **src/__init__.py**
  - [ ] Line 11: `from .config import *`

## Next Steps

1. **Prioritization**:
   - [ ] Focus first on syntax errors as they can prevent code from running
   - [ ] Address undefined names next, as they likely cause runtime errors
   - [ ] Handle unused imports and variables last as they're primarily style/efficiency issues

2. **Implementation Approach**:
   - [ ] For each error, determine if functionality should be integrated (preferred) or code should be removed
   - [ ] Document any design decisions in code comments
   - [ ] Verify that changes don't introduce new lint errors

3. **Testing**:
   - [ ] Run tests after each significant change to ensure functionality is preserved
   - [ ] Re-run the linter to confirm errors have been addressed
