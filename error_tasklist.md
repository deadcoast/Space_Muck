# Space Muck Error Tasklist

This document tracks linting errors identified by flake8 that need to be addressed throughout the codebase. The approach should be to **properly integrate** functionality rather than just commenting out code or silencing warnings.

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

- [ ] **src/generators/base_generator.py** (4 issues)
  - [ ] Line 760, 763, 769, 1064: whitespace before ':'

- [ ] **src/tests/verifications/verify_asteroid_generator.py** (3 issues)
  - [ ] Line 222: multiple instances of whitespace before ':'

- [ ] **Remaining E203 errors**
  - [ ] Group remaining occurrences by file for cleanup

## 5. Module Level Import Not at Top of File (E402) - 14 occurrences

Imports not properly placed:

- [ ] **src/tests/verifications/verify_base_generator_optimizations.py** (3 issues)
  - [ ] Lines 18, 21, 22: imports not at top of file

- [ ] **src/tests/tools/test_dependency_config.py** (2 issues)
  - [ ] Lines 19, 22: imports not at top of file

- [ ] **src/tests/benchmarks/benchmark_base_generator.py** (2 issues)
  - [ ] Lines 22, 25: imports not at top of file

- [ ] **Remaining E402 errors**
  - [ ] Address remaining files with import placement issues

## 6. Unused Imports (F401) - 11 occurrences

Imports declared but never used:

- [ ] **src/config/__init__.py** (multiple issues)
  - [ ] Line 24: '.config.DEBUG_CONFIG', '.config.LOG_LEVEL', '.config.PLAYER_CONFIG', etc.

- [ ] **Remaining F401 errors**
  - [ ] Identify and address all other unused imports

## 7. Unused Local Variables (F841) - 4 occurrences

Variables assigned but never used:

- [ ] **src/ui/shop.py** (2 issues)
  - [ ] Line 467: '_results' assigned but never used
  - [ ] Line 478: '_metrics' assigned but never used

- [ ] **src/main.py** (1 issue)
  - [ ] Line 3945: '_success' assigned but never used 

- [ ] **src/ui/ui_base/ui_system_init.py** (1 issue)
  - [ ] Line 56: '_initialization_status' assigned but never used

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

## Line Too Long Errors (E501) - 167 occurrences

Lines exceeding 100 characters need to be refactored:

- [ ] **src/systems/combat_system.py** (15 issues)
  - [ ] Line 205, 256-258, 272, 356, 360, 366, 370, 414, 418, 424, 428, 505, 514, 714

- [ ] **src/ui/shop.py** (16 issues)
  - [ ] Line 107, 121, 135, 149, 163, 182, 196, 210, 224, 246, 261, 275, 289, 309, 330, 345

- [ ] **src/generators/base_generator.py** (16 issues)
  - [ ] Line 137, 276, 418, 497, 587, 1331, 1339, 1382, 1582, 1646, 1649, 2116

- [ ] **src/algorithms/economy_decision.py** (13 issues)
  - [ ] Line 73, 164, 174, 189, 190, 195, 510, 524, 673, 675, 777

- [ ] **src/algorithms/population_models.py** (8 issues)
  - [ ] Line 7, 9, 57, 61, 78, 103, 256

- [ ] **src/ui/ui_element/ui_element.py** (7 issues)
  - [ ] Line 246, 467, 619, 857, 934, 1212, 1234

- [ ] **src/utils/gpu_utils.py** (6 issues)
  - [ ] Line 224, 256, 356, 382, 624, 719

- [ ] **src/tests/verifications/verify_asteroid_generator.py** (5 issues)
  - [ ] Line 142, 194, 207, 233, 301

- [ ] **src/ui/ui_element/ascii_chain_visualizer.py** (4 issues)
  - [ ] Line 350, 355, 361, 389

- [ ] **src/tests/benchmarks/benchmark_base_generator.py** (4 issues)
  - [ ] Line 57, 71, 86, 99

- [ ] **src/utils/value_generator.py** (3 issues)
  - [ ] Line 70, 118, 142

- [ ] **src/generators/mineral_generator.py** (3 issues)
  - [ ] Line 146, 172, 197

- [ ] **Other files with E501** (67 issues)
  - [ ] Check remaining files including:
    - [ ] src/ui/ui_element/ascii_box.py
    - [ ] src/ui/ui_element/ascii_crew_management_panel.py
    - [ ] src/ui/ui_element/ascii_template_listview.py
    - [ ] src/ui/ui_element/ascii_template_manager.py
    - [ ] src/utils/cellular_automaton_utils.py
    - [ ] src/utils/dependency_injection.py

## Blank Line Contains Whitespace (W293) - 81 occurrences

Blank lines with whitespace need cleanup:

- [ ] **src/tests/tools/test_combat_system.py** (42 issues)
  - [ ] Lines 32, 35, 38, 41, 48, 51, 57, 61, 64, 72, 75, 78, 84, 87, 91, 96, 100, 106, 109, 112, 165, 168, 174, 176, 180, 190, 193, 200, 214, 217, 223, 225, 228, 231, 235, 241, 255, 259, 267, 281, 285, 289

- [ ] **src/tests/tools/test_encounter_generator.py** (12 issues)
  - [ ] Lines 40, 43, 49, 53, 57, 60, 69, 84, 88, 178, 186, 265

- [ ] **src/ui/ui_base/ascii_ui.py** (8 issues)
  - [ ] Lines 50, 57, 60, 70, 267, 270, 278, 281

- [ ] **src/ui/ui_element/ui_element.py** (5 issues)
  - [ ] Lines 324, 567, 788, 946, 1176

- [ ] **src/tests/tools/regression_tests.py** (4 issues)
  - [ ] Lines 35, 46, 56, 73

- [ ] **Other files with W293** (10 issues)
  - [ ] Check remaining files with trailing whitespace in blank lines

## Redefinition of Unused Variables (F811) - 36 occurrences

Variables redefined but not used:

- [ ] **src/generators/symbiote_evolution_generator.py** (7 issues)
  - [ ] Line 62: redefinition of unused 'AdvancedPatternAnalyzer' from line 53
  - [ ] Line 62: redefinition of unused 'ColonyMerger' from line 53
  - [ ] Line 62: redefinition of unused 'VoronoiTerritoryPartitioner' from line 53
  - [ ] Line 75-77: redefinitions of unused classes

- [ ] **src/utils/gpu_utils.py** (5 issues)
  - [ ] Line 47: redefinition of unused 'numba' from line 42
  - [ ] Line 48: redefinition of unused 'cuda' from line 43
  - [ ] Line 62, 74: redefinition of unused 'cp' from lines 41, 65

- [ ] **src/algorithms/economy_decision.py** (3 issues)
  - [ ] Line 96: redefinition of unused 'np' from line 67
  - [ ] Line 105: redefinition of unused 'pygame' from line 92

- [ ] **src/utils/gpu_dbscan.py** (2 issues)
  - [ ] Line 42: redefinition of unused 'DBSCAN' from line 31

- [ ] **src/ui/ui_helpers/animation_helper.py** (1 issue)
  - [ ] Line 14: redefinition of unused 'UIStyle' from line 11

- [ ] **Remaining F811 issues** (18 issues)
  - [ ] Address all other redefinitions of unused variables

## Whitespace Before Colon (E203) - 29 occurrences

Improper spacing in slice notation:

- [ ] **src/algorithms/pde_spatial.py** (8 issues)
  - [ ] Line 241, 256: multiple instances of whitespace before ':'
  - [ ] Line 530, 531, 537, 538: whitespace before ':'

- [ ] **src/generators/base_generator.py** (4 issues)
  - [ ] Line 760, 763, 769, 1064: whitespace before ':'

- [ ] **src/tests/verifications/verify_asteroid_generator.py** (3 issues)
  - [ ] Line 222: multiple instances of whitespace before ':'
  
- [ ] **src/utils/gpu_dbscan.py** (1 issue)
  - [ ] Line 218: whitespace before ':'

- [ ] **src/ui/ui_element/ascii_chain_visualizer.py** (1 issue)
  - [ ] Line 167: whitespace before ':'

- [ ] **Remaining E203 issues** (12 issues)
  - [ ] Address all other whitespace before colon issues

## Other Linting Issues

- [ ] **Unexpected indentation (E116)** - 2 occurrences
- [ ] **Block comment should start with '# ' (E265)** - 1 occurrence

## Implementation Strategy

1. **Prioritization**:
   - [ ] Start with high-count issues that can be fixed systematically
   - [ ] Focus on one error category at a time to apply consistent fixes

2. **File-by-File Process**:
   - [ ] Address files with multiple issues first to maximize efficiency
   - [ ] Use consistent refactoring patterns for similar issues

3. **Systematic Approach**:
   - [ ] For line length issues (E501): Break long strings, use line continuation for expressions
   - [ ] For whitespace issues (W293): Use automated tools to clean trailing whitespace 
   - [ ] For import issues (F811, F401, E402): Follow project's import organization pattern
   - [ ] For spacing issues (E203): Apply consistent slice notation formatting

4. **Testing**:
   - [ ] Run tests after each significant batch of changes
   - [ ] Re-run linter to verify fixes and identify any new issues

5. **Documentation**:
   - [ ] Update this file regularly to track progress
   - [ ] Mark completed items with [X] as they are addressed

6. **Verification**:
   - [ ] Confirm all 345 errors are addressed:
     - [ ] 167 E501 (line too long)
     - [ ] 81 W293 (blank line whitespace)
     - [ ] 36 F811 (redefinition of unused)
     - [ ] 29 E203 (whitespace before colon)
     - [ ] 14 E402 (import not at top)
     - [ ] 11 F401 (unused imports)
     - [ ] 4 F841 (unused local variables)
     - [ ] 2 E116 (unexpected indentation)
     - [ ] 1 E265 (block comment format)
