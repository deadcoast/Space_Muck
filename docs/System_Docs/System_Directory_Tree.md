# Directory Tree


```
/Users/deadcoast/PycharmProjects/Space_Muck
├── assets/
├── benchmark_results/
│   └── benchmark_log.txt
├── docs/
│   ├── .legacy_files/
│   │   ├── game_design/
│   │   │   ├── algorithm_design.md
│   │   │   └── game_ideaology.md
│   │   ├── PhaseOne_CodeBase_Docs/
│   │   │   ├── CodeBase_Architecture.md
│   │   │   ├── CodeBase_Error_Fixes.md
│   │   │   ├── CodeBase_Mapping.md
│   │   │   └── CodeBase_Scratchpad.md
│   │   ├── PhaseTwo_CodeBase_Docs/
│   │   │   ├── Categories/
│   │   │   │   ├── CombatSystem_Design.md
│   │   │   │   ├── GPU_Acceleration_Guide.md
│   │   │   │   └── GPU_Hardware_Compatibility.md
│   │   │   ├── CodeBase_Architecture.md
│   │   │   ├── CodeBase_Error_Fixes.md
│   │   │   ├── CodeBase_Mapping.md
│   │   │   └── CodeBase_Scratchpad.md
│   │   ├── RADS.py
│   │   ├── space_muck_directory.py
│   │   └── symbiote_algorithm.py
│   └── System_Docs/
│       ├── System_Restructuring/
│       │   ├── System_Architecture_Diagrams.md
│       │   ├── System_Architecture_Integration.md
│       │   ├── System_Restructuring_Plan.md
│       │   └── System_Scratchpad.md
│       ├── System_Development_History.md
│       ├── System_Directory_Tree.md
│       ├── Windsurf_System_Architecture.md
│       └── Windsurf_System_Integration.md
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── cellular_automaton.py
│   │   └── symbiote_algorithm.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── dependency_settings.py
│   ├── demos/
│   │   ├── __init__.py
│   │   └── symbiote_evolution_demo.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── base_entity.py
│   │   ├── enemy_ship.py
│   │   ├── fleet.py
│   │   ├── miner_entity.py
│   │   └── player.py
│   ├── examples/
│   │   ├── __init__.py
│   │   └── dependency_config_example.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── asteroid_field.py
│   │   ├── asteroid_generator.py
│   │   ├── base_generator.py
│   │   ├── encounter_generator.py
│   │   ├── procedural_generator.py
│   │   └── symbiote_evolution_generator.py
│   ├── systems/
│   │   ├── __init__.py
│   │   ├── combat_system.py
│   │   ├── encounter_generator.py
│   │   ├── fleet_manager.py
│   │   └── trading_system.py
│   ├── tests/
│   │   ├── benchmarks/
│   │   │   ├── __init__.py
│   │   │   ├── benchmark_base_generator.py
│   │   │   ├── benchmark_comprehensive_gpu.py
│   │   │   ├── benchmark_gpu_acceleration.py
│   │   │   ├── benchmark_gpu_noise_generation.py
│   │   │   ├── benchmark_parallel_processing.py
│   │   │   ├── benchmark_procedural_generation.py
│   │   │   └── README.md
│   │   ├── tools/
│   │   │   ├── check_linting.py
│   │   │   ├── integration_tests.py
│   │   │   ├── mock_data_test.py
│   │   │   ├── performance_tests.py
│   │   │   ├── README.md
│   │   │   ├── regression_tests.py
│   │   │   ├── test_combat_system.py
│   │   │   ├── test_dependency_config.py
│   │   │   └── test_encounter_generator.py
│   │   ├── verifications/
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   ├── verify_asteroid_generator.py
│   │   │   ├── verify_base_entity.py
│   │   │   ├── verify_base_generator_optimizations.py
│   │   │   ├── verify_fleet.py
│   │   │   ├── verify_player.py
│   │   │   ├── verify_procedural_generator.py
│   │   │   └── verify_symbiote_evolution_generator.py
│   │   ├── __init__.py
│   │   ├── directory_tree_simple.py
│   │   ├── directory_tree_tests.md
│   │   ├── README.md
│   │   ├── rich_directory_tree_tests.md
│   │   ├── test_asteroid_generator.py
│   │   ├── test_base_entity.py
│   │   ├── test_base_generator.py
│   │   ├── test_cellular_automaton_utils.py
│   │   ├── test_fleet.py
│   │   ├── test_fleet_manager.py
│   │   ├── test_gpu_clustering.py
│   │   ├── test_gpu_utils.py
│   │   ├── test_imports.py
│   │   ├── test_miner_entity.py
│   │   ├── test_pattern_generator.py
│   │   ├── test_player.py
│   │   ├── test_procedural_generator.py
│   │   ├── test_symbiote_evolution_generator.py
│   │   ├── test_trading_system.py
│   │   ├── test_value_generator.py
│   │   ├── test_value_generator_gpu.py
│   │   └── test_visualization.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── draw_utils.py
│   │   ├── notification.py
│   │   ├── renderers.py
│   │   └── shop.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── cellular_automaton_utils.py
│   │   ├── dependency_config.py
│   │   ├── dependency_injection.py
│   │   ├── gpu_utils.py
│   │   ├── import_standards.py
│   │   ├── logging_setup.py
│   │   ├── noise_generator.py
│   │   ├── noise_generator_gpu.py
│   │   ├── pattern_generator.py
│   │   ├── value_generator.py
│   │   ├── value_generator_gpu.py
│   │   └── visualization.py
│   ├── __init__.py
│   └── main.py
├── visualizations/
│   ├── asteroid_generator_test.png
│   ├── ca_benchmark_results.png
│   ├── clustering_benchmark_results.png
│   ├── symbiote_evolution_test.png
│   └── symbiote_mineral_impact.png
├── .sourcery.yaml
├── .windsurfrules
├── optional-requirements.txt
├── pylintrc
├── README.md
├── requirements.txt
├── rich_directory_tree.py
├── run_tests.sh
└── system_architechture_integration.md

```

# Notes
