#!/usr/bin/env python3

from project_map_parser import ProjectMapParser

def main():
    """Test the project map parser implementation."""
    # Sample project map content
    test_map = """
Project Structure:
    /space_muck
    ├── main.py
    ├── generators/
    │   ├── asteroid_generator.py
    │   ├── asteroid_field.py
    │   └── symbiote_evolution_generator.py
    ├── systems/
    │   ├── fleet_manager.py
    │   ├── trading_system.py
    │   └── exploration_manager.py
    └── ui/
        ├── ascii_ui.py
        └── renderers.py

Enhancement Targets:
    1. asteroid_generator.py: Add resource distribution patterns
    2. fleet_manager.py: Implement advanced formation logic
    3. trading_system.py: Add dynamic pricing model
    4. ascii_ui.py: Improve rendering performance

Dependencies Found:
    Primary:
    - numpy
    - networkx
    - pygame
    - scipy

    Secondary:
    - logging
    - pathlib
    - typing
    - random
    """
    
    print("Testing Project Map Parser:")
    print("==========================")
    
    # Initialize parser with test map
    parser = ProjectMapParser(test_map)
    
    # Test structure parsing
    print("\nModule Paths:")
    for path in parser.get_module_paths():
        print(f"  - {path}")
    
    # Test enhancement parsing
    print("\nEnhancements by Module:")
    test_modules = ['asteroid_generator.py', 'fleet_manager.py']
    for module in test_modules:
        enhancements = parser.get_enhancements_for_module(module)
        print(f"\n{module}:")
        for enhancement in enhancements:
            print(f"  - {enhancement}")
    
    # Test dependency parsing
    print("\nDependencies:")
    for dep_type in ['primary', 'secondary']:
        deps = parser.get_dependencies_by_type(dep_type)
        print(f"\n{dep_type.title()}:")
        for dep in deps:
            print(f"  - {dep}")
    
    # Test map generation
    print("\nGenerated Map:")
    print("==============")
    print(parser.generate_map())

if __name__ == "__main__":
    main()
