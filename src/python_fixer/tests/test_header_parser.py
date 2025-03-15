#!/usr/bin/env python3
# Standard library imports
from pathlib import Path

# Third-party library imports

# Local application imports
from header_map_parser import HeaderMapParser


def main():
    """Test the header map parser implementation."""

    # Initialize parser
    parser = HeaderMapParser()

    # Test with a sample module
    test_content = """
# -----------------------------
# ASTEROID GENERATOR
# -----------------------------
#
# Parent: generators
# Dependencies: numpy, random, logging
#
# MAP: /project_root/generators
# EFFECT: Generates asteroid fields and distributions
# NAMING: Generator[Type]

def generate_asteroid():
    pass
"""

    print("Testing Header Parsing:")
    print("======================")

    # Test header parsing
    header_info = parser.parse_header(test_content)
    print("\nParsed Header Info:")
    for key, value in header_info.items():
        print(f"{key}: {value}")

    # Test import analysis
    imports = parser.analyze_imports(test_content)
    print("\nDetected Imports:")
    for imp in imports:
        print(f"  - {imp}")

    # Test header generation with updates
    updates = {
        "dependencies": imports,  # Update dependencies with actual imports
        "effect": "Generates procedural asteroid fields with resource distribution",
    }

    updated_content, changed = parser.update_header(test_content, updates)
    if changed:
        print("\nUpdated Header:")
        print(updated_content.split("import")[0])  # Show just the header part

    # Test with actual module
    print("\nTesting with actual module:")
    print("==========================")
    module_path = (
        Path(__file__).parent.parent.parent / "generators" / "asteroid_generator.py"
    )

    if module_path.exists():
        module_info = parser.extract_module_info(module_path)
        print("\nExtracted Module Info:")
        for key, value in module_info.items():
            if isinstance(value, list):
                print(f"\n{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"\n{key}: {value}")


if __name__ == "__main__":
    main()
