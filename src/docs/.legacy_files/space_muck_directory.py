#!/usr/bin/env python3
"""
Space_Muck Project Structure Generator

This script creates a modular directory structure for the Space_Muck game,
breaking down the monolithic RADS.py into smaller, more maintainable modules.
"""

# Standard library imports

# Third-party library imports

# Local application imports
from pathlib import Path

def create_directory_structure():
    """Create the recommended directory structure for the Space_Muck project."""

    # Project root directory (assuming we're running from project root)
    project_root = Path.cwd()
    print(f"Creating structure in: {project_root}")

    # Define the structure as nested dictionaries
    structure = {
        "src": {
            "__init__.py": "",
            "main.py": "",
            "config.py": "",
            "entities": {
                "__init__.py": "",
                "player.py": "",
                "miner_entity.py": "",
                "fleet.py": "",
            },
            "world": {
                "__init__.py": "",
                "asteroid_field.py": "",
                "procedural_generation.py": "",
            },
            "ui": {
                "__init__.py": "",
                "shop.py": "",
                "notification.py": "",
                "renderers.py": "",
                "draw_utils.py": "",
            },
            "algorithms": {
                "__init__.py": "",
                "symbiote_algorithm.py": "",
                "cellular_automaton.py": "",
            },
            "utils": {
                "__init__.py": "",
                "logging_setup.py": "",
            },
        },
        "assets": {},
        "tests": {
            "__init__.py": "",
        },
        "requirements.txt": "pygame\nnumpy\nscipy\nscikit-learn\nnetworkx\nperlin-noise",
        "README.md": "# Space_Muck\n\nA procedural generation asteroid mining game with symbiotic alien races.",
    }

    created_count = {"dirs": 0, "files": 0}

    def create_structure(base_path, structure_dict):
        """Recursively create directories and files based on the structure dictionary."""
        for name, content in structure_dict.items():
            path = base_path / name

            if isinstance(content, dict):
                # Create directory
                if not path.exists():
                    print(f"Creating directory: {path}")
                    path.mkdir(exist_ok=True)
                    created_count["dirs"] += 1
                create_structure(path, content)
            else:
                # Create file with content
                if not path.exists():
                    print(f"Creating file: {path}")
                    with open(path, "w") as f:
                        f.write(content)
                    created_count["files"] += 1

    # Create the structure
    try:
        create_structure(project_root, structure)
        print(
            f"Created {created_count['dirs']} directories and {created_count['files']} files"
        )
    except Exception as e:
        print(f"Error creating structure: {e}")
        import traceback

        traceback.print_exc()

if __name__ == "__main__":
    create_directory_structure()
    print("Directory structure creation complete!")
    print("Next steps:")
    print("1. Move the existing code into appropriate modules")
    print("2. Update imports in each file")
    print("3. Test each module individually")
