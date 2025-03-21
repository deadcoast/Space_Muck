"""
lazy_menu.py

This module provides an interactive ASCII menu for the lazy formatting conversion tool.
"""

# Standard library imports
import os
import pathlib
import logging
from typing import List

# Local application imports
from .convert_to_lazy_formatting import transform_file

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class InteractiveMenu:
    """Interactive ASCII menu for the lazy formatting conversion tool."""

    def __init__(self):
        """Initialize the menu with default settings."""
        self.files_to_process: List[pathlib.Path] = []
        self.options = {
            "dry_run": False,
            "verbose": False, 
            "skip_complex": False
        }
        self.exit_requested = False
        self.width = 80
        self.results = {
            "success_count": 0,
            "error_count": 0,
            "skipped_count": 0,
            "total_files": 0
        }
        self.last_message = ""

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def draw_header(self):
        """Draw the menu header."""
        print("=" * self.width)
        print(" " * ((self.width - 30) // 2) + "LAZY FORMATTING CONVERSION TOOL")
        print("=" * self.width)
        print("Convert eager logging calls (f-strings, .format) into lazy-format style.")
        print("-" * self.width)
        
        if self.last_message:
            print(f"\n>>> {self.last_message}")
            print("-" * self.width)

    def draw_file_list(self):
        """Draw the list of files to process."""
        print("\nFILES TO PROCESS:")
        if not self.files_to_process:
            print("  (No files selected)")
        else:
            for i, file in enumerate(self.files_to_process, 1):
                print(f"  {i}. {file}")
        print(f"Total: {len(self.files_to_process)} files")

    def draw_options(self):
        """Draw the available options."""
        print("\nOPTIONS:")
        print(f"  [D] Dry run: {'ON' if self.options['dry_run'] else 'OFF'}")
        print(f"  [V] Verbose output: {'ON' if self.options['verbose'] else 'OFF'}")
        print(f"  [S] Skip complex expressions: {'ON' if self.options['skip_complex'] else 'OFF'}")

    def draw_actions(self):
        """Draw the available actions."""
        print("\nACTIONS:")
        print("  [1] Add file(s)")
        print("  [2] Add directory (all .py files)")
        print("  [3] Remove file from list")
        print("  [4] Clear file list")
        print("  [5] Run conversion")
        print("  [6] Exit")

    def draw_results(self):
        """Draw conversion results if available."""
        if any(v > 0 for v in self.results.values()):
            print("\nRESULTS:")
            print(f"  Files processed successfully: {self.results['success_count']}")
            print(f"  Files with errors: {self.results['error_count']}")
            print(f"  Files skipped: {self.results['skipped_count']}")
            print(f"  Total files: {self.results['total_files']}")

    def draw_menu(self):
        """Draw the complete menu."""
        self.clear_screen()
        self.draw_header()
        self.draw_file_list()
        self.draw_options()
        self.draw_actions()
        self.draw_results()
        print("\nEnter your choice: ", end="")

    def add_file(self):
        """Add a file to the list of files to process."""
        file_path = input("Enter the path to a Python file: ").strip()
        p = pathlib.Path(file_path)
        
        if not p.exists():
            self.last_message = f"Error: File '{file_path}' does not exist."
            return
            
        if p.suffix != '.py':
            self.last_message = f"Error: '{file_path}' is not a Python file."
            return
            
        if p in self.files_to_process:
            self.last_message = f"File '{file_path}' is already in the list."
            return
            
        self.files_to_process.append(p)
        self.last_message = f"Added '{file_path}' to the list."

    def add_directory(self):
        """Add all Python files from a directory to the list."""
        if not (dir_path := input("Enter the directory path: ").strip()):
            self.last_message = "Error: Empty directory path provided."
            return
            
        p = pathlib.Path(dir_path)
        
        if not p.exists() or not p.is_dir():
            self.last_message = f"Error: '{dir_path}' is not a valid directory."
            return
            
        count_before = len(self.files_to_process)
        existing_files = set(self.files_to_process)
        
        # Add all .py files, avoiding duplicates
        for py_file in p.rglob("*.py"):
            if py_file not in existing_files:
                self.files_to_process.append(py_file)
                existing_files.add(py_file)
        
        new_count = len(self.files_to_process) - count_before
        self.last_message = f"Added {new_count} Python files from '{dir_path}'."

    def remove_file(self):
        """Remove a file from the list."""
        if not self.files_to_process:
            self.last_message = "No files to remove."
            return
            
        try:
            if not (index := int(input(f"Enter file number to remove (1-{len(self.files_to_process)}): "))):
                self.last_message = "Invalid index. Index must be positive."
            elif 1 <= index <= len(self.files_to_process):
                removed = self.files_to_process.pop(index - 1)
                self.last_message = f"Removed '{removed}' from the list."
            else:
                self.last_message = f"Invalid index. Must be between 1 and {len(self.files_to_process)}."
        except ValueError:
            self.last_message = "Invalid input. Please enter a number."

    def clear_files(self):
        """Clear the list of files to process."""
        file_count = len(self.files_to_process)
        self.files_to_process.clear()
        self.last_message = f"Removed {file_count} files from the list."

    def toggle_option(self, option: str):
        """Toggle an option."""
        if option in self.options:
            self.options[option] = not self.options[option]
            self.last_message = f"Option '{option}' is now {'ON' if self.options[option] else 'OFF'}."
        else:
            self.last_message = f"Unknown option: '{option}'."

    def run_conversion(self):
        """Run the conversion on the selected files."""
        if not self.files_to_process:
            self.last_message = "No files to process. Please add files first."
            return
            
        self.clear_screen()
        print("=" * self.width)
        print(" " * ((self.width - 20) // 2) + "RUNNING CONVERSION")
        print("=" * self.width)
        
        self.results = {
            "success_count": 0,
            "error_count": 0,
            "skipped_count": 0,
            "total_files": len(self.files_to_process)
        }
        
        for py_file in self.files_to_process:
            if self.options['verbose']:
                print(f"Transforming {py_file} ...")
                
            try:
                # Process file and update result counter based on success/failure
                if transform_file(str(py_file), in_place=(not self.options['dry_run'])):
                    self.results["success_count"] += 1
                else:
                    self.results["error_count"] += 1
            except Exception as e:
                print(f"Error processing {py_file}: {e}")
                self.results["error_count"] += 1
                
        print("\nConversion complete!")
        print(f"Files processed successfully: {self.results['success_count']}")
        print(f"Files with errors: {self.results['error_count']}")
        print(f"Total files: {self.results['total_files']}")
        
        input("\nPress Enter to return to the menu...")
        self.last_message = "Conversion completed."

    def handle_input(self):
        """Handle user input."""
        choice = input().strip().upper()
        
        if choice == '1':
            self.add_file()
        elif choice == '2':
            self.add_directory()
        elif choice == '3':
            self.remove_file()
        elif choice == '4':
            self.clear_files()
        elif choice == '5':
            self.run_conversion()
        elif choice in ('6', 'Q', 'EXIT'):
            self.exit_requested = True
            self.last_message = "Exiting..."
        elif choice == 'D':
            self.toggle_option('dry_run')
        elif choice == 'V':
            self.toggle_option('verbose')
        elif choice == 'S':
            self.toggle_option('skip_complex')
        else:
            self.last_message = f"Invalid choice: '{choice}'. Please try again."

    def run(self):
        """Run the interactive menu loop."""
        while not self.exit_requested:
            self.draw_menu()
            self.handle_input()
        
        print("Thank you for using the Lazy Formatting Conversion Tool!")


def run_interactive_menu():
    """Run the interactive menu."""
    menu = InteractiveMenu()
    try:
        menu.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("Exiting...")


if __name__ == "__main__":
    run_interactive_menu()
