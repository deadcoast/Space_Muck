#!/usr/bin/env python3

import logging
from pathlib import Path
from project_analysis import ProjectAnalyzer

def print_section(title: str, underline_char: str = '=') -> None:
    """Print a section title with consistent formatting."""
    print(f"\n{title}")
    print(underline_char * len(title))

def print_structure_summary(structure: dict) -> None:
    """Print project structure summary.
    
    Args:
        structure: Dictionary containing project structure information
    """
    print_section("Project Structure Summary")
    total_modules = len(structure['modules'])
    print(f"Total Python modules analyzed: {total_modules}")


def print_enhancement_analysis(enhancements: list) -> None:
    """Print enhancement analysis results.
    
    Args:
        enhancements: List of enhancement suggestions per module
    """
    print_section("Enhancement Opportunities")
    if not enhancements:
        print("No immediate enhancement needs identified")
        return
        
    for enhancement in sorted(enhancements, key=lambda x: len(x['suggestions']), reverse=True):
        module = enhancement['module']
        suggestions = enhancement['suggestions']
        print(f"\n{module} ({len(suggestions)} suggestions):")
        for suggestion in sorted(suggestions):
            print(f"  - {suggestion}")


def print_dependency_analysis(dependencies: dict) -> None:
    """Print dependency analysis results.
    
    Args:
        dependencies: Dictionary containing dependency analysis results
    """
    print_section("Dependency Analysis")
    
    # Primary dependencies
    if primary_deps := dependencies['primary']:
        print("\nHigh-Impact Modules (>2 dependents):")
        for dep in sorted(primary_deps):
            print(f"  - {dep}")
    else:
        print("No high-impact modules found")
        
    # Cyclic dependencies
    if cycles := dependencies['cycles']:
        print("\nCyclic Dependencies (potential refactoring targets):")
        for cycle in cycles:
            print(f"  - {' -> '.join(cycle)}")
    else:
        print("\nNo cyclic dependencies found - Good job!")

def analyze_space_muck() -> None:
    """Analyze the Space Muck project structure and dependencies."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get the project src directory path
    project_src = Path(__file__).parent.parent.parent
    
    try:
        # Create analyzer instance
        analyzer = ProjectAnalyzer(str(project_src))
        analysis = analyzer.analyze_project()
        
        # Print analysis results
        print_structure_summary(analysis['structure'])
        
        # Print dependency information
        print_dependency_analysis(analysis['dependencies'])
            
        # Print enhancement suggestions
        print_enhancement_analysis(analysis['enhancements'])
                
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

def main() -> None:
    """Main entry point."""
    try:
        analyze_space_muck()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
