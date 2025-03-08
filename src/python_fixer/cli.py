#!/usr/bin/env python3
"""
Main entry point for the Python Import Fixer tool.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from python_fixer.core.analyzer import ProjectAnalyzer
from python_fixer.logging.structured import StructuredLogger
from python_fixer.web.dashboard import run_dashboard


class CLIError(Exception):
    """Base exception for CLI errors."""

    pass


class ValidationError(CLIError):
    """Raised when input validation fails."""

    pass


class ProjectError(CLIError):
    """Raised when project-related operations fail."""

    pass


def validate_project_path(path: Path) -> None:
    """Validate the project path.

    Args:
        path: Path to validate

    Raises:
        ValidationError: If path validation fails
    """
    if not path.exists():
        raise ValidationError(f"Project path '{path}' does not exist")

    if not path.is_dir():
        raise ValidationError(f"'{path}' is not a directory")

    # Check if it's a Python project (has .py files)
    py_files = list(path.rglob("*.py"))
    if not py_files:
        raise ValidationError(f"No Python files found in '{path}'")


def validate_port(port: int) -> None:
    """Validate the port number.

    Args:
        port: Port number to validate

    Raises:
        ValidationError: If port validation fails
    """
    if not 1 <= port <= 65535:
        raise ValidationError(
            f"Invalid port number: {port}. Must be between 1 and 65535"
        )

    if port < 1024 and os.geteuid() != 0:
        raise ValidationError(
            f"Port {port} requires root privileges. Please use a port number >= 1024"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix Python import and class structure issues in your project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Initialize a new project
  python -m python_fixer init /path/to/project
  
  # Analyze imports in a project
  python -m python_fixer analyze /path/to/project
  
  # Fix imports interactively
  python -m python_fixer fix /path/to/project
  
  # Launch the web dashboard
  python -m python_fixer dashboard /path/to/project --port 8000
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a project")
    init_parser.add_argument("project_path", type=Path, help="Path to the project")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze project imports")
    analyze_parser.add_argument("project_path", type=Path, help="Path to the project")

    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Fix import issues")
    fix_parser.add_argument("project_path", type=Path, help="Path to the project")
    fix_parser.add_argument(
        "--mode",
        choices=["interactive", "automatic"],
        default="interactive",
        help="Fix mode (default: interactive)",
    )

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dashboard_parser.add_argument("project_path", type=Path, help="Path to the project")
    dashboard_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Dashboard host (default: localhost)",
    )
    dashboard_parser.add_argument(
        "--port", type=int, default=8000, help="Dashboard port (default: 8000)"
    )
    dashboard_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    # Global options
    for p in [init_parser, analyze_parser, fix_parser, dashboard_parser]:
        p.add_argument(
            "--log-file",
            type=Path,
            default=Path("import_fixes.log"),
            help="Path for the log file (default: import_fixes.log)",
        )
        p.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose output"
        )

    return parser.parse_args()


def print_analysis_summary(results: dict) -> None:
    """Print a summary of the analysis results."""
    print("\n\033[92m✓ Analysis complete\033[0m")
    print(
        f"\033[96m  - Files analyzed: {len(results.get('structure', {}).get('modules', []))}"
    )
    print(
        f"  - Circular dependencies: {len(results.get('dependencies', {}).get('cycles', []))}"
    )
    print(f"  - Enhancement suggestions: {len(results.get('enhancements', []))}\033[0m")


def print_fixes_summary(fixes: dict) -> None:
    """Print a summary of the applied fixes."""
    print("\n\033[92m✓ Fixes applied successfully\033[0m")
    print(f"\033[96m  - Files modified: {fixes.get('files_modified', 0)}")
    print(f"  - Imports fixed: {fixes.get('imports_fixed', 0)}\033[0m")


def print_cli_header() -> None:
    """Print the CLI header with buttons."""
    print("""
┌─────────────────── [ ERROR-HANDLER CLI ] ────────────────────┐
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│  │ RUN │ │ VAL │ │ LOG │ │ CFG │ │ FIX │ │ SYS │ │ HLP │     │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘     │
│                                                              │""")

def print_error_section(error_msg: str, details: List[str] = None) -> None:
    """Print the error output section."""
    print("""
│  ┏━━━━━━━━━━━━━━━━━━ ERROR OUTPUT ━━━━━━━━━━━━━━━━━━━━━━┓   │
│  ┃                                                       ┃   │
│  ┃  ⚠ {:<52} ┃   │""".format(error_msg))
    
    if details:
        for detail in details:
            print("│  ┃  ✗ {:<52} ┃   │".format(detail))
    
    print("""
│  ┃                                                       ┃   │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   │
│                                                              │""")

def print_suggestions(suggestions: List[str]) -> None:
    """Print the suggestions section."""
    print("""
│  ┏━━━━━━━━━━━━━━━━━━ SUGGESTIONS ━━━━━━━━━━━━━━━━━━━━━━┓   │
│  ┃                                                       ┃   │""")
    
    for suggestion in suggestions:
        print("│  ┃  ▶ {:<52} ┃   │".format(suggestion))
    
    print("""
│  ┃                                                       ┃   │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   │
│                                                              │
│  System: Linux | User: admin | Version: 2.1.3 | Log: Active  │
└──────────────────────────────────────────────────────────────┘""")

def print_command_section(command: str) -> None:
    """Print the command input section."""
    print("""
│  ┏━━━━━━━━━━━━━━━━━━ COMMAND INPUT ━━━━━━━━━━━━━━━━━━━━━━┓   │
│  ┃                                                       ┃   │
│  ┃  {:<54} ┃   │
│  ┃                                                       ┃   │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   │
│                                                              │""".format(command))

def print_usage_examples() -> None:
    """Print example usage information."""
    print("\nExample usage:")
    print("  python -m python_fixer init /path/to/project")
    print("  python -m python_fixer analyze /path/to/project")
    print("  python -m python_fixer fix /path/to/project --mode interactive")
    print("  python -m python_fixer dashboard /path/to/project --port 8000")


def handle_error(
    error: Exception, args: argparse.Namespace = None, logger: StructuredLogger = None
) -> int:
    """Handle an error and return the appropriate exit code.

    Args:
        error: The error to handle
        args: Optional parsed command line arguments
        logger: Optional logger instance

    Returns:
        Exit code to return from the program
    """
    error_msg = str(error)
    command = f"$ {' '.join(sys.argv)}"
    
    print_cli_header()
    print_command_section(command)
    
    details = []
    if isinstance(error, ValidationError):
        if 'project' in error_msg.lower():
            details.append("Directory contains no Python files")
        elif 'port' in error_msg.lower():
            details.append("Port number is invalid or requires root privileges")
    
    print_error_section(error_msg, details)
    
    suggestions = [
        f"Try: {sys.argv[0]} validate --path {args.project_path if args else './project'}",
        f"Try: {sys.argv[0]} --help for more information"
    ]
    print_suggestions(suggestions)
    
    if args and args.verbose:
        import traceback
        traceback.print_exc()
    
    if logger:
        try:
            logger.error("Command failed", exc_info=error)
        except Exception as log_error:
            print("\033[93mWarning: Unable to write to log file: " + str(log_error) + "\033[0m")
    
    return 1


def print_dashboard_info(host: str, port: int, reload: bool) -> None:
    """Print information about the dashboard."""
    print(f"\033[96m  - URL: http://{host}:{port}")
    print(f"  - Auto-reload: {'enabled' if reload else 'disabled'}\033[0m")


def main():
    args = None
    logger = None

    try:
        # Parse and validate arguments
        args = parse_args()
        validate_project_path(args.project_path)
        if args.command == "dashboard":
            validate_port(args.port)

        # Setup logging
        logger = StructuredLogger(
            name="python_fixer",
            log_file=args.log_file,
            level="DEBUG" if args.verbose else "INFO",
        )
        if args.command == "init":
            print(f"\033[94mInitializing project at {args.project_path}...\033[0m")
            analyzer = ProjectAnalyzer(args.project_path)
            analyzer.initialize_project()
            print("\033[92m✓ Project initialized successfully\033[0m")

        elif args.command == "analyze":
            print(f"\033[94mAnalyzing project at {args.project_path}...\033[0m")
            analyzer = ProjectAnalyzer(args.project_path)
            results = analyzer.analyze_project()
            logger.info("Analysis complete", extra={"metrics": results})

            print_analysis_summary(results)

        elif args.command == "fix":
            print(
                f"\033[94mFixing imports in {args.project_path} (mode: {args.mode})...\033[0m"
            )
            analyzer = ProjectAnalyzer(args.project_path)
            fixes = analyzer.fix_project(mode=args.mode)
            logger.info("Fixes complete", extra={"metrics": fixes})

            print_fixes_summary(fixes)

        elif args.command == "dashboard":
            print(f"\033[94mLaunching dashboard for {args.project_path}...\033[0m")
            print_dashboard_info(args.host, args.port, args.reload)

            try:
                run_dashboard(
                    project_path=args.project_path,
                    host=args.host,
                    port=args.port,
                    reload=args.reload,
                    log_level="DEBUG" if args.verbose else "INFO",
                )
            except KeyboardInterrupt:
                print("\n\033[93m⚠ Dashboard stopped by user\033[0m")
            except Exception as e:
                print(f"\n\033[91m✗ Failed to start dashboard: {e}\033[0m")
                logger.error("Dashboard failed", exc_info=e)
                return 1

        return 0

    except ValidationError as e:
        return handle_error(e, args, logger)
    except KeyboardInterrupt:
        print("\n\033[93m⚠ Operation cancelled by user\033[0m")
        return 130
    except Exception as e:
        return handle_error(e, args, logger)


if __name__ == "__main__":
    sys.exit(main())
