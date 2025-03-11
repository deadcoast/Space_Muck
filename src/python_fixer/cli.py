#!/usr/bin/env python3
"""
Main entry point for the Python Import Fixer tool.
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, List

# Import core dependencies
from python_fixer.core.analyzer import ProjectAnalyzer
from python_fixer.core.signals import SignalManager
from python_fixer.logging.structured import StructuredLogger

# Optional web dashboard import
run_dashboard = None
if importlib.util.find_spec("python_fixer.web.dashboard") is not None:
    from python_fixer.web.dashboard import run_dashboard

# Initialize signal manager and logger
signal_manager = SignalManager()
logger = StructuredLogger.get_logger(__name__)


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


def verify_imports() -> None:
    """Verify that all required modules are available.

    Raises:
        ImportError: If any required module is not available
    """
    required_modules = [
        "python_fixer.core.analyzer",
        "python_fixer.core.signals",
        "python_fixer.logging.structured",
    ]

    for module in required_modules:
        if not importlib.util.find_spec(module):
            raise ImportError(f"Required module not found: {module}")


def print_import_paths() -> None:
    """Print the current Python import paths."""
    print("\n\033[96mPython Import Paths:\033[0m")
    for path in sys.path:
        if os.path.exists(path):
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (not found)")
    print()


def parse_args():
    verify_imports()
    print_import_paths()
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
    print("Adding subparsers...")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = _extracted_from_parse_args_26(
        "Setting up init command...",
        subparsers,
        "init",
        "Initialize a project",
    )
    analyze_parser = _extracted_from_parse_args_26(
        "Setting up analyze command...",
        subparsers,
        "analyze",
        "Analyze project imports",
    )
    fix_parser = _extracted_from_parse_args_26(
        "Setting up fix command...", subparsers, "fix", "Fix import issues"
    )
    fix_parser.add_argument(
        "--mode",
        choices=["interactive", "automatic"],
        default="interactive",
        help="Fix mode (default: interactive)",
    )

    dashboard_parser = _extracted_from_parse_args_26(
        "Setting up dashboard command...",
        subparsers,
        "dashboard",
        "Launch web dashboard",
    )
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

    print("Adding global options...")
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

    print("Parsing arguments...")
    try:
        args = parser.parse_args()
        print(f"Parsed arguments: {args}")
        return args
    except Exception as e:
        print(f"Error parsing arguments: {str(e)}")
        raise


# TODO Rename this here and in `parse_args`
def _extracted_from_parse_args_26(arg0, subparsers, arg2, help):
    print(arg0)
    result = subparsers.add_parser(arg2, help=help)
    result.add_argument("project_path", type=Path, help="Path to the project")

    return result


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
    print(
        """
┌─────────────────── [ ERROR-HANDLER CLI ] ────────────────────┐
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│  │ RUN │ │ VAL │ │ LOG │ │ CFG │ │ FIX │ │ SYS │ │ HLP │     │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘     │
│                                                              │"""
    )


def print_error_section(error_msg: str, details: List[str] = None) -> None:
    """Print the error output section."""
    print(
        """
│  ┏━━━━━━━━━━━━━━━━━━ ERROR OUTPUT ━━━━━━━━━━━━━━━━━━━━━━┓   │
│  ┃                                                       ┃   │
│  ┃  ⚠ {:<52} ┃   │""".format(
            error_msg
        )
    )

    if details:
        for detail in details:
            print("│  ┃  ✗ {:<52} ┃   │".format(detail))

    print(
        """
│  ┃                                                       ┃   │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   │
│                                                              │"""
    )


def print_suggestions(suggestions: List[str]) -> None:
    """Print the suggestions section."""
    print(
        """
│  ┏━━━━━━━━━━━━━━━━━━ SUGGESTIONS ━━━━━━━━━━━━━━━━━━━━━━┓   │
│  ┃                                                       ┃   │"""
    )

    for suggestion in suggestions:
        print("│  ┃  ▶ {:<52} ┃   │".format(suggestion))

    print(
        """
│  ┃                                                       ┃   │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   │
│                                                              │
│  System: Linux | User: admin | Version: 2.1.3 | Log: Active  │
└──────────────────────────────────────────────────────────────┘"""
    )


def print_command_section(command: str) -> None:
    """Print the command input section."""
    print(
        """
│  ┏━━━━━━━━━━━━━━━━━━ COMMAND INPUT ━━━━━━━━━━━━━━━━━━━━━━┓   │
│  ┃                                                       ┃   │
│  ┃  {:<54} ┃                                             ┃   │
│  ┃                                                       ┃   │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   │
│                                                              │""".format(
            command
        )
    )


def print_usage_examples() -> None:
    """Print example usage information."""
    print("\nExample usage:")
    print("  python -m python_fixer init /path/to/project")
    print("  python -m python_fixer analyze /path/to/project")
    print("  python -m python_fixer fix /path/to/project --mode interactive")
    print("  python -m python_fixer dashboard /path/to/project --port 8000")


def handle_error(
    error: Exception, args: argparse.Namespace = None, logger: Any = None
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
        if "project" in error_msg.lower():
            details.append("Directory contains no Python files")
        elif "port" in error_msg.lower():
            details.append("Port number is invalid or requires root privileges")

    print_error_section(error_msg, details)

    suggestions = [
        f"Try: {sys.argv[0]} validate --path {args.project_path if args else './project'}",
        f"Try: {sys.argv[0]} --help for more information",
    ]
    print_suggestions(suggestions)

    if args and args.verbose:
        import traceback

        traceback.print_exc()

    if logger:
        try:
            logger.error("Command failed", exc_info=error)
        except Exception as log_error:
            print(
                "\033[93mWarning: Unable to write to log file: "
                + str(log_error)
                + "\033[0m"
            )

    return 1


def print_dashboard_info(host: str, port: int, reload: bool) -> None:
    """Print information about the dashboard."""
    print(f"\033[96m  - URL: http://{host}:{port}")
    print(f"  - Auto-reload: {'enabled' if reload else 'disabled'}\033[0m")


def setup_args_and_logging():
    """Parse and validate arguments, setup logging.

    Returns:
        Tuple[argparse.Namespace, StructuredLogger]: Parsed arguments and logger

    Raises:
        ImportError: If required modules are not available
        ValidationError: If argument validation fails
    """
    print("Setting up arguments and logging...")
    try:
        # Verify imports first
        verify_imports()

        # Show import paths for debugging
        if "--verbose" in sys.argv:
            print_import_paths()

        return _extracted_from_setup_args_and_logging_9()
    except ImportError as e:
        print(f"\033[91mError: Required module not available - {str(e)}\033[0m")
        raise
    except Exception as e:
        print(f"Error in setup: {str(e)}")
        raise


# TODO Rename this here and in `setup_args_and_logging`
def parse_and_validate_args():
    """Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Parsed and validated arguments

    Raises:
        ValidationError: If validation of arguments fails
        Exception: If argument parsing fails
    """
    print("Parsing arguments...")
    args = parse_args()
    print(f"Command: {args.command}")

    print("Converting project path...")
    args.project_path = Path(args.project_path)
    print(f"Project path: {args.project_path}")

    print("Validating project path...")
    validate_project_path(args.project_path)
    print("Project path validation successful")

    if args.command == "dashboard":
        print("Validating port...")
        validate_port(args.port)
        print("Port validation successful")

    return args


def _extracted_from_setup_args_and_logging_9():
    """Parse arguments and set up logging with proper error handling.

    Returns:
        Tuple[argparse.Namespace, StructuredLogger]: Parsed arguments and logger

    Raises:
        ValidationError: If validation of arguments fails
        Exception: If argument parsing or logger setup fails
    """
    logger = None
    try:
        args = parse_and_validate_args()

        print("Setting up logger...")
        log_file = Path(args.log_file) if args.log_file else None
        print(f"Log file: {log_file}")

        logger = StructuredLogger(
            name="python_fixer",
            log_file=log_file,
            level="DEBUG" if args.verbose else "INFO",
        )
        print("Logger setup successful")

        # Log successful setup
        logger.info(
            "Arguments and logging setup complete",
            extra={
                "command": args.command,
                "project_path": str(args.project_path),
                "log_file": str(log_file) if log_file else None,
                "verbose": args.verbose,
            },
        )

        return args, logger

    except ValidationError as e:
        if logger:
            logger.error("Validation error during setup", exc_info=e)
        raise
    except Exception as e:
        if logger:
            logger.error("Unexpected error during setup", exc_info=e)
        raise


def main():
    args = None
    logger = None
    analyzer = None
    dashboard_process = None

    def cleanup():
        """Cleanup function for graceful shutdown."""
        nonlocal analyzer, dashboard_process, logger
        try:
            if analyzer:
                logger.info("Cleaning up analyzer resources...")
                analyzer.cleanup()
            if dashboard_process:
                logger.info("Shutting down dashboard server...")
                dashboard_process.terminate()
        except Exception as e:
            if logger:
                logger.error(f"Error during cleanup: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Cleanup error details:")

    with signal_manager.handler(cleanup):
        try:
            # Debug import paths
            print("\033[94mDebug: Import Resolution\033[0m")
            print("Python executable:", sys.executable)
            print("Python version:", sys.version)
            print("Python path:")
            for p in sys.path:
                print(f"  {p}")

            print("\nAttempting imports...")
            import python_fixer

            print("\033[92m✓ Successfully imported python_fixer\033[0m")
            print("python_fixer location:", python_fixer.__file__)

            print("\nSetting up arguments and logging...")
            args, logger = setup_args_and_logging()
            print("Arguments and logging setup complete")

            print("Creating default config...")
            config = {
                "verbose": args.verbose,
                "enable_caching": True,
                "cache_dir": ".python_fixer_cache",
                "max_workers": 4,
                "enable_type_checking": True,
                "enable_complexity_analysis": True,
            }
            print(f"Config created: {config}")

            if args.command == "init":
                print(f"\033[94mInitializing project at {args.project_path}...\033[0m")
                try:
                    initialize_project(args, config)
                except (ValidationError, ProjectError) as e:
                    print(f"\033[91m✗ {str(e)}\033[0m")
                    logger.error("Initialization failed", exc_info=e)
                    return 1

            elif args.command == "analyze":
                print(f"\033[94mAnalyzing project at {args.project_path}...\033[0m")
                analyzer = ProjectAnalyzer(
                    args.project_path, config=config, backup=True
                )
                results = analyzer.analyze_project()
                logger.info("Analysis complete", extra={"metrics": results})
                print_analysis_summary(results)

            elif args.command == "fix":
                print(
                    f"\033[94mFixing imports in {args.project_path} (mode: {args.mode})...\033[0m"
                )
                analyzer = ProjectAnalyzer(
                    args.project_path, config=config, backup=True
                )
                fixes = analyzer.fix_project(mode=args.mode)
                logger.info("Fixes complete", extra={"metrics": fixes})
                print_fixes_summary(fixes)

            elif args.command == "dashboard":
                if not run_dashboard:
                    raise ValidationError(
                        "Web dashboard not available. Please install optional dependencies:"
                        "\n  pip install python-fixer[web]"
                    )

                print(f"\033[94mLaunching dashboard for {args.project_path}...\033[0m")
                print_dashboard_info(args.host, args.port, args.reload)

                dashboard_process = run_dashboard(
                    project_path=args.project_path,
                    host=args.host,
                    port=args.port,
                    reload=args.reload,
                    log_level="DEBUG" if args.verbose else "INFO",
                )

            return 0

        except ValidationError as e:
            return handle_error(e, args, logger)
        except KeyboardInterrupt:
            logger.warning("Operation cancelled by user")
            return 130
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.exception("Error details:")
            return handle_error(e, args, logger)


# TODO Rename this here and in `main`
def initialize_project(args, config):
    """Initialize a project with proper error handling.

    Args:
        args: Command line arguments
        config: Configuration dictionary

    Raises:
        ValidationError: If project validation fails
        ProjectError: If project initialization fails
    """
    analyzer = None
    logger = StructuredLogger.get_logger(__name__)

    def cleanup():
        """Cleanup function for graceful shutdown."""
        try:
            if analyzer:
                logger.info("Cleaning up analyzer resources...")
                analyzer.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.exception("Cleanup error details:")

    with signal_manager.handler(cleanup):
        try:
            logger.info("Starting project initialization")
            print("\033[94mValidating project path...\033[0m")
            try:
                validate_project_path(args.project_path)
                print("\033[92m✓ Project path validation successful\033[0m")
                logger.info("Project path validation successful")
            except ValidationError as e:
                print(f"\033[91m✗ Project path validation failed: {str(e)}\033[0m")
                logger.error("Project path validation failed", exc_info=e)
                raise

            print("\033[94mCreating ProjectAnalyzer instance...\033[0m")
            try:
                print(f"  Project path: {args.project_path}")
                print(f"  Config: {config}")
                analyzer = ProjectAnalyzer(
                    args.project_path, config=config, backup=True
                )
                print("\033[92m✓ ProjectAnalyzer instance created successfully\033[0m")
                logger.info("ProjectAnalyzer instance created successfully")
            except Exception as e:
                print(
                    f"\033[91m✗ Failed to create ProjectAnalyzer instance: {str(e)}\033[0m"
                )
                logger.error("Failed to create ProjectAnalyzer instance", exc_info=e)
                raise ProjectError(f"Failed to create analyzer: {str(e)}") from e

            print("\033[94mInitializing project...\033[0m")
            try:
                analyzer.initialize_project()
                print("\033[92m✓ Project initialized successfully\033[0m")
                logger.info("Project initialized successfully")
            except Exception as e:
                print(f"\033[91m✗ Failed to initialize project: {str(e)}\033[0m")
                logger.error("Failed to initialize project", exc_info=e)
                raise ProjectError(f"Failed to initialize project: {str(e)}") from e

        except ValidationError:
            raise
        except Exception as e:
            print(f"\033[91m✗ Unexpected error during initialization: {str(e)}\033[0m")
            logger.error("Unexpected error during initialization", exc_info=e)
            raise ProjectError(f"Failed to initialize project: {str(e)}") from e


if __name__ == "__main__":
    sys.exit(main())
