#!/usr/bin/env python3
"""
Main entry point for the Python Import Fixer tool.
"""

import argparse
import sys
from pathlib import Path

from python_fixer.core.analyzer import ProjectAnalyzer
from python_fixer.logging.structured import StructuredLogger
from python_fixer.web.dashboard import run_dashboard


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix Python import and class structure issues in your project"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize a project')
    init_parser.add_argument('project_path', type=Path, help='Path to the project')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze project imports')
    analyze_parser.add_argument('project_path', type=Path, help='Path to the project')
    
    # Fix command
    fix_parser = subparsers.add_parser('fix', help='Fix import issues')
    fix_parser.add_argument('project_path', type=Path, help='Path to the project')
    fix_parser.add_argument('--mode', choices=['interactive', 'automatic'], default='interactive',
                          help='Fix mode (default: interactive)')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch web dashboard')
    dashboard_parser.add_argument('project_path', type=Path, help='Path to the project')
    dashboard_parser.add_argument('--host', type=str, default='localhost', help='Dashboard host (default: localhost)')
    dashboard_parser.add_argument('--port', type=int, default=8000, help='Dashboard port (default: 8000)')
    dashboard_parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    # Global options
    for p in [init_parser, analyze_parser, fix_parser, dashboard_parser]:
        p.add_argument('--log-file', type=Path, default=Path('import_fixes.log'),
                      help='Path for the log file (default: import_fixes.log)')
        p.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()


def print_analysis_summary(results: dict) -> None:
    """Print a summary of the analysis results."""
    print("\n\033[92m✓ Analysis complete\033[0m")
    print(f"\033[96m  - Files analyzed: {len(results.get('structure', {}).get('modules', []))}")
    print(f"  - Circular dependencies: {len(results.get('dependencies', {}).get('cycles', []))}")
    print(f"  - Enhancement suggestions: {len(results.get('enhancements', []))}\033[0m")


def print_fixes_summary(fixes: dict) -> None:
    """Print a summary of the applied fixes."""
    print("\n\033[92m✓ Fixes applied successfully\033[0m")
    print(f"\033[96m  - Files modified: {fixes.get('files_modified', 0)}")
    print(f"  - Imports fixed: {fixes.get('imports_fixed', 0)}\033[0m")


def print_dashboard_info(host: str, port: int, reload: bool) -> None:
    """Print information about the dashboard."""
    print(f"\033[96m  - URL: http://{host}:{port}")
    print(f"  - Auto-reload: {'enabled' if reload else 'disabled'}\033[0m")


def main():
    args = parse_args()
    
    # Validate project path
    if not args.project_path.exists():
        print(f"\033[91mError: Project path '{args.project_path}' does not exist\033[0m")
        return 1
    
    if not args.project_path.is_dir():
        print(f"\033[91mError: '{args.project_path}' is not a directory\033[0m")
        return 1
    
    # Setup logging
    logger = StructuredLogger(
        name="python_fixer",
        log_file=args.log_file,
        level="DEBUG" if args.verbose else "INFO"
    )
    
    try:
        if args.command == 'init':
            print(f"\033[94mInitializing project at {args.project_path}...\033[0m")
            analyzer = ProjectAnalyzer(args.project_path)
            analyzer.initialize_project()
            print("\033[92m✓ Project initialized successfully\033[0m")
            
        elif args.command == 'analyze':
            print(f"\033[94mAnalyzing project at {args.project_path}...\033[0m")
            analyzer = ProjectAnalyzer(args.project_path)
            results = analyzer.analyze_project()
            logger.info("Analysis complete", extra={"metrics": results})
            
            print_analysis_summary(results)
            
        elif args.command == 'fix':
            print(f"\033[94mFixing imports in {args.project_path} (mode: {args.mode})...\033[0m")
            analyzer = ProjectAnalyzer(args.project_path)
            fixes = analyzer.fix_project(mode=args.mode)
            logger.info("Fixes complete", extra={"metrics": fixes})
            
            print_fixes_summary(fixes)
            
        elif args.command == 'dashboard':
            print(f"\033[94mLaunching dashboard for {args.project_path}...\033[0m")
            print_dashboard_info(args.host, args.port, args.reload)
            
            try:
                run_dashboard(
                    project_path=args.project_path,
                    host=args.host,
                    port=args.port,
                    reload=args.reload,
                    log_level="DEBUG" if args.verbose else "INFO"
                )
            except KeyboardInterrupt:
                print("\n\033[93m⚠ Dashboard stopped by user\033[0m")
            except Exception as e:
                print(f"\n\033[91m✗ Failed to start dashboard: {e}\033[0m")
                logger.error("Dashboard failed", exc_info=e)
                return 1
            
        return 0
        
    except KeyboardInterrupt:
        print("\n\033[93m⚠ Operation cancelled by user\033[0m")
        return 130
    except Exception as e:
        print(f"\n\033[91m✗ {args.command} failed: {str(e)}\033[0m")
        logger.error(f"{args.command} failed", exc_info=e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
