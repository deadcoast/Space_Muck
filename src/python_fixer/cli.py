#!/usr/bin/env python3
"""
Main entry point for the Python Import Fixer tool.
"""

import argparse
import sys
from pathlib import Path

from python_fixer.core.analyzer import ProjectAnalyzer
from python_fixer.logging.structured import StructuredLogger


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
    dashboard_parser.add_argument('--port', type=int, default=8000, help='Dashboard port')
    
    # Global options
    for p in [init_parser, analyze_parser, fix_parser, dashboard_parser]:
        p.add_argument('--log-file', type=Path, default=Path('import_fixes.log'),
                      help='Path for the log file (default: import_fixes.log)')
        p.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate project path
    if not args.project_path.exists():
        print(f"Error: Project path '{args.project_path}' does not exist")
        return 1
    
    # Setup logging
    logger = StructuredLogger(
        name="python_fixer",
        log_file=args.log_file,
        level="DEBUG" if args.verbose else "INFO"
    )
    
    try:
        if args.command == 'init':
            logger.info(f"Initializing project at {args.project_path}")
            analyzer = ProjectAnalyzer(args.project_path)
            analyzer.initialize_project()
            
        elif args.command == 'analyze':
            logger.info(f"Analyzing project at {args.project_path}")
            analyzer = ProjectAnalyzer(args.project_path)
            results = analyzer.analyze_project()
            logger.info("Analysis complete", extra={"metrics": results})
            
        elif args.command == 'fix':
            logger.info(f"Fixing imports in {args.project_path} (mode: {args.mode})")
            analyzer = ProjectAnalyzer(args.project_path)
            fixes = analyzer.fix_project(mode=args.mode)
            logger.info("Fixes complete", extra={"metrics": fixes})
            
        elif args.command == 'dashboard':
            logger.info(f"Launching dashboard for {args.project_path} on port {args.port}")
            # TODO: Implement dashboard
            logger.warning("Dashboard feature not yet implemented")
            
        return 0
        
    except Exception as e:
        logger.error(f"{args.command} failed", exc_info=e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
