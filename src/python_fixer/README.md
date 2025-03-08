# Python Fixer

An advanced Python codebase analysis and optimization tool that helps maintain clean, efficient, and well-structured Python projects.

## Features

- **Static Analysis**
  - Import dependency tracking and optimization
  - Cyclomatic complexity analysis
  - Dead code detection
  - Type hint validation

- **Code Optimization**
  - Automatic import restructuring
  - Circular dependency resolution
  - Code quality improvements
  - Performance optimizations

- **Visualization**
  - Import graph visualization
  - Dependency chain analysis
  - Module clustering visualization
  - Performance metrics charts

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from python_fixer import EnhancedAnalyzer, SmartFixer
from pathlib import Path

# Initialize analyzer
root_path = Path("your/project")
analyzer = EnhancedAnalyzer(root_path)

# Run analysis
metrics = analyzer.analyze_project()

# Apply fixes
fixer = SmartFixer(root_path)
report = fixer.fix_project()
```

## Configuration

Create a `python_fixer.yaml` in your project root:

```yaml
enable_type_checking: true
enable_complexity_analysis: true
max_workers: 4

fix_strategies:
  - name: circular_dependencies
    enabled: true
    priority: 1
  - name: import_optimization
    enabled: true
    priority: 2
```

## Advanced Usage

### Custom Fix Strategies

```python
from python_fixer import SmartFixer, FixStrategy

def custom_fix(node):
    # Your custom fix logic
    pass

strategy = FixStrategy(
    priority=1,
    impact=0.8,
    risk=0.2,
    requires_manual_review=False,
    description="Custom fix strategy",
    fix_function=custom_fix
)

fixer = SmartFixer(root_path)
fixer.strategies.append(strategy)
fixer.fix_project()
```

### Performance Optimization

```python
from python_fixer import EnhancedAnalyzer

analyzer = EnhancedAnalyzer(
    root_path,
    config={
        "max_workers": 8,
        "enable_caching": True,
        "cache_dir": ".python_fixer_cache"
    }
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details

# Python Import Fixer

An advanced tool for analyzing and fixing Python module imports, dependencies, and structural issues.

## Overview

Python Import Fixer is a comprehensive solution for analyzing, organizing, and fixing import relationships in Python projects. Using advanced static analysis, machine learning, and graph theory, it identifies problematic import patterns, circular dependencies, and structural issues, then provides intelligent fixes with minimal manual intervention.

## Key Features

- **Advanced Import Analysis**: Identify relative imports, circular dependencies, and unused imports
- **Intelligent Refactoring**: Automatically fix import issues with context-aware transformations
- **Interactive CLI**: Fix imports through a rich command-line interface with visual feedback
- **Web Dashboard**: Monitor and manage fixes through an intuitive web interface
- **ML-Enhanced Type Inference**: Detect and suggest type hints based on usage patterns
- **Comprehensive Reporting**: Generate detailed reports with visualizations and metrics
- **Git Integration**: Safe fixing with automatic Git branch management
- **Undo/Redo Capability**: Track and reverse changes as needed

## Installation

### Requirements

- Python 3.8+
- pip or Poetry

### Install from PyPI

```bash
pip install python-import-fixer
```

### Install from Source

```bash
git clone https://github.com/yourusername/python-import-fixer.git
cd python-import-fixer
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Initialize a project (creates configuration)
fix-imports init /path/to/project

# Analyze imports without making changes
fix-imports analyze /path/to/project

# Fix imports interactively
fix-imports fix /path/to/project --mode interactive

# Fix imports automatically
fix-imports fix /path/to/project --mode automatic

# Launch web dashboard
fix-imports dashboard /path/to/project
```

### Python API Usage

```python
from python_fixer import EnhancedAnalyzer, SmartFixer
from pathlib import Path

# Analyze project imports
analyzer = EnhancedAnalyzer(Path("/path/to/project"))
analysis = analyzer.analyze_project()

# Apply fixes
fixer = SmartFixer(Path("/path/to/project"), backup=True)
results = fixer.fix_project()

# Generate report
print(f"Fixed {results['total_fixes']} import issues")
```

## Configuration

Configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `max_workers` | Number of parallel workers | 4 |
| `backup` | Create backups before fixing | true |
| `fix_mode` | Fix application mode (interactive, automatic, dry-run, safe) | interactive |
| `git_integration` | Enable Git integration | true |
| `exclude_patterns` | Patterns to exclude from analysis | ["venv", "*.pyc", "__pycache__"] |

## Advanced Features

### Machine Learning Type Inference

Python Import Fixer uses neural networks to infer probable type hints based on variable names, usage patterns, and context. This helps improve code safety even in projects without explicit type annotations.

### Dependency Graph Analysis

The tool analyzes and visualizes module dependencies using advanced graph algorithms, identifying optimal refactoring targets to reduce coupling and improve code organization.

### Interactive Web Dashboard

Monitor your project's import health, visualize dependency relationships, and manage fixes through an intuitive web interface:

```bash
fix-imports dashboard /path/to/project --port 8000
```

Then open `http://localhost:8000` in your browser.

## Examples

### Fixing Circular Dependencies

Circular dependencies can be difficult to resolve manually. Python Import Fixer identifies and breaks these cycles with minimal code changes:

```python
# Before
# file_a.py
from file_b import ClassB

class ClassA:
    def method(self):
        return ClassB()

# file_b.py
from file_a import ClassA

class ClassB:
    def method(self):
        return ClassA()

# After
# file_a.py
from typing import Protocol

class ClassBProtocol(Protocol):
    def method(self): ...

class ClassA:
    def method(self):
        from file_b import ClassB
        return ClassB()

# file_b.py
from typing import Protocol

class ClassAProtocol(Protocol):
    def method(self): ...

class ClassB:
    def method(self):
        from file_a import ClassA
        return ClassA()
```

### Converting Relative Imports

Python Import Fixer can automatically convert relative imports to absolute ones:

```python
# Before
from ..utils import helper

# After
from myproject.utils import helper
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Python Import Fixer builds on several open-source technologies:
- [libcst](https://github.com/Instagram/LibCST) for accurate code parsing
- [networkx](https://networkx.org/) for dependency graph analysis
- [rich](https://github.com/Textualize/rich) for beautiful terminal output
- [fastapi](https://fastapi.tiangolo.com/) for the web dashboard
- [typer](https://typer.tiangolo.com/) for the command-line interface

╭─────────────── SPACE_MUCK ────────────────╮
│                                           │
│  ╭─────────────────────────────────────╮  │
│  │                                     │  │
│  │  PYTHON@SYSTEM:~$ _                 │  │
│  │                                     │  │
│  ╰─────────────────────────────────────╯  │
│                                           │
│  [COMMANDS]  [FILES]  [SYSTEM]  [HELP]    │
│                                           │
╰───────────────────────────────────────────╯


┌─────────────────── [ ERROR-HANDLER CLI ] ────────────────────┐
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│  │ RUN │ │ VAL │ │ LOG │ │ CFG │ │ FIX │ │ SYS │ │ HLP │     │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘     │
│                                                              │
│  ┏━━━━━━━━━━━━━━━━━━ COMMAND INPUT ━━━━━━━━━━━━━━━━━━━━━━┓   │
│  ┃                                                       ┃   │
│  ┃  $ project-cli run --port 8080 --path ./myproject _   ┃   │
│  ┃                                                       ┃   │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   │
│                                                              │
│  ┏━━━━━━━━━━━━━━━━━━ ERROR OUTPUT ━━━━━━━━━━━━━━━━━━━━━━━┓   │
│  ┃                                                       ┃   │
│  ┃  ⚠ ValidationError: Invalid project path              ┃   │
│  ┃  ✗ Directory './myproject' contains no Python files   ┃   │
│  ┃  ! Additional context available with --verbose        ┃   │
│  ┃                                                       ┃   │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   │
│                                                              │
│  ┏━━━━━━━━━━━━━━━━━━ SUGGESTIONS ━━━━━━━━━━━━━━━━━━━━━━━━┓   │
│  ┃                                                       ┃   │
│  ┃  ▶ Try: project-cli validate --path ./myproject       ┃   │
│  ┃  ▶ Try: project-cli run --path ./correct/path         ┃   │
│  ┃  ▶ See: project-cli --help for more information       ┃   │
│  ┃                                                       ┃   │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   │
│                                                              │
│  System: Linux | User: admin | Version: 2.1.3 | Log: Active  │
└──────────────────────────────────────────────────────────────┘