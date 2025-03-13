# Standard library imports

# Third-party library imports

# Local application imports
from setuptools import setup, find_packages

setup(
    name="python_fixer",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "rich",  # Required for console output
        "toml",  # Required for configuration
    ],
    extras_require={
        "dev": [
            "libcst",  # For static analysis
            "networkx",  # For dependency graphs
            "matplotlib",  # For visualization
            "mypy",  # For type checking
            "numpy",  # For numerical operations
            "radon",  # For complexity analysis
            "rope",  # For refactoring
            "sympy",  # For symbolic math
        ]
    },
    entry_points={
        "console_scripts": [
            "python-fixer=python_fixer.cli:main",
        ],
    },
)
