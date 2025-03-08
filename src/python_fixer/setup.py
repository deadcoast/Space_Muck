from setuptools import setup, find_packages

setup(
    name="python_fixer",
    version="0.1.0",
    description="Fix Python import and class structure issues in your project",
    author="Codeium",
    packages=find_packages(include=["python_fixer", "python_fixer.*"]),
    entry_points={
        "console_scripts": [
            "python_fixer=python_fixer.cli:main",
        ],
    },
    install_requires=[
        "pathlib",
        "typing",
        "libcst",
        "matplotlib",
        "mypy",
        "networkx",
        "numpy",
        "rope",
        "sympy",
        "toml",
        "radon",
        "rich",
        "pydantic",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "test": [
            "pytest",
            "pytest-cov",
        ],
    },
    python_requires=">=3.7",
)
