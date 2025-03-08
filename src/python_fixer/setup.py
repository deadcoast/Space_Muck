from setuptools import setup, find_packages

setup(
    name="python_fixer",
    version="0.1.0",
    description="Fix Python import and class structure issues in your project",
    author="Codeium",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "networkx",
        "numpy",
        "rich",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
        ],
    },
    python_requires=">=3.7",
)
)
