from setuptools import setup

# This file is only here for backward compatibility.
# All configuration is in pyproject.toml
setup(
    name="python_fixer",
    packages=["python_fixer"],
    package_dir={"python_fixer": "."},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",
)
