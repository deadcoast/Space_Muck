"""
Simple script to check for linting issues in the benchmark_comprehensive_gpu.py file.
"""

import ast
import sys

def check_file(file_path):
    """Check a Python file for syntax errors and basic issues."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        # Parse the file to check for syntax errors
        ast.parse(content)
        print(f"✅ {file_path} is syntactically correct")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False

if __name__ == "__main__":
    file_path = "src/tests/bunchmarks/benchmark_comprehensive_gpu.py"
    if check_file(file_path):
        print("No syntax errors found!")
    else:
        sys.exit(1)
