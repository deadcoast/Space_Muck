"""
Tests for import resolution functionality.

This test suite covers:
1. Relative imports
2. Namespace resolution
3. Circular import prevention
"""

from pathlib import Path

import pytest

from python_fixer.core.analyzer import ImportAnalyzer

# Test fixtures
@pytest.fixture
def temp_package_structure(tmp_path: Path) -> Path:
    """Create a temporary package structure for testing imports.
    
    Structure:
    temp_pkg/
    ├── __init__.py
    ├── module_a.py
    ├── subpkg1/
    │   ├── __init__.py
    │   ├── module_b.py
    │   └── subpkg2/
    │       ├── __init__.py
    │       └── module_c.py
    └── subpkg3/
        ├── __init__.py
        └── module_d.py
    """
    pkg_root = tmp_path / "temp_pkg"
    pkg_root.mkdir()
    
    # Create main package __init__.py
    (pkg_root / "__init__.py").write_text("from .module_a import ClassA")
    
    # Create module_a.py
    (pkg_root / "module_a.py").write_text("""
from .subpkg1.module_b import ClassB

class ClassA:
    def __init__(self):
        self.b = ClassB()
""")

    # Create subpkg1
    subpkg1 = pkg_root / "subpkg1"
    subpkg1.mkdir()
    (subpkg1 / "__init__.py").write_text("")
    
    # Create module_b.py
    (subpkg1 / "module_b.py").write_text("""
from ..module_a import ClassA
from .subpkg2.module_c import ClassC

class ClassB:
    def __init__(self):
        self.c = ClassC()
""")

    # Create subpkg2
    subpkg2 = subpkg1 / "subpkg2"
    subpkg2.mkdir()
    (subpkg2 / "__init__.py").write_text("")
    
    # Create module_c.py
    (subpkg2 / "module_c.py").write_text("""
from ...subpkg3.module_d import ClassD

class ClassC:
    def __init__(self):
        self.d = ClassD()
""")

    # Create subpkg3
    subpkg3 = pkg_root / "subpkg3"
    subpkg3.mkdir()
    (subpkg3 / "__init__.py").write_text("")
    
    # Create module_d.py
    (subpkg3 / "module_d.py").write_text("""
class ClassD:
    pass
""")
    
    return pkg_root

# Test relative imports
def test_parent_directory_imports(temp_package_structure: Path) -> None:
    """Test imports using parent directory notation (..)."""
    analyzer = ImportAnalyzer(temp_package_structure / "subpkg1" / "module_b.py")
    imports = analyzer.analyze_imports()
    
    assert any(imp.module == "..module_a" for imp in imports), "Parent directory import not found"
    assert any(imp.imported_names == ["ClassA"] for imp in imports), "ClassA import not found"

def test_sibling_imports(temp_package_structure: Path) -> None:
    """Test imports from sibling modules (.)."""
    analyzer = ImportAnalyzer(temp_package_structure / "subpkg1" / "module_b.py")
    imports = analyzer.analyze_imports()
    
    assert any(imp.module == ".subpkg2.module_c" for imp in imports), "Sibling directory import not found"
    assert any(imp.imported_names == ["ClassC"] for imp in imports), "ClassC import not found"

def test_nested_package_imports(temp_package_structure: Path) -> None:
    """Test imports from deeply nested packages (...)."""
    analyzer = ImportAnalyzer(temp_package_structure / "subpkg1" / "subpkg2" / "module_c.py")
    imports = analyzer.analyze_imports()
    
    assert any(imp.module == "...subpkg3.module_d" for imp in imports), "Nested package import not found"
    assert any(imp.imported_names == ["ClassD"] for imp in imports), "ClassD import not found"

def test_init_imports(temp_package_structure: Path) -> None:
    """Test imports in __init__.py files."""
    analyzer = ImportAnalyzer(temp_package_structure / "__init__.py")
    imports = analyzer.analyze_imports()
    
    assert any(imp.module == ".module_a" for imp in imports), "Module import in __init__.py not found"
    assert any(imp.imported_names == ["ClassA"] for imp in imports), "ClassA import in __init__.py not found"

@pytest.fixture
def circular_import_structure(tmp_path: Path) -> Path:
    """Create a temporary package structure with circular imports.
    
    Structure:
    circular_pkg/
    ├── __init__.py
    ├── module_x.py  # imports from module_y
    └── module_y.py  # imports from module_x
    """
    pkg_root = tmp_path / "circular_pkg"
    pkg_root.mkdir()
    
    # Create package __init__.py
    (pkg_root / "__init__.py").write_text("")
    
    # Create module_x.py with import from module_y
    (pkg_root / "module_x.py").write_text("""
        from .module_y import ClassY
        
        class ClassX:
            def __init__(self):
                self.y = ClassY()
    """)
    
    # Create module_y.py with import from module_x
    (pkg_root / "module_y.py").write_text("""
        from .module_x import ClassX
        
        class ClassY:
            def __init__(self):
                self.x = ClassX()
    """)
    
    return pkg_root

def test_detect_direct_circular_import(circular_import_structure: Path) -> None:
    """Test detection of direct circular imports between two modules."""
    analyzer_x = ImportAnalyzer(circular_import_structure / "module_x.py")
    imports_x = analyzer_x.analyze_imports()
    
    analyzer_y = ImportAnalyzer(circular_import_structure / "module_y.py")
    imports_y = analyzer_y.analyze_imports()
    
    # Verify both modules import from each other
    assert any(imp.module == ".module_y" for imp in imports_x), "Import from module_y not found"
    assert any(imp.module == ".module_x" for imp in imports_y), "Import from module_x not found"
    
    # Verify imported class names
    assert any(imp.imported_names == ["ClassY"] for imp in imports_x), "ClassY import not found"
    assert any(imp.imported_names == ["ClassX"] for imp in imports_y), "ClassX import not found"

@pytest.fixture
def indirect_circular_structure(tmp_path: Path) -> Path:
    """Create a package structure with indirect circular imports.
    
    Structure:
    indirect_pkg/
    ├── __init__.py
    ├── module_a.py  # imports from module_b
    ├── module_b.py  # imports from module_c
    └── module_c.py  # imports from module_a
    """
    pkg_root = tmp_path / "indirect_pkg"
    pkg_root.mkdir()
    
    # Create package __init__.py
    (pkg_root / "__init__.py").write_text("")
    
    # Create module_a.py
    (pkg_root / "module_a.py").write_text("""
        from .module_b import ClassB
        
        class ClassA:
            def __init__(self):
                self.b = ClassB()
    """)
    
    # Create module_b.py
    (pkg_root / "module_b.py").write_text("""
        from .module_c import ClassC
        
        class ClassB:
            def __init__(self):
                self.c = ClassC()
    """)
    
    # Create module_c.py
    (pkg_root / "module_c.py").write_text("""
        from .module_a import ClassA
        
        class ClassC:
            def __init__(self):
                self.a = ClassA()
    """)
    
    return pkg_root

def test_detect_indirect_circular_import(indirect_circular_structure: Path) -> None:
    """Test detection of indirect circular imports through multiple modules."""
    analyzer_a = ImportAnalyzer(indirect_circular_structure / "module_a.py")
    imports_a = analyzer_a.analyze_imports()
    
    analyzer_b = ImportAnalyzer(indirect_circular_structure / "module_b.py")
    imports_b = analyzer_b.analyze_imports()
    
    analyzer_c = ImportAnalyzer(indirect_circular_structure / "module_c.py")
    imports_c = analyzer_c.analyze_imports()
    
    # Verify the circular import chain
    assert any(imp.module == ".module_b" for imp in imports_a), "Import from module_b not found"
    assert any(imp.module == ".module_c" for imp in imports_b), "Import from module_c not found"
    assert any(imp.module == ".module_a" for imp in imports_c), "Import from module_a not found"
    
    # Verify imported class names
    assert any(imp.imported_names == ["ClassB"] for imp in imports_a), "ClassB import not found"
    assert any(imp.imported_names == ["ClassC"] for imp in imports_b), "ClassC import not found"
    assert any(imp.imported_names == ["ClassA"] for imp in imports_c), "ClassA import not found"
