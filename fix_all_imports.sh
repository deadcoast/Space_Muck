#!/bin/bash
# Fix all imports in the Space Muck project

# Set the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$PROJECT_ROOT/src"

echo "=== Space Muck Import Fixer ==="
echo "Project root: $PROJECT_ROOT"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Add the project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Step 1: Fix general imports
echo "Step 1: Fixing general imports..."
python3 -m src.utils.fix_imports "$SRC_DIR"
echo

# Step 2: Fix UI-specific imports
echo "Step 2: Fixing UI-specific imports..."
python3 -m src.utils.fix_ui_imports "$SRC_DIR/ui"
echo

# Step 3: Fix specific problematic files
echo "Step 3: Fixing specific problematic files..."

# Fix the crew management panel demo
echo "Fixing ascii_crew_management_panel_demo.py..."
python3 -m src.utils.fix_imports "$SRC_DIR/demos/ascii_crew_management_panel_demo.py"

# Fix the crew management panel
echo "Fixing ascii_crew_management_panel.py..."
python3 -m src.utils.fix_imports "$SRC_DIR/ui/ui_element/ascii_crew_management_panel.py"

# Fix the base UI files
echo "Fixing ascii_base.py..."
python3 -m src.utils.fix_imports "$SRC_DIR/ui/ui_base/ascii_base.py"

echo "Fixing ascii_ui.py..."
python3 -m src.utils.fix_imports "$SRC_DIR/ui/ui_base/ascii_ui.py"

echo
echo "=== Import fixing complete ==="
echo "Run your tests to ensure everything works correctly."
echo "If you encounter any issues, you can run the fixers with --dry-run to see what changes would be made without actually making them."
