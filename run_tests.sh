#!/bin/bash
echo "Running tests for Space Muck"
cd "$(dirname "$0")"
python -m unittest discover -s tests
