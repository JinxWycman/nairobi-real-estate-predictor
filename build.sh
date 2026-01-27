#!/bin/bash
echo "=================================================="
echo "FORCING PYTHON BUILD - THIS IS NOT A NODE.JS APP"
echo "=================================================="

# Explicitly use Python
export PYTHON_VERSION=3.9.18
python --version

# Install only Python packages
pip install --upgrade pip
pip install -r requirements.txt

echo "=================================================="
echo "BUILD COMPLETE - Python app ready!"
echo "=================================================="
