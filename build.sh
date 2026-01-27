#!/bin/bash
echo "=========================================="
echo "Starting Python/Streamlit build..."
echo "This is NOT a Node.js application"
echo "=========================================="

# Show Python version
python --version
pip --version

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "=========================================="
echo "Build complete! Starting Streamlit app..."
echo "=========================================="
