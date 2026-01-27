#!/bin/bash
echo "Setting up environment..."

# Check if PORT is set, use default if not
if [ -z "" ]; then
    export PORT=8501
    echo "PORT not set, using default: "
else
    echo "Using PORT from environment: "
fi

# Make scripts executable
chmod +x start.sh
