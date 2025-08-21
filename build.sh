#!/bin/bash
# Build script for deployment platforms

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

echo "Build completed successfully!"
