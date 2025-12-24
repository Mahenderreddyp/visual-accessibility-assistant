#!/bin/bash
# setup.sh: Initialize the environment

echo "Creating directories..."
mkdir data/raw data/processed models/checkpoints models/final models/gguf

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! Ready to run scripts."