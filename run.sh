#!/bin/bash

# Activating virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    python3 main.py
else
    echo "Virtual environment not found. Please run ./setup.sh first."
fi
