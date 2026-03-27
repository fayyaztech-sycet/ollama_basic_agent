#!/bin/bash

# Exit on error
set -e

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Ensure pip is up to date and install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install requests yt-dlp psutil

echo "Setup complete! Use ./run.sh to start the agent."
