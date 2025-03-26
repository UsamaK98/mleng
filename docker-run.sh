#!/bin/bash

# Docker run script for Parliamentary Meeting Analyzer
# This script builds and runs the application using Docker Compose

# Exit on error
set -e

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed or not in PATH"
    exit 1
fi

# Determine the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Build and run containers
echo "Building and starting Docker containers..."
docker-compose up --build -d

echo
echo "Parliamentary Meeting Analyzer is now running in Docker containers."
echo "The application should be available at: http://localhost:8501"
echo
echo "To view logs, run: docker-compose logs -f"
echo "To stop containers, run: docker-compose down" 