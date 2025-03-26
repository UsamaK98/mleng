#!/bin/bash

# Evaluation script for Parliamentary Meeting Analyzer
# This script activates the conda environment and runs the evaluation

# Exit on error
set -e

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Determine the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate conda environment
echo "Activating conda environment 'mentor360'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mentor360 || { echo "Error: Failed to activate conda environment 'mentor360'. Make sure it exists with Python 3.10."; exit 1; }

# Run evaluation
echo "Running evaluation script..."
python src/evaluation/run_evaluation.py

echo "Evaluation complete. Results are saved in the output-new directory." 