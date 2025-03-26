@echo off
REM Evaluation script for Parliamentary Meeting Analyzer
REM This script activates the conda environment and runs the evaluation

REM Change to the script directory
cd /d "%~dp0"

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: conda is not installed or not in PATH
    exit /b 1
)

REM Activate conda environment
echo Activating conda environment 'mentor360'...
call conda activate mentor360
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate conda environment 'mentor360'. Make sure it exists with Python 3.10.
    exit /b 1
)

REM Run evaluation
echo Running evaluation script...
python src/evaluation/run_evaluation.py

echo Evaluation complete. Results are saved in the output-new directory. 