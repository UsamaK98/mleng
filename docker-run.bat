@echo off
REM Docker run script for Parliamentary Meeting Analyzer
REM This script builds and runs the application using Docker Compose

REM Change to the script directory
cd /d "%~dp0"

REM Check if Docker is available
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Docker is not installed or not in PATH
    exit /b 1
)

REM Check if Docker Compose is available
where docker-compose >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Docker Compose is not installed or not in PATH
    exit /b 1
)

REM Build and run containers
echo Building and starting Docker containers...
docker-compose up --build -d

echo.
echo Parliamentary Meeting Analyzer is now running in Docker containers.
echo The application should be available at: http://localhost:8501
echo.
echo To view logs, run: docker-compose logs -f
echo To stop containers, run: docker-compose down 