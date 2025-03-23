"""
A wrapper script for the Streamlit app that ensures proper import paths.
Run this instead of app.py directly to avoid import issues.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import and run the Streamlit app
from src.ui.app import main

if __name__ == "__main__":
    main() 