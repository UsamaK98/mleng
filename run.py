"""
Runner script for the Parliamentary Minutes Agentic Chatbot
"""
import argparse
import os
import subprocess
import sys
import time
from threading import Thread

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import API_PORT, UI_PORT


def run_data_ingestion(force_recreate=False):
    """Run data ingestion process"""
    from src.data.ingestion import ingest_minutes
    
    print("Starting data ingestion process...")
    result = ingest_minutes(force_recreate=force_recreate)
    print(f"Data ingestion complete: {result}")


def run_api_server():
    """Run the FastAPI server"""
    from src.api.app import start_api_server
    
    print(f"Starting API server on port {API_PORT}...")
    start_api_server()


def run_ui_server():
    """Run the Streamlit UI server"""
    print(f"Starting UI server on port {UI_PORT}...")
    subprocess.run([
        "streamlit", "run", 
        "src/ui/streamlit_app.py",
        "--server.port", str(UI_PORT),
        "--server.address", "0.0.0.0"
    ])


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Parliamentary Minutes Agentic Chatbot")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreate vector store during ingestion")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--ui-only", action="store_true", help="Run only the UI server")
    args = parser.parse_args()
    
    if args.ingest:
        run_data_ingestion(force_recreate=args.force_recreate)
        if not (args.api_only or args.ui_only):
            # If no other flags are specified, exit after ingestion
            return
    
    if args.api_only:
        # Run only the API server
        run_api_server()
    elif args.ui_only:
        # Run only the UI server
        run_ui_server()
    else:
        # For the combined mode, we'll use subprocess for both servers to avoid threading issues
        print("Starting both API and UI servers...")
        print("Open a new terminal window and run: python run.py --api-only")
        print("Wait a few seconds for the API to start, then continue with UI")
        time.sleep(3)
        run_ui_server()


if __name__ == "__main__":
    main() 