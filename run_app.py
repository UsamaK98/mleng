"""
Improved runner script for the Parliamentary Minutes Agentic Chatbot
This script ensures the backend is properly initialized before starting the frontend
"""
import argparse
import os
import subprocess
import sys
import time
import requests
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import API_PORT, UI_PORT, API_HOST, QDRANT_HOST, QDRANT_PORT, OLLAMA_BASE_URL


def check_qdrant():
    """Check if Qdrant is running"""
    try:
        response = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections")
        if response.status_code == 200:
            print("✅ Qdrant is running")
            return True
        else:
            print("❌ Qdrant is not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Qdrant is not running. Please start Docker and run Qdrant.")
        return False


def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            print("✅ Ollama is running")
            models = response.json().get("models", [])
            mistral_available = any("mistral" in model.get("name", "").lower() for model in models)
            if mistral_available:
                print("✅ Mistral model is available")
            else:
                print("⚠️ Mistral model not found. You may need to run: docker exec ollama ollama pull mistral")
            return True
        else:
            print("❌ Ollama is not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running. Please start Docker and run Ollama.")
        return False


def check_data_files():
    """Check if required data files exist"""
    from config.config import MINUTES_CSV, SPEAKERS_CSV
    
    minutes_exists = Path(MINUTES_CSV).exists()
    speakers_exists = Path(SPEAKERS_CSV).exists()
    
    if minutes_exists and speakers_exists:
        print("✅ Data files exist")
        return True
    else:
        print("❌ Data files missing. Need to run ingestion.")
        return False


def run_data_ingestion(force_recreate=False):
    """Run data ingestion process"""
    from src.data.ingestion import ingest_minutes
    
    print("\n🔄 Starting data ingestion process...")
    result = ingest_minutes(force_recreate=force_recreate)
    print(f"Data ingestion complete: {result}")


def start_api_server(wait_for_startup=True):
    """Start the FastAPI server in a new process"""
    print(f"\n🚀 Starting API server on port {API_PORT}...")
    
    # Use subprocess to start the API server
    api_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "src.api.app:app", "--host", API_HOST, "--port", str(API_PORT)]
    )
    
    # Wait for the API server to start up
    if wait_for_startup:
        max_retries = 10
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(f"http://{API_HOST}:{API_PORT}")
                if response.status_code == 200:
                    print(f"✅ API server is running at http://{API_HOST}:{API_PORT}")
                    break
                else:
                    print(f"⏳ Waiting for API server to start... ({retries+1}/{max_retries})")
            except requests.exceptions.ConnectionError:
                print(f"⏳ Waiting for API server to start... ({retries+1}/{max_retries})")
            
            retries += 1
            time.sleep(1)
        
        if retries == max_retries:
            print("❌ Couldn't confirm API server startup. Frontend may not work correctly.")
    
    return api_process


def start_ui_server():
    """Start the Streamlit UI server"""
    print(f"\n🚀 Starting UI server on port {UI_PORT}...")
    print(f"📊 UI will be available at http://localhost:{UI_PORT}")
    
    # Use subprocess to start the UI server
    subprocess.run([
        "streamlit", "run", 
        "src/ui/streamlit_app.py",
        "--server.port", str(UI_PORT),
        "--server.address", "0.0.0.0"
    ])


def main():
    """Main entry point with improved initialization sequence"""
    parser = argparse.ArgumentParser(description="Parliamentary Minutes Agentic Chatbot")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreate vector store during ingestion")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--ui-only", action="store_true", help="Run only the UI server")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    args = parser.parse_args()
    
    # Check dependencies unless skipped
    if not args.skip_checks:
        print("\n🔍 Checking dependencies...")
        qdrant_ok = check_qdrant()
        ollama_ok = check_ollama()
        
        if not (qdrant_ok and ollama_ok):
            print("\n❌ Some dependencies are not available. Please ensure Qdrant and Ollama are running.")
            print("   You can use --skip-checks to bypass these checks.")
            return
        
        # Check if data files exist
        data_exists = check_data_files()
        if not data_exists and not args.ingest:
            print("\n⚠️ Data files are missing. Would you like to run data ingestion? (y/n)")
            choice = input().lower()
            if choice in ['y', 'yes']:
                args.ingest = True
                args.force_recreate = True
            else:
                print("⚠️ Proceeding without data ingestion. The application may not work correctly.")
    
    # Run data ingestion if requested
    if args.ingest:
        run_data_ingestion(force_recreate=args.force_recreate)
    
    # Start the API server if running API-only or full app
    api_process = None
    if args.api_only or not args.ui_only:
        api_process = start_api_server(wait_for_startup=not args.api_only)
    
    # Start the UI server if running UI-only or full app
    if args.ui_only or not args.api_only:
        try:
            start_ui_server()
        finally:
            # Clean up the API process if it was started
            if api_process is not None:
                print("\n🛑 Shutting down API server...")
                api_process.terminate()
                api_process.wait()


if __name__ == "__main__":
    print("\n🏛️ Parliamentary Minutes Agentic Chatbot")
    print("----------------------------------------\n")
    main() 