"""
Parliamentary Minutes Agentic Chatbot
Main application entry point
"""
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

<<<<<<< Updated upstream
from config.config import (
    API_HOST, 
    API_PORT, 
    API_DEBUG
)
=======
from src.utils.logger import log
from src.utils.config import config_manager
from src.data.pipeline import data_pipeline

def run_api():
    """Run the FastAPI server."""
    from src.api.app import app
    import uvicorn
    
    port = config_manager.get("api.port", 8000)
    host = config_manager.get("api.host", "127.0.0.1")
    
    log.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

def run_ui():
    """Run the Streamlit UI."""
    streamlit_path = str(project_root / "src" / "ui" / "app_fix.py")
    port = config_manager.get("ui.port", 8501)
    
    log.info(f"Starting Streamlit UI on port {port}")
    
    # Add the project root to PYTHONPATH when running Streamlit
    env = os.environ.copy()
    
    # Add the project root to PYTHONPATH
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root};{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = f"{project_root}"
    
    subprocess.run([
        "streamlit", "run", 
        streamlit_path,
        "--server.port", str(port),
        "--server.address", "127.0.0.1"
    ], env=env)

def run_data_processing():
    """Run the data processing pipeline."""
    log.info("Starting data processing pipeline")
    
    # Get pipeline configuration
    chunk_data = config_manager.get("pipeline.chunk_data", True)
    extract_entities = config_manager.get("pipeline.extract_entities", True)
    build_graph = config_manager.get("pipeline.build_graph", True)
    build_embeddings = config_manager.get("pipeline.build_embeddings", True)
    
    # Run the pipeline with logging
    results = data_pipeline.run_pipeline(
        force_reload=True,
        chunk_data=chunk_data,
        extract_entities=extract_entities,
        build_graph=build_graph,
        build_embeddings=build_embeddings
    )
    
    if results["success"]:
        log.info(f"Data processing completed in {results['timing']['total']:.2f} seconds")
        log.info(f"Processed {results['counts'].get('records', 0)} records")
        log.info(f"Generated {results['counts'].get('chunks', 0)} chunks")
        log.info(f"Extracted {results['counts'].get('entities', 0)} entities")
        log.info(f"Built graph with {results['counts'].get('graph_nodes', 0)} nodes and {results['counts'].get('graph_edges', 0)} edges")
        return True
    else:
        log.error(f"Data processing failed: {results.get('errors', ['Unknown error'])}")
        return False
>>>>>>> Stashed changes

def main():
    """Main application entry point"""
    from src.api.app import start_api_server
    
    # Start the FastAPI server
    start_api_server(host=API_HOST, port=API_PORT, debug=API_DEBUG)

if __name__ == "__main__":
    main() 