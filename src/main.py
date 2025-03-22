"""
Parliamentary Minutes Agentic Chatbot
Main application entry point
"""
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    API_HOST, 
    API_PORT, 
    API_DEBUG
)

def main():
    """Main application entry point"""
    from src.api.app import start_api_server
    
    # Start the FastAPI server
    start_api_server(host=API_HOST, port=API_PORT, debug=API_DEBUG)

if __name__ == "__main__":
    main() 