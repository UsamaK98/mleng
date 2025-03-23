"""
Test script for the Ollama service.

This script tests connecting to Ollama and generating text/embeddings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logging import logger
from src.utils.config import config_manager
from src.services.ollama import OllamaService

def main():
    """Test Ollama service functionality."""
    logger.info("Testing Ollama service")
    
    # Get service configuration
    base_url = config_manager.config.ollama.base_url
    generation_model = config_manager.config.ollama.generation_model
    embed_model = config_manager.config.ollama.embed_model
    
    logger.info(f"Using base URL: {base_url}")
    logger.info(f"Generation model: {generation_model}")
    logger.info(f"Embedding model: {embed_model}")
    
    # Initialize service
    logger.info("Initializing Ollama service...")
    try:
        ollama_service = OllamaService()
        logger.info("Service initialized successfully")
        
        # Test text generation
        logger.info("Testing text generation...")
        prompt = "Summarize the role of parliament in a democratic society in one paragraph."
        
        response, metadata = ollama_service.generate_text(prompt)
        
        logger.info(f"Generation metadata: {metadata}")
        logger.info(f"Generated text: {response}")
        
        # Test embedding generation
        logger.info("Testing embedding generation...")
        texts = [
            "This is a test sentence about parliament.",
            "Another test sentence about legislative procedures."
        ]
        
        embeddings = ollama_service.get_embeddings(texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.info(f"Embedding dimensions: {len(embeddings[0])}")
        
        logger.info("Ollama service test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing Ollama service: {str(e)}")

if __name__ == "__main__":
    main() 