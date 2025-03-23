"""
Test script for VectorStore functionality with available backends.

This script tests the basic functionality of the VectorStore class
with Qdrant and ChromaDB backends.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.services.ollama import OllamaService
from src.storage.vector_db import VectorStore, QDRANT_AVAILABLE, CHROMA_AVAILABLE
from src.utils.config import config_manager

def create_test_data():
    """Create a small dataset for testing."""
    data = {
        'Date': ['2023-06-15', '2023-06-15', '2023-06-16'],
        'Speaker': ['John Smith', 'Jane Doe', 'John Smith'],
        'Content': [
            'We must address the climate change crisis immediately.',
            'Healthcare reform should be our top priority this session.',
            'The budget allocation for education needs to be increased.'
        ]
    }
    return pd.DataFrame(data)

def test_vector_store_qdrant():
    """Test VectorStore with Qdrant backend."""
    if not QDRANT_AVAILABLE:
        logger.warning("Qdrant is not available, skipping test.")
        return False
    
    logger.info("Testing VectorStore with Qdrant backend...")
    
    try:
        # Initialize Ollama service with explicit embedding dimension
        ollama_service = OllamaService()
        
        # Set embedding dimension based on actual embedding size
        # This ensures Qdrant is initialized with the correct vector size
        embedding_test = ollama_service.get_embeddings("test")
        embedding_dim = len(embedding_test)
        ollama_service.embedding_dim = embedding_dim
        logger.info(f"Using embedding dimension: {embedding_dim}")
        
        # Create VectorStore with Qdrant
        vector_store = VectorStore(
            collection_name="test_qdrant",
            ollama_service=ollama_service,
            use_qdrant=True
        )
        
        # Test data
        test_df = create_test_data()
        
        # Store data
        success = vector_store.store_parliamentary_data(test_df)
        if not success:
            logger.error("Failed to store data in Qdrant")
            return False
        
        # Test search
        query = "climate change"
        results = vector_store.search_similar(query, top_k=2)
        
        logger.info(f"Search results for '{query}':")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result['payload'].get('Content', '')[:50]}... (Score: {result['score']:.4f})")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing Qdrant: {str(e)}")
        return False

def test_vector_store_chromadb():
    """Test VectorStore with ChromaDB backend."""
    if not CHROMA_AVAILABLE:
        logger.warning("ChromaDB is not available, skipping test.")
        return False
    
    logger.info("Testing VectorStore with ChromaDB backend...")
    
    try:
        # Initialize Ollama service
        ollama_service = OllamaService()
        
        # Create VectorStore with ChromaDB
        vector_store = VectorStore(
            collection_name="test_chroma",
            ollama_service=ollama_service,
            use_qdrant=False  # Force ChromaDB
        )
        
        # Test data
        test_df = create_test_data()
        
        # Store data
        success = vector_store.store_parliamentary_data(test_df)
        if not success:
            logger.error("Failed to store data in ChromaDB")
            return False
        
        # Test search
        query = "healthcare reform"
        results = vector_store.search_similar(query, top_k=2)
        
        logger.info(f"Search results for '{query}':")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result['payload'].get('Content', '')}... (Score: {result['score']:.4f})")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing ChromaDB: {str(e)}")
        return False

def test_fallback_behavior():
    """Test fallback behavior from Qdrant to ChromaDB."""
    if not CHROMA_AVAILABLE:
        logger.warning("ChromaDB is not available, skipping fallback test.")
        return False
    
    logger.info("Testing fallback behavior from Qdrant to ChromaDB...")
    
    try:
        # Initialize Ollama service
        ollama_service = OllamaService()
        
        # Save original Qdrant configuration
        original_config = None
        
        if QDRANT_AVAILABLE:
            # Store the original config
            if hasattr(config_manager.config, 'qdrant'):
                original_config = {
                    'host': config_manager.config.qdrant.host,
                    'port': config_manager.config.qdrant.port
                }
            
            # Set invalid host and port to force failure
            config_manager.config.qdrant.host = "invalid_host_that_does_not_exist"
            config_manager.config.qdrant.port = 9999
        
        # This should fail to connect to Qdrant and fall back to ChromaDB
        vector_store = VectorStore(
            collection_name="test_fallback",
            ollama_service=ollama_service,
            use_qdrant=True  # This should fail and fall back to ChromaDB
        )
        
        # Restore original configuration if needed
        if original_config:
            config_manager.config.qdrant.host = original_config['host']
            config_manager.config.qdrant.port = original_config['port']
        
        # Verify we're using ChromaDB
        if not vector_store.using_qdrant and vector_store.using_chroma:
            logger.info("Successfully fell back to ChromaDB")
            return True
        else:
            logger.error("Fallback to ChromaDB did not occur as expected")
            return False
    
    except Exception as e:
        logger.error(f"Error testing fallback behavior: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting VectorStore tests...")
    
    # Test with Qdrant
    qdrant_success = test_vector_store_qdrant()
    
    # Test with ChromaDB
    chroma_success = test_vector_store_chromadb()
    
    # Test fallback behavior
    fallback_success = test_fallback_behavior()
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"Qdrant Test: {'Success' if qdrant_success else 'Failed or Skipped'}")
    logger.info(f"ChromaDB Test: {'Success' if chroma_success else 'Failed or Skipped'}")
    logger.info(f"Fallback Test: {'Success' if fallback_success else 'Failed or Skipped'}")
    
    # Overall success
    if (qdrant_success or chroma_success) and fallback_success:
        logger.info("VectorStore implementation is working correctly!")
        return True
    else:
        logger.warning("Some tests failed, check logs for details.")
        return False

if __name__ == "__main__":
    main() 