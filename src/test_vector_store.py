"""
Test script for VectorStore functionality with all backends.

This script tests the basic functionality of the VectorStore class
with Qdrant, ChromaDB, and the SimpleVectorStore fallback.
"""

import sys
import logging
import pandas as pd
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
        # Initialize Ollama service
        ollama_service = OllamaService()
        
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

def test_simple_vector_store():
    """Test VectorStore with SimpleVectorStore fallback."""
    logger.info("Testing VectorStore with SimpleVectorStore fallback...")
    
    # Force both Qdrant and ChromaDB to be unavailable
    original_qdrant_available = QDRANT_AVAILABLE
    original_chroma_available = CHROMA_AVAILABLE
    
    # Monkey patch the availability flags
    import src.storage.vector_db
    src.storage.vector_db.QDRANT_AVAILABLE = False
    src.storage.vector_db.CHROMA_AVAILABLE = False
    
    try:
        # Initialize Ollama service
        ollama_service = OllamaService()
        
        # Create VectorStore (should use SimpleVectorStore)
        vector_store = VectorStore(
            collection_name="test_simple",
            ollama_service=ollama_service
        )
        
        # Verify we're using the simple store
        if not hasattr(vector_store, 'using_simple') or not vector_store.using_simple:
            logger.error("Not using SimpleVectorStore as expected")
            return False
        
        # Test data
        test_df = create_test_data()
        
        # Store data
        success = vector_store.store_parliamentary_data(test_df)
        if not success:
            logger.error("Failed to store data in SimpleVectorStore")
            return False
        
        # Test search
        query = "education budget"
        results = vector_store.search_similar(query, top_k=2)
        
        logger.info(f"Search results for '{query}':")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: {result['payload'].get('Content', '')}... (Score: {result['score']:.4f})")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing SimpleVectorStore: {str(e)}")
        return False
    
    finally:
        # Restore original availability flags
        src.storage.vector_db.QDRANT_AVAILABLE = original_qdrant_available
        src.storage.vector_db.CHROMA_AVAILABLE = original_chroma_available

def test_fallback_behavior():
    """Test automatic fallback behavior."""
    logger.info("Testing fallback behavior...")
    
    try:
        # Initialize Ollama service
        ollama_service = OllamaService()
        
        # Create VectorStore with invalid Qdrant parameters to force fallback
        if QDRANT_AVAILABLE:
            # Save original configuration
            original_host = config_manager.config.qdrant.host
            original_port = config_manager.config.qdrant.port
            
            # Temporarily modify configuration to force fallback
            config_manager.config.qdrant.host = "invalid_host"
            config_manager.config.qdrant.port = 9999
        
        # This should fall back to ChromaDB or SimpleVectorStore
        vector_store = VectorStore(
            collection_name="test_fallback",
            ollama_service=ollama_service,
            use_qdrant=True  # This should fail and trigger fallback
        )
        
        # Restore original configuration if needed
        if QDRANT_AVAILABLE:
            config_manager.config.qdrant.host = original_host
            config_manager.config.qdrant.port = original_port
        
        # Verify we're using a fallback
        if vector_store.using_qdrant:
            logger.error("Fallback did not occur as expected")
            return False
        
        if vector_store.using_chroma:
            logger.info("Successfully fell back to ChromaDB")
        elif vector_store.using_simple:
            logger.info("Successfully fell back to SimpleVectorStore")
        else:
            logger.error("Unknown fallback state")
            return False
        
        return True
    
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
    
    # Test with SimpleVectorStore
    simple_success = test_simple_vector_store()
    
    # Test fallback behavior
    fallback_success = test_fallback_behavior()
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"Qdrant Test: {'Success' if qdrant_success else 'Failed or Skipped'}")
    logger.info(f"ChromaDB Test: {'Success' if chroma_success else 'Failed or Skipped'}")
    logger.info(f"SimpleVectorStore Test: {'Success' if simple_success else 'Failed or Skipped'}")
    logger.info(f"Fallback Test: {'Success' if fallback_success else 'Failed or Skipped'}")
    
    # Overall success
    if any([qdrant_success, chroma_success, simple_success]) and fallback_success:
        logger.info("VectorStore implementation is working correctly with at least one backend!")
        return True
    else:
        logger.warning("Some tests failed, check logs for details.")
        return False

if __name__ == "__main__":
    main() 