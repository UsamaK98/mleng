"""
Test script for Qdrant vector database connection.

This script performs basic connection tests for Qdrant
to diagnose connectivity issues.
"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config_manager

# Import Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    logger.error(
        "Qdrant client is required. Please install it using `pip install qdrant-client`."
    )
    sys.exit(1)

def test_qdrant_connection():
    """Test connection to Qdrant server."""
    # Get configuration
    config = config_manager.config.qdrant
    
    logger.info(f"Testing connection to Qdrant at {config.host}:{config.port}")
    
    # Try both HTTP and gRPC connections
    for prefer_grpc in [True, False]:
        try:
            logger.info(f"Trying connection with prefer_grpc={prefer_grpc}")
            client = QdrantClient(
                host=config.host,
                port=config.port,
                prefer_grpc=prefer_grpc,
                timeout=config.timeout
            )
            
            # Test connection by getting collections
            collections = client.get_collections()
            logger.info(f"Connection successful with prefer_grpc={prefer_grpc}")
            logger.info(f"Available collections: {[c.name for c in collections.collections]}")
            
            # Try creating a test collection
            test_collection_name = "test_connection"
            try:
                client.create_collection(
                    collection_name=test_collection_name,
                    vectors_config=models.VectorParams(
                        size=config.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Successfully created test collection: {test_collection_name}")
                
                # Clean up by deleting the test collection
                client.delete_collection(collection_name=test_collection_name)
                logger.info(f"Successfully deleted test collection: {test_collection_name}")
            except Exception as e:
                logger.warning(f"Could not create test collection: {str(e)}")
            
            return True
        except Exception as e:
            logger.error(f"Connection failed with prefer_grpc={prefer_grpc}: {str(e)}")
    
    logger.error("All connection attempts failed.")
    
    # Suggest possible solutions
    logger.info("Possible solutions:")
    logger.info("1. Ensure Qdrant server is running")
    logger.info("2. Verify that the port configuration (default: 6333) matches your Qdrant server")
    logger.info("3. Check if a firewall is blocking the connection")
    logger.info("4. Try using the local mode with a file-based storage")
    
    return False

def main():
    """Run the Qdrant connection test."""
    logger.info("Starting Qdrant connection test")
    test_qdrant_connection()
    logger.info("Test completed")

if __name__ == "__main__":
    main() 