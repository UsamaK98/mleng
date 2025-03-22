"""
Test script for Qdrant vector database
"""
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure Qdrant connection
HOST = "localhost"
PORT = 6333
COLLECTION_NAME = "test_collection"
VECTOR_SIZE = 384

def test_qdrant():
    """Test basic Qdrant operations"""
    print("Testing Qdrant connection...")
    
    # Initialize client
    client = QdrantClient(host=HOST, port=PORT)
    
    # List existing collections
    collections = client.get_collections().collections
    print(f"Found {len(collections)} collections:")
    for collection in collections:
        print(f"- {collection.name}")
    
    # Check if our test collection exists and recreate it
    collection_names = [c.name for c in collections]
    if COLLECTION_NAME in collection_names:
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(collection_name=COLLECTION_NAME)
    
    # Create a new collection
    print(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )
    
    # Insert a test vector
    print("Inserting test vectors...")
    vectors = [np.random.rand(VECTOR_SIZE).tolist() for _ in range(5)]
    
    points = [
        models.PointStruct(
            id=i,
            vector=vector,
            payload={"text": f"Test point {i}", "metadata": {"test": True}}
        )
        for i, vector in enumerate(vectors)
    ]
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    # Search for a vector
    print("Testing search...")
    query_vector = np.random.rand(VECTOR_SIZE).tolist()
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3
    )
    
    print(f"Found {len(search_results)} results")
    for i, hit in enumerate(search_results):
        print(f"Result {i+1}: ID={hit.id}, Score={hit.score:.4f}")
    
    return True

if __name__ == "__main__":
    if test_qdrant():
        print("\nQdrant test completed successfully!") 