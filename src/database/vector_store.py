"""
Vector database interface for storing and retrieving document embeddings
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    VECTOR_SIZE
)


class VectorStore:
    """
    Interface for Qdrant vector database
    """
    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = QDRANT_COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE
    ):
        """
        Initialize the vector store
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            vector_size: Size of embedding vectors
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port)
        
        # Check if collection exists, create if not
        self._ensure_collection()
    
    def _ensure_collection(self):
        """
        Ensure that the collection exists, create it if not
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            print(f"Creating collection: {self.collection_name}")
            
            # Create the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            # Add payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.speaker",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.date",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add chunks to the vector store
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'metadata', and 'embedding' keys
            
        Returns:
            List of IDs of the added points
        """
        # Prepare points for batch insertion
        points = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Use integer ID
            point_id = i
            ids.append(point_id)
            
            # Create the point
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk["embedding"].tolist(),
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"]
                    }
                )
            )
        
        # Insert points in batches
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Added {len(points)} points to {self.collection_name}")
        
        return ids
    
    def search(
        self,
        query_vector: np.ndarray,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection
        
        Args:
            query_vector: Query embedding vector
            filter_params: Optional filters to apply (e.g., speaker, date)
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        # Build filter if provided
        search_filter = None
        if filter_params:
            filter_conditions = []
            
            if "speaker" in filter_params:
                filter_conditions.append(
                    models.FieldCondition(
                        key="metadata.speaker",
                        match=models.MatchValue(value=filter_params["speaker"])
                    )
                )
            
            if "date" in filter_params:
                filter_conditions.append(
                    models.FieldCondition(
                        key="metadata.date",
                        match=models.MatchValue(value=filter_params["date"])
                    )
                )
            
            if filter_conditions:
                search_filter = models.Filter(
                    must=filter_conditions
                )
        
        # Perform the search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit
        )
        
        # Format the results
        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload["text"],
                "metadata": hit.payload["metadata"]
            })
        
        return results 