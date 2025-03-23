"""
Vector storage module for Parliamentary Meeting Analyzer.

This module provides functionality for storing and retrieving
vector embeddings for parliamentary data using either Qdrant
or ChromaDB as a fallback.
"""

import os
import time
import uuid
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from src.utils.logging import logger, log_inference
from src.utils.config import config_manager
from src.services.ollama import OllamaService

# Try to import Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("Qdrant client is not available. Will use ChromaDB as fallback.")
    QDRANT_AVAILABLE = False

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB is not available. Install it with 'pip install chromadb'")
    CHROMA_AVAILABLE = False

class VectorStore:
    """Vector storage for parliamentary data using Qdrant or ChromaDB."""
    
    def __init__(
        self,
        collection_name: str = "parliament",
        ollama_service: Optional[OllamaService] = None,
        use_qdrant: bool = True
    ):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the collection.
            ollama_service: OllamaService instance for embedding generation.
            use_qdrant: Whether to try using Qdrant first (if available).
        """
        self.collection_name = collection_name
        
        # Get configuration
        self.config = config_manager.config
        
        # Initialize Ollama service if not provided
        if ollama_service is None:
            try:
                from src.services.ollama import OllamaService
                self.ollama_service = OllamaService()
                logger.info("Initialized OllamaService for VectorStore")
            except Exception as e:
                logger.error(f"Failed to initialize OllamaService: {str(e)}")
                self.ollama_service = None
        else:
            self.ollama_service = ollama_service
            
        # Create cache directory
        self.cache_dir = Path(self.config.cache_dir) / "vector_store"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store client
        self.client = None
        self.chroma_client = None
        self.collection = None
        self.using_qdrant = False
        self.using_chroma = False
        
        # Try to initialize Qdrant first if requested
        if use_qdrant and QDRANT_AVAILABLE:
            try:
                self._init_qdrant()
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant: {str(e)}. Falling back to ChromaDB.")
                use_qdrant = False
        
        # Fall back to ChromaDB if Qdrant is not available or initialization failed
        if not use_qdrant and CHROMA_AVAILABLE:
            try:
                self._init_chromadb()
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {str(e)}")
                raise RuntimeError("Failed to initialize any vector store backend") from e
        
        if not self.using_qdrant and not self.using_chroma:
            logger.error("No vector database backend is available")
            raise RuntimeError(
                "No vector database backend is available. Install either qdrant-client or chromadb."
            )
            
        logger.info(f"Initialized VectorStore with {'Qdrant' if self.using_qdrant else 'ChromaDB'}")
    
    def _init_qdrant(self):
        """Initialize Qdrant client."""
        qdrant_config = self.config.qdrant
        
        if qdrant_config.use_local:
            # Use local file-based storage
            local_path = qdrant_config.local_path
            os.makedirs(local_path, exist_ok=True)
            
            self.client = QdrantClient(
                path=local_path,
                prefer_grpc=qdrant_config.prefer_grpc,
                timeout=qdrant_config.timeout
            )
            logger.info(f"Initialized Qdrant client with local storage at {local_path}")
        else:
            # Use remote Qdrant server
            self.client = QdrantClient(
                host=qdrant_config.host,
                port=qdrant_config.port,
                prefer_grpc=qdrant_config.prefer_grpc,
                timeout=qdrant_config.timeout
            )
            logger.info(f"Initialized Qdrant client at {qdrant_config.host}:{qdrant_config.port}")
        
        # Test connection
        try:
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {str(e)}")
            raise
        
        # Check if collection exists or create it
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=qdrant_config.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        self.using_qdrant = True
    
    def _init_chromadb(self):
        """Initialize ChromaDB client."""
        # Create persistent client with local storage
        chroma_path = os.path.join(self.config.cache_dir, "chroma_db")
        os.makedirs(chroma_path, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        logger.info(f"Initialized ChromaDB client with storage at {chroma_path}")
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._get_chroma_embedding_function()
        )
        logger.info(f"Using ChromaDB collection: {self.collection_name} with {self.collection.count()} items")
        
        self.using_chroma = True
    
    def _get_chroma_embedding_function(self):
        """Get embedding function for ChromaDB."""
        # Define a custom embedding function that uses our Ollama service
        class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, ollama_service):
                super().__init__()
                self.ollama_service = ollama_service
            
            def __call__(self, texts):
                embeddings = []
                for text in texts:
                    embedding = self.ollama_service.generate_embedding(text)
                    embeddings.append(embedding)
                return embeddings
        
        return OllamaEmbeddingFunction(self.ollama_service)
    
    def store_parliamentary_data(self, data: pd.DataFrame) -> bool:
        """Store parliamentary data in the vector store.
        
        Args:
            data: DataFrame containing parliamentary data.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.ollama_service:
            logger.error("OllamaService is required for generating embeddings")
            return False
        
        if self.using_qdrant:
            return self._store_qdrant(data)
        elif self.using_chroma:
            return self._store_chromadb(data)
        else:
            logger.error("No vector database backend available")
            return False
    
    def _store_qdrant(self, data: pd.DataFrame) -> bool:
        """Store data in Qdrant."""
        try:
            batch_size = self.config.qdrant.batch_size
            total_points = 0
            
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                
                points = []
                for _, row in batch.iterrows():
                    # Generate embedding for content
                    content = row["Content"]
                    embedding = self.ollama_service.generate_embedding(content)
                    
                    # Create point
                    point_id = str(uuid.uuid4())
                    
                    # Create payload with all row data
                    payload = {col: row[col] for col in row.index}
                    
                    # Add entry_id for retrieval
                    payload["entry_id"] = point_id
                    
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    )
                
                # Upload batch
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    total_points += len(points)
                    logger.info(f"Stored batch of {len(points)} embeddings ({i+1}-{i+len(batch)} of {len(data)})")
            
            logger.info(f"Successfully stored {total_points} embeddings in Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error storing data in Qdrant: {str(e)}")
            return False
    
    def _store_chromadb(self, data: pd.DataFrame) -> bool:
        """Store data in ChromaDB."""
        try:
            batch_size = 100  # ChromaDB batch size
            total_points = 0
            
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                
                documents = []
                metadatas = []
                ids = []
                
                for _, row in batch.iterrows():
                    # Get content
                    content = row["Content"]
                    
                    # Create unique ID
                    point_id = str(uuid.uuid4())
                    
                    # Create metadata with all row data
                    metadata = {col: str(row[col]) for col in row.index}
                    metadata["entry_id"] = point_id
                    
                    documents.append(content)
                    metadatas.append(metadata)
                    ids.append(point_id)
                
                # Upload batch
                if documents:
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    total_points += len(documents)
                    logger.info(f"Stored batch of {len(documents)} embeddings ({i+1}-{i+len(batch)} of {len(data)})")
            
            logger.info(f"Successfully stored {total_points} embeddings in ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error storing data in ChromaDB: {str(e)}")
            return False
    
    def search_similar(
        self, 
        query_text: str, 
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar content.
        
        Args:
            query_text: Text to search for.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score to include in results.
            
        Returns:
            List of results, each containing score and payload.
        """
        if not self.ollama_service:
            logger.error("OllamaService is required for generating embeddings")
            return []
        
        if self.using_qdrant:
            return self._search_qdrant(query_text, top_k, score_threshold)
        elif self.using_chroma:
            return self._search_chromadb(query_text, top_k)
        else:
            logger.error("No vector database backend available")
            return []
    
    def _search_qdrant(
        self, 
        query_text: str, 
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar content in Qdrant."""
        try:
            # Generate embedding for query
            query_embedding = self.ollama_service.generate_embedding(query_text)
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    "score": point.score,
                    "payload": point.payload
                })
            
            logger.info(f"Found {len(results)} similar items for query: '{query_text[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            return []
    
    def _search_chromadb(
        self, 
        query_text: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar content in ChromaDB."""
        try:
            # Search using query text
            search_result = self.collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            
            # Format results to match Qdrant format
            results = []
            for i in range(len(search_result["ids"][0])):
                results.append({
                    "score": 1.0 - (float(search_result["distances"][0][i]) if "distances" in search_result else 0),
                    "payload": search_result["metadatas"][0][i]
                })
            
            logger.info(f"Found {len(results)} similar items for query: '{query_text[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {str(e)}")
            return []

# Usage example:
# from src.storage.vector_db import VectorStore
# from src.data.loader import ParliamentaryDataLoader
# 
# # Load parliamentary data
# loader = ParliamentaryDataLoader()
# loader.load_data()
# 
# # Get a session's data
# session_data = loader.get_session_data("2024-09-10")
# 
# # Initialize vector store
# vector_store = VectorStore()
# 
# # Store data in vector database
# vector_store.store_parliamentary_data(session_data)
# 
# # Search for similar content
# results = vector_store.search_similar("discussion about climate change", top_k=3)
# for result in results:
#     print(f"Score: {result['score']}, Speaker: {result['Speaker']}")
#     print(f"Content: {result['Content'][:100]}...")
#     print() 