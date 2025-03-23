"""
Vector database module for the Parliamentary Meeting Analyzer.

This module provides integration with Qdrant vector database for storing and
retrieving vector embeddings of parliamentary meeting data. It also includes
ChromaDB as a fallback if Qdrant is unavailable.
"""

import os
import time
import json
import sys
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from uuid import uuid4
from pathlib import Path
from tqdm import tqdm

# Import Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Import ChromaDB (fallback)
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from src.utils.logging import logger
from src.utils.config import config_manager
from src.services.ollama import OllamaService

class VectorStore:
    """Vector database for storing and retrieving embeddings."""
    
    def __init__(
        self,
        collection_name: str = "parliament",
        host: Optional[str] = None,
        port: Optional[int] = None,
        ollama_service: Optional[Any] = None,
        use_chromadb_fallback: bool = True
    ):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use
            host: Qdrant host
            port: Qdrant port
            ollama_service: OllamaService instance for embedding generation
            use_chromadb_fallback: Whether to use ChromaDB as fallback if Qdrant fails
        """
        # Get configuration
        self.config = config_manager.config.qdrant
        self.chromadb_config = config_manager.config.chromadb
        self.host = host or self.config.host
        self.port = port or self.config.port
        self.collection_name = collection_name
        self.vector_size = config_manager.config.ollama.embedding_dim
        self.use_chromadb_fallback = use_chromadb_fallback
        self.using_fallback = False
        
        # Initialize Ollama service if not provided
        if ollama_service is None:
            try:
                from src.services.ollama import OllamaService
                self.ollama_service = OllamaService()
                logger.info("Initialized OllamaService for vector embeddings")
            except Exception as e:
                logger.error(f"Failed to initialize OllamaService: {str(e)}")
                self.ollama_service = None
                sys.exit(1)  # Exit if Ollama service fails
        else:
            self.ollama_service = ollama_service
        
        # Create cache directory
        self.cache_dir = Path(config_manager.config.processed_data_dir) / "vector_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to initialize Qdrant first
        self.client = None
        if QDRANT_AVAILABLE:
            try:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    prefer_grpc=self.config.prefer_grpc,
                    timeout=self.config.timeout
                )
                self._ensure_collection_exists()
                logger.info(f"Initialized VectorStore with Qdrant collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant: {str(e)}")
                self.client = None
                
                # Try ChromaDB fallback if enabled
                if not self.use_chromadb_fallback:
                    logger.error("Fallback to ChromaDB is disabled. Exiting.")
                    sys.exit(1)
        else:
            logger.warning("Qdrant client not available. Will try ChromaDB fallback.")
        
        # Use ChromaDB as fallback if Qdrant fails or is not available
        self.chroma_client = None
        self.chroma_collection = None
        
        if self.client is None and CHROMADB_AVAILABLE and self.use_chromadb_fallback:
            try:
                persist_dir = Path(self.chromadb_config.persist_directory)
                persist_dir.mkdir(parents=True, exist_ok=True)
                
                self.chroma_client = chromadb.PersistentClient(path=str(persist_dir))
                
                # Create a custom embedding function using Ollama
                class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
                    def __init__(self, ollama_service):
                        self.ollama_service = ollama_service
                    
                    def __call__(self, texts):
                        return self.ollama_service.get_embeddings(texts)
                
                # Get existing collections
                collections = self.chroma_client.list_collections()
                collection_names = [c.name for c in collections]
                
                if self.collection_name not in collection_names:
                    self.chroma_collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        embedding_function=OllamaEmbeddingFunction(self.ollama_service)
                    )
                else:
                    self.chroma_collection = self.chroma_client.get_collection(
                        name=self.collection_name,
                        embedding_function=OllamaEmbeddingFunction(self.ollama_service)
                    )
                
                self.using_fallback = True
                logger.info(f"Successfully initialized ChromaDB fallback with collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB fallback: {str(e)}")
                logger.error("Both Qdrant and ChromaDB initialization failed. Exiting.")
                sys.exit(1)
        elif self.client is None and (not CHROMADB_AVAILABLE or not self.use_chromadb_fallback):
            logger.error("ChromaDB fallback not available or disabled. Exiting.")
            sys.exit(1)
            
    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
    
    def store_parliamentary_data(
        self,
        df: pd.DataFrame,
        text_column: str = 'Content',
        metadata_columns: Optional[List[str]] = None,
        batch_size: int = 32,
        use_cache: bool = True
    ) -> bool:
        """Store parliamentary data in the vector database.
        
        Args:
            df: DataFrame containing parliamentary data.
            text_column: Name of the column containing text to embed.
            metadata_columns: List of column names to include as metadata.
            batch_size: Number of records to process in each batch.
            use_cache: Whether to use cached embeddings if available.
            
        Returns:
            True if data stored successfully, False otherwise.
        """
        if df.empty or text_column not in df.columns:
            logger.warning(f"DataFrame is empty or missing {text_column} column")
            return False
        
        # Check if we have entry_id for unique identification
        if 'entry_id' not in df.columns:
            df = df.copy()
            df['entry_id'] = df.index.astype(str)
        
        # Define default metadata columns if not provided
        if metadata_columns is None:
            metadata_columns = [
                'Date', 'Timestamp', 'Speaker', 'Role', 'entry_id', 'Filename'
            ]
            # Add entity columns if they exist
            entity_cols = [col for col in df.columns if col.startswith('entities_')]
            metadata_columns.extend(entity_cols)
        
        # Prepare cache file path based on dataframe hash
        df_hash = str(hash(frozenset(df['entry_id'].astype(str))))[:10]
        cache_file = self.cache_dir / f"embeddings_{df_hash}.json"
        
        # Try to load embeddings from cache if enabled
        embeddings_map = {}
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                embeddings_map = cache_data.get('embeddings', {})
                logger.info(f"Loaded {len(embeddings_map)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {str(e)}")
        
        start_time = time.time()
        texts = df[text_column].fillna("").tolist()
        entry_ids = df['entry_id'].astype(str).tolist()
        
        # If using Qdrant
        if not self.using_fallback:
            # Track new or updated records
            points_to_upsert = []
            
            try:
                # Process in batches
                for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                    batch_texts = texts[i:i+batch_size]
                    batch_ids = entry_ids[i:i+batch_size]
                    batch_indices = list(range(i, min(i+batch_size, len(texts))))
                    
                    # Check which records need embedding generation
                    new_texts = []
                    new_indices = []
                    
                    for j, (text, entry_id) in enumerate(zip(batch_texts, batch_ids)):
                        if not text or not text.strip():
                            continue
                        
                        if entry_id not in embeddings_map:
                            new_texts.append(text)
                            new_indices.append(batch_indices[j])
                    
                    # Generate embeddings for new texts
                    if new_texts:
                        try:
                            new_embeddings = self.ollama_service.get_embeddings(new_texts)
                            
                            # Store new embeddings in cache
                            for idx, embedding in zip(new_indices, new_embeddings):
                                entry_id = entry_ids[idx]
                                embeddings_map[entry_id] = embedding
                        
                        except Exception as e:
                            logger.error(f"Error generating embeddings: {str(e)}")
                            continue
                    
                    # Prepare points for this batch
                    for j, entry_id in enumerate(batch_ids):
                        if entry_id not in embeddings_map or not batch_texts[j].strip():
                            continue
                        
                        # Create metadata dictionary
                        metadata = {}
                        row = df.iloc[batch_indices[j]]
                        
                        for col in metadata_columns:
                            if col in row.index:
                                if pd.isna(row[col]):
                                    continue
                                
                                # Convert date objects to strings
                                if isinstance(row[col], pd.Timestamp):
                                    metadata[col] = row[col].strftime('%Y-%m-%d')
                                else:
                                    metadata[col] = str(row[col])
                        
                        # Add the original text
                        metadata[text_column] = batch_texts[j]
                        
                        # Create point
                        point = models.PointStruct(
                            id=entry_id,
                            vector=embeddings_map[entry_id],
                            payload=metadata
                        )
                        
                        points_to_upsert.append(point)
                
                # Save embeddings cache
                with open(cache_file, 'w') as f:
                    json.dump({'embeddings': embeddings_map}, f)
                
                # Upsert points to Qdrant
                if points_to_upsert:
                    logger.info(f"Upserting {len(points_to_upsert)} points to Qdrant")
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points_to_upsert
                    )
                
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Stored {len(points_to_upsert)} vectors in {duration_ms/1000:.2f}s")
                
                return True
                
            except Exception as e:
                logger.error(f"Error storing data in Qdrant: {str(e)}")
                return False
                
        # If using ChromaDB (fallback)
        else:
            try:
                # Process in batches
                items_to_upsert = []
                
                for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings for ChromaDB"):
                    batch_texts = texts[i:i+batch_size]
                    batch_ids = entry_ids[i:i+batch_size]
                    batch_indices = list(range(i, min(i+batch_size, len(texts))))
                    
                    batch_documents = []
                    batch_metadatas = []
                    batch_ids_to_add = []
                    
                    for j, (text, entry_id) in enumerate(zip(batch_texts, batch_ids)):
                        if not text or not text.strip():
                            continue
                        
                        # Create metadata dictionary
                        metadata = {}
                        row = df.iloc[batch_indices[j]]
                        
                        for col in metadata_columns:
                            if col in row.index:
                                if pd.isna(row[col]):
                                    continue
                                
                                # Convert date objects to strings
                                if isinstance(row[col], pd.Timestamp):
                                    metadata[col] = row[col].strftime('%Y-%m-%d')
                                else:
                                    metadata[col] = str(row[col])
                        
                        batch_documents.append(text)
                        batch_metadatas.append(metadata)
                        batch_ids_to_add.append(entry_id)
                    
                    if batch_documents:
                        items_to_upsert.append({
                            "ids": batch_ids_to_add,
                            "documents": batch_documents,
                            "metadatas": batch_metadatas
                        })
                
                # Upsert to ChromaDB in batches
                total_docs = 0
                for batch in items_to_upsert:
                    self.chroma_collection.upsert(
                        ids=batch["ids"],
                        documents=batch["documents"],
                        metadatas=batch["metadatas"]
                    )
                    total_docs += len(batch["ids"])
                
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Stored {total_docs} documents in ChromaDB in {duration_ms/1000:.2f}s")
                
                return True
                
            except Exception as e:
                logger.error(f"Error storing data in ChromaDB: {str(e)}")
                return False
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for records similar to the query.
        
        Args:
            query: Text query to search for.
            top_k: Number of results to return.
            threshold: Similarity threshold (0-1). If None, uses the configured value.
            filter_dict: Dictionary of metadata filters to apply.
            
        Returns:
            List of similar records with metadata.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Use configured threshold if none provided
        if threshold is None:
            threshold = config_manager.config.graphrag.similarity_threshold
        
        start_time = time.time()
        
        # If using Qdrant
        if not self.using_fallback:
            try:
                # Generate embedding for query
                query_embedding = self.ollama_service.get_embeddings(query)
                
                # Convert filter_dict to Qdrant filter
                filter_obj = None
                if filter_dict:
                    conditions = []
                    for key, value in filter_dict.items():
                        if isinstance(value, list):
                            conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchAny(any=value)
                                )
                            )
                        else:
                            conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchValue(value=value)
                                )
                            )
                    
                    filter_obj = models.Filter(
                        must=conditions
                    )
                
                # Search Qdrant
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=threshold,
                    query_filter=filter_obj
                )
                
                # Format results
                results = []
                for result in search_results:
                    item = {
                        'score': result.score,
                        **result.payload
                    }
                    results.append(item)
                
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Search for '{query[:30]}...' returned {len(results)} results in {duration_ms:.2f}ms")
                
                return results
            
            except Exception as e:
                logger.error(f"Error searching Qdrant: {str(e)}")
                return []
                
        # If using ChromaDB (fallback)
        else:
            try:
                # Convert filter_dict to ChromaDB filter format
                where_filter = None
                if filter_dict:
                    where_filter = {}
                    for key, value in filter_dict.items():
                        where_filter[key] = value
                
                # Search ChromaDB
                search_results = self.chroma_collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where_filter
                )
                
                # Format results
                results = []
                if search_results['documents'] and len(search_results['documents'][0]) > 0:
                    for i, (doc_id, doc, metadata, distance) in enumerate(zip(
                        search_results['ids'][0],
                        search_results['documents'][0],
                        search_results['metadatas'][0],
                        search_results['distances'][0]
                    )):
                        # ChromaDB returns distance, convert to similarity score (cosine)
                        similarity_score = 1.0 - distance
                        
                        # Skip if below threshold
                        if similarity_score < threshold:
                            continue
                            
                        item = {
                            'score': similarity_score,
                            'Content': doc,
                            'entry_id': doc_id,
                            **metadata
                        }
                        results.append(item)
                
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"ChromaDB search for '{query[:30]}...' returned {len(results)} results in {duration_ms:.2f}ms")
                
                return results
                
            except Exception as e:
                logger.error(f"Error searching ChromaDB: {str(e)}")
                return []
    
    def get_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a record by its entry_id.
        
        Args:
            entry_id: Entry ID to retrieve.
            
        Returns:
            Record with metadata if found, None otherwise.
        """
        try:
            # If using Qdrant
            if not self.using_fallback:
                try:
                    points = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=[entry_id]
                    )
                    
                    if not points:
                        return None
                    
                    point = points[0]
                    
                    return {
                        'entry_id': entry_id,
                        **point.payload
                    }
                except Exception as e:
                    logger.error(f"Error retrieving from Qdrant by ID: {str(e)}")
                    return None
            
            # If using ChromaDB (fallback)
            else:
                try:
                    results = self.chroma_collection.get(
                        ids=[entry_id],
                        include=["metadatas", "documents"]
                    )
                    
                    if not results or not results['ids'] or not results['ids'][0]:
                        return None
                    
                    return {
                        'entry_id': entry_id,
                        'Content': results['documents'][0],
                        **results['metadatas'][0]
                    }
                except Exception as e:
                    logger.error(f"Error retrieving from ChromaDB by ID: {str(e)}")
                    return None
        
        except Exception as e:
            logger.error(f"Error retrieving by ID {entry_id}: {str(e)}")
            return None
    
    def delete_collection(self) -> bool:
        """Delete the collection.
        
        Returns:
            True if deleted successfully, False otherwise.
        """
        try:
            # If using Qdrant
            if not self.using_fallback:
                try:
                    self.client.delete_collection(collection_name=self.collection_name)
                    logger.info(f"Deleted Qdrant collection: {self.collection_name}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting Qdrant collection: {str(e)}")
                    return False
            
            # If using ChromaDB (fallback)
            else:
                try:
                    self.chroma_client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted ChromaDB collection: {self.collection_name}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting ChromaDB collection: {str(e)}")
                    return False
        
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary with collection information.
        """
        try:
            # If using Qdrant
            if not self.using_fallback:
                try:
                    collection_info = self.client.get_collection(collection_name=self.collection_name)
                    points_count = self.client.count(collection_name=self.collection_name).count
                    
                    return {
                        'name': self.collection_name,
                        'vector_size': collection_info.config.params.vectors.size,
                        'distance': collection_info.config.params.vectors.distance.name,
                        'points_count': points_count,
                        'status': 'active',
                        'using_fallback': False,
                        'backend': 'qdrant'
                    }
                except Exception as e:
                    logger.error(f"Error getting Qdrant collection info: {str(e)}")
                    return {
                        'name': self.collection_name,
                        'status': 'error',
                        'error': str(e),
                        'using_fallback': False,
                        'backend': 'qdrant'
                    }
            
            # If using ChromaDB (fallback)
            else:
                try:
                    # Get collection info
                    collection_info = self.chroma_collection.get()
                    
                    return {
                        'name': self.collection_name,
                        'count': len(collection_info['ids']) if 'ids' in collection_info else 0,
                        'status': 'active',
                        'using_fallback': True,
                        'backend': 'chromadb'
                    }
                except Exception as e:
                    logger.error(f"Error getting ChromaDB collection info: {str(e)}")
                    return {
                        'name': self.collection_name,
                        'status': 'error',
                        'error': str(e),
                        'using_fallback': True,
                        'backend': 'chromadb'
                    }
        
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {
                'name': self.collection_name,
                'status': 'error',
                'error': str(e)
            }
            
    def __repr__(self) -> str:
        """Return a string representation of the vector store."""
        if not self.using_fallback:
            return f"VectorStore(collection={self.collection_name}, backend=qdrant, host={self.host}, port={self.port})"
        else:
            return f"VectorStore(collection={self.collection_name}, backend=chromadb, fallback=True)"

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