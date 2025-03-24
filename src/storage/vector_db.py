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
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from uuid import uuid4, uuid5, NAMESPACE_DNS
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
        """Initialize vector database.
        
        Args:
            collection_name: Name of the collection to use.
            host: Qdrant host address. If None, uses the configured value.
            port: Qdrant port. If None, uses the configured value.
            ollama_service: Optional OllamaService instance for embedding generation.
            use_chromadb_fallback: Whether to use ChromaDB if Qdrant is unavailable.
        """
        self.collection_name = collection_name
        self.qdrant_host = host or config_manager.config.qdrant.host
        self.qdrant_port = port or config_manager.config.qdrant.port
        
        # Set embedding dimension based on model
        self.ollama_service = ollama_service
        
        # Attempt to get actual embedding dimension by testing the model
        if self.ollama_service:
            try:
                logger.info("Testing embedding model to determine actual dimension")
                test_embedding = self.ollama_service.get_embeddings(["Test embedding dimension"])[0]
                actual_dim = len(test_embedding)
                if actual_dim != config_manager.config.ollama.embedding_dim:
                    logger.warning(f"Embedding dimension mismatch: config specifies {config_manager.config.ollama.embedding_dim}, but model produces {actual_dim}")
                    # Update config value to match actual dimension
                    config_manager.config.ollama.embedding_dim = actual_dim
                    config_manager.config.qdrant.vector_size = actual_dim
                self.embedding_dim = actual_dim
                logger.info(f"Using embedding dimension: {self.embedding_dim}")
            except Exception as e:
                logger.error(f"Error determining embedding dimension: {str(e)}")
                self.embedding_dim = config_manager.config.ollama.embedding_dim
                logger.info(f"Falling back to configured embedding dimension: {self.embedding_dim}")
        else:
            self.embedding_dim = config_manager.config.ollama.embedding_dim
        
        self.use_chromadb_fallback = use_chromadb_fallback
        self.using_fallback = False
        self.client = None
        self.chroma_client = None
        self.collection = None
        
        # Create cache directories
        self.cache_dir = Path(config_manager.config.processed_data_dir) / "vector_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file for tracking cache state
        self.metadata_file = self.cache_dir / "vector_metadata.json"
        self._initialize_metadata()
        
        # Initialize database
        self.initialize_vector_db()
    
    def _initialize_metadata(self):
        """Initialize or load vector cache metadata."""
        if not self.metadata_file.exists():
            metadata = {
                "version": "1.0.0",
                "embedding_model": config_manager.config.ollama.embed_model,
                "embedding_dim": self.embedding_dim,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "collections": {},
                "cache_files": {}
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                # Check if embedding model changed
                if metadata.get("embedding_model") != config_manager.config.ollama.embed_model:
                    logger.warning(f"Embedding model changed from {metadata.get('embedding_model')} to {config_manager.config.ollama.embed_model}")
                    metadata["embedding_model"] = config_manager.config.ollama.embed_model
                    metadata["embedding_dim"] = self.embedding_dim
                    metadata["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(self.metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.error(f"Error loading vector cache metadata: {str(e)}")
                # Re-initialize if corrupt
                metadata = {
                    "version": "1.0.0",
                    "embedding_model": config_manager.config.ollama.embed_model,
                    "embedding_dim": self.embedding_dim,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "collections": {},
                    "cache_files": {}
                }
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
    
    def _update_metadata(self, df_hash: str, file_path: str, num_vectors: int):
        """Update vector cache metadata."""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata["cache_files"][df_hash] = {
                "path": str(file_path),
                "num_vectors": num_vectors,
                "collection": self.collection_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Update collection info
            if self.collection_name not in metadata["collections"]:
                metadata["collections"][self.collection_name] = {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_vectors": num_vectors,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                metadata["collections"][self.collection_name]["num_vectors"] = num_vectors
                metadata["collections"][self.collection_name]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating vector cache metadata: {str(e)}")

    def initialize_vector_db(self) -> bool:
        """Initialize vector database connection.
        
        Returns:
            Whether initialization was successful.
        """
        # Try Qdrant first if available
        if QDRANT_AVAILABLE:
            try:
                # Create data directory for Qdrant
                qdrant_data_dir = Path("data") / "qdrant_data"
                qdrant_data_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if we're using local Qdrant or remote
                if self.qdrant_host == "localhost" or self.qdrant_host == "127.0.0.1":
                    logger.info(f"Initializing local Qdrant client at port {self.qdrant_port}")
                    self.client = QdrantClient(
                        path=str(qdrant_data_dir.absolute()),  # Use local storage for persistence
                        port=self.qdrant_port                  # Enable REST API for debugging
                    )
                else:
                    logger.info(f"Initializing remote Qdrant client at {self.qdrant_host}:{self.qdrant_port}")
                    self.client = QdrantClient(
                        host=self.qdrant_host,
                        port=self.qdrant_port
                    )
                
                # Check if collection exists, if not create it
                collections = self.client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if self.collection_name not in collection_names:
                    logger.info(f"Creating new collection: {self.collection_name}")
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.embedding_dim,
                            distance=models.Distance.COSINE
                        )
                    )
                
                # Get collection info to verify
                self.client.get_collection(collection_name=self.collection_name)
                logger.info(f"Successfully connected to Qdrant collection: {self.collection_name}")
                
                # Add collection info to metadata
                self._register_collection_in_metadata()
                
                return True
            
            except Exception as e:
                logger.error(f"Error initializing Qdrant: {str(e)}")
                if not self.use_chromadb_fallback:
                    return False
                logger.warning("Falling back to ChromaDB")
        
        # Fall back to ChromaDB if Qdrant failed or isn't available
        if CHROMADB_AVAILABLE:
            try:
                self.using_fallback = True
                
                # Create data directory for ChromaDB
                chroma_data_dir = Path("data") / "chromadb_data"
                chroma_data_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize ChromaDB with persistent storage
                logger.info(f"Initializing ChromaDB client with persistent storage at {chroma_data_dir}")
                self.chroma_client = chromadb.PersistentClient(path=str(chroma_data_dir.absolute()))
                
                # Check if collection exists, if not create it
                try:
                    self.collection = self.chroma_client.get_collection(name=self.collection_name)
                    logger.info(f"Found existing ChromaDB collection: {self.collection_name}")
                except ValueError:
                    logger.info(f"Creating new ChromaDB collection: {self.collection_name}")
                    self.collection = self.chroma_client.create_collection(name=self.collection_name)
                
                # Add collection info to metadata
                self._register_collection_in_metadata()
                
                return True
                
            except Exception as e:
                logger.error(f"Error initializing ChromaDB: {str(e)}")
                return False
        
        logger.error("Neither Qdrant nor ChromaDB are available")
        return False
    
    def _register_collection_in_metadata(self):
        """Register the collection in metadata."""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Get current vector count
            vector_count = 0
            if not self.using_fallback:
                vector_count = self.client.count(collection_name=self.collection_name).count
            elif self.collection:
                vector_count = len(self.collection.get()["ids"]) if self.collection.get()["ids"] else 0
            
            # Update or create collection entry
            if self.collection_name not in metadata["collections"]:
                metadata["collections"][self.collection_name] = {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_vectors": vector_count,
                    "using_fallback": self.using_fallback,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                metadata["collections"][self.collection_name]["num_vectors"] = vector_count
                metadata["collections"][self.collection_name]["using_fallback"] = self.using_fallback
                metadata["collections"][self.collection_name]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error registering collection in metadata: {str(e)}")

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
                                
                                # Validate embedding format and dimensions
                                if not isinstance(embedding, list):
                                    logger.error(f"Invalid embedding format for entry_id {entry_id}: {type(embedding)}, skipping entry")
                                    continue
                                
                                # Check embedding dimensions - do not pad
                                if len(embedding) != self.embedding_dim:
                                    logger.error(f"Embedding dimension mismatch for entry_id {entry_id}: expected {self.embedding_dim}, got {len(embedding)}")
                                    logger.error(f"Skipping entry as padding is not allowed")
                                    continue
                                
                                # Convert each element to float
                                try:
                                    embedding = [float(e) if not isinstance(e, float) else e for e in embedding]
                                except (ValueError, TypeError):
                                    logger.error(f"Invalid values in embedding for entry_id {entry_id}, skipping entry")
                                    continue
                                
                                # Store validated embedding
                                embeddings_map[entry_id] = embedding
                                
                        except Exception as e:
                            logger.error(f"Error generating embeddings: {str(e)}")
                            continue
                    
                    # Prepare points for this batch
                    for j, entry_id in enumerate(batch_ids):
                        if entry_id not in embeddings_map or not batch_texts[j].strip():
                            continue
                        
                        # Get embedding
                        embedding = embeddings_map[entry_id]
                        
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
                        try:
                            # Convert to UUID for Qdrant (which requires UUID format)
                            uuid_id = entry_id
                            # If entry_id is a number or doesn't match UUID format, create a deterministic UUID
                            if entry_id.isdigit() or (len(entry_id) < 32):
                                # Create a deterministic UUID based on the entry_id using uuid5
                                uuid_id = str(uuid5(NAMESPACE_DNS, entry_id))
                            
                            point = models.PointStruct(
                                id=uuid_id,
                                vector=embedding,
                                payload=metadata
                            )
                            # Store the original entry_id in the payload for reference
                            point.payload["original_id"] = entry_id
                            points_to_upsert.append(point)
                        except Exception as e:
                            logger.error(f"Error creating point for entry_id {entry_id}: {str(e)}")
                            logger.debug(f"Embedding type: {type(embedding)}, first few values: {embedding[:5] if isinstance(embedding, list) else 'not a list'}")
                
                # Save embeddings cache
                with open(cache_file, 'w') as f:
                    json.dump({'embeddings': embeddings_map}, f)
                
                # Upsert points to Qdrant
                if points_to_upsert:
                    logger.info(f"Upserting {len(points_to_upsert)} points to Qdrant")
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=points_to_upsert
                        )
                    except Exception as e:
                        logger.error(f"Error upserting points to Qdrant: {str(e)}")
                        return False
                
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Stored {len(points_to_upsert)} vectors in {duration_ms/1000:.2f}s")
                
                # Update metadata
                self._update_metadata(df_hash, str(cache_file), len(points_to_upsert))
                
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
                    self.collection.upsert(
                        ids=batch["ids"],
                        documents=batch["documents"],
                        metadatas=batch["metadatas"]
                    )
                    total_docs += len(batch["ids"])
                
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Stored {total_docs} documents in ChromaDB in {duration_ms/1000:.2f}s")
                
                # Update metadata
                self._update_metadata(df_hash, "", total_docs)
                
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
                
                # Check if query embedding dimension matches the expected dimension
                if len(query_embedding[0]) != self.embedding_dim:
                    logger.warning(f"Query embedding dimension mismatch: got {len(query_embedding[0])}, expected {self.embedding_dim}")
                    # Normalize dimensions to match expected
                    emb = query_embedding[0]
                    if len(emb) < self.embedding_dim:
                        # Pad with zeros if too short
                        logger.info(f"Padding query embedding from {len(emb)} to {self.embedding_dim} dimensions")
                        query_embedding[0] = emb + [0.0] * (self.embedding_dim - len(emb))
                    else:
                        # Truncate if too long
                        logger.info(f"Truncating query embedding from {len(emb)} to {self.embedding_dim} dimensions")
                        query_embedding[0] = emb[:self.embedding_dim]
                
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
                    query_vector=query_embedding[0],
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
                search_results = self.collection.query(
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
                    results = self.collection.get(
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
                    collection_info = self.collection.get()
                    
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
            return f"VectorStore(collection={self.collection_name}, backend=qdrant, host={self.qdrant_host}, port={self.qdrant_port})"
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