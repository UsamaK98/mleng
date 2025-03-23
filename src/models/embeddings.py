"""
Text embedding models for vectorizing text
"""
from typing import List, Dict, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import requests
import time

from config.config import VECTOR_SIZE, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


class OllamaEmbeddingModel:
    """
    Model for generating text embeddings using Ollama API with nomic-embed-text
    """
    def __init__(
        self, 
        model_name: str = OLLAMA_EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        vector_size: int = VECTOR_SIZE,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Ollama embedding model
        
        Args:
            model_name: Name of the Ollama model to use (default: nomic-embed-text)
            base_url: Ollama API base URL
            vector_size: Size of embedding vectors (may be ignored based on model)
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.base_url = base_url
        self.vector_size = vector_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Check if model is available in Ollama
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json()["models"]]
                if model_name not in available_models:
                    print(f"Warning: Model '{model_name}' not found in Ollama. Available models: {available_models}")
                    print(f"Please run 'ollama pull {model_name}' to download the model")
                else:
                    print(f"Using Ollama embedding model: {model_name}")
                    # Verify embedding dimensions with a test call
                    test_embedding = self.embed_text("Test text for dimension verification")
                    print(f"Embedding dimensions: {len(test_embedding)}")
            else:
                print(f"Warning: Failed to get model list from Ollama API. Status code: {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama API: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array with embedding
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text,
                        "options": {
                            "temperature": 0.0  # Use deterministic embeddings
                        }
                    },
                    timeout=30  # Add a timeout to prevent hanging
                )
                response.raise_for_status()
                data = response.json()
                return np.array(data["embedding"])
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error generating embedding (attempt {attempt+1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    print(f"Failed to generate embedding after {self.max_retries} attempts: {e}")
                    # Return zero vector as fallback
                    return np.zeros(self.vector_size)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array with embeddings (shape: [len(texts), vector_size])
        """
        all_embeddings = []
        
        # Show progress for large batches
        total_texts = len(texts)
        if total_texts > batch_size:
            print(f"Generating embeddings for {total_texts} texts in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            # Show progress for large batches
            if total_texts > batch_size:
                print(f"Processing batch {i//batch_size + 1}/{(total_texts-1)//batch_size + 1} ({len(batch)} texts)")
            
            for text in batch:
                embedding = self.embed_text(text)
                batch_embeddings.append(embedding)
                # Small delay to avoid overwhelming the API
                time.sleep(0.05)
            
            all_embeddings.extend(batch_embeddings)
        
        result = np.array(all_embeddings)
        
        # Verify dimensions
        if result.shape[0] != len(texts):
            print(f"Warning: Expected {len(texts)} embeddings, but got {result.shape[0]}")
        
        return result
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            
        Returns:
            List of chunk dictionaries with added 'embedding' key
        """
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts)
        
        # Add embeddings to chunks
        for i, embedding in enumerate(embeddings):
            chunks[i]["embedding"] = embedding
        
        return chunks


class EmbeddingModel:
    """
    Model for generating text embeddings using SentenceTransformers
    """
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        vector_size: int = VECTOR_SIZE,
        device: str = None
    ):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the sentence-transformers model
            vector_size: Size of embedding vectors
            device: Device to use for inference ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.vector_size = vector_size
        
        # Auto-detect the device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device} for embeddings")
        
        # Load the model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Verify model dimensions
        if self.model.get_sentence_embedding_dimension() != vector_size:
            print(f"Warning: Model {model_name} outputs vectors of size "
                  f"{self.model.get_sentence_embedding_dimension()}, "
                  f"but {vector_size} was specified")
            
            self.vector_size = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array with embedding
        """
        return self.model.encode(text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array with embeddings (shape: [len(texts), vector_size])
        """
        return self.model.encode(texts, batch_size=batch_size)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            
        Returns:
            List of chunk dictionaries with added 'embedding' key
        """
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts)
        
        # Add embeddings to chunks
        for i, embedding in enumerate(embeddings):
            chunks[i]["embedding"] = embedding
        
        return chunks 