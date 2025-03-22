"""
Text embedding models for vectorizing text
"""
from typing import List, Dict, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from config.config import VECTOR_SIZE


class EmbeddingModel:
    """
    Model for generating text embeddings
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