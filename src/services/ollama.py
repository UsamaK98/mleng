"""
Ollama API service for the Parliamentary Meeting Analyzer.

This module provides a standardized interface for interacting with the Ollama API
for text generation and embedding.
"""

import os
import time
import json
import requests
from typing import List, Dict, Any, Optional, Tuple, Union

from src.utils.logging import logger
from src.utils.config import config_manager

class OllamaService:
    """
    Service for interacting with Ollama API.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the Ollama service.
        
        Args:
            base_url: Base URL for the Ollama API. If None, uses the config value.
        """
        # Get configuration
        self.base_url = base_url or config_manager.config.ollama.base_url
        self.embed_model = config_manager.config.ollama.embed_model
        self.generation_model = config_manager.config.ollama.generation_model
        self.max_tokens = config_manager.config.ollama.max_tokens
        self.temperature = config_manager.config.ollama.temperature
        self.timeout = config_manager.config.ollama.timeout_seconds
        self.embedding_dim = config_manager.config.ollama.embedding_dim
        
        logger.info(f"Initialized OllamaService with base URL: {self.base_url}")
        
        # Check connection to Ollama
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """
        Check connection to Ollama API.
        
        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json().get("models", [])]
                logger.info(f"Connected to Ollama API. Available models: {', '.join(available_models)}")
                
                # Check if requested models are available
                if self.embed_model not in available_models:
                    logger.warning(f"Embedding model {self.embed_model} not available in Ollama")
                
                if self.generation_model not in available_models:
                    logger.warning(f"Generation model {self.generation_model} not available in Ollama")
                
                return True
            else:
                logger.error(f"Failed to connect to Ollama API: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Ollama API: {str(e)}")
            return False
    
    def generate_text(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text using the configured model.
        
        Args:
            prompt: The prompt text.
            **kwargs: Additional parameters to pass to the Ollama API.
            
        Returns:
            Tuple containing the generated text and metadata.
        """
        start_time = time.time()
        
        # Default parameters
        params = {
            "model": kwargs.get("model", self.generation_model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens)
            }
        }
        
        # Add system prompt if provided
        if "system" in kwargs:
            params["system"] = kwargs["system"]
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # Log timing
                duration = time.time() - start_time
                logger.debug(f"Generated {len(generated_text)} characters in {duration:.2f}s")
                
                # Metadata
                metadata = {
                    "model": params["model"],
                    "tokens": result.get("eval_count", 0),
                    "duration": duration
                }
                
                return generated_text, metadata
            else:
                error_message = f"Error generating text: {response.status_code} - {response.text}"
                logger.error(error_message)
                return f"Error: {error_message}", {"error": error_message}
        
        except Exception as e:
            error_message = f"Exception in generate_text: {str(e)}"
            logger.error(error_message)
            return f"Error: {error_message}", {"error": error_message}
    
    def get_embeddings(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in a batch. If None, uses the config value.
            
        Returns:
            List of embeddings, each as a list of floats.
        """
        embeddings = []
        batch_size = batch_size or config_manager.config.ollama.batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        # First check if we need to test the embedding model dimension
        expected_dim = self.embedding_dim
        actual_dim = None
        dim_checked = False
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            
            batch_start_time = time.time()
            logger.debug(f"Processing embedding batch {i//batch_size + 1}/{total_batches} with {len(batch)} texts")
            
            for text in batch:
                try:
                    params = {
                        "model": self.embed_model,
                        "prompt": text
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json=params,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        embedding = result.get("embedding", [])
                        
                        # Check embedding dimension on first successful result if not checked yet
                        if embedding and not dim_checked:
                            actual_dim = len(embedding)
                            if actual_dim != expected_dim:
                                logger.warning(f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}. Using expected dimension of {expected_dim}.")
                            dim_checked = True
                        
                        if embedding:
                            # Ensure embedding has correct dimension
                            if len(embedding) != expected_dim:
                                logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, expected {expected_dim}")
                                # Normalize to expected dimension
                                if len(embedding) < expected_dim:
                                    # Pad with zeros if too short
                                    embedding = embedding + [0.0] * (expected_dim - len(embedding))
                                else:
                                    # Truncate if too long
                                    embedding = embedding[:expected_dim]
                            batch_embeddings.append(embedding)
                        else:
                            logger.error(f"Empty embedding returned for text: {text[:100]}...")
                            # Append zero vector as fallback
                            batch_embeddings.append([0.0] * expected_dim)
                    else:
                        logger.error(f"Error getting embedding: {response.status_code} - {response.text}")
                        # Append zero vector as fallback
                        batch_embeddings.append([0.0] * expected_dim)
                
                except Exception as e:
                    logger.error(f"Exception in get_embeddings: {str(e)}")
                    # Append zero vector as fallback
                    batch_embeddings.append([0.0] * expected_dim)
            
            embeddings.extend(batch_embeddings)
            
            batch_duration = time.time() - batch_start_time
            logger.debug(f"Processed batch in {batch_duration:.2f}s ({len(batch_embeddings)} embeddings)")
        
        total_duration = time.time() - start_time
        logger.info(f"Generated {len(embeddings)} embeddings in {total_duration:.2f}s")
        
        return embeddings

# Example usage:
# from src.services.ollama import OllamaService
# 
# ollama_service = OllamaService()
# 
# # Generate embeddings
# embedding = ollama_service.get_embeddings("What is the meaning of life?")
# 
# # Generate chat completion
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is GraphRAG?"}
# ]
# response = ollama_service.generate_chat_completion(messages)
# 
# # Generate text completion
# completion = ollama_service.generate_completion("Once upon a time") 