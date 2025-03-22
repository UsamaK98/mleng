"""
Retriever module for RAG pipeline
"""
from typing import List, Dict, Any, Optional
import numpy as np

from src.models.embeddings import EmbeddingModel
from src.database.vector_store import VectorStore
from config.config import MAX_DOCUMENTS_RETRIEVED


class MinutesRetriever:
    """
    Retriever for parliamentary minutes
    """
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        max_documents: int = MAX_DOCUMENTS_RETRIEVED
    ):
        """
        Initialize the retriever
        
        Args:
            embedding_model: Model for generating query embeddings
            vector_store: Vector store for retrieval
            max_documents: Maximum number of documents to retrieve
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()
        self.max_documents = max_documents
    
    def retrieve(
        self,
        query: str,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            filter_params: Optional filters to apply (e.g., speaker, date)
            
        Returns:
            List of relevant documents with metadata
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search for similar documents
        results = self.vector_store.search(
            query_vector=query_embedding,
            filter_params=filter_params,
            limit=self.max_documents
        )
        
        return results
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a context string for the LLM
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            speaker = result["metadata"]["speaker"]
            role = result["metadata"]["role"]
            date = result["metadata"]["date"]
            timestamp = result["metadata"]["timestamp"]
            text = result["text"]
            
            # Format speaker with role if available
            speaker_text = f"{speaker}"
            if role and role.strip():
                speaker_text += f" ({role})"
            
            # Add formatted context part
            context_parts.append(
                f"[Document {i}]\n"
                f"Date: {date}, Time: {timestamp}\n"
                f"Speaker: {speaker_text}\n"
                f"Content: {text}\n"
            )
        
        return "\n".join(context_parts) 