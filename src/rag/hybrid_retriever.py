"""
Hybrid retriever that combines dense vector search with sparse (keyword) search
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import heapq

from src.database.vector_store import VectorStore
from src.models.embeddings import EmbeddingModel
from config.config import MAX_DOCUMENTS_RETRIEVED


class HybridRetriever:
    """
    Retriever that combines dense vector search with sparse (keyword) search
    for improved retrieval performance.
    """
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        alpha: float = 0.5,  # Weight for combining dense and sparse results
        max_documents: int = MAX_DOCUMENTS_RETRIEVED
    ):
        """
        Initialize the hybrid retriever
        
        Args:
            embedding_model: Model for generating query embeddings
            vector_store: Vector store for retrieval
            alpha: Weight for combining dense and sparse results (0.0 to 1.0)
                  Higher values give more weight to dense (vector) search
            max_documents: Maximum number of documents to retrieve
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()
        self.alpha = alpha
        self.max_documents = max_documents
        
        # TF-IDF vectorizer for sparse retrieval
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            max_df=0.85,         # Ignore terms that appear in >85% of documents
            min_df=2             # Ignore terms that appear in <2 documents
        )
        
        # Document cache for sparse search
        self.document_texts = []
        self.document_ids = []
        self.document_metadata = []
        self.sparse_matrix = None
        
        # Initialize sparse index
        self._initialize_sparse_index()
    
    def _load_documents_from_qdrant(self, batch_size: int = 100, max_docs: int = 5000):
        """
        Load documents from Qdrant for sparse indexing
        
        Args:
            batch_size: Number of documents to fetch in each batch
            max_docs: Maximum number of documents to load
        
        Returns:
            Tuple of (document_ids, document_texts, document_metadata)
        """
        print(f"Loading documents from Qdrant (max: {max_docs})...")
        
        try:
            # Get scroll iterator from Qdrant client
            scroll_iterator = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=batch_size,
                with_payload=True,
                with_vectors=False  # We don't need the vectors for sparse search
            )
            
            doc_ids = []
            doc_texts = []
            doc_metadata = []
            total_loaded = 0
            
            # Iterate through batches
            for batch_number, (batch, _) in enumerate(scroll_iterator):
                if not batch or total_loaded >= max_docs:
                    break
                
                for point in batch:
                    doc_ids.append(point.id)
                    doc_texts.append(point.payload["text"])
                    doc_metadata.append(point.payload["metadata"])
                    total_loaded += 1
                    
                    if total_loaded >= max_docs:
                        break
                
                print(f"Loaded batch {batch_number + 1} ({len(batch)} documents), total: {total_loaded}")
            
            print(f"Successfully loaded {total_loaded} documents from Qdrant")
            return doc_ids, doc_texts, doc_metadata
            
        except Exception as e:
            print(f"Error loading documents from Qdrant: {str(e)}")
            return [], [], []
    
    def _initialize_sparse_index(self, batch_size: int = 100, max_docs: int = 5000):
        """
        Initialize the sparse TF-IDF index by fetching documents from the vector store
        
        Args:
            batch_size: Number of documents to fetch in each batch
            max_docs: Maximum number of documents to load for the sparse index
        """
        print("Initializing sparse TF-IDF index...")
        
        try:
            # Load documents from Qdrant
            doc_ids, doc_texts, doc_metadata = self._load_documents_from_qdrant(
                batch_size=batch_size, 
                max_docs=max_docs
            )
            
            self.document_ids = doc_ids
            self.document_texts = doc_texts
            self.document_metadata = doc_metadata
            
            # Build TF-IDF matrix
            if self.document_texts:
                self.sparse_matrix = self.tfidf.fit_transform(self.document_texts)
                print(f"Built TF-IDF index with {len(self.document_texts)} documents")
            else:
                print("No documents found for TF-IDF indexing")
        
        except Exception as e:
            print(f"Error initializing sparse index: {str(e)}")
            print("Continuing with dense vector search only")
    
    def _sparse_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform sparse keyword search using TF-IDF
        
        Args:
            query: Query string
            limit: Maximum number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        if not self.sparse_matrix or not self.document_texts:
            return []
        
        # Transform query to sparse vector
        query_vector = self.tfidf.transform([query])
        
        # Calculate similarity scores (dot product)
        scores = (query_vector @ self.sparse_matrix.T).toarray()[0]
        
        # Get top results
        if limit >= len(scores):
            top_indices = np.argsort(scores)[::-1][:limit]
        else:
            top_indices = np.argpartition(scores, -limit)[-limit:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with non-zero scores
                results.append({
                    "id": self.document_ids[idx],
                    "score": float(scores[idx]),
                    "text": self.document_texts[idx],
                    "metadata": self.document_metadata[idx]
                })
        
        return results
    
    def _combine_results(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]],
        alpha: float
    ) -> List[Dict[str, Any]]:
        """
        Combine results from dense and sparse search
        
        Args:
            dense_results: Results from vector search
            sparse_results: Results from keyword search
            alpha: Weight for dense results (0.0 to 1.0)
            
        Returns:
            Combined and reranked results
        """
        # Create a map for faster lookups
        result_map = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = result["id"]
            result_map[doc_id] = {
                "id": doc_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "dense_score": result["score"],
                "sparse_score": 0.0,
                "combined_score": alpha * result["score"]
            }
        
        # Process sparse results
        for result in sparse_results:
            doc_id = result["id"]
            if doc_id in result_map:
                # Document already exists from dense results
                result_map[doc_id]["sparse_score"] = result["score"]
                result_map[doc_id]["combined_score"] += (1 - alpha) * result["score"]
            else:
                # New document from sparse results
                result_map[doc_id] = {
                    "id": doc_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "dense_score": 0.0,
                    "sparse_score": result["score"],
                    "combined_score": (1 - alpha) * result["score"]
                }
        
        # Convert map to list and sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Format results for consistency with regular retriever
        formatted_results = []
        for result in combined_results:
            formatted_results.append({
                "id": result["id"],
                "score": result["combined_score"],
                "text": result["text"],
                "metadata": result["metadata"]
            })
        
        return formatted_results
    
    def retrieve(
        self,
        query: str,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search
        
        Args:
            query: Query string
            filter_params: Optional filters to apply (e.g., speaker, date)
            limit: Maximum number of results to return (defaults to self.max_documents)
            
        Returns:
            List of relevant documents with metadata and scores
        """
        if limit is None:
            limit = self.max_documents
        
        # Get more results than requested for reranking
        search_limit = min(limit * 2, 20)  # Cap at 20 to avoid excessive retrieval
        
        # Dense vector search
        query_embedding = self.embedding_model.embed_text(query)
        dense_results = self.vector_store.search(
            query_vector=query_embedding,
            filter_params=filter_params,
            limit=search_limit
        )
        
        # Sparse keyword search (if available)
        sparse_results = []
        if self.sparse_matrix is not None:
            sparse_results = self._sparse_search(query, limit=search_limit)
        
        # Combine and rerank results
        combined_results = self._combine_results(
            dense_results, sparse_results, alpha=self.alpha
        )
        
        # Return top results
        return combined_results[:limit]
    
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
            role = result["metadata"].get("role", "")
            date = result["metadata"]["date"]
            timestamp = result["metadata"].get("timestamp", "")
            text = result["text"]
            score = result.get("score", 0.0)
            
            # Format speaker with role if available
            speaker_text = f"{speaker}"
            if role and role.strip():
                speaker_text += f" ({role})"
            
            # Add formatted context part
            context_parts.append(
                f"[Document {i} - Relevance: {score:.4f}]\n"
                f"Date: {date}, Time: {timestamp}\n"
                f"Speaker: {speaker_text}\n"
                f"Content: {text}\n"
            )
        
        return "\n".join(context_parts) 