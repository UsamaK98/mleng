"""
Graph-enhanced RAG model implementation.

This module implements the GraphRAG class which combines traditional 
vector-based retrieval with graph-based reasoning to enhance 
question answering over parliamentary data.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd

from src.services.ollama import OllamaService
from src.graph.knowledge_graph import KnowledgeGraph
from src.storage.vector_db import VectorStore

logger = logging.getLogger(__name__)

class GraphRAG:
    """
    Graph-enhanced RAG model for parliamentary data.
    
    This class implements a hybrid retrieval system that combines:
    1. Traditional vector-based retrieval
    2. Graph-based reasoning and path exploration
    3. Hybrid approaches that combine both methods
    
    The system can operate in three modes:
    - "vector": Pure vector-based retrieval
    - "graph": Pure graph-based retrieval
    - "hybrid": Combined vector and graph retrieval
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        ollama_service: OllamaService,
        model_name: str = "llama3"
    ):
        """
        Initialize the GraphRAG model.
        
        Args:
            vector_store: Vector store for embedding-based retrieval
            knowledge_graph: Knowledge graph for graph-based retrieval
            ollama_service: Service for LLM generation
            model_name: Name of the LLM model to use
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.ollama_service = ollama_service
        self.model_name = model_name
        
        logger.info(f"Initialized GraphRAG with model '{model_name}'")
        self._check_components()
    
    def _check_components(self):
        """Check if all components are properly initialized and log their status."""
        # Check vector store
        if self.vector_store:
            if hasattr(self.vector_store, 'using_qdrant') and self.vector_store.using_qdrant:
                logger.info("Using Qdrant for vector storage")
            elif hasattr(self.vector_store, 'using_chroma') and self.vector_store.using_chroma:
                logger.info("Using ChromaDB for vector storage")
            else:
                logger.warning("Vector store type could not be determined")
        else:
            logger.warning("Vector store not initialized")
        
        # Check knowledge graph
        if self.knowledge_graph:
            logger.info(f"Knowledge graph has {len(self.knowledge_graph.graph.nodes)} nodes and {len(self.knowledge_graph.graph.edges)} edges")
        else:
            logger.warning("Knowledge graph not initialized")
        
        # Check ollama service
        if self.ollama_service:
            logger.info(f"Ollama service initialized with model base: {self.ollama_service.model_base}")
        else:
            logger.warning("Ollama service not initialized")
    
    def _vector_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform vector-based retrieval.
        
        Args:
            query: User query
            top_k: Number of top results to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        logger.debug(f"Performing vector retrieval for query: '{query}'")
        try:
            results = self.vector_store.search_similar(query, top_k=top_k)
            logger.debug(f"Vector retrieval found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in vector retrieval: {str(e)}")
            return []
    
    def _graph_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform graph-based retrieval.
        
        Args:
            query: User query
            top_k: Number of top results to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        logger.debug(f"Performing graph retrieval for query: '{query}'")
        try:
            # Extract entities from query
            entities = self.knowledge_graph.extract_entities_from_text(query)
            logger.debug(f"Extracted entities from query: {entities}")
            
            if not entities:
                logger.warning("No entities found in query for graph retrieval")
                return []
            
            # Find relevant statements using graph traversal
            results = []
            for entity in entities:
                paths = self.knowledge_graph.find_paths_from_entity(entity, max_length=2)
                for path in paths:
                    # Get the statements associated with these paths
                    statements = self.knowledge_graph.get_statements_for_path(path)
                    for statement in statements:
                        if isinstance(statement, dict) and 'payload' in statement:
                            # Already in the right format
                            results.append(statement)
                        else:
                            # Convert to expected format
                            doc = {
                                'id': f"graph_{len(results)}",
                                'payload': statement if isinstance(statement, dict) else {'Content': statement},
                                'score': 1.0,  # Default score for graph results
                                'source': 'graph'
                            }
                            results.append(doc)
            
            # Sort and limit results
            results = sorted(results, key=lambda x: x.get('score', 0.0), reverse=True)[:top_k]
            logger.debug(f"Graph retrieval found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in graph retrieval: {str(e)}")
            return []
    
    def _hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform hybrid retrieval combining vector and graph approaches.
        
        Args:
            query: User query
            top_k: Number of top results to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        logger.debug(f"Performing hybrid retrieval for query: '{query}'")
        try:
            # Get results from both methods
            vector_results = self._vector_retrieval(query, top_k=top_k)
            graph_results = self._graph_retrieval(query, top_k=top_k)
            
            # Combine and deduplicate results
            combined_results = []
            seen_content = set()
            
            # Process vector results first
            for result in vector_results:
                content = result.get('payload', {}).get('Content', '')
                if content and content not in seen_content:
                    seen_content.add(content)
                    result['source'] = 'vector'
                    combined_results.append(result)
            
            # Then add unique graph results
            for result in graph_results:
                content = result.get('payload', {}).get('Content', '')
                if content and content not in seen_content:
                    seen_content.add(content)
                    result['source'] = 'graph'
                    combined_results.append(result)
            
            # Sort by score and limit
            combined_results = sorted(combined_results, key=lambda x: x.get('score', 0.0), reverse=True)[:top_k]
            logger.debug(f"Hybrid retrieval found {len(combined_results)} results")
            return combined_results
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []
    
    def retrieve(self, query: str, mode: str = "hybrid", top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents using the specified mode.
        
        Args:
            query: User query
            mode: Retrieval mode ("vector", "graph", or "hybrid")
            top_k: Number of top results to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        logger.info(f"Retrieving documents for query: '{query}' using {mode} mode")
        
        if mode == "vector":
            return self._vector_retrieval(query, top_k)
        elif mode == "graph":
            return self._graph_retrieval(query, top_k)
        elif mode == "hybrid":
            return self._hybrid_retrieval(query, top_k)
        else:
            logger.warning(f"Unknown retrieval mode: {mode}, falling back to hybrid")
            return self._hybrid_retrieval(query, top_k)
    
    def _format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved results into a context for the LLM.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results):
            content = result.get('payload', {}).get('Content', '')
            date = result.get('payload', {}).get('Date', 'Unknown date')
            speaker = result.get('payload', {}).get('Speaker', 'Unknown speaker')
            source = result.get('source', 'unknown')
            score = result.get('score', 0.0)
            
            context_part = f"[{i+1}] On {date}, {speaker} said: \"{content}\""
            if source and score:
                context_part += f" (Source: {source}, Relevance: {score:.2f})"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _generate_prompt(self, query: str, context: str) -> str:
        """
        Generate a prompt for the LLM.
        
        Args:
            query: User query
            context: Formatted context from retrieved documents
            
        Returns:
            Complete prompt string
        """
        return f"""You are an assistant for analyzing parliamentary sessions. Answer the question based on the context provided.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using the LLM.
        
        Args:
            prompt: Complete prompt for the LLM
            
        Returns:
            Generated answer
        """
        try:
            if not self.ollama_service:
                return "LLM service is not available."
            
            result = self.ollama_service.generate_text(prompt, model=self.model_name)
            
            # Handle different possible return formats
            if isinstance(result, tuple) and len(result) >= 1:
                # If result is a tuple (text, additional_info)
                return result[0]
            elif isinstance(result, dict) and 'text' in result:
                # If result is a dictionary with 'text' key
                return result['text']
            elif isinstance(result, str):
                # If result is directly a string
                return result
            else:
                logger.error(f"Unexpected response format from LLM: {type(result)}")
                return "Failed to generate a response due to unexpected format."
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"An error occurred while generating the answer: {str(e)}"
    
    def answer_query(
        self, 
        query: str, 
        mode: str = "hybrid", 
        top_k: int = 5
    ) -> Tuple[str, List[Dict], str]:
        """
        Answer a user query using the specified retrieval mode.
        
        Args:
            query: User query
            mode: Retrieval mode ("vector", "graph", or "hybrid")
            top_k: Number of top results to retrieve
            
        Returns:
            Tuple containing (generated_answer, retrieved_documents, formatted_prompt)
        """
        logger.info(f"Answering query: '{query}' using {mode} mode")
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(query, mode=mode, top_k=top_k)
            
            # If no documents were retrieved
            if not retrieved_docs:
                logger.warning(f"No documents retrieved for query: '{query}'")
                empty_response = "I couldn't find any relevant information to answer your question."
                return empty_response, [], ""
            
            # Format context from retrieved documents
            formatted_context = self._format_context(retrieved_docs)
            
            # Generate prompt
            prompt = self._generate_prompt(query, formatted_context)
            
            # Generate answer
            answer = self._generate_answer(prompt)
            
            return answer, retrieved_docs, prompt
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            error_message = f"An error occurred while processing your query: {str(e)}"
            return error_message, [], "" 