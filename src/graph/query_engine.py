"""
GraphRAG query engine for Parliamentary Meeting Analyzer.
Handles complex queries using a combination of graph traversal and vector search.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from src.utils.logger import log
from src.utils.config import config_manager
from src.models.ollama_interface import ollama
from src.data.vector_store import vector_store
from src.graph.knowledge_graph import knowledge_graph

class GraphRAGQueryEngine:
    """
    GraphRAG query engine for complex queries.
    Combines graph traversal and vector search for enhanced retrieval.
    """
    
    def __init__(self):
        """Initialize the GraphRAG query engine."""
        self.max_results = config_manager.get("graphrag.max_nodes_per_query", 50)
        self.similarity_threshold = config_manager.get("graphrag.similarity_threshold", 0.75)
        log.info(f"GraphRAG query engine initialized")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding for the query.
        
        Args:
            query (str): The query text.
            
        Returns:
            List[float]: The query embedding.
        """
        embedding = ollama.generate_embeddings(query)
        log.debug(f"Generated query embedding for: {query[:30]}...")
        return embedding
    
    def retrieve_context(
        self, 
        query: str, 
        context_type: Optional[str] = None, 
        context_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context using GraphRAG approach.
        
        Args:
            query (str): The query text.
            context_type (Optional[str]): Type of context to retrieve 
                (e.g., "speaker", "session", "entity").
            context_filter (Optional[Dict[str, Any]]): Additional filters.
            top_k (int): Number of results to return.
            
        Returns:
            List[Dict[str, Any]]: List of context items with metadata.
        """
        # Start with semantic search
        vector_results = vector_store.search(query, top_k=top_k, filter_dict=context_filter)
        
        # Extract graph nodes from vector results
        graph_nodes = []
        for result in vector_results:
            metadata = result.get("metadata", {})
            
            # Check for speaker nodes
            if "Speaker" in metadata:
                speaker = metadata["Speaker"]
                if knowledge_graph.node_exists(speaker):
                    graph_nodes.append(speaker)
            
            # Check for session nodes
            if "Date" in metadata:
                session = metadata["Date"]
                if knowledge_graph.node_exists(session):
                    graph_nodes.append(session)
        
        # Get expanded context using graph traversal
        expanded_context = self._expand_context_from_graph(graph_nodes, context_type)
        
        # Merge results
        merged_results = self._merge_results(vector_results, expanded_context)
        
        log.info(f"Retrieved {len(merged_results)} context items for query: {query[:30]}...")
        return merged_results
    
    def _expand_context_from_graph(
        self, 
        seed_nodes: List[str], 
        node_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Expand context from seed nodes by finding related nodes in the knowledge graph.
        
        Args:
            seed_nodes (List[str]): List of seed node IDs.
            node_type (Optional[str]): Type of nodes to include in expansion.
            
        Returns:
            List[Dict[str, Any]]: List of expanded context items.
        """
        if not seed_nodes:
            return []
        
        expanded_nodes = set()
        expanded_context = []
        
        # Add seed nodes to expanded set
        for node in seed_nodes:
            if node not in expanded_nodes:
                expanded_nodes.add(node)
                
                # Only include nodes of specified type if requested
                if node_type:
                    if knowledge_graph.graph.nodes[node].get('node_type') != node_type:
                        continue
                
                # Get node data
                node_data = dict(knowledge_graph.graph.nodes[node])
                
                # Add to expanded context
                expanded_context.append({
                    "id": node,
                    "text": node_data.get('text', ''),
                    "metadata": {
                        "node_type": node_data.get('node_type'),
                        "name": node_data.get('name', node),
                        "date": node_data.get('date', '')
                    },
                    "score": 1.0  # Maximum score for direct matches
                })
        
        # Expand to neighbors
        for node in seed_nodes:
            for neighbor in knowledge_graph.graph.neighbors(node):
                if neighbor not in expanded_nodes:
                    expanded_nodes.add(neighbor)
                    
                    # Only include nodes of specified type if requested
                    if node_type:
                        if knowledge_graph.graph.nodes[neighbor].get('node_type') != node_type:
                            continue
                    
                    # Get node data
                    node_data = dict(knowledge_graph.graph.nodes[neighbor])
                    
                    # Add to expanded context
                    expanded_context.append({
                        "id": neighbor,
                        "text": node_data.get('text', ''),
                        "metadata": {
                            "node_type": node_data.get('node_type'),
                            "name": node_data.get('name', neighbor),
                            "date": node_data.get('date', '')
                        },
                        "score": 0.9  # Slightly lower score for immediate neighbors
                    })
        
        return expanded_context
    
    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge vector search results with graph expansion results.
        
        Args:
            vector_results (List[Dict[str, Any]]): Results from vector search.
            graph_results (List[Dict[str, Any]]): Results from graph expansion.
            
        Returns:
            List[Dict[str, Any]]: Merged results.
        """
        # Start with vector results
        merged = {result.get("id"): result for result in vector_results}
        
        # Add graph results if not already included
        for result in graph_results:
            result_id = result.get("id")
            if result_id and result_id not in merged:
                # Assign a score based on position in graph results
                # (lower than vector results but still relevant)
                result["score"] = 0.5
                merged[result_id] = result
        
        # Convert to list and sort by score
        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Limit to max results
        return merged_list[:self.max_results]
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the retrieved context.
        
        Args:
            query (str): The query text.
            context (List[Dict[str, Any]]): Retrieved context.
            
        Returns:
            str: Generated response.
        """
        if not context:
            log.warning("No context available for response generation")
            return "I don't have enough information to answer that question about the parliamentary meetings."
        
        # Prepare context text
        context_text = self._format_context_for_prompt(context)
        
        # Create prompt
        system_prompt = """You are an AI assistant specializing in analyzing parliamentary meeting minutes.
        Use the provided context to answer the user's question. The context includes information from
        parliamentary debates, including speakers, their roles, and the content of their contributions.
        
        If the information needed to answer the question is not in the context, say "I don't have
        enough information to answer that question fully" and then provide the closest relevant information you have.
        
        Always be factual, objective, and base your response strictly on the provided context."""
        
        prompt = f"""Context information from parliamentary minutes:
        {context_text}
        
        User question: {query}
        
        Please provide a comprehensive and accurate answer based on the context provided."""
        
        # Generate response
        response = ollama.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for more factual responses
        )
        
        log.info(f"Generated response for query: {query[:30]}...")
        return response
    
    def _format_context_for_prompt(self, context: List[Dict[str, Any]]) -> str:
        """
        Format context items for inclusion in the prompt.
        
        Args:
            context (List[Dict[str, Any]]): Context items.
            
        Returns:
            str: Formatted context text.
        """
        formatted_items = []
        
        for item in context:
            metadata = item.get("metadata", {})
            item_type = item.get("node_type")
            
            if item_type == knowledge_graph.NODE_SPEAKER:
                name = metadata.get("name", "Unknown")
                role = metadata.get("role", "")
                formatted_items.append(f"Speaker: {name} (Role: {role})")
                
            elif item_type == knowledge_graph.NODE_SESSION:
                date = metadata.get("date", "Unknown")
                formatted_items.append(f"Session Date: {date}")
                
            elif item_type == knowledge_graph.NODE_ENTITY:
                text = metadata.get("text", "")
                entity_type = metadata.get("node_type", "")
                formatted_items.append(f"Entity: {text} (Type: {entity_type})")
                
            elif "text" in metadata:  # Content from vector search
                text = metadata.get("text", "")
                speaker = metadata.get("Speaker", "Unknown")
                date = metadata.get("Date", "Unknown")
                formatted_items.append(f"Content from {speaker} on {date}: {text}")
        
        return "\n\n".join(formatted_items)
    
    def query(self, query_text: str, context_type: Optional[str] = None, 
            context_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a complete query pipeline.
        
        Args:
            query_text (str): The query text.
            context_type (Optional[str]): Type of context to retrieve.
            context_filter (Optional[Dict[str, Any]]): Additional filters.
            
        Returns:
            Dict[str, Any]: Query results including context and response.
        """
        log.info(f"Processing query: {query_text}")
        
        # Retrieve context
        context = self.retrieve_context(query_text, context_type, context_filter)
        
        # Generate response
        response = self.generate_response(query_text, context)
        
        return {
            "query": query_text,
            "context": context,
            "response": response
        }


# Create a singleton instance
graph_rag_query_engine = GraphRAGQueryEngine() 