"""
GraphRAG Query Processor for Parliamentary Meeting Analyzer.

This module provides implementation of Graph-enhanced RAG (GraphRAG) capabilities,
combining knowledge graph traversal with vector similarity search for complex queries.
"""

import time
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
import json
import re

from src.utils.logging import logger
from src.utils.config import config_manager
from src.models.graph import KnowledgeGraph
from src.services.ollama import OllamaService

class GraphRAG:
    """GraphRAG query processor combining knowledge graph with vector search."""
    
    def __init__(
        self,
        kg: KnowledgeGraph,
        ollama_service: OllamaService,
        vector_store = None,
        collection_name: str = "parliament_data"
    ):
        """Initialize the GraphRAG processor.
        
        Args:
            kg: KnowledgeGraph instance.
            ollama_service: OllamaService instance for embedding generation and inference.
            vector_store: Optional vector store instance. If None, the module will work
                          with graph-only mode.
            collection_name: Name of the vector store collection.
        """
        self.kg = kg
        self.ollama_service = ollama_service
        self.vector_store = vector_store
        self.collection_name = collection_name
        
        # Get configuration
        self.config = config_manager.config.graphrag
        
        # Cache directory for storing query results
        self.cache_dir = Path(config_manager.config.processed_data_dir) / "graphrag_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized GraphRAG processor")
    
    def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        max_results: int = 10,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Process a query using GraphRAG.
        
        Args:
            query_text: Text of the query.
            mode: Query mode, one of 'graph', 'vector', or 'hybrid'.
            max_results: Maximum number of results to return.
            include_sources: Whether to include source information in the response.
            
        Returns:
            Dictionary with query results.
        """
        start_time = time.time()
        
        # Identify query type and entities
        query_info = self._analyze_query(query_text)
        query_type = query_info["query_type"]
        
        logger.info(f"Processing query: '{query_text}' (Type: {query_type})")
        
        # Initialize results
        all_results = {}
        graph_results = []
        vector_results = []
        answer = ""
        
        # Process query based on mode
        if mode in ["graph", "hybrid"]:
            graph_results = self._process_graph_query(query_text, query_info)
        
        if mode in ["vector", "hybrid"] and self.vector_store:
            vector_results = self._process_vector_query(query_text, max_results)
        
        # Combine results based on query type and mode
        combined_results = self._combine_results(
            graph_results, 
            vector_results, 
            query_type, 
            mode, 
            max_results
        )
        
        # Generate answer
        if combined_results:
            answer = self._generate_answer(query_text, combined_results, query_type)
        else:
            answer = "I couldn't find any relevant information to answer your question."
        
        # Prepare response
        all_results = {
            "query": query_text,
            "query_type": query_type,
            "answer": answer,
            "time_taken": time.time() - start_time,
        }
        
        # Include sources if requested
        if include_sources:
            # Format sources for display
            sources = self._format_sources(combined_results, query_type)
            all_results["sources"] = sources
        
        logger.info(f"Query processed in {time.time() - start_time:.2f}s")
        
        return all_results
    
    def _analyze_query(self, query_text: str) -> Dict[str, Any]:
        """Analyze the query to determine its type and extract relevant entities.
        
        Args:
            query_text: Text of the query.
            
        Returns:
            Dictionary with query analysis information.
        """
        # Prepare prompt for query analysis
        prompt = f"""
        Analyze the following query and identify its type and relevant entities:
        "{query_text}"
        
        Classify the query as one of these types:
        1. SPEAKER_TOPIC - Questions about what a specific person said about a topic
        2. TOPIC_SUMMARY - Questions requesting a summary of discussions about a topic
        3. SPEAKER_INTERACTION - Questions about interactions between speakers
        4. TEMPORAL - Questions about events during a specific time period
        5. COMPARATIVE - Questions comparing different topics or speakers
        6. FACTUAL - Simple factual questions
        7. OTHER - Other types of questions
        
        Extract the following from the query:
        - Speakers mentioned (if any)
        - Topics mentioned (if any)
        - Organizations mentioned (if any)
        - Time periods mentioned (if any)
        - Other relevant entities
        
        Return the result as a JSON object with the following structure:
        {{
            "query_type": "QUERY_TYPE",
            "speakers": ["Speaker1", "Speaker2"],
            "topics": ["Topic1", "Topic2"],
            "organizations": ["Org1", "Org2"],
            "time_periods": ["Period1", "Period2"],
            "other_entities": ["Entity1", "Entity2"]
        }}
        Only include non-empty fields in the JSON.
        """
        
        # Get analysis from LLM
        response_tuple = self.ollama_service.generate_text(prompt)
        
        # The generate_text method returns a tuple (text, metadata)
        raw_response = response_tuple[0] if isinstance(response_tuple, tuple) else response_tuple
        
        # Extract JSON from response
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'({[\s\S]*})', raw_response)
            if json_match:
                json_str = json_match.group(1)
                query_info = json.loads(json_str)
            else:
                # Fallback to simple analysis
                query_info = {
                    "query_type": "OTHER",
                    "topics": self._extract_basic_entities(query_text)
                }
        except Exception as e:
            logger.warning(f"Error parsing query analysis: {str(e)}")
            # Fallback to simple analysis
            query_info = {
                "query_type": "OTHER",
                "topics": self._extract_basic_entities(query_text)
            }
        
        return query_info
    
    def _extract_basic_entities(self, text: str) -> List[str]:
        """Extract basic entities from text as fallback.
        
        Args:
            text: Text to extract entities from.
            
        Returns:
            List of extracted entities.
        """
        # Simple keyword extraction by removing stopwords and taking nouns
        stopwords = ["what", "who", "when", "where", "why", "how", "did", "does",
                    "is", "are", "was", "were", "do", "the", "a", "an", "in", "on",
                    "at", "by", "for", "with", "about", "from", "to", "of"]
        
        words = text.lower().split()
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        
        return keywords
    
    def _process_graph_query(
        self, 
        query_text: str, 
        query_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a query using the knowledge graph.
        
        Args:
            query_text: Text of the query.
            query_info: Query analysis information.
            
        Returns:
            List of results from graph query.
        """
        query_type = query_info["query_type"]
        results = []
        
        try:
            # Process different query types
            if query_type == "SPEAKER_TOPIC":
                # Find statements where a speaker discussed a topic
                if "speakers" in query_info and "topics" in query_info:
                    for speaker in query_info["speakers"]:
                        for topic in query_info["topics"]:
                            # Find speaker node
                            speaker_nodes = self.kg.search_by_keyword(speaker, ["person"])
                            topic_nodes = self.kg.search_by_keyword(topic, ["topic"])
                            
                            if speaker_nodes and topic_nodes:
                                speaker_node = speaker_nodes[0]
                                topic_node = topic_nodes[0]
                                
                                # Get connected statements
                                speaker_id = speaker_node["id"]
                                topic_id = topic_node["id"]
                                
                                # Find statements where speaker mentions topic
                                # First get statements by speaker
                                speaker_conn = self.kg.get_node_connections(
                                    speaker_id, 
                                    max_hops=1, 
                                    relation_types=["made_statement"]
                                )
                                
                                # Then find which statements mention the topic
                                for node in speaker_conn.get("nodes", []):
                                    if node.get("type") == "statement":
                                        # Check if this statement is connected to the topic
                                        stmt_conn = self.kg.get_node_connections(
                                            node["id"], 
                                            max_hops=1, 
                                            relation_types=["about_topic", "mentions"]
                                        )
                                        
                                        for related_node in stmt_conn.get("nodes", []):
                                            if related_node.get("id") == topic_id:
                                                # This statement is about the topic
                                                results.append({
                                                    "statement": node,
                                                    "speaker": speaker_node,
                                                    "topic": topic_node,
                                                    "source": "graph",
                                                    "relevance": 1.0
                                                })
            
            elif query_type == "SPEAKER_INTERACTION":
                # Find interactions between speakers
                if "speakers" in query_info and len(query_info["speakers"]) >= 2:
                    speakers = query_info["speakers"]
                    
                    for i, speaker1 in enumerate(speakers):
                        for speaker2 in speakers[i+1:]:
                            # Find speaker nodes
                            speaker1_nodes = self.kg.search_by_keyword(speaker1, ["person"])
                            speaker2_nodes = self.kg.search_by_keyword(speaker2, ["person"])
                            
                            if speaker1_nodes and speaker2_nodes:
                                speaker1_node = speaker1_nodes[0]
                                speaker2_node = speaker2_nodes[0]
                                
                                # Find interactions (statements that respond to each other)
                                speaker1_id = speaker1_node["id"]
                                speaker2_id = speaker2_node["id"]
                                
                                # Get statements by both speakers
                                speaker1_conn = self.kg.get_node_connections(
                                    speaker1_id, 
                                    max_hops=1, 
                                    relation_types=["made_statement"]
                                )
                                
                                speaker2_conn = self.kg.get_node_connections(
                                    speaker2_id, 
                                    max_hops=1, 
                                    relation_types=["made_statement"]
                                )
                                
                                # Find statements that respond to each other
                                speaker1_stmts = [
                                    node for node in speaker1_conn.get("nodes", [])
                                    if node.get("type") == "statement"
                                ]
                                
                                speaker2_stmts = [
                                    node for node in speaker2_conn.get("nodes", [])
                                    if node.get("type") == "statement"
                                ]
                                
                                # For each statement by speaker1, check if any statements by speaker2 respond to it
                                for stmt1 in speaker1_stmts:
                                    stmt1_conn = self.kg.get_node_connections(
                                        stmt1["id"], 
                                        max_hops=1, 
                                        relation_types=["responds_to"]
                                    )
                                    
                                    for node in stmt1_conn.get("nodes", []):
                                        if node.get("type") == "statement" and node["id"] in [s["id"] for s in speaker2_stmts]:
                                            # Found an interaction
                                            results.append({
                                                "statement1": stmt1,
                                                "statement2": node,
                                                "speaker1": speaker1_node,
                                                "speaker2": speaker2_node,
                                                "source": "graph",
                                                "relevance": 1.0
                                            })
            
            elif query_type == "TOPIC_SUMMARY":
                # Find statements about a topic
                if "topics" in query_info:
                    for topic in query_info["topics"]:
                        topic_nodes = self.kg.search_by_keyword(topic, ["topic"])
                        
                        if topic_nodes:
                            topic_node = topic_nodes[0]
                            topic_id = topic_node["id"]
                            
                            # Get statements about this topic
                            topic_conn = self.kg.get_node_connections(
                                topic_id, 
                                max_hops=1, 
                                relation_types=["about_topic", "mentions"]
                            )
                            
                            # Collect statement nodes
                            statements = []
                            for node in topic_conn.get("nodes", []):
                                if node.get("type") == "statement":
                                    # Get speaker for this statement
                                    stmt_conn = self.kg.get_node_connections(
                                        node["id"], 
                                        max_hops=1, 
                                        relation_types=["made_statement"]
                                    )
                                    
                                    speakers = [
                                        n for n in stmt_conn.get("nodes", [])
                                        if n.get("type") == "person" and n.get("id") != node.get("id")
                                    ]
                                    
                                    if speakers:
                                        speaker = speakers[0]
                                        statements.append({
                                            "statement": node,
                                            "speaker": speaker,
                                            "topic": topic_node,
                                            "source": "graph",
                                            "relevance": 1.0
                                        })
                            
                            # Sort statements by date
                            statements.sort(key=lambda x: x["statement"].get("date", ""))
                            results.extend(statements)
            
            elif query_type == "TEMPORAL":
                # Find statements during a specific time period
                if "time_periods" in query_info:
                    for period in query_info["time_periods"]:
                        # Search for session nodes with matching dates
                        session_nodes = self.kg.search_by_keyword(period, ["session"])
                        
                        for session in session_nodes:
                            session_id = session["id"]
                            
                            # Get statements in this session
                            session_conn = self.kg.get_node_connections(
                                session_id, 
                                max_hops=1, 
                                relation_types=["part_of"]
                            )
                            
                            statements = []
                            for node in session_conn.get("nodes", []):
                                if node.get("type") == "statement":
                                    # Get speaker for this statement
                                    stmt_conn = self.kg.get_node_connections(
                                        node["id"], 
                                        max_hops=1, 
                                        relation_types=["made_statement"]
                                    )
                                    
                                    speakers = [
                                        n for n in stmt_conn.get("nodes", [])
                                        if n.get("type") == "person" and n.get("id") != node.get("id")
                                    ]
                                    
                                    if speakers:
                                        speaker = speakers[0]
                                        statements.append({
                                            "statement": node,
                                            "speaker": speaker,
                                            "session": session,
                                            "source": "graph",
                                            "relevance": 1.0
                                        })
                            
                            # Add statements to results
                            results.extend(statements)
            
            else:
                # Generic query - search by keywords in both topics and statements
                keywords = (
                    query_info.get("topics", []) +
                    query_info.get("other_entities", [])
                )
                
                if not keywords and query_type == "OTHER":
                    keywords = self._extract_basic_entities(query_text)
                
                for keyword in keywords:
                    # Search statements
                    statement_nodes = self.kg.search_by_keyword(keyword, ["statement"])
                    
                    for stmt in statement_nodes:
                        # Get speaker for this statement
                        stmt_conn = self.kg.get_node_connections(
                            stmt["id"], 
                            max_hops=1, 
                            relation_types=["made_statement"]
                        )
                        
                        speakers = [
                            n for n in stmt_conn.get("nodes", [])
                            if n.get("type") == "person" and n.get("id") != stmt.get("id")
                        ]
                        
                        if speakers:
                            speaker = speakers[0]
                            results.append({
                                "statement": stmt,
                                "speaker": speaker,
                                "keyword": keyword,
                                "source": "graph",
                                "relevance": 1.0
                            })
        
        except Exception as e:
            logger.error(f"Error processing graph query: {str(e)}")
        
        return results
    
    def _process_vector_query(
        self, 
        query_text: str, 
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Process a query using vector similarity search.
        
        Args:
            query_text: Text of the query.
            max_results: Maximum number of results to return.
            
        Returns:
            List of results from vector search.
        """
        if not self.vector_store:
            return []
        
        try:
            # Search for similar content using vector store
            vector_results = self.vector_store.search_similar(
                query_text, 
                top_k=max_results
            )
            
            # Format results
            formatted_results = []
            for result in vector_results:
                # Create statement-like structure
                statement = {
                    "id": f"statement:{result['payload'].get('entry_id', '')}",
                    "type": "statement",
                    "content": result["payload"].get("Content", ""),
                    "date": result["payload"].get("Date", ""),
                    "entry_id": result["payload"].get("entry_id", "")
                }
                
                # Create speaker-like structure
                speaker = {
                    "id": f"person:{result['payload'].get('Speaker', '')}",
                    "type": "person",
                    "label": result["payload"].get("Speaker", "")
                }
                
                formatted_results.append({
                    "statement": statement,
                    "speaker": speaker,
                    "source": "vector",
                    "relevance": result["score"]
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error processing vector query: {str(e)}")
            return []
    
    def _combine_results(
        self,
        graph_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        query_type: str,
        mode: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Combine results from graph and vector queries.
        
        Args:
            graph_results: Results from graph query.
            vector_results: Results from vector query.
            query_type: Type of query.
            mode: Query mode (graph, vector, or hybrid).
            max_results: Maximum number of results to return.
            
        Returns:
            Combined results list.
        """
        if mode == "graph":
            combined = graph_results
        elif mode == "vector":
            combined = vector_results
        else:  # hybrid mode
            # Remove duplicates (statements appearing in both result sets)
            seen_ids = set()
            unique_results = []
            
            # Add graph results first (they are more reliable for structured queries)
            for result in graph_results:
                if "statement" in result:
                    stmt_id = result["statement"].get("id")
                    if stmt_id not in seen_ids:
                        seen_ids.add(stmt_id)
                        unique_results.append(result)
            
            # Add vector results that don't duplicate graph results
            for result in vector_results:
                if "statement" in result:
                    stmt_id = result["statement"].get("id")
                    if stmt_id not in seen_ids:
                        seen_ids.add(stmt_id)
                        unique_results.append(result)
            
            combined = unique_results
        
        # Sort based on query type
        if query_type in ["SPEAKER_TOPIC", "TOPIC_SUMMARY"]:
            # Sort by relevance then date
            combined.sort(key=lambda x: (
                -x.get("relevance", 0),
                x.get("statement", {}).get("date", "")
            ))
        elif query_type == "TEMPORAL":
            # Sort by date
            combined.sort(key=lambda x: x.get("statement", {}).get("date", ""))
        else:
            # Sort by relevance
            combined.sort(key=lambda x: -x.get("relevance", 0))
        
        # Limit results
        return combined[:max_results]
    
    def _generate_answer(
        self,
        query_text: str,
        results: List[Dict[str, Any]],
        query_type: str
    ) -> str:
        """Generate a natural language answer to the query.
        
        Args:
            query_text: Text of the query.
            results: Combined results from graph and vector search.
            query_type: Type of query.
            
        Returns:
            Generated answer text.
        """
        # Extract content from results
        context = ""
        
        # Format context based on result type
        for i, result in enumerate(results[:5]):  # Limit to top 5 most relevant results
            if "statement" in result:
                speaker = result.get("speaker", {}).get("label", "Unknown Speaker")
                content = result.get("statement", {}).get("content", "")
                date = result.get("statement", {}).get("date", "")
                
                context += f"[{i+1}] On {date}, {speaker} said: \"{content}\"\n\n"
        
        # Prepare prompt for answer generation
        prompt = f"""
        Based on the following information from parliamentary records, please answer this question:
        
        Question: {query_text}
        
        Relevant information:
        {context}
        
        Please provide a clear, concise answer drawing only on the information above.
        Focus on directly answering the question rather than summarizing all the information.
        If the provided information is not sufficient to answer the question fully, acknowledge this in your response.
        """
        
        # Generate answer
        answer_text, _ = self.ollama_service.generate_text(prompt)
        
        # Clean up answer (remove any leading/trailing quotes and special characters)
        answer_text = answer_text.strip().strip('"').strip()
        
        return answer_text
    
    def _format_sources(
        self,
        results: List[Dict[str, Any]],
        query_type: str
    ) -> List[Dict[str, Any]]:
        """Format the source information for display.
        
        Args:
            results: Combined results from graph and vector search.
            query_type: Type of query.
            
        Returns:
            Formatted sources list.
        """
        sources = []
        
        for result in results:
            if "statement" in result:
                source = {
                    "speaker": result.get("speaker", {}).get("label", "Unknown"),
                    "content": result.get("statement", {}).get("content", ""),
                    "date": result.get("statement", {}).get("date", ""),
                    "source_type": result.get("source", "unknown"),
                    "relevance": result.get("relevance", 0)
                }
                
                # Add topic information if available
                if "topic" in result:
                    source["topic"] = result.get("topic", {}).get("label", "")
                
                sources.append(source)
        
        return sources

# Usage example:
# from src.models.graph import KnowledgeGraph
# from src.models.graphrag import GraphRAG
# from src.services.ollama import OllamaService
# from src.storage.vector_db import VectorStore
# from src.data.loader import ParliamentaryDataLoader
# 
# # Initialize services
# loader = ParliamentaryDataLoader()
# loader.load_data()
# 
# # Get a session's data
# session_data = loader.get_session_data("2024-09-10")
# 
# # Initialize Ollama service
# ollama_service = OllamaService(model_name="llama3")
# 
# # Initialize vector store
# vector_store = VectorStore(
#     collection_name="parliament", 
#     ollama_service=ollama_service
# )
# vector_store.store_parliamentary_data(session_data)
# 
# # Initialize knowledge graph
# kg = KnowledgeGraph()
# kg.build_from_parliamentary_data(session_data)
# 
# # Initialize GraphRAG
# graphrag = GraphRAG(
#     kg=kg,
#     ollama_service=ollama_service,
#     vector_store=vector_store
# )
# 
# # Process a query
# results = graphrag.query(
#     "What did the Prime Minister say about healthcare funding?",
#     mode="hybrid",
#     max_results=5
# )
# 
# print(results["answer"]) 