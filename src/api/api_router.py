"""
API router module for Parliamentary Meeting Analyzer.
Defines the REST API endpoints for accessing the data and insights.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.utils.logger import log
from src.data.loader import data_loader
from src.data.pipeline import data_pipeline
from src.graph.query_engine import graph_rag_query_engine

# Create main router
router = APIRouter(prefix="/api", tags=["Parliamentary Meeting Analyzer"])

# Create subrouters
data_router = APIRouter(prefix="/data", tags=["Data"])
graph_router = APIRouter(prefix="/graph", tags=["Graph"])
query_router = APIRouter(prefix="/query", tags=["Query"])
analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ----------------------------------------
# Data Endpoints
# ----------------------------------------

@data_router.get("/sessions", response_model=List[str])
async def get_sessions():
    """Get list of all available session dates."""
    try:
        return data_loader.get_sessions()
    except Exception as e:
        log.error(f"Error getting sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@data_router.get("/session/{date}")
async def get_session_data(date: str):
    """
    Get data for a specific session.
    
    Args:
        date (str): The session date in format YYYY-MM-DD
    """
    try:
        session_data = data_loader.get_data_by_session(date)
        if session_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for session date {date}")
        
        # Convert DataFrame to dict for JSON response
        return {
            "date": date,
            "count": len(session_data),
            "speakers": session_data["Speaker"].unique().tolist(),
            "records": session_data.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting session data for {date}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@data_router.get("/speakers", response_model=List[str])
async def get_speakers():
    """Get list of all available speakers."""
    try:
        return data_loader.get_speakers()
    except Exception as e:
        log.error(f"Error getting speakers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@data_router.get("/speaker/{name}")
async def get_speaker_data(name: str):
    """
    Get data for a specific speaker.
    
    Args:
        name (str): The speaker name
    """
    try:
        speaker_data = data_loader.get_data_by_speaker(name)
        if speaker_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for speaker {name}")
        
        speaker_info = data_loader.get_speaker_info(name)
        
        # Convert DataFrame to dict for JSON response
        return {
            "name": name,
            "info": speaker_info,
            "count": len(speaker_data),
            "sessions": speaker_data["Date"].unique().tolist(),
            "records": speaker_data.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting speaker data for {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@data_router.get("/process/session/{date}")
async def process_session(date: str):
    """
    Process a specific session for entities and insights.
    
    Args:
        date (str): The session date in format YYYY-MM-DD
    """
    try:
        result = data_pipeline.process_session(date)
        if not result.get("success", False):
            raise HTTPException(status_code=404, detail=result.get("error", f"Failed to process session {date}"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error processing session {date}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@data_router.get("/process/speaker/{name}")
async def process_speaker(name: str):
    """
    Process a specific speaker for entities and insights.
    
    Args:
        name (str): The speaker name
    """
    try:
        result = data_pipeline.process_speaker(name)
        if not result.get("success", False):
            raise HTTPException(status_code=404, detail=result.get("error", f"Failed to process speaker {name}"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error processing speaker {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------
# Graph Endpoints
# ----------------------------------------

@graph_router.get("/communities")
async def get_communities():
    """Get community information from the knowledge graph."""
    try:
        from src.graph.knowledge_graph import knowledge_graph
        
        communities = knowledge_graph.communities
        if not communities:
            # If communities are not detected yet, build the graph
            knowledge_graph.build_graph()
            communities = knowledge_graph.communities
        
        # Count nodes per community
        community_counts = {}
        for node, community_id in communities.items():
            if community_id not in community_counts:
                community_counts[community_id] = {
                    "id": community_id,
                    "count": 0,
                    "nodes": []
                }
            community_counts[community_id]["count"] += 1
            community_counts[community_id]["nodes"].append(node)
        
        # Get topics for each community
        for community_id, data in community_counts.items():
            topics = knowledge_graph.get_community_topics(community_id)
            data["topics"] = topics
        
        return list(community_counts.values())
    except Exception as e:
        log.error(f"Error getting communities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@graph_router.get("/node/{node_id}")
async def get_node_info(node_id: str):
    """
    Get information about a specific node in the knowledge graph.
    
    Args:
        node_id (str): The node ID
    """
    try:
        from src.graph.knowledge_graph import knowledge_graph
        
        if not knowledge_graph.has_node(node_id):
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found in the graph")
        
        # Get node data
        node_data = knowledge_graph.get_node_data(node_id)
        
        # Get connected nodes
        neighbors = list(knowledge_graph.graph.neighbors(node_id))
        
        # Get node community
        community_id = knowledge_graph.get_node_community(node_id)
        
        return {
            "id": node_id,
            "data": node_data,
            "neighbors": neighbors,
            "community": community_id
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting node info for {node_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------
# Query Endpoints
# ----------------------------------------

@query_router.get("/semantic")
async def semantic_search(
    query: str,
    max_results: int = Query(5, ge=1, le=20),
    threshold: float = Query(0.7, ge=0, le=1)
):
    """
    Perform semantic search on parliamentary data.
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
        threshold (float): Similarity threshold (0-1)
    """
    try:
        from src.data.vector_store import vector_store
        
        results = vector_store.search(query, max_results, threshold)
        return {
            "query": query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        log.error(f"Error performing semantic search for '{query}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@query_router.post("/rag")
async def graph_rag_query(
    query: str,
    context_type: Optional[str] = None,
    filters: Dict[str, Any] = {},
    max_results: int = Query(10, ge=1, le=30),
    include_context: bool = True
):
    """
    Perform RAG query using the knowledge graph and vector store.
    
    Args:
        query (str): The user query
        context_type (str, optional): Type of context to prioritize
        filters (Dict[str, Any]): Filters to apply to the search
        max_results (int): Maximum number of results to return
        include_context (bool): Whether to include context in the response
    """
    try:
        response = graph_rag_query_engine.query(
            query=query,
            context_type=context_type,
            filters=filters,
            max_results=max_results
        )
        
        if not include_context:
            # Remove context from response to reduce payload size
            if "context" in response:
                del response["context"]
        
        return response
    except Exception as e:
        log.error(f"Error performing RAG query for '{query}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------
# Analytics Endpoints
# ----------------------------------------

@analytics_router.get("/entity-frequency")
async def get_entity_frequency(
    entity_type: Optional[str] = None,
    session_date: Optional[str] = None,
    speaker_name: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get frequency analysis of entities.
    
    Args:
        entity_type (str, optional): Filter by entity type
        session_date (str, optional): Filter by session date
        speaker_name (str, optional): Filter by speaker name
        limit (int): Maximum number of entities to return
    """
    try:
        from src.graph.knowledge_graph import knowledge_graph
        
        # Filter nodes by type
        entity_nodes = {}
        
        # Get all nodes of specified type or all entity types if not specified
        for node_id, data in knowledge_graph.graph.nodes(data=True):
            node_type = data.get("node_type")
            if node_type == "entity" or node_type == entity_type:
                entity_label = data.get("label", "unknown")
                if entity_type and entity_label != entity_type:
                    continue
                
                # Apply session filter if specified
                if session_date:
                    connected_to_session = False
                    for neighbor in knowledge_graph.graph.neighbors(node_id):
                        neighbor_data = knowledge_graph.graph.nodes[neighbor]
                        if neighbor_data.get("node_type") == "session" and neighbor_data.get("date") == session_date:
                            connected_to_session = True
                            break
                    if not connected_to_session:
                        continue
                
                # Apply speaker filter if specified
                if speaker_name:
                    connected_to_speaker = False
                    for neighbor in knowledge_graph.graph.neighbors(node_id):
                        neighbor_data = knowledge_graph.graph.nodes[neighbor]
                        if neighbor_data.get("node_type") == "speaker" and neighbor_data.get("name") == speaker_name:
                            connected_to_speaker = True
                            break
                    if not connected_to_speaker:
                        continue
                
                # Count entity mentions
                count = data.get("count", 1)
                name = data.get("name", node_id)
                
                entity_nodes[name] = {
                    "id": node_id,
                    "name": name,
                    "type": entity_label,
                    "count": count
                }
        
        # Sort by count and limit results
        sorted_entities = sorted(
            entity_nodes.values(),
            key=lambda x: x["count"],
            reverse=True
        )[:limit]
        
        return {
            "entity_type": entity_type,
            "session": session_date,
            "speaker": speaker_name,
            "count": len(sorted_entities),
            "entities": sorted_entities
        }
    except Exception as e:
        log.error(f"Error getting entity frequency: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------
# Include subrouters in main router
# ----------------------------------------

router.include_router(data_router)
router.include_router(graph_router)
router.include_router(query_router)
router.include_router(analytics_router) 