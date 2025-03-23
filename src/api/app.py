"""
FastAPI application for the Parliamentary Minutes Agentic Chatbot
"""
import os
import sys
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.pipeline import RAGPipeline
from src.api.analytics_routes import router as analytics_router
from config.config import API_HOST, API_PORT, API_DEBUG


# Initialize FastAPI app
app = FastAPI(
    title="Parliamentary Minutes Agentic Chatbot",
    description="API for querying Scottish Parliament meeting minutes",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include analytics routes
app.include_router(analytics_router)

# Cache for different pipeline instances (standard and hybrid)
pipeline_cache = {}

def get_pipeline(use_hybrid: bool = False, hybrid_alpha: float = 0.5):
    """
    Get or initialize the RAG pipeline with optional hybrid retrieval
    
    Args:
        use_hybrid: Whether to use hybrid retrieval
        hybrid_alpha: Weight for combining dense and sparse results (higher = more weight to dense)
        
    Returns:
        Initialized RAG pipeline
    """
    global pipeline_cache
    
    # Create a cache key based on the retrieval settings
    cache_key = f"hybrid_{use_hybrid}_alpha_{hybrid_alpha}" if use_hybrid else "standard"
    
    # Check if the pipeline is already in the cache
    if cache_key not in pipeline_cache:
        pipeline_cache[cache_key] = RAGPipeline(
            use_hybrid_retrieval=use_hybrid,
            hybrid_alpha=hybrid_alpha
        )
    
    return pipeline_cache[cache_key]


# Request and response models
class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    filters: Optional[Dict[str, Any]] = None
    structured_output: bool = False
    use_hybrid: bool = False
    hybrid_alpha: float = 0.5


class EntityRequest(BaseModel):
    """Entity request model"""
    entity: str
    use_hybrid: bool = False
    hybrid_alpha: float = 0.5


class TopicRequest(BaseModel):
    """Topic request model"""
    topic: str
    use_hybrid: bool = False
    hybrid_alpha: float = 0.5


# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Parliamentary Minutes Agentic Chatbot API",
        "version": "1.0.0",
        "endpoints": [
            "/query", 
            "/entity",
            "/topic",
            "/metadata",
            "/analytics/speakers",
            "/analytics/sessions",
            "/analytics/relationships",
            "/analytics/sentiment"
        ],
        "features": [
            "Hybrid retrieval (dense + sparse search)",
            "Speaker analytics and comparison",
            "Session analysis",
            "Relationship mapping",
            "Sentiment analysis"
        ],
        "analytics_endpoints": {
            "speaker_analytics": [
                "/analytics/speakers - Get top speakers",
                "/analytics/speakers/{speaker_name} - Get speaker statistics",
                "/analytics/speakers/compare?speaker1=X&speaker2=Y - Compare two speakers"
            ],
            "session_analytics": [
                "/analytics/sessions - Get session timeline",
                "/analytics/sessions/{session_date} - Get session statistics",
                "/analytics/sessions/compare?session1=X&session2=Y - Compare two sessions"
            ],
            "relationship_analytics": [
                "/analytics/relationships/network - Get speaker interaction network",
                "/analytics/relationships/influencers - Get key influencers",
                "/analytics/relationships/communities - Get speaker communities"
            ],
            "sentiment_analytics": [
                "/analytics/sentiment/overall - Get overall sentiment statistics",
                "/analytics/sentiment/by-speaker - Get sentiment by speaker",
                "/analytics/sentiment/by-session - Get sentiment by session",
                "/analytics/sentiment/outliers - Find emotional outliers",
                "/analytics/sentiment/by-role - Get sentiment by role",
                "/analytics/sentiment/keywords - Get sentiment-associated keywords"
            ]
        }
    }


@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Process a general query about parliamentary minutes
    
    Optional hybrid retrieval combines dense vector search with sparse keyword search
    for potentially improved results.
    """
    pipeline = get_pipeline(
        use_hybrid=request.use_hybrid, 
        hybrid_alpha=request.hybrid_alpha
    )
    
    try:
        result = pipeline.process_query(
            query=request.query,
            filters=request.filters,
            structured_output=request.structured_output
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/entity")
async def process_entity_query(request: EntityRequest):
    """
    Process a query about a specific entity (speaker)
    
    Optional hybrid retrieval combines dense vector search with sparse keyword search
    for potentially improved results.
    """
    pipeline = get_pipeline(
        use_hybrid=request.use_hybrid, 
        hybrid_alpha=request.hybrid_alpha
    )
    
    try:
        result = pipeline.process_entity_query(entity=request.entity)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topic")
async def process_topic_query(request: TopicRequest):
    """
    Process a query about a specific topic
    
    Optional hybrid retrieval combines dense vector search with sparse keyword search
    for potentially improved results.
    """
    pipeline = get_pipeline(
        use_hybrid=request.use_hybrid, 
        hybrid_alpha=request.hybrid_alpha
    )
    
    try:
        result = pipeline.process_topic_query(topic=request.topic)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metadata")
async def get_metadata():
    """
    Get metadata about the parliamentary minutes dataset
    """
    pipeline = get_pipeline()
    try:
        return pipeline.get_metadata()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_api_server(host: str = API_HOST, port: int = API_PORT, debug: bool = API_DEBUG):
    """
    Start the FastAPI server
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Whether to run in debug mode
    """
    # Disable reload when running in a thread to avoid signal handling issues
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    start_api_server() 