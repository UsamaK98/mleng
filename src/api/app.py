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

# Initialize RAG pipeline
rag_pipeline = None

def get_pipeline():
    """
    Get or initialize the RAG pipeline
    """
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline


# Request and response models
class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    filters: Optional[Dict[str, Any]] = None
    structured_output: bool = False


class EntityRequest(BaseModel):
    """Entity request model"""
    entity: str


class TopicRequest(BaseModel):
    """Topic request model"""
    topic: str


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
            "/metadata"
        ]
    }


@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Process a general query about parliamentary minutes
    """
    pipeline = get_pipeline()
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
    """
    pipeline = get_pipeline()
    try:
        result = pipeline.process_entity_query(entity=request.entity)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topic")
async def process_topic_query(request: TopicRequest):
    """
    Process a query about a specific topic
    """
    pipeline = get_pipeline()
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