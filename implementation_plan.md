# Parliamentary Minutes Agentic Chatbot: Implementation Plan

## Project Overview
An agentic chatbot system for interacting with parliamentary meeting minutes, enabling users to query by entity, topic, or other parameters and receive structured insights from the data.

## System Architecture
```
┌───────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ User Interface    │     │  Retrieval System   │     │  Knowledge Base  │
│ - FastAPI backend │────►│  - RAG pipeline     │────►│  - Qdrant        │
│ - Streamlit UI    │     │  - Query processing │     │  - Postgres      │
└───────────────────┘     └─────────────────────┘     └──────────────────┘
         ▲                          │                          ▲
         │                          ▼                          │
         │                ┌─────────────────────┐             │
         └────────────────┤ Language Model      │─────────────┘
                          │ - Ollama (local)    │
                          │ - nomic-embed-text  │
                          └─────────────────────┘
```

## Phase 1: Foundation
- [x] Analyze dataset and visualizations
- [x] Set up project structure
- [x] Configure environment and dependencies
- [x] Implement data processing pipeline
- [x] Create vector database schema
- [x] Implement data loader for vector database
- [x] Create utilities for text chunking and embedding generation
- [x] Build initial query understanding module

**Deliverables:**
- [x] Project structure with necessary modules
- [x] Environment configuration
- [x] Data processing utilities
- [x] Vector database configuration and initial data loading

## Phase 2: Core Functionality
- [x] Implement RAG pipeline
- [x] Configure Ollama integration
- [x] Create prompt templates for different query types
- [x] Build FastAPI endpoints
- [x] Implement basic query routing logic
- [x] Create response formatter
- [x] Develop basic Streamlit interface
- [x] Add simple query input and response display
- [x] Create evaluation metrics for response quality

**Deliverables:**
- [x] Working FastAPI backend with RAG capabilities
- [x] Basic Streamlit frontend for queries
- [x] Functional end-to-end query-response system
- [x] Initial evaluation metrics

## Phase 3: Advanced Features
- [x] Implement hybrid search capabilities
- [x] Add better error handling and suggestions for entity queries
- [x] Add speaker and topic analytics
- [x] Enable cross-session analysis
- [x] Develop relationship mapping between speakers
- [x] Add sentiment analysis for contributions
- [x] Enhance UI with visualizations
- [x] Create dynamic filtering capabilities
- [x] Optimize embedding generation with nomic-embed-text

**Deliverables:**
- [x] Enhanced search capabilities with hybrid retrieval
- [x] Speaker and topic analytics features
- [x] Advanced UI with visualization components
- [x] Interactive visualizations for analytics data
- [x] Integration with nomic-embed-text for better embeddings

## Phase 4: Final Refinement (Completed)
- [x] UI Simplification and Enhancement
  - [x] Create informative landing page with dataset statistics
  - [x] Remove chat tab for more focused interaction
  - [x] Add topic and speaker buttons for direct querying
  - [x] Implement session search tab
  - [x] Streamline analytics interface
- [x] Performance Optimization
  - [x] Implement nomic-embed-text for better embeddings
  - [ ] Optimize vector search operations

**Deliverables:**
- [x] Simplified and intuitive UI focused on topic/speaker/session search
- [x] Informative landing page with dataset overview
- [x] Improved embedding quality with nomic-embed-text
- [x] Better performance through optimizations

## Technology Stack
- **Backend**: FastAPI, Python
- **Vector Database**: Qdrant (running in Docker)
- **LLM Provider**: Ollama (local deployment)
- **Embedding Model**: nomic-embed-text (via Ollama)
- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy, sentence-transformers
- **Visualization**: matplotlib, plotly
- **Analytics**: spaCy, TextBlob, networkx
- **Environment**: Conda, Poetry
- **Evaluation**: custom metrics for relevance and accuracy

## Timeline
- Phase 1: Completed
- Phase 2: Completed
- Phase 3: Completed
- Phase 4: Completed

## Evaluation Criteria
- Query understanding accuracy
- Response relevance and accuracy
- Response generation time
- User satisfaction
- Resource utilization 