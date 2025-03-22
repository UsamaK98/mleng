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
                          │ - GPU acceleration  │
                          └─────────────────────┘
```

## Phase 1: Foundation
- [x] Analyze dataset and visualizations
- [ ] Set up project structure
- [ ] Configure environment and dependencies
- [ ] Implement data processing pipeline
- [ ] Create vector database schema
- [ ] Implement data loader for vector database
- [ ] Create utilities for text chunking and embedding generation
- [ ] Build initial query understanding module

**Deliverables:**
- Project structure with necessary modules
- Environment configuration
- Data processing utilities
- Vector database configuration and initial data loading

## Phase 2: Core Functionality
- [ ] Implement RAG pipeline
- [ ] Configure Ollama integration
- [ ] Create prompt templates for different query types
- [ ] Build FastAPI endpoints
- [ ] Implement basic query routing logic
- [ ] Create response formatter
- [ ] Develop basic Streamlit interface
- [ ] Add simple query input and response display
- [ ] Create evaluation metrics for response quality

**Deliverables:**
- Working FastAPI backend with RAG capabilities
- Basic Streamlit frontend for queries
- Functional end-to-end query-response system
- Initial evaluation metrics

## Phase 3: Advanced Features
- [ ] Implement hybrid search capabilities
- [ ] Add speaker and topic analytics
- [ ] Enable cross-session analysis
- [ ] Develop relationship mapping between speakers
- [ ] Add sentiment analysis for contributions
- [ ] Enhance UI with visualizations
- [ ] Create dynamic filtering capabilities
- [ ] Optimize query performance
- [ ] Implement GPU acceleration

**Deliverables:**
- Enhanced search capabilities
- Speaker and topic analytics features
- Advanced UI with visualization components
- Performance optimizations

## Phase 4: Refinement
- [ ] Collect and integrate user feedback
- [ ] Improve response quality based on evaluations
- [ ] Add more sophisticated prompt engineering
- [ ] Optimize performance bottlenecks
- [ ] Enhance error handling and logging
- [ ] Create comprehensive documentation
- [ ] Prepare final deployment package

**Deliverables:**
- Refined response generation
- Performance optimizations
- Comprehensive documentation
- Production-ready deployment package

## Technology Stack
- **Backend**: FastAPI, Python
- **Vector Database**: Qdrant (running in Docker)
- **LLM Provider**: Ollama (local deployment)
- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy, sentence-transformers
- **Visualization**: matplotlib, plotly
- **Evaluation**: custom metrics for relevance and accuracy

## Timeline
- Phase 1: 1-2 weeks
- Phase 2: 2-3 weeks
- Phase 3: 2-3 weeks
- Phase 4: 1-2 weeks

## Evaluation Criteria
- Query understanding accuracy
- Response relevance and accuracy
- Response generation time
- User satisfaction
- Resource utilization 