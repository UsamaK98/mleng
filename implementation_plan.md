# Parliamentary Meeting Analyzer Implementation Plan

## Project Overview
This document outlines the implementation plan for the Parliamentary Meeting Analyzer, which uses GraphRAG technology to allow users to query and analyze parliamentary meeting minutes. The system incorporates GLiNER for named entity recognition, Ollama for embedding generation and chat functionality, and presents a user-friendly interface via Streamlit.

## Phase 1: Infrastructure Setup

### 1. Project Structure (Estimated time: 1 day)
- [x] Create main project directory
- [x] Set up required directories:
  - `src/`: Source code
  - `data/`: Data files (processed and raw)
  - `models/`: For storing model configurations and metadata
  - `logs/`: For application logs (added to .gitignore)
  - `config/`: For configuration files
  - `assets/`: For static files (images, CSS, etc.)
  - `app/`: Streamlit application files
  - `tests/`: For unit and integration tests

### 2. Environment Setup (Estimated time: 0.5 day)
- [x] Create requirements.txt with necessary dependencies
- [x] Create .gitignore file (include logs directory)
- [x] Set up virtual environment
- [x] Document environment setup process

### 3. Ollama Interface (Estimated time: 1 day)
- [x] Create a standardized Ollama interface class in `src/services/ollama.py`
- [x] Implement methods for:
  - Text embedding generation
  - Chat completion with qwq model
  - Parameter handling (temperature, top_k, etc.)
- [x] Add error handling and connection management
- [x] Write docstrings and usage examples

### 4. Logging System (Estimated time: 0.5 day)
- [x] Create logging module in `src/utils/logging.py`
- [x] Implement structured logging for:
  - Model interactions
  - Query processing
  - Error handling
  - Performance metrics
- [x] Configure log rotation and format
- [x] Ensure logs are saved to the logs directory

### 5. Configuration System (Estimated time: 1 day)
- [x] Create configuration module in `src/utils/config.py`
- [x] Define default configurations for:
  - Ollama model parameters (chat, embedding)
  - GLiNER parameters
  - GraphRAG parameters
  - Application settings
- [x] Implement configuration loading and saving
- [x] Create utility functions for accessing configuration

## Phase 2: Data Processing

### 1. Data Loading (Estimated time: 0.5 day)
- [x] Create data loading module in `src/data/loader.py`
- [x] Implement functions to read from CSV files:
  - parliamentary_minutes.csv
  - speakers_list.csv
- [x] Create data models/schema for loaded data
- [x] Add data validation and cleaning

### 2. GLiNER Integration (Estimated time: 1 day)
- [x] Create NER module in `src/models/ner.py`
- [x] Implement GLiNER model integration
- [x] Define entity types relevant to parliamentary data:
  - People (MPs, witnesses)
  - Organizations
  - Locations
  - Legislation
  - Topics/Themes
  - Dates/Times
- [x] Create entity extraction pipeline
- [x] Implement caching for performance

### 3. Entity Relationship Extraction (Estimated time: 1.5 days)
- [x] Create relationship extraction module in `src/models/relationship.py`
- [x] Define relationship types:
  - Speaker-to-Topic
  - Speaker-to-Speaker (responses)
  - Speaker-to-Organization
  - Topic-to-Legislation
- [x] Implement rule-based relationship extraction
- [x] Enhance with Ollama-based extraction for complex relationships
- [x] Create visualization helpers for relationships

### 4. Knowledge Graph Construction (Estimated time: 1.5 days)
- [x] Create graph module in `src/models/graph.py`
- [x] Implement NetworkX-based graph construction
- [x] Define node and edge schema
- [x] Create functions for:
  - Adding entity nodes
  - Adding relationship edges
  - Graph querying
  - Graph serialization/deserialization
- [x] Implement community detection

### 5. Vector Storage (Estimated time: 1 day)
- [x] Create vector database module in `src/storage/vector_db.py`
- [x] Implement Qdrant integration
- [x] Create embedding generation pipeline using Ollama
- [x] Define collection schema for different entity types
- [x] Implement search and retrieval functions
- [x] Add ChromaDB fallback mechanism

### 6. GraphRAG Implementation (Estimated time: 2 days)
- [x] Create GraphRAG module in `src/models/graphrag.py`
- [x] Implement core GraphRAG functionality following Microsoft's approach
- [x] Create query analyzer and decomposer
- [x] Implement hybrid retrieval combining:
  - Graph traversal
  - Vector similarity
- [x] Create context merging mechanism
- [x] Optimize for GPU usage

## Phase 3: Streamlit UI Development (Estimated time: 3 days)

### 1. UI Framework (Estimated time: 0.5 day)
- [x] Set up Streamlit application structure
- [x] Create base layout and navigation
- [x] Implement theme and styling
- [x] Design responsive UI elements

### 2. Query Interface (Estimated time: 1 day)
- [x] Create query input component with:
  - Text input
  - Query mode selection (Graph, Vector, Hybrid)
  - Parameter controls (relevance threshold, max results)
- [x] Implement query submission and loading indicators
- [x] Add query history tracking
- [x] Create example query suggestions

### 3. Results Visualization (Estimated time: 1 day)
- [x] Implement response display component
- [x] Create knowledge graph visualization using NetworkX and Streamlit components
- [x] Design entity highlighting in responses
- [x] Add source attribution and confidence scores

### 4. Advanced Features (Estimated time: 0.5 day)
- [ ] Implement interface for:
  - Entity exploration and visualization
  - Persistent caching for NER and embeddings
  - Speaker selection interface

## Phase 4: Analysis Features Implementation (Estimated time: 4 days)

### 1. Home Dashboard (Estimated time: 0.5 day)
- [ ] Create informative home page with:
  - Dataset overview and statistics
  - Application capabilities explanation
  - Tab descriptions and navigation guide
  - Visual summary of parliamentary data
- [ ] Add dataset timeline visualization
- [ ] Include entity type distribution chart

### 2. Entity Explorer (Estimated time: 1 day)
- [ ] Design intuitive entity visualization interface with:
  - Categorized entity listing (People, Organizations, etc.)
  - Entity frequency metrics
  - Entity relationship visualization
  - Entity filtering and sorting options
- [ ] Implement entity relationship graph
- [ ] Create entity co-occurrence matrix

### 3. Speaker Query Interface (Estimated time: 1 day)
- [ ] Implement speaker-focused chat interface:
  - Interactive speaker selection buttons
  - Dynamic context loading for selected speaker
  - Chat history for each speaker
  - LLM integration via Ollama
- [ ] Develop context retrieval optimization for speakers
- [ ] Create speaker profile summary generation

### 4. Statistical Analysis (Estimated time: 1 day)
- [ ] Implement dashboard for:
  - Speaker participation metrics
  - Topic frequency analysis
  - Sentiment analysis over time
  - Entity co-occurrence patterns
- [ ] Create visualization components for each metric
- [ ] Add time-based filtering

## Phase 5: Integration and Optimization (Estimated time: 3 days)

### 1. Performance Optimization (Estimated time: 1 day)
- [ ] Implement caching for:
  - Frequent queries
  - Embedding generation
  - Graph traversals
- [ ] Add batch processing for large datasets
- [ ] Optimize memory usage for graph operations
- [ ] Implement lazy loading for UI components

### 2. Security Enhancements (Estimated time: 0.5 day)
- [ ] Add authentication system
- [ ] Implement role-based access control
- [ ] Create secure API endpoints
- [ ] Add input validation and sanitization

### 3. Testing and Validation (Estimated time: 1 day)
- [ ] Create comprehensive test suite:
  - Unit tests for core components
  - Integration tests for system workflows
  - Performance benchmarks
  - UI testing
- [ ] Implement continuous integration
- [ ] Create validation datasets

### 4. Documentation and Deployment (Estimated time: 0.5 day)
- [ ] Complete code documentation
- [ ] Create user manual
- [ ] Write technical documentation
- [ ] Prepare deployment configurations:
  - Docker containerization
  - Cloud deployment options
  - Local installation guide

## Current Status

- Phase 1 (Infrastructure Setup): ✅ Complete
- Phase 2 (Data Processing): ✅ Complete
- Phase 3 (Streamlit UI Development): 🟡 75% Complete
- Phase 4 (Analysis Features): 🔴 Not Started
- Phase 5 (Integration and Optimization): 🔴 Not Started

## Next Steps
1. Complete remaining items in Phase 3 (Advanced UI Features):
   - Implement persistent caching for NER and embeddings
   - Create speaker selection interface
2. Begin implementation of Phase 4:
   - Develop Home Dashboard
   - Create Entity Explorer
   - Build Speaker Query Interface
   - Implement Statistical Analysis
3. Prepare test datasets for validation
4. Set up continuous integration workflow 