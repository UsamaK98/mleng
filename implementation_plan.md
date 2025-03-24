# Parliamentary Meeting Analyzer - Implementation Plan

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
  - Query template creation
  - Result export options

### 5. Additional UI Tabs (Estimated time: 2 days)
- [x] Implement Home Tab:
  - Overview of the application
  - Dataset statistics and summary
  - Navigation guidance
  - Feature explanations
  
- [x] Implement Entity List Tab:
  - Categorized display of all extracted entities
  - Entity filtering and search
  - Entity relationship visualization
  - Entity details view
  
- [x] Implement Speaker Query Tab:
  - Interactive speaker selection buttons
  - Speaker-specific context retrieval
  - Chat interface with Ollama/qwq model
  - Speaker activity summary

### 6. Caching Optimization (Estimated time: 1 day)
- [x] Review and improve NER model caching:
  - Persistent storage of extracted entities
  - Versioning of entity data
  - Efficient loading mechanisms
  
- [x] Enhance embedding caching:
  - Persistent vector storage
  - Metadata tracking for embeddings
  - Session-independent cache access
  
- [x] Implement cache management UI:
  - Cache status indicators
  - Manual cache refresh options
  - Cache integrity validation

## Phase 4: Analysis Features Implementation (Estimated time: 4 days)

### 1. Statistical Analysis (Estimated time: 1 day)
- [x] Implement dashboard for:
  - Speaker participation metrics (speaking time, frequency)
  - Topic frequency analysis
  - Entity co-occurrence patterns
- [x] Create visualization components:
  - Bar charts for participation
  - Word clouds for topic frequency
  - Line charts for activity over time
  - Heat maps for entity co-occurrence
- [x] Add time-based filtering:
  - Date range selection
  - Meeting-specific filtering
  - Topic-specific filtering
- [ ] Implement export functionality for analysis results:
  - CSV export
  - Chart image export
  - Summary report generation

### 2. Document Comparison (Estimated time: 1 day)
- [ ] Create interface for comparing multiple meeting minutes
- [ ] Implement similarity scoring between documents
- [ ] Design visualization for document relationships
- [ ] Add entity-based comparison views

### 3. Network Analysis (Estimated time: 1 day)
- [ ] Implement advanced graph analytics:
  - Centrality measures for key entities
  - Community detection for related topics
  - Path analysis between entities
  - Influence mapping
- [ ] Create interactive graph exploration tools
- [ ] Add filtering and highlighting based on network metrics

### 4. Pattern Detection (Estimated time: 1 day)
- [ ] Implement algorithms for:
  - Recurring topic identification
  - Speaker position changes over time
  - Coalition and opposition pattern detection
  - Emerging issue identification
- [ ] Create alert system for pattern detection
- [ ] Design visualization for temporal patterns

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
- Phase 3 (Streamlit UI Development): ✅ Complete
- Phase 4 (Analysis Features): 🟡 25% Started
- Phase 5 (Integration and Optimization): 🔴 Not Started

## Next Steps
1. Continue implementation of Analysis Features in Phase 4
   - Add document comparison functionality
   - Implement advanced network analysis
   - Add pattern detection features
2. Prepare test datasets for validation
3. Set up continuous integration workflow
4. Begin planning for Phase 5 optimizations

## Phase 1: Caching Optimization and Storage Solution

### 1. Persistent Storage for Embeddings (COMPLETED)
- Enhanced the `VectorStore` class to store embeddings persistently
- Added metadata tracking for cache state
- Implemented initialization checks and recovery

### 2. Persistent Storage for Named Entities (COMPLETED)
- Enhanced the `EntityExtractor` class for better caching
- Added versioning and metadata tracking
- Improved error handling for cached entities

### 3. Cache Management Utility (COMPLETED)
- Created the `CacheManager` class for cache validation and status reporting
- Implemented methods for retrieving cache status and clearing caches
- Added helper functions for cache maintenance

## Phase 2: UI Enhancements

### 1. Home Tab (COMPLETED)
- Created a new home tab component that displays:
  - Application overview and explanation of features
  - Dataset statistics (documents, speakers, entities)
  - System status showing cache availability
  - Quick tips for navigation

### 2. Entity List Tab (COMPLETED)
- Created a new entity list tab component that displays:
  - Categorized entities by type
  - Filtering options by entity types, dates, speakers
  - Entity search functionality
  - Entity frequency visualizations
  - Entity co-occurrence matrix

### 3. Speaker Query Tab (COMPLETED)
- Created a new speaker query tab component that enables:
  - Speaker-focused search functionality
  - Topic-based filtering of statements
  - Speaker analytics with activity metrics
  - Speaker relationship network visualization
  - Advanced query capabilities

## Phase 3: GraphRAG Enhancements (PENDING)

### 1. Improved Query Analysis
- Enhance query type detection for more accurate processing
- Add support for comparative queries between speakers
- Implement context-aware query processing

### 2. Graph Traversal Optimization
- Optimize path finding algorithms for faster responses
- Implement prioritized node expansion based on relevance
- Add depth-limited graph exploration for complex queries

### 3. Query Result Visualization
- Create visual representations of query results
- Implement interactive result exploration
- Add support for saving and comparing query results

## Phase 4: Statistical Analysis Dashboard (PENDING)

### 1. Speaker Statistics
- Track speaker activity over time
- Analyze speaker sentiment on various topics
- Compare speaking patterns between different speakers

### 2. Topic Analysis
- Track topic trends over time
- Identify correlations between topics
- Analyze topic distribution across sessions

### 3. Network Analysis
- Identify key influencers in the speaker network
- Analyze community structures in discussions
- Measure centrality and importance of entities

## Implementation Timeline

| Task | Estimated Completion | Status |
|------|----------------------|--------|
| Caching Optimization | Week 1 | COMPLETED |
| UI Enhancements | Week 2 | COMPLETED |
| GraphRAG Enhancements | Week 3 | PENDING |
| Statistical Analysis | Week 4 | PENDING | 