# Parliamentary Meeting Analyzer Implementation Plan

## Project Overview
This document outlines the implementation plan for the Parliamentary Meeting Analyzer, which uses GraphRAG technology to allow users to query and analyze parliamentary meeting minutes. The system incorporates GLiNER for named entity recognition, Ollama for embedding generation and chat functionality, and presents a user-friendly interface via Streamlit.

## Phase 1: Infrastructure Setup

### 1. Project Structure (Estimated time: 1 day)
- [x] Create main project directory
- [ ] Set up required directories:
  - `src/`: Source code
  - `data/`: Data files (processed and raw)
  - `models/`: For storing model configurations and metadata
  - `logs/`: For application logs (added to .gitignore)
  - `config/`: For configuration files
  - `assets/`: For static files (images, CSS, etc.)
  - `app/`: Streamlit application files

### 2. Environment Setup (Estimated time: 0.5 day)
- [ ] Create requirements.txt with necessary dependencies
- [ ] Create .gitignore file (include logs directory)
- [ ] Set up virtual environment
- [ ] Document environment setup process

### 3. Ollama Interface (Estimated time: 1 day)
- [ ] Create a standardized Ollama interface class in `src/services/ollama.py`
- [ ] Implement methods for:
  - Text embedding generation
  - Chat completion with qwq model
  - Parameter handling (temperature, top_k, etc.)
- [ ] Add error handling and connection management
- [ ] Write docstrings and usage examples

### 4. Logging System (Estimated time: 0.5 day)
- [ ] Create logging module in `src/utils/logging.py`
- [ ] Implement structured logging for:
  - Model interactions
  - Query processing
  - Error handling
  - Performance metrics
- [ ] Configure log rotation and format
- [ ] Ensure logs are saved to the logs directory

### 5. Configuration System (Estimated time: 1 day)
- [ ] Create configuration module in `src/utils/config.py`
- [ ] Define default configurations for:
  - Ollama model parameters (chat, embedding)
  - GLiNER parameters
  - GraphRAG parameters
  - Application settings
- [ ] Implement configuration loading and saving
- [ ] Create utility functions for accessing configuration

## Phase 2: Data Processing

### 1. Data Loading (Estimated time: 0.5 day)
- [ ] Create data loading module in `src/data/loader.py`
- [ ] Implement functions to read from CSV files:
  - parliamentary_minutes.csv
  - speakers_list.csv
- [ ] Create data models/schema for loaded data
- [ ] Add data validation and cleaning

### 2. GLiNER Integration (Estimated time: 1 day)
- [ ] Create NER module in `src/models/ner.py`
- [ ] Implement GLiNER model integration
- [ ] Define entity types relevant to parliamentary data:
  - People (MPs, witnesses)
  - Organizations
  - Locations
  - Legislation
  - Topics/Themes
  - Dates/Times
- [ ] Create entity extraction pipeline
- [ ] Implement caching for performance

### 3. Entity Relationship Extraction (Estimated time: 1.5 days)
- [ ] Create relationship extraction module in `src/models/relationship.py`
- [ ] Define relationship types:
  - Speaker-to-Topic
  - Speaker-to-Speaker (responses)
  - Speaker-to-Organization
  - Topic-to-Legislation
- [ ] Implement rule-based relationship extraction
- [ ] Enhance with Ollama-based extraction for complex relationships
- [ ] Create visualization helpers for relationships

### 4. Knowledge Graph Construction (Estimated time: 1.5 days)
- [ ] Create graph module in `src/models/graph.py`
- [ ] Implement NetworkX-based graph construction
- [ ] Define node and edge schema
- [ ] Create functions for:
  - Adding entity nodes
  - Adding relationship edges
  - Graph querying
  - Graph serialization/deserialization
- [ ] Implement community detection

### 5. Vector Storage (Estimated time: 1 day)
- [ ] Create vector database module in `src/storage/vector_db.py`
- [ ] Implement Qdrant integration
- [ ] Create embedding generation pipeline using Ollama
- [ ] Define collection schema for different entity types
- [ ] Implement search and retrieval functions

### 6. GraphRAG Implementation (Estimated time: 2 days)
- [ ] Create GraphRAG module in `src/models/graphrag.py`
- [ ] Implement core GraphRAG functionality following Microsoft's approach
- [ ] Create query analyzer and decomposer
- [ ] Implement hybrid retrieval combining:
  - Graph traversal
  - Vector similarity
- [ ] Create context merging mechanism
- [ ] Optimize for GPU usage

## Expected Deliverables for Phases 1-2
- Fully functional backend infrastructure
- Entity extraction and relationship identification system
- Knowledge graph with parliamentary meeting data
- Vector embeddings stored in Qdrant
- GraphRAG query system ready for integration with UI

## Dependencies
- Python 3.10+
- Streamlit
- FastAPI
- GLiNER
- Ollama
- NetworkX
- Qdrant client
- Pandas
- PyTorch (GPU-enabled)

## Next Steps
After completion of Phases 1-2, we will proceed with:
- Phase 3: Streamlit UI Development
- Phase 4: Analysis Features Implementation
- Phase 5: Integration and Optimization 