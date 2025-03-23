# Parliamentary Minutes Agentic Chatbot

An AI-powered system for querying and analyzing Scottish Parliament meeting minutes using Retrieval-Augmented Generation (RAG) technology. This system provides an intelligent interface for searching, analyzing, and extracting insights from parliamentary proceedings.

## Overview

The Parliamentary Minutes Agentic Chatbot allows users to query the Scottish Parliament meeting minutes in natural language. The system uses advanced RAG techniques to retrieve relevant information and generate accurate, contextual responses based on the parliamentary data.

Key features:
- **Natural language queries** about parliamentary proceedings
- **Entity-based search** for finding information about specific MSPs
- **Topic-based search** for exploring discussions on particular subjects
- **Hybrid search capabilities** combining semantic and keyword search
- **Analytics dashboard** with speaker and session insights
- **Interactive visualizations** for parliamentary data exploration

## System Architecture

The system is built with a modular architecture:

1. **Vector Database (Qdrant)**: Stores parliamentary text chunks and their embeddings for semantic search
2. **Embedding Model**: Uses nomic-embed-text for high-quality text embeddings that capture semantic meaning
3. **LLM Integration**: Connects with Ollama to provide local LLM capabilities
4. **Hybrid Search**: Combines semantic and keyword-based search for improved results
5. **Analytics Module**: Provides speaker, session, and relationship analysis tools
6. **API Layer**: FastAPI endpoints for querying the data and retrieving analytics
7. **Streamlit UI**: Interactive user interface for exploring the parliamentary data

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Node.js 18+ (for UI development)
- GPU recommended but not required

## Getting Started

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/parliamentary-minutes-chatbot.git
   cd parliamentary-minutes-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run Docker services (Qdrant):
   ```
   docker-compose up -d
   ```

4. Install the Ollama model for embeddings:
   ```
   ollama pull nomic-embed-text
   ```

5. Start the application:
   ```
   python run_app.py
   ```

### Data Ingestion

The first time you run the application, it will check if data has been ingested and guide you through the process if needed:

1. The script will verify that the required data files exist
2. If data hasn't been ingested yet, it will prompt you to start ingestion
3. Data will be processed, chunked, embedded with nomic-embed-text, and stored in Qdrant

## Usage

After starting the application:

1. Access the UI at http://localhost:8501
2. Choose between searching by topic, speaker, or viewing analytics
3. Enter your query in natural language or select from suggested topics/speakers
4. View the results and explore related information

## API Endpoints

The API is available at http://localhost:8000 with the following endpoints:

- `GET /health`: Check API status
- `POST /query`: Submit a general query about parliamentary minutes
- `POST /query/entity`: Query about a specific parliamentary entity (person, organization)
- `POST /query/topic`: Query about a specific topic
- `GET /analytics/speakers`: Get speaker analytics
- `GET /analytics/sessions`: Get session analytics
- `GET /analytics/relationships`: Get speaker relationship data

## Example Queries

- "What did Nicola Sturgeon say about education reform in March 2022?"
- "Summarize the debate on healthcare funding from the last session"
- "What are the main points raised by John Swinney about climate policy?"
- "Compare Nicola Sturgeon and Ruth Davidson's positions on independence"

## Features

### Advanced Search Capabilities

The system uses hybrid search that combines:
- Dense vector search using nomic-embed-text embeddings
- Sparse keyword search using TF-IDF
- Results are combined with a weighted algorithm for improved relevance

### Analytics Features

The system provides detailed analytics:
- **Speaker Analytics**: Contribution counts, word usage, sentiment analysis
- **Session Analysis**: Topic modeling, participation statistics, timeline views
- **Relationship Mapping**: Speaker interaction networks and influence analysis
- **Sentiment Analysis**: Emotional content analysis across speakers and topics

## Project Structure

```
parliamentary-minutes-chatbot/
├── config/                  # Configuration files
├── src/                     # Source code
│   ├── api/                 # API endpoints and routes
│   ├── analytics/           # Analytics modules
│   ├── data/                # Data processing utilities
│   ├── database/            # Database connectors
│   ├── models/              # Embedding and LLM interfaces
│   ├── rag/                 # RAG pipeline components
│   └── utils/               # Utility functions
├── ui/                      # Streamlit UI code
├── tests/                   # Unit and integration tests
├── docker-compose.yml       # Docker configuration
├── run_app.py               # Application entry point
└── requirements.txt         # Python dependencies
```

## Technology Choices

- **Embeddings**: Using the state-of-the-art nomic-embed-text model for high-quality semantic representations
- **Vector DB**: Qdrant for efficient vector search capabilities
- **LLM**: Ollama for local inference with various models
- **Analytics**: Custom modules for speaker, session, and interaction analysis
- **Visualization**: Interactive charts with Plotly and Streamlit

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Scottish Parliament for providing the parliamentary minutes data
- The open-source AI community for tools and models
- Contributors to the project 