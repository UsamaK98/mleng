# Parliamentary Minutes Agentic Chatbot

An AI-powered chatbot system for querying and analyzing Scottish Parliament meeting minutes, leveraging RAG (Retrieval-Augmented Generation) technology.

## 🔍 Overview

This project provides an intelligent interface to search, analyze, and extract insights from Scottish Parliament meeting minutes. The system uses a vector database to store document embeddings, allowing for semantic retrieval of relevant content when users ask questions. The main features include:

- **Natural language querying** of parliamentary minutes
- **Speaker analysis** to examine contributions by specific MPs
- **Topic exploration** to find discussions on specific subjects
- **Document citations** linking responses to source material
- **Interactive UI** for easy engagement with the data
- **Advanced analytics** including sentiment analysis and relationship mapping
- **Hybrid search** combining dense vector and sparse keyword search

## 🏗️ System Architecture

The system is built with a modular architecture, consisting of several key components:

- **Vector Database (Qdrant)**: Stores document embeddings for semantic search
- **Embedding Model**: Converts text to vector embeddings using sentence-transformers
- **LLM Integration**: Connects to Ollama for generating responses
- **API Layer**: FastAPI backend exposes endpoints for queries
- **UI**: Streamlit frontend for user interaction
- **Analytics Engine**: Provides insights on speakers, sessions, and sentiment

```
┌───────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ User Interface    │     │  Retrieval System   │     │  Knowledge Base  │
│ - Streamlit UI    │────►│  - RAG pipeline     │────►│  - Qdrant        │
│ - Interactive UI  │     │  - Query processing │     │  - Document store│
└───────────────────┘     └─────────────────────┘     └──────────────────┘
         ▲                          │                          ▲
         │                          ▼                          │
         │                ┌─────────────────────┐             │
         └────────────────┤ Language Model      │─────────────┘
                          │ - Ollama (local)    │
                          │ - Mistral model     │
                          └─────────────────────┘
                                    ▲
                                    │
                          ┌─────────────────────┐
                          │  Analytics Engine   │
                          │ - Speaker analysis  │
                          │ - Session analysis  │
                          │ - Sentiment analysis│
                          └─────────────────────┘
```

## 📋 Prerequisites

- **Python 3.9+**
- **Docker** for running Qdrant and Ollama services
- **Conda** (recommended) for managing dependencies
- **Poetry** (optional) for alternative dependency management
- **GPU recommended** for faster embedding generation and inference (but not required)

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/parliamentary-minutes-chatbot.git
cd parliamentary-minutes-chatbot
```

### 2. Set up the environment

You can choose between Conda (recommended) or Poetry for managing your dependencies:

#### Option A: Using Conda (Recommended)

```bash
# Create a new conda environment
conda env create -f environment.yml

# Activate the environment
conda activate parlminutes

# Install the spaCy model (if needed)
python -m spacy download en_core_web_sm
```

#### Option B: Using Poetry

```bash
# Install Poetry if you don't have it
# Windows: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
# macOS/Linux: curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Install the spaCy model (if needed)
python -m spacy download en_core_web_sm
```

### 3. Run necessary Docker services

#### Qdrant Vector Database

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

#### Ollama (for LLM inference)

```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull mistral
```

### 4. Data Ingestion

Process the parliamentary minutes and create embeddings:

```bash
python run_app.py --ingest
```

Add `--force-recreate` if you need to rebuild the vector database:

```bash
python run_app.py --ingest --force-recreate
```

### 5. Run the Application

Start both the API and UI with:

```bash
python run_app.py
```

The application will check if all dependencies are available and then:
- Start the FastAPI backend on port 8000
- Start the Streamlit UI on port 8501

You can access the UI at http://localhost:8501 and the API at http://localhost:8000

## 🔌 API Endpoints

The system provides the following API endpoints:

### Main Endpoints

- **POST /query**: Process general queries about parliamentary minutes
  ```json
  {
    "query": "What discussions took place about healthcare?",
    "filters": {"date": "2024-06-26"},
    "use_hybrid": true
  }
  ```

- **POST /entity**: Process queries about specific entities (speakers)
  ```json
  {
    "entity": "John Smith",
    "use_hybrid": true
  }
  ```

- **POST /topic**: Process queries about specific topics
  ```json
  {
    "topic": "education",
    "use_hybrid": true
  }
  ```

- **GET /metadata**: Retrieve metadata about the parliamentary minutes dataset

### Analytics Endpoints

#### Speaker Analytics
- **GET /analytics/speakers**: Get top speakers by contribution count
- **GET /analytics/speakers/{speaker_name}**: Get detailed statistics for a specific speaker
- **GET /analytics/speakers/compare**: Compare two speakers (query params: speaker1, speaker2)

#### Session Analytics
- **GET /analytics/sessions**: Get timeline of all sessions with key metrics
- **GET /analytics/sessions/{session_date}**: Get statistics for a specific session
- **GET /analytics/sessions/compare**: Compare two sessions (query params: session1, session2)

#### Relationship Analytics
- **GET /analytics/relationships/network**: Get speaker interaction network
- **GET /analytics/relationships/influencers**: Get key influencers in the speaker network
- **GET /analytics/relationships/communities**: Get communities of speakers

#### Sentiment Analytics
- **GET /analytics/sentiment/overall**: Get overall sentiment statistics
- **GET /analytics/sentiment/by-speaker**: Get sentiment analysis by speaker
- **GET /analytics/sentiment/by-session**: Get sentiment trends across sessions
- **GET /analytics/sentiment/outliers**: Find emotional outliers in contributions
- **GET /analytics/sentiment/by-role**: Compare sentiment between different speaker roles
- **GET /analytics/sentiment/keywords**: Get keywords associated with positive and negative sentiment

## 💡 Example Queries

### General Questions
- "What was discussed in the session on January 7, 2025?"
- "Summarize the key points from the June 26, 2024 meeting"
- "What was the main topic of discussion in September 2024?"

### Entity Queries
- "What did Jane Doe contribute to the discussion on healthcare?"
- "How many times did John Smith speak in the October meeting?"
- "What topics does Mary Johnson typically discuss?"

### Topic Queries
- "Summarize all discussions about education in the parliament"
- "What was said about climate change initiatives?"
- "How did the parliament address budget concerns?"

## 📊 Analytics Features

The system includes several advanced analytics features:

### Speaker Analytics
- Track contribution patterns of individual speakers
- Compare speaking styles, topics, and participation between speakers
- Identify most active speakers and their areas of focus

### Session Analytics
- Analyze content and flow of individual parliamentary sessions
- Track changes in discussion topics across sessions
- Compare different sessions based on participants and topics

### Relationship Mapping
- Generate interaction networks between speakers
- Identify key influencers in parliamentary discussions
- Detect communities of speakers who frequently interact

### Sentiment Analysis
- Analyze emotional tone of parliamentary discourse
- Track sentiment trends over time and by speaker
- Identify keywords associated with positive and negative sentiment
- Compare sentiment patterns between different roles and groups

## 📂 Project Structure

```
parliamentary-minutes-chatbot/
├── config/                 # Configuration settings
├── project-info/           
│   └── data/               # Raw parliamentary minutes data
├── src/
│   ├── analytics/          # Analytics modules
│   │   ├── speaker_analysis.py     # Speaker analytics
│   │   ├── session_analysis.py     # Session analysis
│   │   ├── relationship_mapper.py  # Speaker relationship mapping
│   │   └── sentiment_analyzer.py   # Sentiment analysis
│   ├── api/                # FastAPI backend
│   │   ├── app.py          # Main API server
│   │   └── analytics_routes.py     # Analytics API endpoints
│   ├── data/               # Data processing utilities
│   ├── database/           # Vector database interface
│   ├── models/             # LLM interface and embeddings
│   ├── rag/                # RAG pipeline components
│   └── ui/                 # Streamlit user interface
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── output/                 # Processed data outputs
├── environment.yml         # Conda environment definition
├── pyproject.toml          # Poetry configuration
├── .gitignore              # Git ignore file
├── README.md               # This file
└── run_app.py              # Main runner script
```

## 🔜 Future Improvements

- **Multi-modal analysis** to include visual content from sessions
- **Timeline visualization** to track issues over extended periods
- **Speech pattern analysis** to identify rhetorical techniques
- **Argument extraction** to map positions on key issues
- **Enhanced relationship graphs** with interactive visualizations
- **Policy impact analysis** to track how discussions influence outcomes

## 📝 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

This project uses data from the Scottish Parliament. The system is built with open-source technologies including Qdrant, sentence-transformers, Ollama, FastAPI, and Streamlit. 