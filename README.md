# Parliamentary Minutes Agentic Chatbot

An AI-powered chatbot for interacting with Scottish Parliament meeting minutes, enabling users to query by entity name, topic, or free-form questions.

## Features

- **Natural Language Querying**: Ask questions about parliamentary proceedings in plain English
- **Speaker Analysis**: Search for specific speakers and view their contributions
- **Topic Exploration**: Explore discussions on specific topics across sessions
- **Document Citations**: All responses include citations to the source documents
- **Interactive UI**: User-friendly interface with chat and search capabilities

## System Architecture

The system uses a Retrieval-Augmented Generation (RAG) pipeline with the following components:

- **Vector Database**: Qdrant for semantic search of parliamentary content
- **Embedding Model**: Sentence transformers for generating text embeddings
- **LLM Integration**: Local LLM deployment via Ollama
- **API Layer**: FastAPI for backend services
- **UI**: Streamlit for the user interface

## Prerequisites

- Python 3.8+
- Docker (for running Qdrant and Ollama)
- GPU recommended for faster embedding generation and inference

### Docker Services

Make sure you have the following Docker services running:

1. **Qdrant**:
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 -v ./qdrant_data:/qdrant/storage qdrant/qdrant
   ```

2. **Ollama** (with models):
   ```bash
   docker run -d -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
   ```

3. Pull a model in Ollama:
   ```bash
   docker exec -it <ollama_container_id> ollama pull mistral
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd parliamentary-minutes-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ingest the data (this will process the parliamentary minutes and create embeddings):
   ```bash
   python run.py --ingest
   ```

## Usage

### Running the Application

Run the complete application (API + UI):
```bash
python run.py
```

This will start:
- FastAPI server on http://localhost:8000
- Streamlit UI on http://localhost:8501

### Running Components Separately

Run only the API server:
```bash
python run.py --api-only
```

Run only the UI:
```bash
python run.py --ui-only
```

### API Endpoints

- `POST /query`: Process general queries about parliamentary minutes
- `POST /entity`: Process queries about specific speakers
- `POST /topic`: Process queries about specific topics
- `GET /metadata`: Get metadata about the parliamentary dataset

## Example Queries

### General Questions
- "What did Kate Forbes say about European funding?"
- "What were the main points discussed in the October 2024 session?"
- "Summarize the debate about healthcare from January 2025"

### Entity Queries
- "Kate Forbes"
- "The Convener"
- "Stephen Boyle"

### Topic Queries
- "European structural funds"
- "budget"
- "healthcare"

## Project Structure

```
├── config/              # Configuration files
├── output/              # Processed data and visualizations
├── project-info/        # Project documentation
│   └── data/            # Raw parliamentary minutes
├── src/                 # Source code
│   ├── api/             # FastAPI application
│   ├── data/            # Data processing modules
│   ├── database/        # Database interfaces
│   ├── models/          # ML models
│   ├── rag/             # RAG pipeline components
│   ├── ui/              # Streamlit UI
│   └── utils/           # Utility functions
├── implementation_plan.md # Development plan
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── run.py               # Main runner script
```

## Next Steps and Future Improvements

- Implement hybrid search combining dense and sparse retrievals
- Add relationship mapping between speakers
- Enable cross-session analysis
- Add sentiment analysis for contributions
- Enhance UI with more visualizations
- Implement GPU optimization for better performance 