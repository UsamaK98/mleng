# Parliamentary Minutes Agentic Chatbot

An AI-powered chatbot system for querying and analyzing Scottish Parliament meeting minutes, leveraging RAG (Retrieval-Augmented Generation) technology.

## 🔍 Overview

This project provides an intelligent interface to search, analyze, and extract insights from Scottish Parliament meeting minutes. The system uses a vector database to store document embeddings, allowing for semantic retrieval of relevant content when users ask questions. The main features include:

- **Natural language querying** of parliamentary minutes
- **Speaker analysis** to examine contributions by specific MPs
- **Topic exploration** to find discussions on specific subjects
- **Document citations** linking responses to source material
- **Interactive UI** for easy engagement with the data

## 🏗️ System Architecture

The system is built with a modular architecture, consisting of several key components:

- **Vector Database (Qdrant)**: Stores document embeddings for semantic search
- **Embedding Model**: Converts text to vector embeddings using sentence-transformers
- **LLM Integration**: Connects to Ollama for generating responses
- **API Layer**: FastAPI backend exposes endpoints for queries
- **UI**: Streamlit frontend for user interaction

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
```

## 📋 Prerequisites

- **Python 3.9+**
- **Docker** for running Qdrant and Ollama services
- **GPU recommended** for faster embedding generation and inference (but not required)

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/parliamentary-minutes-chatbot.git
cd parliamentary-minutes-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
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

- **POST /query**: Process general queries about parliamentary minutes
  ```json
  {
    "query": "What discussions took place about healthcare?",
    "filters": {"date": "2024-06-26"}
  }
  ```

- **POST /entity**: Process queries about specific entities (speakers)
  ```json
  {
    "entity": "John Smith"
  }
  ```

- **POST /topic**: Process queries about specific topics
  ```json
  {
    "topic": "education"
  }
  ```

- **GET /metadata**: Retrieve metadata about the parliamentary minutes dataset

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

## 📂 Project Structure

```
parliamentary-minutes-chatbot/
├── config/                 # Configuration settings
├── project-info/           
│   └── data/               # Raw parliamentary minutes data
├── src/
│   ├── api/                # FastAPI backend
│   ├── data/               # Data processing utilities
│   ├── database/           # Vector database interface
│   ├── models/             # LLM interface and embeddings
│   ├── rag/                # RAG pipeline components
│   └── ui/                 # Streamlit user interface
├── output/                 # Processed data outputs
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── run.py                  # Original run script
└── run_app.py              # Improved runner script
```

## 🔜 Future Improvements

- **Hybrid search implementation** combining dense and sparse retrieval
- **Relationship mapping** between speakers and topics
- **Cross-session analysis** to track issues over time
- **Sentiment analysis** for contributions
- **UI enhancements** with more visualizations
- **GPU optimization** for faster processing

## 📝 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

This project uses data from the Scottish Parliament. The system is built with open-source technologies including Qdrant, sentence-transformers, Ollama, FastAPI, and Streamlit. 