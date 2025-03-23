# Parliamentary Meeting Minutes Analysis with GraphRAG

This system provides AI-powered analysis of parliamentary meeting minutes using a combination of knowledge graph and vector storage technologies (GraphRAG).

## Features

- Parliamentary data loading and preprocessing
- Named Entity Recognition using GLiNER
- Knowledge graph construction and visualization
- Vector embeddings for semantic search
- GraphRAG query processing combining graph and vector search
- Interactive web interface with Streamlit

## Setup

### Prerequisites

- Python 3.9+
- Conda (for virtual environment management)
- Ollama (for local LLM support)

### Virtual Environment Setup

This project uses a conda virtual environment. To set up and activate the environment:

```bash
# Create the environment
conda create -n mentor360 python=3.9

# Activate the environment
conda activate mentor360

# Install dependencies
pip install -r requirements.txt
```

### Ollama Setup

1. Install Ollama by following the instructions at [https://ollama.ai/](https://ollama.ai/)
2. Pull the required model:
   ```bash
   ollama pull llama3
   ```
3. Ensure the Ollama service is running before starting the application:
   ```bash
   # On Windows
   ollama serve
   
   # On macOS/Linux
   sudo systemctl start ollama
   ```

### Configuration

The application uses a configuration system that can be customized:

1. Default configuration is loaded from `src/utils/config.py`
2. You can override settings by creating a `config.json` file in the project root
3. Environment variables can also override configuration settings

## Running the Application

### Demo Script

To run the GraphRAG demo script that shows the core functionality:

```bash
conda activate mentor360
python src/demo/graphrag_demo.py
```

This will demonstrate:
- Loading parliamentary data
- Extracting entities
- Building a knowledge graph
- Performing queries using graph mode, vector mode, and hybrid mode

### Web Application

To run the Streamlit web interface:

```bash
conda activate mentor360
streamlit run src/web/app.py
```

Then open your browser to the URL displayed in the console (typically http://localhost:8501).

## Project Structure

- `src/data/`: Data loading and preprocessing
- `src/models/`: Core models including NER, knowledge graph, and GraphRAG
- `src/services/`: External service integrations (e.g., Ollama)
- `src/storage/`: Vector storage implementation
- `src/utils/`: Utility functions for logging, configuration, etc.
- `src/web/`: Streamlit web application
- `src/demo/`: Demo scripts

## Troubleshooting

### Ollama Service Issues

If you encounter errors related to the Ollama service:

1. Ensure Ollama is installed and running
2. Check that you have pulled the required model (`ollama pull llama3`)
3. The application will attempt to initialize the Ollama service automatically if not provided

### Embedding Dimensions

If you encounter an error about `embedding_dimensions`, ensure your Ollama configuration in `config.json` uses `embedding_dim` instead:

```json
{
  "ollama": {
    "base_url": "http://localhost:11434",
    "model_name": "llama3",
    "embedding_dim": 4096
  }
}
```

### Streamlit Errors

If you encounter errors with the Streamlit application:

1. Ensure you have the correct version of Streamlit installed (specified in requirements.txt)
2. Try clearing the Streamlit cache: `streamlit cache clear`
3. Check your Python version (3.9 recommended)

## Data Requirements

Parliamentary meeting minutes should be in a CSV format with columns for date, speaker, and content.

## License

This project is licensed under the terms of the MIT license. 