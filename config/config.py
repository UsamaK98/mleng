"""
Parliamentary Minutes Agentic Chatbot
Configuration settings
"""
import os
from pathlib import Path

# Project base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, "project-info", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data processing settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
MIN_CHUNK_SIZE = 100

# Vector database settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "parliamentary_minutes"
VECTOR_SIZE = 384

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:12b"

# Database settings
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "parliamentary_db"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"

# API settings
API_HOST = "127.0.0.1"
API_PORT = 8000
API_DEBUG = True

# UI settings
UI_PORT = 8501
UI_THEME = "light"

# Named paths to data files
MINUTES_CSV = os.path.join(OUTPUT_DIR, "parliamentary_minutes.csv")
SPEAKERS_CSV = os.path.join(OUTPUT_DIR, "speakers_list.csv")

# RAG settings
MAX_DOCUMENTS_RETRIEVED = 7
TEMPERATURE = 0.2
SYSTEM_PROMPT = """
You are an expert assistant specializing in Scottish Parliament meeting minutes.
Your task is to provide accurate, factual responses based on the information in the meeting transcripts.
Only answer questions based on the provided context. If you're unsure, admit that you don't know.
Always cite the date and speaker when providing information from the minutes.
""" 