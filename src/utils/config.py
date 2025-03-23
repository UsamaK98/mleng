"""
Configuration module for the Parliamentary Meeting Analyzer.

This module manages configuration settings for the application, including
model parameters, GraphRAG settings, and application preferences.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass, field, asdict

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(exist_ok=True)

# Default config file path
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.json"

class DotDict(dict):
    """Dot notation access to dictionary attributes."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            elif isinstance(value, list):
                self[key] = [DotDict(item) if isinstance(item, dict) else item for item in value]

@dataclass
class OllamaConfig:
    """Configuration for Ollama API."""
    base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text:latest"
    generation_model: str = "gemma3:12b"
    embedding_dim: int = 3072
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout_seconds: int = 120
    batch_size: int = 10

@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "parliament"
    vector_size: int = 3072  # Same as embedding_dim
    distance: str = "Cosine"
    batch_size: int = 100
    use_local: bool = True
    local_path: str = str(Path(__file__).parent.parent.parent / "data" / "qdrant_data")
    prefer_grpc: bool = True
    timeout: float = 30.0

@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB vector database (fallback)."""
    collection_name: str = "parliament"
    persist_directory: str = str(Path(__file__).parent.parent.parent / "data" / "chromadb_data")
    embedding_function: str = "default"  # Use default embeddings from OllamaService
    batch_size: int = 100
    distance: str = "cosine"

@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG."""
    max_hops: int = 2
    max_nodes: int = 50
    node_types: List[str] = field(default_factory=lambda: [
        "person", "topic", "organization", "legislation", "location"
    ])
    relation_types: List[str] = field(default_factory=lambda: [
        "made_statement", "about_topic", "mentions_person", 
        "participated_in", "part_of", "responds_to", 
        "mentions_organization", "references_legislation",
        "mentions_location", "related_topic", "affiliated_with"
    ])
    cache_expiry_seconds: int = 3600
    similarity_threshold: float = 0.7

@dataclass
class GLiNERConfig:
    """Configuration for GLiNER NER model."""
    model_name: str = "EmergentMethods/gliner_large_news-v2.1"
    use_gpu: bool = False
    entity_types: List[str] = field(default_factory=lambda: [
        "person", "organization", "location", "legislation", "topic", "date", "time", 
        "event", "facility", "vehicle", "number"
    ])
    batch_size: int = 8
    cache_results: bool = True

@dataclass
class Config:
    """Main configuration class for Parliamentary Meeting Analyzer."""
    base_dir: str = str(Path(__file__).parent.parent.parent)
    data_dir: str = str(Path(__file__).parent.parent.parent / "output")
    output_dir: str = str(Path(__file__).parent.parent.parent / "output" / "output-new")
    processed_data_dir: str = str(Path(__file__).parent.parent.parent / "processed_data")
    log_dir: str = str(Path(__file__).parent.parent.parent / "logs")
    cache_dir: str = str(Path(__file__).parent.parent.parent / "cache")
    
    # Component configurations
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    gliner: GLiNERConfig = field(default_factory=GLiNERConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    
    # UI configuration
    ui_title: str = "Parliamentary Meeting Analyzer"
    ui_description: str = "AI-powered parliamentary meeting analysis and knowledge exploration"
    ui_theme: str = "light"

class ConfigManager:
    """Configuration manager singleton."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize configuration."""
        # Create default configuration
        self.config = Config()
        
        # Set appropriate paths
        base_dir = Path(__file__).parent.parent.parent
        
        # Override with environment variables
        if os.environ.get("DATA_DIR"):
            self.config.data_dir = os.environ.get("DATA_DIR")
        
        if os.environ.get("OUTPUT_DIR"):
            self.config.output_dir = os.environ.get("OUTPUT_DIR")
        
        # Ensure directories exist
        for dir_path in [
            self.config.output_dir,
            self.config.processed_data_dir,
            self.config.log_dir,
            self.config.cache_dir
        ]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Load config from file if exists
        config_file = base_dir / "config.json"
        if config_file.exists():
            self.load_config(config_file)
    
    def load_config(self, config_file: Path) -> bool:
        """Load configuration from file.
        
        Args:
            config_file: Path to configuration file.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Convert to DotDict for dot notation access
            config_dict = DotDict(config_data)
            
            # Update configuration
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    if isinstance(value, dict) and key in ["ollama", "graphrag", "gliner", "qdrant", "chromadb"]:
                        # Handle nested config objects
                        if key == "ollama":
                            for k, v in value.items():
                                setattr(self.config.ollama, k, v)
                        elif key == "graphrag":
                            for k, v in value.items():
                                setattr(self.config.graphrag, k, v)
                        elif key == "gliner":
                            for k, v in value.items():
                                setattr(self.config.gliner, k, v)
                        elif key == "qdrant":
                            for k, v in value.items():
                                setattr(self.config.qdrant, k, v)
                        elif key == "chromadb":
                            for k, v in value.items():
                                setattr(self.config.chromadb, k, v)
                    else:
                        setattr(self.config, key, value)
            
            return True
        
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            return False
    
    def save_config(self, config_file: Path) -> bool:
        """Save configuration to file.
        
        Args:
            config_file: Path to save configuration to.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            # Convert to dictionary
            config_dict = asdict(self.config)
            
            # Write to file
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
            return False

# Create singleton instance
config_manager = ConfigManager()

# Usage example:
# from src.utils.config import config_manager
# 
# # Get current configuration
# current_config = config_manager.config
# 
# # Update configuration
# config_manager.update_config({
#     "ollama": {
#         "chat_temperature": 0.8
#     }
# })
#
# # Access configuration values
# chat_model = config_manager.config.ollama.chat_model 