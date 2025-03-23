"""
Cache management utility module.

This module provides functionality for managing, validating, and reporting on cache
status for various components of the system including NER and vector storage.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.utils.logging import logger
from src.utils.config import config_manager

class CacheManager:
    """Cache management utility class."""
    
    def __init__(self):
        """Initialize the cache manager."""
        self.processed_data_dir = Path(config_manager.config.processed_data_dir)
        self.ner_cache_dir = self.processed_data_dir / "ner_cache"
        self.vector_cache_dir = self.processed_data_dir / "vector_cache"
        self.graph_cache_dir = self.processed_data_dir / "graph_cache"
        self.graphrag_cache_dir = self.processed_data_dir / "graphrag_cache"
        
        # Ensure cache directories exist
        self.ner_cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_cache_dir.mkdir(parents=True, exist_ok=True)
        self.graph_cache_dir.mkdir(parents=True, exist_ok=True)
        self.graphrag_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata files
        self.ner_metadata_file = self.ner_cache_dir / "metadata.json"
        self.vector_metadata_file = self.vector_cache_dir / "vector_metadata.json"
        self.graph_metadata_file = self.graph_cache_dir / "graph_metadata.json"
        self.graphrag_metadata_file = self.graphrag_cache_dir / "graphrag_metadata.json"
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get the status of all caches.
        
        Returns:
            Dictionary containing cache status information.
        """
        # Get basic status for each cache
        ner_cache_status = self._get_ner_cache_status()
        vector_cache_status = self._get_vector_cache_status()
        graph_cache_status = self._get_graph_cache_status()
        graphrag_cache_status = self._get_graphrag_cache_status()
        
        # Add validation status to each cache
        if ner_cache_status["exists"]:
            ner_cache_status["valid"] = self._validate_ner_cache()
        
        if vector_cache_status["exists"]:
            vector_cache_status["valid"] = self._validate_vector_cache()
        
        if graph_cache_status["exists"]:
            graph_cache_status["valid"] = self._validate_graph_cache()
        
        if graphrag_cache_status["exists"]:
            graphrag_cache_status["valid"] = self._validate_graphrag_cache()
        
        status = {
            "ner_cache": ner_cache_status,
            "vector_cache": vector_cache_status,
            "graph_cache": graph_cache_status,
            "graphrag_cache": graphrag_cache_status,
            "last_checked": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Calculate overall statistics
        cache_sizes = {}
        for cache_type in ["ner_cache", "vector_cache", "graph_cache", "graphrag_cache"]:
            if status[cache_type]["exists"]:
                cache_sizes[cache_type] = status[cache_type]["size_mb"]
        
        status["total_cache_size_mb"] = sum(cache_sizes.values())
        status["cache_count"] = sum(1 for s in status.values() if isinstance(s, dict) and s.get("exists", False))
        
        return status
    
    def _get_ner_cache_status(self) -> Dict[str, Any]:
        """Get the status of the NER cache.
        
        Returns:
            Dictionary containing NER cache status information.
        """
        if not self.ner_cache_dir.exists():
            return {"exists": False}
        
        cache_files = list(self.ner_cache_dir.glob("entities_*.json"))
        
        # Try to load metadata
        metadata = {}
        if self.ner_metadata_file.exists():
            try:
                with open(self.ner_metadata_file, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load NER cache metadata: {str(e)}")
        
        # Calculate cache size
        total_size_bytes = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        return {
            "exists": True,
            "file_count": len(cache_files),
            "size_mb": total_size_mb,
            "model_name": metadata.get("model_name", "unknown"),
            "entity_types": metadata.get("entity_types", []),
            "last_updated": metadata.get("last_updated", "unknown")
        }
    
    def _get_vector_cache_status(self) -> Dict[str, Any]:
        """Get the status of the vector cache.
        
        Returns:
            Dictionary containing vector cache status information.
        """
        if not self.vector_cache_dir.exists():
            return {"exists": False}
        
        cache_files = list(self.vector_cache_dir.glob("embeddings_*.json"))
        
        # Try to load metadata
        metadata = {}
        if self.vector_metadata_file.exists():
            try:
                with open(self.vector_metadata_file, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load vector cache metadata: {str(e)}")
        
        # Calculate cache size
        total_size_bytes = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        # Get collection info
        collections = metadata.get("collections", {})
        
        return {
            "exists": True,
            "file_count": len(cache_files),
            "size_mb": total_size_mb,
            "embedding_model": metadata.get("embedding_model", "unknown"),
            "embedding_dim": metadata.get("embedding_dim", 0),
            "collections": collections,
            "last_updated": metadata.get("last_updated", "unknown")
        }
    
    def _get_graph_cache_status(self) -> Dict[str, Any]:
        """Get the status of the graph cache.
        
        Returns:
            Dictionary containing graph cache status information.
        """
        if not self.graph_cache_dir.exists():
            return {"exists": False}
        
        cache_files = list(self.graph_cache_dir.glob("*.json"))
        
        # Calculate cache size
        total_size_bytes = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        # Get latest modification time
        last_updated = "unknown"
        if cache_files:
            latest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
            last_updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_file.stat().st_mtime))
        
        return {
            "exists": True,
            "file_count": len(cache_files),
            "size_mb": total_size_mb,
            "last_updated": last_updated
        }
    
    def _get_graphrag_cache_status(self) -> Dict[str, Any]:
        """Get the status of the GraphRAG cache.
        
        Returns:
            Dictionary containing GraphRAG cache status information.
        """
        if not self.graphrag_cache_dir.exists():
            return {"exists": False}
        
        cache_files = list(self.graphrag_cache_dir.glob("*.json"))
        
        # Calculate cache size
        total_size_bytes = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        # Get latest modification time
        last_updated = "unknown"
        if cache_files:
            latest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
            last_updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_file.stat().st_mtime))
        
        return {
            "exists": True,
            "file_count": len(cache_files),
            "size_mb": total_size_mb,
            "last_updated": last_updated
        }
    
    def validate_caches(self) -> Dict[str, bool]:
        """Validate all caches to ensure they are usable.
        
        Returns:
            Dictionary mapping cache types to validation status.
        """
        return {
            "ner_cache": self._validate_ner_cache(),
            "vector_cache": self._validate_vector_cache(),
            "graph_cache": self._validate_graph_cache(),
            "graphrag_cache": self._validate_graphrag_cache()
        }
    
    def _validate_ner_cache(self) -> bool:
        """Validate the NER cache.
        
        Returns:
            Whether the cache is valid.
        """
        if not self.ner_cache_dir.exists() or not self.ner_metadata_file.exists():
            return False
        
        try:
            with open(self.ner_metadata_file, "r") as f:
                metadata = json.load(f)
            
            # Check if metadata has required fields
            required_fields = ["model_name", "entity_types", "version"]
            for field in required_fields:
                if field not in metadata:
                    logger.warning(f"NER cache metadata missing required field: {field}")
                    return False
            
            # Check if at least one cache file exists
            cache_files = list(self.ner_cache_dir.glob("entities_*.json"))
            if not cache_files:
                logger.warning("NER cache directory exists but contains no cache files")
                return False
            
            # Validate a random cache file
            sample_file = cache_files[0]
            with open(sample_file, "r") as f:
                entity_map = json.load(f)
            
            if not isinstance(entity_map, dict):
                logger.warning("NER cache file does not contain a valid entity map")
                return False
            
            return True
        
        except Exception as e:
            logger.warning(f"Error validating NER cache: {str(e)}")
            return False
    
    def _validate_vector_cache(self) -> bool:
        """Validate the vector cache.
        
        Returns:
            Whether the cache is valid.
        """
        # Similar validation logic for vector cache
        if not self.vector_cache_dir.exists() or not self.vector_metadata_file.exists():
            return False
        
        try:
            with open(self.vector_metadata_file, "r") as f:
                metadata = json.load(f)
            
            # Check if metadata has required fields
            required_fields = ["embedding_model", "embedding_dim", "version"]
            for field in required_fields:
                if field not in metadata:
                    logger.warning(f"Vector cache metadata missing required field: {field}")
                    return False
            
            return True
        
        except Exception as e:
            logger.warning(f"Error validating vector cache: {str(e)}")
            return False
    
    def _validate_graph_cache(self) -> bool:
        """Validate the graph cache.
        
        Returns:
            Whether the cache is valid.
        """
        # Basic validation - just check if files exist
        if not self.graph_cache_dir.exists():
            return False
        
        cache_files = list(self.graph_cache_dir.glob("*.json"))
        return len(cache_files) > 0
    
    def _validate_graphrag_cache(self) -> bool:
        """Validate the GraphRAG cache.
        
        Returns:
            Whether the cache is valid.
        """
        # Basic validation - just check if files exist
        if not self.graphrag_cache_dir.exists():
            return False
        
        cache_files = list(self.graphrag_cache_dir.glob("*.json"))
        return len(cache_files) > 0
    
    def clear_cache(self, cache_type: str) -> bool:
        """Clear a specific cache type.
        
        Args:
            cache_type: The type of cache to clear ("ner", "vector", "graph", "graphrag", or "all").
            
        Returns:
            Whether the cache was successfully cleared.
        """
        try:
            if cache_type == "all":
                self._clear_directory(self.ner_cache_dir)
                self._clear_directory(self.vector_cache_dir)
                self._clear_directory(self.graph_cache_dir)
                self._clear_directory(self.graphrag_cache_dir)
                logger.info("All caches cleared successfully")
                return True
            
            elif cache_type == "ner":
                self._clear_directory(self.ner_cache_dir)
                logger.info("NER cache cleared successfully")
                return True
            
            elif cache_type == "vector":
                self._clear_directory(self.vector_cache_dir)
                logger.info("Vector cache cleared successfully")
                return True
            
            elif cache_type == "graph":
                self._clear_directory(self.graph_cache_dir)
                logger.info("Graph cache cleared successfully")
                return True
            
            elif cache_type == "graphrag":
                self._clear_directory(self.graphrag_cache_dir)
                logger.info("GraphRAG cache cleared successfully")
                return True
            
            else:
                logger.warning(f"Unknown cache type: {cache_type}")
                return False
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def _clear_directory(self, directory: Path) -> None:
        """Clear all files in a directory.
        
        Args:
            directory: The directory to clear.
        """
        if not directory.exists():
            return
        
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()


# Create singleton instance
cache_manager = CacheManager() 