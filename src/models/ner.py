"""
Named Entity Recognition module using GLiNER.

This module provides functionality for extracting named entities from parliamentary
meeting minutes using the GLiNER model.
"""

import os
import time
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Import GLiNER
try:
    from gliner import GLiNER
except ImportError:
    raise ImportError(
        "GLiNER is required for NER functionality. "
        "Please install it using `pip install gliner`."
    )

from src.utils.logging import logger, ModelLogger
from src.utils.config import config_manager

class EntityExtractor:
    """Entity extraction using GLiNER model."""
    
    def __init__(self, model_name: Optional[str] = None, use_gpu: Optional[bool] = None):
        """Initialize the entity extractor.
        
        Args:
            model_name: Name of the GLiNER model to use. If None, uses the configured value.
            use_gpu: Whether to use GPU for inference. If None, uses the configured value.
        """
        self.model_name = model_name or config_manager.config.gliner.model_name
        self.use_gpu = use_gpu if use_gpu is not None else config_manager.config.gliner.use_gpu
        self.entity_types = config_manager.config.gliner.entity_types
        self.batch_size = config_manager.config.gliner.batch_size
        self.model = None
        
        # Create cache directory
        self.cache_dir = Path(config_manager.config.processed_data_dir) / "ner_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized EntityExtractor with model: {self.model_name}")
    
    def load_model(self) -> bool:
        """Load the GLiNER model.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self.model is not None:
            return True
        
        try:
            start_time = time.time()
            logger.info(f"Loading GLiNER model: {self.model_name}")
            
            # Try with fallback models if the specified one fails
            model_options = [
                self.model_name,
                "gliner/gliner-large",
                "gliner/gliner-base"
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"Attempting to load model: {model_name}")
                    self.model = GLiNER.from_pretrained(
                        model_name,
                        device="cuda" if self.use_gpu else "cpu"
                    )
                    self.model_name = model_name  # Update model name to successful one
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {str(e)}")
            
            if self.model is None:
                logger.error("All model loading attempts failed")
                return False
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"GLiNER model {self.model_name} loaded in {duration_ms:.2f}ms")
            return True
        
        except Exception as e:
            logger.error(f"Error loading GLiNER model: {str(e)}")
            return False
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from a single text.
        
        Args:
            text: Text to extract entities from.
            
        Returns:
            List of entity dictionaries with 'text', 'label', 'start', 'end' keys.
        """
        if not text or not text.strip():
            return []
        
        if self.model is None and not self.load_model():
            logger.error("Model not loaded, cannot extract entities")
            return []
        
        start_time = time.time()
        
        try:
            # Extract entities using GLiNER
            gliner_entities = self.model.predict_entities(
                text, 
                self.entity_types
            )
            
            # Convert to standard format with start/end positions
            entities = []
            for entity in gliner_entities:
                entities.append({
                    'text': entity["text"],
                    'label': entity["label"],
                    'start': entity.get("start", 0),  # If GLiNER doesn't provide position, default to 0
                    'end': entity.get("end", len(entity["text"]))
                })
            
            duration_ms = (time.time() - start_time) * 1000
            ModelLogger.log_inference(
                model_name=f"gliner-{self.model_name.split('/')[-1]}",
                input_text=text[:100] + "...",
                output_text=str(entities)[:100] + "...",
                duration_ms=duration_ms
            )
            
            return entities
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Error extracting entities: {str(e)}")
            ModelLogger.log_inference(
                model_name=f"gliner-{self.model_name.split('/')[-1]}",
                input_text=text[:100] + "...",
                output_text="",
                duration_ms=duration_ms,
                success=False
            )
            return []
    
    def extract_entities_from_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'Content',
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
        """Extract entities from a dataframe of parliamentary minutes.
        
        Args:
            df: DataFrame containing parliamentary minutes.
            text_column: Name of the column containing text to extract entities from.
            use_cache: Whether to use cached results if available.
            
        Returns:
            Tuple of (Enhanced DataFrame with entity columns, Entity mapping dictionary)
        """
        if df.empty or text_column not in df.columns:
            logger.warning(f"DataFrame is empty or missing {text_column} column")
            return df.copy(), {}
        
        # Check if we have entry_id for unique identification
        if 'entry_id' not in df.columns:
            df = df.copy()
            df['entry_id'] = df.index
        
        # Prepare entity cache file path based on dataframe hash
        df_hash = str(hash(frozenset(df['entry_id'].astype(str))))[:10]
        cache_file = self.cache_dir / f"entities_{df_hash}.json"
        
        # Try to load from cache if enabled
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    entity_map = json.load(f)
                logger.info(f"Loaded {len(entity_map)} cached entity records")
                return self._enhance_dataframe_with_entities(df, entity_map), entity_map
            except Exception as e:
                logger.warning(f"Failed to load entity cache: {str(e)}")
        
        # Ensure model is loaded
        if self.model is None and not self.load_model():
            logger.error("Model not loaded, cannot extract entities")
            return df.copy(), {}
        
        start_time = time.time()
        entity_map = {}
        
        logger.info(f"Extracting entities from {len(df)} records")
        
        # Batch process texts for efficiency
        texts = df[text_column].fillna("").tolist()
        entry_ids = df['entry_id'].astype(str).tolist()
        
        try:
            # Process in batches
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting entities"):
                batch_texts = texts[i:i+self.batch_size]
                batch_ids = entry_ids[i:i+self.batch_size]
                
                for j, (text, entry_id) in enumerate(zip(batch_texts, batch_ids)):
                    if not text or not text.strip():
                        entity_map[entry_id] = []
                        continue
                    
                    entities = self.extract_entities_from_text(text)
                    entity_map[entry_id] = entities
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(entity_map, f)
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Extracted entities from {len(df)} records in {duration_ms/1000:.2f}s")
            
            # Enhance dataframe with extracted entities
            enhanced_df = self._enhance_dataframe_with_entities(df, entity_map)
            
            return enhanced_df, entity_map
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Error batch extracting entities: {str(e)}")
            return df.copy(), {}
    
    def _enhance_dataframe_with_entities(
        self, 
        df: pd.DataFrame, 
        entity_map: Dict[str, List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Add entity information to the dataframe.
        
        Args:
            df: Original dataframe.
            entity_map: Dictionary mapping entry_id to entity lists.
            
        Returns:
            Enhanced dataframe with entity columns.
        """
        enhanced_df = df.copy()
        
        # Add columns for each entity type
        for entity_type in self.entity_types:
            col_name = f"entities_{entity_type}"
            enhanced_df[col_name] = ""
        
        # Add entities to each row
        for idx, row in enhanced_df.iterrows():
            entry_id = str(row['entry_id'])
            if entry_id in entity_map:
                entities = entity_map[entry_id]
                
                # Group entities by type
                for entity_type in self.entity_types:
                    type_entities = [e['text'] for e in entities if e['label'] == entity_type]
                    if type_entities:
                        enhanced_df.at[idx, f"entities_{entity_type}"] = ", ".join(type_entities)
        
        return enhanced_df

# Usage example:
# from src.models.ner import EntityExtractor
# from src.data.loader import ParliamentaryDataLoader
# 
# # Load parliamentary data
# loader = ParliamentaryDataLoader()
# loader.load_data()
# 
# # Get a session's data
# session_data = loader.get_session_data("2024-09-10")
# 
# # Extract entities
# extractor = EntityExtractor()
# enhanced_df, entity_map = extractor.extract_entities_from_dataframe(session_data)
# 
# # Check extracted entities
# print(enhanced_df[["Speaker", "entities_person", "entities_organization", "entities_topic"]].head()) 