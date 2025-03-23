"""
Test script for the GLiNER model.

This script tests loading and using the GLiNER model with the 
specified configuration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logging import logger
from src.utils.config import config_manager
from gliner import GLiNER

def main():
    """Test GLiNER model loading and entity extraction."""
    logger.info("Testing GLiNER model")
    
    # Get model configuration
    model_name = config_manager.config.gliner.model_name
    entity_types = config_manager.config.gliner.entity_types
    use_gpu = config_manager.config.gliner.use_gpu
    
    logger.info(f"Using model: {model_name}")
    logger.info(f"Entity types: {entity_types}")
    
    # Load model
    logger.info("Loading GLiNER model...")
    model = GLiNER.from_pretrained(
        model_name,
        device="cuda" if use_gpu else "cpu"
    )
    logger.info("Model loaded successfully")
    
    # Test with sample text
    test_text = """
    The Parliamentary Budget Committee met on Wednesday to discuss the new healthcare 
    funding bill proposed by Prime Minister Johnson. The committee, led by Dr. Sarah Wilson, 
    reviewed the £500 million allocation scheduled for implementation on January 15, 2025. 
    Representatives from the National Health Service and the Ministry of Finance also attended.
    The meeting took place at the Westminster Parliament building in London.
    """
    
    logger.info("Extracting entities from sample text...")
    entities = model.predict_entities(test_text, entity_types)
    
    # Print results
    logger.info(f"Found {len(entities)} entities:")
    for entity in entities:
        logger.info(f"{entity['text']} => {entity['label']}")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main() 