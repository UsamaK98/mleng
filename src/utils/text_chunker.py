"""
Text chunking utilities for splitting documents
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import re

from config.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


class TextChunker:
    """
    Utility for chunking text documents
    """
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE, 
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum size of chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        
        # Split by sentences for more natural chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                # If current chunk is big enough, add it to chunks
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                
                # Start a new chunk, possibly with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    # Use the last part of the previous chunk for overlap
                    words = current_chunk.split()
                    overlap_words = words[-int(self.chunk_overlap/10):]  # Approx. 10 chars per word
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty and meets size requirements
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_metadata = metadata.copy() if metadata else {}
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def chunk_minutes_entry(self, entry: pd.Series) -> List[Dict[str, Any]]:
        """
        Create chunks from a parliamentary minutes entry
        
        Args:
            entry: Series representing a row from the minutes DataFrame
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Extract metadata from the entry
        metadata = {
            "speaker": entry["Speaker"],
            "role": entry["Role"],
            "date": entry["Date"],
            "timestamp": entry["Timestamp"],
            "source": entry["Filename"]
        }
        
        # Chunk the content
        return self.chunk_text(entry["Content"], metadata)
    
    def chunk_minutes_df(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a DataFrame of minutes into chunks
        
        Args:
            df: DataFrame of minutes entries
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        all_chunks = []
        
        for _, row in df.iterrows():
            chunks = self.chunk_minutes_entry(row)
            all_chunks.extend(chunks)
        
        return all_chunks 