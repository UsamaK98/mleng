"""
Data ingestion script for loading parliamentary minutes into the vector database
"""
import argparse
import pandas as pd
from tqdm import tqdm

from src.data.data_loader import MinutesDataLoader
from src.utils.text_chunker import TextChunker
from src.models.embeddings import EmbeddingModel, OllamaEmbeddingModel
from src.database.vector_store import VectorStore


def ingest_minutes(force_recreate: bool = False, use_ollama_embeddings: bool = False):
    """
    Main ingestion function
    
    Args:
        force_recreate: If True, recreate the vector collection even if it exists
        use_ollama_embeddings: If True, use Ollama embedding model instead of SentenceTransformers
    """
    print("Starting data ingestion process...")
    
    # Initialize components
    data_loader = MinutesDataLoader()
    chunker = TextChunker()
    
    # Select the embedding model based on configuration
    if use_ollama_embeddings:
        print("Using Ollama for embeddings (nomic-embed-text)...")
        embedding_model = OllamaEmbeddingModel()
    else:
        print("Using SentenceTransformers for embeddings...")
        embedding_model = EmbeddingModel()
        
    vector_store = VectorStore()
    
    # If force_recreate is True, recreate the collection
    if force_recreate:
        print("Force recreating the vector collection...")
        try:
            vector_store.client.delete_collection(collection_name=vector_store.collection_name)
            print(f"Deleted collection: {vector_store.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
        
        vector_store._ensure_collection()
    
    # Load the data
    print("Loading minutes data...")
    minutes_df, speakers_df = data_loader.load_data()
    
    # Process each session date separately to show progress
    session_dates = data_loader.get_sessions()
    print(f"Found {len(session_dates)} session dates")
    
    total_chunks = 0
    
    for date in session_dates:
        print(f"Processing session: {date}")
        date_df = data_loader.get_minutes_by_date(date)
        
        # Chunk the documents
        print("Chunking documents...")
        chunks = chunker.chunk_minutes_df(date_df)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        chunks_with_embeddings = embedding_model.embed_chunks(chunks)
        
        # Store in vector database
        print("Storing in vector database...")
        chunk_ids = vector_store.add_chunks(chunks_with_embeddings)
        
        total_chunks += len(chunks)
        print(f"Processed {len(chunks)} chunks for session {date}")
    
    print(f"Ingestion complete. Total chunks: {total_chunks}")
    
    return {
        "total_chunks": total_chunks,
        "session_dates": session_dates,
        "total_speakers": len(speakers_df),
        "total_entries": len(minutes_df)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest parliamentary minutes data into vector database")
    parser.add_argument("--force", action="store_true", help="Force recreate vector collection")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama embedding model (nomic-embed-text)")
    args = parser.parse_args()
    
    ingest_minutes(force_recreate=args.force, use_ollama_embeddings=args.ollama) 