"""
Simple test runner for the Parliamentary Minutes Agentic Chatbot
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.pipeline import RAGPipeline
from src.data.data_loader import MinutesDataLoader

def test_data_loader():
    """Test data loader functionality"""
    print("Testing data loader...")
    data_loader = MinutesDataLoader()
    minutes_df, speakers_df = data_loader.load_data()
    
    print(f"Loaded {len(minutes_df)} minutes entries and {len(speakers_df)} speakers")
    
    # Show some sample data
    print("\nSample minutes data:")
    print(minutes_df.head(2))
    
    print("\nSample speakers data:")
    print(speakers_df.head(2))
    
    return True

def test_rag_pipeline():
    """Test RAG pipeline functionality"""
    print("Testing RAG pipeline...")
    pipeline = RAGPipeline()
    
    # Get metadata
    metadata = pipeline.get_metadata()
    print("Dataset metadata:")
    print(f"Total sessions: {metadata['total_sessions']}")
    print(f"Total speakers: {metadata['total_speakers']}")
    
    # Try a simple query
    print("\nTesting a simple query...")
    response = pipeline.process_query("What topics were discussed in the most recent session?")
    
    print("\nQuery response:")
    print(response["answer"])
    
    return True

if __name__ == "__main__":
    print("Running component tests...\n")
    
    # Test data loader
    if test_data_loader():
        print("\nData loader test completed successfully!\n")
    
    # Test RAG pipeline
    if test_rag_pipeline():
        print("\nRAG pipeline test completed successfully!") 