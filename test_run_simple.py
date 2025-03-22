"""
Simple test runner for the Parliamentary Minutes Agentic Chatbot
Testing the data loader only
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    
    # Show some sessions
    print("\nSessions:")
    sessions = data_loader.get_sessions()
    for session in sessions:
        print(f"- {session}")
    
    # Show some speakers
    print("\nTop 5 speakers:")
    speakers = data_loader.get_speakers()[:5]
    for speaker in speakers:
        print(f"- {speaker}")
    
    return True

if __name__ == "__main__":
    print("Running data loader test...\n")
    
    if test_data_loader():
        print("\nData loader test completed successfully!") 