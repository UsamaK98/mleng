"""
Data loader for parliamentary minutes
"""
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional

from config.config import (
    MINUTES_CSV, 
    SPEAKERS_CSV
)

class MinutesDataLoader:
    """
    Loads and preprocesses parliamentary minutes data
    """
    def __init__(self, minutes_path: str = MINUTES_CSV, speakers_path: str = SPEAKERS_CSV):
        """
        Initialize the data loader
        
        Args:
            minutes_path: Path to the minutes CSV file
            speakers_path: Path to the speakers CSV file
        """
        self.minutes_path = minutes_path
        self.speakers_path = speakers_path
        self.minutes_df = None
        self.speakers_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the minutes and speakers data from CSV files
        
        Returns:
            Tuple of (minutes_df, speakers_df)
        """
        if not os.path.exists(self.minutes_path):
            raise FileNotFoundError(f"Minutes file not found: {self.minutes_path}")
        
        if not os.path.exists(self.speakers_path):
            raise FileNotFoundError(f"Speakers file not found: {self.speakers_path}")
        
        self.minutes_df = pd.read_csv(self.minutes_path)
        self.speakers_df = pd.read_csv(self.speakers_path)
        
        print(f"Loaded {len(self.minutes_df)} minutes entries and {len(self.speakers_df)} speakers")
        
        return self.minutes_df, self.speakers_df
    
    def get_sessions(self) -> List[str]:
        """
        Get list of unique session dates
        
        Returns:
            List of session dates
        """
        if self.minutes_df is None:
            self.load_data()
        
        return sorted(self.minutes_df['Date'].unique().tolist())
    
    def get_speakers(self) -> List[str]:
        """
        Get list of unique speakers
        
        Returns:
            List of speaker names
        """
        if self.speakers_df is None:
            self.load_data()
        
        return self.speakers_df['Speaker'].tolist()
    
    def get_speaker_info(self, speaker_name: str) -> Optional[Dict]:
        """
        Get information about a specific speaker
        
        Args:
            speaker_name: Name of the speaker
            
        Returns:
            Dictionary with speaker information or None if not found
        """
        if self.speakers_df is None:
            self.load_data()
        
        speaker_row = self.speakers_df[self.speakers_df['Speaker'] == speaker_name]
        
        if speaker_row.empty:
            return None
        
        return speaker_row.iloc[0].to_dict()
    
    def get_minutes_by_speaker(self, speaker_name: str) -> pd.DataFrame:
        """
        Get all minutes entries for a specific speaker
        
        Args:
            speaker_name: Name of the speaker
            
        Returns:
            DataFrame with filtered minutes
        """
        if self.minutes_df is None:
            self.load_data()
        
        return self.minutes_df[self.minutes_df['Speaker'] == speaker_name]
    
    def get_minutes_by_date(self, date: str) -> pd.DataFrame:
        """
        Get all minutes entries for a specific date
        
        Args:
            date: Session date in format YYYY-MM-DD
            
        Returns:
            DataFrame with filtered minutes
        """
        if self.minutes_df is None:
            self.load_data()
        
        return self.minutes_df[self.minutes_df['Date'] == date]
    
    def get_speaker_session_contributions(self, speaker_name: str, date: str) -> pd.DataFrame:
        """
        Get all contributions from a specific speaker on a specific date
        
        Args:
            speaker_name: Name of the speaker
            date: Session date in format YYYY-MM-DD
            
        Returns:
            DataFrame with filtered minutes
        """
        if self.minutes_df is None:
            self.load_data()
        
        return self.minutes_df[(self.minutes_df['Speaker'] == speaker_name) & 
                               (self.minutes_df['Date'] == date)] 