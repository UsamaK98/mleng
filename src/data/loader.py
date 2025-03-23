"""
Data loader module for the Parliamentary Meeting Analyzer.

This module handles loading and preprocessing parliamentary meeting data
from CSV files into structured formats for analysis.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from src.utils.logging import logger, DataLogger
from src.utils.config import config_manager

class ParliamentaryDataLoader:
    """Loader for parliamentary meeting data."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files. If None, uses the configured value.
        """
        self.data_dir = data_dir or config_manager.config.data_dir
        self.minutes_df = None
        self.speakers_df = None
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} doesn't exist")
    
    def load_data(self, force_reload: bool = False) -> bool:
        """Load parliamentary minutes and speakers data.
        
        Args:
            force_reload: Whether to force reload data even if already loaded.
            
        Returns:
            True if data loaded successfully, False otherwise.
        """
        if self.minutes_df is not None and self.speakers_df is not None and not force_reload:
            logger.debug("Data already loaded, skipping load")
            return True
        
        start_time = time.time()
        success = True
        
        try:
            # Load parliamentary minutes
            minutes_path = os.path.join(self.data_dir, "parliamentary_minutes.csv")
            if not os.path.exists(minutes_path):
                logger.error(f"Minutes file not found at {minutes_path}")
                success = False
            else:
                self.minutes_df = pd.read_csv(minutes_path)
                logger.info(f"Loaded {len(self.minutes_df)} minutes entries")
            
            # Load speakers list if available
            speakers_path = os.path.join(self.data_dir, "speakers_list.csv")
            if os.path.exists(speakers_path):
                self.speakers_df = pd.read_csv(speakers_path)
                logger.info(f"Loaded {len(self.speakers_df)} speakers")
            else:
                # If speakers list not available, create from minutes data
                if self.minutes_df is not None:
                    self._extract_speakers_from_minutes()
                else:
                    success = False
            
            # Clean and preprocess data
            if success:
                self._preprocess_data()
            
            duration_ms = (time.time() - start_time) * 1000
            DataLogger.log_data_load(
                data_source=self.data_dir,
                num_records=len(self.minutes_df) if self.minutes_df is not None else 0,
                duration_ms=duration_ms,
                success=success
            )
            
            return success
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Error loading data: {str(e)}")
            DataLogger.log_data_load(
                data_source=self.data_dir,
                num_records=0,
                duration_ms=duration_ms,
                success=False
            )
            return False
    
    def _extract_speakers_from_minutes(self) -> None:
        """Extract speaker information from minutes data when speakers CSV is not available."""
        logger.info("Extracting speakers from minutes data")
        
        if self.minutes_df is None:
            logger.warning("No minutes data available to extract speakers")
            return
        
        # Group by Speaker and aggregate data
        speaker_stats = self.minutes_df.groupby('Speaker').agg(
            Role=('Role', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
            Number_of_Contributions=('Content', 'count'),
            Total_Words=('Content', lambda x: sum(len(str(text).split()) for text in x)),
        ).reset_index()
        
        # Calculate average words per contribution
        speaker_stats['Average_Words_Per_Contribution'] = (
            speaker_stats['Total_Words'] / speaker_stats['Number_of_Contributions']
        )
        
        # Sort by number of contributions
        self.speakers_df = speaker_stats.sort_values('Number_of_Contributions', ascending=False)
        logger.info(f"Extracted {len(self.speakers_df)} speakers from minutes")
    
    def _preprocess_data(self) -> None:
        """Clean and preprocess loaded data."""
        if self.minutes_df is None:
            return
        
        # Clean data
        # Convert date strings to datetime objects
        if 'Date' in self.minutes_df.columns:
            self.minutes_df['Date'] = pd.to_datetime(self.minutes_df['Date'])
        
        # Fill missing timestamps with previous/next values or empty string
        if 'Timestamp' in self.minutes_df.columns:
            self.minutes_df['Timestamp'] = self.minutes_df['Timestamp'].fillna('')
        
        # Remove duplicates
        self.minutes_df = self.minutes_df.drop_duplicates().reset_index(drop=True)
        
        # Sort by date and timestamp
        if 'Date' in self.minutes_df.columns and 'Timestamp' in self.minutes_df.columns:
            self.minutes_df = self.minutes_df.sort_values(['Date', 'Timestamp']).reset_index(drop=True)
        
        # Add unique ID for each entry
        self.minutes_df['entry_id'] = self.minutes_df.index
        
        logger.info("Preprocessing completed")
    
    def get_unique_dates(self) -> List[str]:
        """Get list of unique session dates in the dataset.
        
        Returns:
            List of date strings in YYYY-MM-DD format.
        """
        if self.minutes_df is None:
            self.load_data()
            if self.minutes_df is None:
                return []
        
        if 'Date' in self.minutes_df.columns:
            dates = self.minutes_df['Date'].dt.strftime('%Y-%m-%d').unique().tolist()
            return sorted(dates)
        
        return []
    
    def get_unique_speakers(self) -> List[str]:
        """Get list of unique speakers in the dataset.
        
        Returns:
            List of speaker names.
        """
        if self.speakers_df is None:
            self.load_data()
            if self.speakers_df is None:
                return []
        
        return self.speakers_df['Speaker'].unique().tolist()
    
    def get_speaker_info(self, speaker_name: str) -> Dict[str, Any]:
        """Get detailed information about a speaker.
        
        Args:
            speaker_name: Name of the speaker.
            
        Returns:
            Dictionary of speaker information.
        """
        if self.speakers_df is None:
            self.load_data()
            if self.speakers_df is None:
                return {}
        
        speaker_row = self.speakers_df[self.speakers_df['Speaker'] == speaker_name]
        if len(speaker_row) == 0:
            return {}
        
        speaker_info = speaker_row.iloc[0].to_dict()
        
        # Add dates the speaker was present
        if self.minutes_df is not None and 'Date' in self.minutes_df.columns:
            speaker_dates = self.minutes_df[self.minutes_df['Speaker'] == speaker_name]
            if len(speaker_dates) > 0:
                dates = speaker_dates['Date'].dt.strftime('%Y-%m-%d').unique().tolist()
                speaker_info['Dates_Present'] = sorted(dates)
            else:
                speaker_info['Dates_Present'] = []
        
        return speaker_info
    
    def get_session_data(self, date_str: str) -> pd.DataFrame:
        """Get data for a specific session date.
        
        Args:
            date_str: Date string in YYYY-MM-DD format.
            
        Returns:
            DataFrame containing the session data.
        """
        if self.minutes_df is None:
            self.load_data()
            if self.minutes_df is None:
                return pd.DataFrame()
        
        try:
            date = pd.to_datetime(date_str)
            return self.minutes_df[self.minutes_df['Date'].dt.date == date.date()].copy()
        except:
            logger.error(f"Invalid date format: {date_str}")
            return pd.DataFrame()
    
    def get_speaker_contributions(self, speaker_name: str) -> pd.DataFrame:
        """Get all contributions by a specific speaker.
        
        Args:
            speaker_name: Name of the speaker.
            
        Returns:
            DataFrame containing the speaker's contributions.
        """
        if self.minutes_df is None:
            self.load_data()
            if self.minutes_df is None:
                return pd.DataFrame()
        
        return self.minutes_df[self.minutes_df['Speaker'] == speaker_name].copy()
    
    def search_content(self, search_term: str) -> pd.DataFrame:
        """Search for contributions containing a specific term.
        
        Args:
            search_term: Term to search for.
            
        Returns:
            DataFrame of contributions containing the search term.
        """
        if self.minutes_df is None:
            self.load_data()
            if self.minutes_df is None:
                return pd.DataFrame()
        
        if 'Content' not in self.minutes_df.columns:
            return pd.DataFrame()
        
        # Case-insensitive search
        mask = self.minutes_df['Content'].str.contains(search_term, case=False, na=False)
        return self.minutes_df[mask].copy()

# Usage example:
# from src.data.loader import ParliamentaryDataLoader
# 
# loader = ParliamentaryDataLoader()
# loader.load_data()
# 
# # Get unique session dates
# dates = loader.get_unique_dates()
# 
# # Get session data for a specific date
# session_data = loader.get_session_data("2024-09-10")
# 
# # Get contributions by a specific speaker
# speaker_contributions = loader.get_speaker_contributions("John Smith")
# 
# # Search for content containing a term
# climate_discussions = loader.search_content("climate change") 