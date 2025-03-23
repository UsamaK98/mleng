"""
Sentiment analyzer module for parliamentary minutes
"""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import re

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Install with 'pip install textblob'")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not available for visualization")


class SentimentAnalyzer:
    """
    Sentiment analyzer for parliamentary minutes
    """
    def __init__(self, data_loader=None):
        """
        Initialize sentiment analyzer
        
        Args:
            data_loader: Data loader with minutes and speaker data
        """
        self.data_loader = data_loader
        self.minutes_df = None
        self.speakers_df = None
        self.sentiment_df = None
        
        # Check required packages
        if not TEXTBLOB_AVAILABLE:
            print("Warning: TextBlob package not available. Sentiment analysis will be limited.")
        
        # Load data if data_loader is provided
        if self.data_loader:
            self.load_data()
    
    def load_data(self):
        """
        Load data from the data loader
        """
        if not self.data_loader:
            raise ValueError("No data loader provided")
        
        self.minutes_df, self.speakers_df = self.data_loader.load_data()
        print(f"Loaded {len(self.minutes_df)} minutes entries and {len(self.speakers_df)} speakers")
        
        # Compute sentiment scores
        if TEXTBLOB_AVAILABLE:
            self._compute_sentiment_scores()
    
    def _compute_sentiment_scores(self):
        """
        Compute sentiment scores for all contributions
        """
        if not TEXTBLOB_AVAILABLE:
            print("TextBlob not available. Cannot compute sentiment scores.")
            return
        
        print("Computing sentiment scores for all contributions...")
        
        # Make a copy of the minutes DataFrame
        self.sentiment_df = self.minutes_df.copy()
        
        # Add sentiment columns
        self.sentiment_df['polarity'] = 0.0
        self.sentiment_df['subjectivity'] = 0.0
        
        # Process each contribution
        for idx, row in self.sentiment_df.iterrows():
            content = row['Content']
            blob = TextBlob(content)
            
            self.sentiment_df.at[idx, 'polarity'] = blob.sentiment.polarity
            self.sentiment_df.at[idx, 'subjectivity'] = blob.sentiment.subjectivity
        
        print("Sentiment analysis complete")
    
    def get_overall_sentiment_stats(self) -> Dict[str, Any]:
        """
        Get overall sentiment statistics
        
        Returns:
            Dictionary with sentiment statistics
        """
        if self.sentiment_df is None:
            if not TEXTBLOB_AVAILABLE:
                return {"error": "TextBlob not available for sentiment analysis"}
            elif self.minutes_df is not None:
                self._compute_sentiment_scores()
            else:
                return {"error": "Data not loaded"}
        
        stats = {
            "avg_polarity": self.sentiment_df['polarity'].mean(),
            "polarity_std": self.sentiment_df['polarity'].std(),
            "min_polarity": self.sentiment_df['polarity'].min(),
            "max_polarity": self.sentiment_df['polarity'].max(),
            "avg_subjectivity": self.sentiment_df['subjectivity'].mean(),
            "subjectivity_std": self.sentiment_df['subjectivity'].std(),
            "neutral_percentage": (
                (self.sentiment_df['polarity'].abs() < 0.1).sum() / len(self.sentiment_df) * 100
                if len(self.sentiment_df) > 0 else 0
            ),
            "positive_percentage": (
                (self.sentiment_df['polarity'] > 0.1).sum() / len(self.sentiment_df) * 100
                if len(self.sentiment_df) > 0 else 0
            ),
            "negative_percentage": (
                (self.sentiment_df['polarity'] < -0.1).sum() / len(self.sentiment_df) * 100
                if len(self.sentiment_df) > 0 else 0
            ),
            "total_contributions": len(self.sentiment_df)
        }
        
        return stats
    
    def get_sentiment_by_speaker(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get sentiment statistics by speaker
        
        Args:
            limit: Maximum number of speakers to include (sorted by number of contributions)
            
        Returns:
            List of speaker sentiment data
        """
        if self.sentiment_df is None:
            if not TEXTBLOB_AVAILABLE:
                return [{"error": "TextBlob not available for sentiment analysis"}]
            elif self.minutes_df is not None:
                self._compute_sentiment_scores()
            else:
                return [{"error": "Data not loaded"}]
        
        # Group by speaker
        speaker_groups = self.sentiment_df.groupby('Speaker')
        
        # Calculate stats for each speaker
        speaker_stats = []
        for speaker, group in speaker_groups:
            if len(group) < 3:  # Skip speakers with very few contributions
                continue
                
            stats = {
                "speaker": speaker,
                "role": self.speakers_df[self.speakers_df['Speaker'] == speaker]['Role/Organization'].iloc[0] 
                        if speaker in self.speakers_df['Speaker'].values else "Unknown",
                "contributions": len(group),
                "avg_polarity": group['polarity'].mean(),
                "polarity_std": group['polarity'].std(),
                "avg_subjectivity": group['subjectivity'].mean(),
                "positive_percentage": (group['polarity'] > 0.1).sum() / len(group) * 100,
                "negative_percentage": (group['polarity'] < -0.1).sum() / len(group) * 100,
                "neutral_percentage": (group['polarity'].abs() < 0.1).sum() / len(group) * 100
            }
            speaker_stats.append(stats)
        
        # Sort by number of contributions
        speaker_stats.sort(key=lambda x: x["contributions"], reverse=True)
        
        # Limit the number of speakers if requested
        if limit is not None:
            speaker_stats = speaker_stats[:limit]
        
        return speaker_stats
    
    def get_sentiment_by_session(self) -> List[Dict[str, Any]]:
        """
        Get sentiment trends across sessions
        
        Returns:
            List of session sentiment data
        """
        if self.sentiment_df is None:
            if not TEXTBLOB_AVAILABLE:
                return [{"error": "TextBlob not available for sentiment analysis"}]
            elif self.minutes_df is not None:
                self._compute_sentiment_scores()
            else:
                return [{"error": "Data not loaded"}]
        
        # Group by date
        session_groups = self.sentiment_df.groupby('Date')
        
        # Calculate stats for each session
        session_stats = []
        for date, group in session_groups:
            stats = {
                "date": date,
                "contributions": len(group),
                "unique_speakers": group['Speaker'].nunique(),
                "avg_polarity": group['polarity'].mean(),
                "polarity_std": group['polarity'].std(),
                "avg_subjectivity": group['subjectivity'].mean(),
                "positive_percentage": (group['polarity'] > 0.1).sum() / len(group) * 100,
                "negative_percentage": (group['polarity'] < -0.1).sum() / len(group) * 100,
                "neutral_percentage": (group['polarity'].abs() < 0.1).sum() / len(group) * 100,
                "most_positive_speaker": None,
                "most_negative_speaker": None
            }
            
            # Add most positive and negative speakers
            if group['Speaker'].nunique() > 1:
                # Group by speaker within the session
                speaker_sentiment = group.groupby('Speaker')['polarity'].mean()
                
                if not speaker_sentiment.empty:
                    # Most positive speaker
                    most_positive_idx = speaker_sentiment.idxmax()
                    most_positive_val = speaker_sentiment.max()
                    
                    # Most negative speaker
                    most_negative_idx = speaker_sentiment.idxmin()
                    most_negative_val = speaker_sentiment.min()
                    
                    stats["most_positive_speaker"] = {
                        "name": most_positive_idx,
                        "polarity": most_positive_val
                    }
                    
                    stats["most_negative_speaker"] = {
                        "name": most_negative_idx,
                        "polarity": most_negative_val
                    }
            
            session_stats.append(stats)
        
        # Sort by date
        session_stats.sort(key=lambda x: x["date"])
        
        return session_stats
    
    def find_emotional_outliers(self, threshold: float = 2.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find emotional outliers (very positive or negative contributions)
        
        Args:
            threshold: Standard deviation threshold for outliers
            
        Returns:
            Dictionary with positive and negative outliers
        """
        if self.sentiment_df is None:
            if not TEXTBLOB_AVAILABLE:
                return {"error": "TextBlob not available for sentiment analysis"}
            elif self.minutes_df is not None:
                self._compute_sentiment_scores()
            else:
                return {"error": "Data not loaded"}
        
        # Calculate mean and standard deviation
        mean_polarity = self.sentiment_df['polarity'].mean()
        std_polarity = self.sentiment_df['polarity'].std()
        
        # Define thresholds for outliers
        positive_threshold = mean_polarity + (threshold * std_polarity)
        negative_threshold = mean_polarity - (threshold * std_polarity)
        
        # Find positive outliers
        positive_outliers = self.sentiment_df[self.sentiment_df['polarity'] > positive_threshold]
        
        # Find negative outliers
        negative_outliers = self.sentiment_df[self.sentiment_df['polarity'] < negative_threshold]
        
        # Format results
        positive_results = []
        for _, row in positive_outliers.iterrows():
            positive_results.append({
                "speaker": row['Speaker'],
                "date": row['Date'],
                "polarity": row['polarity'],
                "subjectivity": row['subjectivity'],
                "content_preview": row['Content'][:150] + "..." if len(row['Content']) > 150 else row['Content']
            })
        
        negative_results = []
        for _, row in negative_outliers.iterrows():
            negative_results.append({
                "speaker": row['Speaker'],
                "date": row['Date'],
                "polarity": row['polarity'],
                "subjectivity": row['subjectivity'],
                "content_preview": row['Content'][:150] + "..." if len(row['Content']) > 150 else row['Content']
            })
        
        return {
            "positive_outliers": positive_results,
            "negative_outliers": negative_results,
            "thresholds": {
                "mean_polarity": mean_polarity,
                "std_polarity": std_polarity,
                "positive_threshold": positive_threshold,
                "negative_threshold": negative_threshold
            }
        }
    
    def compare_sentiment_between_roles(self) -> Dict[str, Any]:
        """
        Compare sentiment between different speaker roles
        
        Returns:
            Dictionary with sentiment comparison by role
        """
        if self.sentiment_df is None or self.speakers_df is None:
            if not TEXTBLOB_AVAILABLE:
                return {"error": "TextBlob not available for sentiment analysis"}
            elif self.minutes_df is not None:
                self._compute_sentiment_scores()
            else:
                return {"error": "Data not loaded"}
        
        # Merge minutes with speaker info to get roles
        merged_df = pd.merge(
            self.sentiment_df,
            self.speakers_df[['Speaker', 'Role/Organization']],
            on='Speaker',
            how='left'
        )
        
        # Group by role
        role_groups = merged_df.groupby('Role/Organization')
        
        # Calculate stats for each role
        role_stats = []
        for role, group in role_groups:
            if role is None or pd.isna(role) or role == "":
                role = "Unknown"
                
            if len(group) < 3:  # Skip roles with very few contributions
                continue
                
            stats = {
                "role": role,
                "speakers": group['Speaker'].nunique(),
                "contributions": len(group),
                "avg_polarity": group['polarity'].mean(),
                "polarity_std": group['polarity'].std(),
                "avg_subjectivity": group['subjectivity'].mean(),
                "positive_percentage": (group['polarity'] > 0.1).sum() / len(group) * 100,
                "negative_percentage": (group['polarity'] < -0.1).sum() / len(group) * 100,
                "neutral_percentage": (group['polarity'].abs() < 0.1).sum() / len(group) * 100
            }
            role_stats.append(stats)
        
        # Sort by number of contributions
        role_stats.sort(key=lambda x: x["contributions"], reverse=True)
        
        return {
            "role_stats": role_stats,
            "total_roles": len(role_stats)
        }
    
    def get_topic_sentiment_keywords(self, num_keywords: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify keywords associated with positive and negative sentiment
        
        Args:
            num_keywords: Number of keywords to return for each sentiment
            
        Returns:
            Dictionary with positive and negative keywords
        """
        if self.sentiment_df is None:
            if not TEXTBLOB_AVAILABLE:
                return {"error": "TextBlob not available for sentiment analysis"}
            elif self.minutes_df is not None:
                self._compute_sentiment_scores()
            else:
                return {"error": "Data not loaded"}
        
        # Define positive and negative content
        positive_content = self.sentiment_df[self.sentiment_df['polarity'] > 0.2]['Content']
        negative_content = self.sentiment_df[self.sentiment_df['polarity'] < -0.2]['Content']
        
        # Extract keywords from positive content
        positive_keywords = self._extract_keywords(positive_content, num_keywords)
        
        # Extract keywords from negative content
        negative_keywords = self._extract_keywords(negative_content, num_keywords)
        
        return {
            "positive_keywords": positive_keywords,
            "negative_keywords": negative_keywords
        }
    
    def _extract_keywords(self, content_series: pd.Series, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Extract keywords from a series of content
        
        Args:
            content_series: Series of text content
            limit: Maximum number of keywords to return
            
        Returns:
            List of keyword data
        """
        if content_series.empty:
            return []
            
        # Combine all text
        all_text = " ".join(content_series.tolist())
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', '', all_text.lower())
        words = text.split()
        
        # Remove common words
        stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'it', 
                        'as', 'with', 'be', 'on', 'this', 'by', 'have', 'has', 'had',
                        'not', 'we', 'they', 'are', 'were', 'been', 'being', 'am', 'was',
                        'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might',
                        'must', 'our', 'their', 'your', 'my', 'his', 'her', 'its', 'there'])
        
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Return most common
        keywords = []
        for word, count in word_counts.most_common(limit):
            keywords.append({
                "word": word,
                "count": count,
                "frequency_percentage": count / len(filtered_words) * 100 if filtered_words else 0
            })
        
        return keywords
    
    def generate_sentiment_report(self, output_file: str = "sentiment_report.json") -> Dict[str, Any]:
        """
        Generate a comprehensive sentiment report
        
        Args:
            output_file: Path to save the JSON report
            
        Returns:
            Dictionary with comprehensive sentiment analysis
        """
        # Gather all sentiment data
        report = {
            "overall_stats": self.get_overall_sentiment_stats(),
            "sentiment_by_speaker": self.get_sentiment_by_speaker(limit=10),
            "sentiment_by_session": self.get_sentiment_by_session(),
            "emotional_outliers": self.find_emotional_outliers(),
            "sentiment_by_role": self.compare_sentiment_between_roles(),
            "sentiment_keywords": self.get_topic_sentiment_keywords()
        }
        
        # Save to JSON file
        try:
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Sentiment report saved to {output_file}")
        except Exception as e:
            print(f"Error saving sentiment report: {e}")
        
        return report 