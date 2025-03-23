"""
Speaker analytics module for parliamentary minutes
"""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import json
from datetime import datetime

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class SpeakerAnalytics:
    """
    Analytics for parliamentary speakers
    """
    def __init__(self, data_loader=None):
        """
        Initialize speaker analytics
        
        Args:
            data_loader: Data loader with minutes and speaker data
        """
        self.data_loader = data_loader
        self.minutes_df = None
        self.speakers_df = None
        
        # NLP components
        self.nlp = None
        
        # Initialize NLP if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("Loaded spaCy model for NLP analysis")
            except Exception as e:
                print(f"Could not load spaCy model: {e}")
        
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
    
    def get_speaker_stats(self, speaker_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a specific speaker
        
        Args:
            speaker_name: Name of the speaker
            
        Returns:
            Dictionary with speaker statistics
        """
        if self.minutes_df is None or self.speakers_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Try to find the speaker
        if speaker_name not in self.speakers_df['Speaker'].values:
            close_matches = [
                s for s in self.speakers_df['Speaker'].values 
                if speaker_name.lower() in s.lower()
            ]
            
            if not close_matches:
                return {
                    "error": f"Speaker '{speaker_name}' not found",
                    "found": False
                }
            
            # Use the first close match
            speaker_name = close_matches[0]
        
        # Get all contributions by this speaker
        speaker_contributions = self.minutes_df[
            self.minutes_df['Speaker'] == speaker_name
        ]
        
        # Get speaker info
        speaker_info = self.speakers_df[
            self.speakers_df['Speaker'] == speaker_name
        ].iloc[0].to_dict() if len(self.speakers_df[self.speakers_df['Speaker'] == speaker_name]) > 0 else {}
        
        # Basic stats
        total_contributions = len(speaker_contributions)
        total_words = speaker_contributions['Content'].str.split().str.len().sum() if total_contributions > 0 else 0
        avg_words = total_words / total_contributions if total_contributions > 0 else 0
        
        # Session participation
        sessions = speaker_contributions['Date'].unique().tolist() if total_contributions > 0 else []
        
        # Most frequent topics (using keyword extraction)
        topics = {}
        topic_counts = self._extract_speaker_topics(speaker_contributions) if total_contributions > 0 else {}
        
        # Sort topics by frequency
        topics = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Sentiment analysis
        sentiment = {}
        if TEXTBLOB_AVAILABLE and total_contributions > 0:
            sentiment = self._analyze_sentiment(speaker_contributions)
        
        # Interaction pattern (who they respond to or who responds to them)
        interactions = self._analyze_interactions(speaker_name) if total_contributions > 0 else {}
        
        # Language complexity
        language_metrics = self._analyze_language_complexity(speaker_contributions) if total_contributions > 0 else {}
        
        # Time trends
        time_trends = self._analyze_time_trends(speaker_contributions) if total_contributions > 0 else {}
        
        return {
            "name": speaker_name,
            "role": speaker_info.get("Role/Organization", "Unknown"),
            "total_contributions": total_contributions,
            "total_words": total_words,
            "avg_words_per_contribution": avg_words,
            "sessions_attended": sessions,
            "frequent_topics": topics,
            "interactions": interactions,
            "sentiment": sentiment,
            "language_metrics": language_metrics,
            "time_trends": time_trends,
            "found": True
        }
    
    def compare_speakers(self, speaker1: str, speaker2: str) -> Dict[str, Any]:
        """
        Compare two speakers based on their contributions
        
        Args:
            speaker1: First speaker name
            speaker2: Second speaker name
            
        Returns:
            Dictionary with comparison results
        """
        if self.minutes_df is None or self.speakers_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get stats for both speakers
        stats1 = self.get_speaker_stats(speaker1)
        stats2 = self.get_speaker_stats(speaker2)
        
        if not stats1.get("found", False) or not stats2.get("found", False):
            return {
                "error": "One or both speakers not found",
                "found": False
            }
        
        # Compare basic stats
        comparison = {
            "speakers": [speaker1, speaker2],
            "contributions": [stats1["total_contributions"], stats2["total_contributions"]],
            "total_words": [stats1["total_words"], stats2["total_words"]],
            "avg_words": [stats1["avg_words_per_contribution"], stats2["avg_words_per_contribution"]],
            "sessions": {
                "common": list(set(stats1["sessions_attended"]) & set(stats2["sessions_attended"])),
                "speaker1_only": list(set(stats1["sessions_attended"]) - set(stats2["sessions_attended"])),
                "speaker2_only": list(set(stats2["sessions_attended"]) - set(stats1["sessions_attended"])),
            },
            "topics": {
                "common": list(set(stats1["frequent_topics"].keys()) & set(stats2["frequent_topics"].keys())),
                "speaker1_only": list(set(stats1["frequent_topics"].keys()) - set(stats2["frequent_topics"].keys())),
                "speaker2_only": list(set(stats2["frequent_topics"].keys()) - set(stats1["frequent_topics"].keys())),
            },
            "found": True
        }
        
        # Check for direct interactions
        interactions1 = stats1.get("interactions", {}).get("responds_to", {})
        interactions2 = stats2.get("interactions", {}).get("responds_to", {})
        
        direct_interactions = {
            "speaker1_to_speaker2": interactions1.get(speaker2, 0),
            "speaker2_to_speaker1": interactions2.get(speaker1, 0),
            "total": interactions1.get(speaker2, 0) + interactions2.get(speaker1, 0)
        }
        
        comparison["direct_interactions"] = direct_interactions
        
        # Compare sentiment if available
        if "sentiment" in stats1 and "sentiment" in stats2:
            comparison["sentiment"] = {
                "polarity": [stats1["sentiment"].get("avg_polarity", 0), stats2["sentiment"].get("avg_polarity", 0)],
                "subjectivity": [stats1["sentiment"].get("avg_subjectivity", 0), stats2["sentiment"].get("avg_subjectivity", 0)]
            }
        
        return comparison
    
    def get_top_speakers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top speakers by contribution count
        
        Args:
            limit: Maximum number of speakers to return
            
        Returns:
            List of top speakers with their stats
        """
        if self.minutes_df is None or self.speakers_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get top speakers by contribution count
        top_speakers = self.speakers_df.nlargest(limit, "Number_of_Contributions")
        
        results = []
        for _, row in top_speakers.iterrows():
            results.append({
                "name": row["Speaker"],
                "role": row["Role/Organization"],
                "contributions": row["Number_of_Contributions"],
                "total_words": row["Total_Words"],
                "avg_words_per_contribution": row["Average_Words_Per_Contribution"]
            })
        
        return results
    
    def _extract_speaker_topics(self, contributions_df: pd.DataFrame) -> Dict[str, int]:
        """
        Extract topics from a speaker's contributions
        
        Args:
            contributions_df: DataFrame with the speaker's contributions
            
        Returns:
            Dictionary mapping topics to their frequency
        """
        if contributions_df.empty:
            return {}
        
        topic_counts = Counter()
        
        # Combine all text for analysis
        all_text = " ".join(contributions_df["Content"].tolist())
        
        if self.nlp:
            # Use spaCy for named entity recognition and noun chunk extraction
            doc = self.nlp(all_text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ["ORG", "GPE", "LAW", "PERSON"]:
                    topic_counts[ent.text] += 1
            
            # Extract noun chunks (potential topics)
            for chunk in doc.noun_chunks:
                # Filter out single words and common words
                if len(chunk.text.split()) > 1 and len(chunk.text) > 5:
                    topic_counts[chunk.text] += 1
        else:
            # Fallback to simple keyword extraction
            # Remove punctuation and lowercase
            text = re.sub(r'[^\w\s]', '', all_text.lower())
            words = text.split()
            
            # Remove common words
            stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'it', 'as', 'with', 'be', 'on', 'this', 'by'])
            filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
            
            # Count word frequencies
            topic_counts.update(filtered_words)
        
        return dict(topic_counts.most_common(20))
    
    def _analyze_sentiment(self, contributions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sentiment in a speaker's contributions
        
        Args:
            contributions_df: DataFrame with the speaker's contributions
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not TEXTBLOB_AVAILABLE or contributions_df.empty:
            return {}
        
        # Calculate sentiment for each contribution
        polarities = []
        subjectivities = []
        
        for _, row in contributions_df.iterrows():
            text = row["Content"]
            blob = TextBlob(text)
            polarities.append(blob.sentiment.polarity)
            subjectivities.append(blob.sentiment.subjectivity)
        
        return {
            "avg_polarity": np.mean(polarities) if polarities else 0,
            "avg_subjectivity": np.mean(subjectivities) if subjectivities else 0,
            "polarity_std": np.std(polarities) if polarities else 0,
            "polarity_min": min(polarities) if polarities else 0,
            "polarity_max": max(polarities) if polarities else 0,
            "subjectivity_std": np.std(subjectivities) if subjectivities else 0
        }
    
    def _analyze_interactions(self, speaker_name: str) -> Dict[str, Any]:
        """
        Analyze interactions between speakers
        
        Args:
            speaker_name: Name of the speaker
            
        Returns:
            Dictionary with interaction analysis results
        """
        if self.minutes_df is None or self.minutes_df.empty:
            return {}
        
        # Get all sessions where the speaker participated
        speaker_sessions = self.minutes_df[self.minutes_df["Speaker"] == speaker_name]["Date"].unique()
        
        # Initialize counters
        responds_to = Counter()
        responded_by = Counter()
        
        # Analyze each session separately
        for session in speaker_sessions:
            session_df = self.minutes_df[self.minutes_df["Date"] == session].sort_values("Timestamp")
            
            # Find interactions
            speakers = session_df["Speaker"].tolist()
            indices = [i for i, s in enumerate(speakers) if s == speaker_name]
            
            for idx in indices:
                # Check who spoke before (speaker responds to)
                if idx > 0:
                    responds_to[speakers[idx-1]] += 1
                
                # Check who spoke after (responded by)
                if idx < len(speakers) - 1:
                    responded_by[speakers[idx+1]] += 1
        
        # Filter out self-responses
        if speaker_name in responds_to:
            del responds_to[speaker_name]
        if speaker_name in responded_by:
            del responded_by[speaker_name]
        
        return {
            "responds_to": dict(responds_to.most_common(5)),
            "responded_by": dict(responded_by.most_common(5)),
            "total_interactions": sum(responds_to.values()) + sum(responded_by.values())
        }
    
    def _analyze_language_complexity(self, contributions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze language complexity in a speaker's contributions
        
        Args:
            contributions_df: DataFrame with the speaker's contributions
            
        Returns:
            Dictionary with language complexity metrics
        """
        if contributions_df.empty:
            return {}
        
        # Calculate average sentence length
        sentence_lengths = []
        for content in contributions_df["Content"]:
            # Simple sentence splitting by punctuation
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for sentence in sentences:
                words = re.findall(r'\b\w+\b', sentence)
                if words:
                    sentence_lengths.append(len(words))
        
        # Calculate word length
        all_words = []
        for content in contributions_df["Content"]:
            words = re.findall(r'\b\w+\b', content.lower())
            all_words.extend(words)
        
        word_lengths = [len(word) for word in all_words if word]
        
        # Calculate vocabulary size (unique words)
        vocabulary_size = len(set(all_words))
        
        return {
            "avg_sentence_length": np.mean(sentence_lengths) if sentence_lengths else 0,
            "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
            "avg_word_length": np.mean(word_lengths) if word_lengths else 0,
            "vocabulary_size": vocabulary_size,
            "lexical_diversity": vocabulary_size / len(all_words) if all_words else 0
        }
    
    def _analyze_time_trends(self, contributions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze time trends in a speaker's contributions
        
        Args:
            contributions_df: DataFrame with the speaker's contributions
            
        Returns:
            Dictionary with time trend analysis
        """
        if contributions_df.empty:
            return {}
        
        # Group by date and count contributions
        contributions_by_date = contributions_df.groupby("Date").size().to_dict()
        
        # Calculate average words per contribution per date
        words_by_date = {}
        for date, group in contributions_df.groupby("Date"):
            words_by_date[date] = group["Content"].str.split().str.len().mean()
        
        # Sort dates
        sorted_dates = sorted(contributions_by_date.keys())
        
        # Format for time series visualization
        time_series = []
        for date in sorted_dates:
            time_series.append({
                "date": date,
                "contributions": contributions_by_date[date],
                "avg_words": words_by_date.get(date, 0)
            })
        
        return {
            "time_series": time_series,
            "first_contribution": sorted_dates[0] if sorted_dates else None,
            "last_contribution": sorted_dates[-1] if sorted_dates else None,
            "total_contribution_days": len(sorted_dates)
        } 