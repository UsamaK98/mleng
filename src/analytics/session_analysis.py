"""
Session analytics module for parliamentary minutes
"""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SessionAnalyzer:
    """
    Analytics for parliamentary sessions
    """
    def __init__(self, data_loader=None):
        """
        Initialize session analyzer
        
        Args:
            data_loader: Data loader with minutes and speaker data
        """
        self.data_loader = data_loader
        self.minutes_df = None
        self.speakers_df = None
        
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
    
    def get_session_stats(self, date: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a specific session
        
        Args:
            date: Date of the session
            
        Returns:
            Dictionary with session statistics
        """
        if self.minutes_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check if the session exists
        if date not in self.minutes_df['Date'].unique():
            return {
                "error": f"Session on date '{date}' not found",
                "found": False
            }
        
        # Get all contributions in this session
        session_df = self.minutes_df[self.minutes_df['Date'] == date]
        
        # Basic stats
        total_contributions = len(session_df)
        unique_speakers = session_df['Speaker'].nunique()
        
        # Top speakers
        speaker_counts = session_df['Speaker'].value_counts().head(10).to_dict()
        
        # Get content length distribution
        word_counts = session_df['Content'].str.split().str.len()
        content_length_stats = {
            "mean": word_counts.mean(),
            "median": word_counts.median(),
            "min": word_counts.min(),
            "max": word_counts.max(),
            "total_words": word_counts.sum()
        }
        
        # Extract common terms
        common_terms = self._extract_common_terms(session_df)
        
        # Session flow
        session_flow = self._analyze_session_flow(session_df)
        
        return {
            "date": date,
            "total_contributions": total_contributions,
            "unique_speakers": unique_speakers,
            "top_speakers": speaker_counts,
            "content_length_stats": content_length_stats,
            "common_terms": common_terms,
            "session_flow": session_flow,
            "found": True
        }
    
    def compare_sessions(self, date1: str, date2: str) -> Dict[str, Any]:
        """
        Compare two sessions to identify similarities and differences
        
        Args:
            date1: Date of the first session
            date2: Date of the second session
            
        Returns:
            Dictionary with comparison results
        """
        if self.minutes_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get stats for both sessions
        stats1 = self.get_session_stats(date1)
        stats2 = self.get_session_stats(date2)
        
        if not stats1.get("found", False) or not stats2.get("found", False):
            return {
                "error": "One or both sessions not found",
                "found": False
            }
        
        # Get session data
        session1_df = self.minutes_df[self.minutes_df['Date'] == date1]
        session2_df = self.minutes_df[self.minutes_df['Date'] == date2]
        
        # Common speakers
        speakers1 = set(session1_df['Speaker'].unique())
        speakers2 = set(session2_df['Speaker'].unique())
        common_speakers = speakers1.intersection(speakers2)
        
        # Topic similarity
        topic_similarity = self._compute_topic_similarity(session1_df, session2_df)
        
        # Unique aspects of each session
        unique_topics1 = self._extract_unique_topics(session1_df, session2_df)
        unique_topics2 = self._extract_unique_topics(session2_df, session1_df)
        
        # Basic comparison stats
        comparison = {
            "sessions": [date1, date2],
            "contributions": [stats1["total_contributions"], stats2["total_contributions"]],
            "unique_speakers": [stats1["unique_speakers"], stats2["unique_speakers"]],
            "total_words": [
                stats1["content_length_stats"]["total_words"], 
                stats2["content_length_stats"]["total_words"]
            ],
            "common_speakers": {
                "count": len(common_speakers),
                "speakers": list(common_speakers)
            },
            "topic_similarity": topic_similarity,
            "unique_topics_session1": unique_topics1,
            "unique_topics_session2": unique_topics2,
            "found": True
        }
        
        return comparison
    
    def get_session_timeline(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get a timeline of sessions with key metrics
        
        Args:
            limit: Maximum number of sessions to return (newest first)
            
        Returns:
            List of session information
        """
        if self.minutes_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Group by date and get stats
        session_stats = []
        
        # Sort dates
        dates = sorted(self.minutes_df['Date'].unique(), reverse=True)
        
        # Limit if specified
        if limit:
            dates = dates[:limit]
        
        for date in dates:
            session_df = self.minutes_df[self.minutes_df['Date'] == date]
            
            # Basic stats
            contributions = len(session_df)
            speakers = session_df['Speaker'].nunique()
            word_count = session_df['Content'].str.split().str.len().sum()
            
            # Most active speaker
            if not session_df.empty:
                most_active = session_df['Speaker'].value_counts().idxmax()
                most_active_count = session_df['Speaker'].value_counts().max()
            else:
                most_active = "Unknown"
                most_active_count = 0
            
            session_stats.append({
                "date": date,
                "contributions": contributions,
                "unique_speakers": speakers,
                "word_count": word_count,
                "most_active_speaker": most_active,
                "most_active_count": most_active_count
            })
        
        return session_stats
    
    def _extract_common_terms(self, session_df: pd.DataFrame, limit: int = 20) -> Dict[str, int]:
        """
        Extract common terms from a session
        
        Args:
            session_df: DataFrame with session contributions
            limit: Maximum number of terms to return
            
        Returns:
            Dictionary mapping terms to their frequency
        """
        if session_df.empty:
            return {}
        
        # Combine all text
        all_text = " ".join(session_df["Content"].tolist())
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', '', all_text.lower())
        words = text.split()
        
        # Remove common words
        stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'it', 
                        'as', 'with', 'be', 'on', 'this', 'by', 'have', 'has', 'had',
                        'not', 'we', 'they', 'are', 'were', 'been', 'being', 'am', 'was'])
        
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Return most common
        return dict(word_counts.most_common(limit))
    
    def _compute_topic_similarity(self, session1_df: pd.DataFrame, session2_df: pd.DataFrame) -> float:
        """
        Compute topic similarity between two sessions
        
        Args:
            session1_df: DataFrame with first session contributions
            session2_df: DataFrame with second session contributions
            
        Returns:
            Similarity score (0-1)
        """
        if session1_df.empty or session2_df.empty or not SKLEARN_AVAILABLE:
            return 0.0
        
        # Combine all text from each session
        text1 = " ".join(session1_df["Content"].tolist())
        text2 = " ".join(session2_df["Content"].tolist())
        
        # Use TF-IDF to compute similarity
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error computing topic similarity: {e}")
            return 0.0
    
    def _extract_unique_topics(self, session1_df: pd.DataFrame, session2_df: pd.DataFrame, 
                             limit: int = 10) -> List[str]:
        """
        Extract topics that are unique to session1 compared to session2
        
        Args:
            session1_df: DataFrame with first session contributions
            session2_df: DataFrame with second session contributions
            limit: Maximum number of topics to return
            
        Returns:
            List of unique topics
        """
        if session1_df.empty:
            return []
        
        if not SKLEARN_AVAILABLE:
            # Fallback to simple word frequency comparison
            terms1 = self._extract_common_terms(session1_df, limit=50)
            terms2 = self._extract_common_terms(session2_df, limit=50)
            
            # Find terms in session1 that aren't in session2
            unique_terms = [term for term in terms1.keys() if term not in terms2]
            
            return unique_terms[:limit]
        
        # More sophisticated approach with TF-IDF
        try:
            # Combine all text from each session
            docs1 = session1_df["Content"].tolist()
            all_text2 = " ".join(session2_df["Content"].tolist())
            
            # Vectorize
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            vectorizer.fit([all_text2] + docs1)  # Include session2 in the vocabulary
            
            # Transform session1 documents
            tfidf_matrix = vectorizer.transform(docs1)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Find top terms for each document in session1
            unique_topics = set()
            
            for doc_idx in range(tfidf_matrix.shape[0]):
                doc_vector = tfidf_matrix[doc_idx]
                
                # Get non-zero elements and their indices
                non_zero = doc_vector.nonzero()[1]
                scores = zip(non_zero, [doc_vector[0, x] for x in non_zero])
                
                # Sort by score
                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
                
                # Get top terms
                for idx, score in sorted_scores[:5]:  # Top 5 terms per document
                    term = feature_names[idx]
                    if len(term) > 3:  # Filter short terms
                        unique_topics.add(term)
                
                if len(unique_topics) >= limit:
                    break
            
            return list(unique_topics)[:limit]
            
        except Exception as e:
            print(f"Error extracting unique topics: {e}")
            return []
    
    def _analyze_session_flow(self, session_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the flow of a session (speaker transitions, etc.)
        
        Args:
            session_df: DataFrame with session contributions
            
        Returns:
            Dictionary with session flow analysis
        """
        if session_df.empty:
            return {}
        
        # Sort by timestamp
        session_df = session_df.sort_values("Timestamp")
        
        # Get sequence of speakers
        speakers = session_df["Speaker"].tolist()
        
        # Count transitions
        transitions = {}
        for i in range(len(speakers) - 1):
            speaker1 = speakers[i]
            speaker2 = speakers[i + 1]
            
            # Skip self-transitions
            if speaker1 == speaker2:
                continue
            
            key = f"{speaker1} → {speaker2}"
            transitions[key] = transitions.get(key, 0) + 1
        
        # Get top transitions
        top_transitions = dict(sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Calculate speaker engagement
        engagement = {}
        for i, speaker in enumerate(speakers):
            if speaker not in engagement:
                engagement[speaker] = {
                    "contributions": 0,
                    "position": []
                }
            
            engagement[speaker]["contributions"] += 1
            engagement[speaker]["position"].append(i)
        
        # Calculate distribution of contributions
        for speaker in engagement:
            positions = engagement[speaker]["position"]
            engagement[speaker]["avg_position"] = np.mean(positions) / len(speakers) if positions else 0
            engagement[speaker]["early_contributions"] = sum(1 for p in positions if p < len(speakers) / 3)
            engagement[speaker]["middle_contributions"] = sum(1 for p in positions if len(speakers) / 3 <= p < 2 * len(speakers) / 3)
            engagement[speaker]["late_contributions"] = sum(1 for p in positions if p >= 2 * len(speakers) / 3)
            
            # Remove position list to reduce data size
            del engagement[speaker]["position"]
        
        return {
            "top_transitions": top_transitions,
            "speaker_engagement": engagement
        } 