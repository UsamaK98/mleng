"""
RAG pipeline for parliamentary minutes
"""
from typing import Dict, Any, Optional, List
import json

from src.rag.retriever import MinutesRetriever
from src.rag.llm import LLMInterface
from src.data.data_loader import MinutesDataLoader


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline
    """
    def __init__(
        self,
        retriever: Optional[MinutesRetriever] = None,
        llm: Optional[LLMInterface] = None,
        data_loader: Optional[MinutesDataLoader] = None
    ):
        """
        Initialize the RAG pipeline
        
        Args:
            retriever: Document retriever
            llm: Language model interface
            data_loader: Data loader for parliamentary minutes
        """
        self.retriever = retriever or MinutesRetriever()
        self.llm = llm or LLMInterface()
        self.data_loader = data_loader or MinutesDataLoader()
        
        # Load data if not already loaded
        try:
            self.data_loader.load_data()
        except Exception as e:
            print(f"Warning: Could not load data: {e}")
    
    def process_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        structured_output: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline
        
        Args:
            query: User query
            filters: Optional filters to apply (e.g., speaker, date)
            structured_output: Whether to return structured JSON
            
        Returns:
            Response dictionary with answer and metadata
        """
        # Retrieve relevant documents
        results = self.retriever.retrieve(query, filter_params=filters)
        
        # Format context from retrieved documents
        context = self.retriever.format_context(results)
        
        # Generate response
        if structured_output:
            response = self.llm.generate_structured_output(query, context)
        else:
            answer = self.llm.generate(query, context)
            
            # Format response
            response = {
                "answer": answer,
                "sources": [
                    {
                        "speaker": r["metadata"]["speaker"],
                        "date": r["metadata"]["date"],
                        "timestamp": r["metadata"]["timestamp"],
                        "relevance_score": r["score"]
                    }
                    for r in results
                ],
                "query": query,
                "filters": filters or {}
            }
        
        return response
    
    def process_entity_query(self, entity: str) -> Dict[str, Any]:
        """
        Process a query about a specific entity (speaker)
        
        Args:
            entity: Name of the speaker
            
        Returns:
            Structured information about the entity
        """
        # Check if entity is a known speaker
        speaker_info = self.data_loader.get_speaker_info(entity)
        
        if not speaker_info:
            # Try to find a partial match
            speakers = self.data_loader.get_speakers()
            partial_matches = [s for s in speakers if entity.lower() in s.lower()]
            
            if not partial_matches:
                return {
                    "answer": f"No information found for entity: {entity}",
                    "entity": entity,
                    "found": False
                }
            
            # Use the first partial match
            entity = partial_matches[0]
            speaker_info = self.data_loader.get_speaker_info(entity)
        
        # Get all contributions by this speaker
        contributions_df = self.data_loader.get_minutes_by_speaker(entity)
        
        # Get the dates when they spoke
        meeting_dates = sorted(contributions_df["Date"].unique().tolist())
        
        # Get total number of contributions
        total_contributions = len(contributions_df)
        
        # Structure the key contributions by date
        contributions_by_date = {}
        for date in meeting_dates:
            date_contribs = contributions_df[contributions_df["Date"] == date]
            contributions_by_date[date] = len(date_contribs)
        
        # Prepare the output
        output = {
            "entity": entity,
            "role": speaker_info.get("Role/Organization", ""),
            "found": True,
            "meeting_dates_present": meeting_dates,
            "total_contributions": total_contributions,
            "contributions_by_date": contributions_by_date,
            "speaker_stats": {
                "total_words": speaker_info.get("Total_Words", 0),
                "avg_words_per_contribution": speaker_info.get("Average_Words_Per_Contribution", 0)
            }
        }
        
        # For the first few contributions, get some examples
        if total_contributions > 0:
            sample_size = min(3, total_contributions)
            sample_contributions = contributions_df.sample(n=sample_size) if total_contributions > sample_size else contributions_df
            
            output["sample_contributions"] = []
            for _, contrib in sample_contributions.iterrows():
                output["sample_contributions"].append({
                    "date": contrib["Date"],
                    "timestamp": contrib["Timestamp"],
                    "content": contrib["Content"][:200] + "..." if len(contrib["Content"]) > 200 else contrib["Content"]
                })
        
        # Generate a summary of the entity's contributions
        if total_contributions > 0:
            # Retrieve documents about this entity for context
            filter_params = {"speaker": entity}
            results = self.retriever.retrieve(f"What are the main topics discussed by {entity}?", filter_params=filter_params)
            context = self.retriever.format_context(results)
            
            # Generate a summary
            summary_prompt = f"Provide a brief summary of {entity}'s contributions and main discussion topics in the parliamentary minutes."
            summary = self.llm.generate(summary_prompt, context)
            
            output["summary"] = summary
        
        return output
    
    def process_topic_query(self, topic: str) -> Dict[str, Any]:
        """
        Process a query about a specific topic
        
        Args:
            topic: Topic to search for
            
        Returns:
            Information about discussions of the topic
        """
        # Create a search query for the topic
        query = f"What discussions were there about {topic}?"
        
        # Retrieve relevant documents
        results = self.retriever.retrieve(query)
        
        if not results:
            return {
                "answer": f"No information found about topic: {topic}",
                "topic": topic,
                "found": False
            }
        
        # Format context
        context = self.retriever.format_context(results)
        
        # Generate a summary of the topic discussions
        summary_prompt = f"Summarize the key points of discussion about {topic} in the parliamentary minutes."
        summary = self.llm.generate(summary_prompt, context)
        
        # Extract the speakers who discussed this topic
        speakers = list(set(r["metadata"]["speaker"] for r in results))
        
        # Extract the dates when this topic was discussed
        dates = list(set(r["metadata"]["date"] for r in results))
        
        # Format response
        response = {
            "topic": topic,
            "found": True,
            "summary": summary,
            "speakers_discussing": speakers,
            "dates_discussed": sorted(dates),
            "sources": [
                {
                    "speaker": r["metadata"]["speaker"],
                    "date": r["metadata"]["date"],
                    "timestamp": r["metadata"]["timestamp"],
                    "excerpt": r["text"][:150] + "..." if len(r["text"]) > 150 else r["text"]
                }
                for r in results
            ]
        }
        
        return response
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the parliamentary minutes dataset
        
        Returns:
            Dictionary with dataset metadata
        """
        # Get unique sessions
        sessions = self.data_loader.get_sessions()
        
        # Get unique speakers
        speakers = self.data_loader.get_speakers()
        
        # Get top speakers by contribution count
        top_speakers = self.data_loader.speakers_df.nlargest(10, "Number_of_Contributions")
        top_speakers_list = []
        
        for _, row in top_speakers.iterrows():
            top_speakers_list.append({
                "name": row["Speaker"],
                "role": row["Role/Organization"],
                "contributions": row["Number_of_Contributions"],
                "total_words": row["Total_Words"]
            })
        
        return {
            "total_sessions": len(sessions),
            "session_dates": sessions,
            "total_speakers": len(speakers),
            "top_speakers": top_speakers_list,
            "data_summary": f"The dataset contains {len(sessions)} parliamentary sessions with {len(speakers)} unique speakers."
        } 