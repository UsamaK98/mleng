"""
API routes for analytics features
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.data.data_loader import MinutesDataLoader
from src.analytics.speaker_analysis import SpeakerAnalytics
from src.analytics.session_analysis import SessionAnalyzer
from src.analytics.relationship_mapper import RelationshipMapper
from src.analytics.sentiment_analyzer import SentimentAnalyzer

# Create router
router = APIRouter(prefix="/analytics", tags=["analytics"])

# Models for response
class AnalyticsResponse(BaseModel):
    """Base response model for analytics endpoints"""
    status: str = "success"
    data: Dict[str, Any] = {}
    message: Optional[str] = None


# Dependency for data loader
def get_data_loader():
    """Get data loader instance"""
    return MinutesDataLoader()


# Speaker analytics routes
@router.get("/speakers", response_model=AnalyticsResponse)
async def get_top_speakers(
    limit: int = Query(10, description="Number of top speakers to return"),
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get top speakers by contribution count"""
    try:
        analytics = SpeakerAnalytics(data_loader)
        top_speakers = analytics.get_top_speakers(limit)
        return {
            "status": "success",
            "data": {
                "top_speakers": top_speakers
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speakers/{speaker_name}", response_model=AnalyticsResponse)
async def get_speaker_stats(
    speaker_name: str,
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get statistics for a specific speaker"""
    try:
        analytics = SpeakerAnalytics(data_loader)
        stats = analytics.get_speaker_stats(speaker_name)
        if not stats:
            raise HTTPException(status_code=404, detail=f"Speaker '{speaker_name}' not found")
        return {
            "status": "success",
            "data": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speakers/compare", response_model=AnalyticsResponse)
async def compare_speakers(
    speaker1: str = Query(..., description="First speaker name"),
    speaker2: str = Query(..., description="Second speaker name"),
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Compare two speakers"""
    try:
        analytics = SpeakerAnalytics(data_loader)
        comparison = analytics.compare_speakers(speaker1, speaker2)
        if not comparison:
            raise HTTPException(status_code=404, detail=f"One or both speakers not found")
        return {
            "status": "success",
            "data": comparison
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Session analytics routes
@router.get("/sessions", response_model=AnalyticsResponse)
async def get_session_timeline(
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get timeline of sessions with key metrics"""
    try:
        analyzer = SessionAnalyzer(data_loader)
        timeline = analyzer.get_session_timeline()
        return {
            "status": "success",
            "data": {
                "timeline": timeline
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_date}", response_model=AnalyticsResponse)
async def get_session_stats(
    session_date: str,
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get statistics for a specific session by date"""
    try:
        analyzer = SessionAnalyzer(data_loader)
        stats = analyzer.get_session_stats(session_date)
        if not stats:
            raise HTTPException(status_code=404, detail=f"Session on date '{session_date}' not found")
        return {
            "status": "success",
            "data": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/compare", response_model=AnalyticsResponse)
async def compare_sessions(
    session1: str = Query(..., description="First session date"),
    session2: str = Query(..., description="Second session date"),
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Compare two sessions"""
    try:
        analyzer = SessionAnalyzer(data_loader)
        comparison = analyzer.compare_sessions(session1, session2)
        if not comparison:
            raise HTTPException(status_code=404, detail=f"One or both sessions not found")
        return {
            "status": "success",
            "data": comparison
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Relationship analytics routes
@router.get("/relationships/network", response_model=AnalyticsResponse)
async def get_interaction_network(
    min_interactions: int = Query(2, description="Minimum number of interactions to include"),
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get speaker interaction network"""
    try:
        mapper = RelationshipMapper(data_loader)
        network = mapper.get_interaction_network(min_weight=min_interactions)
        return {
            "status": "success",
            "data": network
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships/influencers", response_model=AnalyticsResponse)
async def get_key_influencers(
    limit: int = Query(5, description="Number of influencers to return"),
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get key influencers in the speaker network"""
    try:
        mapper = RelationshipMapper(data_loader)
        influencers = mapper.get_key_influencers(limit=limit)
        return {
            "status": "success",
            "data": {
                "influencers": influencers
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships/communities", response_model=AnalyticsResponse)
async def get_speaker_communities(
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get communities of speakers"""
    try:
        mapper = RelationshipMapper(data_loader)
        communities = mapper.get_speaker_communities()
        return {
            "status": "success",
            "data": communities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Sentiment analysis routes
@router.get("/sentiment/overall", response_model=AnalyticsResponse)
async def get_overall_sentiment(
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get overall sentiment statistics"""
    try:
        analyzer = SentimentAnalyzer(data_loader)
        stats = analyzer.get_overall_sentiment_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/by-speaker", response_model=AnalyticsResponse)
async def get_sentiment_by_speaker(
    limit: int = Query(10, description="Number of speakers to include"),
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get sentiment statistics by speaker"""
    try:
        analyzer = SentimentAnalyzer(data_loader)
        stats = analyzer.get_sentiment_by_speaker(limit=limit)
        return {
            "status": "success",
            "data": {
                "speakers": stats
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/by-session", response_model=AnalyticsResponse)
async def get_sentiment_by_session(
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get sentiment trends across sessions"""
    try:
        analyzer = SentimentAnalyzer(data_loader)
        stats = analyzer.get_sentiment_by_session()
        return {
            "status": "success",
            "data": {
                "sessions": stats
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/outliers", response_model=AnalyticsResponse)
async def get_sentiment_outliers(
    threshold: float = Query(2.0, description="Standard deviation threshold for outliers"),
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Find emotional outliers in contributions"""
    try:
        analyzer = SentimentAnalyzer(data_loader)
        outliers = analyzer.find_emotional_outliers(threshold=threshold)
        return {
            "status": "success",
            "data": outliers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/by-role", response_model=AnalyticsResponse)
async def get_sentiment_by_role(
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Compare sentiment between different speaker roles"""
    try:
        analyzer = SentimentAnalyzer(data_loader)
        comparison = analyzer.compare_sentiment_between_roles()
        return {
            "status": "success",
            "data": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/keywords", response_model=AnalyticsResponse)
async def get_sentiment_keywords(
    num_keywords: int = Query(20, description="Number of keywords to return for each sentiment"),
    data_loader: MinutesDataLoader = Depends(get_data_loader)
):
    """Get keywords associated with positive and negative sentiment"""
    try:
        analyzer = SentimentAnalyzer(data_loader)
        keywords = analyzer.get_topic_sentiment_keywords(num_keywords=num_keywords)
        return {
            "status": "success",
            "data": keywords
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 