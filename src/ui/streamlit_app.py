"""
Streamlit UI for the Parliamentary Minutes Agentic Chatbot - Simplified UI
"""
import os
import sys
import json
import requests
import streamlit as st

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import API_HOST, API_PORT, UI_THEME
from src.ui.components.landing_page import render_landing_page, render_quick_search
from src.ui.components.speaker_view import render_speaker_view

# Set page configuration
st.set_page_config(
    page_title="Parliamentary Minutes Explorer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Parliamentary Minutes Explorer\nInteractive exploration of Scottish Parliament meeting minutes."
    }
)

# Define API base URL
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"


# Helper functions
def fetch_metadata():
    """Fetch metadata about the parliamentary dataset"""
    try:
        response = requests.get(f"{API_BASE_URL}/metadata")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching metadata: {e}")
        return None


def query_api(endpoint, payload):
    """Query the API with the given endpoint and payload"""
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying API: {e}")
        return None


# UI elements
def render_sidebar():
    """Render the sidebar with navigation and filters"""
    st.sidebar.title("🏛️ Navigation")
    
    # Add navigation buttons
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
    
    if st.sidebar.button("🧑‍💼 Speakers", use_container_width=True):
        st.session_state.page = "speakers"
    
    if st.sidebar.button("📋 Topics", use_container_width=True):
        st.session_state.page = "topics"
    
    if st.sidebar.button("📅 Sessions", use_container_width=True):
        st.session_state.page = "sessions"
    
    st.sidebar.markdown("---")
    
    # Display metadata
    metadata = fetch_metadata()
    
    if metadata:
        # Quick search
        query = st.sidebar.text_input("Quick Search:", key="sidebar_search")
        if st.sidebar.button("Search", key="sidebar_search_btn"):
            if query:
                st.session_state.search_query = query
                st.session_state.page = "search_results"
        
        st.sidebar.markdown("---")
        
        st.sidebar.header("Dataset Info")
        st.sidebar.write(f"Total Sessions: {metadata['total_sessions']}")
        st.sidebar.write(f"Total Speakers: {metadata['total_speakers']}")
        
        # Display top speakers in sidebar
        with st.sidebar.expander("Top Speakers"):
            for speaker in metadata.get('top_speakers', [])[:5]:
                cols = st.sidebar.columns([3, 1])
                cols[0].write(f"{speaker['name']}")
                if cols[1].button("View", key=f"sidebar_speaker_{speaker['name']}"):
                    st.session_state.selected_speaker = speaker['name']
                    st.session_state.page = "speaker_detail"


def render_search_results(query):
    """Render search results for a given query"""
    st.title("🔍 Search Results")
    st.subheader(f"Results for: \"{query}\"")
    
    with st.spinner("Searching..."):
        # Try search as entity query first
        entity_response = query_api("entity", {"entity": query})
        
        if entity_response and entity_response.get("found", False):
            st.success(f"Found information for speaker: {query}")
            
            if st.button("View Speaker Details"):
                st.session_state.selected_speaker = query
                st.session_state.page = "speaker_detail"
                st.experimental_rerun()
            
            # Show a brief summary
            st.write("### Summary")
            st.write(entity_response.get("summary", "No summary available."))
        else:
            # Try as a regular query
            response = query_api("query", {
                "query": query,
                "filters": {},
                "structured_output": False
            })
            
            if response and "answer" in response:
                st.write(response["answer"])
                
                # Display sources if available
                if "sources" in response and response["sources"]:
                    with st.expander("Sources"):
                        for source in response["sources"]:
                            st.write(f"**Speaker:** {source['speaker']}  \n"
                                     f"**Date:** {source['date']}  \n"
                                     f"**Time:** {source['timestamp']}  \n"
                                     f"**Relevance:** {source['relevance_score']:.2f}")
            else:
                st.warning("No results found for your search query.")
                st.info("Try searching for a specific speaker, topic, or session date.")


def render_topic_view():
    """Render the topic view page"""
    st.title("📋 Topics")
    st.write("Explore parliamentary discussions by topic")
    
    with st.spinner("Loading topics..."):
        # Get topics from API
        response = requests.get(f"{API_BASE_URL}/topics")
        
        if response.status_code == 200:
            topics = response.json().get("topics", [])
            
            if topics:
                # Display topic grid
                cols = st.columns(3)
                
                for i, topic in enumerate(topics):
                    col_idx = i % 3
                    with cols[col_idx]:
                        with st.container():
                            st.markdown(f"### {topic['name']}")
                            st.write(topic['description'][:100] + "..." if len(topic['description']) > 100 else topic['description'])
                            
                            if st.button("Explore", key=f"topic_{i}"):
                                st.session_state.selected_topic = topic['name']
                                st.session_state.page = "topic_detail"
                                st.experimental_rerun()
            else:
                st.info("No topics available")
        else:
            st.error("Failed to load topics")


def render_session_view():
    """Render the session view page"""
    st.title("📅 Parliamentary Sessions")
    st.write("Browse and explore parliamentary sessions")
    
    metadata = fetch_metadata()
    
    if metadata and "session_dates" in metadata:
        # Display session dates in reverse chronological order
        sessions = metadata["session_dates"]
        sessions.sort(reverse=True)  # Sort newest first
        
        # Allow filtering by year
        years = sorted(set(s.split("-")[0] for s in sessions), reverse=True)
        
        selected_year = st.selectbox("Filter by year:", ["All"] + years)
        
        # Filter sessions by selected year
        if selected_year != "All":
            filtered_sessions = [s for s in sessions if s.startswith(selected_year)]
        else:
            filtered_sessions = sessions
        
        # Display sessions in a grid
        if filtered_sessions:
            cols = st.columns(3)
            
            for i, session_date in enumerate(filtered_sessions):
                col_idx = i % 3
                with cols[col_idx]:
                    with st.container():
                        st.markdown(f"### {session_date}")
                        
                        if st.button("View Details", key=f"session_{i}"):
                            st.session_state.selected_session = session_date
                            st.session_state.page = "session_detail"
                            st.experimental_rerun()
        else:
            st.info(f"No sessions found for {selected_year}")
    else:
        st.error("Failed to load session data")


def render_session_detail(session_date):
    """Render details for a specific parliamentary session"""
    st.title(f"📅 Session: {session_date}")
    
    with st.spinner("Loading session details..."):
        # Query API for session details
        response = query_api("session", {"date": session_date})
        
        if response and response.get("found", False):
            # Display session information
            st.write("### Session Overview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if "summary" in response:
                    st.write(response["summary"])
                
                # Topics discussed
                if "topics_discussed" in response and response["topics_discussed"]:
                    st.write("### Topics Discussed")
                    for topic, weight in response["topics_discussed"].items():
                        st.write(f"- **{topic}**: {weight:.2f}")
            
            with col2:
                # Session statistics
                st.write("### Statistics")
                
                if "statistics" in response:
                    stats = response["statistics"]
                    st.write(f"Total Contributions: {stats.get('total_contributions', 0)}")
                    st.write(f"Unique Speakers: {stats.get('unique_speakers', 0)}")
                    
                    # Top speakers in session
                    if "top_speakers" in stats and stats["top_speakers"]:
                        st.write("### Top Speakers")
                        for speaker in stats["top_speakers"][:3]:
                            st.write(f"- **{speaker['name']}**: {speaker['contributions']} contributions")
            
            # Session timeline
            st.write("### Session Timeline")
            
            if "timeline" in response and response["timeline"]:
                for entry in response["timeline"]:
                    with st.expander(f"{entry['timestamp']} - {entry['speaker']}"):
                        st.write(entry["content"])
            else:
                st.info("No timeline available for this session")
        else:
            st.warning(f"No details found for session: {session_date}")


def render_topic_detail(topic_name):
    """Render details for a specific topic"""
    st.title(f"📋 Topic: {topic_name}")
    
    with st.spinner("Loading topic details..."):
        # Query API for topic details
        response = query_api("topic", {"topic": topic_name})
        
        if response and response.get("found", False):
            # Display topic information
            st.write("### Topic Overview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if "description" in response:
                    st.write(response["description"])
                
                # Related topics
                if "related_topics" in response and response["related_topics"]:
                    st.write("### Related Topics")
                    for topic, similarity in response["related_topics"].items():
                        st.write(f"- **{topic}**: {similarity:.2f}")
            
            with col2:
                # Topic statistics
                st.write("### Statistics")
                
                if "statistics" in response:
                    stats = response["statistics"]
                    st.write(f"Total Mentions: {stats.get('total_mentions', 0)}")
                    st.write(f"Sessions Discussed: {stats.get('sessions_mentioned', 0)}")
                    
                    # Top speakers for topic
                    if "top_speakers" in stats and stats["top_speakers"]:
                        st.write("### Key Speakers")
                        for speaker in stats["top_speakers"][:3]:
                            st.write(f"- **{speaker['name']}**: {speaker['mentions']} mentions")
            
            # Topic mentions
            st.write("### Sample Mentions")
            
            if "sample_mentions" in response and response["sample_mentions"]:
                for mention in response["sample_mentions"]:
                    with st.expander(f"{mention['date']} - {mention['speaker']}"):
                        st.write(mention["content"])
                        st.caption(f"Session: {mention['date']}, Time: {mention['timestamp']}")
            else:
                st.info("No sample mentions available for this topic")
        else:
            st.warning(f"No details found for topic: {topic_name}")


def main():
    """Main app entry point with UI state management"""
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Render sidebar (always present)
    render_sidebar()
    
    # Handle page navigation
    if st.session_state.page == "home":
        # Render landing page
        metadata = fetch_metadata()
        render_landing_page(metadata)
        
        # Handle quick search from landing page
        query = render_quick_search()
        if query:
            st.session_state.search_query = query
            st.session_state.page = "search_results"
            st.experimental_rerun()
    
    elif st.session_state.page == "speakers":
        # Render speaker view
        render_speaker_view()
    
    elif st.session_state.page == "speaker_detail":
        # Render details for selected speaker
        render_speaker_view(st.session_state.selected_speaker)
    
    elif st.session_state.page == "topics":
        # Render topic view
        render_topic_view()
    
    elif st.session_state.page == "topic_detail":
        # Render details for selected topic
        render_topic_detail(st.session_state.selected_topic)
    
    elif st.session_state.page == "sessions":
        # Render session view
        render_session_view()
    
    elif st.session_state.page == "session_detail":
        # Render details for selected session
        render_session_detail(st.session_state.selected_session)
    
    elif st.session_state.page == "search_results":
        # Render search results
        render_search_results(st.session_state.search_query)


if __name__ == "__main__":
    main() 