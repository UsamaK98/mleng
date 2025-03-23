"""
Streamlit UI for the Parliamentary Minutes Agentic Chatbot
"""
import os
import sys
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import API_HOST, API_PORT, UI_THEME

# Set page configuration
st.set_page_config(
    page_title="Parliamentary Minutes Chatbot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Parliamentary Minutes Agentic Chatbot\nAI-powered chatbot for Scottish Parliament meeting minutes."
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
    """Render the sidebar with metadata and filters"""
    st.sidebar.title("Parliamentary Minutes Chatbot")
    
    metadata = fetch_metadata()
    
    if metadata:
        st.sidebar.header("Dataset Info")
        st.sidebar.write(f"Total Sessions: {metadata['total_sessions']}")
        st.sidebar.write(f"Total Speakers: {metadata['total_speakers']}")
        
        # Display session dates
        with st.sidebar.expander("Session Dates"):
            for date in metadata.get('session_dates', []):
                st.write(date)
        
        # Display top speakers
        with st.sidebar.expander("Top Speakers"):
            for speaker in metadata.get('top_speakers', [])[:5]:
                st.write(f"**{speaker['name']}** ({speaker['contributions']} contributions)")
    
    # Add filtering options
    st.sidebar.header("Filters")
    
    # Add filter instructions
    st.sidebar.info("Use the filters in each tab to refine your queries.")


def render_chat_interface():
    """Render the main chat interface"""
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.write(content)
            
            # Display sources if available
            if role == "assistant" and "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"**Speaker:** {source['speaker']}  \n"
                                 f"**Date:** {source['date']}  \n"
                                 f"**Time:** {source['timestamp']}  \n"
                                 f"**Relevance:** {source['relevance_score']:.2f}")
    
    # Chat input
    if prompt := st.chat_input("Ask about parliamentary minutes..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Try to detect if it's an entity query first
                entity_pattern = r"(what|who|where|when|tell me about|information on|search for)\s+(.*?)\s+(said|spoke|discussed|mentioned|contributions?)"
                
                entity_match = re.search(entity_pattern, prompt.lower())
                entity_query = False
                
                if entity_match:
                    potential_entity = entity_match.group(2).strip()
                    # Try entity query first
                    response = query_api("entity", {
                        "entity": potential_entity,
                        "use_hybrid": True
                    })
                    
                    if response and response.get("found", True):
                        entity_query = True
                
                # If not an entity query or entity not found, do regular query
                if not entity_query:
                    response = query_api("query", {
                        "query": prompt,
                        "filters": {},
                        "structured_output": False
                    })
                
                if response:
                    if "answer" in response:
                        st.write(response["answer"])
                    
                    # Display sources if available
                    if "sources" in response and response["sources"]:
                        with st.expander("Sources"):
                            for source in response["sources"]:
                                st.write(f"**Speaker:** {source['speaker']}  \n"
                                         f"**Date:** {source['date']}  \n"
                                         f"**Time:** {source['timestamp']}  \n"
                                         f"**Relevance:** {source['relevance_score']:.2f}")
                    
                    # If entity suggestions are available
                    if not response.get("found", True) and "available_speakers" in response:
                        st.write("### Try asking about these speakers instead:")
                        cols = st.columns(min(3, len(response["available_speakers"])))
                        
                        for i, speaker in enumerate(response["available_speakers"]):
                            with cols[i % 3]:
                                st.button(speaker, key=f"chat_alt_speaker_{i}")
                    
                    # Save response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.get("answer", "I processed your query but couldn't generate a proper response."),
                        "sources": response.get("sources", [])
                    })
                else:
                    st.error("Failed to get a response. Please try again.")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "I'm sorry, I couldn't process your request. Please try again."
                    })


def render_entity_search():
    """Render the entity search interface"""
    st.header("🧑‍💼 Search by Speaker")
    
    entity = st.text_input("Enter speaker name:", key="entity_input")
    
    if st.button("Search", key="entity_search"):
        if entity:
            with st.spinner("Searching..."):
                response = query_api("entity", {"entity": entity})
                
                if response and response.get("found", False):
                    # Display entity info
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader(response["entity"])
                        st.write(f"**Role:** {response.get('role', 'Unknown')}")
                        
                        if "summary" in response:
                            st.write("### Summary")
                            st.write(response["summary"])
                    
                    with col2:
                        st.write("### Statistics")
                        st.write(f"Total Contributions: {response.get('total_contributions', 0)}")
                        st.write(f"Total Words: {response.get('speaker_stats', {}).get('total_words', 0)}")
                        st.write(f"Avg Words per Contribution: {response.get('speaker_stats', {}).get('avg_words_per_contribution', 0):.1f}")
                    
                    # Meeting dates
                    st.write("### Meeting Dates Present")
                    dates_df = pd.DataFrame({
                        "Date": response.get("meeting_dates_present", []),
                        "Contributions": [response.get("contributions_by_date", {}).get(d, 0) 
                                         for d in response.get("meeting_dates_present", [])]
                    })
                    
                    if not dates_df.empty:
                        st.dataframe(dates_df)
                    
                    # Sample contributions
                    if "sample_contributions" in response and response["sample_contributions"]:
                        st.write("### Sample Contributions")
                        for i, contrib in enumerate(response["sample_contributions"], 1):
                            with st.expander(f"Contribution {i} - {contrib['date']}"):
                                st.write(f"**Date:** {contrib['date']}")
                                st.write(f"**Time:** {contrib['timestamp']}")
                                st.write(f"**Content:** {contrib['content']}")
                else:
                    st.warning(response.get("answer", f"No information found for speaker: {entity}"))
                    
                    # Display suggested speakers if available
                    if response and "available_speakers" in response and response["available_speakers"]:
                        st.write("### Try searching for these speakers instead:")
                        cols = st.columns(min(3, len(response["available_speakers"])))
                        
                        for i, speaker in enumerate(response["available_speakers"]):
                            with cols[i % 3]:
                                if st.button(speaker, key=f"alt_speaker_{i}"):
                                    # "Click" the button with this speaker instead
                                    st.session_state.entity_input = speaker
                                    st.experimental_rerun()
        else:
            st.warning("Please enter a speaker name")


def render_topic_search():
    """Render the topic search interface"""
    st.header("📋 Search by Topic")
    
    topic = st.text_input("Enter topic:", key="topic_input")
    
    if st.button("Search", key="topic_search"):
        if topic:
            with st.spinner("Searching..."):
                response = query_api("topic", {"topic": topic})
                
                if response and response.get("found", False):
                    # Display topic summary
                    st.subheader(f"Topic: {response['topic']}")
                    
                    # Summary
                    st.write("### Summary")
                    st.write(response["summary"])
                    
                    # Metadata
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Speakers Discussing This Topic")
                        if "speakers_discussing" in response:
                            for speaker in response["speakers_discussing"]:
                                st.write(f"- {speaker}")
                    
                    with col2:
                        st.write("### Dates Discussed")
                        if "dates_discussed" in response:
                            for date in response["dates_discussed"]:
                                st.write(f"- {date}")
                    
                    # Source excerpts
                    if "sources" in response and response["sources"]:
                        st.write("### Source Excerpts")
                        for i, source in enumerate(response["sources"], 1):
                            with st.expander(f"Source {i} - {source['speaker']} ({source['date']})"):
                                st.write(f"**Speaker:** {source['speaker']}")
                                st.write(f"**Date:** {source['date']}")
                                st.write(f"**Time:** {source['timestamp']}")
                                st.write(f"**Excerpt:** {source['excerpt']}")
                else:
                    st.warning(f"No information found for topic: {topic}")
        else:
            st.warning("Please enter a topic")


def render_analytics_interface():
    """Render the analytics interface with multiple sub-tabs"""
    st.header("📊 Parliamentary Analytics")
    
    # Create sub-tabs for different analytics features
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "👤 Speaker Analytics", 
        "📅 Session Analytics", 
        "🔗 Relationship Mapping",
        "😀 Sentiment Analysis"
    ])
    
    with subtab1:
        render_speaker_analytics()
    
    with subtab2:
        render_session_analytics()
    
    with subtab3:
        render_relationship_analytics()
    
    with subtab4:
        render_sentiment_analytics()


def render_speaker_analytics():
    """Render the speaker analytics interface"""
    st.subheader("Speaker Analytics")
    
    # Add controls for speaker analytics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Analyze and compare parliamentary speakers based on their contributions.")
        
        # Get top speakers
        if st.button("View Top Speakers", key="top_speakers_btn"):
            try:
                response = requests.get(f"{API_BASE_URL}/analytics/speakers")
                if response.status_code == 200:
                    data = response.json()
                    if data["status"] == "success":
                        top_speakers = data["data"]["top_speakers"]
                        
                        # Create a DataFrame for better display
                        df = pd.DataFrame(top_speakers)
                        if not df.empty:
                            st.write("### Top Speakers by Contribution Count")
                            st.dataframe(df)
                        else:
                            st.info("No speaker data available.")
                    else:
                        st.error(f"Error: {data.get('message', 'Unknown error')}")
                else:
                    st.error(f"API Error: Status code {response.status_code}")
            except Exception as e:
                st.error(f"Failed to fetch speaker data: {e}")
        
        # Speaker comparison
        st.write("### Compare Speakers")
        speaker1 = st.text_input("First Speaker:", key="compare_speaker1")
        speaker2 = st.text_input("Second Speaker:", key="compare_speaker2")
        
        if st.button("Compare", key="compare_speakers_btn") and speaker1 and speaker2:
            try:
                response = requests.get(
                    f"{API_BASE_URL}/analytics/speakers/compare",
                    params={"speaker1": speaker1, "speaker2": speaker2}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data["status"] == "success":
                        comparison = data["data"]
                        
                        # Display comparison data
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"### {speaker1}")
                            st.write(f"**Role:** {comparison.get('speaker1_stats', {}).get('role', 'Unknown')}")
                            st.write(f"**Contributions:** {comparison.get('speaker1_stats', {}).get('contributions', 0)}")
                            st.write(f"**Total Words:** {comparison.get('speaker1_stats', {}).get('total_words', 0)}")
                            st.write(f"**Avg Words/Contribution:** {comparison.get('speaker1_stats', {}).get('avg_words_per_contribution', 0):.1f}")
                        
                        with col_b:
                            st.write(f"### {speaker2}")
                            st.write(f"**Role:** {comparison.get('speaker2_stats', {}).get('role', 'Unknown')}")
                            st.write(f"**Contributions:** {comparison.get('speaker2_stats', {}).get('contributions', 0)}")
                            st.write(f"**Total Words:** {comparison.get('speaker2_stats', {}).get('total_words', 0)}")
                            st.write(f"**Avg Words/Contribution:** {comparison.get('speaker2_stats', {}).get('avg_words_per_contribution', 0):.1f}")
                        
                        # Show common topics if available
                        if "common_topics" in comparison:
                            st.write("### Common Topics")
                            st.write(", ".join(comparison["common_topics"]))
                        
                        # Show interaction data if available
                        if "interactions" in comparison:
                            st.write("### Interactions")
                            st.write(f"Direct Interactions: {comparison['interactions'].get('direct_interactions', 0)}")
                            st.write(f"Most Common Session: {comparison['interactions'].get('most_common_session', 'None')}")
                    else:
                        st.error(f"Error: {data.get('message', 'Unknown error')}")
                else:
                    st.error(f"API Error: Status code {response.status_code}")
            except Exception as e:
                st.error(f"Failed to compare speakers: {e}")
    
    with col2:
        st.write("### Speaker Details")
        speaker_name = st.text_input("Enter speaker name:", key="analytics_speaker_name")
        
        if st.button("Get Details", key="get_speaker_details") and speaker_name:
            try:
                response = requests.get(f"{API_BASE_URL}/analytics/speakers/{speaker_name}")
                
                if response.status_code == 200:
                    data = response.json()
                    if data["status"] == "success":
                        stats = data["data"]
                        
                        st.write(f"### {speaker_name}")
                        st.write(f"**Role:** {stats.get('role', 'Unknown')}")
                        st.write(f"**Total Contributions:** {stats.get('total_contributions', 0)}")
                        st.write(f"**Total Words:** {stats.get('total_words', 0)}")
                        st.write(f"**Sessions Attended:** {stats.get('sessions_attended', 0)}")
                        
                        if "frequent_topics" in stats:
                            st.write("### Frequent Topics")
                            st.write(", ".join(stats["frequent_topics"]))
                        
                        if "sentiment" in stats:
                            sentiment = stats["sentiment"]
                            st.write("### Sentiment Analysis")
                            st.write(f"Average Polarity: {sentiment.get('average_polarity', 0):.2f}")
                            st.write(f"Subjectivity: {sentiment.get('subjectivity', 0):.2f}")
                    else:
                        st.warning(f"No data found for speaker: {speaker_name}")
                else:
                    st.error(f"API Error: Status code {response.status_code}")
            except Exception as e:
                st.error(f"Failed to get speaker details: {e}")


def render_session_analytics():
    """Render the session analytics interface"""
    st.subheader("Session Analytics")
    
    st.write("Analyze parliamentary sessions and track changes over time.")
    
    # Session timeline
    if st.button("View Session Timeline", key="session_timeline_btn"):
        try:
            response = requests.get(f"{API_BASE_URL}/analytics/sessions")
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    timeline = data["data"]["timeline"]
                    
                    if timeline:
                        # Create a DataFrame for better display
                        df = pd.DataFrame(timeline)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                        
                        st.write("### Session Timeline")
                        st.dataframe(df)
                        
                        # Create a chart of contributions over time
                        st.write("### Contributions Over Time")
                        chart_data = df[['date', 'contributions']]
                        st.line_chart(chart_data.set_index('date'))
                    else:
                        st.info("No session timeline data available.")
                else:
                    st.error(f"Error: {data.get('message', 'Unknown error')}")
            else:
                st.error(f"API Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to fetch session timeline: {e}")
    
    # Session comparison
    st.write("### Compare Sessions")
    col1, col2 = st.columns(2)
    
    with col1:
        session1 = st.text_input("First Session Date (YYYY-MM-DD):", key="compare_session1")
    
    with col2:
        session2 = st.text_input("Second Session Date (YYYY-MM-DD):", key="compare_session2")
    
    if st.button("Compare", key="compare_sessions_btn") and session1 and session2:
        try:
            response = requests.get(
                f"{API_BASE_URL}/analytics/sessions/compare",
                params={"session1": session1, "session2": session2}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    comparison = data["data"]
                    
                    # Display comparison data
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"### Session: {session1}")
                        session1_stats = comparison.get("session1_stats", {})
                        st.write(f"**Contributions:** {session1_stats.get('total_contributions', 0)}")
                        st.write(f"**Unique Speakers:** {session1_stats.get('unique_speakers', 0)}")
                        st.write(f"**Top Speaker:** {session1_stats.get('top_speakers', ['Unknown'])[0] if session1_stats.get('top_speakers') else 'Unknown'}")
                    
                    with col_b:
                        st.write(f"### Session: {session2}")
                        session2_stats = comparison.get("session2_stats", {})
                        st.write(f"**Contributions:** {session2_stats.get('total_contributions', 0)}")
                        st.write(f"**Unique Speakers:** {session2_stats.get('unique_speakers', 0)}")
                        st.write(f"**Top Speaker:** {session2_stats.get('top_speakers', ['Unknown'])[0] if session2_stats.get('top_speakers') else 'Unknown'}")
                    
                    # Show common speakers if available
                    if "common_speakers" in comparison:
                        st.write("### Common Speakers")
                        st.write(", ".join(comparison["common_speakers"]))
                    
                    # Show topic similarity
                    if "topic_similarity" in comparison:
                        st.write(f"### Topic Similarity: {comparison['topic_similarity']:.2f}")
                else:
                    st.error(f"Error: {data.get('message', 'Unknown error')}")
            else:
                st.error(f"API Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to compare sessions: {e}")
    
    # Individual session analysis
    st.write("### Session Details")
    session_date = st.text_input("Enter session date (YYYY-MM-DD):", key="analytics_session_date")
    
    if st.button("Get Details", key="get_session_details") and session_date:
        try:
            response = requests.get(f"{API_BASE_URL}/analytics/sessions/{session_date}")
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    stats = data["data"]
                    
                    st.write(f"### Session: {session_date}")
                    st.write(f"**Total Contributions:** {stats.get('total_contributions', 0)}")
                    st.write(f"**Unique Speakers:** {stats.get('unique_speakers', 0)}")
                    
                    # Top speakers
                    if "top_speakers" in stats and stats["top_speakers"]:
                        st.write("### Top Speakers")
                        for speaker in stats["top_speakers"][:5]:
                            st.write(f"- {speaker}")
                    
                    # Common terms
                    if "common_terms" in stats and stats["common_terms"]:
                        st.write("### Common Terms")
                        st.write(", ".join(stats["common_terms"]))
                else:
                    st.warning(f"No data found for session: {session_date}")
            else:
                st.error(f"API Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to get session details: {e}")


def render_relationship_analytics():
    """Render the relationship analytics interface"""
    st.subheader("Relationship Mapping")
    
    st.write("Analyze and visualize relationships between speakers based on their interactions.")
    
    # Interaction network
    col1, col2 = st.columns([3, 1])
    
    with col2:
        min_interactions = st.slider(
            "Minimum Interactions", 
            min_value=1, 
            max_value=10, 
            value=2, 
            help="Minimum number of interactions between speakers to include in the network"
        )
    
    with col1:
        if st.button("Generate Network", key="gen_network_btn"):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/analytics/relationships/network",
                    params={"min_interactions": min_interactions}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data["status"] == "success":
                        network_data = data["data"]
                        
                        if "nodes" in network_data and "edges" in network_data:
                            st.write("### Speaker Interaction Network")
                            st.json(network_data)
                            
                            # In a real implementation, you would use a visualization library
                            st.info("Network visualization would be rendered here using a graph visualization library.")
                        else:
                            st.info("Not enough interaction data available with current settings.")
                    else:
                        st.error(f"Error: {data.get('message', 'Unknown error')}")
                else:
                    st.error(f"API Error: Status code {response.status_code}")
            except Exception as e:
                st.error(f"Failed to generate network: {e}")
    
    # Key influencers
    st.write("### Key Influencers")
    
    num_influencers = st.slider("Number of Influencers", min_value=3, max_value=15, value=5)
    
    if st.button("Find Key Influencers", key="find_influencers_btn"):
        try:
            response = requests.get(
                f"{API_BASE_URL}/analytics/relationships/influencers",
                params={"limit": num_influencers}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    influencers = data["data"]["influencers"]
                    
                    if influencers:
                        # Create a DataFrame for better display
                        df = pd.DataFrame(influencers)
                        st.dataframe(df)
                    else:
                        st.info("No influencer data available.")
                else:
                    st.error(f"Error: {data.get('message', 'Unknown error')}")
            else:
                st.error(f"API Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to find key influencers: {e}")
    
    # Community detection
    st.write("### Speaker Communities")
    
    if st.button("Detect Communities", key="detect_communities_btn"):
        try:
            response = requests.get(f"{API_BASE_URL}/analytics/relationships/communities")
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    communities = data["data"]
                    
                    if "communities" in communities and communities["communities"]:
                        st.write(f"Detected {len(communities['communities'])} communities")
                        
                        for i, community in enumerate(communities["communities"], 1):
                            with st.expander(f"Community {i} - {len(community)} members"):
                                st.write(", ".join(community))
                    else:
                        st.info("No community data available.")
                else:
                    st.error(f"Error: {data.get('message', 'Unknown error')}")
            else:
                st.error(f"API Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to detect communities: {e}")


def render_sentiment_analytics():
    """Render the sentiment analytics interface"""
    st.subheader("Sentiment Analysis")
    
    st.write("Analyze sentiment trends in parliamentary discussions.")
    
    # Overall sentiment
    if st.button("Overall Sentiment Statistics", key="overall_sentiment_btn"):
        try:
            response = requests.get(f"{API_BASE_URL}/analytics/sentiment/overall")
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    stats = data["data"]
                    
                    # Display overall sentiment stats
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Polarity", f"{stats.get('avg_polarity', 0):.3f}")
                        
                    with col2:
                        st.metric("Positive Percentage", f"{stats.get('positive_percentage', 0):.1f}%")
                        
                    with col3:
                        st.metric("Negative Percentage", f"{stats.get('negative_percentage', 0):.1f}%")
                    
                    st.write("### Sentiment Distribution")
                    neutral_pct = stats.get('neutral_percentage', 0)
                    positive_pct = stats.get('positive_percentage', 0)
                    negative_pct = stats.get('negative_percentage', 0)
                    
                    # Create a simple bar chart
                    chart_data = pd.DataFrame({
                        'Category': ['Positive', 'Neutral', 'Negative'],
                        'Percentage': [positive_pct, neutral_pct, negative_pct]
                    })
                    st.bar_chart(chart_data.set_index('Category'))
                else:
                    st.error(f"Error: {data.get('message', 'Unknown error')}")
            else:
                st.error(f"API Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to get overall sentiment: {e}")
    
    # Sentiment by speaker
    st.write("### Sentiment by Speaker")
    
    num_speakers = st.slider("Number of Speakers", min_value=5, max_value=20, value=10)
    
    if st.button("Analyze Speaker Sentiment", key="speaker_sentiment_btn"):
        try:
            response = requests.get(
                f"{API_BASE_URL}/analytics/sentiment/by-speaker",
                params={"limit": num_speakers}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    speakers = data["data"]["speakers"]
                    
                    if speakers:
                        # Create a DataFrame for better display
                        df = pd.DataFrame(speakers)
                        
                        # Basic table view
                        st.dataframe(df)
                        
                        # Create a chart of average polarity by speaker
                        st.write("### Average Sentiment Polarity by Speaker")
                        chart_data = df[['speaker', 'avg_polarity']].set_index('speaker')
                        st.bar_chart(chart_data)
                    else:
                        st.info("No speaker sentiment data available.")
                else:
                    st.error(f"Error: {data.get('message', 'Unknown error')}")
            else:
                st.error(f"API Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to analyze speaker sentiment: {e}")
    
    # Sentiment by session
    st.write("### Sentiment Trends by Session")
    
    if st.button("Analyze Session Sentiment", key="session_sentiment_btn"):
        try:
            response = requests.get(f"{API_BASE_URL}/analytics/sentiment/by-session")
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    sessions = data["data"]["sessions"]
                    
                    if sessions:
                        # Create a DataFrame for better display
                        df = pd.DataFrame(sessions)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                        
                        # Create a chart of sentiment over time
                        st.write("### Sentiment Polarity Over Time")
                        chart_data = df[['date', 'avg_polarity']]
                        st.line_chart(chart_data.set_index('date'))
                        
                        # Create a chart of positive/negative percentages
                        st.write("### Positive vs Negative Sentiment Over Time")
                        chart_data = df[['date', 'positive_percentage', 'negative_percentage']]
                        st.line_chart(chart_data.set_index('date'))
                    else:
                        st.info("No session sentiment data available.")
                else:
                    st.error(f"Error: {data.get('message', 'Unknown error')}")
            else:
                st.error(f"API Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to analyze session sentiment: {e}")
    
    # Sentiment keywords
    st.write("### Sentiment-Associated Keywords")
    
    num_keywords = st.slider("Number of Keywords", min_value=5, max_value=30, value=15)
    
    if st.button("Extract Sentiment Keywords", key="sentiment_keywords_btn"):
        try:
            response = requests.get(
                f"{API_BASE_URL}/analytics/sentiment/keywords",
                params={"num_keywords": num_keywords}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    keywords = data["data"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Positive Keywords")
                        if "positive_keywords" in keywords and keywords["positive_keywords"]:
                            df_positive = pd.DataFrame(keywords["positive_keywords"])
                            st.dataframe(df_positive)
                        else:
                            st.info("No positive keywords data available.")
                    
                    with col2:
                        st.write("### Negative Keywords")
                        if "negative_keywords" in keywords and keywords["negative_keywords"]:
                            df_negative = pd.DataFrame(keywords["negative_keywords"])
                            st.dataframe(df_negative)
                        else:
                            st.info("No negative keywords data available.")
                else:
                    st.error(f"Error: {data.get('message', 'Unknown error')}")
            else:
                st.error(f"API Error: Status code {response.status_code}")
        except Exception as e:
            st.error(f"Failed to extract sentiment keywords: {e}")


def main():
    """Main application"""
    # Render sidebar
    render_sidebar()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🧑‍💼 Speaker Search", "📋 Topic Search", "📊 Analytics"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_entity_search()
    
    with tab3:
        render_topic_search()
        
    with tab4:
        render_analytics_interface()


if __name__ == "__main__":
    main() 