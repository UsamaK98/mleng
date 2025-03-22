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
    st.header("💬 Chat with Parliamentary Minutes")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "sources" in message:
                st.write(message["content"])
                
                # Display sources if available
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"**Speaker:** {source['speaker']}  \n"
                                 f"**Date:** {source['date']}  \n"
                                 f"**Time:** {source['timestamp']}  \n"
                                 f"**Relevance:** {source['relevance_score']:.2f}")
            else:
                st.write(message["content"])
    
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
                response = query_api("query", {
                    "query": prompt,
                    "filters": {},
                    "structured_output": False
                })
                
                if response:
                    st.write(response["answer"])
                    
                    # Display sources if available
                    if "sources" in response and response["sources"]:
                        with st.expander("Sources"):
                            for source in response["sources"]:
                                st.write(f"**Speaker:** {source['speaker']}  \n"
                                         f"**Date:** {source['date']}  \n"
                                         f"**Time:** {source['timestamp']}  \n"
                                         f"**Relevance:** {source['relevance_score']:.2f}")
                    
                    # Save response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response["answer"],
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
                    st.warning(f"No information found for speaker: {entity}")
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


def main():
    """Main application"""
    # Render sidebar
    render_sidebar()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧑‍💼 Speaker Search", "📋 Topic Search"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_entity_search()
    
    with tab3:
        render_topic_search()


if __name__ == "__main__":
    main() 