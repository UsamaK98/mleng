"""
Streamlit application for Parliamentary Meeting Analyzer.
Provides a user-friendly interface for exploring and analyzing parliamentary meeting data.
"""

import os
import time
import json
import pandas as pd
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple

from src.utils.logger import log
from src.utils.config import config_manager

# Configure page
st.set_page_config(
    page_title="Parliamentary Meeting Analyzer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API endpoint
API_BASE_URL = config_manager.get("api.url", "http://localhost:8000")

# -----------------------------------------
# Helper functions
# -----------------------------------------

def get_api_url(endpoint: str) -> str:
    """Get the full API URL for a given endpoint."""
    return f"{API_BASE_URL}{endpoint}"

def api_request(endpoint: str, method: str = "GET", params: Dict = None, json_data: Dict = None) -> Dict:
    """Make a request to the API."""
    url = get_api_url(endpoint)
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, params=params, json=json_data)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return {"error": f"Unsupported HTTP method: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return {"error": f"API Error ({response.status_code}): {response.text}"}
            
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return {"error": str(e)}

def check_api_health() -> bool:
    """Check if the API is healthy and available."""
    try:
        response = requests.get(get_api_url("/health"), timeout=2)
        return response.status_code == 200
    except:
        return False

# -----------------------------------------
# App components
# -----------------------------------------

def render_header():
    """Render the application header."""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/1986/1986935.png", width=100)
    
    with col2:
        st.title("Parliamentary Meeting Analyzer")
        st.markdown("Explore insights from parliamentary meeting minutes using AI and graph technologies.")

def render_sidebar():
    """Render the sidebar navigation menu."""
    st.sidebar.title("Navigation")
    
    # Main navigation
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Data Explorer", "Entity Analysis", "Graph View", "Q&A"]
    )
    
    # API connection status
    api_status = check_api_health()
    status_color = "green" if api_status else "red"
    st.sidebar.markdown(f"API Status: <span style='color:{status_color}'>{'Connected' if api_status else 'Disconnected'}</span>", unsafe_allow_html=True)
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application analyzes parliamentary meeting minutes using natural language processing "
        "and graph technologies to extract insights and answer questions about the content."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by: MLEng Team")
    st.sidebar.markdown("Version: 0.1.0")
    
    return page

def home_page():
    """Render the home page."""
    st.header("Welcome to the Parliamentary Meeting Analyzer")
    
    st.markdown("""
    This application helps you explore and analyze parliamentary meeting minutes using advanced AI technologies:
    
    - **Data Explorer**: Browse meeting minutes by session date or speaker
    - **Entity Analysis**: Discover key entities mentioned in the meetings
    - **Graph View**: Visualize connections between speakers, topics, and entities
    - **Q&A**: Ask questions about the meeting content using AI
    
    Get started by selecting a page from the sidebar navigation menu.
    """)
    
    # Display key statistics
    st.subheader("Key Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    try:
        # Get session data
        sessions = api_request("/api/data/sessions")
        if not isinstance(sessions, list):
            sessions = []
            
        # Get speaker data
        speakers = api_request("/api/data/speakers")
        if not isinstance(speakers, list):
            speakers = []
        
        # Get communities data
        communities = api_request("/api/graph/communities")
        if "error" in communities:
            community_count = 0
        else:
            community_count = len(communities)
        
        with col1:
            st.metric("Sessions", len(sessions))
            
        with col2:
            st.metric("Speakers", len(speakers))
            
        with col3:
            st.metric("Topic Communities", community_count)
            
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")
    
    # Recent sessions
    if len(sessions) > 0:
        st.subheader("Recent Sessions")
        
        # Sort sessions by date (assuming YYYY-MM-DD format)
        recent_sessions = sorted(sessions, reverse=True)[:5]
        
        for session in recent_sessions:
            st.markdown(f"- [{session}](/Data_Explorer?session={session})")
    
def data_explorer_page():
    """Render the data explorer page."""
    st.header("Data Explorer")
    st.markdown("Browse and analyze parliamentary meeting minutes by session date or speaker.")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["By Session", "By Speaker"])
    
    with tab1:
        st.subheader("Explore by Session")
        
        # Get sessions
        sessions = api_request("/api/data/sessions")
        if "error" in sessions:
            st.error(sessions["error"])
            return
            
        # Sort sessions by date
        sessions = sorted(sessions, reverse=True)
        
        # Session selector
        selected_session = st.selectbox("Select a session date:", sessions)
        
        if selected_session:
            # Get session data
            session_data = api_request(f"/api/data/session/{selected_session}")
            
            if "error" in session_data:
                st.error(session_data["error"])
            else:
                # Display session info
                st.markdown(f"**Date:** {selected_session}")
                st.markdown(f"**Contributions:** {session_data['count']}")
                st.markdown(f"**Speakers:** {', '.join(session_data['speakers'])}")
                
                # Process and analyze the session
                if st.button("Process and Analyze Session"):
                    with st.spinner("Processing..."):
                        process_result = api_request(f"/api/data/process/session/{selected_session}")
                        
                        if "error" in process_result:
                            st.error(process_result["error"])
                        else:
                            # Display entity information
                            st.subheader("Entities Mentioned")
                            
                            # Group entities by type
                            entity_types = {}
                            for entity in process_result["entities"]:
                                entity_type = entity.get("label", "Unknown")
                                if entity_type not in entity_types:
                                    entity_types[entity_type] = []
                                entity_types[entity_type].append(entity)
                            
                            # Display entity counts by type
                            entity_count_data = {
                                "Type": list(entity_types.keys()),
                                "Count": [len(entities) for entities in entity_types.values()]
                            }
                            
                            entity_df = pd.DataFrame(entity_count_data)
                            fig = px.bar(entity_df, x="Type", y="Count", title="Entity Counts by Type")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display entities by type
                            for entity_type, entities in entity_types.items():
                                with st.expander(f"{entity_type} ({len(entities)})"):
                                    entities_table = []
                                    for entity in entities:
                                        entities_table.append({
                                            "Text": entity.get("text", ""),
                                            "Confidence": f"{entity.get('score', 0):.2f}"
                                        })
                                    
                                    st.table(pd.DataFrame(entities_table))
                
                # Display session content
                with st.expander("Session Contributions"):
                    # Convert to DataFrame
                    df = pd.DataFrame(session_data["records"])
                    
                    # Sort by original order
                    if "Order" in df.columns:
                        df = df.sort_values("Order")
                    
                    # Display as a table with speaker and content
                    for _, row in df.iterrows():
                        st.markdown(f"**{row['Speaker']}**: {row['Content']}")
                        st.markdown("---")
        
    with tab2:
        st.subheader("Explore by Speaker")
        
        # Get speakers
        speakers = api_request("/api/data/speakers")
        if "error" in speakers:
            st.error(speakers["error"])
            return
            
        # Sort speakers alphabetically
        speakers = sorted(speakers)
        
        # Speaker selector
        selected_speaker = st.selectbox("Select a speaker:", speakers)
        
        if selected_speaker:
            # Get speaker data
            speaker_data = api_request(f"/api/data/speaker/{selected_speaker}")
            
            if "error" in speaker_data:
                st.error(speaker_data["error"])
            else:
                # Display speaker info
                st.markdown(f"**Name:** {selected_speaker}")
                st.markdown(f"**Contributions:** {speaker_data['count']}")
                st.markdown(f"**Sessions:** {', '.join(speaker_data['sessions'])}")
                
                if 'info' in speaker_data and speaker_data['info']:
                    st.markdown(f"**Role:** {speaker_data['info'].get('role', 'Unknown')}")
                    st.markdown(f"**Party:** {speaker_data['info'].get('party', 'Unknown')}")
                
                # Process and analyze the speaker
                if st.button("Process and Analyze Speaker"):
                    with st.spinner("Processing..."):
                        process_result = api_request(f"/api/data/process/speaker/{selected_speaker}")
                        
                        if "error" in process_result:
                            st.error(process_result["error"])
                        else:
                            # Display entity information
                            st.subheader("Entities Mentioned")
                            
                            # Group entities by type
                            entity_types = {}
                            for entity in process_result["entities"]:
                                entity_type = entity.get("label", "Unknown")
                                if entity_type not in entity_types:
                                    entity_types[entity_type] = []
                                entity_types[entity_type].append(entity)
                            
                            # Display entity counts by type
                            entity_count_data = {
                                "Type": list(entity_types.keys()),
                                "Count": [len(entities) for entities in entity_types.values()]
                            }
                            
                            entity_df = pd.DataFrame(entity_count_data)
                            fig = px.bar(entity_df, x="Type", y="Count", title="Entity Counts by Type")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display entities by type
                            for entity_type, entities in entity_types.items():
                                with st.expander(f"{entity_type} ({len(entities)})"):
                                    entities_table = []
                                    for entity in entities:
                                        entities_table.append({
                                            "Text": entity.get("text", ""),
                                            "Confidence": f"{entity.get('score', 0):.2f}"
                                        })
                                    
                                    st.table(pd.DataFrame(entities_table))
                
                # Display speaker contributions
                with st.expander("Speaker Contributions"):
                    # Convert to DataFrame
                    df = pd.DataFrame(speaker_data["records"])
                    
                    # Sort by date
                    df = df.sort_values("Date")
                    
                    # Group by date
                    for date, group in df.groupby("Date"):
                        st.markdown(f"**Session: {date}**")
                        
                        # Display each contribution
                        for _, row in group.iterrows():
                            st.markdown(f"{row['Content']}")
                            st.markdown("---")

def entity_analysis_page():
    """Render the entity analysis page."""
    st.header("Entity Analysis")
    st.markdown("Analyze entities mentioned in parliamentary meeting minutes.")
    
    # Filters
    st.subheader("Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get sessions
        sessions = api_request("/api/data/sessions")
        if "error" in sessions:
            sessions = []
        
        # Add "All Sessions" option
        session_options = ["All Sessions"] + sorted(sessions, reverse=True)
        selected_session = st.selectbox("Session:", session_options)
        
        # Convert "All Sessions" to None for API
        session_param = None if selected_session == "All Sessions" else selected_session
    
    with col2:
        # Get speakers
        speakers = api_request("/api/data/speakers")
        if "error" in speakers:
            speakers = []
        
        # Add "All Speakers" option
        speaker_options = ["All Speakers"] + sorted(speakers)
        selected_speaker = st.selectbox("Speaker:", speaker_options)
        
        # Convert "All Speakers" to None for API
        speaker_param = None if selected_speaker == "All Speakers" else selected_speaker
    
    with col3:
        # Entity types
        entity_types = ["All Types", "PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY", "PERCENT", "TIME", "FACILITY", "PRODUCT", "LAW", "WORK_OF_ART"]
        selected_entity_type = st.selectbox("Entity Type:", entity_types)
        
        # Convert "All Types" to None for API
        entity_type_param = None if selected_entity_type == "All Types" else selected_entity_type
    
    # Set result limit
    limit = st.slider("Number of results:", min_value=10, max_value=100, value=20, step=10)
    
    # Get entity frequency data
    if st.button("Analyze Entities"):
        with st.spinner("Analyzing..."):
            # Construct query parameters
            params = {
                "limit": limit
            }
            
            if session_param:
                params["session_date"] = session_param
                
            if speaker_param:
                params["speaker_name"] = speaker_param
                
            if entity_type_param:
                params["entity_type"] = entity_type_param
            
            # Make API request
            entity_data = api_request("/api/analytics/entity-frequency", params=params)
            
            if "error" in entity_data:
                st.error(entity_data["error"])
            else:
                st.success(f"Found {entity_data['count']} entities matching your criteria.")
                
                # Display entity distribution chart
                if entity_data['count'] > 0:
                    entities_df = pd.DataFrame(entity_data["entities"])
                    
                    # Sort by count descending
                    entities_df = entities_df.sort_values("count", ascending=False)
                    
                    # Create bar chart
                    fig = px.bar(
                        entities_df, 
                        x="name", 
                        y="count",
                        title="Entity Frequency",
                        labels={"name": "Entity", "count": "Mentions"},
                        color="type" if "type" in entities_df.columns else None
                    )
                    
                    # Improve layout
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        xaxis={'categoryorder':'total descending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display entity table
                    st.subheader("Entity Details")
                    st.dataframe(entities_df)

def graph_view_page():
    """Render the graph visualization page."""
    st.header("Graph View")
    st.markdown("Visualize the knowledge graph of parliamentary meeting data.")
    
    # Get communities data
    communities = api_request("/api/graph/communities")
    
    if "error" in communities:
        st.error(communities["error"])
        st.warning("Graph may not be built yet. Make sure to process data first.")
    else:
        # Display community statistics
        st.subheader("Topic Communities")
        
        # Create a DataFrame for communities
        community_data = []
        for community in communities:
            # Extract community information
            community_id = community.get("id")
            node_count = community.get("count", 0)
            topics = ", ".join(community.get("topics", ["Unknown"]))
            
            community_data.append({
                "ID": community_id,
                "Nodes": node_count,
                "Topics": topics
            })
        
        community_df = pd.DataFrame(community_data)
        
        # Show community table
        st.dataframe(community_df)
        
        # Community selector for detailed view
        st.subheader("Explore Community")
        
        selected_community = st.selectbox(
            "Select a community to explore:",
            [f"Community {row['ID']}: {row['Topics']}" for _, row in community_df.iterrows()]
        )
        
        if selected_community:
            # Extract community ID from selection
            community_id = int(selected_community.split(":")[0].replace("Community ", "").strip())
            
            # Find the selected community
            selected_community_data = None
            for community in communities:
                if community.get("id") == community_id:
                    selected_community_data = community
                    break
            
            if selected_community_data:
                # Display nodes in this community
                st.markdown(f"**Nodes in this community:** {len(selected_community_data.get('nodes', []))}")
                st.markdown(f"**Topics:** {', '.join(selected_community_data.get('topics', ['Unknown']))}")
                
                # Get information about nodes in this community
                node_data = []
                
                for node_id in selected_community_data.get("nodes", [])[:20]:  # Limit to first 20 nodes
                    # Get node info
                    node_info = api_request(f"/api/graph/node/{node_id}")
                    
                    if "error" not in node_info:
                        node_type = node_info.get("data", {}).get("node_type", "unknown")
                        node_name = node_info.get("data", {}).get("name", node_id)
                        
                        node_data.append({
                            "ID": node_id,
                            "Name": node_name,
                            "Type": node_type,
                            "Neighbors": len(node_info.get("neighbors", []))
                        })
                
                # Display node table
                if node_data:
                    node_df = pd.DataFrame(node_data)
                    st.dataframe(node_df)
                    
                    # Node selector for detailed view
                    selected_node = st.selectbox(
                        "Select a node to explore:",
                        [f"{row['Name']} ({row['Type']})" for _, row in node_df.iterrows()]
                    )
                    
                    if selected_node:
                        # Extract node name from selection
                        node_name = selected_node.split(" (")[0]
                        
                        # Find node ID
                        node_id = None
                        for _, row in node_df.iterrows():
                            if row["Name"] == node_name:
                                node_id = row["ID"]
                                break
                        
                        if node_id:
                            # Get detailed node info
                            node_detail = api_request(f"/api/graph/node/{node_id}")
                            
                            if "error" not in node_detail:
                                st.json(node_detail)

def qa_page():
    """Render the Q&A page."""
    st.header("Question & Answer")
    st.markdown("Ask questions about the parliamentary meeting data and get AI-powered answers.")
    
    # Query input
    query = st.text_input("Enter your question:", "What were the main topics discussed in the most recent session?")
    
    # Query options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        context_type = st.selectbox(
            "Context Type:",
            ["All", "speaker", "session", "entity", "PERSON", "ORGANIZATION", "LOCATION"]
        )
        # Convert "All" to None for API
        context_type_param = None if context_type == "All" else context_type
    
    with col2:
        max_results = st.slider("Maximum context items:", min_value=5, max_value=30, value=10, step=5)
    
    with col3:
        include_context = st.checkbox("Show context sources", value=False)
    
    # Process query
    if st.button("Submit Question"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                # Prepare request data
                json_data = {
                    "query": query,
                    "context_type": context_type_param,
                    "max_results": max_results,
                    "include_context": include_context,
                    "filters": {}  # Can add filters based on UI selections if needed
                }
                
                # Make API request
                response = api_request(
                    "/api/query/rag",
                    method="POST",
                    json_data=json_data
                )
                
                if "error" in response:
                    st.error(response["error"])
                else:
                    # Display answer
                    st.subheader("Answer")
                    st.markdown(response.get("response", "No answer generated."))
                    
                    # Display context if requested
                    if include_context and "context" in response:
                        st.subheader("Sources")
                        
                        for i, ctx in enumerate(response["context"]):
                            with st.expander(f"Source {i+1} (Score: {ctx.get('score', 0):.2f})"):
                                st.markdown(f"**Text:** {ctx.get('text', '')}")
                                
                                if "metadata" in ctx:
                                    metadata = ctx["metadata"]
                                    st.markdown(f"**Date:** {metadata.get('Date', 'Unknown')}")
                                    st.markdown(f"**Speaker:** {metadata.get('Speaker', 'Unknown')}")
                    
                    # Display query details
                    with st.expander("Query Details"):
                        st.json({
                            "query": query,
                            "context_type": context_type_param,
                            "max_results": max_results,
                            "execution_time": response.get("execution_time", 0)
                        })

# -----------------------------------------
# Main app
# -----------------------------------------

def main():
    """Main application entry point."""
    # Render header
    render_header()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "Home":
        home_page()
    elif page == "Data Explorer":
        data_explorer_page()
    elif page == "Entity Analysis":
        entity_analysis_page()
    elif page == "Graph View":
        graph_view_page()
    elif page == "Q&A":
        qa_page()

# Run the app
if __name__ == "__main__":
    main() 