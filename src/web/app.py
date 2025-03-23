"""
Streamlit web application for Parliamentary Meeting Analyzer.

This application provides a user-friendly interface to explore and query
parliamentary meeting data using GraphRAG and knowledge graph visualization.
"""

import os
import sys
import time
import json
import pandas as pd
import streamlit as st
import networkx as nx
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import logger
from src.utils.config import config_manager
from src.data.loader import ParliamentaryDataLoader
from src.models.ner import EntityExtractor
from src.models.graph import KnowledgeGraph
from src.models.graphrag import GraphRAG
from src.services.ollama import OllamaService
from src.storage.vector_db import VectorStore
from src.utils.graph_visualization import create_plotly_graph, export_community_data

# Set page configuration
st.set_page_config(
    page_title=config_manager.config.ui_title,
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data_loader = None
    st.session_state.ollama_service = None
    st.session_state.knowledge_graph = None
    st.session_state.vector_store = None
    st.session_state.graphrag = None
    st.session_state.entity_extractor = None
    st.session_state.sessions_data = {}
    st.session_state.current_session_data = None
    st.session_state.entity_map = None
    st.session_state.graph_built = False
    st.session_state.available_dates = []
    st.session_state.selected_dates = []
    st.session_state.query_history = []

def initialize_services():
    """Initialize all services required for the application."""
    with st.spinner("Initializing services..."):
        # Initialize data loader
        st.session_state.data_loader = ParliamentaryDataLoader()
        
        # Initialize Ollama service
        st.session_state.ollama_service = OllamaService()
        
        # Initialize entity extractor
        st.session_state.entity_extractor = EntityExtractor()
        
        # Initialize knowledge graph
        st.session_state.knowledge_graph = KnowledgeGraph()
        
        # Initialize vector store
        st.session_state.vector_store = VectorStore(
            collection_name="parliament",
            ollama_service=st.session_state.ollama_service
        )
        
        # Mark as initialized
        st.session_state.initialized = True
        st.success("Services initialized successfully!")

def load_data():
    """Load and preprocess parliamentary data."""
    with st.spinner("Loading parliamentary data..."):
        # Load data
        st.session_state.data_loader.load_data()
        
        # Get available dates
        st.session_state.available_dates = st.session_state.data_loader.get_unique_dates()
        
        if not st.session_state.available_dates:
            st.error("No session dates found in the data")
            return False
        
        return True

def load_selected_sessions():
    """Load data for selected sessions."""
    if not st.session_state.selected_dates:
        st.warning("Please select at least one session date")
        return False
    
    with st.spinner(f"Loading data for {len(st.session_state.selected_dates)} selected sessions..."):
        # Concatenate data from selected sessions
        session_dfs = []
        for date in st.session_state.selected_dates:
            if date not in st.session_state.sessions_data:
                session_df = st.session_state.data_loader.get_session_data(date)
                if not session_df.empty:
                    st.session_state.sessions_data[date] = session_df
            
            if date in st.session_state.sessions_data:
                session_dfs.append(st.session_state.sessions_data[date])
        
        if not session_dfs:
            st.error("No data found for the selected sessions")
            return False
        
        st.session_state.current_session_data = pd.concat(session_dfs, ignore_index=True)
        
        # Log info
        st.info(f"Loaded {len(st.session_state.current_session_data)} parliamentary statements")
        
        return True

def extract_entities():
    """Extract entities from the current session data."""
    if st.session_state.current_session_data is None:
        st.warning("Please load session data first")
        return False
    
    with st.spinner("Extracting entities using GLiNER..."):
        # Extract entities
        enhanced_df, entity_map = st.session_state.entity_extractor.extract_entities_from_dataframe(
            st.session_state.current_session_data
        )
        
        st.session_state.current_session_data = enhanced_df
        st.session_state.entity_map = entity_map
        
        # Log info
        st.info(f"Extracted entities for {len(entity_map)} statements")
        
        return True

def build_knowledge_graph():
    """Build knowledge graph from the current session data."""
    if st.session_state.current_session_data is None or st.session_state.entity_map is None:
        st.warning("Please load session data and extract entities first")
        return False
    
    with st.spinner("Building knowledge graph..."):
        # Build graph
        success = st.session_state.knowledge_graph.build_from_parliamentary_data(
            st.session_state.current_session_data,
            st.session_state.entity_map
        )
        
        if not success:
            st.error("Failed to build knowledge graph")
            return False
        
        # Get graph statistics
        graph_stats = st.session_state.knowledge_graph.get_graph_statistics()
        
        # Store in session state
        st.session_state.graph_built = True
        st.session_state.graph_stats = graph_stats
        
        # Log info
        st.success(f"Knowledge graph built with {graph_stats['num_nodes']} nodes and {graph_stats['num_edges']} edges")
        
        return True

def build_vector_store():
    """Build vector store from the current session data."""
    if st.session_state.current_session_data is None:
        st.warning("Please load session data first")
        return False
    
    with st.spinner("Building vector store..."):
        # Store data in vector store
        st.session_state.vector_store.store_parliamentary_data(st.session_state.current_session_data)
        
        # Log info
        st.success("Vector store built successfully")
        
        return True

def initialize_graphrag():
    """Initialize GraphRAG with knowledge graph and vector store."""
    if not st.session_state.graph_built:
        st.warning("Please build knowledge graph first")
        return False
    
    with st.spinner("Initializing GraphRAG..."):
        # Initialize GraphRAG
        st.session_state.graphrag = GraphRAG(
            kg=st.session_state.knowledge_graph,
            ollama_service=st.session_state.ollama_service,
            vector_store=st.session_state.vector_store
        )
        
        # Log info
        st.success("GraphRAG initialized successfully")
        
        return True

def process_query(query, mode="hybrid", max_results=10):
    """Process a query using GraphRAG."""
    if st.session_state.graphrag is None:
        st.warning("Please initialize GraphRAG first")
        return None
    
    with st.spinner(f"Processing query in {mode} mode..."):
        start_time = time.time()
        
        # Process query
        result = st.session_state.graphrag.query(
            query,
            mode=mode,
            max_results=max_results,
            include_sources=True
        )
        
        duration = time.time() - start_time
        
        # Add to query history
        query_record = {
            "query": query,
            "mode": mode,
            "result": result,
            "time": duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.session_state.query_history.append(query_record)
        
        return result

def render_sidebar():
    """Render the application sidebar."""
    st.sidebar.title("Parliamentary Meeting Analyzer")
    
    # Initialize services if needed
    if not st.session_state.initialized:
        if st.sidebar.button("🚀 Initialize Services"):
            initialize_services()
    
    # Data loading section
    st.sidebar.header("Data")
    
    if st.session_state.initialized:
        if st.session_state.data_loader is None or not st.session_state.available_dates:
            if st.sidebar.button("📥 Load Data"):
                load_data()
        else:
            # Session selection
            st.sidebar.subheader("Select Sessions")
            selected_dates = st.sidebar.multiselect(
                "Choose session dates:",
                st.session_state.available_dates,
                default=st.session_state.selected_dates
            )
            
            if selected_dates != st.session_state.selected_dates:
                st.session_state.selected_dates = selected_dates
                st.session_state.graph_built = False
            
            if st.sidebar.button("✅ Load Selected Sessions"):
                if load_selected_sessions():
                    st.session_state.entity_map = None
                    st.session_state.graph_built = False
            
            # Entity extraction
            if st.session_state.current_session_data is not None:
                if st.session_state.entity_map is None:
                    if st.sidebar.button("🔍 Extract Entities"):
                        extract_entities()
                else:
                    st.sidebar.success("✓ Entities extracted")
            
            # Knowledge graph building
            if st.session_state.entity_map is not None:
                if not st.session_state.graph_built:
                    if st.sidebar.button("🕸️ Build Knowledge Graph"):
                        if build_knowledge_graph():
                            build_vector_store()
                            initialize_graphrag()
                else:
                    st.sidebar.success("✓ Knowledge graph built")
    
    # Settings section
    st.sidebar.header("Settings")
    
    if st.session_state.initialized:
        # Query settings
        st.sidebar.subheader("Query Settings")
        
        query_mode = st.sidebar.radio(
            "Query mode:",
            ["hybrid", "graph", "vector"],
            horizontal=True
        )
        
        max_results = st.sidebar.slider(
            "Max results:",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
        
        # Advanced settings (collapsible)
        with st.sidebar.expander("Advanced Settings"):
            st.write("Graph Settings")
            max_hops = st.slider(
                "Max hops for graph traversal:",
                min_value=1,
                max_value=5,
                value=2
            )
            
            # Update config
            if max_hops != config_manager.config.graphrag.max_hops:
                config_manager.config.graphrag.max_hops = max_hops
        
        # Return query settings
        return {
            "query_mode": query_mode,
            "max_results": max_results
        }
    
    return None

def render_graph_section():
    """Render the knowledge graph visualization section."""
    st.header("Knowledge Graph Visualization")
    
    if not st.session_state.graph_built:
        st.info("Please build the knowledge graph first")
        return
    
    # Graph statistics
    st.subheader("Graph Statistics")
    
    stats = st.session_state.graph_stats
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", stats["num_nodes"])
    with col2:
        st.metric("Edges", stats["num_edges"])
    with col3:
        st.metric("Communities", stats.get("num_communities", 0))
    
    # Node type distribution
    st.subheader("Node Types")
    
    node_counts = stats.get("node_counts", {})
    if node_counts:
        fig = px.pie(
            values=list(node_counts.values()),
            names=list(node_counts.keys()),
            title="Node Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Graph visualization
    st.subheader("Interactive Graph")
    
    # Control max nodes to display
    max_nodes = st.slider(
        "Max nodes to display:",
        min_value=50,
        max_value=500,
        value=100,
        step=50
    )
    
    # Create interactive graph visualization
    try:
        with st.spinner("Generating interactive graph visualization..."):
            fig = create_plotly_graph(st.session_state.knowledge_graph, max_nodes=max_nodes)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating graph visualization: {str(e)}")
    
    # Community analysis
    if stats.get("num_communities", 0) > 0:
        st.subheader("Community Analysis")
        
        # Export community data
        community_data = export_community_data(st.session_state.knowledge_graph)
        
        # Show top communities
        communities = community_data.get("communities", [])
        communities.sort(key=lambda x: x.get("size", 0), reverse=True)
        
        top_communities = communities[:5]  # Show top 5 communities
        
        for i, community in enumerate(top_communities):
            with st.expander(f"Community {community['community_id']} ({community['size']} nodes)"):
                # Show key entities
                key_entities = community.get("key_entities", {})
                
                for entity_type, entities in key_entities.items():
                    if entities:
                        st.write(f"**{entity_type.capitalize()}**: {', '.join(entities)}")
                
                # Show node type distribution
                node_types = community.get("node_types", {})
                if node_types:
                    fig = px.bar(
                        x=list(node_types.keys()),
                        y=list(node_types.values()),
                        title="Node Types in Community"
                    )
                    st.plotly_chart(fig, use_container_width=True)

def render_data_explorer():
    """Render the data explorer section."""
    st.header("Data Explorer")
    
    if st.session_state.current_session_data is None:
        st.info("Please load session data first")
        return
    
    # Session data overview
    st.subheader("Session Data Overview")
    
    df = st.session_state.current_session_data
    
    # Show basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Statements", len(df))
    with col2:
        st.metric("Unique Speakers", df["Speaker"].nunique())
    with col3:
        st.metric("Session Dates", df["Date"].dt.date.nunique())
    
    # Speaker distribution
    st.subheader("Speaker Contribution Analysis")
    
    speaker_counts = df["Speaker"].value_counts().head(10)
    fig = px.bar(
        x=speaker_counts.index,
        y=speaker_counts.values,
        title="Top 10 Speakers by Number of Statements"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Entity analysis
    if st.session_state.entity_map is not None:
        st.subheader("Entity Analysis")
        
        # Group entities by type
        entities_by_type = {}
        
        for _, entities in st.session_state.entity_map.items():
            for entity in entities:
                entity_type = entity.get("label")
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = {}
                
                entity_text = entity.get("text")
                if entity_text not in entities_by_type[entity_type]:
                    entities_by_type[entity_type][entity_text] = 0
                
                entities_by_type[entity_type][entity_text] += 1
        
        # Show top entities by type
        tabs = st.tabs(list(entities_by_type.keys()))
        
        for i, entity_type in enumerate(entities_by_type.keys()):
            with tabs[i]:
                # Sort entities by frequency
                sorted_entities = sorted(
                    entities_by_type[entity_type].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Create DataFrame
                entity_df = pd.DataFrame(
                    sorted_entities[:20],
                    columns=["Entity", "Frequency"]
                )
                
                # Show chart
                fig = px.bar(
                    entity_df,
                    x="Entity",
                    y="Frequency",
                    title=f"Top 20 {entity_type} Entities"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.subheader("Data Preview")
    
    # Add filters
    with st.expander("Filters"):
        # Date filter
        dates = sorted(df["Date"].dt.date.unique())
        selected_date = st.selectbox("Filter by date:", ["All"] + [str(d) for d in dates])
        
        # Speaker filter
        speakers = sorted(df["Speaker"].unique())
        selected_speaker = st.selectbox("Filter by speaker:", ["All"] + list(speakers))
        
        # Text search
        search_text = st.text_input("Search in content:")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_date != "All":
        filtered_df = filtered_df[filtered_df["Date"].dt.date.astype(str) == selected_date]
    
    if selected_speaker != "All":
        filtered_df = filtered_df[filtered_df["Speaker"] == selected_speaker]
    
    if search_text:
        filtered_df = filtered_df[filtered_df["Content"].str.contains(search_text, case=False)]
    
    # Show filtered data
    st.write(f"Showing {len(filtered_df)} of {len(df)} statements")
    st.dataframe(
        filtered_df[["Date", "Speaker", "Content"]],
        use_container_width=True,
        height=400
    )

def render_query_section(settings):
    """Render the query section."""
    st.header("GraphRAG Query")
    
    if not st.session_state.graph_built or st.session_state.graphrag is None:
        st.info("Please build the knowledge graph and initialize GraphRAG first")
        return
    
    # Query input
    query = st.text_input("Enter your query:")
    
    # Examples dropdown
    examples = [
        "What were the main topics discussed in the parliament?",
        "What did the Prime Minister say about healthcare?",
        "What were the interactions between the speakers on education policy?",
        "What legislation was mentioned in recent sessions?",
        "Summarize the discussion about climate change."
    ]
    
    selected_example = st.selectbox("Or select an example query:", [""] + examples)
    if selected_example and not query:
        query = selected_example
    
    # Mode selector
    col1, col2 = st.columns([3, 1])
    with col2:
        mode = st.radio(
            "Query mode:",
            ["hybrid", "graph", "vector"],
            horizontal=True,
            index=["hybrid", "graph", "vector"].index(settings["query_mode"])
        )
    
    # Process query
    if query:
        if st.button("Submit Query"):
            result = process_query(
                query,
                mode=mode,
                max_results=settings["max_results"]
            )
            
            if result:
                # Display answer
                st.subheader("Answer")
                st.write(result["answer"])
                
                # Display query info
                st.subheader("Query Information")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Query Type", result["query_type"])
                with col2:
                    st.metric("Processing Time", f"{result['time_taken']:.2f}s")
                with col3:
                    st.metric("Sources", len(result.get("sources", [])))
                
                # Display sources
                if "sources" in result and result["sources"]:
                    st.subheader("Sources")
                    
                    with st.expander("View Sources", expanded=True):
                        for i, source in enumerate(result["sources"]):
                            st.markdown(
                                f"**{i+1}. {source['speaker']} on {source['date']}:**\n\n"
                                f"{source['content']}\n\n"
                                f"*Relevance: {source['relevance']:.2f} - Source: {source['source_type']}*"
                            )
                            st.divider()
    
    # Query history
    if st.session_state.query_history:
        st.header("Query History")
        
        for i, query_record in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(
                f"{query_record['timestamp']} - '{query_record['query'][:50]}...' ({query_record['mode']} mode)"
            ):
                st.write(f"**Query:** {query_record['query']}")
                st.write(f"**Answer:** {query_record['result']['answer']}")
                st.write(f"**Query Type:** {query_record['result']['query_type']}")
                st.write(f"**Processing Time:** {query_record['time']:.2f}s")
                st.write(f"**Sources:** {len(query_record['result'].get('sources', []))}")

def main():
    """Main application function."""
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["GraphRAG Query", "Knowledge Graph", "Data Explorer"])
    
    with tab1:
        if settings:
            render_query_section(settings)
    
    with tab2:
        render_graph_section()
    
    with tab3:
        render_data_explorer()

if __name__ == "__main__":
    main() 