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
from src.utils.cache_manager import cache_manager

# Import tab components
from src.web.components.home_tab import render_home_tab
from src.web.components.entity_list_tab import render_entity_list_tab
from src.web.components.speaker_query_tab import render_speaker_query_tab

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
    
    # Check if entities are already loaded from cache
    if st.session_state.entity_map is not None:
        logger.info("Entities already extracted, skipping extraction")
        return True
    
    with st.spinner("Extracting entities using GLiNER..."):
        # Check if cache exists and is valid
        cache_status = cache_manager.get_cache_status()
        # Check for cache existence, validity is optional
        if cache_status["ner_cache"]["exists"]:
            valid = cache_status["ner_cache"].get("valid", False)
            if valid:
                logger.info("Using valid cached entities")
            else:
                logger.info("Cache exists but may not be validated, using anyway")
        
        # Extract entities (will use cache if available)
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
    
    # Check if vector store already has data
    vector_info = st.session_state.vector_store.get_collection_info()
    if vector_info.get('points_count', 0) > 0 or vector_info.get('count', 0) > 0:
        logger.info("Vector store already contains data, skipping build")
        return True
    
    # Check if cache exists and is valid
    cache_status = cache_manager.get_cache_status()
    if cache_status["vector_cache"]["exists"]:
        valid = cache_status["vector_cache"].get("valid", False)
        if valid:
            logger.info("Using valid cached vector embeddings")
        else:
            logger.info("Vector cache exists but may not be validated, using anyway")
    
    with st.spinner("Building vector store..."):
        # Store data in vector store
        success = st.session_state.vector_store.store_parliamentary_data(st.session_state.current_session_data)
        
        if not success:
            st.error("Failed to build vector store")
            return False
        
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
    """Render the graph visualization section."""
    st.header("Knowledge Graph")
    
    if not st.session_state.graph_built:
        st.info("Please build the knowledge graph first")
        return
    
    # Get graph statistics
    stats = st.session_state.graph_stats
    
    # Display graph statistics
    st.subheader("Graph Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", stats.get("num_nodes", 0))
    with col2:
        st.metric("Edges", stats.get("num_edges", 0))
    with col3:
        st.metric("Communities", stats.get("num_communities", 0))
    
    # Node type distribution
    st.subheader("Node Type Distribution")
    
    # Extract node types and counts
    node_types = stats.get("node_types", {})
    if node_types:
        fig = px.bar(
            x=list(node_types.keys()),
            y=list(node_types.values()),
            title="Node Types"
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
            # Pass the actual graph object instead of the knowledge_graph object
            graph_obj = st.session_state.knowledge_graph.graph
            fig = create_plotly_graph(
                graph=graph_obj, 
                title="Parliamentary Knowledge Graph"
            )
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
    """Render data explorer section."""
    st.header("Data Explorer")
    
    if not st.session_state.available_dates:
        st.warning("No data loaded yet")
        return
    
    # Display session date selector
    st.subheader("Session Data")
    
    # Display the current data frame
    if st.session_state.current_session_data is not None:
        st.subheader("Current Session Data")
        st.dataframe(st.session_state.current_session_data, height=400)
        
        # Show speaker stats
        st.subheader("Speaker Statistics")
        speaker_counts = st.session_state.current_session_data["Speaker"].value_counts().reset_index()
        speaker_counts.columns = ["Speaker", "Contributions"]
        
        st.bar_chart(speaker_counts.set_index("Speaker"))
    else:
        st.info("Select dates to view session data")

def auto_initialize():
    """Automatically initialize all components on startup."""
    # Initialize basic services
    if not st.session_state.initialized:
        initialize_services()
        st.experimental_rerun()
    
    # Load data if not already loaded
    if not st.session_state.available_dates and st.session_state.initialized:
        success = load_data()
        if success:
            st.experimental_rerun()
    
    # If we have data but no session data loaded, load the selected dates
    if (len(st.session_state.available_dates) > 0 and 
        st.session_state.current_session_data is None and
        not st.session_state.selected_dates):
        
        # Set default selection (first few dates)
        num_dates_to_load = min(3, len(st.session_state.available_dates))
        st.session_state.selected_dates = st.session_state.available_dates[:num_dates_to_load]
        
        # Load selected sessions
        success = load_selected_sessions()
        if success:
            st.experimental_rerun()
    
    # Check for entity cache and load from cache if available
    if (st.session_state.current_session_data is not None and 
        st.session_state.entity_map is None):
        
        # Check if we should use cache
        cache_status = cache_manager.get_cache_status()
        if cache_status["ner_cache"]["exists"]:
            logger.info("Entity cache exists, using cached entities")
            # Extract entities will load from cache if available
            success = extract_entities()
            if success:
                st.experimental_rerun()
    
    # Build knowledge graph if entities are extracted but graph not built
    if (st.session_state.entity_map is not None and 
        not st.session_state.graph_built):
        
        # Check if we should use cache
        cache_status = cache_manager.get_cache_status()
        if cache_status["graph_cache"]["exists"]:
            valid = cache_status["graph_cache"].get("valid", False)
            if valid:
                logger.info("Using valid graph cache")
            else:
                logger.info("Graph cache exists but may not be validated, using anyway")
        
        success = build_knowledge_graph()
        if success:
            st.experimental_rerun()
    
    # Initialize vector store if graph is built
    if (st.session_state.graph_built and 
        st.session_state.vector_store is not None and
        st.session_state.current_session_data is not None):
        
        # Check if vector store has data
        vector_info = st.session_state.vector_store.get_collection_info()
        if vector_info.get('points_count', 0) == 0 and vector_info.get('count', 0) == 0:
            # Vector store is empty, build it
            success = build_vector_store()
            if success:
                st.experimental_rerun()
    
    # Initialize GraphRAG if needed
    if (st.session_state.graph_built and 
        st.session_state.vector_store is not None and
        st.session_state.graphrag is None):
        
        success = initialize_graphrag()
        if success:
            st.experimental_rerun()

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
            index=["hybrid", "graph", "vector"].index(settings["query_mode"]),
            key="query_section_mode"
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
    # Auto-initialize components
    auto_initialize()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Home", 
        "Entity List", 
        "Speaker Query", 
        "GraphRAG Query", 
        "Knowledge Graph"
    ])
    
    with tab1:
        # Home tab
        render_home_tab(
            data_loader=st.session_state.data_loader,
            entity_extractor=st.session_state.entity_extractor,
            vector_store=st.session_state.vector_store
        )
    
    with tab2:
        # Entity List tab
        render_entity_list_tab(
            data_loader=st.session_state.data_loader,
            entity_extractor=st.session_state.entity_extractor,
            knowledge_graph=st.session_state.knowledge_graph
        )
    
    with tab3:
        # Speaker Query tab
        render_speaker_query_tab(
            data_loader=st.session_state.data_loader,
            ollama_service=st.session_state.ollama_service,
            vector_store=st.session_state.vector_store,
            knowledge_graph=st.session_state.knowledge_graph,
            graphrag=st.session_state.graphrag
        )
    
    with tab4:
        # GraphRAG Query tab (existing)
        if settings:
            render_query_section(settings)
    
    with tab5:
        # Knowledge Graph tab (existing)
        render_graph_section()

if __name__ == "__main__":
    main() 