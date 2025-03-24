"""
Home tab component for the Parliamentary Meeting Analyzer application.

This component provides an overview of the application, dataset statistics,
and explanations of the different features.
"""

import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

from src.utils.logging import logger
from src.utils.cache_manager import cache_manager
from src.utils.config import config_manager

def render_home_tab(data_loader=None, entity_extractor=None, vector_store=None):
    """Render the home tab.
    
    Args:
        data_loader: ParliamentaryDataLoader instance
        entity_extractor: EntityExtractor instance
        vector_store: VectorStore instance
    """
    st.title("Parliamentary Meeting Minutes Analysis")
    
    # Application description
    st.markdown("""
    ## Welcome to the Parliamentary Meeting Minutes Analysis Tool
    
    This application uses advanced AI techniques to analyze parliamentary meeting minutes,
    extract insights, and allow you to explore the data in various ways. The system combines 
    knowledge graph technology with vector embeddings (GraphRAG) to provide accurate and
    contextual responses to your queries.
    """)
    
    # Create columns for the features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Features
        
        - **Query Analysis**: Ask questions about the parliamentary data using natural language
        - **Entity Extraction**: Identify people, organizations, topics, and other entities
        - **Knowledge Graph**: Visualize relationships between entities
        - **Speaker Analysis**: Explore what specific speakers discussed
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Technologies
        
        - **GLiNER**: For named entity recognition
        - **GraphRAG**: Combining knowledge graphs with semantic search
        - **Ollama**: For embeddings and language model responses
        - **NetworkX**: For graph representation and analysis
        """)
    
    # Dataset statistics
    st.markdown("## Dataset Overview")
    
    # Check for data in session state first, then fall back to data_loader
    if hasattr(st.session_state, 'current_session_data') and st.session_state.current_session_data is not None:
        df = st.session_state.current_session_data
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents", len(df))
        
        with col2:
            num_speakers = df['Speaker'].nunique() if 'Speaker' in df.columns else 0
            st.metric("Speakers", num_speakers)
        
        with col3:
            if 'Date' in df.columns:
                dates = pd.to_datetime(df['Date']).dt.date.nunique()
                st.metric("Meeting Dates", dates)
            else:
                st.metric("Meeting Dates", 0)
        
        with col4:
            if entity_extractor:
                # Get entity counts from cache if possible
                entity_count = 0
                cache_status = cache_manager.get_cache_status()
                if cache_status["ner_cache"]["exists"]:
                    entity_count = cache_status["ner_cache"].get("entity_count", 0)
                
                if entity_count == 0 and hasattr(entity_extractor, 'extract_entities_from_dataframe'):
                    # Count entities from data if cache doesn't have the count
                    try:
                        _, entity_map = entity_extractor.extract_entities_from_dataframe(df, use_cache=True)
                        entity_count = sum(len(entities) for entities in entity_map.values())
                    except Exception as e:
                        logger.error(f"Error counting entities: {str(e)}")
                
                st.metric("Extracted Entities", entity_count)
            else:
                st.metric("Extracted Entities", 0)
        
        # Show speaker distribution
        if 'Speaker' in df.columns:
            st.subheader("Speaker Distribution")
            
            speaker_counts = df['Speaker'].value_counts().reset_index()
            speaker_counts.columns = ['Speaker', 'Count']
            
            # Limit to top 10 speakers for display
            top_speakers = speaker_counts.head(10)
            
            fig = px.bar(
                top_speakers, 
                x='Speaker', 
                y='Count',
                title='Top 10 Speakers by Number of Contributions',
                color='Count',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show date distribution if available
        if 'Date' in df.columns:
            st.subheader("Timeline of Meetings")
            
            df['Date'] = pd.to_datetime(df['Date'])
            date_counts = df.groupby(df['Date'].dt.date).size().reset_index()
            date_counts.columns = ['Date', 'Count']
            
            fig = px.line(
                date_counts, 
                x='Date', 
                y='Count',
                title='Number of Contributions Over Time',
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif data_loader and hasattr(data_loader, 'df') and data_loader.df is not None:
        # Fallback to data_loader if session state doesn't have data
        df = data_loader.df
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents", len(df))
        
        with col2:
            num_speakers = df['Speaker'].nunique() if 'Speaker' in df.columns else 0
            st.metric("Speakers", num_speakers)
        
        with col3:
            if 'Date' in df.columns:
                dates = pd.to_datetime(df['Date']).dt.date.nunique()
                st.metric("Meeting Dates", dates)
            else:
                st.metric("Meeting Dates", 0)
        
        with col4:
            if entity_extractor:
                # Get entity counts from cache if possible
                entity_count = 0
                cache_status = cache_manager.get_cache_status()
                if cache_status["ner_cache"]["exists"]:
                    entity_count = cache_status["ner_cache"].get("entity_count", 0)
                
                if entity_count == 0 and hasattr(entity_extractor, 'extract_entities_from_dataframe'):
                    # Count entities from data if cache doesn't have the count
                    try:
                        _, entity_map = entity_extractor.extract_entities_from_dataframe(df, use_cache=True)
                        entity_count = sum(len(entities) for entities in entity_map.values())
                    except Exception as e:
                        logger.error(f"Error counting entities: {str(e)}")
                
                st.metric("Extracted Entities", entity_count)
            else:
                st.metric("Extracted Entities", 0)
        
        # Show speaker distribution
        if 'Speaker' in df.columns:
            st.subheader("Speaker Distribution")
            
            speaker_counts = df['Speaker'].value_counts().reset_index()
            speaker_counts.columns = ['Speaker', 'Count']
            
            # Limit to top 10 speakers for display
            top_speakers = speaker_counts.head(10)
            
            fig = px.bar(
                top_speakers, 
                x='Speaker', 
                y='Count',
                title='Top 10 Speakers by Number of Contributions',
                color='Count',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show date distribution if available
        if 'Date' in df.columns:
            st.subheader("Timeline of Meetings")
            
            df['Date'] = pd.to_datetime(df['Date'])
            date_counts = df.groupby(df['Date'].dt.date).size().reset_index()
            date_counts.columns = ['Date', 'Count']
            
            fig = px.line(
                date_counts, 
                x='Date', 
                y='Count',
                title='Number of Contributions Over Time',
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Dataset statistics will appear here once data is loaded.")
    
    # Cache status
    st.markdown("## System Status")
    
    cache_status = cache_manager.get_cache_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ner_status = "✅ Available" if cache_status["ner_cache"]["exists"] else "❌ Not Available"
        st.markdown(f"**Entity Cache**: {ner_status}")
        if cache_status["ner_cache"]["exists"]:
            st.markdown(f"Size: {cache_status['ner_cache']['size_mb']:.2f} MB")
    
    with col2:
        vector_status = "✅ Available" if cache_status["vector_cache"]["exists"] else "❌ Not Available"
        st.markdown(f"**Vector Cache**: {vector_status}")
        if cache_status["vector_cache"]["exists"]:
            st.markdown(f"Size: {cache_status['vector_cache']['size_mb']:.2f} MB")
    
    with col3:
        graph_status = "✅ Available" if cache_status["graph_cache"]["exists"] else "❌ Not Available"
        st.markdown(f"**Graph Cache**: {graph_status}")
        if cache_status["graph_cache"]["exists"]:
            st.markdown(f"Size: {cache_status['graph_cache']['size_mb']:.2f} MB")
    
    with col4:
        graphrag_status = "✅ Available" if cache_status["graphrag_cache"]["exists"] else "❌ Not Available"
        st.markdown(f"**GraphRAG Cache**: {graphrag_status}")
        if cache_status["graphrag_cache"]["exists"]:
            st.markdown(f"Size: {cache_status['graphrag_cache']['size_mb']:.2f} MB")
    
    # System info
    if st.checkbox("Show System Information"):
        st.subheader("System Information")
        
        # Check if vector store service is running
        vector_store_status = "✅ Connected" if vector_store and hasattr(vector_store, 'client') else "❌ Not Connected"
        vector_store_type = "Qdrant"
        if vector_store and hasattr(vector_store, 'using_fallback') and vector_store.using_fallback:
            vector_store_type = "ChromaDB (Fallback)"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Vector Store**: {vector_store_status}")
            st.markdown(f"**Vector Store Type**: {vector_store_type}")
            st.markdown(f"**Embedding Model**: {config_manager.config.ollama.embed_model}")
            st.markdown(f"**NER Model**: {config_manager.config.gliner.model_name}")
        
        with col2:
            st.markdown(f"**Cache Last Checked**: {cache_status.get('last_checked', 'Unknown')}")
            st.markdown(f"**Total Cache Size**: {cache_status.get('total_cache_size_mb', 0):.2f} MB")
            st.markdown(f"**Configuration File**: {'config.json' if Path('config.json').exists() else 'Default'}")
    
    # Navigation help
    st.markdown("## Navigation")
    
    st.markdown("""
    Use the tabs above to navigate to different features of the application:
    
    - **Home**: This page - overview and status
    - **Query**: Ask questions about the parliamentary data
    - **Entity List**: View all extracted entities categorized by type
    - **Speaker Query**: Chat with specific speaker's content
    - **Statistics**: Explore data analytics and visualizations
    """)
    
    # Help section
    with st.expander("Need help?"):
        st.markdown("""
        ### Quick Tips
        
        1. **First time use**: The system will extract entities and build indexes on first run, which may take some time.
        2. **Query examples**: Try questions like "What did Speaker X say about topic Y?" or "Summarize the discussion about healthcare."
        3. **Entity exploration**: Use the Entity List tab to browse all extracted entities and their relationships.
        4. **Speaker focus**: The Speaker Query tab allows you to interact with content from specific speakers.
        """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Parliamentary Meeting Analyzer - Home",
        page_icon="📝",
        layout="wide"
    )
    render_home_tab() 