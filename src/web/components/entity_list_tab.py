"""
Entity List tab component for the Parliamentary Meeting Analyzer application.

This component displays all extracted entities categorized by type and provides
filtering and visualization options.
"""

import time
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter, defaultdict

from src.utils.logging import logger
from src.utils.config import config_manager
from src.utils.graph_visualization import create_plotly_graph

def render_entity_list_tab(data_loader=None, entity_extractor=None, knowledge_graph=None):
    """Render the entity list tab.
    
    Args:
        data_loader: ParliamentaryDataLoader instance
        entity_extractor: EntityExtractor instance
        knowledge_graph: KnowledgeGraph instance
    """
    st.title("Entity Explorer")
    
    if not entity_extractor:
        st.error("Entity extractor not initialized. Please load data first.")
        return
    
    # Check if data is actually loaded rather than just checking for data_loader
    if not hasattr(st.session_state, 'current_session_data') or st.session_state.current_session_data is None:
        st.error("Data not loaded. Please load data from the sidebar first.")
        st.info("Go to the sidebar and click 'Load Data', then select session dates and click 'Load Selected Sessions'.")
        return
    
    # Use the session data directly
    df = st.session_state.current_session_data
    
    with st.spinner("Loading entities..."):
        try:
            # Extract entities using cached results if available
            enhanced_df, entity_map = entity_extractor.extract_entities_from_dataframe(df, use_cache=True)
            
            if not entity_map:
                st.error("No entities found. Try reprocessing the data.")
                return
            
            # Set up entity categories
            entity_types = entity_extractor.entity_types
            
            # Create entity listings by type
            all_entities = []
            for entry_id, entities in entity_map.items():
                for entity in entities:
                    # Add source document reference
                    entity_copy = entity.copy()
                    entity_copy['entry_id'] = entry_id
                    
                    # Add source document info
                    doc_row = df[df['entry_id'].astype(str) == str(entry_id)]
                    if not doc_row.empty:
                        if 'Date' in doc_row.columns:
                            entity_copy['date'] = doc_row['Date'].iloc[0]
                        if 'Speaker' in doc_row.columns:
                            entity_copy['speaker'] = doc_row['Speaker'].iloc[0]
                    
                    all_entities.append(entity_copy)
            
            # Convert to DataFrame for easier filtering
            entities_df = pd.DataFrame(all_entities)
            
            if entities_df.empty:
                st.error("No entities found in the data.")
                return
            
            # Entity statistics
            entity_counts = entities_df['label'].value_counts()
            
            # Display metrics
            st.subheader("Entity Overview")
            
            metrics_cols = st.columns(len(entity_types) + 1)
            
            with metrics_cols[0]:
                st.metric("Total Entities", len(entities_df))
            
            for i, entity_type in enumerate(entity_types):
                with metrics_cols[i+1]:
                    count = entity_counts.get(entity_type, 0)
                    st.metric(entity_type, count)
            
            # Create tabs for different views
            list_tab, graph_tab, stats_tab = st.tabs(["Entity List", "Entity Graph", "Entity Statistics"])
            
            with list_tab:
                # Filter options
                st.subheader("Filter Entities")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    selected_types = st.multiselect(
                        "Entity Types",
                        options=entity_types,
                        default=entity_types,
                        key="entity_types_filter"
                    )
                
                with col2:
                    if 'date' in entities_df.columns:
                        entities_df['date'] = pd.to_datetime(entities_df['date'])
                        min_date = entities_df['date'].min().date()
                        max_date = entities_df['date'].max().date()
                        
                        date_range = st.date_input(
                            "Date Range",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key="entity_date_filter"
                        )
                
                with col3:
                    if 'speaker' in entities_df.columns:
                        speakers = entities_df['speaker'].dropna().unique()
                        selected_speaker = st.selectbox(
                            "Speaker",
                            options=["All"] + list(speakers),
                            key="entity_speaker_filter"
                        )
                
                # Text search
                search_term = st.text_input("Search Entities", key="entity_search_filter")
                
                # Apply filters
                filtered_df = entities_df.copy()
                
                if selected_types:
                    filtered_df = filtered_df[filtered_df['label'].isin(selected_types)]
                
                if 'date' in filtered_df.columns and len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (filtered_df['date'].dt.date >= start_date) & 
                        (filtered_df['date'].dt.date <= end_date)
                    ]
                
                if 'speaker' in filtered_df.columns and selected_speaker != "All":
                    filtered_df = filtered_df[filtered_df['speaker'] == selected_speaker]
                
                if search_term:
                    filtered_df = filtered_df[
                        filtered_df['text'].str.contains(search_term, case=False, na=False)
                    ]
                
                # Display results
                st.subheader(f"Entities ({len(filtered_df)} results)")
                
                if not filtered_df.empty:
                    # Group by entity type
                    for entity_type in selected_types:
                        type_df = filtered_df[filtered_df['label'] == entity_type]
                        
                        if not type_df.empty:
                            with st.expander(f"{entity_type} ({len(type_df)})", expanded=entity_type == selected_types[0]):
                                # Get unique entities
                                unique_entities = type_df['text'].value_counts().reset_index()
                                unique_entities.columns = ['Entity', 'Count']
                                
                                # Display as table with count
                                st.dataframe(
                                    unique_entities,
                                    column_config={
                                        "Entity": st.column_config.TextColumn("Entity"),
                                        "Count": st.column_config.NumberColumn("Occurrences")
                                    },
                                    use_container_width=True
                                )
                                
                                # Show individual entity occurrences
                                if st.checkbox(f"Show {entity_type} details", key=f"details_{entity_type}"):
                                    st.dataframe(
                                        type_df[['text', 'date', 'speaker', 'entry_id']].rename(
                                            columns={'text': 'Entity', 'entry_id': 'Document ID'}
                                        ),
                                        use_container_width=True
                                    )
                else:
                    st.info("No entities match the current filters.")
            
            with graph_tab:
                st.subheader("Entity Relationship Graph")
                
                # Create graph from entities if knowledge graph is available
                if knowledge_graph and hasattr(knowledge_graph, 'graph'):
                    # Filter options
                    st.markdown("### Graph Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        graph_entity_types = st.multiselect(
                            "Entity Types to Include",
                            options=entity_types,
                            default=entity_types[:min(3, len(entity_types))],  # Default to first 3 types
                            key="graph_entity_filter"
                        )
                    
                    with col2:
                        max_nodes = st.slider(
                            "Maximum Nodes",
                            min_value=10,
                            max_value=100,
                            value=50,
                            step=10,
                            key="graph_max_nodes"
                        )
                    
                    # Filter graph by entity types
                    filtered_graph = knowledge_graph.get_filtered_graph(node_types=graph_entity_types, max_nodes=max_nodes)
                    
                    if filtered_graph.number_of_nodes() > 0:
                        # Create interactive Plotly visualization
                        fig = create_plotly_graph(filtered_graph)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display graph statistics
                        st.markdown("### Graph Statistics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Nodes", filtered_graph.number_of_nodes())
                        
                        with col2:
                            st.metric("Edges", filtered_graph.number_of_edges())
                        
                        with col3:
                            # Calculate connected components
                            connected_components = nx.number_connected_components(filtered_graph.to_undirected())
                            st.metric("Connected Components", connected_components)
                    else:
                        st.info("No graph data available for the selected entity types.")
                else:
                    st.info("Knowledge graph not available. Please load data and extract entities first.")
            
            with stats_tab:
                st.subheader("Entity Statistics")
                
                # Create summary statistics
                entity_frequency = entities_df['text'].value_counts().reset_index()
                entity_frequency.columns = ['Entity', 'Frequency']
                entity_frequency = entity_frequency.head(20)  # Top 20 entities
                
                # Entity frequency chart
                st.markdown("### Top 20 Entities by Frequency")
                fig = go.Figure(go.Bar(
                    x=entity_frequency['Entity'],
                    y=entity_frequency['Frequency'],
                    marker_color='rgb(26, 118, 255)'
                ))
                fig.update_layout(
                    xaxis_tickangle=-45,
                    xaxis_title="Entity",
                    yaxis_title="Frequency",
                    margin=dict(l=20, r=20, t=30, b=70)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Entity type distribution
                st.markdown("### Entity Type Distribution")
                fig = go.Figure(go.Pie(
                    labels=entity_counts.index,
                    values=entity_counts.values,
                    hole=.3
                ))
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Entity by date if available
                if 'date' in entities_df.columns:
                    st.markdown("### Entity Extraction Over Time")
                    
                    # Group by date and entity type
                    entities_df['date'] = pd.to_datetime(entities_df['date'])
                    time_data = entities_df.groupby([
                        entities_df['date'].dt.date, 'label'
                    ]).size().reset_index()
                    time_data.columns = ['Date', 'Entity Type', 'Count']
                    
                    # Create line chart
                    fig = px.line(
                        time_data, 
                        x='Date', 
                        y='Count', 
                        color='Entity Type',
                        title='Entity Types Extracted Over Time'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Co-occurrence matrix for top entities
                st.markdown("### Entity Co-occurrence")
                
                # Calculate co-occurrence data
                if 'entry_id' in entities_df.columns:
                    co_occurrence = calculate_co_occurrence(entities_df, top_n=10)
                    
                    if co_occurrence is not None:
                        fig = go.Figure(data=go.Heatmap(
                            z=co_occurrence.values,
                            x=co_occurrence.columns,
                            y=co_occurrence.index,
                            colorscale='Viridis'
                        ))
                        
                        fig.update_layout(
                            title="Top 10 Entities Co-occurrence",
                            xaxis_title="Entity",
                            yaxis_title="Entity",
                            margin=dict(l=50, r=50, t=50, b=50)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data to generate co-occurrence matrix.")
        
        except Exception as e:
            logger.error(f"Error in Entity List tab: {str(e)}")
            st.error(f"An error occurred while processing entities: {str(e)}")

def calculate_co_occurrence(entities_df, top_n=10):
    """Calculate entity co-occurrence matrix.
    
    Args:
        entities_df: DataFrame containing entity data
        top_n: Number of top entities to include
        
    Returns:
        DataFrame with co-occurrence matrix or None if not enough data
    """
    try:
        # Get top entities
        top_entities = entities_df['text'].value_counts().head(top_n).index.tolist()
        
        if len(top_entities) < 2:
            return None
        
        # Initialize co-occurrence matrix
        co_matrix = pd.DataFrame(
            0, 
            index=top_entities,
            columns=top_entities
        )
        
        # Calculate co-occurrences
        for entry_id in entities_df['entry_id'].unique():
            entry_entities = entities_df[entities_df['entry_id'] == entry_id]['text'].tolist()
            entry_entities = [e for e in entry_entities if e in top_entities]
            
            # Update co-occurrence counts
            for i, entity1 in enumerate(entry_entities):
                for entity2 in entry_entities[i:]:
                    co_matrix.loc[entity1, entity2] += 1
                    co_matrix.loc[entity2, entity1] += 1
        
        # Set diagonal to zero (don't count self-co-occurrence)
        for entity in top_entities:
            co_matrix.loc[entity, entity] = 0
            
        return co_matrix
    
    except Exception as e:
        logger.error(f"Error calculating co-occurrence: {str(e)}")
        return None

if __name__ == "__main__":
    st.set_page_config(
        page_title="Parliamentary Meeting Analyzer - Entity List",
        page_icon="📝",
        layout="wide"
    )
    render_entity_list_tab() 