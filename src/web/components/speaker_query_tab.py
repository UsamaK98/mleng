"""
Speaker Query tab component for the Parliamentary Meeting Analyzer application.

This component allows users to search for statements by specific speakers 
and explore connections between speakers and topics.
"""

import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import networkx as nx
from collections import Counter

from src.utils.logging import logger
from src.utils.config import config_manager
from src.models.graphrag import GraphRAG
from src.utils.graph_visualization import create_plotly_graph

def render_speaker_query_tab(
    data_loader=None,
    ollama_service=None,
    vector_store=None,
    knowledge_graph=None,
    graphrag=None
):
    """Render the speaker query tab.
    
    Args:
        data_loader: ParliamentaryDataLoader instance
        ollama_service: OllamaService instance
        vector_store: VectorStore instance
        knowledge_graph: KnowledgeGraph instance
        graphrag: GraphRAG instance
    """
    st.title("Speaker Query")
    
    # Check if data is actually loaded rather than just checking for data_loader
    if not hasattr(st.session_state, 'current_session_data') or st.session_state.current_session_data is None:
        st.error("Data not loaded. Please load data from the sidebar first.")
        st.info("Go to the sidebar and click 'Load Data', then select session dates and click 'Load Selected Sessions'.")
        return
    
    if not vector_store:
        st.error("Vector store not initialized. Please load data first.")
        return
    
    # Create GraphRAG instance if not provided
    if not graphrag and knowledge_graph and ollama_service and vector_store:
        try:
            # Use the existing components to create a GraphRAG instance
            graphrag = GraphRAG(
                kg=knowledge_graph,
                ollama_service=ollama_service,
                vector_store=vector_store
            )
        except Exception as e:
            st.error(f"Error initializing GraphRAG: {str(e)}")
            # Continue without GraphRAG functionality
    
    # Get data from session state directly
    df = st.session_state.current_session_data
    
    # Extract speakers from data
    speakers = sorted(df['Speaker'].dropna().unique())
    
    # Create query interface
    st.subheader("Search Speaker Statements")
    
    # Query type tabs
    query_type = st.radio(
        "Query Type",
        ["Speaker Search", "Topic Search", "Advanced Query"],
        horizontal=True,
        key="query_type"
    )
    
    # Query builder based on type
    if query_type == "Speaker Search":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_speaker = st.selectbox(
                "Select Speaker",
                options=speakers,
                key="selected_speaker"
            )
        
        with col2:
            topic_keywords = st.text_input(
                "Optional Topic Keywords",
                placeholder="e.g., climate change, economy, healthcare",
                key="topic_keywords"
            )
        
        # Build query
        if selected_speaker:
            if topic_keywords:
                query = f"What did {selected_speaker} say about {topic_keywords}?"
            else:
                query = f"What are the key topics discussed by {selected_speaker}?"
                
    elif query_type == "Topic Search":
        topic = st.text_input(
            "Enter Topic",
            placeholder="e.g., climate change, economy, healthcare",
            key="topic_search"
        )
        
        # Multiple speaker selection
        selected_speakers = st.multiselect(
            "Filter by Speakers (optional)",
            options=speakers,
            key="topic_speakers"
        )
        
        # Build query
        if topic:
            if selected_speakers:
                speakers_list = ", ".join(selected_speakers)
                query = f"What did {speakers_list} say about {topic}?"
            else:
                query = f"What was discussed about {topic}?"
        else:
            query = ""
            
    else:  # Advanced Query
        query = st.text_area(
            "Enter Your Query",
            placeholder="e.g., Compare the positions of Speaker A and Speaker B on climate change.",
            key="advanced_query"
        )
        
        # Show examples
        with st.expander("Query Examples"):
            st.markdown("""
            * What did Speaker X say about topic Y?
            * Compare Speaker A and Speaker B's positions on topic Z.
            * Find statements by Speaker X that mention Speaker Y.
            * What are the most common topics discussed by Speaker X?
            * When did Speaker X first mention topic Y?
            """)
    
    # Query execution
    execute_button = st.button("Search", key="execute_query", type="primary")
    
    if execute_button and query:
        with st.spinner(f"Processing query: {query}"):
            try:
                # Start time for performance tracking
                start_time = time.time()
                
                # Execute query
                if graphrag:
                    # Use GraphRAG for advanced query processing
                    results = graphrag.query(
                        query_text=query,
                        mode="hybrid",
                        max_results=20,
                        include_sources=True
                    )
                    
                    # Display results
                    display_graphrag_results(results)
                    
                else:
                    # Fallback to basic vector search if GraphRAG not available
                    if query_type == "Speaker Search" and selected_speaker:
                        # Filter by speaker
                        filter_dict = {"Speaker": selected_speaker}
                        
                        # Add topic filtering if provided
                        if topic_keywords:
                            # Search with topic keywords
                            search_results = vector_store.search_similar(
                                query=topic_keywords,
                                top_k=20,
                                filter_dict=filter_dict
                            )
                        else:
                            # Get all statements from the speaker
                            matching_entries = df[df['Speaker'] == selected_speaker]['entry_id'].tolist()
                            search_results = []
                            
                            # Get statements by ID
                            for entry_id in matching_entries[:20]:  # Limit to first 20
                                result = vector_store.get_by_id(str(entry_id))
                                if result:
                                    result['score'] = 1.0  # Assign full score for direct matches
                                    search_results.append(result)
                    
                    elif query_type == "Topic Search" and topic:
                        # Filter by speakers if provided
                        filter_dict = None
                        if selected_speakers:
                            filter_dict = {"Speaker": selected_speakers}
                        
                        # Search by topic
                        search_results = vector_store.search_similar(
                            query=topic,
                            top_k=20,
                            filter_dict=filter_dict
                        )
                    
                    else:  # Advanced Query
                        # Parse the query to detect speakers and topics
                        mentioned_speakers = [s for s in speakers if s.lower() in query.lower()]
                        
                        filter_dict = None
                        if mentioned_speakers:
                            filter_dict = {"Speaker": mentioned_speakers}
                        
                        # Execute the search
                        search_results = vector_store.search_similar(
                            query=query,
                            top_k=20,
                            filter_dict=filter_dict
                        )
                    
                    # Display basic search results
                    display_search_results(search_results, query, time.time() - start_time)
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error(f"An error occurred while processing your query: {str(e)}")
    
    # Speaker analytics section
    st.subheader("Speaker Analytics")
    
    with st.expander("Speaker Metrics"):
        # Calculate speaker metrics
        speaker_counts = df['Speaker'].value_counts()
        
        # Display top speakers
        st.markdown("### Most Active Speakers")
        
        # Create bar chart
        fig = px.bar(
            speaker_counts.head(10).reset_index(),
            x='Speaker',
            y='count',
            labels={'count': 'Number of Statements', 'Speaker': 'Speaker Name'},
            title='Top 10 Most Active Speakers'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Speaker timeline if date information is available
        if 'Date' in df.columns:
            st.markdown("### Speaker Activity Timeline")
            
            # Convert to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Group by date and speaker
            timeline_data = df.groupby([df['Date'].dt.date, 'Speaker']).size().reset_index()
            timeline_data.columns = ['Date', 'Speaker', 'Count']
            
            # Get top 5 speakers
            top_speakers = speaker_counts.head(5).index.tolist()
            filtered_timeline = timeline_data[timeline_data['Speaker'].isin(top_speakers)]
            
            # Create line chart
            fig = px.line(
                filtered_timeline,
                x='Date',
                y='Count',
                color='Speaker',
                title='Speaker Activity Over Time (Top 5 Speakers)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Speaker network if knowledge graph is available
        if knowledge_graph and hasattr(knowledge_graph, 'graph'):
            st.markdown("### Speaker Relationship Network")
            
            # Extract speaker nodes and their connections
            speaker_nodes = [
                node for node, attrs in knowledge_graph.graph.nodes(data=True)
                if attrs.get('type') == 'person'
            ]
            
            # Create subgraph of speakers
            speaker_graph = nx.Graph()
            
            for speaker in speaker_nodes:
                neighbors = list(knowledge_graph.graph.neighbors(speaker))
                
                # Look for connections between speakers through shared topics/entities
                for neighbor in neighbors:
                    neighbor_type = knowledge_graph.graph.nodes[neighbor].get('type')
                    
                    if neighbor_type != 'person':  # If not a person, it's a topic/entity
                        # Find other speakers connected to this topic/entity
                        for second_neighbor in knowledge_graph.graph.neighbors(neighbor):
                            if (second_neighbor != speaker and 
                                knowledge_graph.graph.nodes[second_neighbor].get('type') == 'person'):
                                
                                # Add edge between speakers
                                if not speaker_graph.has_edge(speaker, second_neighbor):
                                    speaker_graph.add_edge(
                                        speaker, 
                                        second_neighbor, 
                                        weight=1,
                                        topic=knowledge_graph.graph.nodes[neighbor].get('label', 'unknown')
                                    )
                                else:
                                    # Increment weight if edge exists
                                    speaker_graph[speaker][second_neighbor]['weight'] += 1
            
            # Create visualization if graph is not empty
            if speaker_graph.number_of_nodes() > 0:
                # Limit to largest connected component if graph is too big
                if speaker_graph.number_of_nodes() > 20:
                    largest_cc = max(nx.connected_components(speaker_graph), key=len)
                    speaker_graph = speaker_graph.subgraph(largest_cc).copy()
                
                # Create network graph using custom visualization
                if 'create_plotly_graph' in globals():
                    fig = create_plotly_graph(speaker_graph)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Basic matplotlib fallback
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Draw using spring layout
                    pos = nx.spring_layout(speaker_graph)
                    
                    # Draw nodes
                    nx.draw_networkx_nodes(
                        speaker_graph, 
                        pos, 
                        node_color='skyblue',
                        node_size=100
                    )
                    
                    # Draw edges with weights
                    edge_weights = [speaker_graph[u][v]['weight'] for u, v in speaker_graph.edges()]
                    
                    nx.draw_networkx_edges(
                        speaker_graph, 
                        pos, 
                        width=[w/max(edge_weights)*5 for w in edge_weights],
                        alpha=0.7
                    )
                    
                    # Draw labels
                    nx.draw_networkx_labels(
                        speaker_graph, 
                        pos, 
                        font_size=8
                    )
                    
                    plt.axis('off')
                    st.pyplot(fig)
            else:
                st.info("No speaker relationships found in the knowledge graph.")

def display_search_results(results, query, time_taken):
    """Display basic search results from vector store.
    
    Args:
        results: List of search results
        query: Original query string
        time_taken: Query execution time in seconds
    """
    if not results:
        st.warning("No results found for your query.")
        return
    
    # Display summary
    st.markdown(f"### Results for: {query}")
    st.markdown(f"Found {len(results)} matching statements in {time_taken:.2f} seconds")
    
    # Display results in cards
    for i, result in enumerate(results):
        with st.container(border=True):
            # Extract metadata
            speaker = result.get('Speaker', 'Unknown Speaker')
            date = result.get('Date', 'Unknown Date')
            content = result.get('Content', '')
            score = result.get('score', 0.0)
            
            # Header with speaker and date
            col1, col2, col3 = st.columns([3, 6, 1])
            
            with col1:
                st.markdown(f"**Speaker:** {speaker}")
            
            with col2:
                st.markdown(f"**Date:** {date}")
            
            with col3:
                st.markdown(f"**Score:** {score:.2f}")
            
            # Main content
            st.markdown(content)
            
            # Expand for metadata
            with st.expander("Details"):
                # Remove large content field for cleaner display
                metadata = {k: v for k, v in result.items() if k != 'Content'}
                st.json(metadata)

def display_graphrag_results(results):
    """Display GraphRAG query results.
    
    Args:
        results: GraphRAG query results dictionary
    """
    # Extract key information
    query = results.get('query', '')
    answer = results.get('answer', 'No answer generated.')
    query_type = results.get('query_type', 'general')
    evidence = results.get('evidence', [])
    time_taken = results.get('time_taken', 0)
    
    # Display query info and answer
    st.markdown(f"### Query Analysis")
    
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        st.markdown(f"**Query:** {query}")
    
    with col2:
        st.markdown(f"**Type:** {query_type}")
    
    with col3:
        st.markdown(f"**Time:** {time_taken:.2f}s")
    
    # Display answer
    st.markdown("### Answer")
    st.markdown(answer)
    
    # Display supporting evidence
    if evidence:
        st.markdown("### Supporting Evidence")
        
        # Create tabs for different evidence views
        list_tab, graph_tab = st.tabs(["List View", "Network View"])
        
        with list_tab:
            for i, item in enumerate(evidence):
                with st.container(border=True):
                    # Extract information
                    speaker = item.get('speaker', {}).get('label', 'Unknown Speaker')
                    content = item.get('statement', {}).get('content', '')
                    date = item.get('statement', {}).get('date', 'Unknown Date')
                    relevance = item.get('relevance', 0.0)
                    source = item.get('source', 'unknown')
                    
                    # Header with speaker and date
                    col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
                    
                    with col1:
                        st.markdown(f"**Speaker:** {speaker}")
                    
                    with col2:
                        st.markdown(f"**Date:** {date}")
                    
                    with col3:
                        st.markdown(f"**Relevance:** {relevance:.2f}")
                    
                    with col4:
                        st.markdown(f"**Source:** {source}")
                    
                    # Main content
                    st.markdown(content)
        
        with graph_tab:
            # If network data available, display it
            if 'network' in results:
                st.markdown("Network visualization not implemented yet.")
            else:
                st.info("No network data available for this query.")
    else:
        st.info("No supporting evidence available for this query.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Parliamentary Meeting Analyzer - Speaker Query",
        page_icon="🔍",
        layout="wide"
    )
    render_speaker_query_tab() 