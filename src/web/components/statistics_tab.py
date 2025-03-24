"""
Statistics tab component for the Parliamentary Meeting Analyzer application.

This component displays statistical analysis of parliamentary data, including
speaker participation metrics, topic frequency analysis, and entity co-occurrence patterns.
"""

import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit compatibility

from src.utils.logging import logger
from src.utils.config import config_manager

def render_statistics_tab(data_loader=None, entity_extractor=None, knowledge_graph=None):
    """Render the statistics tab.
    
    Args:
        data_loader: ParliamentaryDataLoader instance
        entity_extractor: EntityExtractor instance
        knowledge_graph: KnowledgeGraph instance
    """
    st.title("Statistical Analysis Dashboard")
    
    # Check if data is actually loaded
    if not hasattr(st.session_state, 'current_session_data') or st.session_state.current_session_data is None:
        st.error("Data not loaded. Please load data from the sidebar first.")
        st.info("Go to the sidebar and click 'Load Data', then select session dates and click 'Load Selected Sessions'.")
        return
    
    # Use the session data directly
    df = st.session_state.current_session_data
    
    # Create tabs for different analyses
    speaker_tab, topic_tab, entity_tab = st.tabs(["Speaker Analysis", "Topic Analysis", "Entity Co-occurrence"])
    
    with speaker_tab:
        render_speaker_statistics(df, entity_extractor)
    
    with topic_tab:
        render_topic_analysis(df, entity_extractor)
    
    with entity_tab:
        render_entity_cooccurrence(df, entity_extractor, knowledge_graph)

def render_speaker_statistics(df, entity_extractor):
    """Render speaker statistics analysis.
    
    Args:
        df: DataFrame containing parliamentary data
        entity_extractor: EntityExtractor instance
    """
    st.subheader("Speaker Participation Analysis")
    
    try:
        # Create filters for time period
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = df[
                    (df['Date'].dt.date >= start_date) & 
                    (df['Date'].dt.date <= end_date)
                ]
            else:
                filtered_df = df
        else:
            filtered_df = df
            st.warning("No date column found in data. Showing all records.")
        
        # Calculate speaker statistics
        if 'Speaker' in filtered_df.columns and 'Content' in filtered_df.columns:
            # Count speeches per speaker
            speaker_counts = filtered_df['Speaker'].value_counts()
            
            # Estimate speaking time (based on content length)
            speaker_content_length = filtered_df.groupby('Speaker')['Content'].apply(
                lambda x: sum(len(text) for text in x)
            )
            
            # Combine statistics
            speaker_stats = pd.DataFrame({
                'Speech Count': speaker_counts,
                'Content Length': speaker_content_length
            }).reset_index()
            
            speaker_stats = speaker_stats.sort_values('Speech Count', ascending=False)
            
            # Show top 10 speakers by default
            top_n = st.slider("Number of speakers to show", 5, 30, 10)
            top_speakers = speaker_stats.head(top_n)
            
            # Display metrics for top speakers
            st.subheader(f"Top {top_n} Speakers")
            
            # Create bar chart for speech count
            fig1 = px.bar(
                top_speakers,
                x='Speaker',
                y='Speech Count',
                title=f"Number of Speeches by Speaker (Top {top_n})",
                color='Speech Count',
                color_continuous_scale='Viridis'
            )
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Create bar chart for content length (as proxy for speaking time)
            fig2 = px.bar(
                top_speakers,
                x='Speaker',
                y='Content Length',
                title=f"Estimated Speaking Volume by Speaker (Top {top_n})",
                color='Content Length',
                color_continuous_scale='Viridis'
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Show time trend if date column exists
            if 'Date' in filtered_df.columns:
                st.subheader("Speaker Activity Over Time")
                
                # Group by date and speaker
                speaker_time_data = filtered_df.groupby([
                    filtered_df['Date'].dt.date, 'Speaker'
                ]).size().reset_index()
                speaker_time_data.columns = ['Date', 'Speaker', 'Count']
                
                # Select speakers to visualize
                selected_speakers = st.multiselect(
                    "Select Speakers to Visualize",
                    options=speaker_stats['Speaker'].tolist(),
                    default=speaker_stats['Speaker'].head(5).tolist()
                )
                
                if selected_speakers:
                    # Filter data for selected speakers
                    selected_data = speaker_time_data[speaker_time_data['Speaker'].isin(selected_speakers)]
                    
                    # Create line chart
                    fig3 = px.line(
                        selected_data,
                        x='Date',
                        y='Count',
                        color='Speaker',
                        title='Speaker Activity Over Time'
                    )
                    st.plotly_chart(fig3, use_container_width=True)
        else:
            st.error("Required columns (Speaker, Content) not found in data.")
    
    except Exception as e:
        logger.error(f"Error in speaker statistics: {str(e)}")
        st.error(f"An error occurred during speaker analysis: {str(e)}")

def render_topic_analysis(df, entity_extractor):
    """Render topic analysis.
    
    Args:
        df: DataFrame containing parliamentary data
        entity_extractor: EntityExtractor instance
    """
    st.subheader("Topic Analysis")
    
    try:
        # Use entity extractor to get topics if available
        if entity_extractor and hasattr(entity_extractor, 'extract_entities_from_dataframe'):
            with st.spinner("Analyzing topics in data..."):
                # Extract entities for topic analysis
                enhanced_df, entity_map = entity_extractor.extract_entities_from_dataframe(df, use_cache=True)
                
                # Filter for only topic entities
                topic_entities = []
                for entry_id, entities in entity_map.items():
                    for entity in entities:
                        if entity['label'].lower() == 'topic':
                            entity_copy = entity.copy()
                            entity_copy['entry_id'] = entry_id
                            
                            # Add source document info
                            doc_row = df[df['entry_id'].astype(str) == str(entry_id)]
                            if not doc_row.empty:
                                if 'Date' in doc_row.columns:
                                    entity_copy['date'] = doc_row['Date'].iloc[0]
                                if 'Speaker' in doc_row.columns:
                                    entity_copy['speaker'] = doc_row['Speaker'].iloc[0]
                            
                            topic_entities.append(entity_copy)
                
                if topic_entities:
                    topics_df = pd.DataFrame(topic_entities)
                    
                    # Topic frequency
                    topic_counts = topics_df['text'].value_counts().reset_index()
                    topic_counts.columns = ['Topic', 'Frequency']
                    
                    # Show top topics
                    st.subheader("Most Frequently Discussed Topics")
                    top_n = st.slider("Number of topics to show", 5, 50, 20, key="topic_slider")
                    top_topics = topic_counts.head(top_n)
                    
                    # Display as bar chart
                    fig1 = px.bar(
                        top_topics,
                        x='Topic',
                        y='Frequency',
                        title=f"Top {top_n} Topics by Frequency",
                        color='Frequency',
                        color_continuous_scale='Viridis'
                    )
                    fig1.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Generate word cloud from topics
                    st.subheader("Topic Word Cloud")
                    
                    # Create a dictionary of words and their counts
                    word_counts = {row['Topic']: row['Frequency'] for _, row in top_topics.iterrows()}
                    
                    # Check if there's enough data for a word cloud
                    if word_counts:
                        # Create and display word cloud
                        try:
                            wordcloud = WordCloud(
                                width=800,
                                height=400,
                                background_color='white',
                                max_words=100,
                                colormap='viridis'
                            ).generate_from_frequencies(word_counts)
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error generating word cloud: {str(e)}")
                    
                    # Topic by speaker analysis
                    if 'speaker' in topics_df.columns:
                        st.subheader("Topics by Speaker")
                        
                        # Get top speakers
                        speaker_counts = topics_df['speaker'].value_counts().head(10)
                        top_speakers = speaker_counts.index.tolist()
                        
                        # Select speakers to analyze
                        selected_speakers = st.multiselect(
                            "Select Speakers",
                            options=top_speakers,
                            default=top_speakers[:3] if len(top_speakers) >= 3 else top_speakers
                        )
                        
                        if selected_speakers:
                            # Filter for selected speakers
                            speaker_topics = topics_df[topics_df['speaker'].isin(selected_speakers)]
                            
                            # Group by speaker and topic
                            speaker_topic_counts = speaker_topics.groupby(['speaker', 'text']).size().reset_index()
                            speaker_topic_counts.columns = ['Speaker', 'Topic', 'Count']
                            
                            # Sort by count within each speaker
                            speaker_topic_counts = speaker_topic_counts.sort_values(['Speaker', 'Count'], ascending=[True, False])
                            
                            # Show top topics for each speaker
                            topics_per_speaker = 5
                            top_speaker_topics = []
                            for speaker in selected_speakers:
                                speaker_data = speaker_topic_counts[speaker_topic_counts['Speaker'] == speaker].head(topics_per_speaker)
                                top_speaker_topics.append(speaker_data)
                            
                            if top_speaker_topics:
                                combined_data = pd.concat(top_speaker_topics)
                                
                                # Create grouped bar chart
                                fig2 = px.bar(
                                    combined_data,
                                    x='Topic',
                                    y='Count',
                                    color='Speaker',
                                    title=f"Top {topics_per_speaker} Topics by Selected Speakers",
                                    barmode='group'
                                )
                                fig2.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No topic entities found in the data. Try adjusting entity extraction settings.")
        else:
            # Fallback to simple word frequency analysis if no entity extractor
            st.info("Entity extractor not available. Using simple word frequency analysis.")
            
            if 'Content' in df.columns:
                # Combine all content
                all_text = " ".join(df['Content'].astype(str))
                
                # Simple word frequency analysis (excluding common words)
                words = all_text.lower().split()
                common_words = set(['the', 'and', 'to', 'of', 'in', 'a', 'is', 'that', 'for', 'it', 'on', 'with', 'as', 'this', 'by', 'be', 'are', 'or', 'at', 'from', 'was', 'we', 'will', 'have', 'has', 'not', 'they', 'an', 'i', 'but', 'would', 'which', 'there', 'been', 'can', 'their', 'you', 'all', 'who', 'so', 'more', 'what', 'about', 'when', 'one', 'also'])
                filtered_words = [word for word in words if word not in common_words and len(word) > 3]
                
                word_counts = Counter(filtered_words)
                top_words = word_counts.most_common(50)
                
                word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                
                # Display bar chart
                fig = px.bar(
                    word_df.head(30),
                    x='Word',
                    y='Frequency',
                    title="Word Frequency Analysis",
                    color='Frequency',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        logger.error(f"Error in topic analysis: {str(e)}")
        st.error(f"An error occurred during topic analysis: {str(e)}")

def render_entity_cooccurrence(df, entity_extractor, knowledge_graph):
    """Render entity co-occurrence analysis.
    
    Args:
        df: DataFrame containing parliamentary data
        entity_extractor: EntityExtractor instance
        knowledge_graph: KnowledgeGraph instance
    """
    st.subheader("Entity Co-occurrence Analysis")
    
    try:
        # Check if entity extractor is available
        if entity_extractor and hasattr(entity_extractor, 'extract_entities_from_dataframe'):
            with st.spinner("Analyzing entity co-occurrences..."):
                # Extract entities
                enhanced_df, entity_map = entity_extractor.extract_entities_from_dataframe(df, use_cache=True)
                
                # Create entity dataframe
                all_entities = []
                for entry_id, entities in entity_map.items():
                    for entity in entities:
                        entity_copy = entity.copy()
                        entity_copy['entry_id'] = entry_id
                        all_entities.append(entity_copy)
                
                if all_entities:
                    entities_df = pd.DataFrame(all_entities)
                    
                    # Get entity types
                    entity_types = entities_df['label'].unique().tolist()
                    
                    # User selection for entity types
                    selected_types = st.multiselect(
                        "Select Entity Types to Analyze",
                        options=entity_types,
                        default=entity_types[:2] if len(entity_types) >= 2 else entity_types
                    )
                    
                    if selected_types:
                        # Filter by selected types
                        filtered_entities = entities_df[entities_df['label'].isin(selected_types)]
                        
                        # Calculate co-occurrence
                        top_n = st.slider("Number of top entities per type", 5, 20, 10, key="entity_slider")
                        
                        # Get top entities of each selected type
                        top_entities_by_type = {}
                        for entity_type in selected_types:
                            type_counts = filtered_entities[filtered_entities['label'] == entity_type]['text'].value_counts().head(top_n)
                            top_entities_by_type[entity_type] = type_counts.index.tolist()
                        
                        # Create flat list of all top entities
                        all_top_entities = []
                        for entity_list in top_entities_by_type.values():
                            all_top_entities.extend(entity_list)
                        
                        # Initialize co-occurrence matrix
                        co_matrix = pd.DataFrame(
                            0,
                            index=all_top_entities,
                            columns=all_top_entities
                        )
                        
                        # Calculate co-occurrences
                        for entry_id in filtered_entities['entry_id'].unique():
                            entry_entities = filtered_entities[filtered_entities['entry_id'] == entry_id]['text'].tolist()
                            entry_entities = [e for e in entry_entities if e in all_top_entities]
                            
                            # Update co-occurrence counts
                            for i, entity1 in enumerate(entry_entities):
                                for entity2 in entry_entities[i:]:
                                    if entity1 != entity2:  # Skip self-pairs
                                        co_matrix.loc[entity1, entity2] += 1
                                        co_matrix.loc[entity2, entity1] += 1
                        
                        # Create heatmap
                        st.subheader("Entity Co-occurrence Heatmap")
                        
                        if not co_matrix.empty and co_matrix.sum().sum() > 0:
                            fig = go.Figure(data=go.Heatmap(
                                z=co_matrix.values,
                                x=co_matrix.columns,
                                y=co_matrix.index,
                                colorscale='Viridis',
                                hoverongaps=False
                            ))
                            
                            fig.update_layout(
                                title="Entity Co-occurrence Matrix",
                                xaxis_title="Entity",
                                yaxis_title="Entity",
                                height=800,
                                xaxis_tickangle=-45
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No significant co-occurrences found between the selected entities.")
                        
                        # Show network visualization if knowledge graph is available
                        if knowledge_graph and hasattr(knowledge_graph, 'get_filtered_graph'):
                            st.subheader("Entity Relationship Network")
                            
                            # Get relevant subgraph for selected entity types
                            filtered_graph = knowledge_graph.get_filtered_graph(
                                node_types=selected_types,
                                max_nodes=min(len(all_top_entities) * 2, 100)  # Limit to reasonable size
                            )
                            
                            if filtered_graph and filtered_graph.number_of_nodes() > 0:
                                # Get the visualization from the knowledge graph module
                                from src.utils.graph_visualization import create_plotly_graph
                                
                                fig = create_plotly_graph(filtered_graph)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No relevant relationships found in the knowledge graph for the selected entity types.")
                    else:
                        st.info("Please select at least one entity type to analyze.")
                else:
                    st.warning("No entities found in the data. Try adjusting entity extraction settings.")
        else:
            st.error("Entity extractor not available. Cannot perform co-occurrence analysis.")
    
    except Exception as e:
        logger.error(f"Error in entity co-occurrence analysis: {str(e)}")
        st.error(f"An error occurred during entity co-occurrence analysis: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Parliamentary Meeting Analyzer - Statistics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    render_statistics_tab() 