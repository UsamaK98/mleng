"""
Landing page component for the Parliamentary Minutes UI
"""
import streamlit as st


def render_landing_page(metadata):
    """
    Render the landing page with summary information and navigation
    
    Args:
        metadata: Dictionary of metadata about the parliamentary dataset
    """
    st.title("🏛️ Parliamentary Minutes Explorer")
    
    # Main info section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Welcome to the Parliamentary Minutes Explorer
        
        This application allows you to explore parliamentary minutes through an intuitive interface.
        
        **Key Features:**
        - Browse by speakers, topics, or sessions
        - View detailed speaker information
        - Analyze parliamentary discussions
        - Explore session content
        """)
        
        # Call-to-action buttons
        st.write("### Explore By:")
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            st.button("🧑‍💼 Speakers", key="landing_speakers_btn", use_container_width=True)
        
        with button_col2:
            st.button("📋 Topics", key="landing_topics_btn", use_container_width=True)
            
        with button_col3:
            st.button("📅 Sessions", key="landing_sessions_btn", use_container_width=True)
    
    with col2:
        if metadata:
            # Dataset statistics
            st.markdown("### Dataset Overview")
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(label="Total Sessions", value=metadata.get('total_sessions', 0))
            
            with metric_col2:
                st.metric(label="Speakers", value=metadata.get('total_speakers', 0))
            
            # Top speakers list
            st.markdown("### Top Speakers")
            top_speakers = metadata.get('top_speakers', [])[:5]
            
            for speaker in top_speakers:
                st.markdown(f"**{speaker['name']}** - {speaker['contributions']} contributions")
        else:
            st.error("Unable to load metadata. Please check if the API is running.")
    
    # Separator
    st.markdown("---")
    
    # Latest sessions section
    st.markdown("### Recent Sessions")
    
    session_dates = metadata.get('session_dates', [])[:3] if metadata else []
    if session_dates:
        cols = st.columns(len(session_dates))
        
        for i, date in enumerate(session_dates):
            with cols[i]:
                st.markdown(f"**{date}**")
                st.button("View Details", key=f"landing_session_{i}")
    else:
        st.info("No session data available")
    
    # Brief instructions
    st.markdown("---")
    st.markdown("""
    ### Getting Started
    
    1. Use the buttons above to navigate to different sections
    2. Click on speakers or sessions to view detailed information
    3. Explore the analytics to gain insights into parliamentary activities
    """)


def render_quick_search():
    """
    Render a quick search component for the landing page
    
    Returns:
        query: The search query entered by the user
    """
    st.markdown("### Quick Search")
    
    query = st.text_input("Search for a speaker, topic, or keyword:", key="landing_search")
    
    if st.button("Search", key="landing_search_btn"):
        return query
    
    return None 