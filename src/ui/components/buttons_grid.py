"""
Button grid components for the Parliamentary Minutes UI
"""
import streamlit as st


def create_button_grid(items, columns=3, on_click=None, key_prefix="btn"):
    """
    Create a grid of buttons
    
    Args:
        items: List of items to create buttons for
        columns: Number of columns in the grid
        on_click: Function to call when a button is clicked (receives item as parameter)
        key_prefix: Prefix for button keys
        
    Returns:
        List of items that were clicked
    """
    if not items:
        st.info("No items available")
        return None
    
    # Create columns
    cols = st.columns(columns)
    clicked_items = []
    
    # Add buttons to columns
    for i, item in enumerate(items):
        col_idx = i % columns
        with cols[col_idx]:
            if st.button(item, key=f"{key_prefix}_{i}"):
                if on_click:
                    on_click(item)
                clicked_items.append(item)
    
    return clicked_items


def create_item_cards(items, columns=3, on_click=None, key_prefix="card"):
    """
    Create a grid of item cards
    
    Args:
        items: List of dictionaries with item details
               Each dict should have at least 'title' and 'description' keys
        columns: Number of columns in the grid
        on_click: Function to call when a card is clicked (receives item as parameter)
        key_prefix: Prefix for button keys
        
    Returns:
        List of items that were clicked
    """
    if not items:
        st.info("No items available")
        return None
    
    # Create columns
    cols = st.columns(columns)
    clicked_items = []
    
    # Add cards to columns
    for i, item in enumerate(items):
        col_idx = i % columns
        with cols[col_idx]:
            with st.container():
                st.markdown(f"**{item['title']}**")
                st.markdown(item.get('description', ''))
                
                if on_click:
                    if st.button("View", key=f"{key_prefix}_{i}"):
                        on_click(item)
                        clicked_items.append(item)
    
    return clicked_items


def create_speaker_grid(speakers, columns=4, on_click=None):
    """
    Create a grid of speaker buttons
    
    Args:
        speakers: List of speaker dictionaries with name, role, and contributions
        columns: Number of columns in the grid
        on_click: Function to call when a speaker is clicked
        
    Returns:
        Selected speaker (if any)
    """
    if not speakers:
        st.info("No speakers available")
        return None
    
    # Create columns
    cols = st.columns(columns)
    selected_speaker = None
    
    # Add speaker buttons to columns
    for i, speaker in enumerate(speakers):
        col_idx = i % columns
        with cols[col_idx]:
            name = speaker.get('name', 'Unknown')
            role = speaker.get('role', '')
            contributions = speaker.get('contributions', 0)
            
            with st.container():
                st.markdown(f"**{name}**")
                st.caption(f"{role}")
                st.caption(f"{contributions} contributions")
                
                if st.button("View", key=f"speaker_{i}"):
                    if on_click:
                        on_click(speaker)
                    selected_speaker = speaker
    
    return selected_speaker


def create_topic_grid(topics, columns=3, on_click=None):
    """
    Create a grid of topic buttons
    
    Args:
        topics: List of topic dictionaries with name and description
        columns: Number of columns in the grid
        on_click: Function to call when a topic is clicked
        
    Returns:
        Selected topic (if any)
    """
    if not topics:
        st.info("No topics available")
        return None
    
    # Create columns
    cols = st.columns(columns)
    selected_topic = None
    
    # Add topic buttons to columns
    for i, topic in enumerate(topics):
        col_idx = i % columns
        with cols[col_idx]:
            name = topic.get('name', 'Unknown')
            description = topic.get('description', '')
            
            with st.container():
                st.markdown(f"**{name}**")
                st.caption(description[:50] + ('...' if len(description) > 50 else ''))
                
                if st.button("View", key=f"topic_{i}"):
                    if on_click:
                        on_click(topic)
                    selected_topic = topic
    
    return selected_topic


def create_session_grid(sessions, columns=3, on_click=None):
    """
    Create a grid of session buttons
    
    Args:
        sessions: List of session dictionaries with date and description
        columns: Number of columns in the grid
        on_click: Function to call when a session is clicked
        
    Returns:
        Selected session (if any)
    """
    if not sessions:
        st.info("No sessions available")
        return None
    
    # Create columns
    cols = st.columns(columns)
    selected_session = None
    
    # Add session buttons to columns
    for i, session in enumerate(sessions):
        col_idx = i % columns
        with cols[col_idx]:
            date = session.get('date', 'Unknown')
            description = session.get('description', '')
            speakers = session.get('total_speakers', 0)
            
            with st.container():
                st.markdown(f"**{date}**")
                st.caption(f"{speakers} speakers")
                
                if st.button("View", key=f"session_{i}"):
                    if on_click:
                        on_click(session)
                    selected_session = session
    
    return selected_session 