"""
Speaker view component for the Parliamentary Minutes UI
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime

from config.config import API_HOST, API_PORT

# Define API base URL
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"


def query_api(endpoint, payload):
    """Query the API with the given endpoint and payload"""
    try:
        response = requests.post(f"{API_BASE_URL}/{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying API: {e}")
        return None


def render_speaker_view(speaker_name=None):
    """
    Render the speaker view with details about a specific speaker
    
    Args:
        speaker_name: Name of the speaker to show details for (optional)
    """
    st.title("🧑‍💼 Speaker Information")
    
    # Speaker search
    if not speaker_name:
        speaker_name = st.text_input("Enter speaker name:", key="speaker_view_input")
    
    if speaker_name:
        with st.spinner("Loading speaker information..."):
            response = query_api("entity", {"entity": speaker_name})
            
            if response and response.get("found", False):
                _display_speaker_details(response)
            else:
                st.warning(response.get("answer", f"No information found for speaker: {speaker_name}"))
                
                # Display suggested speakers if available
                if response and "available_speakers" in response and response["available_speakers"]:
                    st.write("### Try searching for these speakers instead:")
                    cols = st.columns(min(3, len(response["available_speakers"])))
                    
                    for i, speaker in enumerate(response["available_speakers"]):
                        with cols[i % 3]:
                            if st.button(speaker, key=f"speaker_view_alt_{i}"):
                                # Call recursively with this speaker instead
                                render_speaker_view(speaker)
                                return
    else:
        # If no speaker provided, show a list of speakers to choose from
        _display_speaker_list()


def _display_speaker_details(response):
    """
    Display detailed information about a speaker
    
    Args:
        response: API response containing speaker details
    """
    # Header and basic info
    st.header(response["entity"])
    st.subheader(f"Role: {response.get('role', 'Unknown')}")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overview", "Contributions", "Statistics"])
    
    with tab1:
        # Summary and key stats
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if "summary" in response:
                st.write("### Summary")
                st.write(response["summary"])
            
            # Topics frequently discussed
            if "frequent_topics" in response and response["frequent_topics"]:
                st.write("### Frequently Discussed Topics")
                topics = response["frequent_topics"]
                for topic, weight in topics.items():
                    st.write(f"- **{topic}**: {weight:.2f}")
        
        with col2:
            st.write("### Statistics")
            st.write(f"Total Contributions: {response.get('total_contributions', 0)}")
            st.write(f"Total Words: {response.get('speaker_stats', {}).get('total_words', 0)}")
            st.write(f"Avg Words per Contribution: {response.get('speaker_stats', {}).get('avg_words_per_contribution', 0):.1f}")
            
            # Meeting dates present
            st.write("### Meeting Dates")
            if "meeting_dates_present" in response:
                st.write(f"Present in {len(response['meeting_dates_present'])} sessions")
    
    with tab2:
        # Sample contributions
        if "sample_contributions" in response and response["sample_contributions"]:
            st.write("### Sample Contributions")
            for i, contrib in enumerate(response["sample_contributions"], 1):
                with st.expander(f"Contribution {i} - {contrib['date']}"):
                    st.write(f"**Date:** {contrib['date']}")
                    st.write(f"**Time:** {contrib['timestamp']}")
                    st.write(f"**Content:** {contrib['content']}")
        
        # Meeting dates table
        st.write("### Contributions by Date")
        if "meeting_dates_present" in response:
            dates_df = pd.DataFrame({
                "Date": response.get("meeting_dates_present", []),
                "Contributions": [response.get("contributions_by_date", {}).get(d, 0) 
                                 for d in response.get("meeting_dates_present", [])]
            })
            
            if not dates_df.empty:
                # Sort by date (newest first)
                dates_df["Date"] = pd.to_datetime(dates_df["Date"])
                dates_df = dates_df.sort_values("Date", ascending=False)
                dates_df["Date"] = dates_df["Date"].dt.strftime("%Y-%m-%d")
                
                st.dataframe(dates_df)
    
    with tab3:
        # More detailed statistics
        st.write("### Contribution Statistics")
        
        # Create contribution statistics
        stats = response.get("speaker_stats", {})
        
        if stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Contributions", stats.get("total_contributions", 0))
                st.metric("Total Words", stats.get("total_words", 0))
                st.metric("Average Words per Contribution", round(stats.get("avg_words_per_contribution", 0), 1))
            
            with col2:
                st.metric("Sessions Attended", len(response.get("meeting_dates_present", [])))
                st.metric("Unique Words Used", stats.get("unique_words", 0))
                st.metric("Complexity Score", round(stats.get("complexity_score", 0), 2))
        
        # Additional statistics would be displayed here


def _display_speaker_list():
    """Display a list of speakers to select from"""
    try:
        # Get metadata from API
        response = requests.get(f"{API_BASE_URL}/metadata")
        response.raise_for_status()
        metadata = response.json()
        
        st.write("### Select a Speaker")
        
        if "top_speakers" in metadata and metadata["top_speakers"]:
            # Create speaker cards for top speakers
            top_speakers = metadata["top_speakers"][:8]  # Limit to 8 for UI clarity
            
            cols = st.columns(4)  # 4 columns of speakers
            
            for i, speaker in enumerate(top_speakers):
                col_idx = i % 4
                with cols[col_idx]:
                    with st.container():
                        st.markdown(f"**{speaker['name']}**")
                        st.caption(f"Role: {speaker.get('role', 'Unknown')}")
                        st.caption(f"{speaker['contributions']} contributions")
                        
                        if st.button("View", key=f"speaker_list_{i}"):
                            # Call recursively with this speaker
                            render_speaker_view(speaker['name'])
                            return
            
            # Show option to see all speakers
            st.markdown("---")
            st.write("### All Speakers")
            
            # Allow filtering by first letter
            all_speakers = metadata.get("all_speakers", [])
            if all_speakers:
                # Extract first letters and create alphabet filter
                first_letters = sorted(set(s['name'][0].upper() for s in all_speakers if s['name']))
                
                # Create alphabet buttons
                alphabet_cols = st.columns(len(first_letters))
                selected_letter = None
                
                for i, letter in enumerate(first_letters):
                    with alphabet_cols[i]:
                        if st.button(letter, key=f"letter_{letter}"):
                            selected_letter = letter
                
                # Filter speakers by selected letter
                if selected_letter:
                    filtered_speakers = [s for s in all_speakers if s['name'] and s['name'][0].upper() == selected_letter]
                    
                    # Display filtered speakers
                    for speaker in filtered_speakers:
                        if st.button(f"{speaker['name']} ({speaker.get('role', 'Unknown')})", key=f"speaker_filtered_{speaker['name']}"):
                            render_speaker_view(speaker['name'])
                            return
            
        else:
            st.warning("No speaker information available")
    
    except Exception as e:
        st.error(f"Error loading speaker list: {e}")
        st.info("Please enter a speaker name in the search box above to find their information.") 