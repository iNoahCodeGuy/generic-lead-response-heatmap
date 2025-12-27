"""
Presentation Layer - Streamlit UI

Entry point for the application. Handles user interactions and displays visualizations.
"""

import streamlit as st
import pandas as pd
from logic import (
    load_leads,
    combine_team_data,
    compute_response_metrics,
    aggregate_by_team_slot,
    calculate_team_rankings,
    find_slot_winners,
    compute_team_difference
)
from heatmap import (
    create_team_heatmap,
    create_winner_heatmap,
    create_difference_heatmap,
    create_team_legend
)
import io
from typing import List, Dict, Tuple

# Page configuration
st.set_page_config(
    page_title="Multi-Team Response Comparison",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Multi-Team Lead Response Comparison Dashboard")
st.markdown("Compare response performance across multiple sales teams to identify strengths and weaknesses.")

# Initialize session state
if 'leads_df' not in st.session_state:
    st.session_state.leads_df = None
if 'teams' not in st.session_state:
    st.session_state.teams = []
if 'bucket_minutes' not in st.session_state:
    st.session_state.bucket_minutes = 30


def load_data_from_upload(uploaded_files: List) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load data from uploaded files.
    Supports both single file with team column and multiple team files.
    """
    all_dfs = []
    teams = []
    
    for uploaded_file in uploaded_files:
        # Try to read the file
        try:
            df = load_leads(uploaded_file)
            
            # Check if team column exists
            if 'team' in df.columns:
                # Single file with team column
                teams.extend(df['team'].unique().tolist())
                all_dfs.append(df)
            else:
                # Multiple files - derive team name from filename
                team_name = uploaded_file.name.replace('.csv', '').replace('team_', '').replace('_', ' ').title()
                df = load_leads(uploaded_file, team_name=team_name)
                teams.append(team_name)
                all_dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            continue
    
    if not all_dfs:
        return None, []
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    unique_teams = sorted(list(set(teams)))
    
    return combined_df, unique_teams


def get_team_colors(teams: List[str]) -> Dict[str, str]:
    """Generate consistent colors for teams."""
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Yellow-green
        '#17becf'   # Cyan
    ]
    
    return {team: colors[i % len(colors)] for i, team in enumerate(teams)}


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data upload
    st.subheader("Data Upload")
    uploaded_files = st.file_uploader(
        "Upload CSV file(s)",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload a single CSV with 'team' column, or multiple team-specific CSV files"
    )
    
    # Load sample data option
    if st.button("📁 Load Sample Data", type="primary"):
        try:
            import os
            import pathlib
            
            # Get the directory where app.py is located
            current_file = pathlib.Path(__file__).parent.absolute()
            sample_dir = current_file / 'sample_data'
            
            # Try loading all_teams.csv first (single file with team column)
            all_teams_path = sample_dir / 'all_teams.csv'
            if all_teams_path.exists():
                df = load_leads(str(all_teams_path))
                st.session_state.leads_df = df
                st.session_state.teams = sorted(df['team'].unique().tolist())
                st.success(f"✅ Sample data loaded! {len(df)} leads from {len(st.session_state.teams)} team(s)")
            else:
                # Fallback to individual team files
                sample_files = []
                for team_file in ['team_alpha.csv', 'team_beta.csv']:
                    team_path = sample_dir / team_file
                    if team_path.exists():
                        with open(team_path, 'rb') as f:
                            file_obj = io.BytesIO(f.read())
                            file_obj.name = team_file
                            sample_files.append(file_obj)
                
                if sample_files:
                    st.session_state.leads_df, st.session_state.teams = load_data_from_upload(sample_files)
                    st.success(f"✅ Sample data loaded! {len(st.session_state.leads_df)} leads from {len(st.session_state.teams)} team(s)")
                else:
                    st.warning("⚠️ Sample data files not found. Please upload your own data.")
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Process uploaded files
    if uploaded_files:
        if st.button("📥 Load Data"):
            st.session_state.leads_df, st.session_state.teams = load_data_from_upload(uploaded_files)
            if st.session_state.leads_df is not None:
                st.success(f"Loaded {len(st.session_state.leads_df)} leads from {len(st.session_state.teams)} team(s)")
    
    # Bucket size selector
    st.subheader("Time Bucket Size")
    bucket_options = {
        "15 minutes": 15,
        "30 minutes": 30,
        "60 minutes": 60
    }
    selected_bucket = st.selectbox(
        "Select time bucket size",
        options=list(bucket_options.keys()),
        index=1  # Default to 30 minutes
    )
    st.session_state.bucket_minutes = bucket_options[selected_bucket]
    
    # Display current data info
    if st.session_state.leads_df is not None:
        st.subheader("📈 Data Summary")
        st.write(f"**Total Leads:** {len(st.session_state.leads_df)}")
        st.write(f"**Teams:** {', '.join(st.session_state.teams)}")
        st.write(f"**Date Range:** {st.session_state.leads_df['received_time'].min().date()} to {st.session_state.leads_df['received_time'].max().date()}")


# Main content area
if st.session_state.leads_df is None or len(st.session_state.teams) == 0:
    st.info("👈 Please upload data files or load sample data from the sidebar to get started.")
    st.markdown("""
    ### Expected Data Format
    
    **Single CSV with team column:**
    - `lead_id`: Unique identifier
    - `received_time`: When the lead was received
    - `first_contact_time`: When first contact was made (can be empty)
    - `team`: Team name
    
    **Multiple team files:**
    - Same columns but without `team` column
    - Team name derived from filename
    """)
else:
    # Process data
    with st.spinner("Processing data..."):
        # Compute metrics
        leads_df = compute_response_metrics(st.session_state.leads_df)
        
        # Aggregate by team and slot
        aggregated_df = aggregate_by_team_slot(leads_df, st.session_state.bucket_minutes)
        
        # Calculate rankings
        rankings_df = calculate_team_rankings(aggregated_df)
        
        # Get team colors
        team_colors = get_team_colors(st.session_state.teams)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Side-by-Side Heatmaps",
        "🏆 Winner Heatmap",
        "⚔️ Head-to-Head",
        "📈 Leaderboard"
    ])
    
    with tab1:
        st.header("Side-by-Side Team Heatmaps")
        st.markdown("Compare all teams with consistent color scales for fair comparison.")
        
        # Find global min/max for consistent color scale
        zmin = 0
        zmax = 100
        
        # Create heatmaps for each team
        cols = st.columns(min(2, len(st.session_state.teams)))
        for idx, team in enumerate(st.session_state.teams):
            team_data = aggregated_df[aggregated_df['team'] == team]
            fig = create_team_heatmap(team_data, team, bucket_minutes=st.session_state.bucket_minutes, zmin=zmin, zmax=zmax)
            
            col_idx = idx % 2
            with cols[col_idx]:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Winner Heatmap")
        st.markdown("See which team leads at each time slot.")
        
        # Find winners
        winner_df = find_slot_winners(aggregated_df)
        
        # Create winner heatmap
        fig = create_winner_heatmap(winner_df, team_colors, bucket_minutes=st.session_state.bucket_minutes)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display team legend
        st.markdown(create_team_legend(st.session_state.teams, team_colors), unsafe_allow_html=True)
    
    with tab3:
        st.header("Head-to-Head Comparison")
        st.markdown("Direct comparison between any two teams.")
        
        if len(st.session_state.teams) < 2:
            st.warning("Need at least 2 teams for head-to-head comparison.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                team_a = st.selectbox("Select Team A", options=st.session_state.teams, key="team_a")
            with col2:
                team_b = st.selectbox("Select Team B", options=st.session_state.teams, key="team_b")
            
            if team_a == team_b:
                st.warning("Please select two different teams.")
            else:
                # Compute difference
                diff_df = compute_team_difference(aggregated_df, team_a, team_b)
                
                # Create difference heatmap
                fig = create_difference_heatmap(diff_df, team_a, team_b, bucket_minutes=st.session_state.bucket_minutes)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                team_a_data = aggregated_df[aggregated_df['team'] == team_a]
                team_b_data = aggregated_df[aggregated_df['team'] == team_b]
                
                with col1:
                    st.metric(
                        f"{team_a} Overall",
                        f"{team_a_data['pct_within_1hr'].mean():.1f}%",
                        delta=f"{team_a_data['pct_within_1hr'].mean() - team_b_data['pct_within_1hr'].mean():+.1f}%"
                    )
                with col2:
                    st.metric(
                        f"{team_b} Overall",
                        f"{team_b_data['pct_within_1hr'].mean():.1f}%",
                        delta=f"{team_b_data['pct_within_1hr'].mean() - team_a_data['pct_within_1hr'].mean():+.1f}%"
                    )
                with col3:
                    avg_diff = diff_df['difference'].mean()
                    st.metric(
                        "Average Difference",
                        f"{avg_diff:+.1f}%",
                        help=f"Positive means {team_a} is better, negative means {team_b} is better"
                    )
    
    with tab4:
        st.header("Team Leaderboard")
        st.markdown("Overall performance rankings and statistics.")
        
        # Display rankings table
        st.subheader("Overall Rankings")
        display_rankings = rankings_df[['rank', 'team', 'overall_pct', 'total_leads', 'total_contacted_1hr', 'avg_pct', 'min_pct', 'max_pct']].copy()
        display_rankings.columns = ['Rank', 'Team', 'Overall %', 'Total Leads', 'Contacted <1hr', 'Avg %', 'Min %', 'Max %']
        display_rankings['Overall %'] = display_rankings['Overall %'].round(1)
        display_rankings['Avg %'] = display_rankings['Avg %'].round(1)
        display_rankings['Min %'] = display_rankings['Min %'].round(1)
        display_rankings['Max %'] = display_rankings['Max %'].round(1)
        
        st.dataframe(display_rankings, use_container_width=True, hide_index=True)
        
        # Best and worst time slots per team
        st.subheader("Best and Worst Time Slots by Team")
        for team in st.session_state.teams:
            team_data = aggregated_df[aggregated_df['team'] == team].copy()
            if len(team_data) > 0:
                best_slot = team_data.loc[team_data['pct_within_1hr'].idxmax()]
                worst_slot = team_data.loc[team_data['pct_within_1hr'].idxmin()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{team} - Best Slot:**")
                    st.write(f"{best_slot['day_of_week']} {best_slot['time_bucket'].strftime('%H:%M')}: {best_slot['pct_within_1hr']:.1f}% ({best_slot['total_leads']} leads)")
                with col2:
                    st.markdown(f"**{team} - Worst Slot:**")
                    st.write(f"{worst_slot['day_of_week']} {worst_slot['time_bucket'].strftime('%H:%M')}: {worst_slot['pct_within_1hr']:.1f}% ({worst_slot['total_leads']} leads)")
                st.divider()

