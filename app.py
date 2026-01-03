"""
Presentation Layer - Streamlit UI

Entry point for the application. Handles user interactions and displays visualizations.
"""

import streamlit as st
import pandas as pd
from logic import (
    load_leads,
    detect_columns,
    validate_required_columns,
    get_column_mapping_options,
    combine_team_data,
    compute_response_metrics,
    aggregate_by_team_slot,
    calculate_team_rankings,
    find_slot_winners,
    compute_team_difference,
    get_unique_managers,
    filter_by_manager,
    aggregate_by_manager_slot,
    calculate_manager_rankings,
    compute_vs_average,
    compute_improvement_needed,
    compute_weekly_leads_short,
    compute_leads_to_beat_leader,
    compute_leads_to_beat_average
)
from heatmap import (
    create_team_heatmap,
    create_winner_heatmap,
    create_difference_heatmap,
    create_team_legend,
    create_comparison_heatmap
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
if 'managers' not in st.session_state:
    st.session_state.managers = []
if 'selected_manager' not in st.session_state:
    st.session_state.selected_manager = None
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "Teams"
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = {}
if 'use_count_flag' not in st.session_state:
    st.session_state.use_count_flag = False
if 'use_contacted_flag' not in st.session_state:
    st.session_state.use_contacted_flag = False
if 'selected_week' not in st.session_state:
    st.session_state.selected_week = "All Weeks"


def get_complete_weeks(df: pd.DataFrame) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """
    Find complete weeks (Mon-Sun) in the dataset.
    Returns list of (label, start_date, end_date) tuples.
    """
    if 'received_time' not in df.columns:
        return []
    
    # Get min and max dates
    min_date = df['received_time'].min().normalize()
    max_date = df['received_time'].max().normalize()
    
    # Find the first Monday on or after min_date
    days_until_monday = (7 - min_date.dayofweek) % 7
    if min_date.dayofweek != 0:  # If not already Monday
        first_monday = min_date + pd.Timedelta(days=days_until_monday)
    else:
        first_monday = min_date
    
    # Find complete weeks
    weeks = []
    current_monday = first_monday
    
    while current_monday + pd.Timedelta(days=6) <= max_date:
        week_end = current_monday + pd.Timedelta(days=6)
        label = f"{current_monday.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}"
        weeks.append((label, current_monday, week_end))
        current_monday += pd.Timedelta(days=7)
    
    return weeks


def filter_to_week(df: pd.DataFrame, week_start: pd.Timestamp, week_end: pd.Timestamp) -> pd.DataFrame:
    """
    Filter dataframe to only include leads from the specified week.
    """
    # Include the full day of week_end (up to 23:59:59)
    week_end_inclusive = week_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (df['received_time'] >= week_start) & (df['received_time'] <= week_end_inclusive)
    return df[mask].copy()


def load_data_from_upload(uploaded_files: List, column_mappings: Dict[str, Dict[str, str]]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load data from uploaded files.
    Supports both single file with team column and multiple team files.
    Returns: (combined_df, teams, managers)
    """
    all_dfs = []
    teams = []
    
    for uploaded_file in uploaded_files:
        # Try to read the file
        try:
            mapping = column_mappings.get(uploaded_file.name, {})
            df = load_leads(uploaded_file, column_mapping=mapping)
            
            # Check if team column exists
            if 'team' in df.columns:
                # Single file with team column
                teams.extend(df['team'].unique().tolist())
                all_dfs.append(df)
            else:
                # Multiple files - derive team name from filename
                team_name = uploaded_file.name.replace('.csv', '').replace('team_', '').replace('_', ' ').title()
                df = load_leads(uploaded_file, team_name=team_name, column_mapping=mapping)
                teams.append(team_name)
                all_dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            continue
    
    if not all_dfs:
        return None, [], []
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    unique_teams = sorted(list(set(teams)))
    
    # Extract unique managers
    managers = get_unique_managers(combined_df)
    
    return combined_df, unique_teams, managers


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

    # Column mapping UI (shown when files uploaded)
    if uploaded_files:
        st.subheader("Column Mapping")
        st.caption("Select the correct columns if auto-detect fails. Required: lead_id, received_time, first_contact_time.")

        for uploaded_file in uploaded_files:
            # Peek at data without consuming the file
            df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)

            detected = detect_columns(df_preview)
            options = get_column_mapping_options(df_preview)
            stored = st.session_state.column_mappings.get(uploaded_file.name, {})

            with st.expander(f"Mapping for {uploaded_file.name}", expanded=False):
                col1, col2 = st.columns(2)
                required = ['lead_id', 'received_time', 'first_contact_time']
                optional = ['team', 'manager', 'count_for_metrics', 'contacted_flag']

                with col1:
                    st.markdown("**Required**")
                    for key in required:
                        default = stored.get(key) or detected.get(key)
                        opts = options.get(key, [])
                        index = opts.index(default) if default in opts else 0 if opts else 0
                        selection = st.selectbox(
                            f"{key}",
                            opts,
                            index=index if opts else None,
                            key=f"map_req_{uploaded_file.name}_{key}"
                        )
                        if selection:
                            stored[key] = selection

                with col2:
                    st.markdown("**Optional**")
                    for key in optional:
                        default = stored.get(key) or detected.get(key)
                        opts = ["-- Skip --"] + options.get(key, [])
                        index = opts.index(default) if default in opts else 0
                        selection = st.selectbox(
                            f"{key}",
                            opts,
                            index=index,
                            key=f"map_opt_{uploaded_file.name}_{key}"
                        )
                        if selection != "-- Skip --":
                            stored[key] = selection
                        elif key in stored:
                            stored.pop(key, None)

                st.session_state.column_mappings[uploaded_file.name] = stored
    
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
                    result = load_data_from_upload(sample_files)
                    st.session_state.leads_df, st.session_state.teams, st.session_state.managers = result
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
            result = load_data_from_upload(uploaded_files, st.session_state.column_mappings)
            st.session_state.leads_df, st.session_state.teams, st.session_state.managers = result
            if st.session_state.leads_df is not None:
                st.success(f"Loaded {len(st.session_state.leads_df)} leads from {len(st.session_state.teams)} team(s) and {len(st.session_state.managers)} manager(s)")
    
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
    
    # Analysis mode selector
    if st.session_state.leads_df is not None:
        st.subheader("Supervisor Flags")
        has_count_flag = 'count_for_metrics' in st.session_state.leads_df.columns
        has_contacted_flag = 'contacted_flag' in st.session_state.leads_df.columns

        st.session_state.use_count_flag = st.checkbox(
            "Use eligibility flag to include leads",
            value=st.session_state.use_count_flag if has_count_flag else False,
            disabled=not has_count_flag,
            help="When enabled, only leads with the mapped eligibility flag set will count toward metrics."
        )

        st.session_state.use_contacted_flag = st.checkbox(
            "Use contacted-within-hour flag instead of timestamps",
            value=st.session_state.use_contacted_flag if has_contacted_flag else False,
            disabled=not has_contacted_flag,
            help="When enabled, contacted counts come from the mapped boolean flag; timestamps still drive time bucket placement."
        )

        if st.session_state.use_count_flag and not has_count_flag:
            st.warning("Eligibility flag is enabled but no column is mapped; showing all leads.")
        if st.session_state.use_contacted_flag and not has_contacted_flag:
            st.warning("Contacted flag is enabled but no column is mapped; falling back to timestamps.")

        st.subheader("Analysis Mode")
        st.session_state.analysis_mode = st.radio(
            "Choose analysis type",
            options=["Teams", "Managers"],
            help="Compare by sales teams or by individual managers"
        )
        
        # Manager filter (only show if in manager mode and managers exist)
        if st.session_state.analysis_mode == "Managers" and len(st.session_state.managers) > 0:
            st.subheader("Manager Filter")
            manager_options = ["All Managers"] + st.session_state.managers
            selected = st.selectbox(
                "Filter by manager",
                options=manager_options,
                index=0
            )
            st.session_state.selected_manager = None if selected == "All Managers" else selected
    
    # Display current data info
    if st.session_state.leads_df is not None:
        st.subheader("📈 Data Summary")
        st.write(f"**Total Leads:** {len(st.session_state.leads_df)}")
        st.write(f"**Teams:** {', '.join(st.session_state.teams)}")
        if len(st.session_state.managers) > 0:
            st.write(f"**Managers:** {', '.join(st.session_state.managers)}")
        st.write(f"**Date Range:** {st.session_state.leads_df['received_time'].min().date()} to {st.session_state.leads_df['received_time'].max().date()}")
        
        # Week selector
        st.subheader("📅 Week Filter")
        complete_weeks = get_complete_weeks(st.session_state.leads_df)
        
        if complete_weeks:
            week_options = ["All Weeks"] + [w[0] for w in complete_weeks]
            st.session_state.selected_week = st.selectbox(
                "Select week to analyze",
                options=week_options,
                index=0,
                help="Filter to a single complete week (Mon-Sun) to avoid mixing data from different weeks"
            )
            
            if st.session_state.selected_week != "All Weeks":
                # Find the selected week's dates
                selected_idx = week_options.index(st.session_state.selected_week) - 1
                _, week_start, week_end = complete_weeks[selected_idx]
                st.info(f"📅 Showing: {week_start.strftime('%A, %b %d')} to {week_end.strftime('%A, %b %d, %Y')}")
        else:
            st.warning("No complete weeks (Mon-Sun) found in data.")
            st.session_state.selected_week = "All Weeks"



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
    # Process data based on analysis mode
    with st.spinner("Processing data..."):
        use_count_flag = st.session_state.use_count_flag and ('count_for_metrics' in st.session_state.leads_df.columns)
        use_contacted_flag = st.session_state.use_contacted_flag and ('contacted_flag' in st.session_state.leads_df.columns)

        leads_df = compute_response_metrics(
            st.session_state.leads_df,
            use_contacted_flag=use_contacted_flag,
            contacted_flag_column='contacted_flag',
            use_count_flag=use_count_flag,
            count_flag_column='count_for_metrics'
        )
        
        # Apply week filter if selected
        if st.session_state.selected_week != "All Weeks":
            complete_weeks = get_complete_weeks(st.session_state.leads_df)
            week_options = ["All Weeks"] + [w[0] for w in complete_weeks]
            if st.session_state.selected_week in week_options:
                selected_idx = week_options.index(st.session_state.selected_week) - 1
                _, week_start, week_end = complete_weeks[selected_idx]
                leads_df = filter_to_week(leads_df, week_start, week_end)
        
        if st.session_state.analysis_mode == "Teams":
            # Team-based analysis
            aggregated_df = aggregate_by_team_slot(leads_df, st.session_state.bucket_minutes)
            rankings_df = calculate_team_rankings(aggregated_df)
            grouping_key = 'team'
            items = st.session_state.teams
            color_dict = get_team_colors(st.session_state.teams)
        else:
            # Manager-based analysis
            has_manager_col = ('manager' in leads_df.columns) or ('ManagerPreferredName' in leads_df.columns)
            if not has_manager_col:
                st.warning("Manager analysis requires a manager column. Please map a manager column in the Column Mapping section or switch to Teams mode.")
                st.stop()
            if st.session_state.selected_manager:
                # Filter to specific manager
                filtered_leads = filter_by_manager(leads_df, st.session_state.selected_manager)
                aggregated_df = aggregate_by_manager_slot(filtered_leads, st.session_state.bucket_minutes)
                items = [st.session_state.selected_manager]
                analysis_title = f"Manager: {st.session_state.selected_manager}"
            else:
                # All managers
                aggregated_df = aggregate_by_manager_slot(leads_df, st.session_state.bucket_minutes)
                items = st.session_state.managers
                analysis_title = "All Managers"
            
            rankings_df = calculate_manager_rankings(aggregated_df)
            grouping_key = 'manager'
            color_dict = get_team_colors(items)  # Reuse color function for managers
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Side-by-Side Heatmaps",
        "🔍 Deep Dive",
        "🏆 Winner Heatmap",
        "⚔️ Head-to-Head",
        "📈 Leaderboard"
    ])
    
    with tab1:
        st.header(f"Side-by-Side {st.session_state.analysis_mode} Heatmaps")
        st.markdown(f"Compare all {st.session_state.analysis_mode.lower()} with consistent color scales for fair comparison.")
        
        # Show overall contact rates at the top
        st.subheader("Overall 1-Hour Contact Rates")
        metric_cols = st.columns(min(4, len(items)))
        for idx, item in enumerate(items):
            item_stats = rankings_df[rankings_df[grouping_key] == item]
            if len(item_stats) > 0:
                overall_pct = item_stats['overall_pct'].iloc[0]
                rank = int(item_stats['rank'].iloc[0])
                with metric_cols[idx % len(metric_cols)]:
                    st.metric(item, f"{overall_pct:.1f}%", delta=f"Rank #{rank}")
        
        st.divider()
        
        # Find global min/max for consistent color scale
        zmin = 0
        zmax = 100
        
        # Create heatmaps for each item (one per row for readability)
        for idx, item in enumerate(items):
            item_data = aggregated_df[aggregated_df[grouping_key] == item]
            fig = create_team_heatmap(item_data, item, bucket_minutes=st.session_state.bucket_minutes, zmin=zmin, zmax=zmax)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Drill-down section to inspect raw leads
        st.subheader("🔍 Inspect Raw Leads by Time Slot")
        st.caption("Select a day and time slot to see the underlying lead data.")
        
        # Add day_of_week to leads_df if not present
        if 'day_of_week' not in leads_df.columns:
            leads_df['day_of_week'] = leads_df['received_time'].dt.day_name()
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        available_days = [d for d in day_order if d in leads_df['day_of_week'].unique()]
        
        inspect_col1, inspect_col2, inspect_col3 = st.columns(3)
        
        with inspect_col1:
            inspect_day = st.selectbox("Day of Week", options=available_days, key="inspect_day")
        
        with inspect_col2:
            # Generate time slot options based on bucket size
            bucket_mins = st.session_state.bucket_minutes
            time_slots = []
            for hour in range(24):
                for minute in range(0, 60, bucket_mins):
                    time_slots.append(f"{hour:02d}:{minute:02d}")
            inspect_time = st.selectbox("Time Slot", options=time_slots, index=time_slots.index("08:00") if "08:00" in time_slots else 0, key="inspect_time")
        
        with inspect_col3:
            inspect_entity = st.selectbox(f"{st.session_state.analysis_mode[:-1]}", options=["All"] + items, key="inspect_entity")
        
        if st.button("🔎 Show Leads", key="inspect_btn"):
            # Parse selected time
            hour, minute = map(int, inspect_time.split(":"))
            
            # Filter leads for selected slot
            slot_leads = leads_df[leads_df['day_of_week'] == inspect_day].copy()
            
            # Filter by time bucket
            slot_leads['hour'] = slot_leads['received_time'].dt.hour
            slot_leads['minute_bucket'] = (slot_leads['received_time'].dt.minute // bucket_mins) * bucket_mins
            slot_leads = slot_leads[(slot_leads['hour'] == hour) & (slot_leads['minute_bucket'] == minute)]
            
            # Filter by entity if not "All"
            if inspect_entity != "All":
                if grouping_key == 'team':
                    slot_leads = slot_leads[slot_leads['team'] == inspect_entity]
                else:
                    manager_col = 'manager' if 'manager' in slot_leads.columns else 'ManagerPreferredName'
                    slot_leads = slot_leads[slot_leads[manager_col] == inspect_entity]
            
            if len(slot_leads) == 0:
                st.info(f"No leads found for {inspect_day} at {inspect_time}")
            else:
                st.success(f"Found {len(slot_leads)} leads for {inspect_day} at {inspect_time}")
                
                # Use correct column name: contact_within_1hr (from compute_response_metrics)
                contacted_col = 'contact_within_1hr'
                if contacted_col in slot_leads.columns:
                    contacted = slot_leads[contacted_col].sum()
                else:
                    contacted = 0
                st.write(f"**Contacted within 1hr:** {int(contacted)} / {len(slot_leads)} ({100*contacted/len(slot_leads):.1f}%)")
                
                # Display columns to show (use correct column names)
                display_cols = ['lead_id', 'received_time', 'first_contact_time', 'response_minutes']
                if 'team' in slot_leads.columns:
                    display_cols.insert(1, 'team')
                if 'manager' in slot_leads.columns:
                    display_cols.insert(2, 'manager')
                elif 'ManagerPreferredName' in slot_leads.columns:
                    display_cols.insert(2, 'ManagerPreferredName')
                if 'contact_within_1hr' in slot_leads.columns:
                    display_cols.append('contact_within_1hr')
                
                # Filter to available columns
                display_cols = [c for c in display_cols if c in slot_leads.columns]
                
                st.dataframe(
                    slot_leads[display_cols].sort_values('received_time'),
                    use_container_width=True,
                    hide_index=True
                )
    
    with tab2:
        st.header(f"🔍 {st.session_state.analysis_mode[:-1]} Deep Dive")
        st.markdown("Analyze one team/manager's performance against the average of all others.")
        
        # Select team/manager
        selected_item = st.selectbox(
            f"Select {st.session_state.analysis_mode[:-1]} to analyze:",
            options=items,
            key="deep_dive_select"
        )
        
        if selected_item:
            # Get stats for selected item
            item_stats = rankings_df[rankings_df[grouping_key] == selected_item]
            
            if len(item_stats) > 0:
                stats = item_stats.iloc[0]
                
                # Display prominent metrics
                st.subheader(f"{selected_item} - Performance Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Overall 1hr Contact Rate",
                        f"{stats['overall_pct']:.1f}%",
                        delta=f"Rank #{int(stats['rank'])} of {len(items)}"
                    )
                with col2:
                    st.metric("Total Leads", f"{int(stats['total_leads']):,}")
                with col3:
                    st.metric("Contacted <1hr", f"{int(stats['total_contacted_1hr']):,}")
                with col4:
                    avg_all = rankings_df['overall_pct'].mean()
                    diff = stats['overall_pct'] - avg_all
                    st.metric("vs Average", f"{diff:+.1f}%")
                
                st.divider()
                
                # NEW: Leads needed metrics
                st.subheader("🎯 Leads Needed to Improve")
                
                leader_info = compute_leads_to_beat_leader(rankings_df, selected_item, grouping_key)
                avg_info = compute_leads_to_beat_average(rankings_df, selected_item, grouping_key)
                
                goal_col1, goal_col2 = st.columns(2)
                
                with goal_col1:
                    if leader_info['is_leader']:
                        st.success("🏆 **Already #1!**")
                        st.caption(f"Current rate: {leader_info['current_pct']:.1f}%")
                    else:
                        st.metric(
                            f"Leads to Beat #1 ({leader_info['leader']})",
                            f"{leader_info['leads_needed']:,} leads",
                            delta=f"{leader_info['leader_pct'] - leader_info['current_pct']:.1f}% gap",
                            delta_color="inverse"
                        )
                        st.caption(f"You: {leader_info['current_pct']:.1f}% → Need: >{leader_info['leader_pct']:.1f}%")
                
                with goal_col2:
                    if avg_info['is_above_average']:
                        st.success("✅ **Above Average!**")
                        st.caption(f"You: {avg_info['current_pct']:.1f}% | Others Avg: {avg_info['avg_others_pct']:.1f}%")
                    else:
                        st.metric(
                            "Leads to Beat Others' Average",
                            f"{avg_info['leads_needed']:,} leads",
                            delta=f"{avg_info['avg_others_pct'] - avg_info['current_pct']:.1f}% gap",
                            delta_color="inverse"
                        )
                        st.caption(f"You: {avg_info['current_pct']:.1f}% → Need: >{avg_info['avg_others_pct']:.1f}%")
                
                st.divider()
                
                # Choose comparison type
                comparison_type = st.radio(
                    "View performance:",
                    options=["vs Average of All Others", "vs Specific Team"],
                    horizontal=True
                )
                
                if comparison_type == "vs Average of All Others":
                    # Comparison heatmap (simplified - no text overlay)
                    st.subheader("Performance vs Others by Time Slot")
                    st.markdown("🟢 **Green** = Beating others | 🔴 **Red** = Needs improvement")
                    
                    comparison_df = compute_vs_average(aggregated_df, selected_item, grouping_key)
                    fig = create_comparison_heatmap(comparison_df, selected_item, st.session_state.bucket_minutes)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Weekly Leads Short/Ahead Summary
                    st.subheader("📊 Weekly Performance vs Average")
                    weekly_summary = compute_weekly_leads_short(comparison_df)
                    if weekly_summary.empty:
                        st.success("🎉 Already at or above average across all time slots!")
                    else:
                        total_leads_short = int(weekly_summary['leads_short'].sum())
                        total_slots_ahead = int(weekly_summary['slots_ahead'].sum())
                        
                        if total_leads_short > 0:
                            st.metric("Leads Short to Match Average", f"{total_leads_short:,} leads")
                        elif total_slots_ahead > 0:
                            st.success(f"🎉 Beating average in {total_slots_ahead} time slots!")
                        
                        # Display per-day breakdown
                        st.dataframe(
                            weekly_summary.rename(columns={
                                'day_of_week': 'Day',
                                'leads_short': 'Leads Short',
                                'total_leads': 'Total Leads',
                                'slots_behind': 'Slots Behind',
                                'slots_ahead': 'Slots Ahead'
                            }),
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    st.divider()
                    
                    # Top Improvement Opportunities (ranked by impact)
                    st.subheader("🎯 Top Improvement Opportunities")
                    st.caption("Ranked by impact (deficit × lead volume)")
                    improvements = compute_improvement_needed(comparison_df, top_n=5, sort_by='impact_score')
                    if improvements.empty:
                        st.info("Already at or above the benchmark across all time slots.")
                    else:
                        for _, row in improvements.iterrows():
                            leads_short_txt = f" ({int(row['leads_short'])} leads short)" if row['leads_short'] > 0 else ""
                            st.warning(
                                f"**{row['day_of_week']} {row['time_bucket'].strftime('%H:%M')}**: "
                                f"{row['points_needed']:.1f} pts behind{leads_short_txt} — "
                                f"You: {row['target_pct']:.1f}% | Avg: {row['avg_others']:.1f}%"
                            )
                
                else:
                    # Compare against specific team
                    comparison_options = [t for t in items if t != selected_item]
                    if comparison_options:
                        comparison_team = st.selectbox(
                            f"Compare {selected_item} against:",
                            options=comparison_options,
                            key="comparison_team_select"
                        )
                        
                        st.subheader(f"Head-to-Head: {selected_item} vs {comparison_team}")
                        
                        # Compute difference with correct grouping key
                        diff_df = compute_team_difference(aggregated_df, selected_item, comparison_team, grouping_key=grouping_key)
                        
                        # Create difference heatmap
                        fig = create_difference_heatmap(diff_df, selected_item, comparison_team, bucket_minutes=st.session_state.bucket_minutes)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        selected_data = aggregated_df[aggregated_df[grouping_key] == selected_item]
                        comparison_data = aggregated_df[aggregated_df[grouping_key] == comparison_team]
                        
                        selected_pct = selected_data['pct_within_1hr'].mean()
                        comparison_pct = comparison_data['pct_within_1hr'].mean()
                        
                        with col1:
                            st.metric(
                                f"{selected_item}",
                                f"{selected_pct:.1f}%",
                                delta=f"{selected_pct - comparison_pct:+.1f}%"
                            )
                        with col2:
                            st.metric(
                                f"{comparison_team}",
                                f"{comparison_pct:.1f}%",
                                delta=f"{comparison_pct - selected_pct:+.1f}%"
                            )
                        with col3:
                            avg_diff = diff_df['difference'].mean()
                            st.metric(
                                "Average Difference",
                                f"{avg_diff:+.1f}%",
                                help=f"Positive means {selected_item} is better"
                            )

                        st.divider()
                        
                        st.subheader("📊 Performance Breakdown")
                        improvement_slots = diff_df.copy()
                        improvement_slots['points_needed'] = improvement_slots.apply(
                            lambda row: max(0, row['team_b_pct'] - row['team_a_pct']), axis=1
                        )
                        
                        behind_slots = improvement_slots[improvement_slots['points_needed'] > 0]
                        ahead_slots = improvement_slots[improvement_slots['points_needed'] == 0]
                        
                        # Summary metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Time Slots Behind", len(behind_slots))
                        with col2:
                            st.metric("Time Slots Ahead", len(ahead_slots))
                        
                        st.divider()
                        
                        # Show improvement opportunities
                        if not behind_slots.empty:
                            st.subheader("🎯 Where to Improve")
                            behind_display = behind_slots.sort_values('points_needed', ascending=False).head(5)
                            for _, row in behind_display.iterrows():
                                st.warning(
                                    f"**{row['day_of_week']} {row['time_bucket'].strftime('%H:%M')}**: "
                                    f"{row['points_needed']:.1f} pts behind — "
                                    f"{selected_item}: {row['team_a_pct']:.1f}% | {comparison_team}: {row['team_b_pct']:.1f}%"
                                )
                        else:
                            st.success(f"🎉 {selected_item} is already at or above {comparison_team} in every time slot!")
                        
                        # Show strengths
                        if not ahead_slots.empty:
                            st.subheader("💪 Strengths")
                            ahead_display = ahead_slots.sort_values('difference', ascending=False).head(3)
                            for _, row in ahead_display.iterrows():
                                st.success(
                                    f"**{row['day_of_week']} {row['time_bucket'].strftime('%H:%M')}**: "
                                    f"+{abs(row['difference']):.1f} pts ahead — "
                                    f"{selected_item}: {row['team_a_pct']:.1f}% | {comparison_team}: {row['team_b_pct']:.1f}%"
                                )
                    else:
                        st.info(f"No other {st.session_state.analysis_mode.lower()} to compare against.")
    
    with tab3:
        st.header("Winner Heatmap")
        st.markdown(f"See which {st.session_state.analysis_mode.lower()} leads at each time slot.")
        
        # Find winners
        winner_df = find_slot_winners(aggregated_df, grouping_key=grouping_key)
        
        # Create winner heatmap
        fig = create_winner_heatmap(winner_df, color_dict, bucket_minutes=st.session_state.bucket_minutes)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display legend
        st.markdown(create_team_legend(items, color_dict), unsafe_allow_html=True)
    
    with tab4:
        st.header("Head-to-Head Comparison")
        st.markdown(f"Direct comparison between any two {st.session_state.analysis_mode.lower()}.")
        
        if len(items) < 2:
            st.warning(f"Need at least 2 {st.session_state.analysis_mode.lower()} for head-to-head comparison.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                item_a = st.selectbox(f"Select {st.session_state.analysis_mode[:-1]} A", options=items, key="item_a")
            with col2:
                item_b = st.selectbox(f"Select {st.session_state.analysis_mode[:-1]} B", options=items, key="item_b")
            
            if item_a == item_b:
                st.warning(f"Please select two different {st.session_state.analysis_mode.lower()}.")
            else:
                # Compute difference with correct grouping key
                diff_df = compute_team_difference(aggregated_df, item_a, item_b, grouping_key=grouping_key)
                
                # Create difference heatmap
                fig = create_difference_heatmap(diff_df, item_a, item_b, bucket_minutes=st.session_state.bucket_minutes)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                item_a_data = aggregated_df[aggregated_df[grouping_key] == item_a]
                item_b_data = aggregated_df[aggregated_df[grouping_key] == item_b]
                
                with col1:
                    st.metric(
                        f"{item_a} Overall",
                        f"{item_a_data['pct_within_1hr'].mean():.1f}%",
                        delta=f"{item_a_data['pct_within_1hr'].mean() - item_b_data['pct_within_1hr'].mean():+.1f}%"
                    )
                with col2:
                    st.metric(
                        f"{item_b} Overall",
                        f"{item_b_data['pct_within_1hr'].mean():.1f}%",
                        delta=f"{item_b_data['pct_within_1hr'].mean() - item_a_data['pct_within_1hr'].mean():+.1f}%"
                    )
                with col3:
                    avg_diff = diff_df['difference'].mean()
                    st.metric(
                        "Average Difference",
                        f"{avg_diff:+.1f}%",
                        help=f"Positive means {item_a} is better, negative means {item_b} is better"
                    )
    
    with tab5:
        st.header(f"{st.session_state.analysis_mode} Leaderboard")
        st.markdown("Overall performance rankings and statistics.")
        
        # Display rankings table
        st.subheader("Overall Rankings")
        col_name = 'team' if st.session_state.analysis_mode == "Teams" else 'manager'
        display_rankings = rankings_df[['rank', col_name, 'overall_pct', 'total_leads', 'total_contacted_1hr', 'avg_pct', 'min_pct', 'max_pct']].copy()
        display_rankings.columns = ['Rank', st.session_state.analysis_mode[:-1], 'Overall %', 'Total Leads', 'Contacted <1hr', 'Avg %', 'Min %', 'Max %']
        display_rankings['Overall %'] = display_rankings['Overall %'].round(1)
        display_rankings['Avg %'] = display_rankings['Avg %'].round(1)
        display_rankings['Min %'] = display_rankings['Min %'].round(1)
        display_rankings['Max %'] = display_rankings['Max %'].round(1)
        
        st.dataframe(display_rankings, use_container_width=True, hide_index=True)
        
        # Best and worst time slots per item
        st.subheader(f"Best and Worst Time Slots by {st.session_state.analysis_mode[:-1]}")
        for item in items:
            item_data = aggregated_df[aggregated_df[grouping_key] == item].copy()
            if len(item_data) > 0:
                best_slot = item_data.loc[item_data['pct_within_1hr'].idxmax()]
                worst_slot = item_data.loc[item_data['pct_within_1hr'].idxmin()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{item} - Best Slot:**")
                    st.write(f"{best_slot['day_of_week']} {best_slot['time_bucket'].strftime('%H:%M')}: {best_slot['pct_within_1hr']:.1f}% ({best_slot['total_leads']} leads)")
                with col2:
                    st.markdown(f"**{item} - Worst Slot:**")
                    st.write(f"{worst_slot['day_of_week']} {worst_slot['time_bucket'].strftime('%H:%M')}: {worst_slot['pct_within_1hr']:.1f}% ({worst_slot['total_leads']} leads)")
                st.divider()

