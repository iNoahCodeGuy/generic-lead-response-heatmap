"""
app.py - Lead Response Heatmap Dashboard
=========================================

THE PURPOSE OF THIS FILE
------------------------
This is the presentation layer—the face of the application. It handles
everything the user sees and interacts with:

- Sidebar controls (file uploads, settings)
- Page layout and styling
- Displaying visualizations
- Showing summary statistics

WHY STREAMLIT?
--------------
Building a web application traditionally requires HTML, CSS, JavaScript,
and a backend framework. That's a lot of complexity for a data visualization
tool.

Streamlit eliminates this complexity. You write Python. You get a web app.
Every st.something() call creates a UI element:

    st.title("Hello")           → Renders a heading
    st.file_uploader(...)       → Renders a file upload widget
    st.plotly_chart(fig)        → Renders an interactive chart

The entire UI is defined in ~200 lines of Python. No templates. No routes.
No JavaScript. Pure simplicity.

HOW THIS FILE IS STRUCTURED
---------------------------
The file reads top-to-bottom in the order things appear on screen:

1. PAGE CONFIGURATION - Browser tab settings
2. CUSTOM STYLING - CSS for visual polish
3. SIDEBAR - Controls on the left
4. MAIN PAGE HEADER - Title and description
5. DATA LOADING - Get the data ready
6. MAIN CONTENT - The heatmaps and statistics
7. FOOTER - Credits at the bottom

This structure makes the code self-documenting. Want to change the header?
Look in Section 4. Want to add a new control? Look in Section 3.

HOW TO RUN THIS APPLICATION
---------------------------
From the terminal, in the project directory:

    streamlit run app.py

Then open your browser to http://localhost:8501

Streamlit will hot-reload when you save changes to this file.
"""

import streamlit as st
import pandas as pd
import os

# Import from our custom modules
# logic.py handles all data processing
# heatmap.py handles all visualization
from logic import (
    load_leads,
    load_schedule,
    aggregate_by_slot,
    calculate_summary_stats,
    create_schedule_lookup
)
from heatmap import (
    create_performance_heatmap,
    create_schedule_heatmap
)


# =============================================================================
# SECTION 1: PAGE CONFIGURATION
# =============================================================================
#
# THE FIRST STREAMLIT COMMAND
#
# st.set_page_config() MUST be the first Streamlit command in the script.
# If you call st.write() or any other st function before this, you'll get
# an error.
#
# This function configures browser-level settings:
# - page_title: What appears in the browser tab
# - page_icon: The favicon (emoji or image path)
# - layout: "centered" (default, ~600px max) or "wide" (full width)
# - initial_sidebar_state: "expanded" or "collapsed"
# =============================================================================

st.set_page_config(
    page_title="Lead Response Heatmap",
    page_icon="📊",
    layout="wide",                    # Use full browser width for heatmaps
    initial_sidebar_state="expanded"  # Sidebar open by default
)


# =============================================================================
# SECTION 2: CUSTOM STYLING
# =============================================================================
#
# CSS CUSTOMIZATION
#
# Streamlit has good defaults, but sometimes you want custom styling.
# st.markdown() with unsafe_allow_html=True lets you inject CSS.
#
# The CSS below customizes:
# - .main-title: The page heading
# - .subtitle: The description below the heading
# - .metric-card: Card styling for statistics (if needed)
# - .success-box: Success message styling (if needed)
#
# WHY CSS IN PYTHON?
# It's a trade-off. Keeping CSS here means:
# - Everything in one file (simpler deployment)
# - Styling tied to the component it styles
# - Less flexible than a full CSS file
#
# For a larger application, you'd move this to a separate CSS file.
# =============================================================================

st.markdown("""
    <style>
    /* Main title - large, bold, blue */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle - smaller, gray, provides context */
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Metric cards - used for summary statistics */
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Success message box - green background for confirmations */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 3: SIDEBAR - FILE UPLOADS AND CONTROLS
# =============================================================================
#
# THE CONTROL PANEL
#
# All user controls live in the sidebar. This keeps the main area clean
# for visualization while providing easy access to settings.
#
# SIDEBAR ELEMENTS:
# 1. Data Input section (file uploads)
# 2. Use Sample Data checkbox (for quick demos)
# 3. Settings section (bucket size, schedule toggle)
#
# STREAMLIT SIDEBAR API:
# - st.sidebar.header()     → Section heading
# - st.sidebar.expander()   → Collapsible help text
# - st.sidebar.file_uploader() → File upload widget
# - st.sidebar.checkbox()   → Toggle option
# - st.sidebar.selectbox()  → Dropdown menu
#
# The sidebar.* prefix puts the element in the sidebar instead of main area.
# =============================================================================

# -----------------------------------------------------------------------------
# DATA INPUT SECTION
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Data Input")

# Expandable help text explaining file format requirements
# Using an expander keeps the sidebar clean when help isn't needed
with st.sidebar.expander("ℹ️ File Format Requirements"):
    st.markdown("""
    **leads.csv** must have columns:
    - `lead_id` - Unique identifier
    - `received_time` - When lead was received
    - `first_contact_time` - When first contact was made
    - `first_contact_channel` - Call, Email, SMS, etc.
    - `rep_name` - (Optional) Sales rep name
    
    **schedule.csv** must have columns:
    - `rep_name` - Sales rep name
    - `day_of_week` - Monday, Tuesday, etc.
    - `start_time` - Shift start (HH:MM)
    - `end_time` - Shift end (HH:MM)
    """)

# File upload widgets
# type=['csv'] restricts uploads to CSV files only
leads_file = st.sidebar.file_uploader(
    "Upload Leads CSV",
    type=['csv'],
    help="Upload your leads data file"
)

schedule_file = st.sidebar.file_uploader(
    "Upload Schedule CSV (Optional)",
    type=['csv'],
    help="Upload staff schedule for overlay visualization"
)

# -----------------------------------------------------------------------------
# SAMPLE DATA OPTION
# -----------------------------------------------------------------------------
# Visual separator in the sidebar
st.sidebar.markdown("---")

# Checkbox to use included sample data
# value=True means it's checked by default (for easy demos)
use_sample_data = st.sidebar.checkbox(
    "📂 Use Sample Data",
    value=True,
    help="Check this to use the included sample data files for testing"
)

# -----------------------------------------------------------------------------
# SETTINGS SECTION
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")

# Time bucket size dropdown
# index=1 means "30" (the second option) is selected by default
bucket_minutes = st.sidebar.selectbox(
    "Time Bucket Size",
    options=[15, 30, 60],
    index=1,  # Default to 30 minutes
    help="Choose how to group time slots (smaller = more granular)"
)

# Toggle for schedule overlay heatmap
show_schedule = st.sidebar.checkbox(
    "📅 Show Schedule Overlay",
    value=True,
    help="Display a separate heatmap showing staff coverage"
)


# =============================================================================
# SECTION 4: MAIN PAGE HEADER
# =============================================================================
#
# THE FIRST THING USERS SEE
#
# A clear title and description set expectations. Users immediately know:
# - What this tool does (visualize lead response performance)
# - Why it matters (see how quickly your team responds)
#
# We use custom HTML/CSS (from Section 2) for styling that goes beyond
# Streamlit's built-in st.title() and st.write().
# =============================================================================

# Main title with custom styling
st.markdown(
    '<p class="main-title">📊 Lead Response Heatmap Dashboard</p>', 
    unsafe_allow_html=True
)

# Descriptive subtitle
st.markdown(
    '<p class="subtitle">Visualize how quickly your team responds to leads throughout the workweek</p>', 
    unsafe_allow_html=True
)


# =============================================================================
# SECTION 5: DATA LOADING
# =============================================================================
#
# GETTING DATA INTO THE APPLICATION
#
# Data can come from two sources:
# 1. SAMPLE DATA: Pre-included CSV files in the sample_data/ folder
# 2. UPLOADED FILES: User's own CSV files via the file uploader
#
# The load_data() function handles both cases transparently. The rest of
# the application just sees a DataFrame—it doesn't care where it came from.
#
# PATH HANDLING
# -------------
# When running as a Streamlit app, the working directory might not be the
# project folder. We use os.path.dirname(os.path.abspath(__file__)) to
# find the script's directory, then build paths relative to that.
# This ensures sample data is found regardless of where you run from.
# =============================================================================

def load_data():
    """
    Load leads and schedule data based on user selection.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    Data loading has multiple code paths (sample vs upload) and error
    handling. Putting this in a function keeps the main flow clean and
    makes the logic testable.
    
    THE LOGIC
    ---------
    1. If "Use Sample Data" is checked:
       - Load from sample_data/leads.csv and sample_data/schedule.csv
       - Show success/error messages in sidebar
    
    2. If files are uploaded:
       - Load from the uploaded file objects
       - Show success/error messages with row counts
    
    3. If neither:
       - Return (None, None) and the main section will show instructions
    
    STREAMLIT STATE
    ---------------
    This function reads from sidebar widgets (use_sample_data, leads_file,
    schedule_file). In Streamlit, widget values are available as soon as
    they're created—no explicit form submission needed.
    
    Returns
    -------
    tuple (leads_df, schedule_df)
        DataFrames with loaded data, or None if no data available.
    """
    leads_df = None
    schedule_df = None
    
    # Build paths to sample data files
    # __file__ is the path to this script (app.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_leads_path = os.path.join(script_dir, 'sample_data', 'leads.csv')
    sample_schedule_path = os.path.join(script_dir, 'sample_data', 'schedule.csv')
    
    # -------------------------------------------------------------------------
    # Load leads data
    # -------------------------------------------------------------------------
    if use_sample_data:
        # Try to load sample data
        if os.path.exists(sample_leads_path):
            leads_df = load_leads(sample_leads_path)
            st.sidebar.success("✅ Sample leads loaded")
        else:
            st.sidebar.error("❌ Sample leads.csv not found")
    elif leads_file is not None:
        # Load from uploaded file
        try:
            leads_df = load_leads(leads_file)
            st.sidebar.success(f"✅ Loaded {len(leads_df)} leads")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading leads: {str(e)}")
    
    # -------------------------------------------------------------------------
    # Load schedule data
    # -------------------------------------------------------------------------
    if use_sample_data:
        if os.path.exists(sample_schedule_path):
            schedule_df = load_schedule(sample_schedule_path)
            st.sidebar.success("✅ Sample schedule loaded")
        else:
            st.sidebar.warning("⚠️ Sample schedule.csv not found")
    elif schedule_file is not None:
        try:
            schedule_df = load_schedule(schedule_file)
            st.sidebar.success(f"✅ Loaded schedule for {schedule_df['rep_name'].nunique()} reps")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading schedule: {str(e)}")
    
    return leads_df, schedule_df


# Execute the data loading
# This happens on every script run (Streamlit re-runs the entire script on interaction)
leads_df, schedule_df = load_data()


# =============================================================================
# SECTION 6: MAIN CONTENT - ANALYSIS AND VISUALIZATION
# =============================================================================
#
# THE HEART OF THE APPLICATION
#
# This section displays the actual analysis—but only if we have data.
# If no data is loaded, we show instructions instead.
#
# WHEN DATA EXISTS:
# 1. Summary Statistics: Four key metrics at a glance
# 2. Performance Heatmap: The main visualization
# 3. Schedule Heatmap: Optional staffing overlay
# 4. Raw Data Table: Expandable detail view
#
# STREAMLIT LAYOUT TOOLS:
# - st.columns(4): Create 4 equal columns (for metrics)
# - st.metric(): Display a number with label and optional delta
# - st.plotly_chart(): Render a Plotly figure
# - st.expander(): Collapsible section
# - st.dataframe(): Interactive data table
# =============================================================================

if leads_df is not None:
    # =========================================================================
    # WE HAVE DATA - Show the full analysis
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # STEP 1: Summary Statistics
    # -------------------------------------------------------------------------
    # These numbers appear at the top, giving immediate context before users
    # dive into the detailed heatmap.
    
    st.markdown("---")  # Visual separator
    st.subheader("📈 Summary Statistics")
    
    # Calculate summary stats using our logic module
    stats = calculate_summary_stats(leads_df)
    
    # Display in a 4-column layout
    # st.columns() creates column containers; you use `with` to put content in them
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Leads",
            value=f"{stats['total_leads']:,}"  # :, adds thousand separators
        )
    
    with col2:
        st.metric(
            label="Contacted Within 1 Hour",
            value=f"{stats['contacted_within_1hr']:,}",
            delta=f"{stats['pct_within_1hr']}%"  # Shows as green/red indicator
        )
    
    with col3:
        st.metric(
            label="Avg Response Time",
            value=f"{stats['avg_response_minutes']:.0f} min"  # .0f = no decimals
        )
    
    with col4:
        st.metric(
            label="Median Response Time",
            value=f"{stats['median_response_minutes']:.0f} min"
        )
    
    # -------------------------------------------------------------------------
    # STEP 2: Performance Heatmap
    # -------------------------------------------------------------------------
    # The main visualization. This is what users came for.
    
    st.markdown("---")
    st.subheader("🗓️ Performance Heatmap")
    st.markdown(
        "This heatmap shows the percentage of leads contacted within 1 hour, "
        "broken down by day and time slot. **Green = good performance**, **Red = needs improvement**."
    )
    
    # Aggregate raw leads into (day, time) buckets with performance metrics
    aggregated_df = aggregate_by_slot(leads_df, bucket_minutes)
    
    # If we have schedule data, create a lookup for hover text integration
    schedule_lookup = None
    if schedule_df is not None:
        schedule_lookup = create_schedule_lookup(schedule_df, bucket_minutes)
    
    # Create the Plotly heatmap figure
    performance_fig = create_performance_heatmap(
        aggregated_df, 
        bucket_minutes, 
        schedule_lookup
    )
    
    # Render the figure
    # use_container_width=True makes it fill the available width
    st.plotly_chart(performance_fig, use_container_width=True)
    
    # -------------------------------------------------------------------------
    # STEP 3: Schedule Overlay (Optional)
    # -------------------------------------------------------------------------
    # Shows WHO was working WHEN. Helps diagnose staffing issues.
    
    if show_schedule and schedule_df is not None:
        st.markdown("---")
        st.subheader("👥 Staff Schedule Coverage")
        st.markdown(
            "This heatmap shows how many sales reps are scheduled during each time slot. "
            "Compare this with the performance heatmap to identify staffing gaps."
        )
        
        # Create and display the schedule heatmap
        schedule_fig = create_schedule_heatmap(schedule_df, bucket_minutes)
        st.plotly_chart(schedule_fig, use_container_width=True)
    
    # -------------------------------------------------------------------------
    # STEP 4: Raw Data Table (Expandable)
    # -------------------------------------------------------------------------
    # For users who want to see the numbers behind the visualization.
    # Using an expander keeps the default view clean.
    
    with st.expander("📋 View Raw Aggregated Data"):
        st.markdown("This table shows the underlying data used to create the heatmap.")
        
        # Format the data for display
        display_df = aggregated_df.copy()
        
        # Convert time objects to readable strings
        display_df['time_bucket'] = display_df['time_bucket'].apply(
            lambda x: x.strftime('%I:%M %p')
        )
        
        # Rename columns to be user-friendly
        display_df = display_df.rename(columns={
            'day_of_week': 'Day',
            'time_bucket': 'Time Slot',
            'total_leads': 'Total Leads',
            'contacted_within_1hr': 'Contacted <1hr',
            'pct_within_1hr': '% Within 1hr'
        })
        
        # Display as an interactive table
        st.dataframe(display_df, use_container_width=True)

else:
    # =========================================================================
    # NO DATA - Show instructions
    # =========================================================================
    # Guide users on how to get started
    
    st.info(
        "👆 **To get started:**\n\n"
        "1. Check **'Use Sample Data'** in the sidebar to try the demo, OR\n"
        "2. Upload your own **leads.csv** file\n\n"
        "Once data is loaded, the heatmap will appear here."
    )
    
    # Show what they'll see once data is loaded
    st.markdown("---")
    st.subheader("🔍 What You'll See")
    st.markdown("""
    This tool will generate:
    
    - **Performance Heatmap**: A color-coded grid showing response times
      - 🟢 Green cells = Great! Most leads contacted within 1 hour
      - 🟡 Yellow cells = Moderate performance
      - 🔴 Red cells = Problem areas needing attention
    
    - **Summary Statistics**: Key metrics at a glance
      - Total leads analyzed
      - Percentage contacted within 1 hour
      - Average and median response times
    
    - **Schedule Overlay**: (Optional) Staff coverage visualization
      - See which reps are scheduled when
      - Identify understaffed time slots
    """)


# =============================================================================
# SECTION 7: FOOTER
# =============================================================================
#
# CLOSING THE PAGE
#
# A simple footer with credits. Keeps the page looking complete and
# professional.
#
# Using raw HTML for precise centering and styling.
# =============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.9rem;'>"
    "Lead Response Heatmap Dashboard | Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)
