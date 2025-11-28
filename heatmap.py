"""
heatmap.py - Visualization Functions
=====================================

THE PURPOSE OF THIS FILE
------------------------
This module transforms processed data into visual insights. It contains all
Plotly-based visualization code—and nothing else. No data processing. No UI.
Pure visualization.

Why isolate visualization? Because visualization libraries evolve. Today we
use Plotly. Tomorrow you might want Altair, Bokeh, or a custom D3.js solution.
When that happens, you rewrite this file. The data processing (logic.py) and
UI (app.py) remain untouched.

WHAT THIS MODULE CREATES
------------------------
1. PERFORMANCE HEATMAP: The main visualization showing response rates by 
   day and time. Green = good, red = needs improvement.

2. SCHEDULE HEATMAP: A companion visualization showing staff coverage.
   Compare this with performance to answer: "Are slow times understaffed?"

WHY PLOTLY?
-----------
We chose Plotly over Matplotlib for one critical reason: INTERACTIVITY.

Hover over any cell and you see exact values:
- "Monday at 09:00 AM"
- "Total leads: 15"
- "Contacted within 1hr: 12"
- "Success rate: 80%"

This transforms a static image into an investigative tool. Users can explore
the data, not just view it.

THE COLOR PHILOSOPHY
--------------------
Colors carry meaning. Our performance heatmap uses a traffic-light palette:
- RED (#d73027): Danger. 0% success rate. Immediate attention needed.
- YELLOW (#fee08b): Caution. 50% success rate. Room for improvement.
- GREEN (#1a9850): Success. 100% success rate. Keep doing this.

This encoding is intuitive. No legend needed to understand "red = bad."
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# Import from our logic module
from logic import generate_time_buckets, create_schedule_lookup


# =============================================================================
# SECTION 1: MAIN PERFORMANCE HEATMAP
# =============================================================================
#
# THE CENTRAL VISUALIZATION
#
# This heatmap answers the fundamental question: "When do we respond well
# to leads, and when do we respond poorly?"
#
# Structure:
# - Rows: Days of the week (Monday at top, Friday at bottom)
# - Columns: Time slots throughout the day (5 AM to 8 PM)
# - Cell color: Performance percentage (red → yellow → green)
# - Cell text: The exact percentage value
# - Hover: Detailed breakdown (leads, contacts, success rate, scheduled reps)
#
# Reading the heatmap:
# - A column of red cells at 8 AM = morning problem across all days
# - A row of red cells on Friday = Friday-specific problem
# - Scattered red cells = inconsistent performance (harder to diagnose)
# =============================================================================

def create_performance_heatmap(aggregated_df, bucket_minutes, schedule_lookup=None):
    """
    Create the main performance heatmap visualization.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    Numbers in a spreadsheet don't reveal patterns. A heatmap does. At a 
    glance, you see which time slots are green (working well) and which are
    red (need attention). This visual pattern recognition is faster and more
    intuitive than scanning rows of numbers.
    
    THE DATA TRANSFORMATION
    -----------------------
    Input: Aggregated DataFrame with one row per (day, time_slot)
    Output: Plotly Figure with a color-coded matrix
    
    The key step is "pivoting" the data:
    - From: Long format (day, time, percentage) with many rows
    - To: Matrix format (rows=days, columns=times, values=percentages)
    
    This matrix format is what the heatmap visualization requires.
    
    THE COLOR SCALE
    ---------------
    We use a 5-point diverging scale from red to green:
    - 0%: Deep red (#d73027) - Complete failure
    - 25%: Orange (#fc8d59) - Poor performance
    - 50%: Yellow (#fee08b) - Moderate performance
    - 75%: Light green (#d9ef8b) - Good performance
    - 100%: Deep green (#1a9850) - Excellent performance
    
    The scale is FIXED from 0-100, not auto-scaled. This ensures:
    - 50% always looks yellow, regardless of your actual data range
    - You can compare heatmaps across different time periods
    - The visual meaning is consistent and predictable
    
    Parameters
    ----------
    aggregated_df : pandas.DataFrame
        Aggregated data from logic.aggregate_by_slot(). Must contain:
        - day_of_week: "Monday", "Tuesday", etc.
        - time_bucket: datetime.time objects
        - pct_within_1hr: Performance percentage (0-100)
        - total_leads: Count of leads in this slot
        - contacted_within_1hr: Count of successful contacts
        
    bucket_minutes : int
        Size of time buckets (15, 30, or 60 minutes).
        Used only for the chart title.
        
    schedule_lookup : dict, optional
        Pre-computed lookup from create_schedule_lookup().
        If provided, hover text will include scheduled rep names.
        If None, hover text shows performance data only.
    
    Returns
    -------
    plotly.graph_objects.Figure
        The complete heatmap figure, ready for display with st.plotly_chart()
        or fig.show().
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: Prepare the data structure
    # -------------------------------------------------------------------------
    # The heatmap needs a 2D matrix. We start with a "long" DataFrame and
    # need to reshape it into a "wide" matrix format.
    
    # Extract unique time slots and sort chronologically
    all_times = sorted(aggregated_df['time_bucket'].unique())
    
    # Define day ordering (we want Monday at top, not alphabetical)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # Only include days that actually appear in the data
    # (handles cases where data only covers certain days)
    days_in_data = [day for day in day_order if day in aggregated_df['day_of_week'].values]
    
    # -------------------------------------------------------------------------
    # STEP 2: Pivot to matrix format
    # -------------------------------------------------------------------------
    # Transform from:
    #   day_of_week | time_bucket | pct_within_1hr
    #   Monday      | 08:00       | 75.0
    #   Monday      | 08:30       | 82.5
    #   Tuesday     | 08:00       | 68.0
    #
    # To:
    #            | 08:00 | 08:30 | ...
    #   Monday   | 75.0  | 82.5  | ...
    #   Tuesday  | 68.0  | ...   | ...
    
    pivot_df = aggregated_df.pivot(
        index='day_of_week',      # Rows
        columns='time_bucket',     # Columns
        values='pct_within_1hr'    # Cell values
    )
    
    # Ensure rows are in our preferred order (Monday first)
    pivot_df = pivot_df.reindex(days_in_data)
    
    # -------------------------------------------------------------------------
    # STEP 3: Create human-readable labels
    # -------------------------------------------------------------------------
    # datetime.time(8, 0) → "08:00 AM"
    # These labels appear on the x-axis
    time_labels = [format_time_label(t) for t in pivot_df.columns]
    
    # -------------------------------------------------------------------------
    # STEP 4: Build rich hover text
    # -------------------------------------------------------------------------
    # When users hover over a cell, they see detailed information.
    # This is where Plotly's interactivity shines.
    hover_text = create_hover_text(
        aggregated_df, 
        days_in_data, 
        pivot_df.columns.tolist(),
        schedule_lookup
    )
    
    # -------------------------------------------------------------------------
    # STEP 5: Construct the Plotly heatmap
    # -------------------------------------------------------------------------
    # go.Heatmap is a low-level Plotly object that gives us full control
    # over every aspect of the visualization.
    
    fig = go.Figure(data=go.Heatmap(
        # The data matrix - this is what determines cell colors
        z=pivot_df.values,
        
        # Axis labels
        x=time_labels,      # Column headers
        y=days_in_data,     # Row headers
        
        # Hover behavior - show our custom text, hide default extras
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        
        # The color scale: a smooth gradient from red to green
        # Each entry is [position, color] where position is 0-1
        colorscale=[
            [0.0, '#d73027'],    # 0% - Deep red (failure)
            [0.25, '#fc8d59'],   # 25% - Orange (poor)
            [0.5, '#fee08b'],    # 50% - Yellow (moderate)
            [0.75, '#d9ef8b'],   # 75% - Light green (good)
            [1.0, '#1a9850']     # 100% - Deep green (excellent)
        ],
        
        # Fix the color scale range - don't auto-scale to data
        # This ensures 50% always looks yellow, regardless of actual data
        zmin=0,
        zmax=100,
        
        # The color bar (legend) on the right side
        colorbar=dict(
            title='% Contacted<br>Within 1 Hour',
            ticksuffix='%'
        ),
        
        # Show the percentage value inside each cell
        texttemplate='%{z:.0f}%',
        textfont=dict(size=12, color='black')
    ))
    
    # -------------------------------------------------------------------------
    # STEP 6: Style the layout
    # -------------------------------------------------------------------------
    # Layout controls everything outside the heatmap itself:
    # title, axes, margins, size
    
    fig.update_layout(
        # Chart title
        title=dict(
            text=f'Lead Response Performance by Time Slot ({bucket_minutes}-min buckets)',
            font=dict(size=18)
        ),
        
        # X-axis (time slots)
        xaxis=dict(
            title='Time of Day',
            tickangle=-45,        # Rotate labels to prevent overlap
            side='bottom'
        ),
        
        # Y-axis (days)
        yaxis=dict(
            title='Day of Week',
            autorange='reversed'  # Monday at TOP (more intuitive)
        ),
        
        # Figure dimensions
        height=400,
        
        # Margins: left, right, top, bottom
        margin=dict(l=100, r=50, t=80, b=100)
    )
    
    return fig


# =============================================================================
# SECTION 2: SCHEDULE OVERLAY HEATMAP
# =============================================================================
#
# THE STAFFING CONTEXT
#
# Performance problems have causes. Sometimes the cause is staffing:
# - "Response times are slow at 8 AM" → "Only 1 person is scheduled at 8 AM"
# - "Friday afternoons are bad" → "Half the team leaves early on Fridays"
#
# This heatmap visualizes WHO is scheduled WHEN, enabling staffing insights.
#
# Structure:
# - Rows: Days of the week
# - Columns: Time slots (same as performance heatmap for easy comparison)
# - Cell color: Number of reps scheduled (darker blue = more coverage)
# - Cell text: The count
# - Hover: Names of scheduled reps
#
# Reading this alongside the performance heatmap:
# - Red cell + low staff count = understaffing problem (fixable!)
# - Red cell + high staff count = training/process problem
# =============================================================================

def create_schedule_heatmap(schedule_df, bucket_minutes):
    """
    Create a heatmap showing staff coverage throughout the week.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    The performance heatmap shows WHAT is happening (response rates).
    This heatmap shows WHO was there (staffing levels). Together, they
    answer: "Is poor performance correlated with understaffing?"
    
    This is actionable intelligence. If you discover that slow periods
    match understaffed periods, you have a concrete solution: add staff.
    
    THE COLOR SCALE
    ---------------
    We use a blue sequential scale (light → dark):
    - Light/white: 0 reps scheduled (no coverage)
    - Light blue: 1-2 reps
    - Medium blue: 3-4 reps
    - Dark blue: 5+ reps (heavy coverage)
    
    Blue is chosen deliberately. It's distinct from the red/green 
    performance colors, preventing visual confusion when comparing heatmaps.
    
    Parameters
    ----------
    schedule_df : pandas.DataFrame
        Schedule data from logic.load_schedule(). Must contain:
        - rep_name: Name of the sales representative
        - day_of_week: "Monday", "Tuesday", etc.
        - start_time: datetime.time when shift starts
        - end_time: datetime.time when shift ends
        
    bucket_minutes : int
        Size of time buckets (15, 30, or 60 minutes).
        Controls granularity of the visualization.
    
    Returns
    -------
    plotly.graph_objects.Figure
        The schedule heatmap figure, ready for display.
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: Generate the time slots for the visualization
    # -------------------------------------------------------------------------
    # These must match the performance heatmap for easy visual comparison
    time_buckets = generate_time_buckets(bucket_minutes, start_hour=5, end_hour=20)
    
    # -------------------------------------------------------------------------
    # STEP 2: Define the days to display
    # -------------------------------------------------------------------------
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # -------------------------------------------------------------------------
    # STEP 3: Build the staffing matrices
    # -------------------------------------------------------------------------
    # We need two matrices:
    # 1. staff_matrix: Count of reps (for coloring)
    # 2. rep_names_matrix: Names of reps (for hover text)
    
    staff_matrix = []       # 2D list of integers
    rep_names_matrix = []   # 2D list of strings
    
    for day in days:
        row_counts = []
        row_names = []
        
        for time_bucket in time_buckets:
            # Find all reps scheduled during this specific slot
            reps = get_scheduled_reps(schedule_df, day, time_bucket)
            
            # Store the count for coloring
            row_counts.append(len(reps))
            
            # Store names for hover (or a message if nobody scheduled)
            row_names.append(', '.join(reps) if reps else 'No coverage')
        
        staff_matrix.append(row_counts)
        rep_names_matrix.append(row_names)
    
    # -------------------------------------------------------------------------
    # STEP 4: Create human-readable time labels
    # -------------------------------------------------------------------------
    time_labels = [format_time_label(t) for t in time_buckets]
    
    # -------------------------------------------------------------------------
    # STEP 5: Construct the heatmap
    # -------------------------------------------------------------------------
    fig = go.Figure(data=go.Heatmap(
        # The data matrix (staff counts)
        z=staff_matrix,
        
        # Axis labels
        x=time_labels,
        y=days,
        
        # Hover text shows the rep names, not just counts
        text=rep_names_matrix,
        hovertemplate='<b>%{y} at %{x}</b><br>Staff: %{z}<br>Reps: %{text}<extra></extra>',
        
        # Blue sequential color scale (distinct from performance colors)
        colorscale='Blues',
        
        # Set color range based on actual data
        zmin=0,
        zmax=max(max(row) for row in staff_matrix) if staff_matrix else 1,
        
        # Color bar
        colorbar=dict(
            title='# of Reps<br>Scheduled'
        ),
        
        # Show count in each cell
        texttemplate='%{z}',
        textfont=dict(size=11, color='white')
    ))
    
    # -------------------------------------------------------------------------
    # STEP 6: Style the layout
    # -------------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=f'Staff Schedule Coverage ({bucket_minutes}-min buckets)',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Time of Day',
            tickangle=-45,
            side='bottom'
        ),
        yaxis=dict(
            title='Day of Week',
            autorange='reversed'  # Monday at top, consistent with performance heatmap
        ),
        height=350,
        margin=dict(l=100, r=50, t=80, b=100)
    )
    
    return fig


# =============================================================================
# SECTION 3: HELPER FUNCTIONS
# =============================================================================
#
# SUPPORTING UTILITIES
#
# These functions handle specific tasks that are used by the main heatmap
# functions. Each does one thing well:
#
# - format_time_label(): Convert time objects to readable strings
# - create_hover_text(): Build detailed hover information
# - get_scheduled_reps(): Find who's working during a time slot
#
# By extracting these into separate functions, the main functions stay
# focused and readable. It also enables reuse—format_time_label() is
# called by both heatmap functions.
# =============================================================================

def format_time_label(time_obj):
    """
    Convert a time object to a human-readable string like "08:00 AM".
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    datetime.time(8, 0) is great for calculations but terrible for display.
    Users expect to see "08:00 AM", not "datetime.time(8, 0)".
    
    This function bridges the gap between machine representation and 
    human-readable format.
    
    THE FORMAT
    ----------
    We use 12-hour time with AM/PM because it's more familiar to most users.
    "%I:%M %p" produces formats like "08:00 AM", "02:30 PM".
    
    For 24-hour time, change to "%H:%M" → "08:00", "14:30"
    
    Parameters
    ----------
    time_obj : datetime.time
        The time object to format (e.g., datetime.time(8, 30))
    
    Returns
    -------
    str
        Formatted time string (e.g., "08:30 AM")
    """
    # Combine with today's date (we only need the time, but strftime needs datetime)
    dt = datetime.combine(datetime.today(), time_obj)
    
    # Format as 12-hour time with AM/PM
    return dt.strftime('%I:%M %p')


def create_hover_text(aggregated_df, days, time_buckets, schedule_lookup=None):
    """
    Create detailed hover text for each cell in the performance heatmap.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    The heatmap color shows the big picture. Hover text shows the details.
    When a user sees a red cell and hovers over it, they want to know:
    - Exactly what time slot is this?
    - How many leads were there?
    - How many were contacted in time?
    - Who was supposed to be working?
    
    This context helps users investigate problems, not just identify them.
    
    THE FORMAT
    ----------
    Each hover shows (example):
    
        Monday at 08:00 AM
        Total leads: 15
        Contacted within 1hr: 12
        Success rate: 80.0%
        
        Scheduled Reps:
        Alice, Bob, Carol
    
    The schedule section only appears if schedule_lookup is provided.
    
    Parameters
    ----------
    aggregated_df : pandas.DataFrame
        The aggregated performance data.
        
    days : list
        List of days in display order (e.g., ['Monday', 'Tuesday', ...])
        
    time_buckets : list
        List of time bucket values (datetime.time objects)
        
    schedule_lookup : dict, optional
        Mapping from (day, time_bucket) to list of rep names.
        If provided, rep names are included in hover text.
    
    Returns
    -------
    list of lists
        A 2D matrix of strings matching the heatmap dimensions.
        hover_matrix[row][col] is the hover text for that cell.
    """
    hover_matrix = []
    
    for day in days:
        row = []
        
        for time_bucket in time_buckets:
            # Find the aggregated data for this specific cell
            cell_data = aggregated_df[
                (aggregated_df['day_of_week'] == day) & 
                (aggregated_df['time_bucket'] == time_bucket)
            ]
            
            if len(cell_data) > 0:
                # We have data for this cell - build detailed hover text
                pct = cell_data['pct_within_1hr'].values[0]
                total = cell_data['total_leads'].values[0]
                contacted = cell_data['contacted_within_1hr'].values[0]
                
                # Build the hover text with HTML formatting
                # <b> tags make text bold
                # <br> creates line breaks
                text = f"<b>{day} at {format_time_label(time_bucket)}</b><br>"
                text += f"Total leads: {total}<br>"
                text += f"Contacted within 1hr: {int(contacted)}<br>"
                text += f"Success rate: {pct:.1f}%"
                
                # Add scheduled reps if we have schedule data
                if schedule_lookup and (day, time_bucket) in schedule_lookup:
                    reps = schedule_lookup[(day, time_bucket)]
                    text += f"<br><br><b>Scheduled Reps:</b><br>{', '.join(reps)}"
            else:
                # No data for this cell (no leads received during this slot)
                text = f"<b>{day} at {format_time_label(time_bucket)}</b><br>No leads received"
            
            row.append(text)
        
        hover_matrix.append(row)
    
    return hover_matrix


def get_scheduled_reps(schedule_df, day_of_week, time_bucket):
    """
    Find which reps are scheduled during a specific time slot.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    The schedule DataFrame stores shifts (start_time to end_time).
    We need to know who's working at a SPECIFIC time (e.g., 2:00 PM).
    
    This function answers: "Given a day and time, who should be working?"
    
    THE LOGIC
    ---------
    A rep is working during a time slot if:
        start_time <= time_bucket < end_time
    
    We use < for end_time (not <=) to prevent double-counting.
    If Bob works 9-12 and Carol works 12-5, at 12:00:
    - Bob: 9:00 <= 12:00 < 12:00 → FALSE (Bob's shift ended)
    - Carol: 12:00 <= 12:00 < 17:00 → TRUE (Carol's shift started)
    
    Parameters
    ----------
    schedule_df : pandas.DataFrame
        Schedule data with rep_name, day_of_week, start_time, end_time.
        
    day_of_week : str
        The day to check (e.g., "Monday").
        
    time_bucket : datetime.time
        The specific time to check (e.g., datetime.time(14, 0) for 2 PM).
    
    Returns
    -------
    list
        Names of all reps scheduled during this slot.
        Empty list if no coverage.
    """
    # Filter to the requested day first (reduces the data we iterate over)
    day_schedule = schedule_df[schedule_df['day_of_week'] == day_of_week]
    
    # Check each shift to see if it covers this time
    scheduled_reps = []
    for _, row in day_schedule.iterrows():
        # Time is within shift if: start <= time < end
        if row['start_time'] <= time_bucket < row['end_time']:
            scheduled_reps.append(row['rep_name'])
    
    return scheduled_reps


# =============================================================================
# SECTION 4: COMBINED VISUALIZATION
# =============================================================================
#
# CONVENIENCE WRAPPER
#
# Users often want both heatmaps together. This function creates them
# with proper integration: the schedule lookup is computed once and used
# in both the performance heatmap hover text AND the schedule heatmap.
# =============================================================================

def create_combined_view(aggregated_df, schedule_df, bucket_minutes):
    """
    Create both heatmaps together with integrated schedule information.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    Creating both heatmaps involves a shared computation: the schedule lookup.
    Rather than computing it twice (once for performance hover text, once for
    schedule heatmap), we compute it once and share it.
    
    This function encapsulates that optimization and provides a simple
    interface: "Give me data, I'll give you both visualizations."
    
    THE INTEGRATION
    ---------------
    The schedule data appears in two places:
    1. Performance heatmap HOVER TEXT: "Scheduled Reps: Alice, Bob"
    2. Schedule heatmap: Dedicated visualization of coverage
    
    This function ensures both use the same underlying data.
    
    Parameters
    ----------
    aggregated_df : pandas.DataFrame
        Aggregated lead performance data from logic.aggregate_by_slot()
        
    schedule_df : pandas.DataFrame
        Staff schedule data from logic.load_schedule()
        
    bucket_minutes : int
        Size of time buckets in minutes (15, 30, or 60)
    
    Returns
    -------
    tuple (performance_fig, schedule_fig)
        Two Plotly figures ready for display.
        
    Example
    -------
    >>> perf_fig, sched_fig = create_combined_view(agg_df, sched_df, 30)
    >>> st.plotly_chart(perf_fig)
    >>> st.plotly_chart(sched_fig)
    """
    # Compute schedule lookup once (used by performance heatmap hover text)
    schedule_lookup = create_schedule_lookup(schedule_df, bucket_minutes)
    
    # Create performance heatmap with schedule integration
    performance_fig = create_performance_heatmap(
        aggregated_df, 
        bucket_minutes, 
        schedule_lookup
    )
    
    # Create schedule heatmap
    schedule_fig = create_schedule_heatmap(schedule_df, bucket_minutes)
    
    return performance_fig, schedule_fig
