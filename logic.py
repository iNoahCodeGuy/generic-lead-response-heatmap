"""
logic.py - Core Data Processing Functions
==========================================

THE PURPOSE OF THIS FILE
------------------------
This module is the analytical engine of the Lead Response Heatmap Dashboard.
It contains all pandas-based computations—and nothing else. No visualization
code. No UI code. Pure data transformation.

Why isolate the logic? Because business rules change. Today we measure 
"contacted within 1 hour." Tomorrow it might be 30 minutes. Next month you 
might want to track multiple thresholds. When that happens, you modify this 
file and this file alone. The visualization and UI remain untouched.

WHAT THIS MODULE DOES
---------------------
1. LOADING: Reads CSV files and converts strings to proper data types
2. COMPUTING: Calculates response times and success metrics
3. BUCKETING: Groups continuous timestamps into discrete time slots
4. AGGREGATING: Summarizes performance by day and time
5. SCHEDULING: Determines which reps are working during each slot

THE DATA FLOW
-------------
Raw CSV → Parsed DataFrame → Response Metrics Added → Time Buckets Added → 
Grouped & Aggregated → Ready for Visualization

Each function handles one transformation. Chain them together and you have
your analysis pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# =============================================================================
# SECTION 1: LOADING AND PARSING DATA
# =============================================================================
#
# THE PROBLEM: CSV files store everything as text. The timestamp "2024-01-15 
# 09:30:00" is just a string of characters. You cannot subtract one string 
# from another to calculate duration.
#
# THE SOLUTION: Convert text to proper DateTime objects. Then pandas can
# perform time arithmetic: first_contact_time - received_time = response_time.
#
# This section transforms raw text files into analysis-ready DataFrames.
# =============================================================================

def load_leads(file_or_path):
    """
    Load and parse the leads CSV file into an analysis-ready DataFrame.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    CSV files are universal—every CRM can export them. But CSV stores 
    everything as text. This function bridges that gap: text in, typed 
    data out. The returned DataFrame has proper DateTime columns that 
    support arithmetic operations.
    
    THE CRITICAL DETAIL: errors='coerce'
    ------------------------------------
    Some leads are never contacted. Their first_contact_time is blank.
    Rather than failing on these rows, we use errors='coerce' to convert
    blanks to NaT (Not a Time)—the DateTime equivalent of null/None.
    This lets us handle uncontacted leads gracefully in later calculations.
    
    Parameters
    ----------
    file_or_path : str or file-like object
        Either a file path string (e.g., "data/leads.csv") or an uploaded
        file object from Streamlit's file_uploader widget.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns including:
        - lead_id: Unique identifier for each lead
        - received_time: DateTime when the lead arrived
        - first_contact_time: DateTime of first contact (or NaT if never contacted)
        - first_contact_channel: How contact was made (Call, Email, SMS, etc.)
    
    Example
    -------
    >>> df = load_leads("sample_data/leads.csv")
    >>> df['received_time'].dtype
    datetime64[ns]
    """
    # Read the raw CSV—at this point, all columns are strings
    df = pd.read_csv(file_or_path)
    
    # Convert received_time: This should never be blank (every lead has an arrival time)
    df['received_time'] = pd.to_datetime(df['received_time'])
    
    # Convert first_contact_time: May be blank for uncontacted leads
    # errors='coerce' converts unparseable values to NaT instead of raising an error
    df['first_contact_time'] = pd.to_datetime(df['first_contact_time'], errors='coerce')
    
    return df


def load_schedule(file_or_path):
    """
    Load and parse the staff schedule CSV into an analysis-ready DataFrame.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    The schedule data tells us WHO is working WHEN. Combined with performance
    data, this reveals staffing insights: Are slow response times correlated
    with understaffed periods? Do certain reps perform better at certain times?
    
    THE DATA FORMAT
    ---------------
    Each row represents one shift for one rep:
        rep_name,day_of_week,start_time,end_time
        "Alice","Monday","08:00","17:00"
        "Bob","Monday","12:00","21:00"
    
    Times use 24-hour format (HH:MM) for unambiguous parsing.
    
    Parameters
    ----------
    file_or_path : str or file-like object
        Path to schedule CSV or Streamlit file upload object.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - rep_name: Sales representative's name
        - day_of_week: "Monday", "Tuesday", etc.
        - start_time: datetime.time object for shift start
        - end_time: datetime.time object for shift end
    """
    # Read the raw CSV
    df = pd.read_csv(file_or_path)
    
    # Convert time strings to time objects
    # Format "08:00" → datetime.time(8, 0)
    # We parse as datetime first, then extract just the time component
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M').dt.time
    
    return df


# =============================================================================
# SECTION 2: COMPUTING RESPONSE METRICS
# =============================================================================
#
# THE CORE QUESTION: How quickly did we respond to each lead?
#
# This is the fundamental calculation that drives the entire analysis.
# We measure the gap between lead arrival and first contact, then classify
# each lead as a success (contacted within 1 hour) or not.
#
# WHY ONE HOUR?
# Research shows leads contacted within 60 minutes convert at dramatically
# higher rates. This threshold is configurable—you could change it to 30
# minutes or 2 hours based on your business needs.
# =============================================================================

def compute_response_metrics(df):
    """
    Calculate response time metrics for each lead.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    Raw lead data has timestamps. We need metrics. This function transforms
    "received at 09:00, contacted at 09:45" into "response time: 45 minutes,
    contacted within 1 hour: True".
    
    THE CALCULATION
    ---------------
    response_minutes = (first_contact_time - received_time) in minutes
    contact_within_1hr = True if response_minutes <= 60, else False
    
    HANDLING EDGE CASES
    -------------------
    - Never contacted: first_contact_time is NaT → response_minutes is NaN → 
      contact_within_1hr is False
    - Contacted before received (data error): Produces negative response time.
      This indicates a data quality issue that should be investigated.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'received_time' and 'first_contact_time' columns.
        Both must be datetime64 type (use load_leads() to ensure this).
    
    Returns
    -------
    pandas.DataFrame
        Original DataFrame with two new columns:
        - response_minutes: Float, minutes between received and first contact
        - contact_within_1hr: Boolean, True if response_minutes <= 60
    
    Example
    -------
    >>> df = load_leads("leads.csv")
    >>> df = compute_response_metrics(df)
    >>> df[['received_time', 'first_contact_time', 'response_minutes', 'contact_within_1hr']].head()
    """
    # Create a copy to avoid modifying the original DataFrame
    # This is a pandas best practice—functions should not have side effects
    df = df.copy()
    
    # Calculate time difference as a timedelta object
    # timedelta represents a duration: "2 hours, 30 minutes, 15 seconds"
    time_difference = df['first_contact_time'] - df['received_time']
    
    # Convert timedelta to minutes
    # .dt.total_seconds() extracts the duration in seconds
    # Divide by 60 to get minutes as a float (e.g., 45.5 minutes)
    df['response_minutes'] = time_difference.dt.total_seconds() / 60
    
    # Create the success flag: Was this lead contacted within 1 hour?
    # This is the metric we'll aggregate to create the heatmap
    df['contact_within_1hr'] = df['response_minutes'] <= 60
    
    # Handle uncontacted leads explicitly
    # When first_contact_time is NaT, response_minutes is NaN, and the
    # comparison above produces False. But let's be explicit about it
    # to make the code self-documenting.
    df.loc[df['first_contact_time'].isna(), 'contact_within_1hr'] = False
    
    return df


# =============================================================================
# SECTION 3: TIME BUCKETING
# =============================================================================
#
# THE PROBLEM: Raw timestamps are too precise for pattern analysis.
#
# Consider two leads: one arrives at 08:17:23, another at 08:42:51.
# If we try to analyze performance at these exact times, we'll have at most
# one or two leads per timestamp—too few to calculate meaningful percentages.
#
# THE SOLUTION: Group similar times together.
#
# "Time bucketing" rounds timestamps down to fixed intervals:
# - With 30-minute buckets: 08:17 → 08:00, 08:42 → 08:30
# - Now we can group all leads from 08:00-08:30 and calculate: 
#   "What percentage were contacted within 1 hour?"
#
# This is the same principle behind histograms: continuous data becomes
# discrete categories that reveal distribution patterns.
# =============================================================================

def bucket_time(dt, bucket_minutes):
    """
    Round a datetime DOWN to the nearest time bucket boundary.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    Individual timestamps are too granular for analysis. By flooring times
    to bucket boundaries, we can aggregate leads into analyzable groups.
    This is the mathematical foundation of the heatmap visualization.
    
    THE ALGORITHM
    -------------
    1. Convert time to minutes since midnight (08:15 → 495 minutes)
    2. Integer divide by bucket size to find bucket number (495 // 30 = 16)
    3. Multiply back to get bucket start (16 * 30 = 480 minutes)
    4. Convert back to time (480 minutes → 08:00)
    
    The key insight: integer division (//) automatically floors the result.
    495 // 30 = 16, not 16.5. This discards the remainder, achieving our goal.
    
    Parameters
    ----------
    dt : datetime
        The datetime to bucket. Only the time component is used.
    bucket_minutes : int
        Size of each bucket in minutes. Common values: 15, 30, 60.
        Must evenly divide 60 for clean hour boundaries.
    
    Returns
    -------
    datetime.time
        The bucket's start time (the floored value).
    
    Examples
    --------
    With 30-minute buckets:
    >>> bucket_time(datetime(2024, 1, 15, 8, 15), 30)
    datetime.time(8, 0)    # 08:15 → 08:00
    
    >>> bucket_time(datetime(2024, 1, 15, 8, 45), 30)
    datetime.time(8, 30)   # 08:45 → 08:30
    
    >>> bucket_time(datetime(2024, 1, 15, 9, 0), 30)
    datetime.time(9, 0)    # 09:00 → 09:00 (already on boundary)
    """
    # Step 1: Convert to total minutes from midnight
    # 08:15 → 8*60 + 15 = 495 minutes
    total_minutes = dt.hour * 60 + dt.minute
    
    # Step 2: Floor to bucket boundary using integer division
    # 495 // 30 = 16 (bucket number)
    # 16 * 30 = 480 (bucket start in minutes)
    floored_minutes = (total_minutes // bucket_minutes) * bucket_minutes
    
    # Step 3: Convert back to hours and minutes
    # 480 // 60 = 8 (hours)
    # 480 % 60 = 0 (remaining minutes)
    new_hour = floored_minutes // 60
    new_minute = floored_minutes % 60
    
    # Step 4: Return as a time object
    # We use a dummy date (2000-01-01) because we only need the time
    return datetime(2000, 1, 1, new_hour, new_minute).time()


def add_time_buckets(df, bucket_minutes):
    """
    Add time bucket and day-of-week columns to a leads DataFrame.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    The heatmap has two axes: day of week (rows) and time of day (columns).
    This function extracts both dimensions from the received_time timestamp,
    preparing the data for groupby aggregation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'received_time' column (datetime64 type).
    bucket_minutes : int
        Size of time buckets in minutes (15, 30, or 60).
    
    Returns
    -------
    pandas.DataFrame
        Original DataFrame with two new columns:
        - day_of_week: String like "Monday", "Tuesday", etc.
        - time_bucket: datetime.time representing the bucket start
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Extract day of week as a human-readable string
    # pandas .dt.day_name() returns "Monday", "Tuesday", etc.
    df['day_of_week'] = df['received_time'].dt.day_name()
    
    # Apply bucketing to each timestamp
    # .apply() calls bucket_time() for each row
    df['time_bucket'] = df['received_time'].apply(
        lambda x: bucket_time(x, bucket_minutes)
    )
    
    return df


# =============================================================================
# SECTION 4: AGGREGATING DATA FOR HEATMAP
# =============================================================================
#
# THE TRANSFORMATION: Individual leads → Slot-level statistics
#
# Input:  100 individual lead records with timestamps and response times
# Output: 1 row per (day, time_bucket) combination with aggregated metrics
#
# For example, if we have 15 leads on Monday between 09:00-09:30:
# - 12 were contacted within 1 hour
# - 3 were not
# - Performance: 80%
#
# This aggregated data is exactly what the heatmap needs: one value per cell.
# =============================================================================

def aggregate_by_slot(df, bucket_minutes):
    """
    Group leads by (day, time_bucket) and calculate performance metrics.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    This is the analytical core of the application. It transforms a list of
    individual leads into a performance summary by time slot. Each row in the
    output becomes one cell in the heatmap.
    
    THE PIPELINE
    ------------
    This function orchestrates the full transformation:
    1. Compute response metrics (via compute_response_metrics)
    2. Add time buckets (via add_time_buckets)
    3. Group by day and time bucket
    4. Calculate: count, success count, success percentage
    5. Sort for consistent ordering
    
    THE OUTPUT FORMAT
    -----------------
    One row per unique (day_of_week, time_bucket) combination:
    
    | day_of_week | time_bucket | total_leads | contacted_within_1hr | pct_within_1hr |
    |-------------|-------------|-------------|----------------------|----------------|
    | Monday      | 08:00       | 12          | 10                   | 83.3           |
    | Monday      | 08:30       | 8           | 5                    | 62.5           |
    | ...         | ...         | ...         | ...                  | ...            |
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw leads data with received_time and first_contact_time columns.
    bucket_minutes : int
        Size of time buckets in minutes (15, 30, or 60).
    
    Returns
    -------
    pandas.DataFrame
        Aggregated data ready for heatmap visualization.
    """
    # Step 1: Add response time calculations
    # After this, each lead has response_minutes and contact_within_1hr
    df = compute_response_metrics(df)
    
    # Step 2: Add the grouping dimensions
    # After this, each lead has day_of_week and time_bucket
    df = add_time_buckets(df, bucket_minutes)
    
    # Step 3: Group and aggregate
    # groupby() creates groups for each unique (day, time) combination
    # agg() calculates statistics for each group
    aggregated = df.groupby(['day_of_week', 'time_bucket']).agg(
        # Count total leads in this slot
        total_leads=('lead_id', 'count'),
        # Sum of True values = count of leads contacted within 1 hour
        # (True=1, False=0, so sum gives count of successes)
        contacted_within_1hr=('contact_within_1hr', 'sum')
    ).reset_index()
    
    # Step 4: Calculate success percentage
    # This is the value that will be displayed in each heatmap cell
    aggregated['pct_within_1hr'] = (
        aggregated['contacted_within_1hr'] / aggregated['total_leads'] * 100
    ).round(1)
    
    # Step 5: Set proper day ordering
    # Without this, pandas would sort alphabetically (Friday, Monday, Saturday...)
    # We want chronological order (Monday, Tuesday, Wednesday...)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    aggregated['day_of_week'] = pd.Categorical(
        aggregated['day_of_week'], 
        categories=day_order, 
        ordered=True
    )
    
    # Step 6: Sort by day then time for consistent output
    aggregated = aggregated.sort_values(['day_of_week', 'time_bucket'])
    
    return aggregated


# =============================================================================
# SECTION 5: SCHEDULE HELPER FUNCTIONS
# =============================================================================
#
# THE QUESTION: Who was supposed to be working when performance was poor?
#
# The heatmap might show that Monday 2:00 PM has terrible response times.
# The schedule overlay answers: "Was anyone even scheduled then?"
#
# These functions bridge lead data and schedule data, enabling staffing
# analysis alongside performance analysis.
# =============================================================================

def get_reps_for_slot(schedule_df, day_of_week, time_bucket):
    """
    Find which sales reps are scheduled during a specific time slot.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    To overlay staffing information on the performance heatmap, we need to
    know who was scheduled for any given (day, time) combination. This
    function answers: "Who should have been working?"
    
    THE LOGIC
    ---------
    A rep is considered "working" during a time slot if:
        start_time <= time_bucket < end_time
    
    Note the asymmetric comparison: we include the start time but exclude
    the end time. This prevents double-counting at shift boundaries.
    
    Parameters
    ----------
    schedule_df : pandas.DataFrame
        Schedule data with rep_name, day_of_week, start_time, end_time.
    day_of_week : str
        The day to check (e.g., "Monday").
    time_bucket : datetime.time
        The time slot to check (e.g., datetime.time(14, 0) for 2:00 PM).
    
    Returns
    -------
    list
        Names of reps scheduled during this slot. Empty list if none.
    """
    # Filter to the requested day
    day_schedule = schedule_df[schedule_df['day_of_week'] == day_of_week]
    
    # Check each rep's shift to see if it covers this time
    scheduled_reps = []
    for _, row in day_schedule.iterrows():
        # The time_bucket falls within the shift if it's >= start and < end
        if row['start_time'] <= time_bucket < row['end_time']:
            scheduled_reps.append(row['rep_name'])
    
    return scheduled_reps


def create_schedule_lookup(schedule_df, bucket_minutes):
    """
    Build a fast lookup table: (day, time) → list of scheduled reps.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    The heatmap might have 5 days × 30 time slots = 150 cells. Calling
    get_reps_for_slot() 150 times with DataFrame filtering is slow.
    
    Instead, we pre-compute all combinations once and store them in a
    dictionary. Dictionary lookup is O(1)—essentially instant.
    
    THE TRADE-OFF
    -------------
    We spend a little more time upfront (building the lookup) to save
    significant time later (instant lookups during rendering). This is
    a classic space-time trade-off: use memory to save computation.
    
    Parameters
    ----------
    schedule_df : pandas.DataFrame
        Schedule data with rep_name, day_of_week, start_time, end_time.
    bucket_minutes : int
        Size of time buckets in minutes.
    
    Returns
    -------
    dict
        Keys: (day_of_week, time_bucket) tuples
        Values: Lists of rep names scheduled during that slot
        
        Only slots with at least one rep are included (no empty lists stored).
    """
    lookup = {}
    
    # Generate all time buckets for the display range
    time_buckets = generate_time_buckets(bucket_minutes)
    
    # All days of the week (we check all, even if schedule only has weekdays)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Build the lookup by checking every possible (day, time) combination
    for day in days:
        for time_bucket in time_buckets:
            reps = get_reps_for_slot(schedule_df, day, time_bucket)
            if reps:  # Only store non-empty lists
                lookup[(day, time_bucket)] = reps
    
    return lookup


def generate_time_buckets(bucket_minutes, start_hour=5, end_hour=20):
    """
    Generate all time bucket values for the heatmap's time range.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    The heatmap needs a consistent set of time slots for its columns.
    This function generates those slots: 5:00 AM, 5:30 AM, 6:00 AM, ...
    up to (but not including) 8:00 PM.
    
    THE RANGE: 5 AM to 8 PM
    -----------------------
    This default range covers typical business hours with some buffer.
    Leads arriving at 5:00 AM are caught. The day ends at 8:00 PM.
    
    You can adjust start_hour and end_hour to match your business:
    - Call center open 24/7? Use start_hour=0, end_hour=24
    - Only tracking 9-5? Use start_hour=9, end_hour=17
    
    Parameters
    ----------
    bucket_minutes : int
        Size of each bucket in minutes (15, 30, or 60).
    start_hour : int, optional
        Hour to start (default 5 = 5:00 AM).
    end_hour : int, optional
        Hour to end (default 20 = 8:00 PM). This hour is NOT included;
        if end_hour=20, the last bucket starts at 7:30 PM (for 30-min buckets).
    
    Returns
    -------
    list
        List of datetime.time objects, one per bucket.
        
    Example
    -------
    With 60-minute buckets from 5 AM to 8 PM:
    >>> generate_time_buckets(60, start_hour=5, end_hour=20)
    [time(5,0), time(6,0), time(7,0), ..., time(19,0)]  # 15 buckets
    """
    buckets = []
    current_minutes = start_hour * 60  # Convert start hour to minutes
    end_minutes = end_hour * 60        # Convert end hour to minutes
    
    # Generate buckets until we reach the end
    while current_minutes < end_minutes:
        # Convert minutes back to hours and minutes
        hour = current_minutes // 60
        minute = current_minutes % 60
        
        # Create time object and add to list
        buckets.append(datetime(2000, 1, 1, hour, minute).time())
        
        # Move to next bucket
        current_minutes += bucket_minutes
    
    return buckets


# =============================================================================
# SECTION 6: SUMMARY STATISTICS
# =============================================================================
#
# THE COMPLEMENT TO THE HEATMAP
#
# The heatmap shows patterns across time. Summary statistics show the
# big picture: "Overall, how are we doing?"
#
# These numbers appear at the top of the dashboard, giving users immediate
# context before they dive into the detailed heatmap analysis.
# =============================================================================

def calculate_summary_stats(df):
    """
    Calculate overall summary statistics for the leads data.
    
    WHY THIS FUNCTION EXISTS
    ------------------------
    Before examining the heatmap's cell-by-cell detail, users need context:
    - How many leads are we analyzing?
    - What's our overall success rate?
    - What's the typical response time?
    
    These summary statistics provide that context, answering "How are we
    doing overall?" before the heatmap answers "When are we doing well/poorly?"
    
    THE METRICS
    -----------
    - total_leads: Size of the dataset being analyzed
    - contacted_count: Leads that received any response (regardless of timing)
    - contacted_within_1hr: Leads contacted within the target window
    - pct_within_1hr: Success rate as a percentage
    - avg_response_minutes: Mean response time (for contacted leads)
    - median_response_minutes: Median response time (less sensitive to outliers)
    
    WHY BOTH MEAN AND MEDIAN?
    -------------------------
    If one lead took 24 hours to contact, it dramatically skews the average.
    The median is robust to outliers—it's the "typical" response time.
    Showing both gives a complete picture: mean for aggregate impact,
    median for typical experience.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Leads data. Can be raw (this function computes metrics if needed)
        or already processed with compute_response_metrics().
    
    Returns
    -------
    dict
        Dictionary with keys:
        - total_leads (int)
        - contacted_count (int)
        - contacted_within_1hr (int)
        - pct_within_1hr (float, rounded to 1 decimal)
        - avg_response_minutes (float, rounded to 1 decimal)
        - median_response_minutes (float, rounded to 1 decimal)
    """
    # Ensure response metrics exist (compute them if not already present)
    if 'response_minutes' not in df.columns:
        df = compute_response_metrics(df)
    
    # Total leads in the dataset
    total_leads = len(df)
    
    # Leads that were contacted (have a non-null first_contact_time)
    contacted_count = df['first_contact_time'].notna().sum()
    
    # Leads contacted within the target window
    contacted_within_1hr = df['contact_within_1hr'].sum()
    
    # Success percentage (with protection against division by zero)
    pct_within_1hr = (contacted_within_1hr / total_leads * 100) if total_leads > 0 else 0
    
    # Response time statistics (only for leads that were actually contacted)
    contacted_df = df[df['response_minutes'].notna()]
    avg_response = contacted_df['response_minutes'].mean() if len(contacted_df) > 0 else 0
    median_response = contacted_df['response_minutes'].median() if len(contacted_df) > 0 else 0
    
    return {
        'total_leads': total_leads,
        'contacted_count': int(contacted_count),
        'contacted_within_1hr': int(contacted_within_1hr),
        'pct_within_1hr': round(pct_within_1hr, 1),
        'avg_response_minutes': round(avg_response, 1),
        'median_response_minutes': round(median_response, 1)
    }
