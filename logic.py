"""
Business Logic Layer - Data Processing

All data transformations and calculations for lead response analysis.
No visualization code - pure data processing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional, List, Dict, Tuple
import pytz


def load_leads(file_or_path, team_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load leads from CSV file.
    
    Args:
        file_or_path: Path to CSV file or file-like object
        team_name: Optional team name to assign if not in CSV
        
    Returns:
        DataFrame with leads data (times converted to Pacific timezone)
    """
    df = pd.read_csv(file_or_path)
    
    # Convert timestamps - assume UTC if no timezone info
    df['received_time'] = pd.to_datetime(df['received_time'])
    df['first_contact_time'] = pd.to_datetime(df['first_contact_time'], errors='coerce')
    
    # If timestamps are naive (no timezone), assume UTC
    if df['received_time'].dt.tz is None:
        df['received_time'] = df['received_time'].dt.tz_localize('UTC')
    
    if df['first_contact_time'].dt.tz is None:
        df['first_contact_time'] = df['first_contact_time'].dt.tz_localize('UTC')
    
    # Convert to Pacific time
    pacific = pytz.timezone('US/Pacific')
    df['received_time'] = df['received_time'].dt.tz_convert(pacific)
    df['first_contact_time'] = df['first_contact_time'].dt.tz_convert(pacific)
    
    # Assign team name if provided and not in data
    if team_name and 'team' not in df.columns:
        df['team'] = team_name
    elif team_name and 'team' in df.columns:
        # Override existing team column
        df['team'] = team_name
    
    return df


def combine_team_data(dfs: List[Tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    """
    Combine multiple team DataFrames into one.
    
    Args:
        dfs: List of (DataFrame, team_name) tuples
        
    Returns:
        Combined DataFrame with team column
    """
    combined = []
    for df, team_name in dfs:
        df_copy = df.copy()
        if 'team' not in df_copy.columns:
            df_copy['team'] = team_name
        combined.append(df_copy)
    
    return pd.concat(combined, ignore_index=True)


def compute_response_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate response time metrics for each lead.
    
    Adds columns:
    - response_minutes: Time in minutes between received and first contact
    - contact_within_1hr: Boolean indicating if contacted within 1 hour
    
    Args:
        df: DataFrame with received_time and first_contact_time columns
        
    Returns:
        DataFrame with additional metrics columns
    """
    df = df.copy()
    
    # Calculate response time in minutes
    df['response_minutes'] = (
        (df['first_contact_time'] - df['received_time']).dt.total_seconds() / 60
    )
    
    # Mark leads contacted within 1 hour
    df['contact_within_1hr'] = df['response_minutes'] <= 60
    
    # For leads never contacted, set to False
    df['contact_within_1hr'] = df['contact_within_1hr'].fillna(False)
    
    return df


def bucket_time(dt: pd.Timestamp, bucket_minutes: int = 30) -> time:
    """
    Floor a datetime to the nearest time bucket.
    
    Args:
        dt: Datetime to bucket
        bucket_minutes: Size of time bucket in minutes (default 30)
        
    Returns:
        Time object representing the bucket
    """
    total_minutes = dt.hour * 60 + dt.minute
    floored_minutes = (total_minutes // bucket_minutes) * bucket_minutes
    new_hour = floored_minutes // 60
    new_minute = floored_minutes % 60
    return time(new_hour, new_minute)


def aggregate_by_team_slot(df: pd.DataFrame, bucket_minutes: int = 30) -> pd.DataFrame:
    """
    Aggregate leads by team, day of week, and time slot.
    
    Args:
        df: DataFrame with leads data (must have team column)
        bucket_minutes: Size of time bucket in minutes
        
    Returns:
        Aggregated DataFrame with columns:
        - team: Team name
        - day_of_week: Day name (Monday, Tuesday, etc.)
        - time_bucket: Time bucket
        - total_leads: Count of leads in this slot
        - contacted_within_1hr: Count contacted within 1 hour
        - pct_within_1hr: Percentage contacted within 1 hour
    """
    df = df.copy()
    
    # Ensure we have the required columns
    if 'response_minutes' not in df.columns or 'contact_within_1hr' not in df.columns:
        df = compute_response_metrics(df)
    
    # Extract day of week
    df['day_of_week'] = df['received_time'].dt.day_name()
    
    # Create time buckets
    df['time_bucket'] = df['received_time'].apply(
        lambda x: bucket_time(x, bucket_minutes)
    )
    
    # Aggregate by team, day, and time slot
    aggregated = df.groupby(['team', 'day_of_week', 'time_bucket']).agg(
        total_leads=('lead_id', 'count'),
        contacted_within_1hr=('contact_within_1hr', 'sum')
    ).reset_index()
    
    # Calculate percentage
    aggregated['pct_within_1hr'] = (
        aggregated['contacted_within_1hr'] / aggregated['total_leads'] * 100
    )
    
    # Fill NaN with 0 (for slots with no contacts)
    aggregated['pct_within_1hr'] = aggregated['pct_within_1hr'].fillna(0)
    
    return aggregated


def calculate_team_rankings(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall statistics and rankings for each team.
    
    Args:
        aggregated_df: Aggregated DataFrame from aggregate_by_team_slot
        
    Returns:
        DataFrame with team rankings and statistics
    """
    team_stats = aggregated_df.groupby('team').agg(
        total_leads=('total_leads', 'sum'),
        total_contacted_1hr=('contacted_within_1hr', 'sum'),
        avg_pct=('pct_within_1hr', 'mean'),
        min_pct=('pct_within_1hr', 'min'),
        max_pct=('pct_within_1hr', 'max')
    ).reset_index()
    
    # Calculate overall percentage
    team_stats['overall_pct'] = (
        team_stats['total_contacted_1hr'] / team_stats['total_leads'] * 100
    )
    
    # Rank teams by overall percentage
    team_stats['rank'] = team_stats['overall_pct'].rank(ascending=False, method='min').astype(int)
    team_stats = team_stats.sort_values('rank')
    
    return team_stats


def find_slot_winners(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the winning team for each time slot.
    
    Args:
        aggregated_df: Aggregated DataFrame from aggregate_by_team_slot
        
    Returns:
        DataFrame with columns:
        - day_of_week: Day name
        - time_bucket: Time bucket
        - winning_team: Team with highest percentage
        - winning_pct: Highest percentage value
        - margin: Difference between winner and second place
        - all_teams: Dictionary of all team percentages for this slot
    """
    winners = []
    
    for (day, time_bucket), group in aggregated_df.groupby(['day_of_week', 'time_bucket']):
        # Sort by percentage descending
        sorted_group = group.sort_values('pct_within_1hr', ascending=False)
        
        winner = sorted_group.iloc[0]
        winning_team = winner['team']
        winning_pct = winner['pct_within_1hr']
        
        # Calculate margin (difference to second place, or 0 if only one team)
        if len(sorted_group) > 1:
            margin = winning_pct - sorted_group.iloc[1]['pct_within_1hr']
        else:
            margin = winning_pct
        
        # Create dictionary of all team percentages
        all_teams = dict(zip(group['team'], group['pct_within_1hr']))
        
        winners.append({
            'day_of_week': day,
            'time_bucket': time_bucket,
            'winning_team': winning_team,
            'winning_pct': winning_pct,
            'margin': margin,
            'all_teams': all_teams
        })
    
    return pd.DataFrame(winners)


def compute_team_difference(
    aggregated_df: pd.DataFrame, 
    team_a: str, 
    team_b: str
) -> pd.DataFrame:
    """
    Compute head-to-head difference between two teams.
    
    Args:
        aggregated_df: Aggregated DataFrame from aggregate_by_team_slot
        team_a: First team name
        team_b: Second team name
        
    Returns:
        DataFrame with difference metrics (Team A % - Team B %)
    """
    # Filter to only the two teams
    team_df = aggregated_df[aggregated_df['team'].isin([team_a, team_b])].copy()
    
    # Pivot to have teams as columns
    pivot_df = team_df.pivot_table(
        index=['day_of_week', 'time_bucket'],
        columns='team',
        values='pct_within_1hr',
        fill_value=0
    ).reset_index()
    
    # Calculate difference (Team A - Team B)
    if team_a in pivot_df.columns and team_b in pivot_df.columns:
        pivot_df['difference'] = pivot_df[team_a] - pivot_df[team_b]
        pivot_df['team_a_pct'] = pivot_df[team_a]
        pivot_df['team_b_pct'] = pivot_df[team_b]
    else:
        # One or both teams missing data
        pivot_df['difference'] = 0
        pivot_df['team_a_pct'] = pivot_df.get(team_a, 0)
        pivot_df['team_b_pct'] = pivot_df.get(team_b, 0)
    
    return pivot_df[['day_of_week', 'time_bucket', 'difference', 'team_a_pct', 'team_b_pct']]


def get_unique_managers(df: pd.DataFrame) -> List[str]:
    """
    Get list of unique managers from the dataset.
    
    Args:
        df: DataFrame with leads data (must have ManagerPreferredName column)
        
    Returns:
        Sorted list of unique manager names
    """
    if 'ManagerPreferredName' not in df.columns:
        return []
    
    managers = df['ManagerPreferredName'].dropna().unique().tolist()
    return sorted(managers)


def filter_by_manager(df: pd.DataFrame, manager_name: str) -> pd.DataFrame:
    """
    Filter leads by manager.
    
    Args:
        df: DataFrame with leads data
        manager_name: Manager name to filter by
        
    Returns:
        Filtered DataFrame with only leads from the specified manager
    """
    if manager_name is None or 'ManagerPreferredName' not in df.columns:
        return df
    
    return df[df['ManagerPreferredName'] == manager_name].copy()


def aggregate_by_manager_slot(df: pd.DataFrame, bucket_minutes: int = 30) -> pd.DataFrame:
    """
    Aggregate leads by manager, day of week, and time slot.
    
    Args:
        df: DataFrame with leads data (must have ManagerPreferredName column)
        bucket_minutes: Size of time bucket in minutes
        
    Returns:
        Aggregated DataFrame with columns:
        - manager: Manager name
        - day_of_week: Day name (Monday, Tuesday, etc.)
        - time_bucket: Time bucket
        - total_leads: Count of leads in this slot
        - contacted_within_1hr: Count contacted within 1 hour
        - pct_within_1hr: Percentage contacted within 1 hour
    """
    df = df.copy()
    
    # Ensure we have the required columns
    if 'response_minutes' not in df.columns or 'contact_within_1hr' not in df.columns:
        df = compute_response_metrics(df)
    
    if 'ManagerPreferredName' not in df.columns:
        raise ValueError("DataFrame must have ManagerPreferredName column")
    
    # Extract day of week
    df['day_of_week'] = df['received_time'].dt.day_name()
    
    # Create time buckets
    df['time_bucket'] = df['received_time'].apply(
        lambda x: bucket_time(x, bucket_minutes)
    )
    
    # Aggregate by manager, day, and time slot
    aggregated = df.groupby(['ManagerPreferredName', 'day_of_week', 'time_bucket']).agg(
        total_leads=('lead_id', 'count'),
        contacted_within_1hr=('contact_within_1hr', 'sum')
    ).reset_index()
    
    # Rename column for consistency
    aggregated = aggregated.rename(columns={'ManagerPreferredName': 'manager'})
    
    # Calculate percentage
    aggregated['pct_within_1hr'] = (
        aggregated['contacted_within_1hr'] / aggregated['total_leads'] * 100
    )
    
    # Fill NaN with 0 (for slots with no contacts)
    aggregated['pct_within_1hr'] = aggregated['pct_within_1hr'].fillna(0)
    
    return aggregated


def calculate_manager_rankings(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall statistics and rankings for each manager.
    
    Args:
        aggregated_df: Aggregated DataFrame from aggregate_by_manager_slot
        
    Returns:
        DataFrame with manager rankings and statistics
    """
    manager_stats = aggregated_df.groupby('manager').agg(
        total_leads=('total_leads', 'sum'),
        total_contacted_1hr=('contacted_within_1hr', 'sum'),
        avg_pct=('pct_within_1hr', 'mean'),
        min_pct=('pct_within_1hr', 'min'),
        max_pct=('pct_within_1hr', 'max')
    ).reset_index()
    
    # Calculate overall percentage
    manager_stats['overall_pct'] = (
        manager_stats['total_contacted_1hr'] / manager_stats['total_leads'] * 100
    )
    
    # Rank managers by overall percentage
    manager_stats['rank'] = manager_stats['overall_pct'].rank(ascending=False, method='min').astype(int)
    manager_stats = manager_stats.sort_values('rank')
    
    return manager_stats
