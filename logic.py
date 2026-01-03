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


# Standard column name mappings for auto-detect and manual override
STANDARD_COLUMNS = {
    'lead_id': ['lead_id', 'LeadID', 'lead', 'id'],
    'received_time': ['received_time', 'ReceivedTime', 'received', 'time_received', 'created_at'],
    'first_contact_time': ['first_contact_time', 'FirstContactTime', 'contacted_time', 'time_contacted'],
    'team': ['team', 'Team', 'sales_team', 'SalesTeam'],
    'manager': ['ManagerPreferredName', 'manager', 'Manager', 'manager_name', 'ManagerName'],
    'count_for_metrics': [
        'count_for_metrics', 'CountForMetrics', 'counts_flag', 'eligible_for_metrics',
        'SupervisorFlag', 'IncludeInMetrics', 'use_in_metrics'
    ],
    'contacted_flag': [
        'contacted_flag', 'ContactedFlag', 'contacted_1hr', 'Contacted1Hr',
        'ContactWithinHour', 'contacted_within_hour'
    ]
}


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-detect likely column names for required/optional fields."""
    detected: Dict[str, Optional[str]] = {}
    cols = set(df.columns)
    for std, candidates in STANDARD_COLUMNS.items():
        detected[std] = next((c for c in candidates if c in cols), None)
    return detected


def validate_required_columns(detected: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
    """Ensure mandatory columns are present after detection/override."""
    required = ['lead_id', 'received_time', 'first_contact_time']
    missing = [c for c in required if detected.get(c) is None]
    return len(missing) == 0, missing


def get_column_mapping_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Provide ordered options (detected first) for UI selectors."""
    detected = detect_columns(df)
    opts: Dict[str, List[str]] = {}
    for std in STANDARD_COLUMNS.keys():
        first = [detected[std]] if detected.get(std) else []
        remaining = [c for c in df.columns if c not in first]
        opts[std] = first + remaining
    return opts


def load_leads(
    file_or_path,
    team_name: Optional[str] = None,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Load leads from CSV file with auto-detect and optional manual column mapping.
    
    Args:
        file_or_path: Path to CSV file or file-like object
        team_name: Optional team name to assign if not in CSV
        column_mapping: Optional manual overrides {standard_name: actual_column_name}
        
    Returns:
        DataFrame with leads data (times converted to Pacific timezone)
    
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(file_or_path)
    detected = detect_columns(df)

    if column_mapping:
        detected.update(column_mapping)

    ok, missing = validate_required_columns(detected)
    if not ok:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Rename to standard names for downstream logic
    rename_map = {v: k for k, v in detected.items() if v is not None}
    df = df.rename(columns=rename_map)

    # Normalize manager column naming
    if 'manager' in df.columns and 'ManagerPreferredName' not in df.columns:
        df['ManagerPreferredName'] = df['manager']
    if 'ManagerPreferredName' in df.columns and 'manager' not in df.columns:
        df['manager'] = df['ManagerPreferredName']
    
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
    elif 'team' not in df.columns:
        df['team'] = 'Unknown'
    
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


def coerce_bool_series(series: pd.Series) -> pd.Series:
    """Convert heterogeneous truthy/falsy values to booleans, treating blanks as False."""

    def _coerce(value) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            return lowered in {'true', 't', 'yes', 'y', '1', 'contacted', 'within1h', 'within_1h', 'ok'}
        return bool(value)

    return series.apply(_coerce).fillna(False)


def compute_response_metrics(
    df: pd.DataFrame,
    use_contacted_flag: bool = False,
    contacted_flag_column: str = 'contacted_flag',
    use_count_flag: bool = False,
    count_flag_column: str = 'count_for_metrics'
) -> pd.DataFrame:
    """
    Calculate response time metrics for each lead.

    Adds columns:
    - response_minutes: Time in minutes between received and first contact
    - contact_within_1hr: Boolean indicating if contacted within 1 hour (optionally overridden by flag column)

    Args:
        df: DataFrame with received_time and first_contact_time columns
        use_contacted_flag: If True, use a boolean column to mark contacted-within-1hr instead of timestamps
        contacted_flag_column: Column name to use when use_contacted_flag is True
        use_count_flag: If True, filter to rows where the count/eligibility flag is truthy
        count_flag_column: Column name to use when use_count_flag is True
        
    Returns:
        DataFrame with additional metrics columns (and optionally filtered rows)
    """
    df = df.copy()

    has_count_flag = use_count_flag and count_flag_column in df.columns
    if has_count_flag:
        df[count_flag_column] = coerce_bool_series(df[count_flag_column])
        df = df[df[count_flag_column]]

    # Calculate response time in minutes (still used for placement/ordering)
    df['response_minutes'] = (
        (df['first_contact_time'] - df['received_time']).dt.total_seconds() / 60
    )

    has_contacted_flag = use_contacted_flag and contacted_flag_column in df.columns
    if has_contacted_flag:
        df['contact_within_1hr'] = coerce_bool_series(df[contacted_flag_column])
    else:
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


def find_slot_winners(aggregated_df: pd.DataFrame, grouping_key: str = 'team') -> pd.DataFrame:
    """
    Identify the winning team/manager for each time slot.
    
    Args:
        aggregated_df: Aggregated DataFrame from aggregate_by_team_slot or aggregate_by_manager_slot
        grouping_key: Column name for grouping ('team' or 'manager')
        
    Returns:
        DataFrame with columns:
        - day_of_week: Day name
        - time_bucket: Time bucket
        - winning_team: Team/manager with highest percentage
        - winning_pct: Highest percentage value
        - margin: Difference between winner and second place
        - all_teams: Dictionary of all team/manager percentages for this slot
    """
    winners = []
    
    for (day, time_bucket), group in aggregated_df.groupby(['day_of_week', 'time_bucket']):
        # Sort by percentage descending
        sorted_group = group.sort_values('pct_within_1hr', ascending=False)
        
        winner = sorted_group.iloc[0]
        winning_team = winner[grouping_key]
        winning_pct = winner['pct_within_1hr']
        
        # Calculate margin (difference to second place, or 0 if only one team)
        if len(sorted_group) > 1:
            margin = winning_pct - sorted_group.iloc[1]['pct_within_1hr']
        else:
            margin = winning_pct
        
        # Create dictionary of all team/manager percentages
        all_teams = dict(zip(group[grouping_key], group['pct_within_1hr']))
        
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
    team_b: str,
    grouping_key: str = 'team'
) -> pd.DataFrame:
    """
    Compute head-to-head difference between two teams/managers.
    
    Args:
        aggregated_df: Aggregated DataFrame from aggregate_by_team_slot or aggregate_by_manager_slot
        team_a: First team/manager name
        team_b: Second team/manager name
        grouping_key: Column name for grouping ('team' or 'manager')
        
    Returns:
        DataFrame with difference metrics (Team A % - Team B %)
    """
    # Determine which grouping column exists
    if grouping_key not in aggregated_df.columns:
        if 'team' in aggregated_df.columns:
            grouping_key = 'team'
        elif 'manager' in aggregated_df.columns:
            grouping_key = 'manager'
        else:
            raise ValueError("DataFrame must have either 'team' or 'manager' column")
    
    # Filter to only the two teams/managers
    team_df = aggregated_df[aggregated_df[grouping_key].isin([team_a, team_b])].copy()
    
    # Pivot to have teams as columns
    pivot_df = team_df.pivot_table(
        index=['day_of_week', 'time_bucket'],
        columns=grouping_key,
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
    """Return sorted unique manager names using either manager or ManagerPreferredName."""
    col = 'manager' if 'manager' in df.columns else 'ManagerPreferredName'
    if col not in df.columns:
        return []
    managers = df[col].dropna().unique().tolist()
    return sorted(managers)


def filter_by_manager(df: pd.DataFrame, manager_name: str) -> pd.DataFrame:
    """Filter to rows for the given manager, handling either column name."""
    col = 'manager' if 'manager' in df.columns else 'ManagerPreferredName'
    if manager_name is None or col not in df.columns:
        return df
    return df[df[col] == manager_name].copy()


def aggregate_by_manager_slot(df: pd.DataFrame, bucket_minutes: int = 30) -> pd.DataFrame:
    """Aggregate by manager (supports manager or ManagerPreferredName)."""
    df = df.copy()

    if 'response_minutes' not in df.columns or 'contact_within_1hr' not in df.columns:
        df = compute_response_metrics(df)

    # Ensure we have a manager column; if missing, create a placeholder
    col = 'manager' if 'manager' in df.columns else 'ManagerPreferredName'
    if col not in df.columns:
        df['manager'] = 'Unknown'
        col = 'manager'

    df['day_of_week'] = df['received_time'].dt.day_name()
    df['time_bucket'] = df['received_time'].apply(lambda x: bucket_time(x, bucket_minutes))

    aggregated = df.groupby([col, 'day_of_week', 'time_bucket']).agg(
        total_leads=('lead_id', 'count'),
        contacted_within_1hr=('contact_within_1hr', 'sum')
    ).reset_index()

    aggregated = aggregated.rename(columns={col: 'manager'})
    aggregated['pct_within_1hr'] = (
        aggregated['contacted_within_1hr'] / aggregated['total_leads'] * 100
    ).fillna(0)

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


def compute_vs_average(aggregated_df: pd.DataFrame, target: str, grouping_key: str = 'team') -> pd.DataFrame:
    """
    Compare a single team/manager against the average of all others per slot.
    
    Args:
        aggregated_df: Aggregated DataFrame (from aggregate_by_team_slot or aggregate_by_manager_slot)
        target: Name of the target team/manager
        grouping_key: Column name for grouping ('team' or 'manager')
        
    Returns:
        DataFrame with columns:
        - day_of_week, time_bucket
        - target_pct: Target's percentage
        - avg_others: Average of all other teams/managers
        - difference: target_pct - avg_others (positive = beating others)
        - points_needed: Percentage points needed to tie the benchmark (0 if already ahead)
        - total_leads: Target's total leads in this slot
        - contacted_1hr: Target's contacted within 1hr count
        - leads_short: Additional leads needed to tie (0 if already ahead)
        - impact_score: points_needed * total_leads (for prioritization)
    """
    import math
    results = []
    
    for (day, time_bucket), group in aggregated_df.groupby(['day_of_week', 'time_bucket']):
        target_data = group[group[grouping_key] == target]
        others_data = group[group[grouping_key] != target]
        
        target_pct = target_data['pct_within_1hr'].iloc[0] if len(target_data) > 0 else 0
        avg_others = others_data['pct_within_1hr'].mean() if len(others_data) > 0 else 0
        total_leads = int(target_data['total_leads'].iloc[0]) if len(target_data) > 0 else 0
        contacted_1hr = int(target_data['contacted_within_1hr'].iloc[0]) if len(target_data) > 0 else 0
        
        diff = target_pct - avg_others
        points_needed = max(0, -diff)
        
        # Calculate leads short: how many more contacts needed to reach avg_others percentage
        if points_needed > 0 and total_leads > 0:
            target_contacts_needed = math.ceil((avg_others / 100) * total_leads)
            leads_short = max(0, target_contacts_needed - contacted_1hr)
        else:
            leads_short = 0
        
        impact_score = points_needed * total_leads
        
        results.append({
            'day_of_week': day,
            'time_bucket': time_bucket,
            'target_pct': target_pct,
            'avg_others': avg_others,
            'difference': diff,
            'points_needed': points_needed,
            'total_leads': total_leads,
            'contacted_1hr': contacted_1hr,
            'leads_short': leads_short,
            'impact_score': impact_score
        })
    
    return pd.DataFrame(results)


def compute_improvement_needed(comparison_df: pd.DataFrame, top_n: int = 5, sort_by: str = 'impact_score') -> pd.DataFrame:
    """
    Return top slots where target needs the most improvement to tie the benchmark.
    
    Args:
        comparison_df: DataFrame from compute_vs_average
        top_n: Number of top slots to return
        sort_by: Column to sort by ('impact_score', 'points_needed', or 'leads_short')
    
    Returns:
        DataFrame with top improvement opportunities
    """
    df = comparison_df.copy()
    if 'points_needed' not in df.columns:
        df['points_needed'] = df.apply(
            lambda row: max(0, (row.get('avg_others', 0) - row.get('target_pct', 0))), axis=1
        )

    improvement = df[df['points_needed'] > 0]
    
    # Sort by specified column (default to impact_score for prioritization)
    if sort_by in improvement.columns:
        return improvement.sort_values(sort_by, ascending=False).head(top_n)
    return improvement.sort_values('points_needed', ascending=False).head(top_n)


def compute_weekly_leads_short(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate leads_short/ahead by day of week for weekly summary.
    Shows where supervisor is behind and ahead of average.
    
    Args:
        comparison_df: DataFrame from compute_vs_average
        
    Returns:
        DataFrame with day_of_week, leads_short, total_leads, slots_behind, slots_ahead
    """
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    behind_slots = comparison_df[comparison_df['points_needed'] > 0]
    ahead_slots = comparison_df[comparison_df['points_needed'] == 0]
    
    if behind_slots.empty and ahead_slots.empty:
        return pd.DataFrame(columns=['day_of_week', 'leads_short', 'total_leads', 'slots_behind', 'slots_ahead'])
    
    # Summarize behind slots
    summary_behind = behind_slots.groupby('day_of_week').agg({
        'leads_short': 'sum',
        'total_leads': 'sum',
        'points_needed': 'count'
    }).rename(columns={'points_needed': 'slots_behind'}).reset_index()
    
    # Summarize ahead slots
    if not ahead_slots.empty:
        summary_ahead = ahead_slots.groupby('day_of_week').size().reset_index(name='slots_ahead')
        summary = summary_behind.merge(summary_ahead, on='day_of_week', how='outer').fillna(0)
    else:
        summary = summary_behind.copy()
        summary['slots_ahead'] = 0
    
    if summary.empty:
        return pd.DataFrame(columns=['day_of_week', 'leads_short', 'total_leads', 'slots_behind', 'slots_ahead'])
    
    summary['day_order'] = summary['day_of_week'].apply(lambda x: day_order.index(x) if x in day_order else 7)
    summary = summary.sort_values('day_order').drop(columns=['day_order'])
    summary['leads_short'] = summary['leads_short'].fillna(0).astype(int)
    summary['slots_behind'] = summary['slots_behind'].fillna(0).astype(int)
    summary['slots_ahead'] = summary['slots_ahead'].fillna(0).astype(int)
    
    return summary


def compute_leads_to_beat_leader(rankings_df: pd.DataFrame, target_entity: str, grouping_key: str = 'team') -> dict:
    """
    Calculate how many additional leads need to be contacted within 1hr
    for the target entity to become the #1 ranked entity.
    
    Returns dict with:
    - leads_needed: number of additional contacts needed (0 if already #1)
    - current_rank: current rank
    - current_pct: current percentage
    - leader: name of current leader
    - leader_pct: leader's percentage
    """
    target_row = rankings_df[rankings_df[grouping_key] == target_entity].iloc[0]
    leader_row = rankings_df[rankings_df['rank'] == 1].iloc[0]
    
    current_rank = int(target_row['rank'])
    current_pct = target_row['overall_pct']
    total_leads = target_row['total_leads']
    contacted = target_row['total_contacted_1hr']
    
    leader_name = leader_row[grouping_key]
    leader_pct = leader_row['overall_pct']
    
    if current_rank == 1:
        return {
            'leads_needed': 0,
            'current_rank': 1,
            'current_pct': current_pct,
            'leader': target_entity,
            'leader_pct': current_pct,
            'is_leader': True
        }
    
    # To beat leader, need: (contacted + X) / total_leads > leader_pct / 100
    # X > (leader_pct / 100) * total_leads - contacted
    # Add small buffer (0.1%) to ensure we pass, not just tie
    target_pct = leader_pct + 0.1
    leads_needed = max(0, int((target_pct / 100) * total_leads - contacted) + 1)
    
    return {
        'leads_needed': leads_needed,
        'current_rank': current_rank,
        'current_pct': current_pct,
        'leader': leader_name,
        'leader_pct': leader_pct,
        'is_leader': False
    }


def compute_leads_to_beat_average(rankings_df: pd.DataFrame, target_entity: str, grouping_key: str = 'team') -> dict:
    """
    Calculate how many additional leads need to be contacted within 1hr
    for the target entity to beat the average of all OTHER entities.
    
    Returns dict with:
    - leads_needed: number of additional contacts needed (0 if already above)
    - current_pct: current percentage
    - avg_others_pct: average percentage of other entities
    - is_above_average: whether already beating average
    """
    target_row = rankings_df[rankings_df[grouping_key] == target_entity].iloc[0]
    others = rankings_df[rankings_df[grouping_key] != target_entity]
    
    current_pct = target_row['overall_pct']
    total_leads = target_row['total_leads']
    contacted = target_row['total_contacted_1hr']
    
    # Calculate average of others (weighted by their total leads)
    if len(others) > 0:
        avg_others_pct = (others['total_contacted_1hr'].sum() / others['total_leads'].sum()) * 100
    else:
        avg_others_pct = current_pct
    
    if current_pct >= avg_others_pct:
        return {
            'leads_needed': 0,
            'current_pct': current_pct,
            'avg_others_pct': avg_others_pct,
            'is_above_average': True
        }
    
    # To beat average: (contacted + X) / total_leads > avg_others_pct / 100
    target_pct = avg_others_pct + 0.1
    leads_needed = max(0, int((target_pct / 100) * total_leads - contacted) + 1)
    
    return {
        'leads_needed': leads_needed,
        'current_pct': current_pct,
        'avg_others_pct': avg_others_pct,
        'is_above_average': False
    }
