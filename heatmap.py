"""
Visualization Layer - Plotly Charts

Converts processed data into visual heatmaps.
No data processing - pure visualization.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import numpy as np
from datetime import time


def generate_all_time_buckets(bucket_minutes: int = 30) -> List[time]:
    """
    Generate all possible time buckets for a 24-hour period.
    
    Args:
        bucket_minutes: Size of time bucket in minutes
        
    Returns:
        List of time objects for all buckets in a day
    """
    buckets = []
    total_minutes = 24 * 60
    for minutes in range(0, total_minutes, bucket_minutes):
        hour = minutes // 60
        minute = minutes % 60
        buckets.append(time(hour, minute))
    return buckets


def create_team_heatmap(
    team_df: pd.DataFrame, 
    team_name: str,
    bucket_minutes: int = 30,
    zmin: float = 0,
    zmax: float = 100
) -> go.Figure:
    """
    Create a performance heatmap for a single team.
    
    Args:
        team_df: Aggregated DataFrame for one team (from aggregate_by_team_slot)
        team_name: Name of the team
        bucket_minutes: Size of time bucket in minutes (to generate complete grid)
        zmin: Minimum value for color scale
        zmax: Maximum value for color scale
        
    Returns:
        Plotly Figure with heatmap
    """
    # Pivot to create matrix (days x time buckets)
    pivot_pct = team_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='pct_within_1hr',
        fill_value=0
    )
    
    pivot_contacted = team_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='contacted_within_1hr',
        fill_value=0,
        aggfunc='sum'
    )
    
    pivot_total = team_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='total_leads',
        fill_value=0,
        aggfunc='sum'
    )
    
    # Always show all 7 days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_pct = pivot_pct.reindex(day_order, fill_value=0)
    pivot_contacted = pivot_contacted.reindex(day_order, fill_value=0)
    pivot_total = pivot_total.reindex(day_order, fill_value=0)
    
    # Generate all possible time buckets for complete grid
    all_time_buckets = generate_all_time_buckets(bucket_minutes)
    pivot_pct = pivot_pct.reindex(columns=all_time_buckets, fill_value=0)
    pivot_contacted = pivot_contacted.reindex(columns=all_time_buckets, fill_value=0)
    pivot_total = pivot_total.reindex(columns=all_time_buckets, fill_value=0)
    
    # Format time labels
    time_labels = [t.strftime('%H:%M') for t in all_time_buckets]
    
    # Create text with counts and percentage
    text_array = []
    for i in range(len(day_order)):
        row = []
        for j in range(len(all_time_buckets)):
            contacted = int(pivot_contacted.iloc[i, j])
            total = int(pivot_total.iloc[i, j])
            pct = pivot_pct.iloc[i, j]
            text = f"{contacted}/{total}<br>{pct:.0f}%"
            row.append(text)
        text_array.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_pct.values,
        x=time_labels,
        y=day_order,
        colorscale=[
            [0.0, '#d73027'],    # Red for 0%
            [0.5, '#fee08b'],    # Yellow for 50%
            [1.0, '#1a9850']     # Green for 100%
        ],
        zmin=zmin,
        zmax=zmax,
        text=text_array,
        texttemplate='%{text}',
        textfont={"size": 10, "color": "black"},
        hovertemplate='<b>%{y} at %{x}</b><br>Contacted: %{customdata[0]}<br>Total Leads: %{customdata[1]}<br>Percentage: %{z:.1f}%<extra></extra>',
        customdata=np.stack([pivot_contacted.values.astype(int), pivot_total.values.astype(int)], axis=-1),
        colorbar=dict(title="% Within 1hr", thickness=15)
    ))
    
    fig.update_layout(
        title=dict(text=f'{team_name} - Response Performance Heatmap', font=dict(size=18)),
        xaxis_title='Time of Day',
        yaxis_title='Day of Week',
        height=500,
        margin=dict(l=100, r=60, t=80, b=80)
    )
    
    # Make axis labels readable
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=12))
    
    return fig


def create_winner_heatmap(
    winner_df: pd.DataFrame,
    team_colors: Dict[str, str],
    bucket_minutes: int = 30
) -> go.Figure:
    """
    Create a heatmap showing which team wins each time slot.
    
    Args:
        winner_df: DataFrame from find_slot_winners
        team_colors: Dictionary mapping team names to colors
        bucket_minutes: Size of time bucket in minutes (to generate complete grid)
        
    Returns:
        Plotly Figure with winner heatmap
    """
    # Create pivot table for winning teams
    pivot_teams = winner_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='winning_team',
        aggfunc='first'
    )
    
    # Create pivot table for winning percentages (for intensity)
    pivot_pct = winner_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='winning_pct',
        aggfunc='first'
    )
    
    # Always show all 7 days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_teams = pivot_teams.reindex(day_order)
    pivot_pct = pivot_pct.reindex(day_order)
    
    # Generate all possible time buckets for complete grid
    all_time_buckets = generate_all_time_buckets(bucket_minutes)
    pivot_teams = pivot_teams.reindex(columns=all_time_buckets)
    pivot_pct = pivot_pct.reindex(columns=all_time_buckets)
    
    # Format time labels
    time_labels = [t.strftime('%H:%M') for t in all_time_buckets]
    
    # Create numeric matrix for colors (map teams to numbers)
    # Filter out NaN values
    teams = sorted(set([t for t in pivot_teams.values.flatten() if pd.notna(t)]))
    if not teams:
        # Fallback if no teams found
        teams = ['Unknown']
    team_to_num = {team: i for i, team in enumerate(teams)}
    # Map teams to numbers, handling NaN values
    numeric_matrix = pivot_teams.apply(lambda col: col.map(lambda x: team_to_num.get(x, -1) if pd.notna(x) else -1))
    
    # Create custom colorscale based on team colors
    colorscale = []
    n_teams = len(teams)
    for i, team in enumerate(teams):
        pos = i / max(n_teams - 1, 1) if n_teams > 1 else 0
        colorscale.append([pos, team_colors.get(team, '#808080')])
    
    # Create hover text with all team info
    hover_text = []
    for day in day_order:
        row_text = []
        for time_bucket in all_time_buckets:
            slot_data = winner_df[
                (winner_df['day_of_week'] == day) & 
                (winner_df['time_bucket'] == time_bucket)
            ]
            if not slot_data.empty:
                all_teams = slot_data.iloc[0]['all_teams']
                team_str = '<br>'.join([f"{t}: {p:.1f}%" for t, p in all_teams.items()])
                row_text.append(f"<b>{day} {time_bucket.strftime('%H:%M')}</b><br>{team_str}")
            else:
                row_text.append("")
        hover_text.append(row_text)
    
    fig = go.Figure(data=go.Heatmap(
        z=numeric_matrix.values,
        x=time_labels,
        y=day_order,
        colorscale=colorscale,
        text=hover_text,
        texttemplate='',
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title="Winning Team",
            tickvals=list(range(len(teams))),
            ticktext=teams
        ),
        zmin=0,
        zmax=len(teams) - 1
    ))
    
    fig.update_layout(
        title=dict(text='Winner Heatmap - Which Team Leads Each Time Slot', font=dict(size=18)),
        xaxis_title='Time of Day',
        yaxis_title='Day of Week',
        height=550,
        margin=dict(l=100, r=60, t=80, b=80)
    )
    
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=12))
    
    return fig


def create_difference_heatmap(
    diff_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    bucket_minutes: int = 30
) -> go.Figure:
    """
    Create a heatmap showing the difference between two teams.
    
    Args:
        diff_df: DataFrame from compute_team_difference
        team_a: First team name
        team_b: Second team name
        bucket_minutes: Size of time bucket in minutes (to generate complete grid)
        
    Returns:
        Plotly Figure with difference heatmap
    """
    # Pivot to create matrix
    pivot_df = diff_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='difference',
        fill_value=0
    )
    
    # Always show all 7 days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_df = pivot_df.reindex(day_order, fill_value=0)
    
    # Generate all possible time buckets for complete grid
    all_time_buckets = generate_all_time_buckets(bucket_minutes)
    pivot_df = pivot_df.reindex(columns=all_time_buckets, fill_value=0)
    
    # Format time labels
    time_labels = [t.strftime('%H:%M') for t in all_time_buckets]
    
    # Create hover text with both team percentages
    hover_text = []
    for day in day_order:
        row_text = []
        for time_bucket in all_time_buckets:
            slot_data = diff_df[
                (diff_df['day_of_week'] == day) & 
                (diff_df['time_bucket'] == time_bucket)
            ]
            if not slot_data.empty:
                row = slot_data.iloc[0]
                diff = row['difference']
                text = f"<b>{day} {time_bucket.strftime('%H:%M')}</b><br>"
                text += f"{team_a}: {row['team_a_pct']:.1f}%<br>"
                text += f"{team_b}: {row['team_b_pct']:.1f}%<br>"
                text += f"Difference: {diff:+.1f}%"
                row_text.append(text)
            else:
                row_text.append("")
        hover_text.append(row_text)
    
    # Find max absolute difference for symmetric color scale
    max_abs_diff = max(abs(pivot_df.values.min()), abs(pivot_df.values.max()))
    if max_abs_diff == 0:
        max_abs_diff = 100
    
    # Create diverging color scale (blue-white-red)
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=time_labels,
        y=day_order,
        colorscale=[
            [0.0, '#2166ac'],    # Blue (Team A better)
            [0.5, '#ffffff'],    # White (tied)
            [1.0, '#b2182b']     # Red (Team B better)
        ],
        zmin=-max_abs_diff,
        zmax=max_abs_diff,
        text=hover_text,
        texttemplate='',
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title=f"Difference ({team_a} - {team_b})")
    ))
    
    fig.update_layout(
        title=dict(text=f'Head-to-Head: {team_a} vs {team_b}', font=dict(size=18)),
        xaxis_title='Time of Day',
        yaxis_title='Day of Week',
        height=550,
        margin=dict(l=100, r=60, t=80, b=80)
    )
    
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=12))
    
    return fig


def create_comparison_heatmap(
    comparison_df: pd.DataFrame,
    target_name: str,
    bucket_minutes: int = 30
) -> go.Figure:
    """
    Create a heatmap showing where target beats average (green) vs needs improvement (red).
    Simplified: no text overlay, details shown on hover only.
    """
    # Pivot to create matrix
    pivot_diff = comparison_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='difference',
        fill_value=0
    )
    
    pivot_target_pct = comparison_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='target_pct',
        fill_value=0
    )
    
    pivot_avg = comparison_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='avg_others',
        fill_value=0
    )
    
    # Always show all 7 days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_diff = pivot_diff.reindex(day_order, fill_value=0)
    pivot_target_pct = pivot_target_pct.reindex(day_order, fill_value=0)
    pivot_avg = pivot_avg.reindex(day_order, fill_value=0)
    
    # Generate all possible time buckets for complete grid
    all_time_buckets = generate_all_time_buckets(bucket_minutes)
    pivot_diff = pivot_diff.reindex(columns=all_time_buckets, fill_value=0)
    pivot_target_pct = pivot_target_pct.reindex(columns=all_time_buckets, fill_value=0)
    pivot_avg = pivot_avg.reindex(columns=all_time_buckets, fill_value=0)
    
    # Format time labels
    time_labels = [t.strftime('%H:%M') for t in all_time_buckets]
    
    # Create hover text (details on hover only, no cell text)
    hover_text = []
    for i, day in enumerate(day_order):
        row = []
        for j, tb in enumerate(all_time_buckets):
            target_pct = pivot_target_pct.iloc[i, j]
            avg_pct = pivot_avg.iloc[i, j]
            diff = pivot_diff.iloc[i, j]
            text = f"<b>{day} {tb.strftime('%H:%M')}</b><br>"
            text += f"{target_name}: {target_pct:.1f}%<br>"
            text += f"Others: {avg_pct:.1f}%<br>"
            text += f"Diff: {diff:+.1f}%"
            row.append(text)
        hover_text.append(row)
    
    # Find max absolute difference for symmetric color scale
    max_abs_diff = max(abs(pivot_diff.values.min()), abs(pivot_diff.values.max()), 1)
    
    # Green for beating others, Red for behind (no text overlay)
    fig = go.Figure(data=go.Heatmap(
        z=pivot_diff.values,
        x=time_labels,
        y=day_order,
        colorscale=[
            [0.0, '#d73027'],    # Red (behind)
            [0.5, '#ffffbf'],    # Yellow (even)
            [1.0, '#1a9850']     # Green (ahead)
        ],
        zmin=-max_abs_diff,
        zmax=max_abs_diff,
        text=hover_text,
        texttemplate='',  # No text on cells
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title="vs Others", thickness=15)
    ))
    
    fig.update_layout(
        title=dict(text=f'{target_name} vs Average of Others', font=dict(size=18)),
        xaxis_title='Time of Day',
        yaxis_title='Day of Week',
        height=550,
        margin=dict(l=100, r=60, t=80, b=80)
    )
    
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=12))
    
    return fig


def create_team_legend(teams: List[str], colors: Dict[str, str]) -> str:
    """
    Create HTML legend for team colors.
    
    Args:
        teams: List of team names
        colors: Dictionary mapping team names to colors
        
    Returns:
        HTML string for legend
    """
    legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin: 10px 0;'>"
    for team in teams:
        color = colors.get(team, '#808080')
        legend_html += f"""
        <div style='display: flex; align-items: center;'>
            <div style='width: 20px; height: 20px; background-color: {color}; 
                        border: 1px solid #ccc; margin-right: 5px;'></div>
            <span>{team}</span>
        </div>
        """
    legend_html += "</div>"
    return legend_html

