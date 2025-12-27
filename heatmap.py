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


def create_team_heatmap(
    team_df: pd.DataFrame, 
    team_name: str,
    zmin: float = 0,
    zmax: float = 100
) -> go.Figure:
    """
    Create a performance heatmap for a single team.
    
    Args:
        team_df: Aggregated DataFrame for one team (from aggregate_by_team_slot)
        team_name: Name of the team
        zmin: Minimum value for color scale
        zmax: Maximum value for color scale
        
    Returns:
        Plotly Figure with heatmap
    """
    # Pivot to create matrix (days x time buckets)
    pivot_df = team_df.pivot_table(
        index='day_of_week',
        columns='time_bucket',
        values='pct_within_1hr',
        fill_value=0
    )
    
    # Order days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_in_data = [d for d in day_order if d in pivot_df.index]
    pivot_df = pivot_df.reindex(days_in_data)
    
    # Format time labels
    time_labels = [t.strftime('%H:%M') for t in sorted(pivot_df.columns)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=time_labels,
        y=days_in_data,
        colorscale=[
            [0.0, '#d73027'],    # Red for 0%
            [0.5, '#fee08b'],    # Yellow for 50%
            [1.0, '#1a9850']     # Green for 100%
        ],
        zmin=zmin,
        zmax=zmax,
        text=[[f"{val:.1f}%" for val in row] for row in pivot_df.values],
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Percentage: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="% Within 1hr")
    ))
    
    fig.update_layout(
        title=f'{team_name} - Response Performance Heatmap',
        xaxis_title='Time of Day',
        yaxis_title='Day of Week',
        height=500,
        width=1000
    )
    
    return fig


def create_winner_heatmap(
    winner_df: pd.DataFrame,
    team_colors: Dict[str, str]
) -> go.Figure:
    """
    Create a heatmap showing which team wins each time slot.
    
    Args:
        winner_df: DataFrame from find_slot_winners
        team_colors: Dictionary mapping team names to colors
        
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
    
    # Order days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_in_data = [d for d in day_order if d in pivot_teams.index]
    pivot_teams = pivot_teams.reindex(days_in_data)
    pivot_pct = pivot_pct.reindex(days_in_data)
    
    # Format time labels
    time_labels = [t.strftime('%H:%M') for t in sorted(pivot_teams.columns)]
    
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
    for day in days_in_data:
        row_text = []
        for time_bucket in sorted(pivot_teams.columns):
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
        y=days_in_data,
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
        title='Winner Heatmap - Which Team Leads Each Time Slot',
        xaxis_title='Time of Day',
        yaxis_title='Day of Week',
        height=500,
        width=1000
    )
    
    return fig


def create_difference_heatmap(
    diff_df: pd.DataFrame,
    team_a: str,
    team_b: str
) -> go.Figure:
    """
    Create a heatmap showing the difference between two teams.
    
    Args:
        diff_df: DataFrame from compute_team_difference
        team_a: First team name
        team_b: Second team name
        
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
    
    # Order days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_in_data = [d for d in day_order if d in pivot_df.index]
    pivot_df = pivot_df.reindex(days_in_data)
    
    # Format time labels
    time_labels = [t.strftime('%H:%M') for t in sorted(pivot_df.columns)]
    
    # Create hover text with both team percentages
    hover_text = []
    for day in days_in_data:
        row_text = []
        for time_bucket in sorted(diff_df['time_bucket'].unique()):
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
        y=days_in_data,
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
        title=f'Head-to-Head: {team_a} vs {team_b}',
        xaxis_title='Time of Day',
        yaxis_title='Day of Week',
        height=500,
        width=1000
    )
    
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

