# Multi-Team Lead Response Comparison Dashboard

A Streamlit application that visualizes and compares lead response performance across multiple sales teams, making it easy to identify which team outperforms others at specific times.

## Features

- **Flexible Data Input**: Upload a single CSV with team column, or multiple team-specific CSV files
- **Side-by-Side Team Heatmaps**: Compare all teams visually with consistent color scales
- **Winner Heatmap**: See which team leads at each time slot
- **Head-to-Head Comparison**: Direct comparison between any two teams
- **Team Leaderboard**: Overall performance rankings and statistics

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Use the startup script (Recommended)
```bash
./run.sh
```

### Option 2: Run directly
```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## Installation

If you need to install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Load Sample Data**: Click the "📁 Load Sample Data" button in the sidebar to instantly load sample data with two teams (Alpha and Beta)

2. **Or Upload Your Own Data**: 
   - Upload a single CSV file with a `team` column, OR
   - Upload multiple CSV files (one per team)
   - Click "📥 Load Data" to process

3. **Explore the Dashboard**:
   - **Side-by-Side Heatmaps**: Compare all teams visually
   - **Winner Heatmap**: See which team leads at each time slot  
   - **Head-to-Head**: Direct comparison between two teams
   - **Leaderboard**: Overall performance rankings

4. Explore the different views:
   - **Side-by-Side Heatmaps**: Compare all teams visually
   - **Winner Heatmap**: See which team leads at each time slot
   - **Head-to-Head**: Direct comparison between two teams
   - **Leaderboard**: Overall performance rankings

## Data Format

### Single CSV with Team Column

```csv
lead_id,received_time,first_contact_time,team
L001,2024-01-15 09:23:00,2024-01-15 09:45:00,Alpha
L002,2024-01-15 09:45:00,2024-01-15 10:55:00,Beta
L003,2024-01-15 10:15:00,,Alpha
```

### Multiple Team Files

Upload separate CSV files (one per team). The team name can be derived from the filename or specified during upload.

## Architecture

- `app.py`: Streamlit UI and orchestration
- `logic.py`: Data processing and aggregation
- `heatmap.py`: Visualization functions

