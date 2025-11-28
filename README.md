# Lead Response Heatmap Dashboard

## The Fundamental Problem

Before we examine the solution, we must understand the problem with perfect clarity.

Sales organizations generate leads. Each lead represents a potential customer who has expressed interest. The critical variable is **response time**—the interval between when a lead arrives and when a sales representative makes first contact.

Research demonstrates that leads contacted within one hour convert at dramatically higher rates than those contacted later. Yet most organizations have no systematic way to identify *when* their response times are weakest.

This application solves that problem. It transforms raw lead data into a visual heatmap that immediately reveals performance patterns across time and day.

---

## The Architecture: Understanding the Structure

The system follows a three-layer architecture. Each layer has a single responsibility. This separation is not arbitrary—it is the foundation upon which maintainable software is built.

```
┌─────────────────────────────────────────────────────────┐
│                      app.py                              │
│              (Presentation Layer - UI)                   │
│         What the user sees and interacts with            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     logic.py                             │
│              (Business Logic Layer)                      │
│         All data transformations and calculations        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    heatmap.py                            │
│              (Visualization Layer)                       │
│         Converts processed data into visual charts       │
└─────────────────────────────────────────────────────────┘
```

Why this structure? Because when you need to change the visualization library, you modify only `heatmap.py`. When business rules change, you modify only `logic.py`. When the UI needs updating, you modify only `app.py`. Each component can be understood, tested, and modified in isolation.

---

## The Technology Stack: Why These Choices

### Python
The foundation. Python's readability makes the code accessible to analysts who are not software engineers. Its data ecosystem is unmatched.

### Pandas (`logic.py`)
For tabular data manipulation, pandas is the correct tool. It handles:
- DateTime parsing and arithmetic
- Grouping and aggregation
- Missing value handling

These operations would require hundreds of lines of manual code. Pandas reduces them to single function calls.

### Plotly (`heatmap.py`)
We chose Plotly over Matplotlib for one critical reason: **interactivity**. Hover over any cell and you see the exact values. This transforms a static image into an investigative tool.

### Streamlit (`app.py`)
Streamlit eliminates the complexity of web development. No HTML. No JavaScript. No CSS frameworks. Pure Python that produces a functional web application.

---

## The Data Flow: Step by Step

Let us trace exactly what happens when the application runs.

### Step 1: Data Ingestion

The process begins in `logic.py` with the `load_leads()` function:

```python
def load_leads(file_or_path):
    df = pd.read_csv(file_or_path)
    df['received_time'] = pd.to_datetime(df['received_time'])
    df['first_contact_time'] = pd.to_datetime(df['first_contact_time'], errors='coerce')
    return df
```

Two operations occur:
1. CSV text is parsed into a DataFrame
2. String timestamps become proper DateTime objects

The `errors='coerce'` parameter is significant. When a lead was never contacted, the `first_contact_time` is blank. Rather than failing, pandas converts these to `NaT` (Not a Time)—the DateTime equivalent of null.

### Step 2: Response Time Calculation

The `compute_response_metrics()` function adds two columns:

```python
df['response_minutes'] = (df['first_contact_time'] - df['received_time']).dt.total_seconds() / 60
df['contact_within_1hr'] = df['response_minutes'] <= 60
```

This is the core calculation. We subtract timestamps to get a duration, convert to minutes, then create a boolean flag.

### Step 3: Time Bucketing

Here is where the analysis becomes powerful. The `bucket_time()` function floors each timestamp to the nearest interval:

```python
def bucket_time(dt, bucket_minutes):
    total_minutes = dt.hour * 60 + dt.minute
    floored_minutes = (total_minutes // bucket_minutes) * bucket_minutes
    new_hour = floored_minutes // 60
    new_minute = floored_minutes % 60
    return datetime(2000, 1, 1, new_hour, new_minute).time()
```

With 30-minute buckets:
- 08:15 → 08:00
- 08:45 → 08:30
- 09:00 → 09:00

This transforms continuous time into discrete buckets that can be grouped and compared.

### Step 4: Aggregation

The `aggregate_by_slot()` function is the analytical core:

```python
aggregated = df.groupby(['day_of_week', 'time_bucket']).agg(
    total_leads=('lead_id', 'count'),
    contacted_within_1hr=('contact_within_1hr', 'sum')
).reset_index()

aggregated['pct_within_1hr'] = (
    aggregated['contacted_within_1hr'] / aggregated['total_leads'] * 100
)
```

For each combination of day and time slot, we calculate:
- How many leads arrived
- How many were contacted within one hour
- What percentage that represents

### Step 5: Visualization

The `create_performance_heatmap()` function in `heatmap.py` transforms this aggregated data into a visual matrix:

```python
fig = go.Figure(data=go.Heatmap(
    z=pivot_df.values,
    x=time_labels,
    y=days_in_data,
    colorscale=[
        [0.0, '#d73027'],    # Red for 0%
        [0.5, '#fee08b'],    # Yellow for 50%
        [1.0, '#1a9850']     # Green for 100%
    ],
    zmin=0,
    zmax=100
))
```

The color scale maps directly to performance:
- **Red**: Failure. Leads are not being contacted quickly.
- **Yellow**: Moderate. Room for improvement.
- **Green**: Success. The system is working.

### Step 6: Presentation

Finally, `app.py` orchestrates everything through Streamlit:

```python
leads_df, schedule_df = load_data()           # Step 1
stats = calculate_summary_stats(leads_df)     # Summary metrics
aggregated_df = aggregate_by_slot(leads_df, bucket_minutes)  # Steps 2-4
performance_fig = create_performance_heatmap(aggregated_df, ...)  # Step 5
st.plotly_chart(performance_fig)              # Step 6
```

The user sees a heatmap. Behind it, these six steps execute in sequence.

---

## The File Structure

```
/HeatMap_Dashboard
│
├── app.py                 # Entry point. Run this with Streamlit.
│                          # Handles: file uploads, user settings, display
│
├── logic.py               # Pure data processing. No visualization.
│                          # Handles: loading, parsing, calculating, aggregating
│
├── heatmap.py             # Pure visualization. No data processing.
│                          # Handles: creating Plotly figures
│
├── requirements.txt       # Dependencies with version constraints
│
└── sample_data/
    ├── leads.csv          # 110 sample lead records
    └── schedule.csv       # Staff shift schedules for 5 reps
```

---

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py

# Open browser to http://localhost:8501
```

---

## Adapting for Enterprise Use

The prototype demonstrates the concept. Enterprise deployment requires additional considerations.

### 1. Data Source Integration

The current system reads CSV files. In production, you would connect to your CRM directly.

```python
# Replace this:
df = pd.read_csv(file_or_path)

# With this:
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@host:5432/crm')
df = pd.read_sql('SELECT * FROM leads WHERE received_time > NOW() - INTERVAL 30 DAY', engine)
```

The logic layer remains unchanged. Only the data source differs.

### 2. Authentication

Streamlit supports authentication through community packages:

```python
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials, cookie_name, key, cookie_expiry_days
)
name, authentication_status, username = authenticator.login()
```

For enterprise deployment, integrate with your organization's SSO provider (Okta, Azure AD, etc.).

### 3. Caching and Performance

With larger datasets, add caching:

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_process_leads(date_range):
    # Expensive database query and processing
    return processed_df
```

This prevents redundant calculations when users interact with the UI.

### 4. Scheduling and Automation

Deploy as a scheduled report:

```python
# Using APScheduler or similar
scheduler.add_job(
    generate_weekly_report,
    trigger='cron',
    day_of_week='mon',
    hour=6
)
```

Email the heatmap to managers every Monday morning.

### 5. Multi-Team Support

Add team filtering to the aggregation:

```python
def aggregate_by_slot(df, bucket_minutes, team=None):
    if team:
        df = df[df['team'] == team]
    # Continue with aggregation
```

Each team sees only their performance.

### 6. Historical Comparison

Add a date range selector and compute period-over-period changes:

```python
current_period = aggregate_by_slot(current_df, bucket_minutes)
previous_period = aggregate_by_slot(previous_df, bucket_minutes)
delta = current_period['pct_within_1hr'] - previous_period['pct_within_1hr']
```

Show not just current performance, but whether it is improving or declining.

### 7. Deployment Options

| Option | Complexity | Best For |
|--------|------------|----------|
| Streamlit Community Cloud | Low | Small teams, quick sharing |
| Docker + Cloud Run | Medium | Scalable, serverless |
| Kubernetes | High | Enterprise, multi-tenant |

For most organizations, a containerized deployment behind your corporate VPN provides the right balance of accessibility and security.

---

## The Principle Behind the Design

The fundamental insight is this: **complexity should be proportional to the problem being solved**.

This application could have been built with React, TypeScript, a REST API, a PostgreSQL database, Redis caching, and Kubernetes orchestration. That would be engineering for engineering's sake.

Instead, we have three Python files totaling approximately 500 lines. A junior developer can read and understand the entire system in an afternoon. Modifications can be made with confidence.

When you scale to enterprise, you add complexity *where the problem demands it*—authentication, caching, database connections—while preserving the simplicity of the core logic.

The best systems are those where the architecture serves the problem, not the other way around.

---

## Summary

1. **Problem**: Identify when lead response times are weakest
2. **Solution**: Visual heatmap of performance by day and time
3. **Architecture**: Three-layer separation (UI, Logic, Visualization)
4. **Stack**: Python, Pandas, Plotly, Streamlit
5. **Data Flow**: Load → Calculate → Bucket → Aggregate → Visualize → Display
6. **Enterprise Path**: Add authentication, database integration, caching, and containerization

The code is the documentation. Read it systematically. Understand each function's purpose. The system will reveal itself.

