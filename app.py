import os
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt
import json
from google.cloud import bigquery
# ---------- GCP Credentials ----------
import os, json, tempfile, streamlit as st
from google.cloud import bigquery

PROJECT_ID = "ba882-f25-class-project-team9"

svc = st.secrets.get("gcp_service_account")
sa_info = None
if svc is not None:
    if isinstance(svc, dict):
        # 你在 Secrets 里用了 TOML 表（见下面方案 B）
        sa_info = dict(svc)
    elif isinstance(svc, str):
        # 你在 Secrets 里用了三引号包的整段 JSON（见下面方案 A）
        sa_info = json.loads(svc.strip())

if sa_info:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(json.dumps(sa_info).encode())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
    os.environ["GCP_PROJECT_ID"] = PROJECT_ID
else:
    # 本地兜底（不进 Git）
    cred_path = os.path.join(os.path.dirname(__file__), "credentials.json")
    if os.path.exists(cred_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        os.environ["GCP_PROJECT_ID"] = PROJECT_ID

client = bigquery.Client(project=PROJECT_ID)

# ----------------------- CONFIG -----------------------
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ba882-f25-class-project-team9")
T_MBTA_STATIONS = f"{PROJECT_ID}.mbta_data.stations_dedup"
T_BB_INFO      = f"{PROJECT_ID}.bluebikes_analysis.station_info"
T_BB_STATUS    = f"{PROJECT_ID}.bluebikes_analysis.station_status"
T_BB_HIST      = f"{PROJECT_ID}.bluebikes_historical_us.JantoSep_historical"
# Prediction tables produced by your Airflow BQML pipeline:
T_PRED_PICKUPS = f"{PROJECT_ID}.bluebikes_analysis.predicted_pickups"
T_PRED_DROPS   = f"{PROJECT_ID}.bluebikes_analysis.predicted_dropoffs"
# NEW: ML Clustering results
T_CLUSTERS     = f"{PROJECT_ID}.bluebikes_analysis.mbta_station_clusters"

client = bigquery.Client(project=PROJECT_ID)

# ----------------------- UTIL: QUERY -----------------------
@st.cache_data(ttl=600)
def run_query(sql: str, _params=None, cache_key: str = "") -> pd.DataFrame:
    _ = cache_key  # ensures re-compute when the key changes
    job_config = bigquery.QueryJobConfig(query_parameters=_params or [])
    return client.query(sql, job_config=job_config).to_dataframe()

# ----------------------- SIDEBAR -----------------------
st.sidebar.header("Filters")
miles = st.sidebar.slider("Radius (miles)", 0.25, 2.0, 1.0, 0.25)
meters = miles * 1609.34
use_avail = st.sidebar.checkbox(
    "Require historical availability (≥30% active last 14 days)", value=False
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Pages",
    [ "Review_BB", "Overview","Summary", "Recommendations", "Bluebikes Clustering", "Bluebikes Demand Prediction", "Availability","Occupancy Prediction" ]
)

# ----------------------- SQL (Coverage) -----------------------
SQL_CORE = f"""
WITH mbta AS (
  SELECT
    station_id,
    station_name,
    CAST(lat AS FLOAT64) AS lat,
    CAST(lng AS FLOAT64) AS lng,
    ST_GEOGPOINT(CAST(lng AS FLOAT64), CAST(lat AS FLOAT64)) AS geog
  FROM `{T_MBTA_STATIONS}`
),
bb_latest AS (
  SELECT
    station_id AS bb_station_id,
    name       AS bb_name,
    CAST(lat AS FLOAT64) AS lat,
    CAST(lon AS FLOAT64) AS lon,
    ST_GEOGPOINT(CAST(lon AS FLOAT64), CAST(lat AS FLOAT64)) AS geog
  FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY snapshot_ts DESC) rn
    FROM `{T_BB_INFO}`
  )
  WHERE rn = 1
),
pairs AS (
  SELECT
    m.station_id, m.station_name,
    m.lat AS mbta_lat, m.lng AS mbta_lng,
    b.bb_station_id,
    CASE WHEN b.bb_station_id IS NOT NULL THEN ST_DISTANCE(m.geog, b.geog) END AS distance_m
  FROM mbta m
  LEFT JOIN bb_latest b
    ON ST_DWITHIN(m.geog, b.geog, @meters)
)
SELECT
  station_id,
  ANY_VALUE(station_name) AS station_name,
  ANY_VALUE(mbta_lat)     AS mbta_lat,
  ANY_VALUE(mbta_lng)     AS mbta_lng,
  COUNT(DISTINCT bb_station_id) AS nearby_bluebikes_count,
  MIN(distance_m)              AS nearest_distance_m,
  (COUNT(DISTINCT bb_station_id) >= 1) AS meets_last_mile
FROM pairs
GROUP BY station_id
ORDER BY meets_last_mile DESC, nearest_distance_m;
"""

SQL_WITH_AVAIL = f"""
WITH window AS (
  SELECT TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY) AS start_ts
),
bb_latest AS (
  SELECT
    station_id AS bb_station_id,
    name       AS bb_name,
    CAST(lat AS FLOAT64) AS lat,
    CAST(lon AS FLOAT64) AS lon,
    ST_GEOGPOINT(CAST(lon AS FLOAT64), CAST(lat AS FLOAT64)) AS geog
  FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY snapshot_ts DESC) rn
    FROM `{T_BB_INFO}`
  ) WHERE rn = 1
),
bb_avail AS (
  SELECT
    station_id AS bb_station_id,
    AVG( IFNULL(num_bikes_available,0) > 0 ) AS active_ratio
  FROM `{T_BB_STATUS}`, window
  WHERE snapshot_ts >= window.start_ts
  GROUP BY bb_station_id
),
mbta AS (
  SELECT
    station_id,
    station_name,
    CAST(lat AS FLOAT64) AS lat,
    CAST(lng AS FLOAT64) AS lng,
    ST_GEOGPOINT(CAST(lng AS FLOAT64), CAST(lat AS FLOAT64)) AS geog
  FROM `{T_MBTA_STATIONS}`
),
pairs AS (
  SELECT
    m.station_id, m.station_name,
    m.lat AS mbta_lat, m.lng AS mbta_lng,
    b.bb_station_id,
    CASE WHEN b.bb_station_id IS NOT NULL THEN ST_DISTANCE(m.geog, b.geog) END AS distance_m,
    a.active_ratio
  FROM mbta m
  LEFT JOIN bb_latest b
    ON ST_DWITHIN(m.geog, b.geog, @meters)
  LEFT JOIN bb_avail a
    ON a.bb_station_id = b.bb_station_id
)
SELECT
  station_id,
  ANY_VALUE(station_name) AS station_name,
  ANY_VALUE(mbta_lat)     AS mbta_lat,
  ANY_VALUE(mbta_lng)     AS mbta_lng,
  COUNTIF(IFNULL(active_ratio,0) >= 0.3) AS nearby_active_bluebikes,
  MIN(CASE WHEN IFNULL(active_ratio,0) >= 0.3 THEN distance_m END) AS nearest_distance_m,
  (COUNTIF(IFNULL(active_ratio,0) >= 0.3) >= 1) AS meets_last_mile
FROM pairs
GROUP BY station_id
ORDER BY meets_last_mile DESC, nearest_distance_m;
"""

# ----------------------- SQL (Availability) -----------------------
SQL_NEAREST_AVAIL = f"""
WITH m AS (
  SELECT
    station_id,
    station_name,
    CAST(lat AS FLOAT64) AS lat,
    CAST(lng AS FLOAT64) AS lng,
    ST_GEOGPOINT(CAST(lng AS FLOAT64), CAST(lat AS FLOAT64)) AS geog
  FROM `{T_MBTA_STATIONS}`
  WHERE station_id = @mbta_station_id
),
bb_latest AS (
  SELECT
    station_id AS bb_station_id,
    name AS bb_name,
    CAST(lat AS FLOAT64) AS lat,
    CAST(lon AS FLOAT64) AS lon,
    ST_GEOGPOINT(CAST(lon AS FLOAT64), CAST(lat AS FLOAT64)) AS geog
  FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY snapshot_ts DESC) rn
    FROM `{T_BB_INFO}`
  ) WHERE rn = 1
),
bb_status_latest AS (
  SELECT
    station_id AS bb_station_id,
    num_bikes_available,
    num_docks_available,
    is_renting,
    is_returning,
    snapshot_ts AS status_snapshot_ts
  FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY snapshot_ts DESC) rn
    FROM `{T_BB_STATUS}`
  ) WHERE rn = 1
),
nearest AS (
  SELECT
    m.station_id AS mbta_station_id,
    m.station_name AS mbta_station_name,
    m.lat AS mbta_lat, m.lng AS mbta_lng,
    b.bb_station_id, b.bb_name,
    b.lat AS bb_lat, b.lon AS bb_lng,
    ST_DISTANCE(m.geog, b.geog) AS distance_m
  FROM m
  CROSS JOIN bb_latest b
  ORDER BY distance_m
  LIMIT 1
)
SELECT
  n.*,
  s.num_bikes_available,
  s.num_docks_available,
  s.is_renting,
  s.is_returning,
  s.status_snapshot_ts
FROM nearest n
LEFT JOIN bb_status_latest s
  ON s.bb_station_id = n.bb_station_id
"""

# ----------------------- PAGE: TITLE -----------------------
st.title("🚲 MBTA × Bluebikes — Last-Mile Coverage")

# ======================= NEW PAGE: ML CLUSTERING =======================
if page == "Bluebikes Clustering":
    st.header("🤖 ML Clustering Analysis — Transit Desert Identification")
    st.caption("Using unsupervised K-means clustering to identify MBTA stations with poor last-mile connectivity")
    
    # Load clustering data - get most recent run
    SQL_CLUSTERS = f"""
    WITH latest_run AS (
      SELECT run_id
      FROM `{T_CLUSTERS}`
      ORDER BY run_id DESC
      LIMIT 1
    )
    SELECT DISTINCT
      c.station_id,
      c.station_name,
      c.lat,
      c.lng,
      c.cluster,
      c.cluster_label,
      c.distance_to_nearest_bluebikes_m,
      c.avg_bikes_available_morning,
      c.avg_bikes_available_evening
    FROM `{T_CLUSTERS}` c
    INNER JOIN latest_run lr ON c.run_id = lr.run_id
    """
    
    with st.spinner("Loading clustering results..."):
        df_clusters = run_query(SQL_CLUSTERS, cache_key="ml_clustering_v1")
    
    if df_clusters.empty:
        st.warning("No clustering results found. Please run the clustering DAG first.")
        st.stop()
    
    # ----------------------- EXECUTIVE SUMMARY -----------------------
    st.subheader("📊 Executive Summary")
    
    # Calculate summary statistics
    total_stations = len(df_clusters)
    transit_deserts = len(df_clusters[df_clusters['cluster_label'] == 'Transit Desert'])
    well_served = len(df_clusters[df_clusters['cluster_label'] == 'Well-Served'])
    moderate = len(df_clusters[df_clusters['cluster_label'] == 'Moderate Coverage'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total MBTA Stations", total_stations)
    with col2:
        st.metric("🔴 Transit Deserts", transit_deserts, 
                 delta=f"{transit_deserts/total_stations*100:.1f}%",
                 delta_color="inverse")
    with col3:
        st.metric("🟢 Well-Served", well_served,
                 delta=f"{well_served/total_stations*100:.1f}%")
    with col4:
        st.metric("🟡 Moderate Coverage", moderate,
                 delta=f"{moderate/total_stations*100:.1f}%")
    
    st.markdown("---")
    
    # ----------------------- CLUSTER CHARACTERISTICS -----------------------
    st.subheader("📈 Cluster Characteristics")
    
    cluster_stats = df_clusters.groupby('cluster_label').agg({
        'station_id': 'count',
        'distance_to_nearest_bluebikes_m': 'mean',
        'avg_bikes_available_morning': 'mean',
        'avg_bikes_available_evening': 'mean'
    }).round(1)
    
    cluster_stats.columns = ['Station Count', 'Avg Distance (m)', 'Avg Morning Bikes', 'Avg Evening Bikes']
    cluster_stats = cluster_stats.reset_index()
    
    # Display as formatted table
    st.dataframe(
        cluster_stats.style.format({
            'Station Count': '{:.0f}',
            'Avg Distance (m)': '{:.0f}',
            'Avg Morning Bikes': '{:.1f}',
            'Avg Evening Bikes': '{:.1f}'
        }).background_gradient(subset=['Avg Distance (m)'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # Visualization of cluster distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribution by Cluster Type**")
        cluster_dist = df_clusters['cluster_label'].value_counts().reset_index()
        cluster_dist.columns = ['cluster_label', 'count']
        
        chart = alt.Chart(cluster_dist).mark_bar().encode(
            x=alt.X('count:Q', title='Number of Stations'),
            y=alt.Y('cluster_label:N', title='Cluster Type',
                   sort=['Transit Desert', 'Moderate Coverage', 'Well-Served']),
            color=alt.Color('cluster_label:N', 
                          scale=alt.Scale(
                              domain=['Transit Desert', 'Moderate Coverage', 'Well-Served'],
                              range=['#d62728', '#ff7f0e', '#2ca02c']
                          ),
                          legend=None)
        ).properties(height=200)
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("**Average Distance to Nearest Bluebikes**")
        avg_dist = df_clusters.groupby('cluster_label')['distance_to_nearest_bluebikes_m'].mean().reset_index()
        
        chart = alt.Chart(avg_dist).mark_bar().encode(
            x=alt.X('distance_to_nearest_bluebikes_m:Q', title='Distance (meters)'),
            y=alt.Y('cluster_label:N', title='Cluster Type',
                   sort=['Transit Desert', 'Moderate Coverage', 'Well-Served']),
            color=alt.Color('cluster_label:N', 
                          scale=alt.Scale(
                              domain=['Transit Desert', 'Moderate Coverage', 'Well-Served'],
                              range=['#d62728', '#ff7f0e', '#2ca02c']
                          ),
                          legend=None)
        ).properties(height=200)
        st.altair_chart(chart, use_container_width=True)
    
    st.markdown("---")
    
    # ----------------------- INTERACTIVE MAP -----------------------
    st.subheader("🗺️ Interactive Cluster Map")
    st.caption("Click on stations to see details. Color indicates cluster type.")
    
    # Prepare data for map
    df_map = df_clusters.copy()
    
    # Define colors for each cluster type
    color_map = {
        'Transit Desert': [214, 39, 40, 200],      # Red
        'Moderate Coverage': [255, 127, 14, 200],  # Orange
        'Well-Served': [44, 160, 44, 200]          # Green
    }
    df_map['color'] = df_map['cluster_label'].map(color_map)
    
    # Filter by cluster type
    cluster_filter = st.multiselect(
        "Filter by cluster type:",
        options=['Transit Desert', 'Moderate Coverage', 'Well-Served'],
        default=['Transit Desert', 'Moderate Coverage', 'Well-Served']
    )
    
    df_map_filtered = df_map[df_map['cluster_label'].isin(cluster_filter)]
    
    # Create pydeck map
    view_state = pdk.ViewState(
        latitude=df_map_filtered['lat'].mean(),
        longitude=df_map_filtered['lng'].mean(),
        zoom=10,
        pitch=0
    )
    
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=df_map_filtered,
        get_position='[lng, lat]',
        get_color='color',
        get_radius=100,
        pickable=True,
        auto_highlight=True
    )
    
    tooltip = {
        "html": "<b>{station_name}</b><br/>"
                "Type: {cluster_label}<br/>"
                "Distance to Bluebikes: {distance_to_nearest_bluebikes_m:.0f}m<br/>"
                "Morning bikes: {avg_bikes_available_morning:.1f}<br/>"
                "Evening bikes: {avg_bikes_available_evening:.1f}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='mapbox://styles/mapbox/light-v10'
    )
    
    st.pydeck_chart(deck, use_container_width=True)
    
    st.markdown("---")
    
    # ----------------------- DETAILED TABLES -----------------------
    st.subheader("📋 Detailed Station Lists")
    
    tab1, tab2, tab3 = st.tabs(["🔴 Transit Deserts", "🟡 Moderate Coverage", "🟢 Well-Served"])
    
    with tab1:
        st.markdown("### Transit Desert Stations")
        st.caption("These stations have poor last-mile connectivity and should be prioritized for improvement.")
        
        transit_desert_df = df_clusters[df_clusters['cluster_label'] == 'Transit Desert'].copy()
        transit_desert_df = transit_desert_df.sort_values('distance_to_nearest_bluebikes_m', ascending=False)
        
        # Display table with formatting
        st.dataframe(
            transit_desert_df[[
                'station_name', 
                'distance_to_nearest_bluebikes_m', 
                'avg_bikes_available_morning',
                'avg_bikes_available_evening'
            ]].rename(columns={
                'station_name': 'Station Name',
                'distance_to_nearest_bluebikes_m': 'Distance to Bluebikes (m)',
                'avg_bikes_available_morning': 'Morning Bikes Available',
                'avg_bikes_available_evening': 'Evening Bikes Available'
            }).style.format({
                'Distance to Bluebikes (m)': '{:.0f}',
                'Morning Bikes Available': '{:.1f}',
                'Evening Bikes Available': '{:.1f}'
            }).background_gradient(subset=['Distance to Bluebikes (m)'], cmap='Reds'),
            use_container_width=True,
            height=400
        )
        
        # Top 10 worst
        st.markdown("**Top 10 Most Isolated Stations**")
        top_10_worst = transit_desert_df.nlargest(10, 'distance_to_nearest_bluebikes_m')
        
        chart = alt.Chart(top_10_worst).mark_bar(color='#d62728').encode(
            y=alt.Y('station_name:N', sort='-x', title=None),
            x=alt.X('distance_to_nearest_bluebikes_m:Q', title='Distance to Nearest Bluebikes (meters)'),
            tooltip=['station_name', 'distance_to_nearest_bluebikes_m']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    
    with tab2:
        st.markdown("### Moderate Coverage Stations")
        st.caption("These stations have decent but not ideal connectivity.")
        
        moderate_df = df_clusters[df_clusters['cluster_label'] == 'Moderate Coverage'].copy()
        moderate_df = moderate_df.sort_values('distance_to_nearest_bluebikes_m', ascending=False)
        
        st.dataframe(
            moderate_df[[
                'station_name', 
                'distance_to_nearest_bluebikes_m', 
                'avg_bikes_available_morning',
                'avg_bikes_available_evening'
            ]].rename(columns={
                'station_name': 'Station Name',
                'distance_to_nearest_bluebikes_m': 'Distance to Bluebikes (m)',
                'avg_bikes_available_morning': 'Morning Bikes Available',
                'avg_bikes_available_evening': 'Evening Bikes Available'
            }).style.format({
                'Distance to Bluebikes (m)': '{:.0f}',
                'Morning Bikes Available': '{:.1f}',
                'Evening Bikes Available': '{:.1f}'
            }),
            use_container_width=True,
            height=400
        )
    
    with tab3:
        st.markdown("### Well-Served Stations")
        st.caption("These stations have excellent last-mile connectivity.")
        
        well_served_df = df_clusters[df_clusters['cluster_label'] == 'Well-Served'].copy()
        well_served_df = well_served_df.sort_values('avg_bikes_available_morning', ascending=False)
        
        st.dataframe(
            well_served_df[[
                'station_name', 
                'distance_to_nearest_bluebikes_m', 
                'avg_bikes_available_morning',
                'avg_bikes_available_evening'
            ]].rename(columns={
                'station_name': 'Station Name',
                'distance_to_nearest_bluebikes_m': 'Distance to Bluebikes (m)',
                'avg_bikes_available_morning': 'Morning Bikes Available',
                'avg_bikes_available_evening': 'Evening Bikes Available'
            }).style.format({
                'Distance to Bluebikes (m)': '{:.0f}',
                'Morning Bikes Available': '{:.1f}',
                'Evening Bikes Available': '{:.1f}'
            }).background_gradient(subset=['Morning Bikes Available'], cmap='Greens'),
            use_container_width=True,
            height=400
        )
    
    st.markdown("---")
    
    # ----------------------- RECOMMENDATIONS -----------------------
    st.subheader("💡 Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### High Priority Actions")
        st.markdown(f"""
        1. **Add Bluebikes stations** near {transit_deserts} Transit Desert MBTA stations
        2. **Focus on** stations with distance > 800m (10-min walk)
        3. **Prioritize** high-ridership MBTA lines
        4. **Increase rebalancing** frequency for stations with morning availability < 2 bikes
        """)
    
    with col2:
        st.markdown("### Key Insights")
        avg_desert_distance = df_clusters[df_clusters['cluster_label'] == 'Transit Desert']['distance_to_nearest_bluebikes_m'].mean()
        avg_served_distance = df_clusters[df_clusters['cluster_label'] == 'Well-Served']['distance_to_nearest_bluebikes_m'].mean()
        
        st.markdown(f"""
        - Transit Deserts average **{avg_desert_distance:.0f}m** from nearest Bluebikes
        - Well-Served stations average **{avg_served_distance:.0f}m** from nearest Bluebikes
        - **{transit_deserts/total_stations*100:.1f}%** of MBTA stations need improvement
        - Morning availability is critical for commuter usage
        """)
    
    # Download button for data
    st.markdown("---")
    csv = df_clusters.to_csv(index=False)
    st.download_button(
        label="📥 Download Clustering Results (CSV)",
        data=csv,
        file_name="mbta_transit_desert_analysis.csv",
        mime="text/csv"
    )

# ----------------------- PAGE: Review_BB -----------------------
elif page == "Review_BB":
    st.header("📋 Bluebikes — System Review (Historical)")
    st.caption("Source: bluebikes_historical.JantoSep_historical")

    YEAR = 2025
    params_year = [bigquery.ScalarQueryParameter("yy", "INT64", YEAR)]

    sql_total = f"""
        SELECT COUNT(1) AS total_rides
        FROM `{T_BB_HIST}`
        WHERE EXTRACT(YEAR FROM started_at) = @yy
    """
    total_rides = int(run_query(sql_total, params_year, cache_key=f"rev_total::{YEAR}").iloc[0]["total_rides"])

    sql_top_start = f"""
        SELECT start_station_name AS station, COUNT(1) AS trips
        FROM `{T_BB_HIST}`
        WHERE EXTRACT(YEAR FROM started_at) = @yy
        GROUP BY station ORDER BY trips DESC LIMIT 10
    """
    sql_top_end = f"""
        SELECT end_station_name AS station, COUNT(1) AS trips
        FROM `{T_BB_HIST}`
        WHERE EXTRACT(YEAR FROM ended_at) = @yy
        GROUP BY station ORDER BY trips DESC LIMIT 10
    """
    df_top_start = run_query(sql_top_start, params_year, cache_key=f"rev_start::{YEAR}")
    df_top_end = run_query(sql_top_end, params_year, cache_key=f"rev_end::{YEAR}")

    sql_weekday = f"""
        SELECT
          FORMAT_TIMESTAMP('%A', started_at, 'America/New_York') AS weekday,
          member_casual, rideable_type, COUNT(1) AS trips
        FROM `{T_BB_HIST}`
        WHERE EXTRACT(YEAR FROM started_at) = @yy
        GROUP BY weekday, member_casual, rideable_type
    """
    df_weekday = run_query(sql_weekday, params_year, cache_key=f"rev_weekday::{YEAR}")
    week_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    if not df_weekday.empty:
        df_weekday["weekday"] = pd.Categorical(df_weekday["weekday"], categories=week_order, ordered=True)

    sql_season = f"""
        SELECT
          CASE
            WHEN EXTRACT(MONTH FROM started_at) IN (12,1,2) THEN 'Winter'
            WHEN EXTRACT(MONTH FROM started_at) IN (3,4,5) THEN 'Spring'
            WHEN EXTRACT(MONTH FROM started_at) IN (6,7,8) THEN 'Summer'
            ELSE 'Fall' END AS season,
          rideable_type, COUNT(1) AS trips
        FROM `{T_BB_HIST}`
        WHERE EXTRACT(YEAR FROM started_at) = @yy
        GROUP BY season, rideable_type
    """
    df_season = run_query(sql_season, params_year, cache_key=f"rev_season::{YEAR}")
    if not df_season.empty:
        df_season["season"] = pd.Categorical(df_season["season"], categories=["Fall","Winter","Spring","Summer"], ordered=True)

    st.subheader("🚲 Start Station Top 10")
    if df_top_start.empty:
        st.info("No data.")
    else:
        st.altair_chart(
            alt.Chart(df_top_start).mark_bar().encode(
                x=alt.X("trips:Q", title="Trips"),
                y=alt.Y("station:N", sort="-x", title="Start Station Name"),
            ).properties(height=300),
            use_container_width=True
        )

    st.subheader("🅿️ End Station Top 10")
    if df_top_end.empty:
        st.info("No data.")
    else:
        st.altair_chart(
            alt.Chart(df_top_end).mark_bar().encode(
                x=alt.X("trips:Q", title="Trips"),
                y=alt.Y("station:N", sort="-x", title="End Station Name"),
            ).properties(height=300),
            use_container_width=True
        )

    st.subheader("📆 Week Day Analysis")
    if df_weekday.empty:
        st.info("No data.")
    else:
        base = (
            alt.Chart(df_weekday)
            .mark_bar()
            .encode(
                x=alt.X("weekday:N", sort=week_order, title=None,
                        axis=alt.Axis(labelAngle=0, labelPadding=8, labelLimit=0)),
                y=alt.Y("trips:Q", title="Total Trips"),
                color=alt.Color("rideable_type:N", title="Rideable Type"),
            )
            .properties(height=360, width=900)
        )
        chart_week = (
            base
            .facet(
                row=alt.Row("member_casual:N", title=None,
                            header=alt.Header(labelFontSize=16, labelPadding=10)),
                columns=1
            )
            .resolve_scale(y="independent")
            .configure_facet(spacing=60)
            .configure_axis(labelFontSize=12, titleFontSize=14)
        )
        st.altair_chart(chart_week, use_container_width=True)

    st.subheader("🌤️ Ride by Season")
    if df_season.empty:
        st.info("No data.")
    else:
        st.altair_chart(
            alt.Chart(df_season).mark_bar().encode(
                x=alt.X("season:N", title="Season"),
                y=alt.Y("trips:Q", title="Total Rides"),
                color=alt.Color("rideable_type:N", title=None),
            ).properties(height=320),
            use_container_width=True
        )

    st.markdown(f"### 🧮 Total Rides in {YEAR}: **{total_rides:,}**")
    st.stop()

# ----------------------- SHARED CAPTION -----------------------
st.caption(
    f"Rule: ≥1 Bluebikes station within {miles:.2f} mile(s) (~{meters:.0f} m). "
    + ("(Availability rule applied)" if use_avail else "")
)

# ----------------------- Fetch core data -----------------------
sql = SQL_WITH_AVAIL if use_avail else SQL_CORE
params = [bigquery.ScalarQueryParameter("meters", "FLOAT64", meters)]

with st.spinner("Querying BigQuery…"):
    df = run_query(sql, params, cache_key=f"core::{use_avail}::{meters:.2f}")

if df.empty:
    st.warning("No results found. Try increasing the radius.")
    st.stop()

if "nearest_distance_m" in df.columns:
    df["nearest_distance_m"] = df["nearest_distance_m"].astype(float).round(1)

meets = df[df["meets_last_mile"]].sort_values("nearest_distance_m", na_position="last")
fails = df[~df["meets_last_mile"]].sort_values("nearest_distance_m", na_position="last")

# ----------------------- MAP HELPER -----------------------
def render_map(df_sub, rgba, title):
    st.subheader(title)
    if df_sub.empty:
        st.info("No stations to display.")
        return
    points = pd.DataFrame({
        "lat": df_sub["mbta_lat"],
        "lon": df_sub["mbta_lng"],
        "name": df_sub["station_name"],
        "color": [rgba] * len(df_sub)
    })
    view = pdk.ViewState(
        latitude=points["lat"].mean() if points["lat"].notna().any() else 42.3601,
        longitude=points["lon"].mean() if points["lon"].notna().any() else -71.0589,
        zoom=10
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius=60,
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[layer], tooltip={"text": "{name}"}))

# ----------------------- Classification (for Recommendations) -----------------------
KEYWORDS_OUTER_CENTER = ("Center", "City Hall", "Downtown", "Square", "Sq")
KEYWORDS_SEASONAL = ("Beach", "Harbor", "Greenbush", "Newburyport", "Gloucester")
KEYWORDS_FAR_END = ("Route", "Parking Lot", "495", "Park", "Terminal", "End")

def classify_station(name: str) -> str:
    n = (name or "").lower()
    if any(k.lower() in n for k in KEYWORDS_OUTER_CENTER):
        return "outer_city_center"
    if any(k.lower() in n for k in KEYWORDS_SEASONAL):
        return "seasonal_or_tourism"
    if any(k.lower() in n for k in KEYWORDS_FAR_END):
        return "far_end_or_park_and_ride"
    return "other"

fails["category"] = fails["station_name"].apply(classify_station)
cat_counts = fails["category"].value_counts().to_dict()

# ----------------------- PAGE: Summary -----------------------
if page == "Summary":
    st.header("📊 Descriptive Summary")

    total_stations = len(df)
    num_meets = len(meets)
    num_fails = len(fails)
    percent_meets = (num_meets / total_stations * 100) if total_stations > 0 else 0.0

    median_distance = (
        df["nearest_distance_m"].median()
        if "nearest_distance_m" in df.columns and df["nearest_distance_m"].notna().any()
        else None
    )
    mean_distance = (
        df["nearest_distance_m"].mean()
        if "nearest_distance_m" in df.columns and df["nearest_distance_m"].notna().any()
        else None
    )
    p90_distance = (
        df["nearest_distance_m"].quantile(0.90)
        if "nearest_distance_m" in df.columns and df["nearest_distance_m"].notna().any()
        else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total MBTA Stations (filtered)", total_stations)
    c2.metric("Stations Meeting Rule", num_meets)
    c3.metric("Stations NOT meeting Rule", num_fails)
    c4.metric("Compliance Rate", f"{percent_meets:.1f}%")

    bullets = []
    if median_distance is not None:
        bullets.append(f"- Median nearest Bluebikes distance: **{median_distance:.1f} m**")
    if mean_distance is not None:
        bullets.append(f"- Mean nearest Bluebikes distance: **{mean_distance:.1f} m**")
    if p90_distance is not None:
        bullets.append(f"- 90th percentile of distance: **{p90_distance:.1f} m**")
    bullets.append(f"- Availability rule: **{'ON (≥30% active last 14 days)' if use_avail else 'OFF'}**")
    bullets.append(f"- Radius rule: **≥1 station within {miles:.2f} mi (~{meters:.0f} m)**")
    st.markdown("\n".join(bullets))

    st.markdown("---")

    if "nearest_distance_m" in df.columns and df["nearest_distance_m"].notna().any():
        c5, c6 = st.columns(2)
        with c5:
            st.subheader("Nearest 10 (m)")
            st.dataframe(
                df.sort_values("nearest_distance_m", na_position="last")
                  .loc[:, ["station_id", "station_name", "nearest_distance_m"]]
                  .head(10)
                  .rename(columns={"nearest_distance_m": "nearest_m"}),
                use_container_width=True
            )
        with c6:
            st.subheader("Farthest 10 (m)")
            st.dataframe(
                df.sort_values("nearest_distance_m", ascending=False, na_position="last")
                  .loc[:, ["station_id", "station_name", "nearest_distance_m"]]
                  .head(10)
                  .rename(columns={"nearest_distance_m": "nearest_m"}),
                use_container_width=True
            )

    st.stop()

# ----------------------- PAGE: Overview -----------------------
elif page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Stations meeting rule", len(meets))
    c2.metric("Stations NOT meeting rule", len(fails))
    c3.metric("Median nearest distance (m)",
              f"{df['nearest_distance_m'].median():.0f}" if df['nearest_distance_m'].notna().any() else "—")

    st.subheader("✅ Stations meeting the last-mile rule")
    st.dataframe(
        meets[["station_id", "station_name", "nearby_bluebikes_count", "nearest_distance_m"]]
          .rename(columns={"nearest_distance_m": "nearest_m"}),
        use_container_width=True
    )

    st.subheader("❌ Stations NOT meeting the last-mile rule")
    st.dataframe(
        fails[["station_id", "station_name"]],
        use_container_width=True
    )

    render_map(meets, [0, 180, 0, 200], "🗺️ Map — Stations meeting the rule")
    render_map(fails, [200, 0, 0, 220], "🗺️ Map — Stations NOT meeting the rule")

# ----------------------- PAGE: Recommendations -----------------------
elif page == "Recommendations":
    st.header("💡 Recommendations")

    st.markdown("#### Summary Snapshot")
    st.markdown(f"""
- Failing MBTA stations: **{len(fails)}**
- Passing MBTA stations: **{len(meets)}**
- Categories (heuristic):
  - Outer city centers: **{cat_counts.get('outer_city_center', 0)}**
  - Seasonal / tourism: **{cat_counts.get('seasonal_or_tourism', 0)}**
  - Far-end / Park & Ride: **{cat_counts.get('far_end_or_park_and_ride', 0)}**
  - Other: **{cat_counts.get('other', 0)}**
""")

    st.markdown("#### Recommended Actions")
    st.markdown("""
- **Short-term (0–6 months)**: Pilot outer city centers with 1–2 docks near MBTA stations; add a "Commuter Rail + Bluebikes" discount.
- **Medium-term (6–24 months)**: For Far-end / Park & Ride stations, run Park-and-Bike pilots and measure uptake before scaling.
- **Long-term (24+ months)**: Seasonal docks in tourism corridors (Apr–Oct); expand multi-modal options (e-scooters, carshare).
""")

    st.markdown("#### Map of Non-Compliant MBTA Stations")
    render_map(fails, [200, 0, 0, 220], "🗺️ MBTA Stations Not Meeting Last-Mile Rule")

    with st.expander("See list of failing stations"):
        st.dataframe(fails[["station_id", "station_name"]], use_container_width=True)

# ----------------------- PAGE: Availability -----------------------
elif page == "Availability":
    st.header("🔎 Availability — Nearest Bluebikes for a Selected MBTA Station")

    with st.form("availability_search_form", clear_on_submit=False):
        station_keyword = st.text_input(
            "Enter MBTA station keyword (e.g., 'Quincy', 'Forest Hills')",
            key="station_kw",
            help="Type a keyword and press Enter or click Search",
        )
        submitted = st.form_submit_button("Search")

    if "last_keyword" not in st.session_state:
        st.session_state["last_keyword"] = ""
    if "has_result" not in st.session_state:
        st.session_state["has_result"] = False

    if station_keyword != st.session_state["last_keyword"]:
        st.session_state["has_result"] = False

    if submitted:
        if not station_keyword.strip():
            st.warning("Please enter a station keyword first.")
            st.stop()
        st.session_state["last_keyword"] = station_keyword
        st.session_state["has_result"] = True

    if st.session_state.get("has_result"):
        with st.spinner("Searching MBTA stations matching your keyword…"):
            search_sql = f"""
                SELECT station_id, station_name
                FROM `{T_MBTA_STATIONS}`
                WHERE LOWER(station_name) LIKE LOWER(CONCAT('%', @kw, '%'))
                ORDER BY station_name
            """
            stations_df = run_query(
                search_sql,
                [bigquery.ScalarQueryParameter("kw", "STRING", st.session_state["last_keyword"])],
                cache_key=f"mbta_search::{st.session_state['last_keyword']}"
            )

        if stations_df.empty:
            st.warning("No matching MBTA stations found. Try a different keyword.")
            st.stop()

        chosen_id = str(stations_df.iloc[0]["station_id"])

        with st.spinner("Finding nearest Bluebikes station and current status…"):
            qparams = [bigquery.ScalarQueryParameter("mbta_station_id", "STRING", chosen_id)]
            nearest_df = run_query(SQL_NEAREST_AVAIL, qparams, cache_key=f"nearest::{chosen_id}")

        if nearest_df.empty:
            st.info("No nearby Bluebikes station found.")
            st.stop()

        row = nearest_df.iloc[0]

        st.markdown(f"### 🚉 MBTA Station: **{row['mbta_station_name']}**")
        st.markdown(f"### 🚲 Nearest Bluebikes Station: **{row['bb_name']}**")
        st.markdown("#### Current Bluebikes Availability and Status")

        col1, col2, col3 = st.columns([1.2, 1.8, 1.2])
        col1.metric("Distance (m)", f"{row['distance_m']:.0f}")
        col2.markdown(
            f"**Status Time (Boston)**<br>"
            f"<span style='font-size:22px;'>{row.get('status_snapshot_ts','')}</span>",
            unsafe_allow_html=True
        )
        renting_val = str(row.get("is_renting"))
        returning_val = str(row.get("is_returning"))
        renting_ok = (renting_val in ("1", "True", "true"))
        returning_ok = (returning_val in ("1", "True", "true"))
        col3.markdown(
            f"<b>Is Renting:</b> {'✅' if renting_ok else '❌'}<br>"
            f"<b>Is Returning:</b> {'✅' if returning_ok else '❌'}",
            unsafe_allow_html=True
        )

        c5, c6 = st.columns([1, 1])
        c5.metric("Bikes Available", int(row.get("num_bikes_available") or 0))
        c6.metric("Docks Available", int(row.get("num_docks_available") or 0))

        mdf = pd.DataFrame([
            {"name": f"MBTA — {row['mbta_station_name']}", "lat": row["mbta_lat"], "lon": row["mbta_lng"], "color": [0, 102, 204, 220]},
            {"name": f"Bluebikes — {row['bb_name']}", "lat": row["bb_lat"], "lon": row["bb_lng"], "color": [0, 200, 100, 220]},
        ])
        line_df = pd.DataFrame([{
            "from_lon": row["mbta_lng"], "from_lat": row["mbta_lat"],
            "to_lon": row["bb_lng"], "to_lat": row["bb_lat"]
        }])

        view = pdk.ViewState(latitude=mdf["lat"].mean(), longitude=mdf["lon"].mean(), zoom=12)

        layer_points = pdk.Layer(
            "ScatterplotLayer",
            data=mdf,
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=70,
            pickable=True
        )
        layer_line = pdk.Layer(
            "LineLayer",
            data=line_df,
            get_source_position='[from_lon, from_lat]',
            get_target_position='[to_lon, to_lat]',
            get_width=4,
            get_color=[255, 140, 0]
        )

        st.pydeck_chart(pdk.Deck(
            initial_view_state=view,
            layers=[layer_points, layer_line],
            tooltip={"text": "{name}"}
        ))

# ----------------------- PAGE: ML_BB -----------------------
elif page == "Bluebikes Demand Prediction":
    st.header("🤖 Predict demand patterns (pickups / dropoffs)")

    # ---- Top predicted pickups ----
    SQL_TOP_PICKS = f"""
      SELECT
        CAST(station_id AS STRING) AS station_id,
        ANY_VALUE(station_name) AS station_name,
        SUM(predicted_pickups) AS pred_pickups
      FROM `{T_PRED_PICKUPS}`
      GROUP BY station_id
      ORDER BY pred_pickups DESC
      LIMIT 15
    """

    # ---- Top predicted dropoffs ----
    SQL_TOP_DROPS = f"""
      SELECT
        CAST(station_id AS STRING) AS station_id,
        ANY_VALUE(station_name) AS station_name,
        SUM(predicted_dropoffs) AS pred_dropoffs
      FROM `{T_PRED_DROPS}`
      GROUP BY station_id
      ORDER BY pred_dropoffs DESC
      LIMIT 15
    """

    # ---- Stations ranked in both lists ----
    SQL_COMMON = f"""
      WITH pick AS (
        SELECT
          CAST(station_id AS STRING) AS station_id,
          ANY_VALUE(station_name) AS station_name,
          SUM(predicted_pickups) AS total_pick
        FROM `{T_PRED_PICKUPS}`
        GROUP BY station_id
        ORDER BY total_pick DESC
        LIMIT 30
      ),
      drop AS (
        SELECT
          CAST(station_id AS STRING) AS station_id,
          ANY_VALUE(station_name) AS station_name,
          SUM(predicted_dropoffs) AS total_drop
        FROM `{T_PRED_DROPS}`
        GROUP BY station_id
        ORDER BY total_drop DESC
        LIMIT 30
      )
      SELECT
        p.station_id,
        COALESCE(p.station_name, d.station_name) AS station_name,
        p.total_pick,
        d.total_drop,
        (IFNULL(p.total_pick,0) + IFNULL(d.total_drop,0)) AS combined_score
      FROM pick p
      JOIN drop d USING (station_id)
      ORDER BY combined_score DESC
      LIMIT 20
    """

    # ---- Station choices for heatmap ----
   # ---- Station choices for heatmap (FIXED) ----
    SQL_STATION_CHOICES = f"""
        WITH unioned AS (
            SELECT CAST(station_id AS STRING) AS station_id, ANY_VALUE(station_name) AS station_name
            FROM `{T_PRED_PICKUPS}`
            GROUP BY station_id
            UNION ALL
            SELECT CAST(station_id AS STRING) AS station_id, ANY_VALUE(station_name) AS station_name
            FROM `{T_PRED_DROPS}`
            GROUP BY station_id
        )
        SELECT station_id, ANY_VALUE(station_name) AS station_name
        FROM unioned
        GROUP BY station_id
        ORDER BY station_name
    """


    with st.spinner("Loading predictions…"):
        df_top_picks = run_query(SQL_TOP_PICKS, cache_key="mlbb::top_picks_fixed")
        df_top_drops = run_query(SQL_TOP_DROPS, cache_key="mlbb::top_drops_fixed")
        df_common    = run_query(SQL_COMMON,    cache_key="mlbb::common_fixed")

    # ---- Display ----
    c1, c2 = st.columns(2, gap="large")

    # PICKUPS
    with c1:
        st.subheader("🏆 Top predicted pickup stations")
        if df_top_picks.empty:
            st.info("No prediction data.")
        else:
            st.dataframe(df_top_picks, use_container_width=True)
            st.altair_chart(
                alt.Chart(df_top_picks).mark_bar().encode(
                    x=alt.X("pred_pickups:Q", title="Predicted Pickups (sum)"),
                    y=alt.Y("station_name:N", sort="-x", title=None)
                ).properties(height=360),
                use_container_width=True
            )

    # DROPOFFS
    with c2:
        st.subheader("🏁 Top predicted dropoff stations")
        if df_top_drops.empty:
            st.info("No prediction data.")
        else:
            st.dataframe(df_top_drops, use_container_width=True)
            st.altair_chart(
                alt.Chart(df_top_drops).mark_bar().encode(
                    x=alt.X("pred_dropoffs:Q", title="Predicted Dropoffs (sum)"),
                    y=alt.Y("station_name:N", sort="-x", title=None)
                ).properties(height=360),
                use_container_width=True
            )

    # COMMON STATIONS
    st.subheader("🔗 Stations ranked in both pickup & dropoff lists")
    if df_common.empty:
        st.info("No overlap between top pickup and top dropoff.")
    else:
        st.dataframe(
            df_common[["station_name", "total_pick", "total_drop", "combined_score"]],
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("🗺️ Select a station to view hourly × weekday prediction heatmap")

    station_choices = run_query(SQL_STATION_CHOICES, cache_key="mlbb::choices_fixed")
    if station_choices.empty:
        st.info("No stations found in prediction tables.")
        st.stop()

    chosen = st.selectbox("Choose station", options=station_choices["station_name"].tolist())
    chosen_id = station_choices.loc[station_choices["station_name"] == chosen, "station_id"].iloc[0]

    # ---- Hour × weekday prediction grid ----
    SQL_HEAT = f"""
      SELECT day_of_week, hour, SUM(predicted_pickups) AS pred_pickups, 0 AS pred_dropoffs
      FROM `{T_PRED_PICKUPS}` WHERE CAST(station_id AS STRING) = @sid
      GROUP BY day_of_week, hour
      UNION ALL
      SELECT day_of_week, hour, 0, SUM(predicted_dropoffs)
      FROM `{T_PRED_DROPS}` WHERE CAST(station_id AS STRING) = @sid
      GROUP BY day_of_week, hour
    """

    heat_df = run_query(
        SQL_HEAT,
        [bigquery.ScalarQueryParameter("sid", "STRING", chosen_id)],
        cache_key=f"mlbb::heat::{chosen_id}"
    )
    if heat_df.empty:
        st.info("No prediction grid for this station.")
        st.stop()

    # ---- Heatmap ----
    dt_long = heat_df.melt(
        id_vars=["day_of_week", "hour"],
        value_vars=["pred_pickups", "pred_dropoffs"],
        var_name="type",
        value_name="value"
    )
    day_map = {1:"Sun",2:"Mon",3:"Tue",4:"Wed",5:"Thu",6:"Fri",7:"Sat"}
    dt_long["weekday"] = dt_long["day_of_week"].map(day_map)

    st.caption("Heatmap of predicted demand by weekday × hour (local time)")
    st.altair_chart(
        alt.Chart(dt_long).mark_rect().encode(
            x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("weekday:O", sort=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"], title="Weekday"),
            color=alt.Color("value:Q", title="Predicted"),
            facet=alt.Facet("type:N", columns=2, title=None)
        ).properties(height=260),
        use_container_width=True
    )
  # ----------------------- PAGE: Occupancy Prediction -----------------------
elif page == "Occupancy Prediction":
    st.header("🚌 MBTA Vehicle Occupancy Prediction (via BigQuery ML)")
    st.caption("Predict real-time train/bus occupancy levels using the deployed Logistic Regression model.")

    # Input fields for filtering
    route_id = st.text_input("Enter Route ID (e.g., Red, 1, 47)", "Red")
    direction_id = st.selectbox("Direction", [0, 1])
    hour_of_day = st.slider("Hour of Day", 0, 23, 8)
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    # SQL for model prediction
    SQL_OCC_PRED = f"""
    SELECT
      predicted_occupancy_status,
      predicted_occupancy_status_probs
    FROM ML.PREDICT(MODEL `{PROJECT_ID}.mbta_ml.occ_lr_min`,
      (SELECT
          '{route_id}' AS route_id,
          {direction_id} AS direction_id,
          '{day_of_week}' AS day_of_week,
          {hour_of_day} AS hour_of_day
      ))
    """

    if st.button("🔮 Run Prediction"):
        with st.spinner("Querying BigQuery ML model..."):
            try:
                df_pred = run_query(SQL_OCC_PRED, cache_key=f"occ_pred::{route_id}::{direction_id}::{day_of_week}::{hour_of_day}")
                if not df_pred.empty:
                    pred_label = df_pred.iloc[0]["predicted_occupancy_status"]
                    st.success(f"### 🚆 Predicted Occupancy: **{pred_label}**")
                    st.json(df_pred.iloc[0]["predicted_occupancy_status_probs"])
                else:
                    st.warning("No prediction result returned from BigQuery ML model.")
            except Exception as e:
                st.error(f"Error running ML prediction: {e}")


