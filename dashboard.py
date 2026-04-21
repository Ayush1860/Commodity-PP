"""
dashboard.py — Streamlit Dashboard for Commodity Price Visualization.

Run with: streamlit run dashboard.py
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import config
from scraper import AgmarknetScraper
from data_processing import clean_pipeline, load_data
from feature_engineering import engineer_features, get_feature_columns
from model import (
    train_model,
    forecast_next_n_days,
    save_model,
    load_model,
    calculate_rmse,
    calculate_mape,
)

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ─────────────────────────── Page Config ─────────────────────────────────
st.set_page_config(
    page_title="Commodity Price Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stMetric .metric-container { background: linear-gradient(135deg, #1e3a5f, #2d5a87); }

    .metric-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; opacity: 0.8; font-weight: 400; }
    .metric-card h1 { margin: 5px 0 0 0; font-size: 1.8rem; font-weight: 700; }
    .metric-card.green { background: linear-gradient(135deg, #0d5e3a, #1a8a5c); }
    .metric-card.blue  { background: linear-gradient(135deg, #1e3a5f, #2d5a87); }
    .metric-card.amber { background: linear-gradient(135deg, #5e3a0d, #8a5c1a); }
    .metric-card.red   { background: linear-gradient(135deg, #5e0d0d, #8a1a1a); }

    .header-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00b09b, #96c93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .header-subtitle {
        text-align: center;
        font-size: 1rem;
        color: #888;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Header ──────────────────────────────────────
st.markdown('<h1 class="header-title">🌾 Commodity Price Prediction Engine</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Powered by XGBoost & Agmarknet Data</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────── Sidebar ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Commodity input
    commodity = st.text_input(
        "Commodity Name",
        value="Wheat",
        help="e.g., Wheat, Tomato, Onion, Rice, Potato"
    )

    state = st.text_input(
        "State",
        value="Madhya Pradesh",
        help="e.g., Madhya Pradesh, Karnataka, Maharashtra"
    )

    st.markdown("---")

    # Date range
    st.subheader("📅 Date Range")
    days_back = st.slider(
        "Historical Days", min_value=30, max_value=730,
        value=365, step=30
    )

    st.markdown("---")

    # Model settings
    st.subheader("🤖 Model Settings")
    forecast_days = st.slider(
        "Forecast Horizon (days)", min_value=1, max_value=30,
        value=7
    )
    tune_model = st.checkbox("Run Optuna Tuning", value=False,
                              help="Optimize hyperparameters (slower but better)")
    n_trials = st.slider(
        "Optuna Trials", min_value=10, max_value=200,
        value=50, disabled=not tune_model
    )

    st.markdown("---")

    # Data source
    st.subheader("📂 Data Source")
    data_source = st.radio(
        "Source", ["Scrape from Agmarknet", "Upload CSV", "Use Existing CSV"],
        index=0
    )

    uploaded_file = None
    existing_csv = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    elif data_source == "Use Existing CSV":
        csv_files = []
        if os.path.exists(config.DATA_DIR):
            csv_files = [f for f in os.listdir(config.DATA_DIR) if f.endswith(".csv")]
        if csv_files:
            existing_csv = st.selectbox("Select CSV", csv_files)
        else:
            st.warning("No CSV files found in data/ directory.")

    st.markdown("---")
    run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)


# ─────────────────────────── Main Content ────────────────────────────────

if run_button:
    # ── 1. Data Loading ──────────────────────────────────────────────────
    raw_df = None

    with st.status("Running prediction pipeline...", expanded=True) as status:

        # Data Extraction
        st.write("📡 **Step 1/5**: Extracting data...")
        if data_source == "Upload CSV" and uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file, parse_dates=["Date"])
            st.write(f"  ✓ Loaded {len(raw_df)} records from uploaded file.")

        elif data_source == "Use Existing CSV" and existing_csv:
            csv_path = os.path.join(config.DATA_DIR, existing_csv)
            raw_df = load_data(csv_path)
            st.write(f"  ✓ Loaded {len(raw_df)} records from {existing_csv}.")

        else:
            try:
                from datetime import datetime, timedelta
                scrape_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                scrape_to = datetime.now().strftime("%Y-%m-%d")
                scraper = AgmarknetScraper()
                raw_df = scraper.fetch_data(
                    commodity_name=commodity,
                    state=state,
                    from_date=scrape_from,
                    to_date=scrape_to,
                )
                if not raw_df.empty:
                    scraper.save_to_csv(raw_df, commodity, state)
                    st.write(f"  ✓ Scraped {len(raw_df)} records from Agmarknet.")
                else:
                    st.error("No data retrieved from Agmarknet. Check your inputs.")
                    st.stop()
            except Exception as e:
                st.error(f"Scraping failed: {e}")
                st.stop()

        if raw_df is None or raw_df.empty:
            st.error("No data available. Please check your inputs.")
            st.stop()

        # Data Cleaning
        st.write("🧹 **Step 2/5**: Cleaning data...")
        clean_df = clean_pipeline(raw_df)
        st.write(f"  ✓ Cleaned: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns.")

        # Feature Engineering
        st.write("⚙️ **Step 3/5**: Engineering features...")
        feature_df = engineer_features(clean_df, forecast_horizon=1)
        feature_cols = get_feature_columns(feature_df)
        st.write(f"  ✓ Created {len(feature_cols)} features.")

        # Model Training
        st.write(f"🤖 **Step 4/5**: Training XGBoost model...")
        model, metrics, importance = train_model(
            feature_df,
            tune=tune_model,
            n_trials=n_trials if tune_model else None,
        )
        st.write(f"  ✓ Model trained. Test RMSE: ₹{metrics['test_rmse']:.2f}")

        # Forecasting
        st.write(f"🔮 **Step 5/5**: Generating {forecast_days}-day forecast...")
        forecast_df = forecast_next_n_days(
            model, clean_df, feature_cols, n_days=forecast_days
        )
        st.write(f"  ✓ Forecast generated!")

        # Save model
        save_model(model, commodity, state, metrics)

        status.update(label="Pipeline complete! ✓", state="complete")

    # Store in session state for persistence
    st.session_state["clean_df"] = clean_df
    st.session_state["feature_df"] = feature_df
    st.session_state["forecast_df"] = forecast_df
    st.session_state["metrics"] = metrics
    st.session_state["importance"] = importance
    st.session_state["commodity"] = commodity
    st.session_state["state"] = state


# ─────────────────────────── Display Results ─────────────────────────────

if "forecast_df" in st.session_state:
    clean_df = st.session_state["clean_df"]
    forecast_df = st.session_state["forecast_df"]
    metrics = st.session_state["metrics"]
    importance = st.session_state["importance"]
    commodity_name = st.session_state.get("commodity", commodity)
    state_name = st.session_state.get("state", state)

    # ── Metrics Row ──────────────────────────────────────────────────────
    st.markdown("### 📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>Test RMSE</h3>
            <h1>₹{metrics['test_rmse']:.2f}</h1>
        </div>""", unsafe_allow_html=True)

    with col2:
        mape_color = "green" if metrics['test_mape'] < 5 else "amber" if metrics['test_mape'] < 10 else "red"
        st.markdown(f"""
        <div class="metric-card {mape_color}">
            <h3>Test MAPE</h3>
            <h1>{metrics['test_mape']:.2f}%</h1>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>Training Samples</h3>
            <h1>{metrics['n_train']:,}</h1>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card blue">
            <h3>Features Used</h3>
            <h1>{metrics['n_features']}</h1>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Price Chart: Historical + Forecast ───────────────────────────────
    st.markdown(f"### 📈 {commodity_name} Price Trend — {state_name}")

    fig = go.Figure()

    # Historical prices
    if "Date" in clean_df.columns and "Modal_Price" in clean_df.columns:
        fig.add_trace(go.Scatter(
            x=clean_df["Date"],
            y=clean_df["Modal_Price"],
            mode="lines",
            name="Modal Price (Historical)",
            line=dict(color="#00b09b", width=2),
            hovertemplate="Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>",
        ))

    if "Min_Price" in clean_df.columns and "Max_Price" in clean_df.columns:
        # Confidence band (Min-Max range)
        fig.add_trace(go.Scatter(
            x=clean_df["Date"],
            y=clean_df["Max_Price"],
            mode="lines",
            name="Max Price",
            line=dict(color="rgba(150,201,61,0.3)", width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=clean_df["Date"],
            y=clean_df["Min_Price"],
            mode="lines",
            name="Price Range (Min-Max)",
            line=dict(color="rgba(150,201,61,0.3)", width=0),
            fill="tonexty",
            fillcolor="rgba(150,201,61,0.1)",
            hoverinfo="skip",
        ))

    # Forecast line
    if not forecast_df.empty:
        # Connect forecast to historical
        last_historical = clean_df.iloc[-1]
        forecast_dates = [last_historical["Date"]] + forecast_df["Date"].tolist()
        forecast_prices = [last_historical["Modal_Price"]] + forecast_df["Predicted_Price"].tolist()

        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_prices,
            mode="lines+markers",
            name=f"Forecast ({len(forecast_df)} days)",
            line=dict(color="#ff6b6b", width=3, dash="dash"),
            marker=dict(size=8, color="#ff6b6b", symbol="diamond"),
            hovertemplate="Date: %{x}<br>Predicted: ₹%{y:.2f}<extra></extra>",
        ))

        # Forecast confidence band (±5% as rough estimate)
        upper = [p * 1.05 for p in forecast_prices]
        lower = [p * 0.95 for p in forecast_prices]
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(255,107,107,0.1)",
            line=dict(color="rgba(255,107,107,0)"),
            name="Confidence Band (±5%)",
            hoverinfo="skip",
        ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Date",
        yaxis_title="Price (₹/Quintal)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Two-Column Layout: Forecast Table + Feature Importance ───────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🔮 Forecast Table")
        display_df = forecast_df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%a, %b %d %Y")
        display_df = display_df.rename(columns={
            "Predicted_Price": "Predicted ₹",
            "Day": "Day Ahead",
        })
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    with col_right:
        st.markdown("### 🏆 Feature Importance (Top 15)")
        top_n = importance.head(15).sort_values("Importance")
        fig_imp = px.bar(
            top_n,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig_imp.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Arrivals Chart ───────────────────────────────────────────────────
    if "Arrivals_Tonnes" in clean_df.columns:
        st.markdown("### 📦 Market Arrivals (Supply Volume)")
        fig_arr = go.Figure()
        fig_arr.add_trace(go.Bar(
            x=clean_df["Date"],
            y=clean_df["Arrivals_Tonnes"],
            name="Arrivals (Tonnes)",
            marker_color="rgba(150,201,61,0.6)",
            hovertemplate="Date: %{x}<br>Arrivals: %{y:.0f} T<extra></extra>",
        ))
        fig_arr.update_layout(
            template="plotly_dark",
            height=300,
            xaxis_title="Date",
            yaxis_title="Arrivals (Tonnes)",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_arr, use_container_width=True)

    # ── Raw Data Explorer ────────────────────────────────────────────────
    with st.expander("📋 View Raw Data"):
        st.dataframe(clean_df, use_container_width=True, height=300)

else:
    # No results yet — show instructions
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px; color: #666;">
        <h2 style="font-size: 3rem; margin-bottom: 10px;">🌾</h2>
        <h3>Welcome to the Commodity Price Prediction Engine</h3>
        <p style="font-size: 1.1rem; max-width: 600px; margin: auto;">
            Configure your commodity, state, and model settings in the sidebar,
            then click <b>🚀 Run Pipeline</b> to scrape data, train the model,
            and generate a price forecast.
        </p>
        <br>
        <p style="font-size: 0.9rem; color: #888;">
            Supported commodities: Wheat, Rice, Tomato, Onion, Potato, Soybean, etc.<br>
            Data source: <a href="https://agmarknet.gov.in" target="_blank">Agmarknet</a> (Govt. of India)
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────── Footer ──────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#555; font-size:0.8rem;">'
    'Commodity Price Prediction Engine • Powered by XGBoost & Agmarknet 2.0 API'
    '</p>',
    unsafe_allow_html=True,
)
