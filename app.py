"""
app.py — P2-ETF-MERTON-ANN
Professional dashboard displaying the latest ETF signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# HuggingFace imports
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

try:
    HF_DATASET_REPO = st.secrets["HF_DATASET_REPO"]
except:
    HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-merton-ann-data")

st.set_page_config(
    page_title="P2 ETF Merton ANN",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_signal(module: str):
    """Load the latest signal for equity or fixed income."""
    try:
        filename = f"signals/{module}_signal.json"
        file_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="/tmp",
            local_dir_use_symlinks=False
        )
        with open(file_path, 'r') as f:
            signal = json.load(f)
        return signal
    except Exception as e:
        st.sidebar.error(f"Error loading {module}: {e}")
        return None

def format_weights(weights: dict, top_n: int = 10) -> pd.DataFrame:
    """Convert weights dict to DataFrame with formatted percentages."""
    df = pd.DataFrame(list(weights.items()), columns=["ETF", "Weight"])
    df = df.sort_values("Weight", ascending=False).reset_index(drop=True)
    df["Weight (%)"] = df["Weight"] * 100
    df["Weight (%)"] = df["Weight (%)"].map("{:.2f}%".format)
    return df.head(top_n)

def add_bar_chart(df, column="Weight", max_weight=1.0):
    """Add a simple horizontal bar chart using Streamlit's progress bar."""
    for _, row in df.iterrows():
        pct = row[column] / max_weight * 100
        st.markdown(f"**{row['ETF']}**")
        st.progress(pct / 100, text=f"{row['Weight (%)']}")

# ----------------------------------------------------------------------
# Sidebar – Debug Info & Last Update
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 System Info")
    st.markdown(f"**Repo:** `{HF_DATASET_REPO}`")
    st.markdown(f"**Token:** {'✅ configured' if HF_TOKEN else '❌ missing'}")
    st.markdown("---")

    # Load signals (once per session)
    @st.cache_resource
    def load_signals():
        return {
            "equity": load_signal("equity"),
            "fi": load_signal("fi")
        }

    signals = load_signals()
    equity_signal = signals["equity"]
    fi_signal = signals["fi"]

    # Last update timestamp (from equity signal if available)
    if equity_signal:
        last_date = equity_signal.get("date", "unknown")
        st.markdown(f"**Last update:** {last_date}")
    else:
        st.markdown("**Last update:** not yet")

    st.markdown("---")
    st.caption("Data from HuggingFace Hub • Model: Merton ANN • Regime: Semi-Markov")

# ----------------------------------------------------------------------
# Main Page – Hero Section
# ----------------------------------------------------------------------
st.title("📈 P2 ETF Merton ANN")
st.markdown(
    "**Merton Optimal Portfolio with Semi-Markov Regime Switching and ANN Feedback Control**  \n"
    "Daily next‑day ETF signals for equity and fixed income universes."
)

if equity_signal:
    next_trading = equity_signal.get("next_trading_date", "N/A")
    st.markdown(f"### 🗓️ **Prediction for:** {next_trading} (US Markets)")
else:
    st.warning("⚠️ No signal data available. Please run the training pipeline first.")

# ----------------------------------------------------------------------
# Equity Module
# ----------------------------------------------------------------------
st.header("📊 Equity Universe")
if equity_signal:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="**Selected ETF**",
            value=equity_signal.get("selected_etf", "N/A"),
            delta=None,
            delta_color="normal"
        )
    with col2:
        exp_return = equity_signal.get("expected_return_annualized", 0) * 100
        st.metric(
            label="**Expected Annual Return**",
            value=f"{exp_return:.1f}%",
            delta=None
        )
    with col3:
        st.metric(
            label="**Regime**",
            value=equity_signal.get("regime", "N/A").replace("-", " ").title()
        )
    with col4:
        st.metric(
            label="**Investment Horizon**",
            value=f"{equity_signal.get('horizon_days', 'N/A')} days"
        )

    # Portfolio weights (top 10)
    weights = equity_signal.get("weights", {})
    if weights:
        st.subheader("Portfolio Weights")
        df_weights = format_weights(weights, top_n=10)
        # Show as a bar chart using columns for better presentation
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.dataframe(df_weights[["ETF", "Weight (%)"]], use_container_width=True)
        with col_right:
            # Horizontal bar chart using progress bars
            for _, row in df_weights.iterrows():
                pct = row["Weight"] * 100
                st.markdown(f"**{row['ETF']}**")
                st.progress(pct / 100, text=f"{row['Weight (%)']}")
    else:
        st.info("No weights available.")

    # Details expander
    with st.expander("🔍 Model Details & Parameters"):
        st.markdown("#### Calibration & Regime Information")
        st.json({
            "selected_etf": equity_signal.get("selected_etf"),
            "expected_return_annualized": equity_signal.get("expected_return_annualized"),
            "horizon_days": equity_signal.get("horizon_days"),
            "window_type": equity_signal.get("window_type"),
            "regime_threshold": equity_signal.get("regime_threshold"),
            "semi_markov_params": equity_signal.get("semi_markov_params", {})
        })
        st.markdown("#### Full Weights (All Assets)")
        all_weights = equity_signal.get("weights", {})
        if all_weights:
            df_all = pd.DataFrame(list(all_weights.items()), columns=["ETF", "Weight"])
            df_all = df_all.sort_values("Weight", ascending=False).reset_index(drop=True)
            df_all["Weight (%)"] = df_all["Weight"] * 100
            df_all["Weight (%)"] = df_all["Weight (%)"].map("{:.2f}%".format)
            st.dataframe(df_all[["ETF", "Weight (%)"]], use_container_width=True)

else:
    st.error("📁 Signal file not found: `signals/equity_signal.json`")
    st.info("Please run the training pipeline to generate signals.")

# ----------------------------------------------------------------------
# Fixed Income Module
# ----------------------------------------------------------------------
st.header("📊 Fixed Income & Real Assets")
if fi_signal:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="**Selected ETF**",
            value=fi_signal.get("selected_etf", "N/A")
        )
    with col2:
        exp_return = fi_signal.get("expected_return_annualized", 0) * 100
        st.metric(
            label="**Expected Annual Return**",
            value=f"{exp_return:.1f}%"
        )
    with col3:
        st.metric(
            label="**Regime**",
            value=fi_signal.get("regime", "N/A").replace("-", " ").title()
        )
    with col4:
        st.metric(
            label="**Investment Horizon**",
            value=f"{fi_signal.get('horizon_days', 'N/A')} days"
        )

    weights = fi_signal.get("weights", {})
    if weights:
        st.subheader("Portfolio Weights")
        df_weights = format_weights(weights, top_n=10)
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.dataframe(df_weights[["ETF", "Weight (%)"]], use_container_width=True)
        with col_right:
            for _, row in df_weights.iterrows():
                pct = row["Weight"] * 100
                st.markdown(f"**{row['ETF']}**")
                st.progress(pct / 100, text=f"{row['Weight (%)']}")

    with st.expander("🔍 Model Details & Parameters"):
        st.markdown("#### Calibration & Regime Information")
        st.json({
            "selected_etf": fi_signal.get("selected_etf"),
            "expected_return_annualized": fi_signal.get("expected_return_annualized"),
            "horizon_days": fi_signal.get("horizon_days"),
            "window_type": fi_signal.get("window_type"),
            "regime_threshold": fi_signal.get("regime_threshold"),
            "semi_markov_params": fi_signal.get("semi_markov_params", {})
        })
        st.markdown("#### Full Weights (All Assets)")
        all_weights = fi_signal.get("weights", {})
        if all_weights:
            df_all = pd.DataFrame(list(all_weights.items()), columns=["ETF", "Weight"])
            df_all = df_all.sort_values("Weight", ascending=False).reset_index(drop=True)
            df_all["Weight (%)"] = df_all["Weight"] * 100
            df_all["Weight (%)"] = df_all["Weight (%)"].map("{:.2f}%".format)
            st.dataframe(df_all[["ETF", "Weight (%)"]], use_container_width=True)

else:
    st.error("📁 Signal file not found: `signals/fi_signal.json`")
    st.info("Please run the training pipeline to generate signals.")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"**Methodology:** Merton intertemporal portfolio with semi-Markov regime switching and an ANN feedback controller.\n\n"
    f"**Data source:** Yahoo Finance (ETF prices) & FRED (macro indicators).\n\n"
    f"**Disclaimer:** This is a research implementation and not financial advice. Past performance does not guarantee future results."
)
