"""
app.py — P2-ETF-MERTON-ANN
Streamlit dashboard displaying the latest ETF signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# HuggingFace imports
from huggingface_hub import hf_hub_download

# Try to get token from secrets or environment
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
    page_icon="📈",
    layout="wide"
)

st.title("📈 P2 ETF Merton ANN")
st.markdown("Merton Optimal Portfolio with Semi-Markov Regime Switching and ANN Feedback Control")

# Debug info
st.sidebar.header("Debug Info")
st.sidebar.write(f"Repo: `{HF_DATASET_REPO}`")
st.sidebar.write(f"Token configured: {'Yes' if HF_TOKEN else 'No'}")

# Load signal function
def load_signal(module: str):
    """Load the latest signal for equity or fixed income."""
    try:
        filename = f"signals/{module}_signal.json"
        st.sidebar.write(f"Attempting to load: {filename}")
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

# Load both signals
equity_signal = load_signal("equity")
fi_signal = load_signal("fi")

# Display equity
st.header("Equity Module")
if equity_signal is not None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected ETF", equity_signal.get("selected_etf", "N/A"))
    with col2:
        exp_return = equity_signal.get("expected_return_annualized", 0) * 100
        st.metric("Exp. Annual Return", f"{exp_return:.1f}%")
    with col3:
        st.metric("Regime", equity_signal.get("regime", "N/A"))
    with col4:
        horizon = equity_signal.get("horizon_days", "N/A")
        st.metric("Horizon (days)", horizon)

    # Show full weights as a table
    weights = equity_signal.get("weights", {})
    if weights:
        st.subheader("Portfolio Weights (Top 10)")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(sorted_weights[:10], columns=["ETF", "Weight"])
        df["Weight"] = df["Weight"] * 100
        st.dataframe(df, use_container_width=True)

    # Show metadata
    with st.expander("Details"):
        st.json(equity_signal)
else:
    st.error("📁 Signal file not found: `signals/equity_signal.json`")
    st.info("The training pipeline may not have completed yet, or the file wasn't uploaded.\n\n"
            "Make sure the pipeline has run at least once and that the HuggingFace token has write permissions.")

# Display fixed income
st.header("Fixed Income Module")
if fi_signal is not None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected ETF", fi_signal.get("selected_etf", "N/A"))
    with col2:
        exp_return = fi_signal.get("expected_return_annualized", 0) * 100
        st.metric("Exp. Annual Return", f"{exp_return:.1f}%")
    with col3:
        st.metric("Regime", fi_signal.get("regime", "N/A"))
    with col4:
        horizon = fi_signal.get("horizon_days", "N/A")
        st.metric("Horizon (days)", horizon)

    weights = fi_signal.get("weights", {})
    if weights:
        st.subheader("Portfolio Weights (Top 10)")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(sorted_weights[:10], columns=["ETF", "Weight"])
        df["Weight"] = df["Weight"] * 100
        st.dataframe(df, use_container_width=True)

    with st.expander("Details"):
        st.json(fi_signal)
else:
    st.error("📁 Signal file not found: `signals/fi_signal.json`")

# Show last update time
st.markdown("---")
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
