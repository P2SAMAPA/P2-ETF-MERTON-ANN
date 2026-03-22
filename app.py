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

def display_calibration_details(signal: dict, module_name: str):
    """Show calibration and regime details in a clean format."""
    if not signal:
        return

    st.markdown("#### 📐 Calibration Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Selected ETF", signal.get("selected_etf", "N/A"))
        st.metric("Expected Annual Return", f"{signal.get('expected_return_annualized', 0)*100:.1f}%")
        st.metric("Investment Horizon", f"{signal.get('horizon_days', 'N/A')} days")
    with col2:
        st.metric("Window Type", signal.get("window_type", "N/A").replace("_", " ").title())
        st.metric("Regime Threshold", f"{signal.get('regime_threshold', 0):.2f}")
        st.metric("Current Regime", signal.get("regime", "N/A").replace("-", " ").title())

    st.markdown("---")
    st.markdown("#### ⏱️ Semi‑Markov Transition Parameters")
    sm = signal.get("semi_markov_params", {})
    if sm:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk‑on → Risk‑off (p_01)", f"{sm.get('p_01', 0):.4f}")
            st.metric("Mean Duration – Risk‑on", f"{sm.get('mean_duration_on', 0):.0f} days")
        with col2:
            st.metric("Risk‑off → Risk‑on (p_10)", f"{sm.get('p_10', 0):.4f}")
            st.metric("Mean Duration – Risk‑off", f"{sm.get('mean_duration_off', 0):.0f} days")

        on_durs = sm.get("risk_on_durations", [])
        off_durs = sm.get("risk_off_durations", [])

        if on_durs or off_durs:
            with st.expander("📜 Regime Duration History"):
                if on_durs:
                    st.markdown(f"**Risk‑on durations (days) — {len(on_durs)} periods**")
                    bullet_list = "\n".join([f"- {d}" for d in on_durs[:10]])
                    st.markdown(bullet_list)
                    if len(on_durs) > 10:
                        st.caption(f"... and {len(on_durs) - 10} more periods")
                if off_durs:
                    st.markdown(f"**Risk‑off durations (days) — {len(off_durs)} periods**")
                    bullet_list = "\n".join([f"- {d}" for d in off_durs[:10]])
                    st.markdown(bullet_list)
                    if len(off_durs) > 10:
                        st.caption(f"... and {len(off_durs) - 10} more periods")
    else:
        st.info("No semi‑Markov parameters available.")

def display_large_table(df: pd.DataFrame):
    """Display a DataFrame with larger font."""
    # Convert to HTML with inline style for larger font
    html = df.to_html(escape=False, index=False)
    st.markdown(
        f"<div style='font-size: 18px'>{html}</div>",
        unsafe_allow_html=True
    )

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

    weights = equity_signal.get("weights", {})
    if weights:
        st.subheader("Portfolio Weights")
        df_weights = format_weights(weights, top_n=10)
        display_large_table(df_weights[["ETF", "Weight (%)"]])
    else:
        st.info("No weights available.")

    with st.expander("🔍 Model Details & Parameters"):
        display_calibration_details(equity_signal, "equity")
        st.markdown("#### 📊 Full Weight Vector (All Assets)")
        all_weights = equity_signal.get("weights", {})
        if all_weights:
            df_all = pd.DataFrame(list(all_weights.items()), columns=["ETF", "Weight"])
            df_all = df_all.sort_values("Weight", ascending=False).reset_index(drop=True)
            df_all["Weight (%)"] = df_all["Weight"] * 100
            df_all["Weight (%)"] = df_all["Weight (%)"].map("{:.2f}%".format)
            display_large_table(df_all[["ETF", "Weight (%)"]])
        else:
            st.info("No weights available.")

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
        display_large_table(df_weights[["ETF", "Weight (%)"]])
    else:
        st.info("No weights available.")

    with st.expander("🔍 Model Details & Parameters"):
        display_calibration_details(fi_signal, "fi")
        st.markdown("#### 📊 Full Weight Vector (All Assets)")
        all_weights = fi_signal.get("weights", {})
        if all_weights:
            df_all = pd.DataFrame(list(all_weights.items()), columns=["ETF", "Weight"])
            df_all = df_all.sort_values("Weight", ascending=False).reset_index(drop=True)
            df_all["Weight (%)"] = df_all["Weight"] * 100
            df_all["Weight (%)"] = df_all["Weight (%)"].map("{:.2f}%".format)
            display_large_table(df_all[["ETF", "Weight (%)"]])
        else:
            st.info("No weights available.")

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
