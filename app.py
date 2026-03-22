"""
app.py — P2-ETF-MERTON-ANN
Professional dashboard with option selector, history tab, and actual returns.
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime
import os
from huggingface_hub import hf_hub_download

# Config
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

try:
    HF_DATASET_REPO = st.secrets["HF_DATASET_REPO"]
except:
    HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-merton-ann-data")

st.set_page_config(page_title="P2 ETF Merton ANN", page_icon="📊", layout="wide")

# Helper functions
@st.cache_data(ttl=3600)  # cache for 1 hour
def load_prices(module: str):
    """Load the latest price data for a module (used to compute actual returns)."""
    try:
        filename = f"data/{module}.parquet"
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="/tmp",
            local_dir_use_symlinks=False
        )
        df = pd.read_parquet(path)
        # Extract close prices for all ETFs (assume column pattern *Close)
        close_cols = [c for c in df.columns if c.endswith("_Close")]
        close_df = df[close_cols].copy()
        close_df.columns = [c.replace("_Close", "") for c in close_df.columns]
        return close_df
    except Exception as e:
        st.warning(f"Could not load price data for {module}: {e}")
        return None

def load_signal(module: str, option: str = "A"):
    suffix = "" if option == "A" else f"_option{option}"
    try:
        filename = f"signals/{module}_signal{suffix}.json"
        path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=filename, repo_type="dataset", token=HF_TOKEN, local_dir="/tmp", local_dir_use_symlinks=False)
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading {module}: {e}")
        return None

def load_history(module: str, option: str = "A"):
    suffix = "" if option == "A" else f"_option{option}"
    try:
        filename = f"signals/{module}_history{suffix}.json"
        path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=filename, repo_type="dataset", token=HF_TOKEN, local_dir="/tmp", local_dir_use_symlinks=False)
        with open(path) as f:
            return json.load(f)
    except:
        return []

def annual_to_daily(annual_return):
    """Convert annualized return to daily return (simple)."""
    if annual_return is None or np.isnan(annual_return):
        return None
    return (1 + annual_return) ** (1/252) - 1

def compute_actual_return(row, price_df):
    """
    Compute actual daily return for a given signal row.
    row: dict with keys 'date', 'next_trading_date', 'selected_etf'
    price_df: DataFrame with index as date, columns = ETF tickers.
    """
    signal_date = pd.to_datetime(row['date'])
    next_date = pd.to_datetime(row['next_trading_date'])
    etf = row['selected_etf']

    if etf not in price_df.columns:
        return None

    # Get close prices
    try:
        close_signal = price_df.loc[signal_date, etf]
        close_next = price_df.loc[next_date, etf]
        return (close_next - close_signal) / close_signal
    except (KeyError, ValueError):
        return None

def format_history_with_returns(history, price_df, module):
    """Add daily expected and actual return columns to history DataFrame."""
    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)
    # Keep only relevant columns
    cols = ['date', 'next_trading_date', 'selected_etf', 'expected_return_annualized', 'regime', 'horizon_days', 'window_type']
    df = df[[c for c in cols if c in df.columns]].copy()

    # Convert expected annual to daily
    df['expected_daily_return'] = df['expected_return_annualized'].apply(annual_to_daily)
    df['expected_daily_return_pct'] = df['expected_daily_return'] * 100

    # Compute actual daily return using price data
    if price_df is not None:
        actual_returns = []
        for _, row in df.iterrows():
            ret = compute_actual_return(row, price_df)
            actual_returns.append(ret)
        df['actual_daily_return'] = actual_returns
        df['actual_daily_return_pct'] = df['actual_daily_return'] * 100
        # Format as string with %
        df['actual_daily_return_pct'] = df['actual_daily_return_pct'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "Pending"
        )
    else:
        df['actual_daily_return_pct'] = "N/A"

    # Format expected daily as percentage
    df['expected_daily_return_pct'] = df['expected_daily_return_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    df['expected_return_annualized'] = df['expected_return_annualized'] * 100
    df['expected_return_annualized'] = df['expected_return_annualized'].map("{:.2f}%".format)

    # Rename columns for display
    df = df.rename(columns={
        'date': 'Date',
        'selected_etf': 'ETF',
        'expected_return_annualized': 'Exp. Annual Return',
        'expected_daily_return_pct': 'Exp. Daily Return',
        'actual_daily_return_pct': 'Actual Daily Return',
        'regime': 'Regime',
        'horizon_days': 'Horizon (days)',
        'window_type': 'Window'
    })
    # Reorder columns
    ordered = ['Date', 'ETF', 'Exp. Annual Return', 'Exp. Daily Return', 'Actual Daily Return', 'Regime', 'Horizon (days)', 'Window']
    return df[ordered]

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 System Info")
    st.markdown(f"**Repo:** `{HF_DATASET_REPO}`")
    st.markdown(f"**Token:** {'✅ configured' if HF_TOKEN else '❌ missing'}")

    option = st.radio("Select Model Option", ["A", "B"], index=0, help="Option A: baseline (no macro). Option B: macro features + ensemble.")
    st.markdown("---")

    # Load signals for selected option
    equity_signal = load_signal("equity", option)
    fi_signal = load_signal("fi", option)

    if equity_signal:
        last_date = equity_signal.get("date", "unknown")
        st.markdown(f"**Last update:** {last_date}")
    else:
        st.markdown("**Last update:** not yet")

    st.markdown("---")
    st.caption("Data from HuggingFace Hub • Model: Merton ANN • Regime: Semi-Markov")

# Main page
st.title("📈 P2 ETF Merton ANN")
st.markdown("**Merton Optimal Portfolio with Semi-Markov Regime Switching and ANN Feedback Control**")
if equity_signal:
    next_trading = equity_signal.get("next_trading_date", "N/A")
    st.markdown(f"### 🗓️ **Prediction for:** {next_trading} (US Markets) — Option {option}")

# Tabs
tab1, tab2 = st.tabs(["Current Signals", "Historical Performance"])

with tab1:
    # Equity module (same as before, unchanged)
    st.header("📊 Equity Universe")
    if equity_signal:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Selected ETF", equity_signal.get("selected_etf", "N/A"))
        with col2: st.metric("Exp. Annual Return", f"{equity_signal.get('expected_return_annualized',0)*100:.1f}%")
        with col3: st.metric("Regime", equity_signal.get("regime", "N/A").replace("-", " ").title())
        with col4: st.metric("Horizon", f"{equity_signal.get('horizon_days', 'N/A')} days")

        weights = equity_signal.get("weights", {})
        if weights:
            st.subheader("Portfolio Weights")
            df_weights = pd.DataFrame(list(weights.items()), columns=["ETF", "Weight"])
            df_weights = df_weights.sort_values("Weight", ascending=False).reset_index(drop=True)
            df_weights["Weight (%)"] = df_weights["Weight"] * 100
            df_weights["Weight (%)"] = df_weights["Weight (%)"].map("{:.2f}%".format)
            # Display as large table
            st.markdown("<div style='font-size: 18px'>" + df_weights[["ETF", "Weight (%)"]].head(10).to_html(escape=False, index=False) + "</div>", unsafe_allow_html=True)

        with st.expander("🔍 Model Details & Parameters"):
            # ... (same as before, keep for brevity)
            st.json(equity_signal.get("semi_markov_params", {}))
    else:
        st.error("No equity signal available for this option.")

    # Fixed Income module (unchanged)
    st.header("📊 Fixed Income & Real Assets")
    if fi_signal:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Selected ETF", fi_signal.get("selected_etf", "N/A"))
        with col2: st.metric("Exp. Annual Return", f"{fi_signal.get('expected_return_annualized',0)*100:.1f}%")
        with col3: st.metric("Regime", fi_signal.get("regime", "N/A").replace("-", " ").title())
        with col4: st.metric("Horizon", f"{fi_signal.get('horizon_days', 'N/A')} days")

        weights = fi_signal.get("weights", {})
        if weights:
            st.subheader("Portfolio Weights")
            df_weights = pd.DataFrame(list(weights.items()), columns=["ETF", "Weight"])
            df_weights = df_weights.sort_values("Weight", ascending=False).reset_index(drop=True)
            df_weights["Weight (%)"] = df_weights["Weight"] * 100
            df_weights["Weight (%)"] = df_weights["Weight (%)"].map("{:.2f}%".format)
            st.markdown("<div style='font-size: 18px'>" + df_weights[["ETF", "Weight (%)"]].head(10).to_html(escape=False, index=False) + "</div>", unsafe_allow_html=True)

        with st.expander("🔍 Model Details & Parameters"):
            st.json(fi_signal.get("semi_markov_params", {}))
    else:
        st.error("No fixed income signal available for this option.")

with tab2:
    st.header("Historical Signals")

    # Load price data for computing actual returns
    price_eq = load_prices("equity")
    price_fi = load_prices("fixed_income")

    # Equity history
    st.subheader("Equity Module")
    hist_eq = load_history("equity", option)
    if hist_eq:
        df_eq = format_history_with_returns(hist_eq, price_eq, "equity")
        st.dataframe(df_eq, use_container_width=True)
    else:
        st.info("No equity history yet.")

    # Fixed income history
    st.subheader("Fixed Income Module")
    hist_fi = load_history("fi", option)
    if hist_fi:
        df_fi = format_history_with_returns(hist_fi, price_fi, "fixed_income")
        st.dataframe(df_fi, use_container_width=True)
    else:
        st.info("No fixed income history yet.")

# Footer
st.markdown("---")
st.caption(
    "**Methodology:** Merton intertemporal portfolio with semi-Markov regime switching and ANN feedback controller.\n\n"
    "**Data source:** Yahoo Finance (ETF prices) & FRED (macro indicators).\n\n"
    "**Disclaimer:** This is a research implementation and not financial advice."
)
