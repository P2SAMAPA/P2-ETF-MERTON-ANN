"""
app.py — P2-ETF-MERTON-ANN
Professional dashboard with option selector and history.
"""

import streamlit as st
import pandas as pd
import json
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

def format_weights(weights, top_n=10):
    df = pd.DataFrame(list(weights.items()), columns=["ETF", "Weight"])
    df = df.sort_values("Weight", ascending=False).reset_index(drop=True)
    df["Weight (%)"] = df["Weight"] * 100
    df["Weight (%)"] = df["Weight (%)"].map("{:.2f}%".format)
    return df.head(top_n)

def display_large_table(df):
    html = df.to_html(escape=False, index=False)
    st.markdown(f"<div style='font-size: 18px'>{html}</div>", unsafe_allow_html=True)

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
    # Equity module
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
            df_weights = format_weights(weights)
            display_large_table(df_weights[["ETF", "Weight (%)"]])

        with st.expander("🔍 Model Details & Parameters"):
            st.markdown("#### Calibration Parameters")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Window Type", equity_signal.get("window_type", "N/A").replace("_", " ").title())
                st.metric("Regime Threshold", f"{equity_signal.get('regime_threshold', 0):.2f}")
            with col2:
                st.metric("Ensemble Models Used", equity_signal.get("ensemble_models_used", 1))
                st.metric("ANN Parameters", equity_signal.get("n_parameters", "N/A"))

            st.markdown("#### Semi‑Markov Parameters")
            sm = equity_signal.get("semi_markov_params", {})
            if sm:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("p_01", f"{sm.get('p_01',0):.4f}")
                    st.metric("Mean Risk‑on Duration", f"{sm.get('mean_duration_on',0):.0f} days")
                with col2:
                    st.metric("p_10", f"{sm.get('p_10',0):.4f}")
                    st.metric("Mean Risk‑off Duration", f"{sm.get('mean_duration_off',0):.0f} days")

                on_durs = sm.get("risk_on_durations", [])
                off_durs = sm.get("risk_off_durations", [])
                if on_durs or off_durs:
                    with st.expander("Regime Duration History"):
                        if on_durs:
                            st.markdown(f"**Risk‑on durations — {len(on_durs)} periods**")
                            st.markdown("\n".join([f"- {d}" for d in on_durs[:10]]))
                        if off_durs:
                            st.markdown(f"**Risk‑off durations — {len(off_durs)} periods**")
                            st.markdown("\n".join([f"- {d}" for d in off_durs[:10]]))

            st.markdown("#### Full Weight Vector")
            all_weights = equity_signal.get("weights", {})
            if all_weights:
                df_all = pd.DataFrame(list(all_weights.items()), columns=["ETF", "Weight"])
                df_all = df_all.sort_values("Weight", ascending=False).reset_index(drop=True)
                df_all["Weight (%)"] = df_all["Weight"] * 100
                df_all["Weight (%)"] = df_all["Weight (%)"].map("{:.2f}%".format)
                display_large_table(df_all[["ETF", "Weight (%)"]])

    else:
        st.error("No equity signal available for this option.")

    # Fixed Income module
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
            df_weights = format_weights(weights)
            display_large_table(df_weights[["ETF", "Weight (%)"]])

        with st.expander("🔍 Model Details & Parameters"):
            # same as equity expander, but using fi_signal
            st.markdown("#### Calibration Parameters")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Window Type", fi_signal.get("window_type", "N/A").replace("_", " ").title())
                st.metric("Regime Threshold", f"{fi_signal.get('regime_threshold', 0):.2f}")
            with col2:
                st.metric("Ensemble Models Used", fi_signal.get("ensemble_models_used", 1))
                st.metric("ANN Parameters", fi_signal.get("n_parameters", "N/A"))

            sm = fi_signal.get("semi_markov_params", {})
            if sm:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("p_01", f"{sm.get('p_01',0):.4f}")
                    st.metric("Mean Risk‑on Duration", f"{sm.get('mean_duration_on',0):.0f} days")
                with col2:
                    st.metric("p_10", f"{sm.get('p_10',0):.4f}")
                    st.metric("Mean Risk‑off Duration", f"{sm.get('mean_duration_off',0):.0f} days")

                on_durs = sm.get("risk_on_durations", [])
                off_durs = sm.get("risk_off_durations", [])
                if on_durs or off_durs:
                    with st.expander("Regime Duration History"):
                        if on_durs:
                            st.markdown(f"**Risk‑on durations — {len(on_durs)} periods**")
                            st.markdown("\n".join([f"- {d}" for d in on_durs[:10]]))
                        if off_durs:
                            st.markdown(f"**Risk‑off durations — {len(off_durs)} periods**")
                            st.markdown("\n".join([f"- {d}" for d in off_durs[:10]]))

            st.markdown("#### Full Weight Vector")
            all_weights = fi_signal.get("weights", {})
            if all_weights:
                df_all = pd.DataFrame(list(all_weights.items()), columns=["ETF", "Weight"])
                df_all = df_all.sort_values("Weight", ascending=False).reset_index(drop=True)
                df_all["Weight (%)"] = df_all["Weight"] * 100
                df_all["Weight (%)"] = df_all["Weight (%)"].map("{:.2f}%".format)
                display_large_table(df_all[["ETF", "Weight (%)"]])

    else:
        st.error("No fixed income signal available for this option.")

with tab2:
    st.header("Historical Signals")
    history_equity = load_history("equity", option)
    history_fi = load_history("fi", option)

    if history_equity:
        st.subheader("Equity Module History")
        df_eq = pd.DataFrame(history_equity)
        df_eq = df_eq[["date", "selected_etf", "expected_return_annualized", "regime", "horizon_days", "window_type"]]
        df_eq["expected_return_annualized"] = df_eq["expected_return_annualized"] * 100
        df_eq = df_eq.rename(columns={
            "date": "Date",
            "selected_etf": "ETF",
            "expected_return_annualized": "Exp. Return (%)",
            "regime": "Regime",
            "horizon_days": "Horizon (days)",
            "window_type": "Window"
        })
        st.dataframe(df_eq, use_container_width=True)
    else:
        st.info("No history for equity module yet.")

    if history_fi:
        st.subheader("Fixed Income Module History")
        df_fi = pd.DataFrame(history_fi)
        df_fi = df_fi[["date", "selected_etf", "expected_return_annualized", "regime", "horizon_days", "window_type"]]
        df_fi["expected_return_annualized"] = df_fi["expected_return_annualized"] * 100
        df_fi = df_fi.rename(columns={
            "date": "Date",
            "selected_etf": "ETF",
            "expected_return_annualized": "Exp. Return (%)",
            "regime": "Regime",
            "horizon_days": "Horizon (days)",
            "window_type": "Window"
        })
        st.dataframe(df_fi, use_container_width=True)
    else:
        st.info("No history for fixed income module yet.")

# Footer
st.markdown("---")
st.caption(
    "**Methodology:** Merton intertemporal portfolio with semi-Markov regime switching and ANN feedback controller.\n\n"
    "**Data source:** Yahoo Finance (ETF prices) & FRED (macro indicators).\n\n"
    "**Disclaimer:** This is a research implementation and not financial advice."
)
