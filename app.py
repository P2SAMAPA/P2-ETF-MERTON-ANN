"""
app.py — P2-ETF-MERTON-ANN
Streamlit dashboard showing current signals, historical performance, and regime visualization.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# HuggingFace
from huggingface_hub import hf_hub_download, HfApi
import os

# Config
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-merton-ann-data")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Page config
st.set_page_config(
    page_title="P2 ETF Merton ANN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .hero-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .regime-risk-on {
        color: #28a745;
        font-weight: bold;
    }
    .regime-risk-off {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_signal(module: str) -> dict:
    """Load current signal from HF."""
    try:
        filename = f"signals/{module}_signal.json"
        print(f"Attempting to load: {filename} from {HF_DATASET_REPO}")

        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            token=HF_TOKEN
        )
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"✓ Successfully loaded {module} signal")
        return data
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            st.error(f"📁 Signal file not found: `signals/{module}_signal.json`\n\n"
                    f"The training pipeline may not have completed yet, or the file wasn't uploaded.\n\n"
                    f"**Debug info:**\n"
                    f"- Repo: `{HF_DATASET_REPO}`\n"
                    f"- Token configured: `{'Yes' if HF_TOKEN else 'No'}`\n"
                    f"- Error: {error_msg[:100]}...")
        else:
            st.error(f"Error loading {module} signal: {e}")
        return {}


@st.cache_data(ttl=3600)
def load_history(module: str) -> pd.DataFrame:
    """Load signal history from HF."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"signals/{module}_history.json",
            token=HF_TOKEN
        )
        with open(path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_data(module: str) -> pd.DataFrame:
    """Load price data from HF."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"data/{module}.parquet",
            token=HF_TOKEN
        )
        return pd.read_parquet(path)
    except:
        return pd.DataFrame()


def render_hero_box(signal: dict, module_name: str):
    """Render the hero box with current signal."""
    etf = signal.get('selected_etf', 'N/A')
    date = signal.get('next_trading_date', 'N/A')
    regime = signal.get('regime', 'N/A')
    exp_return = signal.get('expected_return_annualized', 0)
    horizon = signal.get('horizon_days', 0)

    regime_class = "regime-risk-on" if regime == "risk-on" else "regime-risk-off"

    st.markdown(f"""
    <div class="hero-box">
        <h1>{module_name}</h1>
        <h2>Hold <strong>{etf}</strong></h2>
        <h3>for {date}</h3>
        <p>Regime: <span class="{regime_class}">{regime.upper()}</span> | 
           Horizon: {horizon}d | 
           Expected: {exp_return:.1%} ann.</p>
    </div>
    """, unsafe_allow_html=True)


def render_weights_chart(signal: dict):
    """Render ANN weights bar chart."""
    weights = signal.get('weights', {})
    if not weights:
        return

    df = pd.DataFrame([
        {'ETF': k, 'Weight': v}
        for k, v in weights.items()
    ])
    df = df.sort_values('Weight', ascending=True)

    fig = px.bar(
        df, x='Weight', y='ETF', orientation='h',
        title='ANN Portfolio Weights',
        color='Weight', color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_regime_timeline(history: pd.DataFrame):
    """Render regime timeline."""
    if history.empty or 'regime' not in history.columns:
        return

    history['date'] = pd.to_datetime(history['date'])

    # Create regime indicator (0/1)
    history['regime_num'] = history['regime'].apply(lambda x: 0 if x == 'risk-on' else 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['regime_num'],
        mode='lines',
        fill='tozeroy',
        name='Regime',
        line=dict(color='red'),
        fillcolor='rgba(255,0,0,0.3)'
    ))

    fig.update_layout(
        title='Regime History (0=Risk-On, 1=Risk-Off)',
        yaxis_range=[-0.1, 1.1],
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)


def render_signal_history(history: pd.DataFrame):
    """Render historical ETF selections."""
    if history.empty:
        return

    history['date'] = pd.to_datetime(history['date'])

    # Count ETF selections
    etf_counts = history['selected_etf'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ETF Selection Frequency")
        fig = px.pie(
            values=etf_counts.values,
            names=etf_counts.index,
            title='Historical ETF Selections'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Recent Signals")
        recent = history[['date', 'selected_etf', 'regime', 'expected_return_annualized']].tail(10)
        recent['expected_return_annualized'] = recent['expected_return_annualized'].apply(lambda x: f"{x:.1%}")
        st.dataframe(recent, use_container_width=True)


def render_performance_metrics(history: pd.DataFrame, prices: pd.DataFrame, module: str):
    """Render backtest performance metrics."""
    if history.empty or prices.empty:
        return

    history['date'] = pd.to_datetime(history['date'])

    # Calculate returns (simplified backtest)
    # This is a placeholder - real implementation would compute actual returns
    # from following the signals

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Signals", len(history))

    with col2:
        unique_etfs = history['selected_etf'].nunique()
        st.metric("ETFs Used", unique_etfs)

    with col3:
        avg_horizon = history['horizon_days'].mean() if 'horizon_days' in history.columns else 0
        st.metric("Avg Horizon", f"{avg_horizon:.0f}d")

    with col4:
        latest_regime = history['regime'].iloc[-1] if 'regime' in history.columns else 'N/A'
        st.metric("Current Regime", latest_regime)


def main():
    st.title("📈 P2 ETF Merton ANN")
    st.markdown("*Merton Optimal Portfolio with Semi-Markov Regime Switching and ANN Feedback Control*")

    # Sidebar
    st.sidebar.header("Settings")
    module = st.sidebar.selectbox(
        "Select Module",
        ["Equity", "Fixed Income"],
        index=0
    )

    module_key = "equity" if module == "Equity" else "fixed_income"

    # Load data
    signal = load_signal(module_key)
    history = load_history(module_key)
    prices = load_data(module_key)

    if not signal:
        st.error(f"No signal available for {module}. Please run the training pipeline.")
        return

    # Hero box
    render_hero_box(signal, module)

    # Main content
    tab1, tab2, tab3 = st.tabs(["Current Signal", "Weights", "History & Performance"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Signal Details")
            st.json({
                "Selected ETF": signal.get('selected_etf'),
                "Next Trading Date": signal.get('next_trading_date'),
                "Current Regime": signal.get('regime'),
                "Investment Horizon": f"{signal.get('horizon_days')} days",
                "Calibration Window": signal.get('window_type'),
                "Expected Return (Annualized)": f"{signal.get('expected_return_annualized', 0):.2%}",
                "ANN Parameters": signal.get('n_parameters'),
                "Semi-Markov p_01": f"{signal.get('semi_markov_params', {}).get('p_01', 0):.4f}",
                "Semi-Markov p_10": f"{signal.get('semi_markov_params', {}).get('p_10', 0):.4f}",
            })

        with col2:
            st.subheader("Semi-Markov Parameters")
            sm_params = signal.get('semi_markov_params', {})
            st.json({
                "Mean Duration Risk-On": f"{sm_params.get('mean_duration_on', 0):.0f} days",
                "Mean Duration Risk-Off": f"{sm_params.get('mean_duration_off', 0):.0f} days",
                "Transition Prob (On→Off)": f"{sm_params.get('p_01', 0):.4f}",
                "Transition Prob (Off→On)": f"{sm_params.get('p_10', 0):.4f}",
            })

    with tab2:
        render_weights_chart(signal)

    with tab3:
        st.subheader("Performance Metrics")
        render_performance_metrics(history, prices, module)

        st.subheader("Regime Timeline")
        render_regime_timeline(history)

        st.subheader("Signal History")
        render_signal_history(history)

    # Footer
    st.markdown("---")
    st.markdown(
        "*Research implementation based on: "
        '"Intertemporal Optimal Portfolio Allocation under Regime Switching Using Artificial Neural Networks" '
        "(Carl et al., 2025)*"
    )


if __name__ == "__main__":
    main()
