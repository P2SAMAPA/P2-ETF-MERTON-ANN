"""
app.py — P2-ETF-MERTON-ANN
Professional institutional-grade dashboard with modern UI.
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import os
from huggingface_hub import hf_hub_download
import pandas_market_calendars as mcal
import plotly.graph_objects as go
import plotly.express as px

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2 Asset Management | Merton-ANN Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Professional Styling ──────────────────────────────────────
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-navy: #0f172a;
        --accent-gold: #f59e0b;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --bg-card: #ffffff;
        --bg-slate: #f8fafc;
        --border-light: #e2e8f0;
    }
    
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Signal cards */
    .signal-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-light);
        margin-bottom: 1rem;
    }
    
    .signal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-light);
    }
    
    .signal-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-risk-on {
        background: #dcfce7;
        color: #166534;
    }
    
    .badge-risk-off {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Metric grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: var(--bg-slate);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid var(--border-light);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-value.positive {
        color: var(--accent-green);
    }
    
    .metric-value.negative {
        color: var(--accent-red);
    }
    
    /* ETF Selection highlight */
    .etf-selection {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid var(--accent-gold);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .etf-symbol {
        font-size: 3rem;
        font-weight: 800;
        color: var(--primary-navy);
        letter-spacing: -0.02em;
    }
    
    .etf-allocation {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    /* Confidence indicator */
    .confidence-bar {
        width: 100%;
        height: 8px;
        background: var(--border-light);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-gold) 0%, #fbbf24 100%);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--primary-navy) !important;
    }
    
    /* Table styling */
    .styled-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 0.9rem;
    }
    
    .styled-table th {
        background: var(--primary-navy);
        color: white;
        padding: 0.75rem;
        text-align: left;
        font-weight: 600;
    }
    
    .styled-table td {
        padding: 0.75rem;
        border-bottom: 1px solid var(--border-light);
    }
    
    .styled-table tr:hover {
        background: var(--bg-slate);
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: var(--primary-navy);
        color: white;
        border-radius: 12px;
        text-align: center;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    /* Alert boxes */
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        color: #1e40af;
    }
    
    .alert-warning {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# ── Configuration ────────────────────────────────────────────────────────────
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

try:
    HF_DATASET_REPO = st.secrets["HF_DATASET_REPO"]
except:
    HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-merton-ann-data")

# ── NYSE Calendar ─────────────────────────────────────────────────────────────
nyse = mcal.get_calendar("NYSE")

def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Return the next NYSE trading day after the given date."""
    schedule = nyse.schedule(start_date=date, end_date=date + timedelta(days=10))
    trading_days = schedule.index
    next_days = trading_days[trading_days > date]
    if len(next_days) > 0:
        return next_days[0]
    d = date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

# ── Data Loading Functions ───────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_prices(module: str):
    """Load price data for actual returns calculation."""
    try:
        filename = f"data/{module}.parquet"
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="/tmp",
            force_download=True,
        )
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
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
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO, 
            filename=filename, 
            repo_type="dataset", 
            token=HF_TOKEN, 
            local_dir="/tmp"
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return None

def load_history(module: str, option: str = "A"):
    suffix = "" if option == "A" else f"_option{option}"
    try:
        filename = f"signals/{module}_history{suffix}.json"
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO, 
            filename=filename, 
            repo_type="dataset", 
            token=HF_TOKEN, 
            local_dir="/tmp"
        )
        with open(path) as f:
            return json.load(f)
    except:
        return []

# ── Helper Functions ────────────────────────────────────────────────────────
def annual_to_daily(annual_return):
    if annual_return is None or np.isnan(annual_return):
        return None
    return (1 + annual_return) ** (1/252) - 1

def compute_actual_return(row, price_df):
    signal_date = pd.to_datetime(row['date'])
    next_date = pd.to_datetime(row['next_trading_date'])
    
    # Handle multiple ETFs (new format)
    if 'selected_etfs' in row:
        etfs = row['selected_etfs']
        allocation = row.get('allocation', [1.0/len(etfs)] * len(etfs))
        
        total_return = 0
        valid = False
        for etf, alloc in zip(etfs, allocation):
            if etf not in price_df.columns:
                continue
            try:
                close_signal = price_df.loc[signal_date, etf]
                close_next = price_df.loc[next_date, etf]
                etf_return = (close_next - close_signal) / close_signal
                total_return += etf_return * alloc
                valid = True
            except (KeyError, ValueError):
                continue
        return total_return if valid else None
    else:
        # Legacy single ETF format
        etf = row.get('selected_etf')
        if not etf or etf not in price_df.columns:
            return None
        try:
            close_signal = price_df.loc[signal_date, etf]
            close_next = price_df.loc[next_date, etf]
            return (close_next - close_signal) / close_signal
        except (KeyError, ValueError):
            return None

def format_allocation(selected_etfs, allocation):
    """Format ETF allocation for display."""
    if len(selected_etfs) == 1:
        return f"100% {selected_etfs[0]}"
    parts = [f"{int(alloc*100)}% {etf}" for etf, alloc in zip(selected_etfs, allocation)]
    return " + ".join(parts)

def safe_to_float(value):
    """Convert value to float if possible, otherwise return None."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 1px solid #334155; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">⚙️ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Option Selection
    st.markdown("### Model Configuration")
    option = st.radio(
        "Select Strategy Variant",
        ["Option A: Baseline", "Option B: Enhanced (Macro + Ensemble)"],
        index=0,
        help="Option A: Merton-ANN with regime switching. Option B: Adds macro features and model ensemble."
    )
    option_val = "A" if "A" in option else "B"
    
    st.markdown("---")
    
    # System Status
    st.markdown("### System Status")
    if HF_TOKEN:
        st.markdown("🟢 **Data Connection**: Active")
    else:
        st.markdown("🔴 **Data Connection**: No Token")
    
    # Load signal to show last update
    equity_signal_check = load_signal("equity", option_val)
    if equity_signal_check:
        last_date = equity_signal_check.get("date", "unknown")
        st.markdown(f"📅 **Last Signal**: {last_date}")
        
        # Show validation score if available
        val_score = equity_signal_check.get("validation_score", {})
        if val_score and "sharpe_dispersion" in val_score:
            dispersion = val_score.get("sharpe_dispersion", 0)
            if dispersion > 0.2:
                st.markdown(f"✅ **Model Health**: Good (σ={dispersion:.2f})")
            else:
                st.markdown(f"⚠️ **Model Health**: Low Differentiation")
    else:
        st.markdown("⏳ **Status**: Awaiting first run")
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #94a3b8; padding-top: 1rem;">
        <strong>P2 Asset Management</strong><br>
        Quantitative Strategy Division<br><br>
        Data: Yahoo Finance | FRED<br>
        Execution: GitHub Actions
    </div>
    """, unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
price_eq = load_prices("equity")
price_fi = load_prices("fixed_income")

if price_eq is not None and not price_eq.empty:
    last_data_date = price_eq.index[-1]
    computed_next_date = next_trading_day(last_data_date)
    computed_next_date_str = computed_next_date.strftime("%Y-%m-%d")
else:
    computed_next_date_str = "N/A"

equity_signal = load_signal("equity", option_val)
fi_signal = load_signal("fi", option_val)

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>📈 Merton-ANN Strategy Dashboard</h1>
    <p>Institutional-grade optimal portfolio allocation using semi-Markov regime switching and neural network feedback control</p>
    <div style="margin-top: 1.5rem; display: flex; gap: 2rem; font-size: 0.9rem;">
        <div>🎯 <strong>Next Signal:</strong> {computed_next_date_str}</div>
        <div>🔧 <strong>Variant:</strong> Option {option_val}</div>
        <div>📊 <strong>Universes:</strong> Equity (16 ETFs) | Fixed Income (11 ETFs)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_signals, tab_history, tab_analytics = st.tabs(["📊 Current Signals", "📈 Historical Performance", "🔬 Strategy Analytics"])

# ── Tab 1: Current Signals ────────────────────────────────────────────────────
with tab_signals:
    
    # Equity Section
    st.markdown('<div class="signal-card">', unsafe_allow_html=True)
    
    if equity_signal:
        # Header with regime badge
        regime = equity_signal.get("regime", "unknown")
        regime_class = "badge-risk-on" if regime == "risk-on" else "badge-risk-off"
        regime_icon = "🟢" if regime == "risk-on" else "🔴"
        
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.markdown(f"""
            <div class="signal-header">
                <div class="signal-title">📊 Equity Universe <span style="font-size: 0.9rem; color: #64748b; font-weight: 400;">(vs SPY)</span></div>
                <span class="badge {regime_class}">{regime_icon} {regime.replace("-", " ").title()}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # ETF Selection (handles both single and dual ETF format)
        selected_etfs = equity_signal.get("selected_etfs", [equity_signal.get("selected_etf", "N/A")])
        allocation = equity_signal.get("allocation", [1.0])
        
        st.markdown(f"""
        <div class="etf-selection">
            <div class="etf-symbol">{selected_etfs[0]}{f" + {selected_etfs[1]}" if len(selected_etfs) > 1 else ""}</div>
            <div class="etf-allocation">{format_allocation(selected_etfs, allocation)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Grid
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            exp_ret = equity_signal.get('expected_return_annualized', 0) * 100
            color_class = "positive" if exp_ret > 0 else "negative"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Expected Annual Return</div>
                <div class="metric-value {color_class}">{exp_ret:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            horizon = equity_signal.get('horizon_days', 'N/A')
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Investment Horizon</div>
                <div class="metric-value">{horizon} days</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col3:
            window = equity_signal.get('window_type', 'N/A')
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Calibration Window</div>
                <div class="metric-value">{window.replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col4:
            confidence = equity_signal.get('confidence', 0) * 100
            conf_width = min(max(confidence, 10), 100)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Selection Confidence</div>
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf_width}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Weights Distribution Chart
        weights = equity_signal.get("weights", {})
        if weights:
            st.markdown("#### Portfolio Weight Distribution")
            
            # Sort and filter meaningful weights (>1%)
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            meaningful = [(k, v) for k, v in sorted_weights if v > 0.01]
            
            if meaningful:
                df_weights = pd.DataFrame(meaningful, columns=['ETF', 'Weight'])
                df_weights['Weight_Pct'] = df_weights['Weight'] * 100
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_weights['ETF'],
                        y=df_weights['Weight_Pct'],
                        marker_color=['#f59e0b' if etf in selected_etfs else '#64748b' 
                                     for etf in df_weights['ETF']],
                        text=df_weights['Weight_Pct'].apply(lambda x: f'{x:.1f}%'),
                        textposition='outside',
                    )
                ])
                fig.update_layout(
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Weight (%)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(t=20, b=40),
                    yaxis=dict(gridcolor='#e2e8f0', range=[0, max(df_weights['Weight_Pct']) * 1.2])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Model Details Expander
        with st.expander("🔍 Advanced: Model Parameters & Validation"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Semi-Markov Parameters**")
                sm_params = equity_signal.get("semi_markov_params", {})
                st.json({
                    "Mean Duration Risk-On": f"{sm_params.get('mean_duration_on', 0):.0f} days",
                    "Mean Duration Risk-Off": f"{sm_params.get('mean_duration_off', 0):.0f} days",
                    "P(Risk-On → Risk-Off)": f"{sm_params.get('p_01', 0):.4f}",
                    "P(Risk-Off → Risk-On)": f"{sm_params.get('p_10', 0):.4f}",
                })
            
            with col2:
                st.markdown("**Validation Metrics (1Y Rolling)**")
                val_score = equity_signal.get("validation_score", {})
                if val_score and "error" not in val_score:
                    st.json({
                        "Sharpe Dispersion": f"{val_score.get('sharpe_dispersion', 0):.3f}",
                        "Best vs Median Sharpe": f"{val_score.get('best_sharpe', 0):.2f} vs {val_score.get('median_sharpe', 0):.2f}",
                        "Hit Rate vs Benchmark": f"{val_score.get('hit_vs_benchmark', 0)}",
                    })
                else:
                    st.info("Validation data pending")
            
            # Adaptive window info
            if "adaptive_window" in equity_signal:
                st.markdown(f"**Adaptive Regime Window**: {equity_signal['adaptive_window']} days (selected based on recent Sharpe performance)")
            
            if "ensemble_models_used" in equity_signal:
                st.markdown(f"**Ensemble Models**: Top {equity_signal['ensemble_models_used']} models averaged")
    
    else:
        st.error("⚠️ No equity signal available. Please check data pipeline.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fixed Income Section
    st.markdown('<div class="signal-card">', unsafe_allow_html=True)
    
    if fi_signal:
        regime = fi_signal.get("regime", "unknown")
        regime_class = "badge-risk-on" if regime == "risk-on" else "badge-risk-off"
        regime_icon = "🟢" if regime == "risk-on" else "🔴"
        
        st.markdown(f"""
        <div class="signal-header">
            <div class="signal-title">🏛️ Fixed Income & Real Assets <span style="font-size: 0.9rem; color: #64748b; font-weight: 400;">(vs AGG)</span></div>
            <span class="badge {regime_class}">{regime_icon} {regime.replace("-", " ").title()}</span>
        </div>
        """, unsafe_allow_html=True)
        
        selected_etfs_fi = fi_signal.get("selected_etfs", [fi_signal.get("selected_etf", "N/A")])
        allocation_fi = fi_signal.get("allocation", [1.0])
        
        st.markdown(f"""
        <div class="etf-selection" style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); border-color: #6366f1;">
            <div class="etf-symbol" style="color: #312e81;">{selected_etfs_fi[0]}{f" + {selected_etfs_fi[1]}" if len(selected_etfs_fi) > 1 else ""}</div>
            <div class="etf-allocation">{format_allocation(selected_etfs_fi, allocation_fi)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            exp_ret = fi_signal.get('expected_return_annualized', 0) * 100
            color_class = "positive" if exp_ret > 0 else "negative"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Expected Annual Return</div>
                <div class="metric-value {color_class}">{exp_ret:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            horizon = fi_signal.get('horizon_days', 'N/A')
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Investment Horizon</div>
                <div class="metric-value">{horizon} days</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Regime Indicator</div>
                <div class="metric-value">MOVE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col4:
            confidence = fi_signal.get('confidence', 0) * 100
            conf_width = min(max(confidence, 10), 100)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Selection Confidence</div>
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf_width}%; background: linear-gradient(90deg, #6366f1 0%, #818cf8 100%);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # FI Weights Chart
        weights_fi = fi_signal.get("weights", {})
        if weights_fi:
            st.markdown("#### Portfolio Weight Distribution")
            sorted_weights_fi = sorted(weights_fi.items(), key=lambda x: x[1], reverse=True)
            meaningful_fi = [(k, v) for k, v in sorted_weights_fi if v > 0.01]
            
            if meaningful_fi:
                df_weights_fi = pd.DataFrame(meaningful_fi, columns=['ETF', 'Weight'])
                df_weights_fi['Weight_Pct'] = df_weights_fi['Weight'] * 100
                
                fig_fi = go.Figure(data=[
                    go.Bar(
                        x=df_weights_fi['ETF'],
                        y=df_weights_fi['Weight_Pct'],
                        marker_color=['#6366f1' if etf in selected_etfs_fi else '#94a3b8' 
                                      for etf in df_weights_fi['ETF']],
                        text=df_weights_fi['Weight_Pct'].apply(lambda x: f'{x:.1f}%'),
                        textposition='outside',
                    )
                ])
                fig_fi.update_layout(
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Weight (%)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(t=20, b=40),
                    yaxis=dict(gridcolor='#e2e8f0', range=[0, max(df_weights_fi['Weight_Pct']) * 1.2])
                )
                st.plotly_chart(fig_fi, use_container_width=True)
    
    else:
        st.error("⚠️ No fixed income signal available.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 2: Historical Performance ───────────────────────────────────────────
with tab_history:
    st.markdown('<div class="signal-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Historical Signal Performance")
    
    hist_eq = load_history("equity", option_val)
    hist_fi = load_history("fi", option_val)
    
    if hist_eq:
        st.markdown("#### Equity Module History")
        
        # Process history data with safe numeric conversion
        history_data = []
        for row in hist_eq:
            # Compute actual return safely
            actual_ret_raw = compute_actual_return(row, price_eq) if price_eq is not None else None
            numeric_ret = safe_to_float(actual_ret_raw)
            
            history_data.append({
                'Date': row.get('date', 'N/A'),
                'Signal Date': row.get('next_trading_date', 'N/A'),
                'ETF(s)': ", ".join(row.get('selected_etfs', [row.get('selected_etf', 'N/A')])),
                'Allocation': format_allocation(
                    row.get('selected_etfs', [row.get('selected_etf', 'N/A')]),
                    row.get('allocation', [1.0])
                ),
                'Regime': row.get('regime', 'N/A').replace('-', ' ').title(),
                'Expected': f"{row.get('expected_return_annualized', 0)*100:.1f}%" if row.get('expected_return_annualized') else "N/A",
                'Actual': f"{numeric_ret*100:.2f}%" if numeric_ret is not None else "Pending",
                'Hit': "✅" if numeric_ret and numeric_ret > 0 else ("❌" if numeric_ret else "⏳"),
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, height=400)
        
        # Cumulative performance chart
        if price_eq is not None and len(hist_eq) > 5:
            st.markdown("#### Cumulative Strategy Returns")
            
            returns_series = []
            dates = []
            for row in hist_eq:
                ret = compute_actual_return(row, price_eq)
                numeric_ret = safe_to_float(ret)
                if numeric_ret is not None:
                    returns_series.append(numeric_ret)
                    dates.append(pd.to_datetime(row.get('date')))
            
            if returns_series:
                cumulative = (1 + pd.Series(returns_series)).cumprod()
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=dates,
                    y=cumulative,
                    mode='lines+markers',
                    name='Strategy',
                    line=dict(color='#f59e0b', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(245, 158, 11, 0.1)'
                ))
                fig_cum.add_trace(go.Scatter(
                    x=dates,
                    y=[1] * len(dates),
                    mode='lines',
                    name='Break-even',
                    line=dict(color='#64748b', width=1, dash='dash')
                ))
                fig_cum.update_layout(
                    title="Cumulative Performance (Daily Rebalanced)",
                    xaxis_title="Date",
                    yaxis_title="Growth of $1",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.info("No historical data available yet. Signals will appear here after the first run.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 3: Strategy Analytics ─────────────────────────────────────────────────
with tab_analytics:
    st.markdown('<div class="signal-card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Strategy Methodology & Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Model Architecture
        
        **Core Framework**: Merton's Intertemporal Portfolio Problem
        
        **Key Innovations**:
        - **Semi-Markov Regime Switching**: Captures duration-dependent transitions (avg 4.5 years per regime)
        - **ANN Feedback Controller**: Small MLP (5-10 neurons) trained on synthetic GBM paths
        - **Daily Recalibration**: Parameters updated after each market close
        
        **Risk Management**:
        - Momentum overlay (20-day lookback)
        - Relative strength vs benchmark
        - Confidence-weighted position sizing
        """)
    
    with col2:
        st.markdown("""
        #### Current Implementation
        
        **Equity Universe**: 16 ETFs covering US sectors, styles, and international
        
        **Fixed Income Universe**: 11 ETFs across Treasuries, credit, and real assets
        
        **Regime Detection**:
        - Equity: VIX with adaptive window (21/63/252 days)
        - Fixed Income: MOVE index
        
        **Signal Generation**:
        - Winner-takes-all → Top-2 with temperature scaling
        - Ensemble averaging (Option B)
        - Validation: 1-year rolling Sharpe backtest
        """)
    
    st.markdown("---")
    
    # Performance statistics
    if hist_eq and len(hist_eq) > 10:
        st.markdown("#### Live Performance Statistics")
        
        returns_all = []
        for row in hist_eq:
            ret = compute_actual_return(row, price_eq)
            numeric_ret = safe_to_float(ret)
            if numeric_ret is not None:
                returns_all.append(numeric_ret)
        
        if returns_all:
            col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
            
            total_return = (1 + pd.Series(returns_all)).prod() - 1
            ann_return = (1 + total_return) ** (252 / len(returns_all)) - 1
            volatility = pd.Series(returns_all).std() * np.sqrt(252)
            sharpe = ann_return / volatility if volatility > 0 else 0
            win_rate = sum(1 for r in returns_all if r > 0) / len(returns_all)
            max_dd = (pd.Series(returns_all).cumsum() - pd.Series(returns_all).cumsum().cummax()).min()
            
            with col_stat1:
                st.metric("Total Return", f"{total_return*100:.1f}%")
            with col_stat2:
                st.metric("Ann. Return", f"{ann_return*100:.1f}%")
            with col_stat3:
                st.metric("Ann. Volatility", f"{volatility*100:.1f}%")
            with col_stat4:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            with col_stat5:
                st.metric("Win Rate", f"{win_rate*100:.0f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p><strong>P2 Asset Management</strong> | Quantitative Strategy Division</p>
    <p style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">
        Methodology: Merton (1971) intertemporal portfolio with semi-Markov regime switching (Carl et al., 2025) 
        and neural network feedback control.
    </p>
    <p style="font-size: 0.8rem; opacity: 0.6; margin-top: 1rem;">
        Data Sources: Yahoo Finance (ETF prices), FRED (macro indicators, VIX, MOVE)<br>
        Execution: GitHub Actions (daily retraining at 22:00 UTC) | Infrastructure: HuggingFace Hub + Streamlit<br><br>
        <strong>Disclaimer:</strong> This is a research implementation for educational purposes only. 
        Past performance does not guarantee future results. Not investment advice.
    </p>
</div>
""", unsafe_allow_html=True)
