# config.py — P2-ETF-MERTON-ANN
# Single source of truth for ETF universes, macro series, and HF dataset config.

import os

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-merton-ann-data")
HF_TOKEN        = os.environ.get("HF_TOKEN", "")

# ── FRED ──────────────────────────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ── Equity universe (Option A) ────────────────────────────────────────────────
# Benchmark: SPY (not holdable — benchmark only)
# Regime indicator: ^VIX
EQUITY_BENCHMARK = "SPY"
EQUITY_REGIME    = "^VIX"
EQUITY_ETFS = [
    "QQQ",   # NASDAQ 100
    "IWM",   # Russell 2000 Small Cap
    "IWF",   # Russell 1000 Growth
    "IWD",   # Russell 1000 Value
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XME",   # Metals & Mining
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "GDX",   # Gold Miners
    "EEM",   # Emerging Markets
    "EFA",   # Developed International
]
EQUITY_START = "2007-01-01"

# ── Fixed income + real assets universe (Option B) ────────────────────────────
# Benchmark: AGG (not holdable — benchmark only)
# Regime indicator: ^MOVE
FI_BENCHMARK = "AGG"
FI_REGIME    = "^MOVE"
FI_ETFS = [
    "TLT",   # 20Y+ Treasury
    "IEF",   # 7-10Y Treasury
    "SHY",   # 1-3Y Treasury (near-cash)
    "LQD",   # IG Corporate Bonds
    "HYG",   # High Yield Corporate
    "MBB",   # Mortgage-Backed Securities
    "PFF",   # Preferred Stock
    "GLD",   # Gold
    "SLV",   # Silver
    "VNQ",   # REITs
    "EMB",   # Emerging Market Bonds
]
FI_START = "2007-01-01"

# ── FRED macro series (fetched for both modules) ──────────────────────────────
FRED_SERIES = {
    "DTB3":         "3M T-Bill Rate (risk-free rate)",
    "DGS10":        "10Y Treasury Yield",
    "T10Y2Y":       "10Y-2Y Yield Spread",
    "T10Y3M":       "10Y-3M Yield Spread",
    "BAMLH0A0HYM2": "HY Credit Spread",
    "BAMLC0A0CM":   "IG Credit Spread",
    "DCOILWTICO":   "WTI Crude Oil",
    "DTWEXBGS":     "USD Broad Index",
    "T10YIE":       "10Y Breakeven Inflation",
}
FRED_START = "2005-01-01"   # buffer before earliest ETF start

# ── All tickers to download from yfinance (per module) ───────────────────────
def equity_tickers() -> list:
    return EQUITY_ETFS + [EQUITY_BENCHMARK, EQUITY_REGIME]

def fi_tickers() -> list:
    return FI_ETFS + [FI_BENCHMARK, FI_REGIME]

# ── HF dataset split names ────────────────────────────────────────────────────
SPLIT_EQUITY = "equity"
SPLIT_FI     = "fixed_income"

# ── OHLCV columns stored per ticker ──────────────────────────────────────────
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
