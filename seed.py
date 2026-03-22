"""
seed.py — P2-ETF-MERTON-ANN
-----------------------------
One-time script to seed the full historical dataset to HuggingFace.

Fetches:
  - OHLCV for all equity ETFs + SPY benchmark + ^VIX  (from 2006-01-01)
  - OHLCV for all FI ETFs + AGG benchmark + ^MOVE      (from 2008-01-01)
  - FRED macro series for both modules                  (from 2005-01-01)

Pushes three parquet files to HF dataset P2SAMAPA/p2-etf-merton-ann-data:
  - data/equity.parquet
  - data/fixed_income.parquet
  - data/fred_macro.parquet

Run once via GitHub Actions (seed.yml) or locally:
    HF_TOKEN=xxx FRED_API_KEY=xxx python seed.py
"""

import os
import sys
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from io import BytesIO
from datetime import datetime
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── FRED fetcher ──────────────────────────────────────────────────────────────

def fetch_fred(series_id: str, start: str) -> pd.Series:
    """Fetch a single FRED series. Returns daily Series."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":         series_id,
        "api_key":           cfg.FRED_API_KEY,
        "file_type":         "json",
        "observation_start": start,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        data = {
            pd.Timestamp(o["date"]): float(o["value"])
            for o in obs if o["value"] != "."
        }
        s = pd.Series(data, name=series_id)
        log.info(f"  FRED {series_id}: {len(s)} observations")
        return s
    except Exception as e:
        log.error(f"  FRED {series_id} failed: {e}")
        return pd.Series(name=series_id, dtype=float)


def fetch_all_fred(start: str) -> pd.DataFrame:
    """Fetch all FRED macro series. Returns DataFrame indexed by date."""
    frames = []
    for sid in cfg.FRED_SERIES:
        s = fetch_fred(sid, start=start)
        if not s.empty:
            frames.append(s)
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index().ffill()


# ── yfinance fetcher ──────────────────────────────────────────────────────────

def fetch_ohlcv(tickers: list, start: str) -> pd.DataFrame:
    """
    Download OHLCV for a list of tickers from yfinance.
    Returns flat DataFrame with columns like {TICKER}_Open, {TICKER}_Close etc.
    Volume for regime indicators (^VIX, ^MOVE) is set to 0 if missing.
    """
    log.info(f"  Downloading {len(tickers)} tickers from {start} ...")
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError("yfinance returned empty DataFrame")

    result = pd.DataFrame()
    failed = []

    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df_t = raw.xs(ticker, axis=1, level=1).copy()
            else:
                df_t = raw.copy()

            if df_t.empty:
                raise ValueError("empty")

            for col in cfg.OHLCV_COLS:
                col_name = f"{ticker}_{col}"
                if col in df_t.columns:
                    result[col_name] = df_t[col]
                else:
                    # Volume missing for index tickers like ^VIX, ^MOVE
                    result[col_name] = 0.0

            log.info(f"  ✓ {ticker}: {len(df_t)} rows")
        except Exception as e:
            log.warning(f"  ✗ {ticker} failed: {e}")
            failed.append(ticker)

    if failed:
        log.warning(f"  Failed tickers: {failed}")

    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_module(
    module: str,
    etfs: list,
    benchmark: str,
    regime_ticker: str,
    start: str,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a complete dataset for one module (equity or fixed_income).
    Merges OHLCV prices with macro indicators.
    """
    log.info(f"\nBuilding {module} module ...")
    tickers = etfs + [benchmark, regime_ticker]

    # Fetch OHLCV
    price_df = fetch_ohlcv(tickers, start=start)

    # Align macro to trading days
    macro_aligned = macro_df.reindex(price_df.index, method="ffill")

    # Merge
    df = pd.concat([price_df, macro_aligned], axis=1)
    df = df.sort_index()

    # Drop rows where ALL ETF close prices are missing
    close_cols = [f"{t}_Close" for t in etfs if f"{t}_Close" in df.columns]
    df = df.dropna(subset=close_cols, how="all")

    # Forward-fill macro gaps (weekends/holidays)
    macro_cols = list(cfg.FRED_SERIES.keys())
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    log.info(f"  {module}: {len(df)} rows × {df.shape[1]} cols "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── HF push ───────────────────────────────────────────────────────────────────

def push_to_hf(df: pd.DataFrame, filename: str):
    """Push DataFrame as parquet to HF dataset repo."""
    log.info(f"  Pushing {filename} to HF ...")
    api    = HfApi(token=cfg.HF_TOKEN)
    buf    = BytesIO()
    df_out = df.copy()
    df_out.index.name = "date"
    df_out.to_parquet(buf, index=True)
    buf.seek(0)
    api.upload_file(
        path_or_fileobj=buf.getvalue(),
        path_in_repo=f"data/{filename}",
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=f"seed: {filename} ({len(df)} rows, "
                       f"{df.index[0].date()} → {df.index[-1].date()})",
    )
    log.info(f"  ✓ Pushed data/{filename}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)

    log.info("=" * 60)
    log.info("P2-ETF-MERTON-ANN — Full Historical Seed")
    log.info(f"Run date: {datetime.utcnow().date()}")
    log.info("=" * 60)

    # Fetch macro once — covers both modules
    log.info("\nFetching FRED macro series ...")
    macro_df = fetch_all_fred(start=cfg.FRED_START)
    log.info(f"  Macro: {len(macro_df)} rows, {macro_df.shape[1]} series")

    if not macro_df.empty:
        # Save macro separately
        push_to_hf(macro_df, "fred_macro.parquet")
    else:
        log.warning("Macro DataFrame is empty; not saving macro file.")

    # ── Equity module ─────────────────────────────────────────────────────────
    equity_df = build_module(
        module        = "equity",
        etfs          = cfg.EQUITY_ETFS,
        benchmark     = cfg.EQUITY_BENCHMARK,
        regime_ticker = cfg.EQUITY_REGIME,
        start         = cfg.EQUITY_START,
        macro_df      = macro_df,
    )
    push_to_hf(equity_df, "equity.parquet")

    # ── Fixed income module ───────────────────────────────────────────────────
    fi_df = build_module(
        module        = "fixed_income",
        etfs          = cfg.FI_ETFS,
        benchmark     = cfg.FI_BENCHMARK,
        regime_ticker = cfg.FI_REGIME,
        start         = cfg.FI_START,
        macro_df      = macro_df,
    )
    push_to_hf(fi_df, "fixed_income.parquet")

    log.info("\n✓ Seed complete.")
    log.info(f"  Equity:       {len(equity_df):,} rows "
             f"({equity_df.index[0].date()} → {equity_df.index[-1].date()})")
    log.info(f"  Fixed Income: {len(fi_df):,} rows "
             f"({fi_df.index[0].date()} → {fi_df.index[-1].date()})")


if __name__ == "__main__":
    main()
