"""
daily_data_update.py — P2-ETF-MERTON-ANN
------------------------------------------
Daily incremental update — appends new trading day(s) to both
equity.parquet and fixed_income.parquet on HuggingFace.

Runs Mon-Fri after US market close via daily_data_update.yml.
Safe to run manually anytime — idempotent (deduplicates by date).
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from io import BytesIO
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── HF helpers ────────────────────────────────────────────────────────────────

def load_from_hf(filename: str) -> pd.DataFrame:
    """Load existing parquet from HF dataset."""
    log.info(f"  Loading data/{filename} from HF ...")
    path = hf_hub_download(
        repo_id=cfg.HF_DATASET_REPO,
        filename=f"data/{filename}",
        repo_type="dataset",
        token=cfg.HF_TOKEN,
        force_download=True,
    )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    log.info(f"  Loaded: {len(df)} rows "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def push_to_hf(df: pd.DataFrame, filename: str):
    """Push updated DataFrame as parquet to HF dataset."""
    log.info(f"  Pushing data/{filename} to HF ...")
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
        commit_message=f"update: {filename} → {df.index[-1].date()} "
                       f"({len(df)} rows)",
    )
    log.info(f"  ✓ Pushed data/{filename}")


# ── FRED incremental ──────────────────────────────────────────────────────────

def fetch_fred_incremental(series_id: str, start: str) -> pd.Series:
    """Fetch FRED series from start date."""
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
        return pd.Series(data, name=series_id)
    except Exception as e:
        log.warning(f"  FRED {series_id} failed: {e}")
        return pd.Series(name=series_id, dtype=float)


def fetch_macro_incremental(start: str) -> pd.DataFrame:
    """Fetch all FRED macro series from start date."""
    frames = []
    for sid in cfg.FRED_SERIES:
        s = fetch_fred_incremental(sid, start=start)
        if not s.empty:
            frames.append(s)
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index().ffill()


# ── yfinance incremental ──────────────────────────────────────────────────────

def fetch_ohlcv_incremental(tickers: list, start: str) -> pd.DataFrame:
    """Fetch OHLCV for tickers from start date."""
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    result = pd.DataFrame()
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df_t = raw.xs(ticker, axis=1, level=1).copy()
            else:
                df_t = raw.copy()
            for col in cfg.OHLCV_COLS:
                col_name = f"{ticker}_{col}"
                result[col_name] = df_t[col] if col in df_t.columns else 0.0
        except Exception as e:
            log.warning(f"  {ticker}: {e}")

    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# ── Module updater ────────────────────────────────────────────────────────────

def update_module(
    filename: str,
    etfs: list,
    benchmark: str,
    regime_ticker: str,
) -> int:
    """
    Load existing parquet, fetch new rows, append, deduplicate, push back.
    Returns number of new rows added.
    """
    log.info(f"\nUpdating {filename} ...")

    # Load existing
    existing = load_from_hf(filename)
    last_date = existing.index[-1]

    # Determine fetch start — go back 5 days for safety (catches late data)
    fetch_start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
    log.info(f"  Fetching from {fetch_start} ...")

    # Fetch new price data
    tickers  = etfs + [benchmark, regime_ticker]
    new_prices = fetch_ohlcv_incremental(tickers, start=fetch_start)

    if new_prices.empty:
        log.info("  No new price data available")
        return 0

    # Fetch new macro
    new_macro = fetch_macro_incremental(start=fetch_start)

    # Align and merge new data
    if not new_macro.empty:
        new_macro_aligned = new_macro.reindex(new_prices.index, method="ffill")
        new_df = pd.concat([new_prices, new_macro_aligned], axis=1)
    else:
        new_df = new_prices

    # Combine with existing — deduplicate keeping latest
    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # Forward-fill macro gaps
    for col in cfg.FRED_SERIES:
        if col in combined.columns:
            combined[col] = combined[col].ffill()

    new_rows = len(combined) - len(existing)
    log.info(f"  New rows: {new_rows} | Total: {len(combined)} rows "
             f"(up to {combined.index[-1].date()})")

    if new_rows == 0:
        log.info("  Dataset already up to date — skipping push")
        return 0

    push_to_hf(combined, filename)
    return new_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)

    log.info("=" * 60)
    log.info("P2-ETF-MERTON-ANN — Daily Data Update")
    log.info(f"Run date: {datetime.utcnow().date()}")
    log.info("=" * 60)

    eq_new = update_module(
        filename      = "equity.parquet",
        etfs          = cfg.EQUITY_ETFS,
        benchmark     = cfg.EQUITY_BENCHMARK,
        regime_ticker = cfg.EQUITY_REGIME,
    )

    fi_new = update_module(
        filename      = "fixed_income.parquet",
        etfs          = cfg.FI_ETFS,
        benchmark     = cfg.FI_BENCHMARK,
        regime_ticker = cfg.FI_REGIME,
    )

    log.info("\n✓ Daily update complete.")
    log.info(f"  Equity new rows:       {eq_new}")
    log.info(f"  Fixed income new rows: {fi_new}")

    if eq_new == 0 and fi_new == 0:
        log.info("  Both datasets already current.")


if __name__ == "__main__":
    main()
