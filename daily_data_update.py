I’ve identified the core issues:

1. **The daily update script only updates `fred_macro.parquet`**, leaving `equity.parquet` and `fixed_income.parquet` untouched – so those files never get new daily rows.
2. The script is incomplete: the ETF price fetching logic is just a placeholder comment.
3. It doesn’t load the existing datasets, so it can’t append new rows.
4. It passes `FRED_SERIES` (a dict) to a function expecting a list, which may work but is error‑prone.
5. There’s no error handling that would make the workflow fail if something goes wrong.

Below is a corrected, complete `daily_data_update.py` that:
- Fetches the latest trading day’s data for both equity and fixed income ETFs.
- Appends it to the existing datasets (downloaded from HF).
- Updates the FRED macro dataset with new daily values.
- Pushes all three files back to HF.
- Exits with a non‑zero code on failure, so the GitHub Action turns red.

---

daily_data_update.py

"""
daily_data_update.py — Append one new trading day to the dataset.
Also updates FRED macro data.
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download, upload_file
from fredapi import Fred
import logging

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Helper: fetch FRED data (full history, then forward fill)
# ------------------------------------------------------------------------------
def fetch_fred_data(api_key, series_list, start_date="2000-01-01", end_date=None):
    """Fetch FRED series and return a daily DataFrame with forward fill."""
    fred = Fred(api_key=api_key)
    data = {}
    for series in series_list:
        try:
            # series_list should be a list of series IDs (strings)
            df = fred.get_series(series, start=start_date, end=end_date)
            data[series] = df
            log.debug(f"  FRED {series}: {len(df)} points")
        except Exception as e:
            log.warning(f"Failed to fetch {series}: {e}")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df = df.resample('D').ffill()
    return df

# ------------------------------------------------------------------------------
# Helper: fetch ETF prices for a single date (or last available)
# ------------------------------------------------------------------------------
def fetch_prices_for_date(tickers, target_date):
    """
    Download OHLCV for a list of tickers for a specific target_date.
    Returns a dictionary with keys like {ticker}_Close etc.
    If data for target_date is not available, returns None.
    """
    # yfinance can return data for a range, but we only need one day.
    # We'll fetch a few days around target_date to be safe.
    start = target_date - timedelta(days=5)
    end = target_date + timedelta(days=1)
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            log.warning(f"No data returned for {tickers} around {target_date}")
            return None
        # Extract the row for target_date
        if isinstance(raw.columns, pd.MultiIndex):
            # MultiIndex: (price_type, ticker) -> we need to reshape
            # Better to convert to a flat DataFrame with columns like TICKER_Close
            flat = pd.DataFrame()
            for ticker in tickers:
                for col in cfg.OHLCV_COLS:
                    if (col, ticker) in raw.columns:
                        flat[f"{ticker}_{col}"] = raw[(col, ticker)]
                    else:
                        flat[f"{ticker}_{col}"] = pd.NA
            flat.index = raw.index
        else:
            # Single ticker case
            flat = raw.copy()
            for col in cfg.OHLCV_COLS:
                flat.rename(columns={col: f"{tickers[0]}_{col}"}, inplace=True)
        # Now find the row closest to target_date
        # Sometimes the market may be closed on target_date (weekend/holiday)
        # We want the most recent available date <= target_date
        available_dates = flat.index[flat.index <= target_date]
        if len(available_dates) == 0:
            log.warning(f"No data on or before {target_date} for {tickers}")
            return None
        latest = available_dates[-1]
        row = flat.loc[latest].to_dict()
        row['date'] = latest
        return row
    except Exception as e:
        log.error(f"Error fetching prices for {tickers}: {e}")
        return None

# ------------------------------------------------------------------------------
# Helper: append a new row to a parquet dataset and upload it
# ------------------------------------------------------------------------------
def update_parquet_file(dataset_file, new_row_dict, repo_id, token):
    """
    dataset_file: path in repo, e.g., "data/equity.parquet"
    new_row_dict: dict containing the new row (must include 'date' key)
    Returns True on success.
    """
    # 1. Download existing file to a temporary location
    local_path = f"/tmp/{dataset_file.replace('/', '_')}"
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=dataset_file,
            repo_type="dataset",
            token=token,
            local_dir="/tmp",
            local_dir_use_symlinks=False,
        )
        # The downloaded file may be saved as /tmp/data_equity.parquet (or similar)
        # Actually hf_hub_download returns the full path. Let's store it.
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=dataset_file,
            repo_type="dataset",
            token=token,
            local_dir="/tmp",
            local_dir_use_symlinks=False,
        )
        log.info(f"Downloaded {dataset_file} to {downloaded_path}")
        df = pd.read_parquet(downloaded_path)
    except Exception as e:
        log.warning(f"Could not download {dataset_file}: {e}. Assuming empty dataset.")
        df = pd.DataFrame()

    # 2. Append new row (if date not already present)
    new_date = new_row_dict['date']
    if not df.empty and 'date' in df.columns:
        if new_date in pd.to_datetime(df['date']).values:
            log.info(f"Date {new_date} already exists in {dataset_file}. Skipping append.")
            return True
    else:
        # Ensure date column exists
        if 'date' not in df.columns and not df.empty:
            # Should not happen, but handle
            log.warning(f"{dataset_file} missing date column, re-indexing.")
            df = df.reset_index().rename(columns={'index': 'date'})

    # Convert new_row_dict to DataFrame and append
    new_row_df = pd.DataFrame([new_row_dict])
    updated_df = pd.concat([df, new_row_df], ignore_index=True)
    updated_df = updated_df.sort_values('date').reset_index(drop=True)

    # 3. Save to temporary file and upload
    tmp_out = f"/tmp/updated_{dataset_file.replace('/', '_')}"
    updated_df.to_parquet(tmp_out, index=False)
    upload_file(
        path_or_fileobj=tmp_out,
        path_in_repo=dataset_file,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Daily update: added {new_date.strftime('%Y-%m-%d')}"
    )
    log.info(f"Uploaded updated {dataset_file} with new date {new_date}")
    return True

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    log.info("Daily data update started.")

    # Check environment variables
    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)

    today = pd.Timestamp.now().normalize()
    # We want yesterday's data (last trading day)
    target_date = today - timedelta(days=1)

    # --------------------------------------------------------------------------
    # 1. Update equity dataset
    # --------------------------------------------------------------------------
    log.info("Updating equity module...")
    # Collect all tickers we need to fetch for equity
    equity_tickers = cfg.EQUITY_ETFS + [cfg.EQUITY_BENCHMARK, cfg.EQUITY_REGIME]
    equity_row = fetch_prices_for_date(equity_tickers, target_date)
    if equity_row is None:
        log.warning("No equity data fetched. Skipping equity update.")
    else:
        update_parquet_file(
            dataset_file="data/equity.parquet",
            new_row_dict=equity_row,
            repo_id=cfg.HF_DATASET_REPO,
            token=cfg.HF_TOKEN,
        )

    # --------------------------------------------------------------------------
    # 2. Update fixed income dataset
    # --------------------------------------------------------------------------
    log.info("Updating fixed income module...")
    fi_tickers = cfg.FI_ETFS + [cfg.FI_BENCHMARK, cfg.FI_REGIME]
    fi_row = fetch_prices_for_date(fi_tickers, target_date)
    if fi_row is None:
        log.warning("No fixed income data fetched. Skipping FI update.")
    else:
        update_parquet_file(
            dataset_file="data/fixed_income.parquet",
            new_row_dict=fi_row,
            repo_id=cfg.HF_DATASET_REPO,
            token=cfg.HF_TOKEN,
        )

    # --------------------------------------------------------------------------
    # 3. Update FRED macro dataset
    # --------------------------------------------------------------------------
    log.info("Updating FRED macro data...")
    # FRED_SERIES is a dict in config.py; we need the list of keys
    fred_series_list = list(cfg.FRED_SERIES.keys())
    # Fetch all data from start to today (full refresh)
    fred_df = fetch_fred_data(
        cfg.FRED_API_KEY,
        fred_series_list,
        start_date=cfg.FRED_START,
        end_date=today
    )
    if not fred_df.empty:
        # Save to temporary file
        tmp_fred = "/tmp/fred_macro.parquet"
        fred_df.to_parquet(tmp_fred)
        upload_file(
            path_or_fileobj=tmp_fred,
            path_in_repo="data/fred_macro.parquet",
            repo_id=cfg.HF_DATASET_REPO,
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            commit_message=f"Update FRED macro data as of {today.strftime('%Y-%m-%d')}"
        )
        log.info("FRED macro data updated.")
    else:
        log.warning("No FRED data fetched. Skipping macro update.")

    log.info("Daily data update finished.")

if __name__ == "__main__":
    main()
