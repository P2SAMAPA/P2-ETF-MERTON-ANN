"""
daily_data_update.py — Append one new trading day to the dataset.
Also updates FRED macro data.
"""

import os
import sys
import time
import random
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
# Helper: fetch ETF prices for a single date (or last available) with batching and retries
# ------------------------------------------------------------------------------
def fetch_prices_for_date(tickers, target_date):
    """
    Download OHLCV for a list of tickers for a specific target_date.
    Splits into batches to avoid rate limits, with exponential backoff.
    Returns a dictionary with keys like {ticker}_Close etc.
    If data for target_date is not available, returns None.
    """
    batch_size = 5
    all_rows = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        retries = 3
        for attempt in range(1, retries+1):
            try:
                start = target_date - timedelta(days=5)
                end = target_date + timedelta(days=1)
                raw = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
                if raw.empty:
                    raise ValueError("Empty download")
                # Flatten and get the row for the target_date
                if isinstance(raw.columns, pd.MultiIndex):
                    flat = pd.DataFrame()
                    for ticker in batch:
                        for col in cfg.OHLCV_COLS:
                            if (col, ticker) in raw.columns:
                                flat[f"{ticker}_{col}"] = raw[(col, ticker)]
                            else:
                                flat[f"{ticker}_{col}"] = pd.NA
                    flat.index = raw.index
                else:
                    flat = raw.copy()
                    for col in cfg.OHLCV_COLS:
                        flat.rename(columns={col: f"{batch[0]}_{col}"}, inplace=True)
                available_dates = flat.index[flat.index <= target_date]
                if len(available_dates) == 0:
                    raise ValueError(f"No data on or before {target_date}")
                latest = available_dates[-1]
                row = flat.loc[latest].to_dict()
                row['date'] = latest
                all_rows.append(row)
                break  # success, exit retry loop
            except Exception as e:
                if attempt == retries:
                    log.warning(f"Batch {batch} failed after {retries} attempts: {e}")
                else:
                    sleep_time = 2 ** attempt + random.uniform(0, 1)
                    log.warning(f"Batch {batch} attempt {attempt} failed: {e}. Retrying in {sleep_time:.2f}s")
                    time.sleep(sleep_time)

    if not all_rows:
        return None

    # Merge rows from all batches
    combined = {}
    for row in all_rows:
        combined.update(row)
    return combined


# ------------------------------------------------------------------------------
# Helper: append a new row to a parquet dataset and upload it
# ------------------------------------------------------------------------------
def update_parquet_file(dataset_file, new_row_dict, repo_id, token):
    """
    dataset_file: path in repo, e.g., "data/equity.parquet"
    new_row_dict: dict containing the new row (must include 'date' key)
    Returns True on success.
    """
    local_path = f"/tmp/{dataset_file.replace('/', '_')}"
    try:
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

    new_date = new_row_dict['date']
    if not df.empty and 'date' in df.columns:
        if new_date in pd.to_datetime(df['date']).values:
            log.info(f"Date {new_date} already exists in {dataset_file}. Skipping append.")
            return True
    else:
        if 'date' not in df.columns and not df.empty:
            log.warning(f"{dataset_file} missing date column, re-indexing.")
            df = df.reset_index().rename(columns={'index': 'date'})

    new_row_df = pd.DataFrame([new_row_dict])
    updated_df = pd.concat([df, new_row_df], ignore_index=True)
    updated_df = updated_df.sort_values('date').reset_index(drop=True)

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

    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)

    today = pd.Timestamp.now().normalize()
    target_date = today - timedelta(days=1)

    # --------------------------------------------------------------------------
    # 1. Update equity dataset
    # --------------------------------------------------------------------------
    log.info("Updating equity module...")
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
    fred_series_list = list(cfg.FRED_SERIES.keys())
    fred_df = fetch_fred_data(
        cfg.FRED_API_KEY,
        fred_series_list,
        start_date=cfg.FRED_START,
        end_date=today
    )
    if not fred_df.empty:
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
