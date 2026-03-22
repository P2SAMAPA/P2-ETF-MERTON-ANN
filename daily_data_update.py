"""
daily_data_update.py — Append one new trading day to the dataset.
Also updates FRED macro data.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import (
    EQUITY_ETFS, FI_ETFS, FRED_SERIES, HF_DATASET_REPO, HF_TOKEN, FRED_API_KEY
)
from huggingface_hub import HfApi, upload_file
import requests
from fredapi import Fred

# Helper: fetch FRED data
def fetch_fred_data(api_key, series_list, start_date="2000-01-01", end_date=None):
    fred = Fred(api_key=api_key)
    data = {}
    for series in series_list:
        try:
            df = fred.get_series(series, start=start_date, end=end_date)
            data[series] = df
        except Exception as e:
            print(f"Failed to fetch {series}: {e}")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df = df.resample('D').ffill()
    return df

# Main
def main():
    print("Daily data update started.")
    today = pd.Timestamp.now().normalize()
    # ... existing logic to fetch ETF prices for equity and fixed_income and append ...
    # After that, update FRED macro data
    fred_df = fetch_fred_data(FRED_API_KEY, FRED_SERIES, start_date="2000-01-01", end_date=today)
    if not fred_df.empty:
        # Save locally
        fred_df.to_parquet("/tmp/fred_macro.parquet")
        # Upload to HF
        upload_file(
            path_or_fileobj="/tmp/fred_macro.parquet",
            path_in_repo="data/fred_macro.parquet",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"Update FRED macro data as of {today.strftime('%Y-%m-%d')}"
        )
        print("FRED macro data updated.")
    else:
        print("No FRED data fetched.")
