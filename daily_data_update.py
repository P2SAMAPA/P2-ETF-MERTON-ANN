"""
daily_data_update.py — Append one new trading day to the dataset.
Also updates FRED macro data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from huggingface_hub import HfApi

import config as cfg
import data_utils as du

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Helper: get next trading day (using data_utils)
# ------------------------------------------------------------------------------
def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Return the next NYSE trading day after the given date."""
    # Use the same calendar as seed.py
    trading_days = du.get_trading_days(
        start=date.strftime("%Y-%m-%d"),
        end=(date + timedelta(days=10)).strftime("%Y-%m-%d")
    )
    next_days = trading_days[trading_days > date]
    if len(next_days) > 0:
        return next_days[0]
    # fallback: skip weekends only
    d = date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ------------------------------------------------------------------------------
# Main update
# ------------------------------------------------------------------------------
def main():
    log.info("Daily data update started.")

    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # 1. Load current OHLCV file to get last date
    # --------------------------------------------------------------------------
    try:
        ohlcv_existing = du.load_parquet(cfg.FILE_ETF_OHLCV)
        if ohlcv_existing.empty:
            log.error("No existing OHLCV data – run seed.py first.")
            sys.exit(1)
        last_date = ohlcv_existing.index[-1]
        log.info(f"Last stored OHLCV date: {last_date.date()}")
    except Exception as e:
        log.error(f"Failed to load existing OHLCV: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # 2. Determine target date (next trading day after last_date)
    # --------------------------------------------------------------------------
    target_date = next_trading_day(last_date)
    if target_date.date() > datetime.now().date():
        log.info("Next trading day is in the future – nothing to update.")
        return
    log.info(f"Update window: {target_date.date()} to {target_date.date()}")

    # --------------------------------------------------------------------------
    # 3. Fetch new OHLCV for the target date
    # --------------------------------------------------------------------------
    start = target_date.strftime("%Y-%m-%d")
    end   = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"Fetching OHLCV for {target_date.date()} ...")
    try:
        ohlcv_multi = du.download_ohlcv(cfg.ALL_TICKERS, start=start, end=end)
        if ohlcv_multi.empty:
            log.error("No OHLCV data returned – skipping update.")
            return
        new_ohlcv_flat = du.flatten_ohlcv(ohlcv_multi)
    except Exception as e:
        log.error(f"OHLCV fetch failed: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # 4. Append to etf_ohlcv.parquet
    # --------------------------------------------------------------------------
    # Ensure columns match existing (new_ohlcv_flat may have fewer columns)
    if not ohlcv_existing.empty:
        new_ohlcv_flat = new_ohlcv_flat.reindex(ohlcv_existing.columns, fill_value=np.nan)
    ohlcv_updated = pd.concat([ohlcv_existing, new_ohlcv_flat], axis=0).sort_index()
    du.upload_parquet(ohlcv_updated, cfg.FILE_ETF_OHLCV,
                      f"Daily update: added {target_date.date()}")

    # --------------------------------------------------------------------------
    # 5. Recompute etf_returns.parquet from the updated OHLCV
    # --------------------------------------------------------------------------
    returns = du.compute_returns(ohlcv_updated, cfg.ALL_TICKERS)
    du.upload_parquet(returns, cfg.FILE_ETF_RETURNS,
                      f"Daily update: recomputed returns to {target_date.date()}")

    # --------------------------------------------------------------------------
    # 6. Recompute etf_vol.parquet from the updated returns
    # --------------------------------------------------------------------------
    if not returns.empty:
        vol = pd.DataFrame(index=returns.index)
        for t in cfg.ALL_TICKERS:
            ret_col = f"{t}_ret"
            if ret_col in returns.columns:
                vol[f"{t}_vol"] = returns[ret_col].rolling(cfg.VOL_WINDOW).std() * np.sqrt(252)
        vol = vol.dropna(how="all")
        du.upload_parquet(vol, cfg.FILE_ETF_VOL,
                          f"Daily update: recomputed vol to {target_date.date()}")
    else:
        log.warning("Returns empty – skipping volatility update.")

    # --------------------------------------------------------------------------
    # 7. Fetch FRED macro data for the target date
    # --------------------------------------------------------------------------
    log.info(f"Fetching FRED data for {target_date.date()} ...")
    try:
        macro_new = du.download_fred(start=start, end=start)   # single day
        if macro_new.empty:
            log.warning("FRED data returned empty – using NaNs.")
            macro_new = pd.DataFrame(index=[target_date], columns=cfg.FRED_SERIES.keys(), dtype=float)
    except Exception as e:
        log.error(f"FRED fetch failed: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # 8. Append to macro_fred.parquet
    # --------------------------------------------------------------------------
    macro_existing = du.load_parquet(cfg.FILE_MACRO_FRED)
    if not macro_existing.empty:
        macro_new = macro_new.reindex(macro_existing.columns, fill_value=np.nan)
    macro_updated = pd.concat([macro_existing, macro_new], axis=0).sort_index()
    du.upload_parquet(macro_updated, cfg.FILE_MACRO_FRED,
                      f"Daily update: added macro for {target_date.date()}")

    # --------------------------------------------------------------------------
    # 9. Recompute macro_derived.parquet from the updated macro_fred
    # --------------------------------------------------------------------------
    macro_derived = du.compute_macro_derived(macro_updated)
    du.upload_parquet(macro_derived, cfg.FILE_MACRO_DERIVED,
                      f"Daily update: recomputed derived macro to {target_date.date()}")

    # --------------------------------------------------------------------------
    # 10. Rebuild master.parquet from all updated files
    # --------------------------------------------------------------------------
    master = du.build_master(
        ohlcv_updated,
        returns,
        macro_updated,
        macro_derived
    )
    du.upload_parquet(master, cfg.FILE_MASTER,
                      f"Daily update: rebuilt master to {target_date.date()}")

    log.info("Daily update completed successfully.")


if __name__ == "__main__":
    main()
