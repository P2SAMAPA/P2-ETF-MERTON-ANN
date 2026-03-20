"""
train_predict.py — P2-ETF-MERTON-ANN
Daily pipeline: calibrate → simulate → train 3 ANNs (21/63/126d) → predict winner.
Outputs single ETF selection with NYSE calendar next trading date.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Debug: Check environment variables at startup
print("Environment Check:")
hf_repo_env = os.environ.get('HF_DATASET_REPO', 'NOT SET')
hf_token_env = os.environ.get('HF_TOKEN', '')
print(f"  HF_DATASET_REPO env: {hf_repo_env}")
print(f"  HF_TOKEN env: {'SET (' + str(len(hf_token_env)) + ' chars)' if hf_token_env else 'NOT SET'}")
print(f"  FRED_API_KEY env: {'SET' if os.environ.get('FRED_API_KEY') else 'NOT SET'}")

# Import local modules
from config import (
    HF_DATASET_REPO, HF_TOKEN, FRED_API_KEY,
    EQUITY_ETFS, EQUITY_REGIME, EQUITY_BENCHMARK,
    FI_ETFS, FI_REGIME, FI_BENCHMARK,
    FRED_SERIES
)
from regime_detection import full_regime_analysis, get_current_regime
from calibration import calibrate_both_windows, get_risk_free_rate_from_fred
from simulation import generate_merton_training_data
from ann_model import train_ann_for_horizon, predict_optimal_etf, MertonANN

# HuggingFace
from huggingface_hub import HfApi, hf_hub_download

# Verify HF repo connection (after imports)
try:
    api = HfApi(token=HF_TOKEN)

    # Try to get repo info
    print(f"\nVerifying HF repo: {HF_DATASET_REPO}")
    repo_info = api.repo_info(repo_id=HF_DATASET_REPO, repo_type="dataset")
    print(f"✓ Repo found: {repo_info.id}")
    print(f"  Private: {repo_info.private}")

    # List files in data folder
    print(f"  Listing files in repo:")
    files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
    data_files = [f for f in files if f.startswith("data/")]
    print(f"    Data files: {data_files}")

except Exception as e:
    print(f"✗ ERROR verifying repo: {e}")
    print(f"  This usually means:")
    print(f"  1. HF_DATASET_REPO secret is wrong (check for typos/extra spaces)")
    print(f"  2. HF_TOKEN doesn't have read access")
    print(f"  3. Repo doesn't exist or is private")
    print(f"\nExpected format: P2SAMAPA/p2-etf-merton-ann-data")
    print(f"Current value: {HF_DATASET_REPO}")

# NYSE Calendar
try:
    from pandas_market_calendars import get_calendar
    NYSE = get_calendar("NYSE")
except ImportError:
    NYSE = None


def load_data_from_hf(module: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price and FRED data from HuggingFace dataset.

    Returns: (prices_df, fred_df)
    """
    print(f"\nLoading data for module: {module}")
    print(f"Using repo: {HF_DATASET_REPO}")
    print(f"Token length: {len(HF_TOKEN) if HF_TOKEN else 0}")

    try:
        # Download prices file using hf_hub_download function
        print(f"Downloading data/{module}.parquet...")
        prices_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"data/{module}.parquet",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="/tmp",
            local_dir_use_symlinks=False
        )
        prices = pd.read_parquet(prices_path)
        print(f"✓ Loaded prices: {prices.shape}")

        # Try to load FRED data
        try:
            print("Downloading data/fred_macro.parquet...")
            fred_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename="data/fred_macro.parquet",
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir="/tmp",
                local_dir_use_symlinks=False
            )
            fred_df = pd.read_parquet(fred_path)
            print(f"✓ Loaded FRED data: {fred_df.shape}")
        except Exception as e:
            print(f"⚠ FRED data not available: {e}")
            fred_df = pd.DataFrame()

        return prices, fred_df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        print(traceback.format_exc())
        # Return empty DataFrames as fallback
        return pd.DataFrame(), pd.DataFrame()


def get_next_trading_date(current_date: pd.Timestamp = None) -> pd.Timestamp:
    """
    Get next NYSE trading date using market calendar.
    """
    if current_date is None:
        current_date = pd.Timestamp.now()

    if NYSE is None:
        # Fallback: simple business day
        next_date = current_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:  # Saturday=5, Sunday=6
            next_date += pd.Timedelta(days=1)
        return next_date

    # Get valid trading days
    schedule = NYSE.schedule(start_date=current_date, end_date=current_date + pd.Timedelta(days=10))
    future_dates = schedule[schedule.index > current_date]

    if len(future_dates) > 0:
        return future_dates.index[0]
    else:
        # Fallback
        return current_date + pd.Timedelta(days=1)


def process_module(
    module: str,
    etfs: List[str],
    regime_ticker: str,
    benchmark: str,
    horizons: List[int] = [21, 63, 126],
    eta: float = 0.5,
    n_paths: int = 10000
) -> Dict:
    """
    Process one module (equity or fixed_income).

    Returns signal dict with selected ETF and metadata.
    """
    print(f"\n=== Processing {module} ===")

    # Load data
    prices, fred_df = load_data_from_hf(module)

    if prices.empty:
        return {"error": f"No data for {module}"}

    # Extract regime indicator
    print(f"Available columns: {list(prices.columns)[:10]}...")  # Show first 10
    print(f"Looking for regime ticker: {regime_ticker}")

    regime_prices = None

    # Try different column structures
    if isinstance(prices.columns, pd.MultiIndex):
        # MultiIndex structure (Ticker, Field)
        if regime_ticker in prices.columns.get_level_values(0):
            regime_prices = prices[regime_ticker]['Close'] if 'Close' in prices[regime_ticker].columns else prices[regime_ticker].iloc[:, 0]
    else:
        # Flat column structure - try various naming conventions
        possible_names = [
            regime_ticker,
            regime_ticker.replace('^', ''),
            f"{regime_ticker}_Close",
            regime_ticker.lstrip('^')
        ]

        for name in possible_names:
            if name in prices.columns:
                regime_prices = prices[name]
                print(f"✓ Found regime indicator as: {name}")
                break

    if regime_prices is None:
        print(f"✗ Regime indicator {regime_ticker} not found. Tried: {possible_names}")

    if regime_prices is None:
        return {"error": f"Regime indicator {regime_ticker} not found"}

    # Regime detection
    regime_analysis = full_regime_analysis(regime_prices)
    current_regime = regime_analysis["current_regime"]
    semi_markov_params = regime_analysis["semi_markov_params"]

    print(f"Current regime: {regime_analysis['current_regime_label']}")
    print(f"Threshold: {regime_analysis['threshold']:.2f}")

    # Risk-free rate
    rf_series = get_risk_free_rate_from_fred(fred_df)

    # Calibration (both windows)
    current_date = prices.index[-1]
    calibration_results = calibrate_both_windows(
        prices, etfs, regime_analysis["regime_history"], rf_series, current_date
    )

    best_result = None
    best_annualized_return = -np.inf
    best_horizon = None
    best_window = None

    # Train ANNs for each horizon and window
    for window_type, params in calibration_results.items():
        print(f"\n--- {window_type} window ---")

        for T_days in horizons:
            print(f"Training {T_days}-day horizon...")

            # Generate training data
            training_data = generate_merton_training_data(
                params, semi_markov_params, T_days, n_paths, W0=1.0, eta=eta
            )

            # Train ANN
            model = train_ann_for_horizon(
                training_data, len(etfs), eta, epochs=500, learning_rate=0.01
            )

            # Predict for current state (t=0, W=W0, current_regime)
            selected_idx, weights = predict_optimal_etf(model, 0.0, 0.0, current_regime)

            # Estimate expected return (simplified backtest on synthetic data)
            # Use last 100 training samples for validation
            X_val = training_data["X"][-1000:]
            y_val = training_data["y"][-1000:]

            # Portfolio returns under current regime
            mu = params[current_regime]["mu"]

            # Average expected return of ANN portfolio
            ann_weights = model.predict(X_val)
            expected_returns = ann_weights @ mu  # Annualized
            avg_annualized_return = np.mean(expected_returns)

            print(f"  Expected annualized return: {avg_annualized_return:.2%}")

            # Track best
            if avg_annualized_return > best_annualized_return:
                best_annualized_return = avg_annualized_return
                best_horizon = T_days
                best_window = window_type
                best_result = {
                    "selected_etf": etfs[selected_idx],
                    "selected_idx": int(selected_idx),
                    "weights": {etfs[i]: float(weights[i]) for i in range(len(etfs))},
                    "regime": "risk-on" if current_regime == 0 else "risk-off",
                    "horizon_days": T_days,
                    "window_type": window_type,
                    "expected_return_annualized": float(avg_annualized_return),
                    "model_params": model.get_weights(),
                    "n_parameters": model.count_parameters(),
                }

    # Add metadata
    next_trading_date = get_next_trading_date(current_date)

    signal = {
        "date": current_date.strftime("%Y-%m-%d"),
        "next_trading_date": next_trading_date.strftime("%Y-%m-%d"),
        "module": module,
        **best_result,
        "all_horizons_tested": horizons,
        "semi_markov_params": semi_markov_params,
        "regime_threshold": float(regime_analysis["threshold"]),
    }

    print(f"\n✓ Best: {signal['selected_etf']} ({signal['horizon_days']}d, {signal['window_type']})")
    print(f"  Expected return: {signal['expected_return_annualized']:.2%}")

    return signal


def save_signal_to_hf(signal: Dict, module: str):
    """Save signal JSON to HuggingFace dataset."""
    import traceback

    print(f"\n{'='*60}")
    print(f"SAVE SIGNAL: {module}")
    print(f"{'='*60}")
    print(f"Repo: {HF_DATASET_REPO}")
    print(f"Token available: {'Yes (len=%d)' % len(HF_TOKEN) if HF_TOKEN else 'No'}")

    # Check signal validity - don't save error signals
    if not signal or 'error' in signal:
        print(f"✗ ERROR: Signal for {module} failed - not saving to HF")
        print(f"Signal content: {signal}")
        raise ValueError(f"Cannot save error signal for {module}: {signal.get('error', 'Unknown error')}")

    try:
        api = HfApi(token=HF_TOKEN)

        # Create temp file
        temp_file = f"/tmp/{module}_signal.json"
        os.makedirs("/tmp", exist_ok=True)

        with open(temp_file, 'w') as f:
            json.dump(signal, f, indent=2, default=str)

        print(f"✓ Created temp file: {temp_file}")
        print(f"  File size: {os.path.getsize(temp_file)} bytes")
        print(f"  Selected ETF: {signal.get('selected_etf', 'N/A')}")

        # Ensure signals folder exists by uploading with create_pr set to False
        path_in_repo = f"signals/{module}_signal.json"
        print(f"Uploading to: {path_in_repo}")

        # Upload file - use create_pr=False to commit directly to main
        try:
            result = api.upload_file(
                path_or_fileobj=temp_file,
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Update {module} signal for {signal.get('date', 'unknown')}",
            )
            print(f"✓ Successfully uploaded: {path_in_repo}")
        except Exception as upload_error:
            print(f"⚠ First upload attempt failed: {upload_error}")
            print("Trying alternative upload method...")

            # Alternative: try with explicit branch
            result = api.upload_file(
                path_or_fileobj=temp_file,
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Update {module} signal",
                create_pr=False,
            )
            print(f"✓ Alternative upload succeeded: {path_in_repo}")

        # Also append to history
        try:
            history_file = f"/tmp/{module}_history.json"
            history = []

            try:
                existing_path = hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    filename=f"signals/{module}_history.json",
                    token=HF_TOKEN
                )
                with open(existing_path, 'r') as f:
                    history = json.load(f)
                print(f"✓ Loaded existing history: {len(history)} records")
            except Exception as e:
                print(f"ℹ No existing history (OK for first run)")
                history = []

            history.append(signal)

            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)

            api.upload_file(
                path_or_fileobj=history_file,
                path_in_repo=f"signals/{module}_history.json",
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Update {module} history"
            )
            print(f"✓ Updated history: {len(history)} total records")

        except Exception as e:
            print(f"⚠ History update failed (non-critical): {e}")

        print(f"{'='*60}\n")

    except Exception as e:
        print(f"✗ ERROR saving to HF: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print(f"{'='*60}\n")
        raise


def main():
    """Main daily pipeline."""
    print("=" * 60)
    print("P2-ETF-MERTON-ANN: Daily Training & Prediction")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Parameters
    ETA = 0.5  # Risk aversion
    HORIZONS = [21, 63, 126]  # Days
    N_PATHS = 10000  # Synthetic paths

    results = {}

    # Process Equity Module
    print("\n>>> Starting Equity Module Processing <<<")
    try:
        equity_signal = process_module(
            "equity",
            EQUITY_ETFS,
            EQUITY_REGIME,
            EQUITY_BENCHMARK,
            HORIZONS,
            ETA,
            N_PATHS
        )
        print(f"Equity processing complete. Signal: {equity_signal.get('selected_etf', 'ERROR')}")
        results["equity"] = equity_signal
        save_signal_to_hf(equity_signal, "equity")
    except Exception as e:
        print(f"✗ ERROR in equity processing: {e}")
        import traceback
        print(traceback.format_exc())
        equity_signal = {"error": str(e), "module": "equity"}
        results["equity"] = equity_signal

    # Process Fixed Income Module
    print("\n>>> Starting Fixed Income Module Processing <<<")
    try:
        fi_signal = process_module(
            "fixed_income",
            FI_ETFS,
            FI_REGIME,
            FI_BENCHMARK,
            HORIZONS,
            ETA,
            N_PATHS
        )
        print(f"Fixed Income processing complete. Signal: {fi_signal.get('selected_etf', 'ERROR')}")
        results["fixed_income"] = fi_signal
        save_signal_to_hf(fi_signal, "fi")
    except Exception as e:
        print(f"✗ ERROR in fixed income processing: {e}")
        import traceback
        print(traceback.format_exc())
        fi_signal = {"error": str(e), "module": "fixed_income"}
        results["fixed_income"] = fi_signal

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Equity:      Hold {equity_signal.get('selected_etf', 'N/A')} "
          f"for {equity_signal.get('next_trading_date', 'N/A')} "
          f"({equity_signal.get('expected_return_annualized', 0):.1%} exp)")
    print(f"Fixed Inc:   Hold {fi_signal.get('selected_etf', 'N/A')} "
          f"for {fi_signal.get('next_trading_date', 'N/A')} "
          f"({fi_signal.get('expected_return_annualized', 0):.1%} exp)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"FATAL ERROR: {e}")
        print(traceback.format_exc())
        raise
