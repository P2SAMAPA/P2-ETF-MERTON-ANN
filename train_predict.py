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

from config import (
    HF_DATASET_REPO, HF_TOKEN, FRED_API_KEY,
    EQUITY_ETFS, EQUITY_REGIME, EQUITY_BENCHMARK,
    FI_ETFS, FI_REGIME, FI_BENCHMARK,
    FRED_SERIES
)
from regime_detection import full_regime_analysis, get_current_regime
from calibration import calibrate_both_windows, get_risk_free_rate_from_fred, validate_parameters
from simulation import generate_merton_training_data
from ann_model import train_ann_for_horizon, predict_optimal_etf, MertonANN

from huggingface_hub import HfApi, hf_hub_download

try:
    from pandas_market_calendars import get_calendar
    NYSE = get_calendar("NYSE")
except ImportError:
    NYSE = None


def load_data_from_hf(module: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load price and FRED data from HuggingFace dataset."""
    print(f"\nLoading data for module: {module}")
    print(f"Using repo: {HF_DATASET_REPO}")
    print(f"Token length: {len(HF_TOKEN) if HF_TOKEN else 0}")

    try:
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
        return pd.DataFrame(), pd.DataFrame()


def get_next_trading_date(current_date: pd.Timestamp = None) -> pd.Timestamp:
    """Get next NYSE trading date using market calendar."""
    if current_date is None:
        current_date = pd.Timestamp.now()

    if NYSE is None:
        next_date = current_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)
        return next_date

    schedule = NYSE.schedule(start_date=current_date, end_date=current_date + pd.Timedelta(days=10))
    future_dates = schedule[schedule.index > current_date]
    if len(future_dates) > 0:
        return future_dates.index[0]
    else:
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
    """Process one module (equity or fixed_income)."""
    print(f"\n=== Processing {module} ===")

    prices, fred_df = load_data_from_hf(module)
    if prices.empty:
        return {"error": f"No data for {module}"}

    # Extract regime indicator
    print(f"Available columns: {list(prices.columns)[:10]}...")
    print(f"Looking for regime ticker: {regime_ticker}")

    regime_prices = None
    if isinstance(prices.columns, pd.MultiIndex):
        if regime_ticker in prices.columns.get_level_values(0):
            regime_prices = prices[regime_ticker]['Close'] if 'Close' in prices[regime_ticker].columns else prices[regime_ticker].iloc[:, 0]
    else:
        possible_names = [f"{regime_ticker}_Close", f"{regime_ticker}_close", regime_ticker]
        for name in possible_names:
            if name in prices.columns:
                regime_prices = prices[name]
                print(f"✓ Found regime indicator as: {name}")
                break

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

    for window_type, params in calibration_results.items():
        # Validate parameters before proceeding
        if not validate_parameters(params):
            print(f"⚠ Skipping {window_type} due to invalid parameters (non-positive definite covariance or NaN)")
            continue

        print(f"\n--- {window_type} window ---")

        for T_days in horizons:
            print(f"Training {T_days}-day horizon...")

            n_assets = len(etfs)
            print(f"  DEBUG: n_assets={n_assets}")
            print(f"  DEBUG: params[0] mu shape={params[0]['mu'].shape if hasattr(params[0]['mu'], 'shape') else len(params[0]['mu'])}")

            # Generate training data with safety
            try:
                training_data = generate_merton_training_data(
                    params, semi_markov_params, T_days, n_paths, W0=1.0, eta=eta, n_assets=n_assets
                )
            except Exception as e:
                print(f"  ✗ Error generating training data: {e}")
                continue

            print(f"  DEBUG: training_data X shape={training_data['X'].shape}, y shape={training_data['y'].shape}")

            # Train ANN
            try:
                model = train_ann_for_horizon(
                    training_data, len(etfs), eta, epochs=500, learning_rate=0.01
                )
            except Exception as e:
                print(f"  ✗ Error training ANN: {e}")
                continue

            # Predict for current state (t=0, W=W0, current_regime)
            selected_idx, weights = predict_optimal_etf(model, 0.0, 0.0, current_regime)

            # Validate that weights are not NaN
            if np.any(np.isnan(weights)):
                print(f"  WARNING: NaN in ann_weights, replacing with equal weights")
                weights = np.ones(len(etfs)) / len(etfs)
                selected_idx = np.argmax(weights)

            # Estimate expected return
            X_val = training_data["X"][-1000:]
            y_val = training_data["y"][-1000:]

            ann_weights = model.predict(X_val)
            ann_weights = np.nan_to_num(ann_weights, nan=1.0/len(etfs))

            mu = params[current_regime]["mu"]
            mu = np.nan_to_num(mu, nan=0.0)

            expected_returns = ann_weights @ mu
            avg_annualized_return = np.mean(expected_returns)

            if np.isnan(avg_annualized_return):
                avg_annualized_return = 0.0

            print(f"  Expected annualized return: {avg_annualized_return:.2%}")

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

    if best_result is None:
        print(f"✗ ERROR: No valid training result found for {module}")
        return {"error": f"Training failed - all horizons returned NaN", "module": module}

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

    if not signal or 'error' in signal:
        print(f"✗ ERROR: Signal for {module} failed - not saving to HF")
        print(f"Signal content: {signal}")
        raise ValueError(f"Cannot save error signal for {module}: {signal.get('error', 'Unknown error')}")

    try:
        api = HfApi(token=HF_TOKEN)

        temp_file = f"/tmp/{module}_signal.json"
        os.makedirs("/tmp", exist_ok=True)

        with open(temp_file, 'w') as f:
            json.dump(signal, f, indent=2, default=str)

        print(f"✓ Created temp file: {temp_file}")
        print(f"  File size: {os.path.getsize(temp_file)} bytes")
        print(f"  Selected ETF: {signal.get('selected_etf', 'N/A')}")

        path_in_repo = f"signals/{module}_signal.json"
        print(f"Uploading to: {path_in_repo}")

        result = api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=path_in_repo,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Update {module} signal for {signal.get('date', 'unknown')}",
        )
        print(f"✓ Successfully uploaded: {path_in_repo}")

        # Append to history
        try:
            history_file = f"/tmp/{module}_history.json"
            history = []

            try:
                existing_path = hf_hub_download(
                    repo_id=HF_DATASET_REPO,
                    filename=f"signals/{module}_history.json",
                    token=HF_TOKEN,
                    local_dir="/tmp",
                    local_dir_use_symlinks=False
                )
                with open(existing_path, 'r') as f:
                    history = json.load(f)
                print(f"✓ Loaded existing history: {len(history)} records")
            except Exception:
                print(f"ℹ No existing history (OK for first run)")

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

    ETA = 0.5
    HORIZONS = [21, 63, 126]
    N_PATHS = 10000

    results = {}

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
