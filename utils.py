"""
utils.py — Shared utilities for the P2-ETF-MERTON-ANN pipeline.
Provides data loading, regime detection, calibration, simulation, training, and saving.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import HfApi, hf_hub_download
from sklearn.covariance import LedoitWolf

from config import (
    HF_DATASET_REPO, HF_TOKEN,
    EQUITY_ETFS, EQUITY_REGIME,
    FI_ETFS, FI_REGIME,
    FRED_SERIES
)
from regime_detection import full_regime_analysis
from calibration import compute_returns, get_risk_free_rate_from_fred, calibrate_both_windows
from simulation import generate_merton_training_data
from ann_model import train_ann_for_horizon, predict_optimal_etf

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data_from_hf(module: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load price and FRED data from HuggingFace dataset."""
    print(f"\nLoading data for module: {module}")
    print(f"Using repo: {HF_DATASET_REPO}")
    print(f"Token length: {len(HF_TOKEN) if HF_TOKEN else 0}")

    try:
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

        # Ensure index is datetime
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
            print("  Converted index to datetime")

        try:
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
            if not isinstance(fred_df.index, pd.DatetimeIndex):
                fred_df.index = pd.to_datetime(fred_df.index)
        except Exception as e:
            print(f"⚠ FRED data not available: {e}")
            fred_df = pd.DataFrame()

        return prices, fred_df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()

# ----------------------------------------------------------------------
# Signal saving
# ----------------------------------------------------------------------
def save_signal_to_hf(signal: Dict, module: str, option: str = "A"):
    """Save signal JSON to HuggingFace dataset with option suffix."""
    import traceback

    suffix = "" if option == "A" else f"_option{option}"
    print(f"\n{'='*60}")
    print(f"SAVE SIGNAL: {module} (Option {option})")
    print(f"{'='*60}")
    print(f"Repo: {HF_DATASET_REPO}")
    print(f"Token available: {'Yes (len=%d)' % len(HF_TOKEN) if HF_TOKEN else 'No'}")

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is empty or not set. Cannot upload.")

    if not signal or 'error' in signal:
        print(f"✗ ERROR: Signal for {module} failed - not saving to HF")
        print(f"Signal content: {signal}")
        raise ValueError(f"Cannot save error signal for {module}: {signal.get('error', 'Unknown error')}")

    try:
        api = HfApi(token=HF_TOKEN)

        temp_file = f"/tmp/{module}_signal{suffix}.json"
        os.makedirs("/tmp", exist_ok=True)

        with open(temp_file, 'w') as f:
            json.dump(signal, f, indent=2, default=str)

        print(f"✓ Created temp file: {temp_file}")
        print(f"  File size: {os.path.getsize(temp_file)} bytes")
        print(f"  Selected ETF: {signal.get('selected_etf', 'N/A')}")

        path_in_repo = f"signals/{module}_signal{suffix}.json"
        print(f"Uploading to: {path_in_repo}")

        api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=path_in_repo,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Update {module} signal (Option {option}) for {signal.get('date', 'unknown')}",
        )
        print(f"✓ Successfully uploaded: {path_in_repo}")

        # History handling
        history_file = f"/tmp/{module}_history{suffix}.json"
        history = []
        remote_history_path = f"signals/{module}_history{suffix}.json"

        # Try to download existing history
        try:
            print(f"Attempting to download existing history from {remote_history_path}...")
            existing_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=remote_history_path,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir="/tmp",
                local_dir_use_symlinks=False,
                force_download=True
            )
            with open(existing_path, 'r') as f:
                history = json.load(f)
            print(f"✓ Loaded existing history: {len(history)} records")
        except Exception as e:
            print(f"ℹ No existing history found or download failed: {e}")
            history = []

        # Append new record if not already present
        new_record = {
            "date": signal.get("date"),
            "next_trading_date": signal.get("next_trading_date"),
            "selected_etf": signal.get("selected_etf"),
            "weights": signal.get("weights"),
            "regime": signal.get("regime"),
            "horizon_days": signal.get("horizon_days"),
            "window_type": signal.get("window_type"),
            "expected_return_annualized": signal.get("expected_return_annualized"),
            "generated_at": datetime.utcnow().isoformat()
        }
        # Remove None values for cleaner JSON
        new_record = {k: v for k, v in new_record.items() if v is not None}

        existing_dates = {r.get("date") for r in history if "date" in r}
        if new_record["date"] not in existing_dates:
            history.append(new_record)
            print(f"✓ Appended new record for {new_record['date']}")
        else:
            print(f"ℹ Record for {new_record['date']} already exists – skipping append")

        # Write updated history locally
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)

        # Upload history
        print(f"Uploading updated history to {remote_history_path}...")
        api.upload_file(
            path_or_fileobj=history_file,
            path_in_repo=remote_history_path,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Update {module} history (Option {option}) – {len(history)} records"
        )
        print(f"✓ Updated history: {len(history)} total records")

        print(f"{'='*60}\n")

    except Exception as e:
        print(f"✗ ERROR saving to HF: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print(f"{'='*60}\n")
        raise   # re-raise to make the step fail

# ----------------------------------------------------------------------
# Macro preprocessing (for Option B)
# ----------------------------------------------------------------------
def preprocess_macro(fred_df: pd.DataFrame, returns_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align macro data with returns index, forward fill missing, and compute rolling z-scores.
    Returns a DataFrame with the same index as returns_index and columns for each macro series.
    """
    if fred_df.empty:
        return pd.DataFrame(index=returns_index)

    # Ensure index is datetime
    fred_df.index = pd.to_datetime(fred_df.index)

    # Reindex to returns_index
    macro_aligned = fred_df.reindex(returns_index, method='ffill').fillna(method='ffill')

    # Rolling z-score (window=252 days)
    macro_z = macro_aligned.copy()
    for col in macro_aligned.columns:
        rolling_mean = macro_aligned[col].rolling(window=252, min_periods=30).mean()
        rolling_std = macro_aligned[col].rolling(window=252, min_periods=30).std()
        macro_z[col] = (macro_aligned[col] - rolling_mean) / rolling_std
        macro_z[col] = macro_z[col].fillna(0.0)  # fill NaN with 0

    return macro_z

# ----------------------------------------------------------------------
# Main processing function (common to both options)
# ----------------------------------------------------------------------
def process_module(
    module: str,
    etfs: List[str],
    regime_ticker: str,
    option: str = "A",
    horizons: List[int] = [21, 63, 126],
    eta: float = 0.5,
    n_paths: int = 2000,
    epochs: int = 200
) -> Dict:
    """Process one module (equity or fixed_income) for a given option."""
    print(f"\n=== Processing {module} (Option {option}) ===")

    prices, fred_df = load_data_from_hf(module)
    if prices.empty:
        return {"error": f"No data for {module}"}

    # Extract regime indicator
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
    current_date = prices.index[-1]  # already a Timestamp
    calibration_results = calibrate_both_windows(
        prices, etfs, regime_analysis["regime_history"], rf_series, current_date
    )

    # For Option B, preprocess macro data
    if option == "B":
        returns = compute_returns(prices, etfs)
        macro_z = preprocess_macro(fred_df, returns.index)
        # Get the current macro values (for the last date in returns)
        if not macro_z.empty and current_date in macro_z.index:
            current_macro = macro_z.loc[current_date].values
        else:
            current_macro = np.zeros(len(macro_z.columns)) if not macro_z.empty else np.array([])
        print(f"  Current macro features shape: {current_macro.shape}")
    else:
        macro_z = None
        current_macro = None

    best_result = None
    best_expected_return = -np.inf
    best_horizon = None
    best_window = None
    best_model = None
    best_weights = None

    # Store candidates for ensemble (only Option B)
    candidate_models = []  # (expected_return, model, weights, horizon, window)

    # Fixed hyperparameters
    hidden_size = 10
    learning_rate = 0.01

    for window_type, params in calibration_results.items():
        if not validate_parameters(params):
            print(f"⚠ Parameters for {window_type} appear invalid (NaNs/zeros). Skipping.")
            continue

        print(f"\n--- {window_type} window ---")

        for T_days in horizons:
            print(f"Training {T_days}-day horizon...")

            n_assets = len(etfs)

            # Generate training data
            try:
                training_data = generate_merton_training_data(
                    params, semi_markov_params, T_days, n_paths, W0=1.0, eta=eta,
                    n_assets=n_assets, macro_data=macro_z, option=option
                )
            except Exception as e:
                print(f"  ✗ Error generating training data: {e}")
                continue

            # Train ANN
            try:
                model = train_ann_for_horizon(
                    training_data, len(etfs), eta,
                    epochs=epochs, learning_rate=learning_rate,
                    hidden_size=hidden_size, input_dim=training_data["X"].shape[1]
                )
            except Exception as e:
                print(f"  ✗ Error training ANN: {e}")
                continue

            # Predict for current state (with macro if needed)
            selected_idx, weights = predict_optimal_etf(
                model, 0.0, 0.0, current_regime,
                macro_features=current_macro
            )

            # Evaluate on validation set (last 1000 samples)
            X_val = training_data["X"][-1000:]
            ann_weights = model.predict(X_val)
            ann_weights = np.nan_to_num(ann_weights, nan=1.0/len(etfs))

            mu = params[current_regime]["mu"]
            mu = np.nan_to_num(mu, nan=0.0)

            expected_returns = ann_weights @ mu
            avg_return = np.mean(expected_returns)
            if np.isnan(avg_return):
                avg_return = 0.0

            print(f"  Expected annualized return: {avg_return:.2%}")

            # For Option B, collect models for ensemble
            if option == "B":
                candidate_models.append((avg_return, model, weights, T_days, window_type))
            else:  # Option A
                if avg_return > best_expected_return:
                    best_expected_return = avg_return
                    best_horizon = T_days
                    best_window = window_type
                    best_model = model
                    best_weights = weights

    # Option B ensemble: take average of top 3 models (or fewer if not enough)
    if option == "B":
        if len(candidate_models) == 0:
            return {"error": "No valid training results for Option B"}
        candidate_models.sort(key=lambda x: x[0], reverse=True)
        top_k = min(3, len(candidate_models))
        ensemble_weights = np.zeros(len(etfs))
        for i in range(top_k):
            ensemble_weights += candidate_models[i][2]
        ensemble_weights /= top_k
        selected_idx = np.argmax(ensemble_weights)

        # For metadata, use the best model's horizon and window
        best_expected_return = candidate_models[0][0]
        best_horizon = candidate_models[0][3]
        best_window = candidate_models[0][4]

        best_result = {
            "selected_etf": etfs[selected_idx],
            "selected_idx": int(selected_idx),
            "weights": {etfs[i]: float(ensemble_weights[i]) for i in range(len(etfs))},
            "regime": "risk-on" if current_regime == 0 else "risk-off",
            "horizon_days": best_horizon,
            "window_type": best_window,
            "expected_return_annualized": float(best_expected_return),
            "ensemble_models_used": top_k,
            "n_parameters": candidate_models[0][1].count_parameters() if candidate_models else 0,
        }
    else:  # Option A
        if best_model is None:
            return {"error": "No valid training result for Option A"}
        selected_idx = np.argmax(best_weights)
        best_result = {
            "selected_etf": etfs[selected_idx],
            "selected_idx": int(selected_idx),
            "weights": {etfs[i]: float(best_weights[i]) for i in range(len(etfs))},
            "regime": "risk-on" if current_regime == 0 else "risk-off",
            "horizon_days": best_horizon,
            "window_type": best_window,
            "expected_return_annualized": float(best_expected_return),
            "model_params": best_model.get_weights(),
            "n_parameters": best_model.count_parameters(),
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

# ----------------------------------------------------------------------
# Helper: validate parameters
# ----------------------------------------------------------------------
def validate_parameters(params: Dict[int, Dict]) -> bool:
    """Check if calibrated parameters are valid (no NaNs/infs)."""
    for regime in [0, 1]:
        Sigma = params[regime]["Sigma"]
        mu = params[regime]["mu"]
        if np.any(~np.isfinite(Sigma)) or np.any(~np.isfinite(mu)):
            return False
        if np.all(Sigma == 0):
            return False
    return True

# ----------------------------------------------------------------------
# Get next trading date
# ----------------------------------------------------------------------
try:
    from pandas_market_calendars import get_calendar
    NYSE = get_calendar("NYSE")
except ImportError:
    NYSE = None

def get_next_trading_date(current_date: pd.Timestamp = None) -> pd.Timestamp:
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
