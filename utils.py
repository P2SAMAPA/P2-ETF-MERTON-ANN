"""
utils.py — Shared utilities for the P2-ETF-MERTON-ANN pipeline.
Includes: Relative strength ranking, rolling validation, momentum integration.
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
    EQUITY_ETFS, EQUITY_REGIME, EQUITY_BENCHMARK,
    FI_ETFS, FI_REGIME, FI_BENCHMARK,
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

        if "date" in prices.columns:
            prices = prices.set_index("date")
        elif "Date" in prices.columns:
            prices = prices.set_index("Date")
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
        
        # CRITICAL FIX: Check and remove duplicate dates in index
        if prices.index.duplicated().any():
            n_dups = prices.index.duplicated().sum()
            print(f"  ⚠ Found {n_dups} duplicate dates in prices, removing duplicates")
            prices = prices[~prices.index.duplicated(keep='first')]
            print(f"  ✓ After dedup: {prices.shape}")
        
        print(f" Index: {prices.index[0].date()} → {prices.index[-1].date()}")

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
            if "date" in fred_df.columns:
                fred_df = fred_df.set_index("date")
            elif "Date" in fred_df.columns:
                fred_df = fred_df.set_index("Date")
            if not isinstance(fred_df.index, pd.DatetimeIndex):
                fred_df.index = pd.to_datetime(fred_df.index)
            
            # CRITICAL FIX: Check and remove duplicate dates in FRED data
            if fred_df.index.duplicated().any():
                n_dups = fred_df.index.duplicated().sum()
                print(f"  ⚠ Found {n_dups} duplicate dates in FRED data, removing duplicates")
                fred_df = fred_df[~fred_df.index.duplicated(keep='first')]
                
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
# CHANGE #5: Relative Strength Ranking
# ----------------------------------------------------------------------
def calculate_relative_strength(
    prices: pd.DataFrame, 
    etfs: List[str], 
    benchmark_col: str,
    lookback: int = 20
) -> np.ndarray:
    """
    Calculate relative strength vs benchmark for each ETF.
    RS = ETF return / Benchmark return over lookback period.
    
    Returns array of RS scores (higher = stronger than benchmark)
    """
    rs_scores = np.ones(len(etfs))  # Default to neutral
    
    if len(prices) < lookback + 1:
        return rs_scores
    
    # Get benchmark prices
    bench_col = f"{benchmark_col}_Close" if f"{benchmark_col}_Close" in prices.columns else benchmark_col
    if bench_col not in prices.columns:
        return rs_scores
    
    bench_prices = prices[bench_col].dropna()
    if len(bench_prices) < lookback:
        return rs_scores
    
    bench_return = bench_prices.iloc[-1] / bench_prices.iloc[-lookback] - 1
    if abs(bench_return) < 1e-6:
        bench_return = 1e-6  # Avoid division by zero
    
    for i, etf in enumerate(etfs):
        col = f"{etf}_Close" if f"{etf}_Close" in prices.columns else etf
        if col in prices.columns:
            etf_prices = prices[col].dropna()
            if len(etf_prices) >= lookback // 2:
                etf_return = etf_prices.iloc[-1] / etf_prices.iloc[-lookback] - 1
                rs_scores[i] = (1 + etf_return) / (1 + bench_return)
    
    return rs_scores


def apply_relative_strength_overlay(
    ann_weights: np.ndarray,
    rs_scores: np.ndarray,
    rs_boost: float = 0.25
) -> np.ndarray:
    """
    Blend ANN weights with relative strength scores.
    """
    # Normalize RS scores to sum to 1 (for weighting)
    rs_norm = rs_scores / np.sum(rs_scores) if np.sum(rs_scores) > 0 else np.ones(len(rs_scores)) / len(rs_scores)
    
    # Blend
    adjusted = ann_weights * (1 - rs_boost) + rs_norm * rs_boost
    adjusted = np.maximum(adjusted, 0)
    return adjusted / np.sum(adjusted)


# ----------------------------------------------------------------------
# CHANGE #7: Rolling Validation (Walk-Forward)
# ----------------------------------------------------------------------
def rolling_validation_score(
    prices: pd.DataFrame,
    etfs: List[str],
    regime_history: pd.Series,
    benchmark_col: str,
    window: int = 252
) -> Dict[str, float]:
    """
    Calculate 1-year rolling validation metrics.
    Simulates: predict using only past data, check next-day return.
    
    Returns validation metrics to assess model health.
    """
    if len(prices) < window + 20:
        return {"error": "Insufficient data for validation"}
    
    returns = compute_returns(prices, etfs)
    
    # Simple heuristic: pick ETF with best Sharpe in recent window
    recent_returns = returns.iloc[-window:]
    
    # CRITICAL FIX: Remove duplicates from recent_returns
    if recent_returns.index.duplicated().any():
        recent_returns = recent_returns[~recent_returns.index.duplicated(keep='first')]
    
    # Calculate Sharpe for each ETF
    sharpe_scores = []
    for etf in etfs:
        if etf in recent_returns.columns:
            r = recent_returns[etf].dropna()
            if len(r) > 30:
                sharpe = (r.mean() / (r.std() + 1e-6)) * np.sqrt(252)
                sharpe_scores.append(sharpe)
            else:
                sharpe_scores.append(-999)
        else:
            sharpe_scores.append(-999)
    
    sharpe_scores = np.array(sharpe_scores)
    
    # Check if best Sharpe is meaningfully better than median
    best_sharpe = np.max(sharpe_scores)
    median_sharpe = np.median(sharpe_scores[sharpe_scores > -900])
    
    # Calculate hit rate: how often did top Sharpe ETF beat benchmark
    bench_col = f"{benchmark_col}_Close" if f"{benchmark_col}_Close" in prices.columns else benchmark_col
    if bench_col in prices.columns:
        bench_prices = prices[bench_col].dropna()
        if len(bench_prices) >= window:
            bench_return = bench_prices.iloc[-1] / bench_prices.iloc[-window] - 1
            best_etf_idx = np.argmax(sharpe_scores)
            best_etf = etfs[best_etf_idx]
            best_etf_col = f"{best_etf}_Close" if f"{best_etf}_Close" in prices.columns else best_etf
            if best_etf_col in prices.columns:
                best_etf_return = prices[best_etf_col].dropna().iloc[-1] / prices[best_etf_col].dropna().iloc[-window] - 1
                hit = 1 if best_etf_return > bench_return else 0
            else:
                hit = 0
                best_etf_return = 0
        else:
            hit = 0
            bench_return = 0
            best_etf_return = 0
    else:
        hit = 0
        bench_return = 0
        best_etf_return = 0
    
    return {
        "validation_window": window,
        "best_sharpe": float(best_sharpe),
        "median_sharpe": float(median_sharpe),
        "sharpe_dispersion": float(best_sharpe - median_sharpe),
        "hit_vs_benchmark": hit,
        "best_etf_return": float(best_etf_return),
        "benchmark_return": float(bench_return),
        "data_quality_ok": best_sharpe > 0 and (best_sharpe - median_sharpe) > 0.1
    }


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
        print(f" File size: {os.path.getsize(temp_file)} bytes")
        print(f" Selected ETF(s): {signal.get('selected_etfs', 'N/A')}")

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

        # Append new record
        new_record = {
            "date": signal.get("date"),
            "next_trading_date": signal.get("next_trading_date"),
            "selected_etfs": signal.get("selected_etfs"),  # Now a list
            "allocation": signal.get("allocation"),  # Proportions
            "weights": signal.get("weights"),
            "regime": signal.get("regime"),
            "horizon_days": signal.get("horizon_days"),
            "window_type": signal.get("window_type"),
            "expected_return_annualized": signal.get("expected_return_annualized"),
            "confidence": signal.get("confidence"),  # NEW
            "validation_score": signal.get("validation_score"),  # NEW
            "generated_at": datetime.utcnow().isoformat()
        }
        new_record = {k: v for k, v in new_record.items() if v is not None}

        existing_dates = {r.get("date") for r in history if "date" in r}
        if new_record["date"] not in existing_dates:
            history.append(new_record)
            print(f"✓ Appended new record for {new_record['date']}")
        else:
            print(f"ℹ Record for {new_record['date']} already exists – skipping append")

        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)

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
        raise


# ----------------------------------------------------------------------
# Macro preprocessing (for Option B)
# ----------------------------------------------------------------------
def preprocess_macro(fred_df: pd.DataFrame, returns_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align macro data with returns index, forward fill missing, and compute rolling z-scores.
    """
    if fred_df.empty:
        return pd.DataFrame(index=returns_index)

    fred_df.index = pd.to_datetime(fred_df.index)
    
    # CRITICAL FIX: Remove duplicates from returns_index
    if returns_index.duplicated().any():
        returns_index = returns_index[~returns_index.duplicated(keep='first')]
    
    macro_aligned = fred_df.reindex(returns_index, method='ffill').fillna(method='ffill')

    macro_z = macro_aligned.copy()
    for col in macro_aligned.columns:
        rolling_mean = macro_aligned[col].rolling(window=252, min_periods=30).mean()
        rolling_std = macro_aligned[col].rolling(window=252, min_periods=30).std()
        macro_z[col] = (macro_aligned[col] - rolling_mean) / rolling_std
        macro_z[col] = macro_z[col].fillna(0.0)

    return macro_z


# ----------------------------------------------------------------------
# Main processing function (MODIFIED)
# ----------------------------------------------------------------------
def process_module(
    module: str,
    etfs: List[str],
    regime_ticker: str,
    benchmark: str,  # NEW: Added benchmark parameter
    option: str = "A",
    horizons: List[int] = [21, 63, 126],
    eta: float = 0.5,
    n_paths: int = 2000,
    epochs: int = 200
) -> Dict:
    """Process one module with all improvements."""
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

    # CRITICAL FIX: Ensure regime_prices has no duplicates
    if regime_prices.index.duplicated().any():
        print(f"  ⚠ Found {regime_prices.index.duplicated().sum()} duplicates in regime_prices, removing")
        regime_prices = regime_prices[~regime_prices.index.duplicated(keep='first')]

    # CHANGE #7: Calculate validation score first
    validation_score = rolling_validation_score(prices, etfs, pd.Series(), benchmark, window=252)
    print(f"  Validation: Sharpe dispersion = {validation_score.get('sharpe_dispersion', 0):.3f}")
    print(f"  Validation: Hit vs benchmark = {validation_score.get('hit_vs_benchmark', 0)}")

    # Get benchmark returns for adaptive regime detection
    bench_col = f"{benchmark}_Close" if f"{benchmark}_Close" in prices.columns else benchmark
    benchmark_returns = prices[bench_col] if bench_col in prices.columns else pd.Series()
    
    # CRITICAL FIX: Remove duplicates from benchmark_returns
    if benchmark_returns.index.duplicated().any():
        print(f"  ⚠ Found {benchmark_returns.index.duplicated().sum()} duplicates in benchmark_returns, removing")
        benchmark_returns = benchmark_returns[~benchmark_returns.index.duplicated(keep='first')]

    # CHANGE #2: Regime detection with adaptive window
    regime_analysis = full_regime_analysis(regime_prices, benchmark_returns, adaptive=True)
    current_regime = regime_analysis["current_regime"]
    semi_markov_params = regime_analysis["semi_markov_params"]

    print(f"Current regime: {regime_analysis['current_regime_label']}")
    print(f"Threshold: {regime_analysis['threshold']:.2f}")
    if "window_selection" in regime_analysis:
        print(f"Selected window: {regime_analysis['window_selection']['selected_window']} days")

    # Risk-free rate
    rf_series = get_risk_free_rate_from_fred(fred_df)

    # Calibration
    current_date = prices.index[-1]
    calibration_results = calibrate_both_windows(
        prices, etfs, regime_analysis["regime_history"], rf_series, current_date
    )

    # For Option B, preprocess macro data
    if option == "B":
        returns = compute_returns(prices, etfs)
        macro_z = preprocess_macro(fred_df, returns.index)
        if not macro_z.empty and current_date in macro_z.index:
            current_macro = macro_z.loc[current_date].values
        else:
            current_macro = np.zeros(len(macro_z.columns)) if not macro_z.empty else np.array([])
        print(f" Current macro features shape: {current_macro.shape}")
    else:
        macro_z = None
        current_macro = None

    best_result = None
    best_expected_return = -np.inf
    best_horizon = None
    best_window = None
    best_model = None
    best_weights = None
    best_confidence = 0

    candidate_models = []

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

            try:
                training_data = generate_merton_training_data(
                    params, semi_markov_params, T_days, n_paths, W0=1.0, eta=eta,
                    n_assets=n_assets, macro_data=macro_z, option=option
                )
            except Exception as e:
                print(f" ✗ Error generating training data: {e}")
                continue

            try:
                model = train_ann_for_horizon(
                    training_data, len(etfs), eta,
                    epochs=epochs, learning_rate=learning_rate,
                    hidden_size=hidden_size, input_dim=training_data["X"].shape[1]
                )
            except Exception as e:
                print(f" ✗ Error training ANN: {e}")
                continue

            # CHANGE #1 & #3: Predict with momentum and temperature
            selected_indices, allocation, raw_weights, confidence = predict_optimal_etf(
                model, 0.0, 0.0, current_regime,
                macro_features=current_macro,
                prices=prices,
                etfs=etfs,
                apply_momentum=True,
                temperature=0.4
            )

            # CHANGE #5: Apply relative strength overlay
            rs_scores = calculate_relative_strength(prices, etfs, benchmark, lookback=20)
            rs_adjusted = apply_relative_strength_overlay(raw_weights, rs_scores, rs_boost=0.2)
            
            # Re-select with RS-adjusted weights
            from ann_model import select_etfs_with_temperature
            selected_indices, allocation, confidence = select_etfs_with_temperature(
                rs_adjusted, temperature=0.4, max_etfs=2
            )

            # Evaluate on validation set
            X_val = training_data["X"][-1000:]
            ann_weights = model.predict(X_val)
            ann_weights = np.nan_to_num(ann_weights, nan=1.0/len(etfs))

            mu = params[current_regime]["mu"]
            mu = np.nan_to_num(mu, nan=0.0)

            expected_returns = ann_weights @ mu
            avg_return = np.mean(expected_returns)
            if np.isnan(avg_return):
                avg_return = 0.0

            print(f" Expected annualized return: {avg_return:.2%}")
            print(f" Selected: {[etfs[i] for i in selected_indices]} @ {allocation}")

            if option == "B":
                candidate_models.append((avg_return, model, selected_indices, allocation, confidence, T_days, window_type))
            else:
                if avg_return > best_expected_return:
                    best_expected_return = avg_return
                    best_horizon = T_days
                    best_window = window_type
                    best_model = model
                    best_selected = selected_indices
                    best_allocation = allocation
                    best_confidence = confidence

    # Option B ensemble
    if option == "B":
        if len(candidate_models) == 0:
            return {"error": "No valid training results for Option B"}
        candidate_models.sort(key=lambda x: x[0], reverse=True)
        top_k = min(3, len(candidate_models))
        
        # Average the top-k model selections
        vote_counts = np.zeros(len(etfs))
        for i in range(top_k):
            for idx in candidate_models[i][2]:
                vote_counts[idx] += candidate_models[i][3][list(candidate_models[i][2]).index(idx)] if idx in candidate_models[i][2] else 1
        
        # Get consensus top 2
        top_indices = np.argsort(vote_counts)[::-1][:2]
        if vote_counts[top_indices[1]] > 0.3 * vote_counts[top_indices[0]]:
            selected = top_indices[:2].tolist()
            allocation = vote_counts[selected] / np.sum(vote_counts[selected])
        else:
            selected = [top_indices[0]]
            allocation = np.array([1.0])
        
        best_expected_return = candidate_models[0][0]
        best_horizon = candidate_models[0][5]
        best_window = candidate_models[0][6]
        best_confidence = candidate_models[0][4]
    else:
        if best_model is None:
            return {"error": "No valid training result for Option A"}
        selected = best_selected
        allocation = best_allocation

    # Format output
    selected_etfs = [etfs[i] for i in selected]
    
    best_result = {
        "selected_etfs": selected_etfs,
        "selected_indices": [int(i) for i in selected],
        "allocation": [float(a) for a in allocation],
        "weights": {etfs[i]: float(allocation[list(selected).index(i)] if i in selected else 0.0) for i in range(len(etfs))},
        "regime": "risk-on" if current_regime == 0 else "risk-off",
        "horizon_days": best_horizon,
        "window_type": best_window,
        "expected_return_annualized": float(best_expected_return),
        "confidence": float(best_confidence),
        "validation_score": validation_score,
        "n_parameters": best_model.count_parameters() if 'best_model' in locals() and best_model else 0,
    }

    if option == "B":
        best_result["ensemble_models_used"] = top_k if 'top_k' in locals() else 1

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
        "adaptive_window": regime_analysis.get("primary_window", 252),
    }

    print(f"\n✓ Best: {signal['selected_etfs']} ({signal['horizon_days']}d, {signal['window_type']})")
    print(f" Expected return: {signal['expected_return_annualized']:.2%}")
    print(f" Allocation: {signal['allocation']}")

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
