"""
regime_detection.py — P2-ETF-MERTON-ANN
VIX/MOVE geometric moving average + K-means clustering for regime detection.
Also computes transition probabilities from historical regime durations (semi-Markov property).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple, Dict, Any


def geometric_moving_average(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute geometric moving average: MA_t = exp((1/window) * sum(log(x_{t-i})))
    Handles missing values by forward filling first.
    """
    # Forward fill to handle leading NaNs, then clip to avoid log(0)
    series_filled = series.fillna(method='ffill').fillna(1.0)
    log_series = np.log(np.clip(series_filled, 1e-6, None))
    ma = log_series.rolling(window=window, min_periods=window).mean()
    return np.exp(ma)


def detect_regime_threshold(regime_indicator: pd.Series, window: int = 252) -> float:
    """Compute K-means clustering (K=2) on log-transformed geometric MA."""
    gma = geometric_moving_average(regime_indicator, window)
    gma_clean = gma.dropna()

    if len(gma_clean) < 10:
        print(f"⚠ Insufficient data for regime detection: {len(gma_clean)} points, using median as threshold")
        return float(gma_clean.median()) if len(gma_clean) > 0 else 20.0

    log_gma = np.log(gma_clean).values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(log_gma)

    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold_log = np.mean(centers)
    threshold = np.exp(threshold_log)

    return threshold


def classify_regime(regime_indicator: pd.Series, threshold: float, window: int = 252) -> pd.Series:
    """
    Classify each date as risk-on (0) or risk-off (1) based on threshold.
    Returns a Series with the same index as regime_indicator.
    """
    gma = geometric_moving_average(regime_indicator, window)
    # risk-on = 0 (low vol, below threshold), risk-off = 1 (high vol, above threshold)
    regime = (gma > threshold).astype(int)
    # Fill any NaNs that may appear due to insufficient data for rolling
    regime = regime.fillna(0)  # default to risk-on
    return regime


def compute_regime_durations(regime_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Compute durations of consecutive regime periods."""
    regime_series = regime_series.dropna()
    if len(regime_series) == 0:
        return pd.Series([252]), pd.Series([126])
    regime_changes = regime_series.diff().ne(0).cumsum()
    risk_on_durations = []
    risk_off_durations = []

    for regime_id, group in regime_series.groupby(regime_changes):
        duration = len(group)
        regime_value = group.iloc[0]
        if regime_value == 0:
            risk_on_durations.append(duration)
        else:
            risk_off_durations.append(duration)

    return pd.Series(risk_on_durations), pd.Series(risk_off_durations)


def estimate_semi_markov_parameters(regime_series: pd.Series) -> Dict[str, Any]:
    """Estimate semi-Markov transition parameters from historical regime durations."""
    risk_on_dur, risk_off_dur = compute_regime_durations(regime_series)

    mean_duration_on = risk_on_dur.mean() if len(risk_on_dur) > 0 else 252
    mean_duration_off = risk_off_dur.mean() if len(risk_off_dur) > 0 else 126

    lambda_on = 1.0 / mean_duration_on if mean_duration_on > 0 else 0.004
    lambda_off = 1.0 / mean_duration_off if mean_duration_off > 0 else 0.008

    p_01 = min(lambda_on, 0.5)
    p_10 = min(lambda_off, 0.5)

    return {
        "p_01": p_01,
        "p_10": p_10,
        "mean_duration_on": mean_duration_on,
        "mean_duration_off": mean_duration_off,
        "lambda_on": lambda_on,
        "lambda_off": lambda_off,
        "risk_on_durations": risk_on_dur.tolist(),
        "risk_off_durations": risk_off_dur.tolist(),
    }


def get_current_regime(regime_indicator: pd.Series, threshold: float, window: int = 252) -> int:
    """Get the most recent regime classification."""
    gma = geometric_moving_average(regime_indicator, window)
    current_gma = gma.iloc[-1]
    return 1 if current_gma > threshold else 0


def full_regime_analysis(regime_indicator: pd.Series, window: int = 252) -> Dict[str, Any]:
    """
    Complete regime analysis: threshold, full history, current regime, and semi-Markov params.
    The returned regime_history is a Series with the same index as regime_indicator.
    """
    # Ensure we have a clean series with datetime index
    regime_indicator = regime_indicator.copy()
    regime_indicator.index = pd.to_datetime(regime_indicator.index)
    threshold = detect_regime_threshold(regime_indicator, window)
    regime_history = classify_regime(regime_indicator, threshold, window)

    # Drop any leading NaN that might still exist
    regime_history = regime_history.dropna()
    if len(regime_history) == 0:
        # Fallback: assume risk-on for all
        regime_history = pd.Series(0, index=regime_indicator.index)

    current_regime = regime_history.iloc[-1]
    semi_markov_params = estimate_semi_markov_parameters(regime_history)

    return {
        "threshold": threshold,
        "regime_history": regime_history,
        "current_regime": int(current_regime),
        "current_regime_label": "risk-off" if current_regime == 1 else "risk-on",
        "semi_markov_params": semi_markov_params,
    }
