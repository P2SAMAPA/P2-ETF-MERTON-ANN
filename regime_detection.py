"""
regime_detection.py — P2-ETF-MERTON-ANN
Consensus regime detection using three windows (252, 126, 84 days).
Each window: geometric moving average + K-means threshold.
Majority vote decides final regime.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple, Dict, Any, List


def geometric_moving_average(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute geometric moving average: MA_t = exp((1/window) * sum(log(x_{t-i})))
    Handles missing values by forward filling first.
    """
    series_filled = series.fillna(method='ffill').fillna(1.0)
    log_series = np.log(np.clip(series_filled, 1e-6, None))
    ma = log_series.rolling(window=window, min_periods=window).mean()
    return np.exp(ma)


def detect_threshold(series: pd.Series, window: int = 252) -> float:
    """
    Compute threshold for a given window using K-means on log(geometric MA).
    Fallback to median if insufficient data.
    """
    gma = geometric_moving_average(series, window)
    gma_clean = gma.dropna()

    if len(gma_clean) < 10:
        # Fallback: median of available data
        return float(gma_clean.median()) if len(gma_clean) > 0 else 20.0

    log_gma = np.log(gma_clean).values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(log_gma)
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold_log = np.mean(centers)
    return np.exp(threshold_log)


def classify_with_window(series: pd.Series, window: int) -> pd.Series:
    """
    Classify each date using a specific window.
    Returns Series of 0/1 aligned with input index.
    """
    threshold = detect_threshold(series, window)
    gma = geometric_moving_average(series, window)
    # risk-on = 0 (low vol, below threshold), risk-off = 1 (high vol, above threshold)
    regime = (gma > threshold).astype(int)
    # Fill any leading NaNs with 0 (risk-on) as safe default
    return regime.fillna(0)


def full_regime_analysis(regime_indicator: pd.Series, windows: List[int] = None) -> Dict[str, Any]:
    """
    Perform consensus regime detection using multiple windows.
    Majority vote decides final regime for each day.
    Returns the same dictionary as before, but with regime_history as the consensus.
    """
    if windows is None:
        windows = [252, 126, 84]

    # Ensure index is datetime
    regime_indicator = regime_indicator.copy()
    regime_indicator.index = pd.to_datetime(regime_indicator.index)

    # Get classifications from each window
    classifications = []
    for w in windows:
        cls = classify_with_window(regime_indicator, w)
        classifications.append(cls)

    # Combine into a DataFrame
    df = pd.concat(classifications, axis=1)
    df.columns = [f'window_{w}' for w in windows]

    # Majority vote (rows with 0/1, sum > 1 means risk-off majority)
    # Since there are 3 windows, majority is if sum >= 2
    sum_votes = df.sum(axis=1)
    consensus = (sum_votes >= 2).astype(int)

    # Drop any NaNs (shouldn't happen)
    consensus = consensus.dropna()

    # Get current regime (last value)
    current_regime = int(consensus.iloc[-1])

    # Compute semi-Markov parameters on consensus series
    semi_markov_params = estimate_semi_markov_parameters(consensus)

    # Compute threshold for the primary window (252) for backward compatibility (e.g., display)
    primary_threshold = detect_threshold(regime_indicator, windows[0])

    return {
        "threshold": primary_threshold,
        "regime_history": consensus,
        "current_regime": current_regime,
        "current_regime_label": "risk-off" if current_regime == 1 else "risk-on",
        "semi_markov_params": semi_markov_params,
        "windows_used": windows,
        "window_classifications": {f"window_{w}": cls for w, cls in zip(windows, classifications)}
    }


# ----------------------------------------------------------------------
# The following functions are unchanged from the original
# ----------------------------------------------------------------------
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
    """Legacy function for backward compatibility; uses single window."""
    gma = geometric_moving_average(regime_indicator, window)
    current_gma = gma.iloc[-1]
    return 1 if current_gma > threshold else 0
