"""
regime_detection.py — P2-ETF-MERTON-ANN
Consensus regime detection with adaptive window selection.
CHANGE #2: Selects best window based on recent Sharpe ratio performance.
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
    regime = (gma > threshold).astype(int)
    return regime.fillna(0)

# CHANGE #2: Adaptive window selection based on recent performance
def select_best_regime_window(
    regime_indicator: pd.Series,
    returns: pd.Series,
    windows: List[int] = None,
    eval_period: int = 252
) -> Tuple[int, Dict[str, Any]]:
    """
    Select the regime detection window with best recent Sharpe ratio.
    
    Strategy: Go risk-on when regime=0, risk-off (cash/bonds) when regime=1.
    Evaluates which window gave best Sharpe over last eval_period days.
    
    Args:
        regime_indicator: VIX or MOVE series
        returns: Benchmark returns (e.g., SPY for equity, AGG for FI)
        windows: List of windows to test (default [21, 63, 252])
        eval_period: Days to evaluate (default 252 = 1 year)
    
    Returns:
        best_window: The window with highest Sharpe
        window_analysis: Dict with all window performances
    """
    if windows is None:
        windows = [21, 63, 252]  # 1 month, 3 months, 12 months
    
    # Use recent data for evaluation
    recent_returns = returns.iloc[-eval_period:].dropna()
    if len(recent_returns) < 63:  # Need at least 3 months
        return windows[-1], {"fallback": True, "reason": "insufficient_data"}
    
    window_scores = {}
    
    for w in windows:
        # Get regime classification for this window
        regime = classify_with_window(regime_indicator, w)
        regime_aligned = regime.reindex(recent_returns.index, method='ffill')
        
        # Simple strategy: long when risk-on (regime=0), flat when risk-off (regime=1)
        strategy_returns = recent_returns * (1 - regime_aligned)  # 1 when risk-on, 0 when risk-off
        
        # Calculate Sharpe (annualized)
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else -999
        
        # Also calculate win rate (what % of time we were in market)
        time_in_market = (1 - regime_aligned).mean()
        
        window_scores[w] = {
            "sharpe": sharpe,
            "time_in_market": time_in_market,
            "total_return": strategy_returns.sum()
        }
    
    # Select window with best Sharpe, but penalize windows that are always in or always out
    best_sharpe = -999
    best_window = windows[-1]  # Default to longest
    
    for w, scores in window_scores.items():
        # Penalize if always in market (no regime detection value) or always out
        if 0.1 < scores["time_in_market"] < 0.9:
            adjusted_sharpe = scores["sharpe"]
        else:
            adjusted_sharpe = scores["sharpe"] - 0.5  # Penalty
        
        if adjusted_sharpe > best_sharpe:
            best_sharpe = adjusted_sharpe
            best_window = w
    
    return best_window, {
        "selected_window": best_window,
        "all_scores": window_scores,
        "evaluation_period": eval_period
    }


def full_regime_analysis(
    regime_indicator: pd.Series, 
    benchmark_returns: pd.Series = None,
    windows: List[int] = None,
    adaptive: bool = True
) -> Dict[str, Any]:
    """
    Perform regime detection with adaptive window selection.
    
    Args:
        regime_indicator: VIX or MOVE series
        benchmark_returns: Optional returns for adaptive window selection
        windows: List of windows to consider
        adaptive: If True, use best window based on recent performance
    
    Returns:
        Dict with regime analysis including selected window info
    """
    if windows is None:
        windows = [252, 126, 84]  # Keep original as fallback

    regime_indicator = regime_indicator.copy()
    regime_indicator.index = pd.to_datetime(regime_indicator.index)

    # CHANGE #2: Adaptive window selection if benchmark returns provided
    window_analysis = None
    if adaptive and benchmark_returns is not None and len(benchmark_returns) > 252:
        best_window, window_analysis = select_best_regime_window(
            regime_indicator, benchmark_returns, windows
        )
        primary_window = best_window
        print(f"  Adaptive window selected: {primary_window} days")
    else:
        primary_window = windows[0]  # Default to first (252)

    # Get classifications from each window (for consensus)
    classifications = []
    for w in windows:
        cls = classify_with_window(regime_indicator, w)
        classifications.append(cls)

    df = pd.concat(classifications, axis=1)
    df.columns = [f'window_{w}' for w in windows]

    # Majority vote
    sum_votes = df.sum(axis=1)
    consensus = (sum_votes >= 2).astype(int)
    consensus = consensus.dropna()

    current_regime = int(consensus.iloc[-1])

    # Compute semi-Markov parameters on consensus
    semi_markov_params = estimate_semi_markov_parameters(consensus)

    # Use adaptive window's threshold for display
    primary_threshold = detect_threshold(regime_indicator, primary_window)

    result = {
        "threshold": primary_threshold,
        "regime_history": consensus,
        "current_regime": current_regime,
        "current_regime_label": "risk-off" if current_regime == 1 else "risk-on",
        "semi_markov_params": semi_markov_params,
        "windows_used": windows,
        "primary_window": primary_window,
        "window_classifications": {f"window_{w}": cls for w, cls in zip(windows, classifications)}
    }
    
    if window_analysis:
        result["window_selection"] = window_analysis

    return result


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
