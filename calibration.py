"""
calibration.py — P2-ETF-MERTON-ANN
Estimate μ (mean returns), Σ (covariance), and r (risk-free rate) per regime.
Supports both full history (2007+) and rolling 10-year windows.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Literal


def compute_returns(prices: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Compute daily log returns from price data.
    prices: DataFrame with MultiIndex (Date, Ticker) or columns per ticker
    """
    if isinstance(prices.columns, pd.MultiIndex):
        # Extract close prices from MultiIndex
        close_prices = prices.xs('Close', level=1, axis=1) if 'Close' in prices.columns.get_level_values(1) else prices
    else:
        # Flat column structure - look for _Close suffix
        close_dict = {}
        for ticker in tickers:
            # Try different column naming conventions
            possible_names = [
                f"{ticker}_Close",
                f"{ticker}_close",
                ticker,
            ]
            for name in possible_names:
                if name in prices.columns:
                    close_dict[ticker] = prices[name]
                    break

        if not close_dict:
            # Try to find any column that starts with ticker and contains Close
            for col in prices.columns:
                for ticker in tickers:
                    if col.startswith(ticker) and 'Close' in col:
                        close_dict[ticker] = prices[col]
                        break

        close_prices = pd.DataFrame(close_dict)

    # Ensure we only have requested tickers that were found
    available_tickers = [t for t in tickers if t in close_prices.columns]
    if len(available_tickers) == 0:
        raise ValueError(f"No tickers found in data. Looking for: {tickers[:5]}... Available: {list(prices.columns)[:10]}...")

    close_prices = close_prices[available_tickers]

    # Log returns
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.dropna()


def estimate_parameters(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    risk_free_rates: pd.Series = None,
    window_type: Literal["full", "rolling_10y"] = "full",
    current_date: pd.Timestamp = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Estimate μ, Σ, r for each regime (0=risk-on, 1=risk-off).

    Parameters:
    -----------
    returns : DataFrame of daily log returns (N x M)
    regime_labels : Series of 0/1 regime labels, aligned with returns index
    risk_free_rates : Series of daily risk-free rates (e.g., DTB3/252)
    window_type : "full" for all history, "rolling_10y" for last 10 years
    current_date : reference date for rolling window

    Returns:
    --------
    dict: {0: {"mu": array, "Sigma": array, "r": float}, 1: {...}}
    """
    # Align indices
    data = returns.copy()
    data['regime'] = regime_labels.reindex(data.index)

    if risk_free_rates is not None:
        data['rf'] = risk_free_rates.reindex(data.index)
    else:
        data['rf'] = 0.0

    data = data.dropna()

    # Apply window filter
    if window_type == "rolling_10y" and current_date is not None:
        start_date = current_date - pd.DateOffset(years=10)
        data = data[data.index >= start_date]

    params = {}

    for regime in [0, 1]:
        regime_data = data[data['regime'] == regime]

        if len(regime_data) < 30:  # Minimum observations
            # Fall back to using all data if insufficient regime-specific data
            regime_data = data

        # Extract returns (exclude regime and rf columns)
        regime_returns = regime_data.drop(columns=['regime', 'rf'], errors='ignore')

        # Annualized mean returns (252 trading days)
        mu_daily = regime_returns.mean().values
        mu = mu_daily * 252

        # Annualized covariance
        Sigma_daily = regime_returns.cov().values
        Sigma = Sigma_daily * 252

        # Risk-free rate (annualized)
        r = regime_data['rf'].mean() * 252 if 'rf' in regime_data.columns else 0.02

        params[regime] = {
            "mu": mu,
            "Sigma": Sigma,
            "r": r,
            "n_obs": len(regime_data),
            "tickers": regime_returns.columns.tolist()
        }

    return params


def calibrate_both_windows(
    prices: pd.DataFrame,
    tickers: list,
    regime_labels: pd.Series,
    risk_free_rates: pd.Series = None,
    current_date: pd.Timestamp = None
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """
    Calibrate parameters for both full history and rolling 10-year window.

    Returns:
    --------
    {"full": {0: params, 1: params}, "rolling_10y": {0: params, 1: params}}
    """
    returns = compute_returns(prices, tickers)

    if current_date is None:
        current_date = returns.index[-1]

    results = {}

    for window_type in ["full", "rolling_10y"]:
        params = estimate_parameters(
            returns, regime_labels, risk_free_rates, window_type, current_date
        )
        results[window_type] = params

    return results


def get_risk_free_rate_from_fred(fred_data: pd.DataFrame, date: pd.Timestamp = None) -> pd.Series:
    """
    Extract daily risk-free rate from FRED DTB3 data.
    Returns series of daily rates (as decimals, e.g., 0.05 for 5%).
    """
    if 'DTB3' not in fred_data.columns:
        # Return default if not available
        if date:
            return pd.Series([0.03], index=[date])
        return pd.Series([0.03], index=pd.date_range(start='2000-01-01', periods=1))

    rf = fred_data['DTB3'].ffill() / 100.0  # Convert from percent to decimal
    rf_daily = rf / 252.0  # Convert annual to daily

    return rf_daily


def validate_parameters(params: Dict[int, Dict]) -> bool:
    """
    Check if calibrated parameters are valid (positive definite covariance, etc.)
    """
    for regime in [0, 1]:
        Sigma = params[regime]["Sigma"]

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(Sigma)
        if np.any(eigenvalues < -1e-8):
            return False

        # Check for NaN/Inf
        if not np.all(np.isfinite(Sigma)):
            return False
        if not np.all(np.isfinite(params[regime]["mu"])):
            return False

    return True
