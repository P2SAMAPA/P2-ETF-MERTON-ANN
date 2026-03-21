"""
calibration.py — P2-ETF-MERTON-ANN
Estimate μ (mean returns), Σ (covariance), and r (risk-free rate) per regime.
Supports both full history (2007+) and rolling 10-year windows.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Literal


def compute_returns(prices: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Compute daily log returns from price data."""
    if isinstance(prices.columns, pd.MultiIndex):
        close_prices = prices.xs('Close', level=1, axis=1) if 'Close' in prices.columns.get_level_values(1) else prices
    else:
        close_dict = {}
        for ticker in tickers:
            possible_names = [f"{ticker}_Close", f"{ticker}_close", ticker]
            for name in possible_names:
                if name in prices.columns:
                    close_dict[ticker] = prices[name]
                    break
        close_prices = pd.DataFrame(close_dict)

    available_tickers = [t for t in tickers if t in close_prices.columns]
    if len(available_tickers) == 0:
        raise ValueError(f"No tickers found in data. Looking for: {tickers[:5]}... Available: {list(prices.columns)[:10]}...")

    close_prices = close_prices[available_tickers]
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.dropna()


def estimate_parameters(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    risk_free_rates: pd.Series = None,
    window_type: Literal["full", "rolling_10y"] = "full",
    current_date: pd.Timestamp = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """Estimate μ, Σ, r for each regime (0=risk-on, 1=risk-off)."""
    data = returns.copy()
    # Align regime_labels to returns index
    regime_aligned = regime_labels.reindex(data.index, fill_value=0)
    data['regime'] = regime_aligned
    if risk_free_rates is not None:
        data['rf'] = risk_free_rates.reindex(data.index, fill_value=0.0)
    else:
        data['rf'] = 0.0
    data = data.dropna()

    if window_type == "rolling_10y" and current_date is not None:
        start_date = current_date - pd.DateOffset(years=10)
        data = data[data.index >= start_date]

    # Debug: print first few rows of data to see regime column
    print(f"  DEBUG (calibration): First 5 regime values in {window_type} window: {data['regime'].iloc[:5].values}")
    print(f"  DEBUG (calibration): Mean daily log returns (first 5 columns): {returns.iloc[:, :5].mean().values}")

    params = {}
    for regime in [0, 1]:
        regime_data = data[data['regime'] == regime]
        if len(regime_data) < 30:
            print(f"  WARNING: Regime {regime} has only {len(regime_data)} observations; using all data.")
            regime_data = data  # fallback to all data

        if len(regime_data) == 0:
            # No data at all – use default
            n_assets = len(returns.columns)
            params[regime] = {
                "mu": np.zeros(n_assets),
                "Sigma": np.eye(n_assets) * 1e-4,
                "r": 0.02,
                "n_obs": 0,
                "tickers": returns.columns.tolist()
            }
            continue

        regime_returns = regime_data.drop(columns=['regime', 'rf'], errors='ignore')

        # Drop tickers with zero variance (constant returns)
        variances = regime_returns.var()
        valid_columns = variances[variances > 1e-12].index.tolist()
        if len(valid_columns) < 2:
            valid_columns = regime_returns.columns.tolist()
        regime_returns = regime_returns[valid_columns]

        mu_daily = regime_returns.mean().values
        mu = mu_daily * 252

        # If mu is all zero, fall back to global mean
        if np.allclose(mu, 0.0):
            print(f"  WARNING: Regime {regime} mu is zero. Falling back to global mean.")
            global_mu_daily = returns.mean().values
            global_mu = global_mu_daily * 252
            if len(global_mu) > len(mu):
                mu = global_mu[:len(mu)]
            elif len(global_mu) < len(mu):
                mu_padded = np.zeros(len(mu))
                mu_padded[:len(global_mu)] = global_mu
                mu = mu_padded
            else:
                mu = global_mu

        Sigma_daily = regime_returns.cov().values
        Sigma = Sigma_daily * 252

        # Ensure positive definiteness
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma = Sigma + np.eye(len(mu)) * 1e-6

        # Risk-free rate
        r = regime_data['rf'].mean() * 252 if 'rf' in regime_data.columns else 0.02

        mu = np.nan_to_num(mu, nan=0.0)
        Sigma = np.nan_to_num(Sigma, nan=0.0)

        params[regime] = {
            "mu": mu,
            "Sigma": Sigma,
            "r": r,
            "n_obs": len(regime_data),
            "tickers": valid_columns
        }

    # Ensure both regimes have the same number of assets
    n_assets = max(len(params[0]["mu"]), len(params[1]["mu"]))
    for regime in [0, 1]:
        current_n = len(params[regime]["mu"])
        if current_n < n_assets:
            mu_padded = np.zeros(n_assets)
            mu_padded[:current_n] = params[regime]["mu"]
            Sigma_padded = np.eye(n_assets) * 1e-4
            Sigma_padded[:current_n, :current_n] = params[regime]["Sigma"]
            params[regime]["mu"] = mu_padded
            params[regime]["Sigma"] = Sigma_padded

    return params


def calibrate_both_windows(
    prices: pd.DataFrame,
    tickers: list,
    regime_labels: pd.Series,
    risk_free_rates: pd.Series = None,
    current_date: pd.Timestamp = None
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """Calibrate for both full history and rolling 10-year window."""
    returns = compute_returns(prices, tickers)
    if current_date is None:
        current_date = returns.index[-1]

    results = {}
    for window_type in ["full", "rolling_10y"]:
        params = estimate_parameters(returns, regime_labels, risk_free_rates, window_type, current_date)
        results[window_type] = params
    return results


def get_risk_free_rate_from_fred(fred_data: pd.DataFrame, date: pd.Timestamp = None) -> pd.Series:
    """Extract daily risk-free rate from FRED DTB3 data."""
    if 'DTB3' not in fred_data.columns:
        if date:
            return pd.Series([0.03], index=[date])
        return pd.Series([0.03], index=pd.date_range(start='2000-01-01', periods=1))
    rf = fred_data['DTB3'].ffill() / 100.0
    rf_daily = rf / 252.0
    return rf_daily


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
