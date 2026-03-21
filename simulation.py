"""
simulation.py — P2-ETF-MERTON-ANN
Generate synthetic GBM paths with semi-Markov regime switching.
10,000+ paths for ANN training.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy.linalg import cholesky


def simulate_semi_markov_regime(
    T: int,
    dt: float = 1/252,
    initial_regime: int = 0,
    semi_markov_params: Dict = None
) -> np.ndarray:
    """
    Simulate regime path with semi-Markov (duration-dependent) transitions.
    """
    if semi_markov_params is None:
        semi_markov_params = {
            "p_01": 0.001,
            "p_10": 0.002,
            "mean_duration_on": 1134,
            "mean_duration_off": 252,
        }

    p_01 = semi_markov_params.get("p_01", 0.001)
    p_10 = semi_markov_params.get("p_10", 0.002)

    regime_path = np.zeros(T, dtype=int)
    regime_path[0] = initial_regime

    current_regime = initial_regime
    time_in_regime = 0

    for t in range(1, T):
        if current_regime == 0:
            adjusted_p = min(p_01 * (1 + time_in_regime / 252), 0.5)
            if np.random.random() < adjusted_p:
                current_regime = 1
                time_in_regime = 0
            else:
                time_in_regime += 1
        else:
            adjusted_p = min(p_10 * (1 + time_in_regime / 126), 0.5)
            if np.random.random() < adjusted_p:
                current_regime = 0
                time_in_regime = 0
            else:
                time_in_regime += 1

        regime_path[t] = current_regime

    return regime_path


def simulate_gbm_path(
    T: int,
    mu: np.ndarray,
    Sigma: np.ndarray,
    r: float,
    regime_path: np.ndarray,
    W0: float = 1.0,
    dt: float = 1/252
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate single multi-asset GBM path with regime-switching parameters.
    Returns: (wealth_path, returns_path)
    """
    n_assets = len(mu)
    # Ensure positive-definite covariance
    try:
        L = cholesky(Sigma * dt, lower=True)
    except np.linalg.LinAlgError:
        Sigma_reg = Sigma + np.eye(n_assets) * 1e-6
        L = cholesky(Sigma_reg * dt, lower=True)

    Z = np.random.standard_normal((T, n_assets))
    dW = Z @ L.T

    # Simulate asset prices
    S = np.zeros((T, n_assets))
    S[0] = 1.0
    for t in range(1, T):
        dS = mu * dt + dW[t]
        S[t] = S[t-1] * np.exp(dS)

    # Simple returns for wealth
    returns = np.diff(S, axis=0) / S[:-1]
    returns = np.vstack([np.zeros(n_assets), returns])

    # Wealth evolution with random weights (for diversity)
    wealth = np.zeros(T)
    wealth[0] = W0
    for t in range(1, T):
        weights = np.random.dirichlet(np.ones(n_assets))
        portfolio_return = np.dot(weights, returns[t])
        # Prevent negative wealth
        new_wealth = wealth[t-1] * (1 + portfolio_return)
        wealth[t] = max(new_wealth, 1e-8)   # floor to avoid log(0)

    return wealth, returns


def generate_merton_training_data(
    params: Dict[int, Dict],
    semi_markov_params: Dict,
    T_days: int,
    n_paths: int = 10000,
    W0: float = 1.0,
    eta: float = 0.5,
    n_assets: int = None
) -> Dict[str, np.ndarray]:
    """
    Generate training data using Merton-optimal allocations as targets.
    """
    if n_assets is None:
        n_assets = len(params[0]["mu"])

    if n_assets == 0:
        raise ValueError("n_assets cannot be 0 - check ETF list and calibration")

    print(f"  Generating training data: {n_paths} paths, {T_days} days, {n_assets} assets")

    dt = 1/252
    T_steps = T_days

    X_list = []
    y_list = []

    for _ in range(n_paths):
        initial_regime = np.random.choice([0, 1])
        regime_path = simulate_semi_markov_regime(T_steps, dt, initial_regime, semi_markov_params)

        # Sample points
        n_samples = min(50, T_steps)
        sample_indices = np.linspace(0, T_steps-1, n_samples, dtype=int)

        # Simulate a reference wealth path (only needed for log wealth)
        wealth = W0

        for t in sample_indices:
            current_regime = int(regime_path[t])
            mu = params[current_regime]["mu"]
            Sigma = params[current_regime]["Sigma"]
            r = params[current_regime]["r"]

            # Ensure mu and Sigma are finite and positive definite
            if np.any(~np.isfinite(mu)):
                mu = np.nan_to_num(mu, nan=0.0)
            if np.any(~np.isfinite(Sigma)):
                Sigma = np.eye(n_assets) * 1e-4

            try:
                # Add small ridge to covariance for stability
                Sigma_reg = Sigma + np.eye(n_assets) * 1e-6
                Sigma_inv = np.linalg.inv(Sigma_reg)
                excess_returns = mu - r
                optimal_weights = (1/eta) * Sigma_inv @ excess_returns
                # Project to simplex (softmax)
                optimal_weights = np.exp(optimal_weights - np.max(optimal_weights))
                weights = optimal_weights / (optimal_weights.sum() + 1e-10)
            except np.linalg.LinAlgError:
                weights = np.ones(n_assets) / n_assets
            except Exception:
                weights = np.ones(n_assets) / n_assets

            # Input features
            t_normalized = t / T_steps
            # Clip log wealth to avoid inf
            log_wealth = np.log(max(wealth / W0, 1e-8))

            X_list.append([t_normalized, log_wealth, current_regime])
            y_list.append(weights)

            # Update wealth for next step (simulate with random return)
            random_return = np.random.normal(mu.mean() * dt, 0.01)
            wealth = wealth * (1 + random_return)
            wealth = max(wealth, 1e-8)   # keep positive

    # Convert to arrays and check for NaNs
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        # Fallback: replace NaNs with zeros or equal weights
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=1.0/n_assets)

    return {
        "X": X,
        "y": y,
    }
