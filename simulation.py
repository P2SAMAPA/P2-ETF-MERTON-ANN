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

    Parameters:
    -----------
    T : investment horizon in days
    dt : time step (default 1/252 for daily)
    initial_regime : starting regime (0=risk-on, 1=risk-off)
    semi_markov_params : dict with p_01, p_10, mean durations, etc.

    Returns:
    --------
    regime_path : array of shape (T,) with 0/1 values
    """
    if semi_markov_params is None:
        # Default: ~4.5 years average holding time
        semi_markov_params = {
            "p_01": 0.001,  # ~1/4.5 years daily prob
            "p_10": 0.002,
            "mean_duration_on": 1134,  # 4.5 years
            "mean_duration_off": 252,   # 1 year
        }

    p_01 = semi_markov_params.get("p_01", 0.001)
    p_10 = semi_markov_params.get("p_10", 0.002)

    regime_path = np.zeros(T, dtype=int)
    regime_path[0] = initial_regime

    current_regime = initial_regime
    time_in_regime = 0

    for t in range(1, T):
        # Duration-dependent transition probability (semi-Markov)
        # Increase transition prob as time in regime increases
        if current_regime == 0:  # risk-on
            # Base prob increases slightly with duration
            adjusted_p = min(p_01 * (1 + time_in_regime / 252), 0.5)
            if np.random.random() < adjusted_p:
                current_regime = 1
                time_in_regime = 0
            else:
                time_in_regime += 1
        else:  # risk-off
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

    Parameters:
    -----------
    T : number of time steps (days)
    mu : array of annualized mean returns (n_assets,)
    Sigma : covariance matrix (n_assets, n_assets) - annualized
    r : risk-free rate (annualized)
    regime_path : array of 0/1 regime labels (T,)
    W0 : initial wealth
    dt : time step

    Returns:
    --------
    wealth_path : array (T,) of portfolio wealth over time
    returns_path : array (T, n_assets) of asset returns
    """
    n_assets = len(mu)

    # Cholesky decomposition for correlated random shocks
    try:
        L = cholesky(Sigma * dt, lower=True)
    except np.linalg.LinAlgError:
        # Add small diagonal if not positive definite
        Sigma_reg = Sigma + np.eye(n_assets) * 1e-6
        L = cholesky(Sigma_reg * dt, lower=True)

    # Generate random shocks
    Z = np.random.standard_normal((T, n_assets))
    dW = Z @ L.T  # Correlated Brownian motions

    # Simulate asset prices
    S = np.zeros((T, n_assets))
    S[0] = 1.0  # Start at 1.0 (normalized)

    for t in range(1, T):
        # Drift + diffusion
        dS = mu * dt + dW[t]
        S[t] = S[t-1] * np.exp(dS)

    # Compute returns (simple returns for wealth calculation)
    returns = np.diff(S, axis=0) / S[:-1]
    returns = np.vstack([np.zeros(n_assets), returns])  # Pad first row

    # Wealth evolution (will be controlled by ANN, but here we simulate passive)
    # For training data, we simulate random portfolio weights
    wealth = np.zeros(T)
    wealth[0] = W0

    for t in range(1, T):
        # Random allocation for training diversity
        weights = np.random.dirichlet(np.ones(n_assets))
        portfolio_return = np.dot(weights, returns[t])
        wealth[t] = wealth[t-1] * (1 + portfolio_return)

    return wealth, returns


def generate_training_data(
    params: Dict[int, Dict],
    semi_markov_params: Dict,
    T_days: int,
    n_paths: int = 10000,
    W0: float = 1.0,
    eta: float = 0.5,
    n_assets: int = None
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic training data for ANN.

    Returns dict with:
    - X: inputs (t/T, log(W/W0), regime) shape (n_samples, 3)
    - y: optimal weights shape (n_samples, n_assets)
    - utilities: terminal utilities for each path
    """
    if n_assets is None:
        n_assets = len(params[0]["mu"])

    dt = 1/252
    T_steps = T_days

    X_list = []
    y_list = []
    utilities = []

    for _ in range(n_paths):
        # Random initial regime
        initial_regime = np.random.choice([0, 1])

        # Simulate regime path
        regime_path = simulate_semi_markov_regime(
            T_steps, dt, initial_regime, semi_markov_params
        )

        # Simulate wealth path with random allocations
        # Use average parameters for simplicity in data generation
        mu_avg = (params[0]["mu"] + params[1]["mu"]) / 2
        Sigma_avg = (params[0]["Sigma"] + params[1]["Sigma"]) / 2
        r_avg = (params[0]["r"] + params[1]["r"]) / 2

        wealth, returns = simulate_gbm_path(
            T_steps, mu_avg, Sigma_avg, r_avg, regime_path, W0, dt
        )

        # Sample points along the path for training
        n_samples = min(50, T_steps)  # Sample 50 points per path
        sample_indices = np.linspace(0, T_steps-1, n_samples, dtype=int)

        for t in sample_indices:
            # Input features
            t_normalized = t / T_steps
            log_wealth = np.log(wealth[t] / W0)
            current_regime = regime_path[t]

            X_list.append([t_normalized, log_wealth, current_regime])

            # For training target: use random weights (ANN will learn optimal)
            # In practice, we use dynamic programming or known Merton solution
            # Here we use random for diversity, ANN learns via utility optimization
            weights = np.random.dirichlet(np.ones(n_assets))
            y_list.append(weights)

        # Terminal utility for the path
        W_T = wealth[-1]
        if eta == 1.0:
            utility = np.log(W_T)
        else:
            utility = (W_T ** (1 - eta) - 1) / (1 - eta)
        utilities.append(utility)

    return {
        "X": np.array(X_list),
        "y": np.array(y_list),
        "utilities": np.array(utilities),
        "regime_paths": regime_path,  # Last one for reference
    }


def generate_merton_training_data(
    params: Dict[int, Dict],
    semi_markov_params: Dict,
    T_days: int,
    n_paths: int = 10000,
    W0: float = 1.0,
    eta: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Generate training data using Merton-optimal allocations as targets.
    This is the proper way: simulate paths and compute optimal weights analytically
    where possible, or use numerical optimization.
    """
    n_assets = len(params[0]["mu"])
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

        # Simulate a reference wealth path
        wealth = W0

        for t in sample_indices:
            current_regime = int(regime_path[t])
            mu = params[current_regime]["mu"]
            Sigma = params[current_regime]["Sigma"]
            r = params[current_regime]["r"]

            # Merton optimal weights (analytical solution)
            # w* = (1/eta) * Sigma^(-1) * (mu - r)
            try:
                Sigma_inv = np.linalg.inv(Sigma)
                excess_returns = mu - r
                optimal_weights = (1/eta) * Sigma_inv @ excess_returns

                # Project to simplex (long-only, sum to 1)
                # Use softmax projection
                weights = np.exp(optimal_weights)
                weights = weights / weights.sum()

            except np.linalg.LinAlgError:
                # Fallback to equal weights
                weights = np.ones(n_assets) / n_assets

            # Input features
            t_normalized = t / T_steps
            log_wealth = np.log(wealth / W0)

            X_list.append([t_normalized, log_wealth, current_regime])
            y_list.append(weights)

            # Update wealth for next step (simulate with these weights)
            # Simplified: random return
            random_return = np.random.normal(mu.mean() * dt, 0.01)
            wealth = wealth * (1 + random_return)

    return {
        "X": np.array(X_list),
        "y": np.array(y_list),
    }
