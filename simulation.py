"""
simulation.py — P2-ETF-MERTON-ANN
Generate synthetic GBM paths with semi-Markov regime switching.
Vectorized version for speed.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy.linalg import cholesky


def simulate_semi_markov_regime(
    T: int,
    n_paths: int,
    dt: float = 1/252,
    initial_regime: int = 0,
    semi_markov_params: Dict = None
) -> np.ndarray:
    """
    Simulate regime paths for all paths simultaneously (vectorized).
    Returns array of shape (n_paths, T) with 0/1 values.
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

    # Initialize
    regime_paths = np.zeros((n_paths, T), dtype=int)
    # Random initial regime for each path
    regime_paths[:, 0] = np.random.choice([0, 1], size=n_paths)

    # Current regime and time in regime for each path
    current_regime = regime_paths[:, 0].copy()
    time_in_regime = np.zeros(n_paths, dtype=int)

    for t in range(1, T):
        # For risk-on (0) and risk-off (1) separately
        for reg_val in [0, 1]:
            mask = (current_regime == reg_val)
            if not np.any(mask):
                continue

            # Adjusted transition probability based on duration
            if reg_val == 0:
                base_p = p_01
                adj = 1 + time_in_regime[mask] / 252
            else:
                base_p = p_10
                adj = 1 + time_in_regime[mask] / 126
            p_trans = np.minimum(base_p * adj, 0.5)

            # Determine which paths switch
            switch = np.random.random(size=np.sum(mask)) < p_trans

            # Update regimes and time counters
            new_regime = np.where(switch, 1 - reg_val, reg_val)
            regime_paths[mask, t] = new_regime
            current_regime[mask] = new_regime
            # Reset time for those that switched
            time_in_regime[mask] = np.where(switch, 0, time_in_regime[mask] + 1)

    return regime_paths


def simulate_gbm_paths_vectorized(
    T: int,
    n_paths: int,
    mu: np.ndarray,
    Sigma: np.ndarray,
    r: float,
    regime_paths: np.ndarray,
    W0: float = 1.0,
    dt: float = 1/252
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate multi-asset GBM paths for all paths simultaneously.
    Returns (wealth_paths, returns_paths) where each has shape (n_paths, T).
    """
    n_assets = len(mu)

    # Cholesky of covariance (annual) scaled by dt
    try:
        L = cholesky(Sigma * dt, lower=True)
    except np.linalg.LinAlgError:
        Sigma_reg = Sigma + np.eye(n_assets) * 1e-6
        L = cholesky(Sigma_reg * dt, lower=True)

    # Random shocks (n_paths, T, n_assets)
    Z = np.random.standard_normal((n_paths, T, n_assets))
    dW = Z @ L.T  # (n_paths, T, n_assets)

    # Asset prices (n_paths, T, n_assets)
    S = np.zeros((n_paths, T, n_assets))
    S[:, 0] = 1.0

    # Simple returns (n_paths, T, n_assets)
    returns = np.zeros((n_paths, T, n_assets))

    for t in range(1, T):
        dS = mu * dt + dW[:, t]
        S[:, t] = S[:, t-1] * np.exp(dS)
        returns[:, t] = (S[:, t] - S[:, t-1]) / S[:, t-1]

    # Wealth evolution with random weights (for diversity)
    wealth = np.zeros((n_paths, T))
    wealth[:, 0] = W0

    for t in range(1, T):
        # Random Dirichlet weights for each path (diversified training)
        weights = np.random.dirichlet(np.ones(n_assets), size=n_paths)
        portfolio_return = np.sum(weights * returns[:, t], axis=1)
        new_wealth = wealth[:, t-1] * (1 + portfolio_return)
        wealth[:, t] = np.maximum(new_wealth, 1e-8)   # floor to avoid log(0)

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
    Vectorized version.
    """
    if n_assets is None:
        n_assets = len(params[0]["mu"])

    if n_assets == 0:
        raise ValueError("n_assets cannot be 0 - check ETF list and calibration")

    print(f"  Generating training data: {n_paths} paths, {T_days} days, {n_assets} assets")

    dt = 1/252
    T_steps = T_days

    # Simulate regime paths for all paths at once
    initial_regime = np.random.choice([0, 1], size=n_paths)
    regime_paths = simulate_semi_markov_regime(
        T_steps, n_paths, dt, initial_regime, semi_markov_params
    )

    # We need to generate wealth and returns per path. We'll sample a subset of points for training.
    # Instead of simulating full asset paths for each sample point, we'll simulate full paths once
    # and then sample points along them. But the optimal weights at each time depend on the current
    # regime's mu and Sigma, which vary over time. To keep it simple, we'll simulate one full
    # asset path per path and compute the Merton weights at each time step.

    # Pre-allocate arrays for features and targets
    n_samples_per_path = min(50, T_steps)
    total_samples = n_paths * n_samples_per_path

    X = np.zeros((total_samples, 3), dtype=np.float32)
    y = np.zeros((total_samples, n_assets), dtype=np.float32)

    # For each path, we need to simulate asset returns under the correct regime at each time.
    # This is more complex. We'll do a loop over paths but with vectorized internal steps.
    # Given n_paths is 10k, a loop over paths is still okay if the per-path operations are numpy-vectorized.
    # But we can also simulate regime-specific asset paths for each path. We'll do a hybrid: for each path,
    # generate asset returns using the regime path, compute Merton weights, and sample.

    # For speed, we'll use a simple approach: generate a single set of returns under average parameters,
    # and then compute Merton weights using the actual regime at each sampled point. This approximates
    # the true dynamics while being fast.

    # Average parameters across regimes
    mu_avg = (params[0]["mu"] + params[1]["mu"]) / 2
    Sigma_avg = (params[0]["Sigma"] + params[1]["Sigma"]) / 2
    r_avg = (params[0]["r"] + params[1]["r"]) / 2

    # Simulate one set of asset returns for all paths (vectorized over paths)
    # We'll use the average parameters to generate a single asset price path per path
    # (same drift for all paths, different shocks per path)
    Z = np.random.standard_normal((n_paths, T_steps, n_assets))
    # Need to simulate asset prices using regime-specific drifts? This is complicated.
    # Simpler: use average drifts for all paths. The Merton weights computed at each step
    # will use the actual regime's mu and Sigma, which may not match the realized returns,
    # but that's okay because the ANN learns to map state to optimal weights, not to predict returns.

    # For the training target (optimal weights), we only need the current state's mu and Sigma,
    # not the actual realized returns. The realized returns are used only to evolve wealth.
    # But we can simulate a simple wealth process using a random return drawn from the current regime's distribution.

    # I'll implement a vectorized simulation where each path's wealth evolves using random returns
    # drawn from a normal distribution with the current regime's mean return and a fixed variance.

    # Pre-allocate arrays for sampling
    sample_indices = np.linspace(0, T_steps-1, n_samples_per_path, dtype=int)

    # Initialize wealth for all paths
    wealth = np.full(n_paths, W0, dtype=np.float32)

    # For each sample time step, we compute the state and target for all paths
    # that are still alive. This is a loop over sample points, but it's efficient.
    for idx, t in enumerate(sample_indices):
        # Current regime for each path at time t
        current_regime = regime_paths[:, t].astype(int)

        # Compute Merton optimal weights for each path based on its regime
        # We need to handle each regime separately for vectorization
        # We'll loop over regimes (2 regimes) and fill y for those paths
        for reg in [0, 1]:
            mask = (current_regime == reg)
            if not np.any(mask):
                continue
            mu_reg = params[reg]["mu"]
            Sigma_reg = params[reg]["Sigma"]
            r_reg = params[reg]["r"]

            # Regularized inverse
            try:
                Sigma_reg_inv = np.linalg.inv(Sigma_reg + np.eye(n_assets) * 1e-6)
            except np.linalg.LinAlgError:
                # Fallback to equal weights
                weights = np.ones((np.sum(mask), n_assets)) / n_assets
                y[idx*n_paths + np.where(mask)[0], :] = weights
                continue

            excess = mu_reg - r_reg
            optimal = (1/eta) * Sigma_reg_inv @ excess  # shape (n_assets,)
            # Softmax projection (ensure positivity)
            optimal = np.exp(optimal - np.max(optimal))
            weights = optimal / (optimal.sum() + 1e-10)
            # Broadcast to all paths in this regime
            y[idx*n_paths + np.where(mask)[0], :] = weights

        # Input features
        t_normalized = t / T_steps
        log_wealth = np.log(np.maximum(wealth / W0, 1e-8))
        X[idx*n_paths:(idx+1)*n_paths, 0] = t_normalized
        X[idx*n_paths:(idx+1)*n_paths, 1] = log_wealth
        X[idx*n_paths:(idx+1)*n_paths, 2] = current_regime

        # Update wealth for next step (using random return from the current regime)
        # For simplicity, we draw a random return from the regime's average drift
        # and a fixed volatility to avoid complexity.
        for reg in [0, 1]:
            mask = (current_regime == reg)
            if not np.any(mask):
                continue
            mu_reg = params[reg]["mu"]
            # Use average of mu_reg as drift (could also use asset-specific, but simpler)
            drift = np.mean(mu_reg) * dt
            # Add random shock
            shock = np.random.normal(0, 0.02, size=np.sum(mask))  # 2% daily vol approx
            ret = drift + shock
            wealth[mask] = wealth[mask] * (1 + ret)
        wealth = np.maximum(wealth, 1e-8)

    return {
        "X": X,
        "y": y,
    }
