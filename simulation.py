"""
simulation.py — P2-ETF-MERTON-ANN
Vectorized simulation with optional macro feature bootstrap.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.linalg import cholesky
from sklearn.covariance import LedoitWolf

def simulate_semi_markov_regime_vectorized(
    T: int,
    n_paths: int,
    dt: float = 1/252,
    semi_markov_params: Dict = None
) -> np.ndarray:
    """Vectorized semi-Markov regime simulation."""
    if semi_markov_params is None:
        semi_markov_params = {"p_01": 0.001, "p_10": 0.002}
    p_01 = semi_markov_params.get("p_01", 0.001)
    p_10 = semi_markov_params.get("p_10", 0.002)

    regime_paths = np.zeros((n_paths, T), dtype=int)
    regime_paths[:, 0] = np.random.choice([0, 1], size=n_paths)

    current_regime = regime_paths[:, 0].copy()
    time_in_regime = np.zeros(n_paths, dtype=int)

    for t in range(1, T):
        for reg_val in [0, 1]:
            mask = (current_regime == reg_val)
            if not np.any(mask):
                continue
            if reg_val == 0:
                base_p = p_01
                adj = 1 + time_in_regime[mask] / 252
            else:
                base_p = p_10
                adj = 1 + time_in_regime[mask] / 126
            p_trans = np.minimum(base_p * adj, 0.5)
            switch = np.random.random(size=np.sum(mask)) < p_trans
            new_regime = np.where(switch, 1 - reg_val, reg_val)
            regime_paths[mask, t] = new_regime
            current_regime[mask] = new_regime
            time_in_regime[mask] = np.where(switch, 0, time_in_regime[mask] + 1)

    return regime_paths


def generate_merton_training_data(
    params: Dict[int, Dict],
    semi_markov_params: Dict,
    T_days: int,
    n_paths: int = 10000,
    W0: float = 1.0,
    eta: float = 0.5,
    n_assets: int = None,
    macro_data: pd.DataFrame = None,   # for Option B
    option: str = "A"
) -> Dict[str, np.ndarray]:
    """Generate training data with optional macro features."""
    if n_assets is None:
        n_assets = len(params[0]["mu"])
    if n_assets == 0:
        raise ValueError("n_assets cannot be 0")

    print(f"  Generating training data: {n_paths} paths, {T_days} days, {n_assets} assets, option {option}")

    dt = 1/252
    T_steps = T_days

    # Simulate regime paths
    regime_paths = simulate_semi_markov_regime_vectorized(T_steps, n_paths, dt, semi_markov_params)

    # Macro bootstrapping (if option B and macro_data available)
    if option == "B" and macro_data is not None and not macro_data.empty:
        macro_vals = macro_data.values  # shape (n_macro_days, n_macro_features)
        n_macro_days = macro_vals.shape[0]
        # Random start indices for each path
        start_indices = np.random.randint(0, n_macro_days - T_steps, size=n_paths)
        macro_paths = np.zeros((n_paths, T_steps, macro_vals.shape[1]))
        for i in range(n_paths):
            macro_paths[i] = macro_vals[start_indices[i]:start_indices[i]+T_steps]
    else:
        macro_paths = None

    # We'll sample n_samples_per_path points uniformly
    n_samples_per_path = min(50, T_steps)
    sample_indices = np.linspace(0, T_steps-1, n_samples_per_path, dtype=int)
    total_samples = n_paths * n_samples_per_path

    # Determine feature dimension
    base_dim = 3  # t/T, log_wealth, regime
    if macro_paths is not None:
        macro_dim = macro_paths.shape[2]
    else:
        macro_dim = 0

    X = np.zeros((total_samples, base_dim + macro_dim), dtype=np.float32)
    y = np.zeros((total_samples, n_assets), dtype=np.float32)

    # Wealth simulation (simple random walk for diversity)
    wealth = np.full(n_paths, W0, dtype=np.float32)

    # Precompute Merton weights for each regime (they are state-dependent but not path-dependent)
    # We'll compute on the fly per sample

    idx = 0
    for sample_t in sample_indices:
        t_norm = sample_t / T_steps
        current_regime = regime_paths[:, sample_t].astype(int)

        # For each regime, compute weights for all paths in that regime
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
                # Fallback equal weights
                w = np.ones(n_assets) / n_assets
                y[idx + np.where(mask)[0], :] = w
                continue

            excess = mu_reg - r_reg
            optimal = (1/eta) * Sigma_reg_inv @ excess
            optimal = np.exp(optimal - np.max(optimal))
            w = optimal / (optimal.sum() + 1e-10)
            y[idx + np.where(mask)[0], :] = w

        # Build features
        log_wealth = np.log(np.maximum(wealth / W0, 1e-8))
        if macro_paths is not None:
            macro_t = macro_paths[:, sample_t, :]  # (n_paths, macro_dim)
            X[idx:idx+n_paths, 0] = t_norm
            X[idx:idx+n_paths, 1] = log_wealth
            X[idx:idx+n_paths, 2] = current_regime
            X[idx:idx+n_paths, 3:] = macro_t
        else:
            X[idx:idx+n_paths, 0] = t_norm
            X[idx:idx+n_paths, 1] = log_wealth
            X[idx:idx+n_paths, 2] = current_regime

        # Update wealth for next step (random return based on current regime's average drift)
        # Simplified: use global average drift
        avg_mu = (params[0]["mu"] + params[1]["mu"]) / 2
        drift = np.mean(avg_mu) * dt
        shock = np.random.normal(0, 0.02, size=n_paths)
        wealth = wealth * (1 + drift + shock)
        wealth = np.maximum(wealth, 1e-8)

        idx += n_paths

    return {"X": X, "y": y}
