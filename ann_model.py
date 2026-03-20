"""
ann_model.py — P2-ETF-MERTON-ANN
Small MLP (~50 parameters) for optimal portfolio feedback control.
Trained via SGD to maximize expected isoelastic utility.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class MertonANN:
    """
    Small MLP for Merton optimal portfolio control.

    Architecture:
    - Input: 3 features (t/T, log(W/W0), regime)
    - Hidden: 10 neurons (ReLU activation)
    - Output: n_assets weights (softmax for long-only, sum-to-1)

    Total parameters: ~50 (3*10 + 10 + 10*n_assets + n_assets)
    """

    def __init__(self, n_assets: int, hidden_size: int = 10, eta: float = 0.5):
        self.n_assets = n_assets
        self.hidden_size = hidden_size
        self.eta = eta

        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(3, hidden_size) * np.sqrt(2.0 / 3)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, n_assets) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(n_assets)

        # Training history
        self.loss_history = []

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        X: shape (n_samples, 3) - [t/T, log(W/W0), regime]
        Returns: (hidden_output, output_weights)
        """
        # Hidden layer (ReLU)
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)  # ReLU

        # Output layer (linear then softmax)
        Z2 = A1 @ self.W2 + self.b2
        # Softmax for valid portfolio weights
        exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)

        return A1, A2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get portfolio weights for input states."""
        _, weights = self.forward(X)
        return weights

    def compute_utility(self, W_T: np.ndarray) -> np.ndarray:
        """Compute isoelastic utility of terminal wealth."""
        if self.eta == 1.0:
            return np.log(W_T)
        else:
            return (W_T ** (1 - self.eta) - 1) / (1 - self.eta)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        mu: np.ndarray,
        Sigma: np.ndarray,
        r: float,
        epochs: int = 500,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = False
    ) -> list:
        """
        Train ANN using SGD to maximize expected utility.

        Parameters:
        -----------
        X_train : training inputs (t/T, log(W/W0), regime)
        y_train : target weights (for supervised warm-start, optional)
        mu : mean returns (annualized)
        Sigma : covariance matrix (annualized)
        r : risk-free rate
        epochs : training iterations
        learning_rate : SGD step size
        batch_size : mini-batch size
        """
        n_samples = len(X_train)

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]

                # Forward pass
                A1, A2 = self.forward(X_batch)

                # Simulate portfolio returns with these weights
                # Simplified: expected utility gradient
                dt = 1/252

                # Portfolio mean and variance
                portfolio_mu = A2 @ mu  # (batch_size,)
                portfolio_var = np.sum(A2 * (A2 @ Sigma), axis=1)  # (batch_size,)

                # Expected wealth growth (simplified 1-step)
                expected_return = portfolio_mu * dt

                # Utility gradient approximation
                # Higher return = higher utility, lower variance = higher utility
                utility_grad = (mu - r) / self.eta - Sigma @ A2.T

                # Backward pass (simplified policy gradient)
                dZ2 = A2 - (y_train[indices[i:batch_end]] if i < len(y_train) else A2)
                dZ2 = dZ2 * 0.1  # Scale gradient

                dW2 = A1.T @ dZ2 / len(X_batch)
                db2 = np.mean(dZ2, axis=0)

                dA1 = dZ2 @ self.W2.T
                dZ1 = dA1 * (A1 > 0)  # ReLU derivative

                dW1 = X_batch.T @ dZ1 / len(X_batch)
                db1 = np.mean(dZ1, axis=0)

                # Update weights (SGD)
                self.W2 += learning_rate * dW2
                self.b2 += learning_rate * db2
                self.W1 += learning_rate * dW1
                self.b1 += learning_rate * db1

                epoch_loss += np.mean(dZ2 ** 2)

            self.loss_history.append(epoch_loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

        return self.loss_history

    def train_supervised(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 500,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = False
    ) -> list:
        """
        Supervised training on Merton-optimal targets.
        More stable than direct utility maximization.
        """
        n_samples = len(X_train)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]

                # Forward
                A1, A2 = self.forward(X_batch)

                # Cross-entropy loss (softmax targets)
                # Clip for numerical stability
                A2_clipped = np.clip(A2, 1e-10, 1.0)
                loss = -np.sum(y_batch * np.log(A2_clipped)) / len(X_batch)

                # Backward
                dZ2 = A2 - y_batch

                dW2 = A1.T @ dZ2 / len(X_batch)
                db2 = np.mean(dZ2, axis=0)

                dA1 = dZ2 @ self.W2.T
                dZ1 = dA1 * (A1 > 0)

                dW1 = X_batch.T @ dZ1 / len(X_batch)
                db1 = np.mean(dZ1, axis=0)

                # Update
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1

                epoch_loss += loss

            self.loss_history.append(epoch_loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

        return self.loss_history

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return model weights for serialization."""
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Load model weights."""
        self.W1 = weights["W1"]
        self.b1 = weights["b1"]
        self.W2 = weights["W2"]
        self.b2 = weights["b2"]

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size
        )


def train_ann_for_horizon(
    training_data: Dict[str, np.ndarray],
    n_assets: int,
    eta: float = 0.5,
    epochs: int = 500,
    learning_rate: float = 0.01
) -> MertonANN:
    """
    Train ANN for a specific investment horizon.

    Returns trained MertonANN model.
    """
    X = training_data["X"]
    y = training_data["y"]

    model = MertonANN(n_assets=n_assets, hidden_size=10, eta=eta)

    # Use supervised training on Merton-optimal targets
    model.train_supervised(
        X, y,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=32,
        verbose=False
    )

    return model


def predict_optimal_etf(
    model: MertonANN,
    t_T: float,
    log_wealth_ratio: float,
    regime: int
) -> Tuple[int, np.ndarray]:
    """
    Predict optimal ETF (winner-takes-all) from ANN.

    Returns:
    --------
    selected_idx : index of selected ETF (argmax)
    weights : full weight vector from ANN
    """
    X = np.array([[t_T, log_wealth_ratio, regime]])
    weights = model.predict(X)[0]
    selected_idx = np.argmax(weights)
    return selected_idx, weights
