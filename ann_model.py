"""
ann_model.py — P2-ETF-MERTON-ANN
Small MLP with variable input dimension.
"""

import numpy as np
from typing import Dict, Tuple

class MertonANN:
    def __init__(self, n_assets: int, input_dim: int = 3, hidden_size: int = 10, eta: float = 0.5):
        self.n_assets = n_assets
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.eta = eta

        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_size) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, n_assets) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(n_assets)

        self.loss_history = []

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_clipped = np.clip(X, -10, 10)
        Z1 = X_clipped @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ self.W2 + self.b2
        Z2_clipped = np.clip(Z2, -50, 50)
        exp_Z2 = np.exp(Z2_clipped - np.max(Z2_clipped, axis=1, keepdims=True))
        A2 = exp_Z2 / (np.sum(exp_Z2, axis=1, keepdims=True) + 1e-10)
        return A1, A2

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, weights = self.forward(X)
        return weights

    def compute_utility(self, W_T: np.ndarray) -> np.ndarray:
        if self.eta == 1.0:
            return np.log(np.clip(W_T, 1e-8, None))
        else:
            return (np.clip(W_T, 1e-8, None) ** (1 - self.eta) - 1) / (1 - self.eta)

    def train_supervised(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 500,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        clip_grad: float = 1.0,
        verbose: bool = False
    ) -> list:
        n_samples = len(X_train)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]

                A1, A2 = self.forward(X_batch)

                A2_clipped = np.clip(A2, 1e-10, 1.0)
                loss = -np.sum(y_batch * np.log(A2_clipped)) / len(X_batch)

                dZ2 = A2 - y_batch
                dW2 = A1.T @ dZ2 / len(X_batch)
                db2 = np.mean(dZ2, axis=0)
                dA1 = dZ2 @ self.W2.T
                dZ1 = dA1 * (A1 > 0)
                dW1 = X_batch.T @ dZ1 / len(X_batch)
                db1 = np.mean(dZ1, axis=0)

                grad_norm = np.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2))
                if grad_norm > clip_grad:
                    scale = clip_grad / grad_norm
                    dW1 *= scale
                    db1 *= scale
                    dW2 *= scale
                    db2 *= scale

                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1

                epoch_loss += loss

            self.W1 = np.clip(self.W1, -10, 10)
            self.W2 = np.clip(self.W2, -10, 10)

            self.loss_history.append(epoch_loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return self.loss_history

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_weights(self, weights: Dict[str, np.ndarray]):
        self.W1 = weights["W1"]
        self.b1 = weights["b1"]
        self.W2 = weights["W2"]
        self.b2 = weights["b2"]

    def count_parameters(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


def train_ann_for_horizon(
    training_data: Dict[str, np.ndarray],
    n_assets: int,
    eta: float = 0.5,
    epochs: int = 500,
    learning_rate: float = 0.01,
    hidden_size: int = 10,
    input_dim: int = None
) -> MertonANN:
    X = training_data["X"]
    y = training_data["y"]
    if input_dim is None:
        input_dim = X.shape[1]

    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=1.0/n_assets)

    model = MertonANN(n_assets=n_assets, input_dim=input_dim, hidden_size=hidden_size, eta=eta)
    model.train_supervised(
        X, y,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=32,
        clip_grad=1.0,
        verbose=False
    )
    return model


def predict_optimal_etf(
    model: MertonANN,
    t_T: float,
    log_wealth_ratio: float,
    regime: int,
    macro_features: np.ndarray = None
) -> Tuple[int, np.ndarray]:
    if macro_features is None:
        X = np.array([[t_T, log_wealth_ratio, regime]], dtype=np.float32)
    else:
        # Combine features: [t_T, log_wealth_ratio, regime, macro1, macro2, ...]
        X = np.concatenate([[t_T, log_wealth_ratio, regime], macro_features]).reshape(1, -1).astype(np.float32)

    weights = model.predict(X)[0]
    if np.any(np.isnan(weights)):
        weights = np.ones(model.n_assets) / model.n_assets
    selected_idx = np.argmax(weights)
    return selected_idx, weights
