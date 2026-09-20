from __future__ import annotations

import numpy as np


def gaussian_kernel(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Z_sq = np.sum(Z**2, axis=1, keepdims=True)
    sq_dists = np.maximum(X_sq + Z_sq.T - 2.0 * (X @ Z.T), 0.0)
    return np.exp(-gamma * sq_dists)


def polynomial_kernel(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    return (X @ Z.T + 1.0) ** gamma


def compute_kernel(X: np.ndarray, Z: np.ndarray, kernel_type: str, gamma: float):
    if kernel_type == "gaussian":
        return gaussian_kernel(X, Z, gamma)
    if kernel_type == "polynomial":
        return polynomial_kernel(X, Z, gamma)
    raise ValueError("kernel_type must be 'gaussian' or 'polynomial'.")


def decision_function(
    X_new: np.ndarray,
    X_sv: np.ndarray,
    y_sv: np.ndarray,
    alpha_sv: np.ndarray,
    b: float,
    kernel_type: str,
    gamma: float,
):
    K = compute_kernel(X_sv, X_new, kernel_type, gamma)
    return (alpha_sv * y_sv) @ K + b


def predict(
    X_new: np.ndarray,
    X_sv: np.ndarray,
    y_sv: np.ndarray,
    alpha_sv: np.ndarray,
    b: float,
    kernel_type: str,
    gamma: float,
):
    scores = decision_function(X_new, X_sv, y_sv, alpha_sv, b, kernel_type, gamma)
    return np.where(scores >= 0.0, 1.0, -1.0)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == -1) & (y_true == -1)))
    fp = int(np.sum((y_pred == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))
    return np.array([[tp, fn], [fp, tn]], dtype=int)
