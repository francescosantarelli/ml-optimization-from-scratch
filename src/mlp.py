from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def pack_params(w: np.ndarray, b: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([w.ravel(), b, v])


def unpack_params(params: np.ndarray, n_hidden: int, n_features: int):
    w = params[: n_hidden * n_features].reshape(n_hidden, n_features)
    b = params[n_hidden * n_features : n_hidden * n_features + n_hidden]
    v = params[n_hidden * n_features + n_hidden :]
    return w, b, v


def init_params(n_hidden: int, n_features: int, seed: int = 42) -> np.ndarray:
    """Xavier-style Gaussian initialization."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0 / (n_features + n_hidden))
    w0 = rng.normal(0.0, scale, size=(n_hidden, n_features))
    b0 = np.zeros(n_hidden)
    v0 = rng.normal(0.0, scale, size=n_hidden)
    return pack_params(w0, b0, v0)


def forward(
    X: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    v: np.ndarray,
    sigma: float,
):
    A = X @ w.T + b
    Z = np.tanh(sigma * A)
    scores = Z @ v
    return scores, Z, A


def objective_and_grad(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    sigma: float,
    rho: float,
):
    """Regularized MSE objective and analytical gradient."""
    n_samples, n_features = X.shape
    w, b, v = unpack_params(params, n_hidden, n_features)

    scores, Z, _ = forward(X, w, b, v, sigma)
    residuals = scores - y

    mse = 0.5 / n_samples * np.dot(residuals, residuals)
    reg = 0.5 * rho * np.dot(params, params)
    objective = mse + reg

    delta_out = residuals / n_samples
    grad_v = Z.T @ delta_out + rho * v

    dtanh = 1.0 - Z**2
    delta_hidden = np.outer(delta_out, v) * dtanh * sigma
    grad_w = delta_hidden.T @ X + rho * w
    grad_b = delta_hidden.sum(axis=0) + rho * b

    grad = pack_params(grad_w, grad_b, grad_v)
    return objective, grad


def plain_mse(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    sigma: float,
) -> float:
    _, n_features = X.shape
    w, b, v = unpack_params(params, n_hidden, n_features)
    scores, _, _ = forward(X, w, b, v, sigma)
    return 0.5 / len(y) * np.sum((scores - y) ** 2)


def predict(X: np.ndarray, params: np.ndarray, n_hidden: int, sigma: float):
    _, n_features = X.shape
    w, b, v = unpack_params(params, n_hidden, n_features)
    scores, _, _ = forward(X, w, b, v, sigma)
    return np.where(scores >= 0.0, 1.0, -1.0)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def train_full_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_hidden: int = 10,
    sigma: float = 1.0,
    rho: float = 1e-3,
    seed: int = 42,
    max_iter: int = 2000,
    gtol: float = 1e-5,
    ftol: float = 1e-12,
):
    """Train all MLP parameters jointly with L-BFGS-B."""
    params0 = init_params(n_hidden, X.shape[1], seed=seed)
    result = minimize(
        fun=objective_and_grad,
        x0=params0,
        args=(X, y, n_hidden, sigma, rho),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol, "maxfun": 50000},
    )
    return params0, result
