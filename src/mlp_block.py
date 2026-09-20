from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize

from .mlp import forward, init_params, pack_params, unpack_params


def full_objective(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    sigma: float,
    rho: float,
) -> float:
    n_samples, n_features = X.shape
    w, b, v = unpack_params(params, n_hidden, n_features)
    scores, _, _ = forward(X, w, b, v, sigma)
    residuals = scores - y
    mse = 0.5 / n_samples * np.dot(residuals, residuals)
    reg = 0.5 * rho * np.dot(params, params)
    return mse + reg


def solve_v(Z: np.ndarray, y: np.ndarray, rho: float) -> np.ndarray:
    """Exact minimizer of the convex output-layer subproblem."""
    n_samples, n_hidden = Z.shape
    lhs = (Z.T @ Z) / n_samples + rho * np.eye(n_hidden)
    rhs = (Z.T @ y) / n_samples
    return solve(lhs, rhs, assume_a="pos")


def _unpack_wb(params_wb: np.ndarray, n_hidden: int, n_features: int):
    w = params_wb[: n_hidden * n_features].reshape(n_hidden, n_features)
    b = params_wb[n_hidden * n_features :]
    return w, b


def objective_and_grad_wb(
    params_wb: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    n_hidden: int,
    sigma: float,
    rho: float,
):
    n_samples, n_features = X.shape
    w, b = _unpack_wb(params_wb, n_hidden, n_features)

    A = X @ w.T + b
    Z = np.tanh(sigma * A)
    scores = Z @ v
    residuals = scores - y

    mse = 0.5 / n_samples * np.dot(residuals, residuals)
    reg = 0.5 * rho * (np.dot(w.ravel(), w.ravel()) + np.dot(b, b))
    objective = mse + reg

    delta_out = residuals / n_samples
    dtanh = 1.0 - Z**2
    delta_hidden = np.outer(delta_out, v) * dtanh * sigma

    grad_w = delta_hidden.T @ X + rho * w
    grad_b = delta_hidden.sum(axis=0) + rho * b
    grad = np.concatenate([grad_w.ravel(), grad_b])
    return objective, grad


def train_block_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_hidden: int = 10,
    sigma: float = 1.0,
    rho: float = 1e-3,
    seed: int = 42,
    max_outer: int = 500,
    tol_outer: float = 1e-5,
    tol_rel_objective: float = 1e-8,
    max_inner: int = 500,
    gtol_inner: float = 1e-5,
    ftol_inner: float = 1e-12,
):
    """Alternate exact output-layer updates with L-BFGS-B hidden-layer updates."""
    n_features = X.shape[1]
    params0 = init_params(n_hidden, n_features, seed=seed)
    w, b, v = unpack_params(params0, n_hidden, n_features)
    w, b, v = w.copy(), b.copy(), v.copy()

    history = []
    total_fun_evals = 0
    total_grad_evals = 0
    previous_objective = full_objective(params0, X, y, n_hidden, sigma, rho)

    converged = False
    for outer_it in range(1, max_outer + 1):
        params_old = pack_params(w, b, v)

        _, Z_old, _ = forward(X, w, b, v, sigma)
        v_mid = solve_v(Z_old, y, rho)

        wb0 = np.concatenate([w.ravel(), b])
        result = minimize(
            fun=objective_and_grad_wb,
            x0=wb0,
            args=(X, y, v_mid, n_hidden, sigma, rho),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": max_inner,
                "gtol": gtol_inner,
                "ftol": ftol_inner,
                "maxfun": 20000,
            },
        )

        w_new, b_new = _unpack_wb(result.x, n_hidden, n_features)
        total_fun_evals += result.nfev
        total_grad_evals += result.njev if result.njev is not None else result.nfev

        _, Z_new, _ = forward(X, w_new, b_new, v_mid, sigma)
        v_new = solve_v(Z_new, y, rho)

        params_new = pack_params(w_new, b_new, v_new)
        objective = full_objective(params_new, X, y, n_hidden, sigma, rho)
        max_parameter_change = np.max(np.abs(params_new - params_old))
        relative_change = abs(objective - previous_objective) / (abs(previous_objective) + 1e-12)

        history.append(
            {
                "outer_iteration": outer_it,
                "objective": objective,
                "max_parameter_change": max_parameter_change,
                "relative_objective_change": relative_change,
                "inner_iterations": result.nit,
            }
        )

        w, b, v = w_new, b_new, v_new
        previous_objective = objective

        if max_parameter_change < tol_outer or relative_change < tol_rel_objective:
            converged = True
            break

    return pack_params(w, b, v), history, {
        "outer_iterations": outer_it,
        "total_function_evaluations": total_fun_evals,
        "total_gradient_evaluations": total_grad_evals,
        "converged": converged,
    }
