from __future__ import annotations

import numpy as np

from .svm_common import compute_kernel


def select_mvp(alpha, grad, y, C, tol_sv=1e-8):
    neg_y_grad = -y * grad
    R = ((alpha < C - tol_sv) & (y > 0)) | ((alpha > tol_sv) & (y < 0))
    S = ((alpha < C - tol_sv) & (y < 0)) | ((alpha > tol_sv) & (y > 0))

    R_values = np.where(R, neg_y_grad, -np.inf)
    S_values = np.where(S, neg_y_grad, np.inf)

    i = int(np.argmax(R_values))
    j = int(np.argmin(S_values))
    return i, j, float(R_values[i]), float(S_values[j])


def solve_q2(alpha_i, alpha_j, g_i, g_j, y_i, y_j, Q_ii, Q_jj, Q_ij, C):
    eta = Q_ii + Q_jj - 2.0 * y_i * y_j * Q_ij
    eta = max(eta, 1e-12)

    step_i = -(g_i - y_i * y_j * g_j) / eta

    if y_i * y_j < 0:
        lower = max(0.0, alpha_i - alpha_j)
        upper = min(C, C + alpha_i - alpha_j)
    else:
        lower = max(0.0, alpha_i + alpha_j - C)
        upper = min(C, alpha_i + alpha_j)

    alpha_i_new = float(np.clip(alpha_i + step_i, lower, upper))
    realized_step_i = alpha_i_new - alpha_i
    alpha_j_new = float(alpha_j - y_i * y_j * realized_step_i)
    alpha_j_new = float(np.clip(alpha_j_new, 0.0, C))
    return alpha_i_new, alpha_j_new


def mvp_decomposition(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    kernel_type: str = "gaussian",
    gamma: float = 0.1,
    max_iter: int = 10000,
    tol: float = 1e-3,
    tol_sv: float = 1e-5,
):
    n_samples = X.shape[0]
    K = compute_kernel(X, X, kernel_type, gamma)
    Q = np.outer(y, y) * K

    alpha = np.zeros(n_samples)
    grad = -np.ones(n_samples)
    gap_history = []

    status = "MAX_ITER_REACHED"
    gap = np.inf
    m_alpha = M_alpha = np.nan

    for iteration in range(1, max_iter + 1):
        i, j, m_alpha, M_alpha = select_mvp(alpha, grad, y, C, tol_sv)
        gap = m_alpha - M_alpha
        gap_history.append(gap)

        if gap <= tol:
            status = "CONVERGED"
            break

        alpha_i_new, alpha_j_new = solve_q2(
            alpha[i], alpha[j], grad[i], grad[j], y[i], y[j],
            Q[i, i], Q[j, j], Q[i, j], C,
        )

        d_i = alpha_i_new - alpha[i]
        d_j = alpha_j_new - alpha[j]
        grad += d_i * Q[:, i] + d_j * Q[:, j]
        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new

    obj_val = 0.5 * float(alpha @ Q @ alpha) - float(np.sum(alpha))
    sv_mask = alpha > tol_sv
    margin_mask = (alpha > tol_sv) & (alpha < C - tol_sv)

    if np.any(margin_mask):
        decision_vals = (alpha * y) @ K[:, margin_mask]
        b = float(np.mean(y[margin_mask] - decision_vals))
    elif np.any(sv_mask):
        decision_vals = (alpha * y) @ K[:, sv_mask]
        b = float(np.mean(y[sv_mask] - decision_vals))
    else:
        b = 0.0

    info = {
        "iterations": iteration,
        "status": status,
        "final_gap": float(gap),
        "m_alpha": float(m_alpha),
        "M_alpha": float(M_alpha),
        "gap_history": gap_history,
    }
    return alpha, b, sv_mask, obj_val, info
