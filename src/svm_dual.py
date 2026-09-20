from __future__ import annotations

import numpy as np
import cvxopt
import cvxopt.solvers

from .svm_common import compute_kernel

cvxopt.solvers.options["show_progress"] = False


def build_dual_matrices(X: np.ndarray, y: np.ndarray, kernel_type: str, gamma: float):
    K = compute_kernel(X, X, kernel_type, gamma)
    Q = np.outer(y, y) * K
    return K, Q


def dual_objective(alpha: np.ndarray, Q: np.ndarray) -> float:
    return 0.5 * float(alpha @ Q @ alpha) - float(np.sum(alpha))


def compute_bias(alpha, y, K, C, tol_sv=1e-5) -> float:
    weighted = alpha * y
    raw_scores = K @ weighted

    sv_mask = alpha > tol_sv
    margin_mask = (alpha > tol_sv) & (alpha < C - tol_sv)

    if np.any(margin_mask):
        return float(np.mean(y[margin_mask] - raw_scores[margin_mask]))
    if np.any(sv_mask):
        return float(np.median(y[sv_mask] - raw_scores[sv_mask]))
    return 0.0


def solve_svm_dual(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    kernel_type: str = "gaussian",
    gamma: float = 0.1,
    tol_sv: float = 1e-5,
    solver_jitter: float = 1e-8,
):
    n_samples = X.shape[0]
    K, Q = build_dual_matrices(X, y, kernel_type, gamma)
    Q_solver = Q + solver_jitter * np.eye(n_samples)

    P_mat = cvxopt.matrix(Q_solver, tc="d")
    q_mat = cvxopt.matrix(-np.ones(n_samples), tc="d")

    G = np.vstack([-np.eye(n_samples), np.eye(n_samples)])
    h = np.hstack([np.zeros(n_samples), C * np.ones(n_samples)])

    sol = cvxopt.solvers.qp(
        P_mat,
        q_mat,
        cvxopt.matrix(G, tc="d"),
        cvxopt.matrix(h, tc="d"),
        cvxopt.matrix(y.reshape(1, -1), tc="d"),
        cvxopt.matrix(0.0, tc="d"),
    )

    alpha = np.clip(np.array(sol["x"]).reshape(-1), 0.0, C)
    sv_mask = alpha > tol_sv
    b = compute_bias(alpha, y, K, C, tol_sv=tol_sv)
    obj_val = dual_objective(alpha, Q)
    return alpha, b, sv_mask, obj_val, sol, K, Q


def kkt_violation(alpha: np.ndarray, y: np.ndarray, Q: np.ndarray, C: float, tol_sv=1e-5):
    grad = Q @ alpha - np.ones_like(alpha)

    i_up = ((alpha < C - tol_sv) & (y > 0)) | ((alpha > tol_sv) & (y < 0))
    i_low = ((alpha > tol_sv) & (y > 0)) | ((alpha < C - tol_sv) & (y < 0))

    m_alpha = np.max(-y[i_up] * grad[i_up]) if np.any(i_up) else 0.0
    M_alpha = np.min(-y[i_low] * grad[i_low]) if np.any(i_low) else 0.0
    return max(0.0, float(m_alpha - M_alpha)), float(m_alpha), float(M_alpha)
