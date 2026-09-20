from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset, scale_train_test, split_train_test
from src.svm_common import accuracy, confusion_matrix_binary, predict
from src.svm_dual import kkt_violation, solve_svm_dual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    X_train_raw, X_test_raw, y_train, y_test = split_train_test(X, y, seed=args.seed)
    X_train, X_test, _ = scale_train_test(X_train_raw, X_test_raw)

    C, gamma, kernel_type = 1.0, 0.1, "gaussian"

    start = time.time()
    alpha, b, sv_mask, obj, sol, _, Q = solve_svm_dual(
        X_train, y_train, C=C, gamma=gamma, kernel_type=kernel_type
    )
    elapsed = time.time() - start

    X_sv, y_sv, alpha_sv = X_train[sv_mask], y_train[sv_mask], alpha[sv_mask]
    train_pred = predict(X_train, X_sv, y_sv, alpha_sv, b, kernel_type, gamma)
    test_pred = predict(X_test, X_sv, y_sv, alpha_sv, b, kernel_type, gamma)
    kkt, _, _ = kkt_violation(alpha, y_train, Q, C)

    print("SVM — dual QP with CVXOPT")
    print(f"solver status: {sol.get('status', 'N/A')}")
    print(f"iterations: {sol.get('iterations', 'N/A')}")
    print(f"dual objective: {obj:.6f}")
    print(f"KKT violation: {kkt:.6e}")
    print(f"support vectors: {sv_mask.sum()}")
    print(f"train accuracy: {100*accuracy(y_train, train_pred):.2f}%")
    print(f"test accuracy: {100*accuracy(y_test, test_pred):.2f}%")
    print("test confusion matrix:")
    print(confusion_matrix_binary(y_test, test_pred))
    print(f"runtime: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
